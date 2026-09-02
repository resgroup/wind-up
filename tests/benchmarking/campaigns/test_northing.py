"""Tests for the shared northing step that runs upstream of every method."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns.declaration import CampaignSpec
from benchmarking.campaigns.northing import north_campaign_scada
from benchmarking.synthetic import HOT_COLUMNS
from wind_up.circular_math import circ_diff

_COLUMNS = HOT_COLUMNS
_TURBINES = ("T01", "T02", "T03", "T04")
_START = pd.Timestamp("2018-01-01", tz="UTC")
_RATED = 2300.0


def _index(days: int = 120) -> pd.DatetimeIndex:
    return pd.date_range(start=_START, periods=days * 144, freq="600s", tz="UTC")


def _scada(
    index: pd.DatetimeIndex, offsets: dict[str, list[tuple[pd.Timestamp, float]]]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Long SCADA whose turbines report the site direction minus their own north offset."""
    rng = np.random.default_rng(0)
    site_wd = np.cumsum(rng.normal(0.0, 2.0, len(index))) % 360.0
    frames = []
    for i, turbine in enumerate(_TURBINES):
        applied = np.full(len(index), offsets[turbine][0][1], dtype=float)
        for when, value in offsets[turbine][1:]:
            applied[index >= when] = value
        scatter = np.random.default_rng(10 + i).normal(0.0, 6.0, len(index))
        frames.append(
            pd.DataFrame(
                {
                    _COLUMNS.turbine: turbine,
                    _COLUMNS.active_power: 1200.0,
                    _COLUMNS.active_power_min: 1100.0,
                    _COLUMNS.wind_speed: 9.0,
                    _COLUMNS.wind_speed_sd: 1.0,
                    _COLUMNS.gen_rpm: 1500.0,
                    _COLUMNS.availability: 600.0,
                    _COLUMNS.nacelle_position: (site_wd + scatter - applied) % 360.0,
                },
                index=index,
            )
        )
    return pd.concat(frames), site_wd


def _spec(north_offsets: list[tuple[str, pd.Timestamp, float]] | None) -> CampaignSpec:
    return CampaignSpec(
        upgraded_turbines=["T01"],
        upgrade_timing=_START + pd.Timedelta(days=60),
        candidate_references=[t for t in _TURBINES if t != "T01"],
        excluded_turbines=[],
        coords=dict.fromkeys(_TURBINES, (0.0, 0.0)),
        north_offsets=north_offsets,
        rated_power_kw=_RATED,
        analysis_period=(_START, _START + pd.Timedelta(days=120)),
    )


def _northed(frame: pd.DataFrame, turbine: str) -> np.ndarray:
    rows = frame[frame[_COLUMNS.turbine] == turbine]
    return rows[_COLUMNS.northed("nacelle_position")].to_numpy(dtype=float)


class TestDiscovery:
    """``north_offsets=None`` means discover from the data."""

    def test_writes_a_northed_companion_leaving_the_original_untouched(self) -> None:
        index = _index()
        offsets = {t: [(_START, 20.0 * i)] for i, t in enumerate(_TURBINES)}
        scada, site_wd = _scada(index, offsets)
        era5 = pd.Series(site_wd, index=index)

        out = north_campaign_scada(scada, spec=_spec(None), columns=_COLUMNS, era5_wd=era5)

        assert _COLUMNS.northed("nacelle_position") in out.columns
        assert np.allclose(
            out[_COLUMNS.nacelle_position].to_numpy(dtype=float),
            scada[_COLUMNS.nacelle_position].to_numpy(dtype=float),
        )

    def test_recovers_each_turbines_offset(self) -> None:
        index = _index()
        offsets = {"T01": [(_START, 0.0)], "T02": [(_START, 25.0)], "T03": [(_START, -40.0)], "T04": [(_START, 12.0)]}
        scada, site_wd = _scada(index, offsets)
        era5 = pd.Series(site_wd, index=index)

        out = north_campaign_scada(scada, spec=_spec(None), columns=_COLUMNS, era5_wd=era5)

        for turbine in _TURBINES:
            assert circ_diff(_northed(out, turbine), site_wd).mean() == pytest.approx(0.0, abs=2.0), turbine

    def test_recovers_a_step_change_mid_campaign(self) -> None:
        index = _index()
        step_at = _START + pd.Timedelta(days=70)
        offsets = {t: [(_START, 5.0 * i)] for i, t in enumerate(_TURBINES)}
        offsets["T03"] = [(_START, 10.0), (step_at, 55.0)]
        scada, site_wd = _scada(index, offsets)
        era5 = pd.Series(site_wd, index=index)

        out = north_campaign_scada(scada, spec=_spec(None), columns=_COLUMNS, era5_wd=era5)

        assert circ_diff(_northed(out, "T03"), site_wd).mean() == pytest.approx(0.0, abs=2.0)

    def test_discovery_without_reanalysis_raises(self) -> None:
        index = _index(days=30)
        scada, _ = _scada(index, {t: [(_START, 0.0)] for t in _TURBINES})
        with pytest.raises(ValueError, match="era5_wd"):
            north_campaign_scada(scada, spec=_spec(None), columns=_COLUMNS, era5_wd=None)


class TestDeclared:
    """A supplied table is applied exactly; nothing is discovered."""

    def test_applies_the_declared_offsets(self) -> None:
        index = _index(days=30)
        scada, _ = _scada(index, {t: [(_START, 0.0)] for t in _TURBINES})
        declared = [("T02", _START, 33.0)]

        out = north_campaign_scada(scada, spec=_spec(declared), columns=_COLUMNS, era5_wd=None)

        raw = scada[scada[_COLUMNS.turbine] == "T02"][_COLUMNS.nacelle_position].to_numpy(dtype=float)
        assert _northed(out, "T02") == pytest.approx((raw + 33.0) % 360.0)
        # a turbine with no declared correction is copied through unchanged
        untouched = scada[scada[_COLUMNS.turbine] == "T01"][_COLUMNS.nacelle_position].to_numpy(dtype=float)
        assert _northed(out, "T01") == pytest.approx(untouched % 360.0)

    def test_an_empty_list_applies_no_correction_but_still_writes_the_column(self) -> None:
        index = _index(days=30)
        offsets = {t: [(_START, 30.0)] for t in _TURBINES}
        scada, _ = _scada(index, offsets)

        out = north_campaign_scada(scada, spec=_spec([]), columns=_COLUMNS, era5_wd=None)

        assert _COLUMNS.northed("nacelle_position") in out.columns
        for turbine in _TURBINES:
            raw = scada[scada[_COLUMNS.turbine] == turbine][_COLUMNS.nacelle_position].to_numpy(dtype=float)
            assert _northed(out, turbine) == pytest.approx(raw % 360.0), turbine

    def test_an_empty_list_needs_no_reanalysis(self) -> None:
        index = _index(days=30)
        scada, _ = _scada(index, {t: [(_START, 0.0)] for t in _TURBINES})
        # would raise if this branch tried to discover
        north_campaign_scada(scada, spec=_spec([]), columns=_COLUMNS, era5_wd=None)
