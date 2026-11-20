"""Tests for the study path's northing of every replicate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.harness import replicates as harness_replicates
from benchmarking.harness.replicates import NorthingInputs, StudyConfig, iter_replicates
from benchmarking.synthetic import HOT_COLUMNS, ConstantCpChange
from wind_up.circular_math import circ_diff
from wind_up.layout import Layout

if TYPE_CHECKING:
    import pytest

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


def _spread_across_turbines(directions: np.ndarray) -> float:
    """Median absolute disagreement between each turbine's direction and the first turbine's."""
    reference = np.repeat(directions[:, [0]], directions.shape[1], axis=1)
    return float(np.nanmedian(np.abs(circ_diff(directions, reference))))


def _northed(frame: pd.DataFrame, turbine: str) -> np.ndarray:
    rows = frame[frame[_COLUMNS.turbine] == turbine]
    return rows[_COLUMNS.northed("nacelle_position")].to_numpy(dtype=float)


class TestStudyPath:
    """The study path norths every replicate, so a method sees a table wind-up worked out itself."""

    @staticmethod
    def _study() -> StudyConfig:
        return StudyConfig(
            mode="prepost",
            turbine_subset=list(_TURBINES),
            treatment_start_range=(_START + pd.Timedelta(days=55), _START + pd.Timedelta(days=65)),
            min_pre_months=1,
            campaign_months=[1],
            n_replicates=1,
            seed=0,
        )

    @staticmethod
    def _replicate(*, era5_wd: pd.Series | None) -> pd.DataFrame:
        index = _index()
        offsets = {t: [(_START, 25.0 * i)] for i, t in enumerate(_TURBINES)}
        scada, site_wd = _scada(index, offsets)
        reanalysis = pd.Series(site_wd, index=index) if era5_wd is None else era5_wd
        replicates = list(
            iter_replicates(
                scada,
                profile=[ConstantCpChange(delta=0.05)],
                study=TestStudyPath._study(),
                columns=_COLUMNS,
                northing=None if era5_wd is None else NorthingInputs(era5_wd=reanalysis, layout=None),
                rated_power_kw=_RATED,
            )
        )
        return replicates[0].synthetic_df

    def test_the_layout_reaches_the_northing_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        index = _index()
        scada, site_wd = _scada(index, {t: [(_START, 0.0)] for t in _TURBINES})
        layout = Layout.from_frame(
            pd.DataFrame(
                {
                    "name": list(_TURBINES),
                    "latitude": [55.0 + 0.004 * i for i in range(len(_TURBINES))],
                    "longitude": 0.0,
                    "rotor_diameter_m": 82.0,
                }
            )
        )
        seen = []

        def recording_north_scada(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
            seen.append(kwargs["layout"])
            return frame

        monkeypatch.setattr(harness_replicates, "north_scada", recording_north_scada)
        next(
            iter_replicates(
                scada,
                profile=[ConstantCpChange(delta=0.05)],
                study=self._study(),
                columns=_COLUMNS,
                northing=NorthingInputs(era5_wd=pd.Series(site_wd, index=index), layout=layout),
                rated_power_kw=_RATED,
            )
        )

        assert len(seen) == 1
        assert seen[0] is layout

    def test_no_reanalysis_means_no_northed_column(self) -> None:
        assert _COLUMNS.northed("nacelle_position") not in self._replicate(era5_wd=None).columns

    def test_reanalysis_norths_each_replicate(self) -> None:
        index = _index()
        _, site_wd = _scada(index, {t: [(_START, 25.0 * i)] for i, t in enumerate(_TURBINES)})
        synthetic = self._replicate(era5_wd=pd.Series(site_wd, index=index))

        assert _COLUMNS.northed("nacelle_position") in synthetic.columns
        # Each turbine reports the site direction behind its own offset; northing removes the
        # spread between them, which the raw readings still carry.
        raw = np.column_stack(
            [
                synthetic[synthetic[_COLUMNS.turbine] == t][_COLUMNS.nacelle_position].to_numpy(dtype=float)
                for t in _TURBINES
            ]
        )
        northed = np.column_stack([_northed(synthetic, t) for t in _TURBINES])
        assert _spread_across_turbines(northed) < 0.25 * _spread_across_turbines(raw)
