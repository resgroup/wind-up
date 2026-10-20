"""Tests for the power-model feature builder (method step 6).

Covers the reference feature pivot (active power per reference, original tag names, NaN-preserving),
the waking scenario categoricals, the solar elevation, the outcome extraction and the
reference-only guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.power_model.features import (
    QUALIFIER,
    SOLAR_ELEVATION,
    build_reference_features,
    check_reference_only,
    extract_outcome,
    solar_elevation_frame,
    waking_scenario_features,
    without_constant_scenarios,
)
from benchmarking.harness.waking import WAKING_SCENARIO_COL

_TURBINE = "TurbineName"
_REFS = ("R1", "R2", "R3")
_POWER = "wtc_ActPower_mean"
_AVAIL = "wtc_ScReToOp_timeon"
_WS = "wtc_AcWindSp_mean"


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC", name="timestamp")


def _scada(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Long SCADA with test T1 and references R1, R2, R3, each carrying power + availability + ws."""
    rng = np.random.default_rng(0)
    frames = [
        pd.DataFrame(
            {
                _TURBINE: name,
                _POWER: rng.normal(800, 100, len(idx)),
                _AVAIL: 600.0,
                _WS: rng.normal(8, 2, len(idx)),
            },
            index=idx,
        )
        for name in ("T1", "R1", "R2", "R3")
    ]
    return pd.concat(frames)


class TestBuildReferenceFeatures:
    def test_only_active_power_per_reference(self) -> None:
        idx = _index(12)
        feats = build_reference_features(
            _scada(idx),
            test_wtg="T1",
            references=_REFS,
            turbine_col=_TURBINE,
            active_power_col=_POWER,
        )
        # one column per reference; none for T1; ws and availability not included
        assert list(feats.columns) == [f"{_POWER}{QUALIFIER}{r}" for r in _REFS]
        assert not any(c.startswith((_WS, _AVAIL)) for c in feats.columns)
        assert feats.index.equals(idx)

    def test_keeps_original_tag_names(self) -> None:
        idx = _index(12)
        feats = build_reference_features(
            _scada(idx),
            test_wtg="T1",
            references=_REFS,
            turbine_col=_TURBINE,
            active_power_col=_POWER,
        )
        assert f"{_POWER}{QUALIFIER}R1" in feats.columns

    def test_preserves_nan_rows(self) -> None:
        idx = _index(12)
        scada = _scada(idx)
        scada.loc[(scada[_TURBINE] == "R1") & (scada.index == idx[3]), _POWER] = np.nan
        feats = build_reference_features(
            scada,
            test_wtg="T1",
            references=_REFS,
            turbine_col=_TURBINE,
            active_power_col=_POWER,
        )
        assert len(feats) == len(idx)
        assert np.isnan(feats.loc[idx[3], f"{_POWER}{QUALIFIER}R1"])

    def test_raises_when_the_pool_is_empty(self) -> None:
        idx = _index(12)
        with pytest.raises(ValueError, match="no references supplied"):
            build_reference_features(
                _scada(idx),
                test_wtg="T1",
                references=(),
                turbine_col=_TURBINE,
                active_power_col=_POWER,
            )

    def test_a_turbine_outside_the_pool_contributes_nothing(self) -> None:
        idx = _index(12)
        feats = build_reference_features(
            _scada(idx),
            test_wtg="T1",
            references=("R1", "R2"),
            turbine_col=_TURBINE,
            active_power_col=_POWER,
        )
        # R3 is in the frame but not offered as a reference, so it is not one
        assert not any(c.endswith(f"{QUALIFIER}R3") for c in feats.columns)
        assert len(feats.columns) == 2

    def test_extra_cols_add_per_reference_features(self) -> None:
        idx = _index(12)
        scada = _scada(idx)
        scada["wtc_ActPower_stddev"] = 7.0
        feats = build_reference_features(
            scada,
            test_wtg="T1",
            references=_REFS,
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            extra_cols=("wtc_ActPower_stddev",),
        )
        # three references x two value columns; still nothing from the test turbine
        assert len(feats.columns) == 6
        assert f"wtc_ActPower_stddev{QUALIFIER}R2" in feats.columns
        assert not any(c.endswith(f"{QUALIFIER}T1") for c in feats.columns)

    def test_missing_extra_col_raises(self) -> None:
        idx = _index(12)
        with pytest.raises(ValueError, match="missing required reference-feature columns"):
            build_reference_features(
                _scada(idx),
                test_wtg="T1",
                references=_REFS,
                turbine_col=_TURBINE,
                active_power_col=_POWER,
                extra_cols=("wtc_ActPower_stddev",),
            )

    def test_extra_test_turbine_column_never_reaches_features(self) -> None:
        idx = _index(12)
        scada = _scada(idx)
        # a leak-bait column only present on the test turbine must not appear among features
        scada["wtc_NacWdSp_mean"] = np.where(scada[_TURBINE] == "T1", scada[_POWER], np.nan)
        feats = build_reference_features(
            scada,
            test_wtg="T1",
            references=_REFS,
            turbine_col=_TURBINE,
            active_power_col=_POWER,
        )
        assert not any("NacWdSp" in c for c in feats.columns)
        assert not any(c.endswith(f"{QUALIFIER}T1") for c in feats.columns)


# Denver, CO, and sunrise and sunset there on 19 May 2025 (timeanddate.com, in UTC).
_DENVER = {"latitude": 39.7392, "longitude": -104.9903}
_DENVER_SUNRISE = pd.Timestamp("2025-05-19 11:42", tz="UTC")
_DENVER_SUNSET = pd.Timestamp("2025-05-20 02:11", tz="UTC")
_HILL_OF_TOWIE = {"latitude": 57.506, "longitude": -3.068}
_TEN_MINUTES = pd.Timedelta(minutes=10)


def _elevation(starts: list[pd.Timestamp], *, timebase: pd.Timedelta = _TEN_MINUTES, **site: float) -> np.ndarray:
    index = pd.DatetimeIndex(starts)
    return solar_elevation_frame(index, timebase=timebase, **site)[SOLAR_ELEVATION].to_numpy()


class TestSolarElevationFrame:
    def test_the_sun_is_on_the_horizon_at_sunrise_and_sunset(self) -> None:
        # Each period starts five minutes before the event, so its middle is the event.
        starts = [_DENVER_SUNRISE - _TEN_MINUTES / 2, _DENVER_SUNSET - _TEN_MINUTES / 2]
        np.testing.assert_allclose(_elevation(starts, **_DENVER), [0.0, 0.0], atol=0.2)

    def test_it_is_evaluated_at_the_middle_of_each_period(self) -> None:
        hourly = _elevation([_DENVER_SUNRISE - pd.Timedelta(minutes=30)], timebase=pd.Timedelta(hours=1), **_DENVER)
        instant = _elevation([_DENVER_SUNRISE], timebase=pd.Timedelta(0), **_DENVER)
        assert hourly == pytest.approx(instant)
        start = _elevation([_DENVER_SUNRISE - pd.Timedelta(minutes=30)], timebase=pd.Timedelta(0), **_DENVER)
        assert abs(start[0] - instant[0]) > 1.0

    def test_a_timestamp_in_another_zone_is_the_same_instant(self) -> None:
        utc = _elevation([_DENVER_SUNRISE], **_DENVER)
        local = _elevation([_DENVER_SUNRISE.tz_convert("America/Denver")], **_DENVER)
        np.testing.assert_allclose(local, utc)

    def test_the_index_is_kept(self) -> None:
        index = _index(6)
        out = solar_elevation_frame(index, timebase=_TEN_MINUTES, **_HILL_OF_TOWIE)
        assert list(out.columns) == [SOLAR_ELEVATION]
        assert out.index.equals(index)

    def test_hill_of_towie_midsummer(self) -> None:
        noon, midnight = _elevation(
            [pd.Timestamp("2018-06-21 12:10", tz="UTC"), pd.Timestamp("2018-06-21 00:10", tz="UTC")],
            **_HILL_OF_TOWIE,
        )
        assert noon == pytest.approx(55.9, abs=0.5)
        assert midnight < 0.0

    def test_raises_on_timezone_naive_timestamps(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            _elevation([pd.Timestamp("2025-05-19 11:42")], **_DENVER)


class TestOutcomeAndGuard:
    def test_extract_outcome_returns_test_power_on_index(self) -> None:
        idx = _index(10)
        scada = _scada(idx)
        y = extract_outcome(scada, test_wtg="T1", turbine_col=_TURBINE, active_power_col=_POWER)
        assert y.index.equals(idx)
        expected = scada[scada[_TURBINE] == "T1"][_POWER].to_numpy()
        np.testing.assert_allclose(y.to_numpy(), expected)

    def test_guard_rejects_test_turbine_feature(self) -> None:
        with pytest.raises(ValueError, match="reference-only rule violated"):
            check_reference_only([f"{_POWER}{QUALIFIER}R1", f"{_POWER}{QUALIFIER}T1"], test_wtg="T1")

    def test_guard_passes_for_reference_only_features(self) -> None:
        check_reference_only([f"{_POWER}{QUALIFIER}R1", "temperature_2m"], test_wtg="T1")


_NORTHED_DIR = "northed_wtc_NacPos_mean"
_RAW_DIR = "wtc_NacPos_mean"


def _scada_with_direction(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """The long SCADA above, plus a raw nacelle position and its northed counterpart."""
    scada = _scada(idx)
    rng = np.random.default_rng(1)
    scada[_RAW_DIR] = rng.uniform(0, 360, len(scada))
    scada[_NORTHED_DIR] = (scada[_RAW_DIR] + 30.0) % 360.0
    return scada


class TestReferenceDirectionFeature:
    """Each reference's northed direction enters as sin/cos; the raw column never does."""

    def test_direction_enters_as_sin_and_cos_per_reference(self) -> None:
        idx = _index(12)
        feats = build_reference_features(
            _scada_with_direction(idx),
            test_wtg="T1",
            references=_REFS,
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            direction_col=_NORTHED_DIR,
        )
        for ref in ("R1", "R2", "R3"):
            assert f"{_NORTHED_DIR}_sin{QUALIFIER}{ref}" in feats.columns
            assert f"{_NORTHED_DIR}_cos{QUALIFIER}{ref}" in feats.columns
        # the raw degree column is not a feature: LightGBM cannot see that 359 deg is next to 1 deg
        assert not any(c.startswith(f"{_NORTHED_DIR}{QUALIFIER}") for c in feats.columns)
        assert not any(c.startswith(_RAW_DIR) for c in feats.columns)

    def test_sin_cos_values_are_the_direction_on_the_unit_circle(self) -> None:
        idx = _index(6)
        scada = _scada_with_direction(idx)
        feats = build_reference_features(
            scada,
            test_wtg="T1",
            references=_REFS,
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            direction_col=_NORTHED_DIR,
        )
        expected = scada[scada[_TURBINE] == "R1"][_NORTHED_DIR].to_numpy(dtype=float)
        assert feats[f"{_NORTHED_DIR}_sin{QUALIFIER}R1"].to_numpy() == pytest.approx(np.sin(np.deg2rad(expected)))
        assert feats[f"{_NORTHED_DIR}_cos{QUALIFIER}R1"].to_numpy() == pytest.approx(np.cos(np.deg2rad(expected)))

    def test_test_turbine_direction_is_never_a_feature(self) -> None:
        idx = _index(12)
        feats = build_reference_features(
            _scada_with_direction(idx),
            test_wtg="T1",
            references=_REFS,
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            direction_col=_NORTHED_DIR,
        )
        assert not any(c.endswith(f"{QUALIFIER}T1") for c in feats.columns)

    def test_missing_northed_column_raises_naming_the_shared_step(self) -> None:
        idx = _index(12)
        with pytest.raises(ValueError, match=_NORTHED_DIR):
            build_reference_features(
                _scada(idx),  # no direction columns at all
                test_wtg="T1",
                references=_REFS,
                turbine_col=_TURBINE,
                active_power_col=_POWER,
                direction_col=_NORTHED_DIR,
            )

    def test_a_raw_direction_in_extra_cols_is_dropped_for_the_northed_one(self) -> None:
        idx = _index(12)
        feats = build_reference_features(
            _scada_with_direction(idx),
            test_wtg="T1",
            references=_REFS,
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            extra_cols=(_RAW_DIR,),
            direction_col=_NORTHED_DIR,
        )
        assert not any(c.startswith(_RAW_DIR) for c in feats.columns)
        assert any(c.startswith(f"{_NORTHED_DIR}_sin") for c in feats.columns)

    def test_omitting_direction_col_keeps_the_previous_feature_set(self) -> None:
        idx = _index(12)
        scada = _scada_with_direction(idx)
        without = build_reference_features(
            scada,
            test_wtg="T1",
            references=_REFS,
            turbine_col=_TURBINE,
            active_power_col=_POWER,
        )
        assert len(without.columns) == 3
        assert not any("northed" in c for c in without.columns)


def _scada_with_scenarios(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """The long SCADA above with a waking scenario per record."""
    scada = _scada(idx)
    labels = np.array(["normal", "R2 not waking", "unknown"])
    scada[WAKING_SCENARIO_COL] = labels[np.arange(len(scada)) % len(labels)]
    return scada


class TestWakingScenarioFeatures:
    def test_one_column_for_the_test_turbine_and_each_reference(self) -> None:
        feats = waking_scenario_features(
            _scada_with_scenarios(_index(12)), test_wtg="T1", references=("R2", "R1"), turbine_col=_TURBINE
        )
        assert list(feats.columns) == [f"{WAKING_SCENARIO_COL}{QUALIFIER}{t}" for t in ("test", "R1", "R2")]

    def test_columns_are_categoricals_sharing_one_dtype(self) -> None:
        feats = waking_scenario_features(
            _scada_with_scenarios(_index(12)), test_wtg="T1", references=_REFS, turbine_col=_TURBINE
        )
        dtypes = {str(feats[c].dtype) for c in feats.columns}
        assert dtypes == {"category"}
        assert len({tuple(feats[c].cat.categories) for c in feats.columns}) == 1

    def test_values_are_each_turbines_own_scenario(self) -> None:
        idx = _index(12)
        scada = _scada_with_scenarios(idx)
        feats = waking_scenario_features(scada, test_wtg="T1", references=_REFS, turbine_col=_TURBINE)
        expected = scada[scada[_TURBINE] == "R2"][WAKING_SCENARIO_COL].to_numpy()
        assert feats[f"{WAKING_SCENARIO_COL}{QUALIFIER}R2"].astype(str).to_numpy().tolist() == expected.tolist()

    def test_a_missing_record_is_nan(self) -> None:
        idx = _index(12)
        scada = _scada_with_scenarios(idx)
        scada = scada[~((scada[_TURBINE] == "R1") & (scada.index == idx[4]))]
        feats = waking_scenario_features(scada, test_wtg="T1", references=_REFS, turbine_col=_TURBINE)
        assert pd.isna(feats.loc[idx[4], f"{WAKING_SCENARIO_COL}{QUALIFIER}R1"])

    def test_passes_the_reference_only_guard(self) -> None:
        feats = waking_scenario_features(
            _scada_with_scenarios(_index(12)), test_wtg="T1", references=_REFS, turbine_col=_TURBINE
        )
        check_reference_only(feats.columns.tolist(), test_wtg="T1")

    def test_raises_without_the_scenario_column(self) -> None:
        with pytest.raises(ValueError, match=WAKING_SCENARIO_COL):
            waking_scenario_features(_scada(_index(12)), test_wtg="T1", references=_REFS, turbine_col=_TURBINE)


def test_a_scenario_column_with_one_value_over_the_rows_is_left_out() -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="10min", tz="UTC")
    dtype = pd.CategoricalDtype(["T03 not waking", "normal"])
    features = pd.DataFrame(
        {
            f"{WAKING_SCENARIO_COL}{QUALIFIER}test": pd.Categorical(["normal"] * 3 + ["T03 not waking"], dtype=dtype),
            f"{WAKING_SCENARIO_COL}{QUALIFIER}T02": pd.Categorical(["normal", "T03 not waking"] * 2, dtype=dtype),
            f"power{QUALIFIER}T02": [1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )
    kept, constant = without_constant_scenarios(features, rows=np.array([True, True, True, False]))
    assert constant == [f"{WAKING_SCENARIO_COL}{QUALIFIER}test"]
    assert list(kept.columns) == [f"{WAKING_SCENARIO_COL}{QUALIFIER}T02", f"power{QUALIFIER}T02"]
