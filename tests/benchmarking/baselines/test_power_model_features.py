"""Tests for the power-model feature builder (method step 6).

Covers the reference feature pivot (active power per reference, original tag names, NaN-preserving),
the waking scenario categoricals, the ERA5 feature list, the outcome extraction and the
reference-only guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.power_model.features import (
    ERA5_FEATURES,
    QUALIFIER,
    build_reference_features,
    check_reference_only,
    era5_feature_frame,
    extract_outcome,
    waking_scenario_features,
)
from benchmarking.harness.reanalysis import ERA5_WD, ERA5_WS
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


def _aligned_era5(idx: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(idx)
    return pd.DataFrame(
        {
            "wind_speed_100m": np.linspace(5, 10, n),
            "wind_direction_100m": np.linspace(0, 270, n),
            "wind_speed_10m": np.linspace(4, 8, n),
            "wind_direction_10m": np.linspace(0, 270, n),
            "wind_gusts_10m": np.linspace(6, 12, n),
            "temperature_2m": np.linspace(1, 6, n),
            "surface_pressure": np.linspace(990, 1010, n),
            "relative_humidity_2m": np.linspace(60, 90, n),
            "cloud_cover": np.linspace(0, 100, n),
            ERA5_WS: np.linspace(5, 10, n),
            ERA5_WD: np.linspace(0, 270, n),
        },
        index=idx,
    )


class TestEra5FeatureFrame:
    def test_returns_exactly_the_era5_feature_list(self) -> None:
        out = era5_feature_frame(_aligned_era5(_index(6)))
        assert list(out.columns) == list(ERA5_FEATURES)

    def test_direction_is_sin_and_cos_of_the_100m_direction(self) -> None:
        aligned = _aligned_era5(_index(6))
        out = era5_feature_frame(aligned)
        rad = np.deg2rad(aligned["wind_direction_100m"].to_numpy())
        np.testing.assert_allclose(out["wind_direction_100m_sin"].to_numpy(), np.sin(rad))
        np.testing.assert_allclose(out["wind_direction_100m_cos"].to_numpy(), np.cos(rad))

    def test_shear_and_gust_ratio_follow_their_definitions(self) -> None:
        aligned = _aligned_era5(_index(6))
        out = era5_feature_frame(aligned)
        ws10 = aligned["wind_speed_10m"].to_numpy()
        np.testing.assert_allclose(
            out["shear_exponent"].to_numpy(), np.log(aligned["wind_speed_100m"].to_numpy() / ws10) / np.log(10.0)
        )
        np.testing.assert_allclose(out["gust_ratio"].to_numpy(), aligned["wind_gusts_10m"].to_numpy() / ws10)

    def test_air_density_is_plausible(self) -> None:
        out = era5_feature_frame(_aligned_era5(_index(6)))
        assert out["air_density"].between(1.2, 1.3).all()

    def test_raises_without_an_input(self) -> None:
        with pytest.raises(ValueError, match="surface_pressure"):
            era5_feature_frame(_aligned_era5(_index(6)).drop(columns=["surface_pressure"]))


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
