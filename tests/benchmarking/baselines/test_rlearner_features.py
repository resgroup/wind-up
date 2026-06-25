"""Tests for the R-learner upgrade-invariant feature builder.

The builder turns long source-native SCADA into a wide feature matrix from reference
turbines only (never the test turbine's own signals), keeping original tag names, plus the
outcome/treatment extraction, the ERA5 sin/cos transform, the enforcement guard, and the
(currently no-op) feature-engineering seam.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.rlearner.era5_sync import ERA5_WD, ERA5_WS
from benchmarking.baselines.rlearner.features import (
    QUALIFIER,
    build_reference_features,
    check_upgrade_invariant,
    engineered_reference_features,
    era5_features,
    extract_outcome_and_treatment,
)
from benchmarking.synthetic import ToggleSchedule, treated_mask

_TURBINE = "TurbineName"
_POWER = "wtc_ActPower_mean"
_WS = "wtc_AcWindSp_mean"


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC", name="timestamp")


def _scada(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Long SCADA with test T1 and references R1, R2, each carrying power + wind speed."""
    rng = np.random.default_rng(0)
    frames = [
        pd.DataFrame(
            {_TURBINE: name, _POWER: rng.normal(800, 100, len(idx)), _WS: rng.normal(8, 2, len(idx))},
            index=idx,
        )
        for name in ("T1", "R1", "R2")
    ]
    return pd.concat(frames)


class TestBuildReferenceFeatures:
    def test_includes_only_reference_turbines(self) -> None:
        idx = _index(12)
        feats = build_reference_features(_scada(idx), test_wtg="T1", turbine_col=_TURBINE)
        # two references x two value columns = four feature columns; none for T1
        assert len(feats.columns) == 4
        assert not any(c.endswith(f"{QUALIFIER}T1") for c in feats.columns)
        assert feats.index.equals(idx)

    def test_keeps_original_tag_names(self) -> None:
        idx = _index(12)
        feats = build_reference_features(_scada(idx), test_wtg="T1", turbine_col=_TURBINE)
        assert f"{_POWER}{QUALIFIER}R1" in feats.columns
        assert f"{_WS}{QUALIFIER}R2" in feats.columns

    def test_preserves_nan_rows(self) -> None:
        idx = _index(12)
        scada = _scada(idx)
        scada.loc[(scada[_TURBINE] == "R1") & (scada.index == idx[3]), _WS] = np.nan
        feats = build_reference_features(scada, test_wtg="T1", turbine_col=_TURBINE)
        # the NaN is preserved (no complete-case dropping) and the row is kept
        assert len(feats) == len(idx)
        assert np.isnan(feats.loc[idx[3], f"{_WS}{QUALIFIER}R1"])

    def test_raises_when_no_references(self) -> None:
        idx = _index(12)
        only_test = _scada(idx)
        only_test = only_test[only_test[_TURBINE] == "T1"]
        with pytest.raises(ValueError, match="reference"):
            build_reference_features(only_test, test_wtg="T1", turbine_col=_TURBINE)


class TestGuard:
    def test_rejects_test_turbine_column(self) -> None:
        names = [f"{_POWER}{QUALIFIER}R1", f"{_WS}{QUALIFIER}T1"]
        with pytest.raises(ValueError, match="upgrade-invariant"):
            check_upgrade_invariant(names, test_wtg="T1")

    def test_passes_reference_only_columns(self) -> None:
        names = [f"{_POWER}{QUALIFIER}R1", f"{_WS}{QUALIFIER}R2"]
        check_upgrade_invariant(names, test_wtg="T1")  # no raise


class TestOutcomeTreatment:
    def test_prepost_outcome_and_treatment(self) -> None:
        idx = _index(20)
        scada = _scada(idx)
        upgrade = idx[10]
        y, t = extract_outcome_and_treatment(
            scada, test_wtg="T1", turbine_col=_TURBINE, active_power_col=_POWER, upgrade_timing=upgrade
        )
        expected_y = scada.loc[scada[_TURBINE] == "T1", _POWER]
        assert y.to_numpy() == pytest.approx(expected_y.to_numpy())
        assert t.to_numpy().tolist() == [0] * 10 + [1] * 10

    def test_toggle_treatment(self) -> None:
        idx = _index(40)
        scada = _scada(idx)
        schedule = ToggleSchedule(period=pd.Timedelta(minutes=20), start=idx[0])
        _, t = extract_outcome_and_treatment(
            scada, test_wtg="T1", turbine_col=_TURBINE, active_power_col=_POWER, upgrade_timing=schedule
        )
        assert t.to_numpy().tolist() == np.asarray(treated_mask(idx, schedule)).astype(int).tolist()


class TestEra5Features:
    def test_wd_becomes_sin_cos(self) -> None:
        idx = _index(4)
        aligned = pd.DataFrame({ERA5_WS: [8.0, 9.0, 10.0, 11.0], ERA5_WD: [0.0, 90.0, 180.0, 270.0]}, index=idx)
        feats = era5_features(aligned)
        assert list(feats.columns) == [ERA5_WS, "era5_wd_sin", "era5_wd_cos"]
        assert feats["era5_wd_sin"].to_numpy() == pytest.approx([0.0, 1.0, 0.0, -1.0], abs=1e-9)
        assert feats["era5_wd_cos"].to_numpy() == pytest.approx([1.0, 0.0, -1.0, 0.0], abs=1e-9)


class TestEngineeredSeam:
    def test_currently_adds_no_columns(self) -> None:
        idx = _index(12)
        eng = engineered_reference_features(_scada(idx), test_wtg="T1", turbine_col=_TURBINE)
        # the seam exists (north-corrected yaw etc. go here later) but adds nothing yet
        assert list(eng.columns) == []
        assert eng.index.equals(idx)
