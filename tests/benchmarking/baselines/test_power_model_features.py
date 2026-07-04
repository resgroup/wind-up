"""Tests for the power-model curated, reference-only feature builder.

Covers the reference feature pivot (active power + availability per reference, original tag names,
NaN-preserving), the ERA5 all-columns passthrough with direction sin/cos, the outcome extraction,
and the reference-only enforcement guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.era5_sync import ERA5_WD, ERA5_WS
from benchmarking.baselines.power_model.features import (
    QUALIFIER,
    build_reference_features,
    check_reference_only,
    era5_feature_frame,
    extract_outcome,
)

_TURBINE = "TurbineName"
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
    def test_only_active_power_and_availability_per_reference(self) -> None:
        idx = _index(12)
        feats = build_reference_features(
            _scada(idx), test_wtg="T1", turbine_col=_TURBINE, active_power_col=_POWER, availability_col=_AVAIL
        )
        # three references x two value columns = six feature columns; none for T1; ws not included
        assert len(feats.columns) == 6
        assert not any(c.endswith(f"{QUALIFIER}T1") for c in feats.columns)
        assert not any(c.startswith(_WS) for c in feats.columns)
        assert feats.index.equals(idx)

    def test_keeps_original_tag_names(self) -> None:
        idx = _index(12)
        feats = build_reference_features(
            _scada(idx), test_wtg="T1", turbine_col=_TURBINE, active_power_col=_POWER, availability_col=_AVAIL
        )
        assert f"{_POWER}{QUALIFIER}R1" in feats.columns
        assert f"{_AVAIL}{QUALIFIER}R3" in feats.columns

    def test_preserves_nan_rows(self) -> None:
        idx = _index(12)
        scada = _scada(idx)
        scada.loc[(scada[_TURBINE] == "R1") & (scada.index == idx[3]), _POWER] = np.nan
        feats = build_reference_features(
            scada, test_wtg="T1", turbine_col=_TURBINE, active_power_col=_POWER, availability_col=_AVAIL
        )
        assert len(feats) == len(idx)
        assert np.isnan(feats.loc[idx[3], f"{_POWER}{QUALIFIER}R1"])

    def test_raises_when_no_references(self) -> None:
        idx = _index(12)
        only_test = _scada(idx)
        only_test = only_test[only_test[_TURBINE] == "T1"]
        with pytest.raises(ValueError, match="reference"):
            build_reference_features(
                only_test, test_wtg="T1", turbine_col=_TURBINE, active_power_col=_POWER, availability_col=_AVAIL
            )

    def test_extra_cols_add_per_reference_features(self) -> None:
        idx = _index(12)
        scada = _scada(idx)
        scada["wtc_ActPower_stddev"] = 7.0
        feats = build_reference_features(
            scada,
            test_wtg="T1",
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            availability_col=_AVAIL,
            extra_cols=("wtc_ActPower_stddev",),
        )
        # three references x three value columns; still nothing from the test turbine
        assert len(feats.columns) == 9
        assert f"wtc_ActPower_stddev{QUALIFIER}R2" in feats.columns
        assert not any(c.endswith(f"{QUALIFIER}T1") for c in feats.columns)

    def test_missing_availability_col_raises_even_when_not_featured(self) -> None:
        idx = _index(12)
        scada = _scada(idx).drop(columns=[_AVAIL])
        with pytest.raises(ValueError, match="missing required reference-feature columns"):
            build_reference_features(
                scada,
                test_wtg="T1",
                turbine_col=_TURBINE,
                active_power_col=_POWER,
                availability_col=_AVAIL,
                include_availability=False,
            )

    def test_missing_extra_col_raises(self) -> None:
        idx = _index(12)
        with pytest.raises(ValueError, match="missing required reference-feature columns"):
            build_reference_features(
                _scada(idx),
                test_wtg="T1",
                turbine_col=_TURBINE,
                active_power_col=_POWER,
                availability_col=_AVAIL,
                extra_cols=("wtc_ActPower_stddev",),
            )

    def test_extra_test_turbine_column_never_reaches_features(self) -> None:
        idx = _index(12)
        scada = _scada(idx)
        # a leak-bait column only present on the test turbine must not appear among features
        scada["wtc_NacWdSp_mean"] = np.where(scada[_TURBINE] == "T1", scada[_POWER], np.nan)
        feats = build_reference_features(
            scada, test_wtg="T1", turbine_col=_TURBINE, active_power_col=_POWER, availability_col=_AVAIL
        )
        assert not any("NacWdSp" in c for c in feats.columns)
        assert not any(c.endswith(f"{QUALIFIER}T1") for c in feats.columns)


class TestEra5FeatureFrame:
    def test_passes_through_raw_drops_aliases_and_adds_direction_sin_cos(self) -> None:
        idx = _index(6)
        aligned = pd.DataFrame(
            {
                "wind_speed_100m": np.linspace(5, 10, 6),
                "wind_direction_100m": np.linspace(0, 270, 6),
                "temperature_2m": np.linspace(1, 6, 6),
                ERA5_WS: np.linspace(5, 10, 6),
                ERA5_WD: np.linspace(0, 270, 6),
            },
            index=idx,
        )
        out = era5_feature_frame(aligned)
        # aliases dropped; raw columns kept; direction gains sin/cos companions (raw degrees kept)
        assert ERA5_WS not in out.columns
        assert ERA5_WD not in out.columns
        assert {"wind_speed_100m", "wind_direction_100m", "temperature_2m"} <= set(out.columns)
        assert {"wind_direction_100m_sin", "wind_direction_100m_cos"} <= set(out.columns)
        assert "wind_speed_100m_sin" not in out.columns
        # sin/cos are consistent with the raw degrees
        np.testing.assert_allclose(
            out["wind_direction_100m_sin"].to_numpy(), np.sin(np.deg2rad(aligned["wind_direction_100m"].to_numpy()))
        )


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
