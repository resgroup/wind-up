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
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            availability_col=_AVAIL,
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
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            availability_col=_AVAIL,
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
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            availability_col=_AVAIL,
            direction_col=_NORTHED_DIR,
        )
        assert not any(c.endswith(f"{QUALIFIER}T1") for c in feats.columns)

    def test_missing_northed_column_raises_naming_the_shared_step(self) -> None:
        idx = _index(12)
        with pytest.raises(ValueError, match=_NORTHED_DIR):
            build_reference_features(
                _scada(idx),  # no direction columns at all
                test_wtg="T1",
                turbine_col=_TURBINE,
                active_power_col=_POWER,
                availability_col=_AVAIL,
                direction_col=_NORTHED_DIR,
            )

    def test_a_raw_direction_in_extra_cols_is_dropped_for_the_northed_one(self) -> None:
        idx = _index(12)
        feats = build_reference_features(
            _scada_with_direction(idx),
            test_wtg="T1",
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            availability_col=_AVAIL,
            extra_cols=(_RAW_DIR,),
            direction_col=_NORTHED_DIR,
        )
        assert not any(c.startswith(_RAW_DIR) for c in feats.columns)
        assert any(c.startswith(f"{_NORTHED_DIR}_sin") for c in feats.columns)

    def test_omitting_direction_col_keeps_the_previous_feature_set(self) -> None:
        idx = _index(12)
        scada = _scada_with_direction(idx)
        without = build_reference_features(
            scada, test_wtg="T1", turbine_col=_TURBINE, active_power_col=_POWER, availability_col=_AVAIL
        )
        assert len(without.columns) == 6
        assert not any("northed" in c for c in without.columns)


_RATED = 2300.0
_WAKING_THRESHOLD_KW = 0.05 * _RATED


def _scada_spanning_the_waking_threshold(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Long SCADA whose power straddles 5% of rated, so the waking boolean is not degenerate."""
    scada = _scada(idx)
    ramp = np.linspace(0.0, 4 * _WAKING_THRESHOLD_KW, len(idx))
    for name in ("T1", "R1", "R2", "R3"):
        scada.loc[scada[_TURBINE] == name, _POWER] = ramp
    scada[_RAW_DIR] = 90.0
    scada[_NORTHED_DIR] = 120.0
    return scada


class TestPowerFreeReferences:
    """A screened reference keeps its wake geometry but loses the channels a Cp change corrupts."""

    def _features(self, idx: pd.DatetimeIndex, power_free: tuple[str, ...]) -> pd.DataFrame:
        return build_reference_features(
            _scada_spanning_the_waking_threshold(idx),
            test_wtg="T1",
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            availability_col=_AVAIL,
            direction_col=_NORTHED_DIR,
            include_availability=False,
            power_free=power_free,
            waking_threshold_kw=_WAKING_THRESHOLD_KW,
        )

    def test_a_power_free_reference_loses_its_power_column(self) -> None:
        feats = self._features(_index(24), power_free=("R1",))
        assert f"{_POWER}{QUALIFIER}R1" not in feats.columns

    def test_the_other_references_keep_theirs(self) -> None:
        feats = self._features(_index(24), power_free=("R1",))
        for ref in ("R2", "R3"):
            assert f"{_POWER}{QUALIFIER}{ref}" in feats.columns

    def test_a_power_free_reference_keeps_its_direction(self) -> None:
        """Whether it is casting a wake on its neighbours is a function of where it points."""
        feats = self._features(_index(24), power_free=("R1",))
        assert f"{_NORTHED_DIR}_sin{QUALIFIER}R1" in feats.columns
        assert f"{_NORTHED_DIR}_cos{QUALIFIER}R1" in feats.columns

    def test_a_power_free_reference_gains_a_waking_boolean(self) -> None:
        feats = self._features(_index(24), power_free=("R1",))
        assert f"waking_{_POWER}{QUALIFIER}R1" in feats.columns

    def test_the_waking_boolean_is_true_above_the_threshold_and_false_below(self) -> None:
        idx = _index(24)
        feats = self._features(idx, power_free=("R1",))
        power = _scada_spanning_the_waking_threshold(idx)
        r1 = power[power[_TURBINE] == "R1"][_POWER].to_numpy()
        waking = feats[f"waking_{_POWER}{QUALIFIER}R1"].to_numpy()
        assert (waking == (r1 >= _WAKING_THRESHOLD_KW)).all()

    def test_the_boolean_is_informative_not_degenerate_on_this_fixture(self) -> None:
        feats = self._features(_index(24), power_free=("R1",))
        waking = feats[f"waking_{_POWER}{QUALIFIER}R1"]
        assert 0 < waking.sum() < len(waking)

    def test_good_references_get_no_waking_column(self) -> None:
        """The boolean replaces power where power was removed; it is not a new default feature."""
        feats = self._features(_index(24), power_free=("R1",))
        for ref in ("R2", "R3"):
            assert f"waking_{_POWER}{QUALIFIER}{ref}" not in feats.columns

    def test_no_power_free_references_leaves_the_matrix_untouched(self) -> None:
        """The screen finding nothing must cost the feature matrix nothing."""
        idx = _index(24)
        unscreened = build_reference_features(
            _scada_spanning_the_waking_threshold(idx),
            test_wtg="T1",
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            availability_col=_AVAIL,
            direction_col=_NORTHED_DIR,
            include_availability=False,
        )
        assert list(self._features(idx, power_free=()).columns) == list(unscreened.columns)

    def test_a_power_free_turbine_that_is_not_a_reference_raises(self) -> None:
        with pytest.raises(ValueError, match="R9"):
            self._features(_index(24), power_free=("R9",))

    def test_the_test_turbine_still_contributes_nothing(self) -> None:
        feats = self._features(_index(24), power_free=("R1",))
        assert not any(c.endswith(f"{QUALIFIER}T1") for c in feats.columns)


class TestWakingDtype:
    """The waking column has to survive reindexing onto the full index as a numeric feature."""

    def _features_with_a_gap(self) -> pd.DataFrame:
        idx = _index(24)
        scada = _scada_spanning_the_waking_threshold(idx)
        # R1 misses the last six timestamps, so joining its waking column leaves gaps.
        drop = (scada[_TURBINE] == "R1") & (scada.index >= idx[-6])
        return build_reference_features(
            scada[~drop],
            test_wtg="T1",
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            availability_col=_AVAIL,
            direction_col=_NORTHED_DIR,
            include_availability=False,
            power_free=("R1",),
            waking_threshold_kw=_WAKING_THRESHOLD_KW,
        )

    def test_the_waking_column_is_numeric_not_object(self) -> None:
        """LightGBM rejects an object column, which is what bool + NaN collapses to."""
        feats = self._features_with_a_gap()
        assert feats[f"waking_{_POWER}{QUALIFIER}R1"].dtype.kind == "f"

    def test_a_missing_record_is_nan_rather_than_asserted_not_waking(self) -> None:
        """No record is no knowledge; NaN is preserved for LightGBM as the rest of the matrix is."""
        feats = self._features_with_a_gap()
        assert feats[f"waking_{_POWER}{QUALIFIER}R1"].isna().any()

    def test_every_feature_column_is_numeric(self) -> None:
        feats = self._features_with_a_gap()
        bad = [c for c in feats.columns if feats[c].dtype.kind not in "fiub"]
        assert bad == []
