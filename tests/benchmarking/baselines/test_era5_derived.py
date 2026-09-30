"""Known-input tests for the shared ERA5 derivation utility (Issue 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.era5_derived import (
    ERA5_DERIVATIONS,
    air_density,
    era5_derived_frame,
    gust_margin,
    gust_ratio,
    hub_height_wind_speed,
    shear_exponent,
    vertical_veer,
)

_HOT_HUB_HEIGHT_M = 59.0


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="10min", tz="UTC"))


class TestShearExponent:
    def test_known_value(self) -> None:
        # ws doubles from 10 m to 100 m -> alpha = ln(2)/ln(10)
        alpha = shear_exponent(_series([5.0]), _series([10.0]))
        np.testing.assert_allclose(alpha.to_numpy(), np.log(2.0) / np.log(10.0))

    def test_zero_shear(self) -> None:
        alpha = shear_exponent(_series([8.0]), _series([8.0]))
        np.testing.assert_allclose(alpha.to_numpy(), 0.0)

    def test_calm_is_nan(self) -> None:
        alpha = shear_exponent(_series([0.0, -1.0, 5.0]), _series([5.0, 5.0, 0.0]))
        assert alpha.isna().all()


class TestHubHeightWindSpeed:
    def test_power_law_interpolation(self) -> None:
        # alpha = 0.2, ws_100m = 10 -> ws_59m = 10 * 0.59^0.2
        ws = hub_height_wind_speed(_series([10.0]), _series([0.2]), hub_height_m=_HOT_HUB_HEIGHT_M)
        np.testing.assert_allclose(ws.to_numpy(), 10.0 * 0.59**0.2)

    def test_zero_alpha_keeps_speed(self) -> None:
        ws = hub_height_wind_speed(_series([7.0]), _series([0.0]), hub_height_m=_HOT_HUB_HEIGHT_M)
        np.testing.assert_allclose(ws.to_numpy(), 7.0)

    def test_nan_alpha_propagates(self) -> None:
        ws = hub_height_wind_speed(_series([7.0]), _series([np.nan]), hub_height_m=_HOT_HUB_HEIGHT_M)
        assert ws.isna().all()


class TestGustRatio:
    def test_known_value(self) -> None:
        ratio = gust_ratio(_series([7.5]), _series([5.0]))
        np.testing.assert_allclose(ratio.to_numpy(), 1.5)

    def test_calm_floor_is_nan(self) -> None:
        ratio = gust_ratio(_series([3.0, 3.0]), _series([0.5, 0.0]))
        assert ratio.isna().all()


class TestGustMargin:
    def test_known_value(self) -> None:
        margin = gust_margin(_series([12.0]), _series([9.0]))
        np.testing.assert_allclose(margin.to_numpy(), 3.0)


class TestVerticalVeer:
    @pytest.mark.parametrize(
        ("wd_hi", "wd_lo", "expected"),
        [
            (30.0, 10.0, 20.0),  # simple positive veer
            (10.0, 30.0, -20.0),  # simple negative
            (350.0, 10.0, -20.0),  # wraps across north
            (10.0, 350.0, 20.0),  # wraps the other way
            (180.0, 0.0, 180.0),  # boundary maps to +180, not -180
        ],
    )
    def test_wrapped_difference(self, wd_hi: float, wd_lo: float, expected: float) -> None:
        veer = vertical_veer(_series([wd_hi]), _series([wd_lo]))
        np.testing.assert_allclose(veer.to_numpy(), expected)


class TestAirDensity:
    def test_standard_dry_air(self) -> None:
        # ISA sea level: 15 degC, 1013.25 hPa, dry -> 1.225 kg/m3
        rho = air_density(_series([15.0]), _series([1013.25]), _series([0.0]))
        np.testing.assert_allclose(rho.to_numpy(), 1.225, atol=0.001)

    def test_humidity_reduces_density(self) -> None:
        dry = air_density(_series([15.0]), _series([1013.25]), _series([0.0]))
        moist = air_density(_series([15.0]), _series([1013.25]), _series([100.0]))
        assert float(moist.iloc[0]) < float(dry.iloc[0])
        np.testing.assert_allclose(moist.to_numpy(), 1.217, atol=0.002)


class TestEra5DerivedFrame:
    def _aligned(self, n: int = 4) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC")
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "wind_speed_10m": rng.uniform(3, 10, n),
                "wind_speed_100m": rng.uniform(5, 14, n),
                "wind_gusts_10m": rng.uniform(5, 15, n),
                "wind_direction_10m": rng.uniform(0, 360, n),
                "wind_direction_100m": rng.uniform(0, 360, n),
                "temperature_2m": rng.uniform(0, 20, n),
                "surface_pressure": rng.uniform(980, 1030, n),
                "relative_humidity_2m": rng.uniform(40, 100, n),
            },
            index=idx,
        )

    def test_all_derivations_present_and_finite(self) -> None:
        frame = era5_derived_frame(self._aligned(), derivations=ERA5_DERIVATIONS, hub_height_m=_HOT_HUB_HEIGHT_M)
        assert list(frame.columns) == list(ERA5_DERIVATIONS)
        assert np.isfinite(frame.to_numpy(dtype=float)).all()

    def test_subset_only_builds_requested(self) -> None:
        frame = era5_derived_frame(self._aligned(), derivations=["gust_ratio", "veer"])
        assert list(frame.columns) == ["gust_ratio", "veer"]

    def test_unknown_derivation_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown ERA5 derivation"):
            era5_derived_frame(self._aligned(), derivations=["nonsense"])

    def test_hub_height_required(self) -> None:
        with pytest.raises(ValueError, match="requires hub_height_m"):
            era5_derived_frame(self._aligned(), derivations=["wind_speed_hub"])

    def test_missing_raw_column_raises(self) -> None:
        aligned = self._aligned().drop(columns=["wind_gusts_10m"])
        with pytest.raises(ValueError, match="missing columns"):
            era5_derived_frame(aligned, derivations=["gust_ratio"])
