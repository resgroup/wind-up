"""Derive physically meaningful quantities from raw (synced) ERA5 columns (shared, Issue 9).

The methods consume ERA5 as raw Open-Meteo columns; this module turns those into the derived,
treatment-invariant quantities that actually drive turbine power and its scatter, so every method
*and* the CEM matching step share one implementation:

* ``shear_exponent`` — the power-law exponent ``alpha = ln(ws_100m/ws_10m)/ln(100/10)``; folds the
  collinear 10 m / 100 m speeds into one physical vertical-shear (stability) signal.
* ``wind_speed_hub`` — hub-height wind speed by the shear power law,
  ``ws_hh = ws_100m * (hub_height_m/100)^alpha`` (needs the site's hub height).
* ``gust_ratio`` — ``wind_gusts_10m / wind_speed_10m``, a unitless TI-like turbulence proxy
  (NaN below a calm-wind floor where the ratio degenerates).
* ``veer`` — vertical direction veer ``wind_direction_100m - wind_direction_10m`` wrapped to
  ±180°.
* ``air_density`` — moist-air density from 2 m temperature, surface pressure and relative
  humidity (partial pressures of dry air and vapour, Magnus saturation formula).

All functions are NaN-tolerant (LightGBM handles NaN natively) and operate on the *aligned* ERA5
frame the lag sync produces (original Open-Meteo column names).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

# The derivation vocabulary: each name is both the config token and the output column name.
ERA5_DERIVATIONS: tuple[str, ...] = (
    "shear_exponent",
    "wind_speed_hub",
    "gust_ratio",
    "gust_margin",
    "veer",
    "air_density",
)

# The two ERA5 wind-speed levels the shear exponent is fit between.
_SHEAR_LO_M = 10.0
_SHEAR_HI_M = 100.0
# Below this 10 m wind speed the gust ratio degenerates (tiny denominator) and TI is meaningless.
_GUST_RATIO_MIN_WS = 1.0
# Specific gas constants [J/(kg K)] for dry air and water vapour.
_R_DRY = 287.05
_R_VAPOUR = 461.5
_KELVIN_OFFSET = 273.15
# Magnus saturation-vapour-pressure constants (over water), e_s in Pa for t in degC.
_MAGNUS_A = 611.2
_MAGNUS_B = 17.62
_MAGNUS_C = 243.12


def shear_exponent(ws_lo: pd.Series, ws_hi: pd.Series) -> pd.Series:
    """Power-law shear exponent ``alpha = ln(ws_hi/ws_lo)/ln(hi/lo)``; NaN where either speed <= 0."""
    lo = ws_lo.to_numpy(dtype=float)
    hi = ws_hi.to_numpy(dtype=float)
    positive = (lo > 0) & (hi > 0)
    alpha = np.full(len(lo), np.nan)
    alpha[positive] = np.log(hi[positive] / lo[positive]) / np.log(_SHEAR_HI_M / _SHEAR_LO_M)
    return pd.Series(alpha, index=ws_hi.index, name="shear_exponent")


def hub_height_wind_speed(ws_hi: pd.Series, alpha: pd.Series, *, hub_height_m: float) -> pd.Series:
    """Interpolate to hub height with the shear power law: ``ws_hh = ws_hi * (hh/hi)^alpha``."""
    hi = ws_hi.to_numpy(dtype=float)
    a = alpha.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        ws_hh = hi * (hub_height_m / _SHEAR_HI_M) ** a
    return pd.Series(ws_hh, index=ws_hi.index, name="wind_speed_hub")


def gust_ratio(gusts: pd.Series, ws: pd.Series, *, min_ws: float = _GUST_RATIO_MIN_WS) -> pd.Series:
    """TI-like gust ratio ``gusts/ws``; NaN where ``ws < min_ws`` (calm-wind degenerate denominator)."""
    g = gusts.to_numpy(dtype=float)
    w = ws.to_numpy(dtype=float)
    valid = w >= min_ws
    ratio = np.divide(g, w, out=np.full(len(w), np.nan), where=valid)
    return pd.Series(ratio, index=ws.index, name="gust_ratio")


def gust_margin(gusts: pd.Series, ws: pd.Series) -> pd.Series:
    """Gust margin ``gusts - ws`` [m/s] — an absolute gustiness signal.

    On Hill of Towie this correlates with measured nacelle TI better than the ratio form
    (the calm-wind denominator makes ``gust_ratio`` nearly uncorrelated with TI at 10 min).
    """
    margin = gusts.to_numpy(dtype=float) - ws.to_numpy(dtype=float)
    return pd.Series(margin, index=ws.index, name="gust_margin")


def vertical_veer(wd_hi: pd.Series, wd_lo: pd.Series) -> pd.Series:
    """Vertical direction veer ``wd_hi - wd_lo`` wrapped to (-180, 180] degrees."""
    diff = wd_hi.to_numpy(dtype=float) - wd_lo.to_numpy(dtype=float)
    wrapped = -((180.0 - diff) % 360.0 - 180.0)
    return pd.Series(wrapped, index=wd_hi.index, name="veer")


def air_density(temperature_c: pd.Series, pressure_hpa: pd.Series, relative_humidity_pct: pd.Series) -> pd.Series:
    """Moist-air density [kg/m3] from 2 m temperature [degC], surface pressure [hPa] and RH [%].

    Partial-pressure form: ``rho = p_dry/(R_dry T) + p_vapour/(R_vapour T)`` with the vapour
    pressure from the Magnus saturation formula scaled by relative humidity.
    """
    t_c = temperature_c.to_numpy(dtype=float)
    t_k = t_c + _KELVIN_OFFSET
    p_pa = pressure_hpa.to_numpy(dtype=float) * 100.0
    saturation_pa = _MAGNUS_A * np.exp(_MAGNUS_B * t_c / (_MAGNUS_C + t_c))
    p_vapour = np.clip(relative_humidity_pct.to_numpy(dtype=float), 0.0, 100.0) / 100.0 * saturation_pa
    p_dry = p_pa - p_vapour
    rho = p_dry / (_R_DRY * t_k) + p_vapour / (_R_VAPOUR * t_k)
    return pd.Series(rho, index=temperature_c.index, name="air_density")


# Raw Open-Meteo columns each derivation needs (validated before deriving so a missing input is a
# clear configuration error, not a KeyError deep in numpy).
_REQUIRED_RAW: dict[str, tuple[str, ...]] = {
    "shear_exponent": ("wind_speed_10m", "wind_speed_100m"),
    "wind_speed_hub": ("wind_speed_10m", "wind_speed_100m"),
    "gust_ratio": ("wind_gusts_10m", "wind_speed_10m"),
    "gust_margin": ("wind_gusts_10m", "wind_speed_100m"),
    "veer": ("wind_direction_100m", "wind_direction_10m"),
    "air_density": ("temperature_2m", "surface_pressure", "relative_humidity_2m"),
}


def era5_derived_frame(
    aligned_era5: pd.DataFrame,
    *,
    derivations: Sequence[str],
    hub_height_m: float | None = None,
) -> pd.DataFrame:
    """Build the requested derived columns from an aligned ERA5 frame (one column per derivation).

    :param aligned_era5: ERA5 aligned to the analysis grid (original Open-Meteo column names)
    :param derivations: which of :data:`ERA5_DERIVATIONS` to compute (also the output column names)
    :param hub_height_m: turbine hub height; required by the ``wind_speed_hub`` derivation
    """
    unknown = [d for d in derivations if d not in ERA5_DERIVATIONS]
    if unknown:
        msg = f"unknown ERA5 derivation(s) {unknown}; available: {list(ERA5_DERIVATIONS)}"
        raise ValueError(msg)
    if "wind_speed_hub" in derivations and hub_height_m is None:
        msg = "the wind_speed_hub derivation requires hub_height_m (the site's turbine hub height)."
        raise ValueError(msg)
    missing = sorted({c for d in derivations for c in _REQUIRED_RAW[d]} - set(aligned_era5.columns))
    if missing:
        msg = f"aligned ERA5 is missing columns {missing} required by derivations {list(derivations)}"
        raise ValueError(msg)

    # alpha feeds both shear_exponent and wind_speed_hub; compute it at most once.
    alpha: pd.Series | None = None
    if {"shear_exponent", "wind_speed_hub"} & set(derivations):
        alpha = shear_exponent(aligned_era5["wind_speed_10m"], aligned_era5["wind_speed_100m"])
    out = pd.DataFrame(index=aligned_era5.index)
    for name in derivations:
        if name == "shear_exponent":
            assert alpha is not None  # noqa: S101 - set above whenever this branch is reachable
            out[name] = alpha
        elif name == "wind_speed_hub":
            assert alpha is not None  # noqa: S101 - set above whenever this branch is reachable
            assert hub_height_m is not None  # noqa: S101 - narrowed above; mypy needs the hint
            out[name] = hub_height_wind_speed(aligned_era5["wind_speed_100m"], alpha, hub_height_m=hub_height_m)
        elif name == "gust_ratio":
            out[name] = gust_ratio(aligned_era5["wind_gusts_10m"], aligned_era5["wind_speed_10m"])
        elif name == "gust_margin":
            out[name] = gust_margin(aligned_era5["wind_gusts_10m"], aligned_era5["wind_speed_100m"])
        elif name == "veer":
            out[name] = vertical_veer(aligned_era5["wind_direction_100m"], aligned_era5["wind_direction_10m"])
        elif name == "air_density":
            out[name] = air_density(
                aligned_era5["temperature_2m"],
                aligned_era5["surface_pressure"],
                aligned_era5["relative_humidity_2m"],
            )
    return out
