"""Synthetic turbine-upgrade callables and their resolution.

An upgrade is a callable ``(rows) -> UpgradeEffect`` describing how it changes a test
turbine's treated rows. Upgrades compose: ``apply_upgrades`` resolves a list against the
*original* baseline in a defined order (Cp ratios multiply and are applied through the
Cp core, then a rated-power change, then a nacelle wind-speed change), so the result does
not depend on list order in surprising ways.

Phase 1 implements the four profiles Issue 1 names: constant, wind-speed-dependent and
condition (turbulence-intensity) dependent Cp changes, and rated-power change. Pitch,
rpm, yaw and wake-steering upgrades are future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from benchmarking.synthetic.cp_core import power_from_cp_change, region2_fraction, rpm_from_power_change
from wind_up.constants import DataColumns

if TYPE_CHECKING:
    import pandas as pd

    from benchmarking.synthetic.cp_core import CpCore


@dataclass
class UpgradeEffect:
    """One upgrade's contribution, resolved against the original baseline rows.

    :param cp_ratio: per-row (or scalar) multiplicative Cp ratio, e.g. 1.02 for +2% Cp
    :param ws_factor: multiplicative change to the nacelle wind speed, e.g. 1.01 for +1%
    :param new_rated_power_kw: rated-power override, or None to leave rated unchanged
    """

    cp_ratio: npt.ArrayLike = 1.0
    ws_factor: float = 1.0
    new_rated_power_kw: float | None = None


@dataclass(frozen=True)
class ConstantCpChange:
    """A flat Cp change in region 2 (e.g. blade cleaning, fouling or add-ons)."""

    delta: float
    ws_delta: float = 0.0

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this upgrade."""
        return {"kind": "constant_cp", "delta": self.delta, "ws_delta": self.ws_delta}

    def __call__(self, rows: pd.DataFrame) -> UpgradeEffect:  # noqa: ARG002
        """Return this upgrade's effect on the given rows."""
        return UpgradeEffect(cp_ratio=1.0 + self.delta, ws_factor=1.0 + self.ws_delta)


@dataclass(frozen=True)
class WindSpeedCpChange:
    """A wind-speed-dependent Cp change (e.g. the AeroUp region-2 shape).

    The Cp delta is interpolated over ``ws_points`` against the turbine's original
    wind speed; outside the point range the nearest endpoint delta is held.
    """

    ws_points: tuple[float, ...]
    deltas: tuple[float, ...]
    ws_delta: float = 0.0

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this upgrade."""
        return {
            "kind": "wind_speed_cp",
            "ws_points": list(self.ws_points),
            "deltas": list(self.deltas),
            "ws_delta": self.ws_delta,
        }

    def __call__(self, rows: pd.DataFrame) -> UpgradeEffect:
        """Return this upgrade's effect on the given rows."""
        original_ws = rows[DataColumns.wind_speed_mean].to_numpy(dtype=float)
        delta = np.interp(original_ws, self.ws_points, self.deltas)
        return UpgradeEffect(cp_ratio=1.0 + delta, ws_factor=1.0 + self.ws_delta)


def _condition_series(rows: pd.DataFrame, by: str) -> npt.NDArray[np.float64]:
    """Compute a treatment-invariant condition signal from the original rows.

    ``by="ti"`` is turbulence intensity (WindSpeedSD / WindSpeedMean); any other value is
    treated as the name of a column already present in ``rows``.
    """
    if by == "ti":
        ws = rows[DataColumns.wind_speed_mean].to_numpy(dtype=float)
        sd = rows[DataColumns.wind_speed_sd].to_numpy(dtype=float)
        return sd / ws
    return rows[by].to_numpy(dtype=float)


@dataclass(frozen=True)
class ConditionCpChange:
    """A Cp change that varies with a condition signal of the original data.

    Phase 1 supports ``by="ti"`` (turbulence intensity = WindSpeedSD / WindSpeedMean);
    the Cp delta is interpolated over ``points`` of that condition. ``by`` may also name
    any column already present in the rows.
    """

    by: str
    points: tuple[float, ...]
    deltas: tuple[float, ...]
    ws_delta: float = 0.0

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this upgrade."""
        return {
            "kind": "condition_cp",
            "by": self.by,
            "points": list(self.points),
            "deltas": list(self.deltas),
            "ws_delta": self.ws_delta,
        }

    def __call__(self, rows: pd.DataFrame) -> UpgradeEffect:
        """Return this upgrade's effect on the given rows."""
        condition = _condition_series(rows, self.by)
        delta = np.interp(condition, self.points, self.deltas)
        return UpgradeEffect(cp_ratio=1.0 + delta, ws_factor=1.0 + self.ws_delta)


@dataclass(frozen=True)
class RatedPowerChange:
    """A change to the turbine's rated power.

    A downrate (new < old) caps power at the new rated and leaves region-2 power
    unchanged. An uprate (new > old) lifts the region-3 fraction of power toward the new
    rated (deep region-2 power is essentially unaffected), then caps at the new rated.
    The uprate model is a documented synthetic choice rather than measured physics.
    """

    new_rated_power_kw: float

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this upgrade."""
        return {"kind": "rated_power", "new_rated_power_kw": self.new_rated_power_kw}

    def __call__(self, rows: pd.DataFrame) -> UpgradeEffect:  # noqa: ARG002
        """Return this upgrade's effect on the given rows."""
        return UpgradeEffect(new_rated_power_kw=self.new_rated_power_kw)


def apply_rated_change(
    power_kw: npt.NDArray[np.float64],
    *,
    original_power_kw: npt.NDArray[np.float64],
    old_rated_kw: float,
    new_rated_kw: float,
) -> npt.NDArray[np.float64]:
    """Apply a rated-power change to (already Cp-adjusted) power.

    In future this function should change other test turbine fields especially pitch angle but this is deferred for now.

    See :class:`RatedPowerChange` for the downrate/uprate behaviour.
    """
    if new_rated_kw <= old_rated_kw:
        return np.minimum(power_kw, new_rated_kw)
    region3_fraction = 1.0 - region2_fraction(original_power_kw)
    ratio = new_rated_kw / old_rated_kw
    lifted = power_kw * (1.0 + region3_fraction * (ratio - 1.0))
    return np.minimum(lifted, new_rated_kw)


def apply_upgrades(rows: pd.DataFrame, upgrades: list, cp: CpCore) -> pd.DataFrame:
    """Resolve and apply a list of upgrades to one test turbine's treated rows.

    Cp ratios from all upgrades multiply together and are applied to the original power
    through the Cp core (region-2 weighted, rated-clipped); rpm tracks the resulting
    power change and the nacelle wind speed is scaled by the combined ws factor. The
    input ``rows`` is not mutated.

    :param rows: treated rows for one test turbine, carrying the original SCADA tags
    :param upgrades: upgrade callables to resolve
    :param cp: the turbine's Cp core (rated power and Cp parameters)
    :return: a new frame with modified power, rpm and wind-speed columns
    """
    out = rows.copy()
    n = len(rows)
    baseline_power = rows[DataColumns.active_power_mean].to_numpy(dtype=float)
    baseline_rpm = rows[DataColumns.gen_rpm_mean].to_numpy(dtype=float)
    baseline_ws = rows[DataColumns.wind_speed_mean].to_numpy(dtype=float)

    cp_ratio = np.ones(n)
    ws_factor = 1.0
    new_rated_power_kw: float | None = None
    for upgrade in upgrades:
        effect = upgrade(rows)
        cp_ratio = cp_ratio * np.asarray(effect.cp_ratio, dtype=float)
        ws_factor *= effect.ws_factor
        if effect.new_rated_power_kw is not None:
            new_rated_power_kw = effect.new_rated_power_kw

    new_power = power_from_cp_change(baseline_power, cp_ratio=cp_ratio, rated_power_kw=cp.rated_power_kw)
    if new_rated_power_kw is not None:
        new_power = apply_rated_change(
            new_power,
            original_power_kw=baseline_power,
            old_rated_kw=cp.rated_power_kw,
            new_rated_kw=new_rated_power_kw,
        )
    new_rpm = rpm_from_power_change(baseline_rpm=baseline_rpm, baseline_power_kw=baseline_power, new_power_kw=new_power)

    out[DataColumns.active_power_mean] = new_power
    out[DataColumns.gen_rpm_mean] = new_rpm
    out[DataColumns.wind_speed_mean] = baseline_ws * ws_factor
    return out
