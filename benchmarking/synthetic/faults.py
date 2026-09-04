"""Injected data faults: the pathologies real SCADA carries, with known ground truth.

A fault is injected into the synthetic frame after the upgrades, is invisible to the analyst
(``CampaignSpec`` never carries it), and never moves the ground truth. ``original_df`` is left
untouched, so any movement in an estimate is the fault's doing.

That is what separates a fault from an upgrade: an upgrade is declared and moves the truth; a
fault is undeclared and moves only the estimate.

Most faults hold the truth still by corrupting what a method **measures** rather than what a
turbine **produced**. :class:`ReferenceCpChange` is the exception -- it changes real power on a
reference turbine, and stays truth-neutral because the truth is derived per *test* turbine.
Faults that change power declare ``changes_power``, and the generator refuses to aim one at a
test turbine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from benchmarking.synthetic.upgrades import ConstantCpChange, apply_upgrades

if TYPE_CHECKING:
    from benchmarking.synthetic.cp_core import CpCore
    from benchmarking.synthetic.schema import ColumnSchema


@runtime_checkable
class Fault(Protocol):
    """An undeclared corruption applied to the synthetic frame."""

    @property
    def description(self) -> dict:
        """Serialisable provenance recorded in the dataset's run metadata."""
        ...

    @property
    def changes_power(self) -> bool:
        """Whether this fault alters active power rather than only a reading."""
        ...

    def __call__(self, synthetic_df: pd.DataFrame, *, columns: ColumnSchema, cp: CpCore) -> pd.DataFrame:
        """Return ``synthetic_df`` with this fault injected; ``cp`` converts a Cp change to power."""
        ...


@dataclass(frozen=True)
class NorthingStep:
    """A step change in one turbine's reported direction, from ``at`` to the end of the record.

    The signature of a recalibration or a sensor swap: the turbine's north reference moves and
    nothing else does. Power is untouched, so ground truth is unaffected.

    :param turbine: the turbine whose direction reading steps
    :param at: when the step happens; rows from here on carry the offset
    :param offset_deg: degrees added to the reading (wrapped to 0-360)
    :param role: the :class:`~benchmarking.synthetic.schema.ColumnSchema` direction role to
        corrupt; the nacelle position by default
    """

    turbine: str
    at: pd.Timestamp
    offset_deg: float
    role: str = "nacelle_position"
    changes_power: ClassVar[bool] = False

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this fault."""
        return {
            "kind": "northing_step",
            "turbine": self.turbine,
            "at": str(self.at),
            "offset_deg": float(self.offset_deg),
        }

    def __call__(self, synthetic_df: pd.DataFrame, *, columns: ColumnSchema, cp: CpCore) -> pd.DataFrame:  # noqa: ARG002
        """Return ``synthetic_df`` with ``turbine``'s direction stepped by ``offset_deg`` from ``at``."""
        columns.require_roles([self.role])
        column = getattr(columns, self.role)
        if column not in synthetic_df.columns:
            msg = (
                f"cannot inject a northing step: the {self.role} column {column!r} is not in the frame. "
                f"Columns present: {sorted(synthetic_df.columns)}"
            )
            raise ValueError(msg)
        is_turbine = (synthetic_df[columns.turbine] == self.turbine).to_numpy()
        if not is_turbine.any():
            present = sorted({str(t) for t in synthetic_df[columns.turbine].unique()})
            msg = f"cannot inject a northing step into {self.turbine!r}: it is not in the frame, which has {present}"
            raise ValueError(msg)

        synthetic_df = synthetic_df.copy()
        stepped = is_turbine & np.asarray(synthetic_df.index >= self.at)
        values = synthetic_df[column].to_numpy(dtype=float)
        values[stepped] = (values[stepped] + self.offset_deg) % 360.0
        synthetic_df[column] = values
        return synthetic_df


# The nacelle anemometer channels a gain fault scales together, so turbulence intensity
# (sd / mean) is unchanged and only the wind-speed axis moves. Roles a schema leaves unset
# are skipped, so a source carrying min/max companions picks them up by naming them here.
WIND_SPEED_ROLES = ("wind_speed", "wind_speed_sd")


def _turbine_mask(synthetic_df: pd.DataFrame, *, columns: ColumnSchema, turbine: str, fault: str) -> np.ndarray:
    """Boolean over the frame's rows selecting ``turbine``; raises if it is not present."""
    is_turbine = (synthetic_df[columns.turbine] == turbine).to_numpy()
    if not is_turbine.any():
        present = sorted({str(t) for t in synthetic_df[columns.turbine].unique()})
        msg = f"cannot inject {fault} into {turbine!r}: it is not in the frame, which has {present}"
        raise ValueError(msg)
    return is_turbine


def _gain_columns(
    synthetic_df: pd.DataFrame, *, columns: ColumnSchema, roles: tuple[str, ...], fault: str
) -> list[str]:
    """Column names for ``roles``, skipping those the schema leaves unset; raises on absent columns."""
    # dict.fromkeys dedupes while keeping order: roles is public, and two roles may name the
    # same column, which would otherwise scale that channel twice.
    resolved = list(dict.fromkeys(name for role in roles if (name := getattr(columns, role))))
    if not resolved:
        msg = f"cannot inject {fault}: the schema leaves every named role {list(roles)} unset"
        raise ValueError(msg)
    missing = sorted(c for c in resolved if c not in synthetic_df.columns)
    if missing:
        msg = (
            f"cannot inject {fault}: the columns {missing} are not in the frame. "
            f"Columns present: {sorted(synthetic_df.columns)}"
        )
        raise ValueError(msg)
    return resolved


def _scaled(synthetic_df: pd.DataFrame, *, cols: list[str], factor: np.ndarray) -> pd.DataFrame:
    """Return ``synthetic_df`` with every column in ``cols`` multiplied row-wise by ``factor``."""
    synthetic_df = synthetic_df.copy()
    for col in cols:
        synthetic_df[col] = synthetic_df[col].to_numpy(dtype=float) * factor
    return synthetic_df


@dataclass(frozen=True)
class SensorGainStep:
    """A step change in the gain of one turbine's sensor channels, from ``at`` to the end.

    The signature of a recalibration or a replaced transducer: the readings jump by a constant
    factor and nothing else changes. Power is untouched, so ground truth is unaffected.

    :param turbine: the turbine whose readings step
    :param at: when the step happens; rows from here on carry the gain
    :param gain: the multiplier applied to each named channel (1.5 reads 50% high, 0.5 half)
    :param roles: the :class:`~benchmarking.synthetic.schema.ColumnSchema` roles to scale; the
        anemometer channels by default
    """

    turbine: str
    at: pd.Timestamp
    gain: float
    roles: tuple[str, ...] = WIND_SPEED_ROLES
    changes_power: ClassVar[bool] = False

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this fault."""
        return {
            "kind": "sensor_gain_step",
            "turbine": self.turbine,
            "at": str(self.at),
            "gain": float(self.gain),
            "roles": list(self.roles),
        }

    def __call__(self, synthetic_df: pd.DataFrame, *, columns: ColumnSchema, cp: CpCore) -> pd.DataFrame:  # noqa: ARG002
        """Return ``synthetic_df`` with ``turbine``'s named channels scaled by ``gain`` from ``at``."""
        fault = "a sensor gain step"
        is_turbine = _turbine_mask(synthetic_df, columns=columns, turbine=self.turbine, fault=fault)
        cols = _gain_columns(synthetic_df, columns=columns, roles=self.roles, fault=fault)
        factor = np.ones(len(synthetic_df))
        factor[is_turbine & np.asarray(synthetic_df.index >= self.at)] = float(self.gain)
        return _scaled(synthetic_df, cols=cols, factor=factor)


@dataclass(frozen=True)
class SensorGainDrift:
    """A gain on one turbine's sensor channels ramping linearly from 1.0 to ``gain`` over the record.

    The signature of a slowly degrading or fouling sensor. Because the ramp spans the whole
    analysis period, the baseline and treated periods sit at different mean gains -- the shape
    least likely to cancel when a toggle campaign alternates. Power is untouched, so ground truth
    is unaffected.

    :param turbine: the turbine whose readings drift
    :param gain: the multiplier reached at the last timestamp of the record
    :param roles: the :class:`~benchmarking.synthetic.schema.ColumnSchema` roles to scale; the
        anemometer channels by default
    """

    turbine: str
    gain: float
    roles: tuple[str, ...] = WIND_SPEED_ROLES
    changes_power: ClassVar[bool] = False

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this fault."""
        return {
            "kind": "sensor_gain_drift",
            "turbine": self.turbine,
            "gain": float(self.gain),
            "roles": list(self.roles),
        }

    def __call__(self, synthetic_df: pd.DataFrame, *, columns: ColumnSchema, cp: CpCore) -> pd.DataFrame:  # noqa: ARG002
        """Return ``synthetic_df`` with ``turbine``'s named channels on a 1.0 to ``gain`` ramp."""
        fault = "a sensor gain drift"
        is_turbine = _turbine_mask(synthetic_df, columns=columns, turbine=self.turbine, fault=fault)
        cols = _gain_columns(synthetic_df, columns=columns, roles=self.roles, fault=fault)
        index = pd.DatetimeIndex(synthetic_df.index)
        span = (index.max() - index.min()).total_seconds()
        # A record spanning no time has no ramp to walk: every row sits at the end of it.
        progress = (index - index.min()).total_seconds() / span if span > 0 else np.ones(len(index))
        factor = np.ones(len(synthetic_df))
        ramp = 1.0 + (float(self.gain) - 1.0) * np.asarray(progress, dtype=float)
        factor[is_turbine] = ramp[is_turbine]
        return _scaled(synthetic_df, cols=cols, factor=factor)


@dataclass(frozen=True)
class ReferenceCpChange:
    """A flat Cp change on one turbine, from ``at`` to the end of the record.

    A reference turbine's own undeclared performance change: its retrofit, a blade repair, a
    controller change. Unlike the reading-only faults this moves real power, through the same Cp
    core a declared upgrade uses, so power, generator rpm and wind speed stay consistent.

    Truth-neutral because ground truth is derived per test turbine. The generator refuses to aim
    this at one.

    :param turbine: the turbine whose performance changes
    :param at: when the change happens; rows from here on carry it
    :param delta: the Cp change (0.03 for +3%), applied region-2 weighted
    """

    turbine: str
    at: pd.Timestamp
    delta: float
    changes_power: ClassVar[bool] = True

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this fault."""
        return {"kind": "reference_cp_change", "turbine": self.turbine, "at": str(self.at), "delta": float(self.delta)}

    def __call__(self, synthetic_df: pd.DataFrame, *, columns: ColumnSchema, cp: CpCore) -> pd.DataFrame:
        """Return ``synthetic_df`` with ``turbine``'s power carrying a ``delta`` Cp change from ``at``."""
        is_turbine = _turbine_mask(synthetic_df, columns=columns, turbine=self.turbine, fault="a reference Cp change")
        changed = is_turbine & np.asarray(synthetic_df.index >= self.at)
        if not changed.any():
            return synthetic_df
        synthetic_df = synthetic_df.copy()
        modified = apply_upgrades(
            synthetic_df.loc[changed], [ConstantCpChange(delta=self.delta)], cp=cp, columns=columns
        )
        for col in (columns.active_power, columns.gen_rpm, columns.wind_speed):
            synthetic_df.loc[changed, col] = modified[col].to_numpy()
        return synthetic_df


def apply_faults(synthetic_df: pd.DataFrame, faults: list, *, columns: ColumnSchema, cp: CpCore) -> pd.DataFrame:
    """Apply every fault to ``synthetic_df`` in order, returning the corrupted frame."""
    for fault in faults:
        synthetic_df = fault(synthetic_df, columns=columns, cp=cp)
    return synthetic_df
