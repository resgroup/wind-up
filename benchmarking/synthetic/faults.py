"""Injected data faults: the pathologies real SCADA carries, with known ground truth.

A fault corrupts what a method **measures**, never what the turbine **produced**. It is applied
to the synthetic frame after the upgrades, leaving ``original_df`` untouched, so the true uplift
is unchanged by construction and any movement in an estimate is the fault's doing.

That is what separates a fault from an upgrade: an upgrade changes power and moves the truth; a
fault changes a reading and moves only the estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from benchmarking.synthetic.schema import ColumnSchema


@runtime_checkable
class Fault(Protocol):
    """A measurement corruption applied to the synthetic frame."""

    @property
    def description(self) -> dict:
        """Serialisable provenance recorded in the dataset's run metadata."""
        ...

    def __call__(self, synthetic_df: pd.DataFrame, *, columns: ColumnSchema) -> pd.DataFrame:
        """Return ``synthetic_df`` with this fault injected."""
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

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this fault."""
        return {
            "kind": "northing_step",
            "turbine": self.turbine,
            "at": str(self.at),
            "offset_deg": float(self.offset_deg),
        }

    def __call__(self, synthetic_df: pd.DataFrame, *, columns: ColumnSchema) -> pd.DataFrame:
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

    def __call__(self, synthetic_df: pd.DataFrame, *, columns: ColumnSchema) -> pd.DataFrame:
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

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this fault."""
        return {
            "kind": "sensor_gain_drift",
            "turbine": self.turbine,
            "gain": float(self.gain),
            "roles": list(self.roles),
        }

    def __call__(self, synthetic_df: pd.DataFrame, *, columns: ColumnSchema) -> pd.DataFrame:
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


def apply_faults(synthetic_df: pd.DataFrame, faults: list, *, columns: ColumnSchema) -> pd.DataFrame:
    """Apply every fault to ``synthetic_df`` in order, returning the corrupted frame."""
    for fault in faults:
        synthetic_df = fault(synthetic_df, columns=columns)
    return synthetic_df
