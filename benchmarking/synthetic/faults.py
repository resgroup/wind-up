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

if TYPE_CHECKING:
    import pandas as pd

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


def apply_faults(synthetic_df: pd.DataFrame, faults: list, *, columns: ColumnSchema) -> pd.DataFrame:
    """Apply every fault to ``synthetic_df`` in order, returning the corrupted frame."""
    for fault in faults:
        synthetic_df = fault(synthetic_df, columns=columns)
    return synthetic_df
