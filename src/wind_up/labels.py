"""Record labels: the upgrade states, the analyst's own labels, and the columns they are written to."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

# Upgrade state values.
ON = "on"
OFF = "off"
NEITHER = "neither"
UPGRADE_STATES = (ON, OFF, NEITHER)

# The uses a label may withhold.
UPLIFT = "uplift"
NORTHING = "northing"
USES = (UPLIFT, NORTHING)
# Expand this to the supported uses when a label is created.
ALL_USES = "all"

# Record-label columns naming why a record is not valid for a use; empty where it is valid.
UPLIFT_REASON_COL = "not_valid_uplift_because"
NORTHING_REASON_COL = "not_valid_northing_because"
REASON_SEPARATOR = "; "

FULL_CIRCLE_DEG = 360.0


def upgrade_state_column(upgrade: str) -> str:
    """Return the record-label column holding the upgrade state of the upgrade called ``upgrade``."""
    return f"upgrade_state: {upgrade}"


def settling_reason(upgrade: str) -> str:
    """Return the reason a record settling after a change of the upgrade called ``upgrade`` is not valid."""
    return f"settling after a {upgrade} change"


@dataclass(frozen=True)
class Label:
    """An analyst's label: a named true or false on every record.

    :param name: the label's name, as analyses and reports refer to it
    :param column: the boolean column of the records holding it; False where unknown, never NaN
    :param withholds: a use or an iterable of uses to withhold its records from. :data:`ALL_USES`
        expands to all supported uses. The attribute is stored as a frozen set.
    """

    name: str
    column: str
    withholds: Iterable[str] | str = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Reject an empty name or column and an unknown use, and resolve :data:`ALL_USES` to :data:`USES`."""
        if not self.name.strip() or not self.column.strip():
            msg = f"a label needs a name and a column, got name {self.name!r} and column {self.column!r}"
            raise ValueError(msg)
        withholds = _uses(self.withholds)
        unknown = sorted(withholds - {*USES, ALL_USES})
        if unknown:
            msg = f"label {self.name!r} withholds {unknown}; a label may withhold only {list(USES)} or {ALL_USES!r}"
            raise ValueError(msg)
        object.__setattr__(self, "withholds", frozenset(USES) if ALL_USES in withholds else withholds)


def direction_sector(direction: pd.Series, *, sector: tuple[float, float]) -> pd.Series:
    """Return True where ``direction`` lies in ``sector``, a label to add to the records.

    The sector runs clockwise from its first bound, inclusive, to its second, exclusive, and may pass
    through north; equal bounds are the full circle. A direction that is not finite is not in it.
    The direction must not come from a turbine the label will select records of as a subject.

    :param direction: wind direction in degrees, one value per timestamp
    :param sector: ``(from, to)`` in degrees
    """
    low, high = (float(bound) % FULL_CIRCLE_DEG for bound in sector)
    values = direction.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        wrapped = np.mod(values, FULL_CIRCLE_DEG)
        if low < high:
            inside = (wrapped >= low) & (wrapped < high)
        elif low > high:
            inside = (wrapped >= low) | (wrapped < high)
        else:
            inside = np.isfinite(wrapped)
    return pd.Series(inside & np.isfinite(values), index=direction.index, name=direction.name)


def _uses(withholds: Iterable[str] | str) -> frozenset[str]:
    """Return ``withholds`` as a frozen set, a lone string as one use."""
    return frozenset([withholds] if isinstance(withholds, str) else withholds)
