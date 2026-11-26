"""Operating states for test records, as the facts' record labels give them to a method."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from wind_up.operating_state import (
    FULL_DOWNTIME,
    GENERIC_STATES,
    MISSING,
    NORMAL_OPERATION,
    PARTIAL_DOWNTIME,
    STATE_COL,
    VALID_NORTHING_COL,
    VALID_UPLIFT_COL,
    WAKING_STATE_COL,
    OperatingStateConfig,
    label_operating_states,
)

if TYPE_CHECKING:
    from wind_up.schema import ColumnSchema


def _timebase(scada: pd.DataFrame) -> pd.Timedelta:
    index = pd.DatetimeIndex(scada.index.unique()).sort_values()
    return pd.Timedelta(np.median(np.diff(index.to_numpy()))) if len(index) > 1 else pd.Timedelta(minutes=10)


def generic_states(
    scada: pd.DataFrame, *, columns: ColumnSchema, rated_power_kw: float, timebase: pd.Timedelta | None = None
) -> pd.DataFrame:
    """Return ``scada`` labelled with wind-up's generic operating states; the timebase is inferred when ``None``."""
    return label_operating_states(
        scada,
        columns=columns,
        config=OperatingStateConfig(),
        timebase=_timebase(scada) if timebase is None else timebase,
        rated_power_kw=rated_power_kw,
    )


def counter_states(scada: pd.DataFrame, *, columns: ColumnSchema, timebase: pd.Timedelta | None = None) -> pd.DataFrame:
    """Return ``scada`` labelled from its power and availability counter alone, without the stuck and range rules.

    Missing where either is not finite, full downtime at a counter of 0 or below, partial downtime
    under a full period, normal operation otherwise. Without an availability column only power counts.
    """
    full = (_timebase(scada) if timebase is None else timebase).total_seconds()
    power = scada[columns.active_power].to_numpy(dtype=float)
    has_counter = columns.availability is not None and columns.availability in scada.columns
    counter = scada[columns.availability].to_numpy(dtype=float) if has_counter else np.full(len(scada), full)
    state = np.select(
        [~np.isfinite(power) | ~np.isfinite(counter), counter <= 0, counter < full],
        [MISSING, FULL_DOWNTIME, PARTIAL_DOWNTIME],
        default=NORMAL_OPERATION,
    ).astype(object)
    out = scada.copy()
    out[STATE_COL] = state
    states = pd.Series(state, index=out.index)
    out[VALID_NORTHING_COL] = states.map({k: v.northing for k, v in GENERIC_STATES.items()}).astype(bool)
    out[WAKING_STATE_COL] = states.map({k: v.waking for k, v in GENERIC_STATES.items()})
    out[VALID_UPLIFT_COL] = states.map({k: v.uplift for k, v in GENERIC_STATES.items()}).astype(bool)
    return out
