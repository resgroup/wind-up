"""Operating states: one state per SCADA record of every turbine, each with a validity per later step.

wind-up owns the generic states and their validity: missing, full downtime, partial downtime and
normal operation. A caller adds site states through a label column and a table giving each label's
validity; it may also write a generic state's name into that column. The first matching rule wins:

1. missing: active power or the availability counter is not finite or out of range, the record is
   stuck (compared with the turbine's last record not missing for another reason), the counter reads
   zero while the turbine generates above :data:`COUNTER_CONTRADICTION_RATED` of rated (one of the two
   is wrong, and which one is unknown), or the caller labelled it missing;
2. full downtime: the counter is at or below 0, or the caller labelled it so;
3. partial downtime: the counter is between 0 and a full period, the rotor is parked, or the caller
   labelled it so;
4. the caller's site label;
5. normal operation.

A campaign's declarations then win over these rules: its label overrides first, then its exclusions, which
label their records missing (:func:`apply_declarations`).

:func:`fill_grid` gives every turbine a row at every timestamp, so a record labelled missing reads the same as
one never provided. :func:`label_operating_states` adds the state and its validity columns to a long SCADA frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from benchmarking.synthetic import ColumnSchema

# Output columns.
STATE_COL = "operating_state"
VALID_NORTHING_COL = "valid_northing"
WAKING_STATE_COL = "waking_state"
VALID_UPLIFT_COL = "valid_uplift"
STATE_COLUMNS = (STATE_COL, VALID_NORTHING_COL, WAKING_STATE_COL, VALID_UPLIFT_COL)

# Generic state names.
MISSING = "missing"
FULL_DOWNTIME = "full downtime"
PARTIAL_DOWNTIME = "partial downtime"
NORMAL_OPERATION = "normal operation"

# Waking validity values.
WAKING = "waking"
PART_WAKING = "part waking"
NOT_WAKING = "not waking"
WAKING_MISSING = "missing"
WAKING_VALUES = (WAKING, PART_WAKING, NOT_WAKING, WAKING_MISSING)

# Below this nacelle wind speed (m/s) an unchanged record is a calm, not stuck.
CALM_WIND_SPEED = 1.5

# A counter at or below zero beside power above this share of rated contradicts it: the record is missing.
COUNTER_CONTRADICTION_RATED = 0.05

# Out of range: active power outside (-1, 2) x rated, the counter outside (-0.05, 1.05) x a full period.
_POWER_RANGE_RATED = (-1.0, 2.0)
_COUNTER_RANGE_FULL = (-0.05, 1.05)

# The ColumnSchema roles that are measured signals; the stuck rule reads those set and present.
_SIGNAL_ROLES = (
    "active_power",
    "wind_speed",
    "wind_speed_sd",
    "gen_rpm",
    "availability",
    "active_power_min",
    "pitch",
    "reactive_power",
    "nacelle_position",
    "ambient_temp",
)


@dataclass(frozen=True)
class StateValidity:
    """Whether a state's records are valid for northing and uplift, and its waking state.

    :param northing: valid for northing
    :param waking: one of :data:`WAKING_VALUES`
    :param uplift: valid for uplift
    """

    northing: bool
    waking: str
    uplift: bool

    def __post_init__(self) -> None:
        """Reject an unknown waking value."""
        if self.waking not in WAKING_VALUES:
            msg = f"waking must be one of {list(WAKING_VALUES)}, got {self.waking!r}"
            raise ValueError(msg)


GENERIC_STATES: dict[str, StateValidity] = {
    MISSING: StateValidity(northing=False, waking=WAKING_MISSING, uplift=False),
    FULL_DOWNTIME: StateValidity(northing=False, waking=NOT_WAKING, uplift=False),
    PARTIAL_DOWNTIME: StateValidity(northing=False, waking=PART_WAKING, uplift=False),
    NORMAL_OPERATION: StateValidity(northing=True, waking=WAKING, uplift=True),
}


@dataclass(frozen=True)
class OperatingStateConfig:
    """A site's operating-state declaration.

    :param label_column: the column holding the caller's per-record label; NaN is no label
    :param labels: the validity of each site label; generic state names are reserved
    :param parked_pitch_above_deg: pitch above this is a parked rotor
    :param parked_pitch_below_deg: pitch below this is a parked rotor; at most one of the two is set
    """

    label_column: str | None = None
    labels: dict[str, StateValidity] = field(default_factory=dict)
    parked_pitch_above_deg: float | None = None
    parked_pitch_below_deg: float | None = None

    def __post_init__(self) -> None:
        """Reject generic names in ``labels``, labels without a column, and two parked-pitch rules."""
        generic = sorted(set(self.labels) & set(GENERIC_STATES))
        if generic:
            msg = f"the operating-state labels may not redefine the generic states {generic}"
            raise ValueError(msg)
        if self.labels and self.label_column is None:
            msg = "operating-state labels need a label_column to read them from"
            raise ValueError(msg)
        if self.parked_pitch_above_deg is not None and self.parked_pitch_below_deg is not None:
            msg = "set at most one of parked_pitch_above_deg and parked_pitch_below_deg"
            raise ValueError(msg)

    def validity(self) -> dict[str, StateValidity]:
        """Return every state's validity: the generic states, then the site labels."""
        return {**GENERIC_STATES, **self.labels}


def fill_grid(scada_df: pd.DataFrame, *, turbine_col: str, timebase: pd.Timedelta) -> pd.DataFrame:
    """Return ``scada_df`` with a row for every turbine at every timestamp, ordered by timestamp then turbine.

    The timestamps run from the first to the last, ``timebase`` apart, with any off the grid kept. The added rows
    carry the turbine and NaN everywhere else.
    """
    if scada_df.empty:
        return scada_df
    index = pd.DatetimeIndex(scada_df.index)
    grid = pd.date_range(index.min(), index.max(), freq=timebase).union(index.unique())
    turbines = pd.unique(scada_df[turbine_col].to_numpy())
    present = pd.MultiIndex.from_arrays([index, scada_df[turbine_col].to_numpy()])
    absent = pd.MultiIndex.from_product([grid, turbines]).difference(present)
    filled = scada_df
    if len(absent):
        rows = scada_df.iloc[:0].reindex(pd.RangeIndex(len(absent)))
        rows[turbine_col] = absent.get_level_values(1)
        rows.index = absent.get_level_values(0).rename(scada_df.index.name)
        filled = pd.concat([scada_df, rows])
    order = np.lexsort((filled[turbine_col].astype(str).to_numpy(), pd.DatetimeIndex(filled.index).asi8))
    if (order == np.arange(len(filled))).all():
        return filled
    return filled.iloc[order]


def not_missing(scada_df: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Return True on each record not labelled missing; every record of a frame without the state column."""
    if STATE_COL not in scada_df.columns:
        return np.ones(len(scada_df), dtype=bool)
    return scada_df[STATE_COL].to_numpy() != MISSING


def signal_columns(scada_df: pd.DataFrame, *, columns: ColumnSchema) -> list[str]:
    """Return the measured-signal columns the schema names and ``scada_df`` carries."""
    names = (getattr(columns, role) for role in _SIGNAL_ROLES)
    return list(dict.fromkeys(c for c in names if c is not None and c in scada_df.columns))


def stuck_records(scada_df: pd.DataFrame, *, columns: ColumnSchema) -> pd.Series:
    """Return True where every measured signal equals the turbine's own previous record, outside calms.

    A NaN repeats the last value before it. A turbine's first record is never stuck, and a record
    with nacelle wind speed below :data:`CALM_WIND_SPEED` is a calm, not stuck.

    :param scada_df: long-format SCADA of one or more turbines, indexed by timestamp
    :param columns: the schema ``scada_df`` is keyed by
    :return: a bool Series on ``scada_df``'s index, in its order
    """
    signals = signal_columns(scada_df, columns=columns)
    if not signals or scada_df.empty:
        return pd.Series(data=False, index=scada_df.index)
    turbine = scada_df[columns.turbine].to_numpy()
    order = np.lexsort((pd.DatetimeIndex(scada_df.index).asi8, pd.factorize(turbine)[0]))
    ordered = scada_df[signals].iloc[order].reset_index(drop=True)
    groups = pd.Series(turbine[order])
    filled = ordered.groupby(groups, sort=False).ffill().fillna(0)
    previous = filled.groupby(groups, sort=False).shift(1)
    first = groups.ne(groups.shift(1)).to_numpy()
    frozen = (filled == previous).all(axis=1).to_numpy() & ~first
    if columns.wind_speed in scada_df.columns:
        frozen &= ~(ordered[columns.wind_speed].to_numpy(dtype=float) < CALM_WIND_SPEED)
    stuck = np.empty(len(scada_df), dtype=bool)
    stuck[order] = frozen
    return pd.Series(stuck, index=scada_df.index)


def counter_contradicts_power(*, power: np.ndarray, counter: np.ndarray, rated_power_kw: float) -> np.ndarray:
    """Return True where the availability counter reads zero while power is above the contradiction share of rated."""
    with np.errstate(invalid="ignore"):
        return (counter <= 0) & (power > COUNTER_CONTRADICTION_RATED * rated_power_kw)


def label_operating_states(
    scada_df: pd.DataFrame,
    *,
    columns: ColumnSchema,
    config: OperatingStateConfig,
    timebase: pd.Timedelta,
    rated_power_kw: float,
) -> pd.DataFrame:
    """Return a copy of ``scada_df`` with each record's operating state and its validity added.

    Adds :data:`STATE_COL`, :data:`VALID_NORTHING_COL`, :data:`WAKING_STATE_COL` and
    :data:`VALID_UPLIFT_COL`. Raises if ``config`` names a label column ``scada_df`` lacks, or the
    column holds a label that is neither a site label nor a generic state.

    :param scada_df: long-format SCADA of one or more turbines, indexed by timestamp
    :param columns: the schema ``scada_df`` is keyed by
    :param config: the site's labels and parked-pitch rule
    :param timebase: the records' period; a full period of the availability counter is its seconds
    :param rated_power_kw: rated power, which bounds the in-range active power
    """
    full = timebase.total_seconds()
    power = scada_df[columns.active_power].to_numpy(dtype=float)
    counter = scada_df[columns.availability].to_numpy(dtype=float)
    site = _site_labels(scada_df, config=config)
    power_lo, power_hi = (f * rated_power_kw for f in _POWER_RANGE_RATED)
    counter_lo, counter_hi = (f * full for f in _COUNTER_RANGE_FULL)
    missing = (
        ~np.isfinite(power)
        | ~np.isfinite(counter)
        | (power <= power_lo)
        | (power >= power_hi)
        | (counter <= counter_lo)
        | (counter >= counter_hi)
        | counter_contradicts_power(power=power, counter=counter, rated_power_kw=rated_power_kw)
        | (site == MISSING)
    )
    signals = signal_columns(scada_df, columns=columns)
    unmissing = scada_df.assign(**{c: scada_df[c].where(~missing) for c in signals})
    missing |= stuck_records(unmissing, columns=columns).to_numpy()
    full_downtime = (counter <= 0) | (site == FULL_DOWNTIME)
    partial_downtime = ((counter > 0) & (counter < full)) | _parked(scada_df, columns=columns, config=config)
    partial_downtime |= site == PARTIAL_DOWNTIME
    has_site = pd.notna(site) & ~np.isin(site, list(GENERIC_STATES))
    state = np.select(
        [missing, full_downtime, partial_downtime, has_site],
        [np.full(len(site), MISSING), np.full(len(site), FULL_DOWNTIME), np.full(len(site), PARTIAL_DOWNTIME), site],
        default=NORMAL_OPERATION,
    ).astype(object)

    return _with_states(scada_df.copy(), state=state, config=config)


def apply_declarations(
    labelled: pd.DataFrame,
    *,
    turbine_col: str,
    config: OperatingStateConfig,
    exclusions: Sequence[tuple[str | None, pd.Timestamp, pd.Timestamp]],
    overrides: Sequence[tuple[str | None, pd.Timestamp, pd.Timestamp, str]],
) -> pd.DataFrame:
    """Return a copy of a labelled frame with a campaign's declarations applied over its labels.

    Each label override's records take its state, then each exclusion's records are missing; the validity
    columns follow. A window is ``(turbine, start, end)``, end exclusive, turbine None for every turbine.
    Raises on an override state ``config`` does not define.
    """
    validity = config.validity()
    unknown = sorted({state for *_, state in overrides} - set(validity))
    if unknown:
        msg = f"the label overrides name {unknown}, which are neither declared labels nor generic states"
        raise ValueError(msg)
    state = labelled[STATE_COL].to_numpy(dtype=object).copy()
    for who, start, end, named in overrides:
        state[declared_records(labelled, turbine_col=turbine_col, windows=[(who, start, end)])] = named
    state[declared_records(labelled, turbine_col=turbine_col, windows=exclusions)] = MISSING
    return _with_states(labelled.copy(), state=state, config=config)


def declared_records(
    scada_df: pd.DataFrame,
    *,
    turbine_col: str,
    windows: Sequence[
        tuple[str | None, pd.Timestamp, pd.Timestamp] | tuple[str | None, pd.Timestamp, pd.Timestamp, str]
    ],
) -> npt.NDArray[np.bool_]:
    """Return True on each record inside a ``(turbine, start, end, ...)`` window of its turbine or of None."""
    index = pd.DatetimeIndex(scada_df.index)
    turbines = scada_df[turbine_col].astype(str).to_numpy()
    inside = np.zeros(len(scada_df), dtype=bool)
    for who, start, end, *_ in windows:
        mine = np.asarray((index >= start) & (index < end))
        inside |= mine if who is None else mine & (turbines == who)
    return inside


def _with_states(labelled: pd.DataFrame, *, state: np.ndarray, config: OperatingStateConfig) -> pd.DataFrame:
    """Write ``state`` and its validity columns into ``labelled`` and return it."""
    validity = config.validity()
    labelled[STATE_COL] = state
    states = pd.Series(state, index=labelled.index)
    labelled[VALID_NORTHING_COL] = states.map({k: v.northing for k, v in validity.items()}).astype(bool)
    labelled[WAKING_STATE_COL] = states.map({k: v.waking for k, v in validity.items()})
    labelled[VALID_UPLIFT_COL] = states.map({k: v.uplift for k, v in validity.items()}).astype(bool)
    return labelled


def state_hours(labelled: pd.DataFrame, *, columns: ColumnSchema, timebase: pd.Timedelta) -> pd.DataFrame:
    """Return the hours each turbine spent in each operating state: ``turbine``, ``operating_state``, ``hours``."""
    counts = labelled.groupby([columns.turbine, STATE_COL]).size().rename("records").reset_index()
    counts["hours"] = counts["records"] * (timebase / pd.Timedelta(hours=1))
    return counts.rename(columns={columns.turbine: "turbine"})[["turbine", STATE_COL, "hours"]]


def _site_labels(scada_df: pd.DataFrame, *, config: OperatingStateConfig) -> np.ndarray:
    """Return the caller's labels as an object array, None where there is none; raise on an unknown label."""
    if config.label_column is None:
        return np.full(len(scada_df), None, dtype=object)
    if config.label_column not in scada_df.columns:
        msg = f"the operating-state label column {config.label_column!r} is not in the SCADA"
        raise ValueError(msg)
    raw = scada_df[config.label_column]
    labels = raw.astype(object).where(raw.notna(), None).to_numpy()
    unknown = sorted({str(v) for v in labels if v is not None} - set(config.validity()))
    if unknown:
        msg = (
            f"the operating-state label column {config.label_column!r} holds {unknown}, which are neither "
            f"declared labels {sorted(config.labels)} nor generic states {list(GENERIC_STATES)}"
        )
        raise ValueError(msg)
    return labels


def _parked(scada_df: pd.DataFrame, *, columns: ColumnSchema, config: OperatingStateConfig) -> np.ndarray:
    """Return True where pitch is beyond the declared parked threshold; False without a rule or a pitch signal."""
    if columns.pitch is None or columns.pitch not in scada_df.columns:
        return np.zeros(len(scada_df), dtype=bool)
    pitch = scada_df[columns.pitch].to_numpy(dtype=float)
    if config.parked_pitch_above_deg is not None:
        return pitch > config.parked_pitch_above_deg
    if config.parked_pitch_below_deg is not None:
        return pitch < config.parked_pitch_below_deg
    return np.zeros(len(scada_df), dtype=bool)
