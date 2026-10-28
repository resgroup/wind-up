"""Waking scenarios: which turbines are upwind of each turbine in each record, and which are not waking.

:func:`add_waking_scenarios` adds to a long, step-2 labelled and northed SCADA frame, for each record:

- :data:`DIRECTION_COL`: the direction used to find the turbines upwind. A turbine that is not a test
  turbine uses its own northed nacelle position where the record is valid for northing. Otherwise the
  circular mean of up to :data:`DONOR_COUNT` of the nearest turbines that are not test turbines and
  have a valid direction in that record is used. :data:`DIRECTION_SOURCE_COL` says which.
- :data:`UPWIND_COL`: every layout turbine within the IEC disturbed sector at that direction, in name
  order and joined by ``"; "``, or :data:`NONE`, or :data:`UNKNOWN` without a direction.
- :data:`WAKING_SCENARIO_COL`: :data:`NORMAL` when every upwind turbine is waking; otherwise the upwind
  turbines that are not, with their waking state, e.g. ``"T03 not waking; T04 part waking"``; or
  :data:`UNKNOWN` without a direction or when an upwind turbine's waking state is missing.

A layout turbine with no SCADA record at a timestamp has a missing waking state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from benchmarking.harness.operating_state import (
    NOT_WAKING,
    PART_WAKING,
    VALID_NORTHING_COL,
    WAKING,
    WAKING_MISSING,
    WAKING_STATE_COL,
)
from wind_up.circular_math import circ_diff
from wind_up.layout import NAME_COL, ROTOR_DIAMETER_COL, iec_disturbed_sector_deg

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarking.synthetic import ColumnSchema
    from wind_up.layout import Layout

# Output columns.
DIRECTION_COL = "waking_direction"
DIRECTION_SOURCE_COL = "waking_direction_source"
UPWIND_COL = "upwind"
WAKING_SCENARIO_COL = "waking_scenario"
WAKING_COLUMNS = (DIRECTION_COL, DIRECTION_SOURCE_COL, UPWIND_COL, WAKING_SCENARIO_COL)

# Column values.
NORMAL = "normal"
UNKNOWN = "unknown"
NONE = "none"
OWN = "own"
NEIGHBOURS = "neighbours"

DONOR_COUNT = 4

_SEPARATOR = "; "
# Waking states as codes; 0 is waking.
_STATE_CODES = {WAKING: 0, NOT_WAKING: 1, PART_WAKING: 2, WAKING_MISSING: 3}
_CODE_NAMES = {code: name for name, code in _STATE_CODES.items()}
_MISSING_CODE = _STATE_CODES[WAKING_MISSING]


@dataclass(frozen=True)
class WakingDirections:
    """Each turbine's directions on a shared timestamp index, one column per turbine.

    :param index: the timestamps
    :param turbines: the turbines, in column order
    :param own: the turbine's northed direction where valid, else NaN
    :param neighbours: the circular mean of its nearest valid donors, else NaN
    :param is_test: whether each turbine is a test turbine
    """

    index: pd.DatetimeIndex
    turbines: tuple[str, ...]
    own: npt.NDArray[np.float64]
    neighbours: npt.NDArray[np.float64]
    is_test: npt.NDArray[np.bool_]

    @property
    def uses_own(self) -> npt.NDArray[np.bool_]:
        """Whether each timestamp and turbine uses its own direction."""
        return np.isfinite(self.own) & ~self.is_test[None, :]

    @property
    def direction(self) -> npt.NDArray[np.float64]:
        """The direction used: its own where :attr:`uses_own`, else its neighbours'."""
        return np.where(self.uses_own, self.own, self.neighbours)

    def source(self) -> npt.NDArray[np.object_]:
        """Return :data:`OWN`, :data:`NEIGHBOURS` or :data:`UNKNOWN` per timestamp and turbine."""
        source = np.full(self.own.shape, UNKNOWN, dtype=object)
        source[np.isfinite(self.neighbours)] = NEIGHBOURS
        source[self.uses_own] = OWN
        return source


def waking_directions(
    scada_df: pd.DataFrame, *, columns: ColumnSchema, layout: Layout, test_turbines: Sequence[str]
) -> WakingDirections:
    """Return every SCADA turbine's direction for finding the turbines upwind of it.

    :param scada_df: long SCADA labelled by step 2 and carrying the northed nacelle position
    :param columns: the schema ``scada_df`` is keyed by
    :param layout: the farm layout; must name every SCADA turbine
    :param test_turbines: the campaign's test turbines, whose own direction is never used
    """
    index = pd.DatetimeIndex(scada_df.index.unique()).sort_values()
    turbines = tuple(sorted(str(t) for t in scada_df[columns.turbine].unique()))
    northed = _wide(scada_df, columns=columns, col=columns.northed("nacelle_position"), index=index, turbines=turbines)
    valid = _wide(scada_df, columns=columns, col=VALID_NORTHING_COL, index=index, turbines=turbines)
    own = np.where(np.isfinite(northed) & (valid == 1.0), northed, np.nan)
    rows = [layout.index_of(t) for t in turbines]
    distance = layout.distance_m[np.ix_(rows, rows)]
    is_test = np.isin(turbines, list(test_turbines))
    neighbours = np.full(own.shape, np.nan)
    for j in range(len(turbines)):
        donors = [k for k in np.argsort(distance[j], kind="stable") if k != j and not is_test[k]]
        if donors:
            neighbours[:, j] = _nearest_valid_mean(own[:, donors])
    return WakingDirections(index=index, turbines=turbines, own=own, neighbours=neighbours, is_test=is_test)


def add_waking_scenarios(
    scada_df: pd.DataFrame,
    *,
    columns: ColumnSchema,
    layout: Layout,
    test_turbines: Sequence[str],
    directions: WakingDirections | None = None,
) -> pd.DataFrame:
    """Return a copy of ``scada_df`` with :data:`WAKING_COLUMNS` added to every record.

    :param scada_df: long SCADA labelled by step 2 and carrying the northed nacelle position
    :param columns: the schema ``scada_df`` is keyed by
    :param layout: the farm layout; must name every SCADA turbine
    :param test_turbines: the campaign's test turbines, whose own direction is never used
    :param directions: :func:`waking_directions` of the same arguments, for a caller that already holds it
    """
    found = (
        directions
        if directions is not None
        else waking_directions(scada_df, columns=columns, layout=layout, test_turbines=test_turbines)
    )
    names = [str(n) for n in layout.frame[NAME_COL]]
    states = _state_codes(scada_df, columns=columns, index=found.index, turbines=names)
    diameters = layout.frame[ROTOR_DIAMETER_COL].to_numpy(dtype=float)
    direction = found.direction
    upwind = np.empty(direction.shape, dtype=object)
    scenario = np.empty(direction.shape, dtype=object)
    for j, turbine in enumerate(found.turbines):
        target = layout.index_of(turbine)
        upwind[:, j], scenario[:, j] = _labels(
            direction[:, j],
            bearing=layout.bearing_deg[target],
            sector=iec_disturbed_sector_deg(layout.distance_m[target] / diameters),
            target=target,
            states=states,
            names=names,
        )

    out = scada_df.copy()
    turbine_col = out[columns.turbine].astype(str).to_numpy()
    row = found.index.get_indexer(pd.DatetimeIndex(out.index))
    col = pd.Index(found.turbines).get_indexer(turbine_col)
    if (row < 0).any() or (col < 0).any():
        missing_times = int(np.count_nonzero(row < 0))
        missing_turbines = sorted(set(turbine_col[col < 0]))
        msg = (
            f"the directions do not cover scada_df: {missing_times} row(s) at timestamps and the turbines "
            f"{missing_turbines} are not in them; build the directions from the same frame"
        )
        raise ValueError(msg)
    out[DIRECTION_COL] = direction[row, col]
    out[DIRECTION_SOURCE_COL] = found.source()[row, col]
    out[UPWIND_COL] = upwind[row, col]
    out[WAKING_SCENARIO_COL] = scenario[row, col]
    return out


def upwind_matrix(
    direction: npt.NDArray[np.float64],
    *,
    bearing: npt.NDArray[np.float64],
    sector: npt.NDArray[np.float64],
    target: int,
) -> npt.NDArray[np.bool_]:
    """Return which turbines are upwind of ``target`` at each direction, shape ``(directions, turbines)``.

    A turbine is upwind when the bearing to it is within half its disturbed sector of the direction.
    A NaN direction has no turbine upwind.
    """
    relative = np.abs(circ_diff(bearing[None, :], direction[:, None]))
    with np.errstate(invalid="ignore"):
        upwind = relative < sector[None, :] / 2
    upwind[:, target] = False
    return np.asarray(upwind)


def _labels(
    direction: npt.NDArray[np.float64],
    *,
    bearing: npt.NDArray[np.float64],
    sector: npt.NDArray[np.float64],
    target: int,
    states: npt.NDArray[np.int8],
    names: Sequence[str],
) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.object_]]:
    """Return one turbine's upwind and waking-scenario labels per timestamp."""
    order = np.argsort(names, kind="stable")
    upwind = upwind_matrix(direction, bearing=bearing, sector=sector, target=target)[:, order]
    codes = np.where(upwind, states[:, order], 0).astype(np.int8)
    ordered = [names[i] for i in order]

    patterns, which = np.unique(upwind, axis=0, return_inverse=True)
    upwind_text = np.array(
        [_SEPARATOR.join(n for n, up in zip(ordered, p, strict=True) if up) or NONE for p in patterns], dtype=object
    )[which.ravel()]
    patterns, which = np.unique(codes, axis=0, return_inverse=True)
    scenario_text = np.array([_scenario(p, names=ordered) for p in patterns], dtype=object)[which.ravel()]

    unknown = ~np.isfinite(direction)
    upwind_text[unknown] = UNKNOWN
    scenario_text[unknown] = UNKNOWN
    return upwind_text, scenario_text


def _scenario(codes: npt.NDArray[np.int8], *, names: Sequence[str]) -> str:
    """Return the waking-scenario label of one row of upwind state codes."""
    if (codes == _MISSING_CODE).any():
        return UNKNOWN
    abnormal = [f"{n} {_CODE_NAMES[int(c)]}" for n, c in zip(names, codes, strict=True) if c]
    return _SEPARATOR.join(abnormal) or NORMAL


def _nearest_valid_mean(donors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Circular mean of the first :data:`DONOR_COUNT` finite values of each row, NaN where there are none."""
    finite = np.isfinite(donors)
    take = finite & (np.cumsum(finite, axis=1) <= DONOR_COUNT)
    rad = np.deg2rad(np.where(take, donors, 0.0))
    sin = np.where(take, np.sin(rad), 0.0).sum(axis=1)
    cos = np.where(take, np.cos(rad), 0.0).sum(axis=1)
    mean = np.rad2deg(np.arctan2(sin, cos)) % 360.0
    return np.where(take.any(axis=1), mean, np.nan)


def _wide(
    scada_df: pd.DataFrame, *, columns: ColumnSchema, col: str, index: pd.DatetimeIndex, turbines: Sequence[str]
) -> npt.NDArray[np.float64]:
    """Return ``col`` as a ``(timestamps, turbines)`` float array, NaN where a turbine has no record."""
    out = np.full((len(index), len(turbines)), np.nan)
    turbine_of = scada_df[columns.turbine].astype(str).to_numpy()
    values = scada_df[col].to_numpy(dtype=float, na_value=np.nan)
    row = index.get_indexer(pd.DatetimeIndex(scada_df.index))
    for j, turbine in enumerate(turbines):
        mine = turbine_of == turbine
        out[row[mine], j] = values[mine]
    return out


def _state_codes(
    scada_df: pd.DataFrame, *, columns: ColumnSchema, index: pd.DatetimeIndex, turbines: Sequence[str]
) -> npt.NDArray[np.int8]:
    """Return each turbine's waking state as a code, ``(timestamps, turbines)``; missing where it has no record."""
    out = np.full((len(index), len(turbines)), _MISSING_CODE, dtype=np.int8)
    turbine_of = scada_df[columns.turbine].astype(str).to_numpy()
    codes = scada_df[WAKING_STATE_COL].map(_STATE_CODES).fillna(_MISSING_CODE).to_numpy(dtype=np.int8)
    row = index.get_indexer(pd.DatetimeIndex(scada_df.index))
    for j, turbine in enumerate(turbines):
        mine = turbine_of == turbine
        out[row[mine], j] = codes[mine]
    return out
