"""The facts of a wind farm and what was done to it, as every method receives them.

:class:`WindFarmFacts` holds the records as measured, what their columns mean, the turbines, the
upgrades and the analyst's declarations. :meth:`WindFarmFacts.record_labels` labels every record
with its operating state, its state under each upgrade and the analyst's labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import pandas as pd

from wind_up.analysis_warnings import AnalysisWarning
from wind_up.labels import (
    NEITHER,
    NORTHING,
    NORTHING_REASON_COL,
    OFF,
    ON,
    REASON_SEPARATOR,
    UPGRADE_STATES,
    UPLIFT,
    UPLIFT_REASON_COL,
    Label,
    upgrade_state_column,
)
from wind_up.layout import LATITUDE_COL, LONGITUDE_COL, NAME_COL, ROTOR_DIAMETER_COL, Layout
from wind_up.operating_state import (
    STATE_COL,
    STATE_COLUMNS,
    VALID_NORTHING_COL,
    VALID_UPLIFT_COL,
    OperatingStateConfig,
    apply_declarations,
    fill_grid,
    label_operating_states,
    signal_columns,
)
from wind_up.record_selection import NEAR_CHANGEOVER, ChangedInterval, changed_times, turbines_within_reach
from wind_up.toggle import ToggleSchedule, is_toggle, treated_mask, validate_toggle_df
from wind_up.wake_reach import MAX_REACH_D, reach_from_uplift

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from wind_up.analysis import AnalysisSpec
    from wind_up.record_selection import ChangedTimes, WorksWindow
    from wind_up.schema import ColumnSchema

RATED_POWER_COL = "rated_power_kw"
HUB_HEIGHT_COL = "hub_height_m"
REQUIRED_TURBINE_COLUMNS = (NAME_COL, LATITUDE_COL, LONGITUDE_COL, RATED_POWER_COL)

# The record-label column an analysis's comparison is written to.
SIDE_COL = "side"

# The step of the method the labels belong to, for the warnings register.
LABELS_STEP = 2

NO_TIME = pd.Timedelta(0)

Window = tuple[pd.Timestamp, pd.Timestamp]
Period = tuple[pd.Timestamp | None, pd.Timestamp | None]
Exclusion = tuple[str | None, pd.Timestamp, pd.Timestamp]
StateOverride = tuple[str | None, pd.Timestamp, pd.Timestamp, str]
NorthOffset = tuple[str, pd.Timestamp, float]
Timing: TypeAlias = pd.Timestamp | Mapping[str, pd.Timestamp] | ToggleSchedule | pd.DataFrame


@dataclass(frozen=True)
class Upgrade:
    """What was done to a group of turbines, and when.

    :param name: the upgrade's name
    :param turbines: the turbines that received it
    :param timing: the upgrade date, a date per turbine, a toggle schedule, or a toggle frame with
        boolean ``toggle_on`` and ``toggle_off`` columns on a time index. ``None`` when every turbine
        has an installation window: the upgrade is then on from the end of its last window.
    :param settling_time: toggle only. Records from each change between on and off until this long
        after it are in neither state.
    :param installation_windows: per turbine, the windows ``[start, end)`` when the upgrade was being
        fitted; their records are in neither state
    :param upgrade_state_overrides: ``(turbine or None, start, end, state)``: the upgrade state of a
        turbine, or of every turbine, over ``[start, end)`` where the timing does not describe it
    :param toggle_cycle: toggle only: one on and off cycle of the controller. A toggle schedule's period
        when omitted; for a toggle frame, the median gap between its on runs when omitted, and checked
        against them when given.
    :param toggle_datum: toggle only: any boundary of the controller's cycle grid. A toggle schedule's
        start when omitted, else the first record of a toggle frame in the on or off state.
    :param expected_uplift: the uplift expected, a fraction, from which the reach of the upgrade's
        wake change follows; at most one of this and ``wake_reach_d``
    :param wake_reach_d: the reach of the upgrade's wake change, in rotor diameters
    """

    name: str
    turbines: Sequence[str]
    timing: Timing | None = None
    settling_time: pd.Timedelta = NO_TIME
    installation_windows: Mapping[str, Sequence[Window]] = field(default_factory=dict)
    upgrade_state_overrides: Sequence[StateOverride] = ()
    toggle_cycle: pd.Timedelta | None = None
    toggle_datum: pd.Timestamp | None = None
    expected_uplift: float | None = None
    wake_reach_d: float | None = None

    def __post_init__(self) -> None:
        """Freeze the collections and reject a declaration that contradicts itself."""
        object.__setattr__(self, "turbines", tuple(str(t) for t in self.turbines))
        object.__setattr__(
            self,
            "installation_windows",
            {
                str(t): tuple((pd.Timestamp(a), pd.Timestamp(b)) for a, b in w)
                for t, w in self.installation_windows.items()
            },
        )
        object.__setattr__(
            self,
            "upgrade_state_overrides",
            tuple((who, pd.Timestamp(a), pd.Timestamp(b), state) for who, a, b, state in self.upgrade_state_overrides),
        )
        if not self.name.strip():
            msg = "an upgrade needs a name"
            raise ValueError(msg)
        if not self.turbines or len(set(self.turbines)) != len(self.turbines):
            msg = f"upgrade {self.name!r} needs at least one turbine, each named once; got {list(self.turbines)}"
            raise ValueError(msg)
        if self.expected_uplift is not None and self.wake_reach_d is not None:
            msg = f"upgrade {self.name!r} declares both expected_uplift and wake_reach_d; give at most one"
            raise ValueError(msg)
        self._check_windows()
        self._check_timing()

    def _check_windows(self) -> None:
        """Reject a window of a turbine outside the upgrade, a window that ends before it starts, an unknown state."""
        stray = sorted(set(self.installation_windows) - set(self.turbines))
        if stray:
            msg = f"upgrade {self.name!r} has installation windows for {stray}, which did not receive it"
            raise ValueError(msg)
        windows = [w for ws in self.installation_windows.values() for w in ws]
        spans = [*windows, *((a, b) for _, a, b, _ in self.upgrade_state_overrides)]
        backwards = [(a, b) for a, b in spans if b < a]
        if backwards:
            msg = f"upgrade {self.name!r} has a window that ends before it starts: {backwards[0]}"
            raise ValueError(msg)
        unknown = sorted({state for *_, state in self.upgrade_state_overrides} - set(UPGRADE_STATES))
        if unknown:
            msg = f"upgrade {self.name!r} overrides to {unknown}; an upgrade state is one of {list(UPGRADE_STATES)}"
            raise ValueError(msg)

    def _check_timing(self) -> None:
        """Reject a timing that leaves a turbine without a date, and settling outside a toggle."""
        if isinstance(self.timing, pd.DataFrame):
            validate_toggle_df(self.timing)
        if self.settling_time < NO_TIME or (self.settling_time > NO_TIME and self.mode != "toggle"):
            msg = f"upgrade {self.name!r}: settling_time is for a toggle and cannot be negative"
            raise ValueError(msg)
        if self.mode == "toggle":
            self._check_cycle()
            return
        if self.toggle_cycle is not None or self.toggle_datum is not None:
            msg = f"upgrade {self.name!r}: toggle_cycle and toggle_datum are for a toggle"
            raise ValueError(msg)
        if isinstance(self.timing, Mapping):
            object.__setattr__(self, "timing", {str(t): pd.Timestamp(d) for t, d in self.timing.items()})
        dated = set(self.timing) if isinstance(self.timing, Mapping) else set(self.turbines)
        if self.timing is None:
            dated = {t for t, w in self.installation_windows.items() if w}
        undated = sorted(set(self.turbines) - dated)
        if undated:
            msg = f"upgrade {self.name!r} gives no upgrade date or installation window for {undated}"
            raise ValueError(msg)

    def _check_cycle(self) -> None:
        """Reject a cycle that is not positive, or that disagrees with the schedule or the frame's runs."""
        if self.toggle_cycle is None:
            return
        if self.toggle_cycle <= NO_TIME:
            msg = f"upgrade {self.name!r}: toggle_cycle must be positive, got {self.toggle_cycle}"
            raise ValueError(msg)
        stated = self._timing_cycle()
        if stated != self.toggle_cycle:
            msg = (
                f"upgrade {self.name!r}: toggle_cycle {self.toggle_cycle} disagrees with the toggle timing, "
                f"whose cycle is {stated}"
            )
            raise ValueError(msg)

    @property
    def cycle(self) -> pd.Timedelta:
        """Return a toggle's cycle: declared, a schedule's period, or a frame's median gap between on runs."""
        if self.mode != "toggle":
            msg = f"upgrade {self.name!r} is not a toggle"
            raise ValueError(msg)
        if self.toggle_cycle is not None:
            return self.toggle_cycle
        return self._timing_cycle()

    def _timing_cycle(self) -> pd.Timedelta:
        """Return the cycle the timing states: a schedule's period, or a frame's median gap between on runs.

        An on run starts at an on record whose previous record is not on, so records in neither state
        between the runs, settling among them, do not shorten the cycle.
        """
        if isinstance(self.timing, ToggleSchedule):
            return self.timing.period
        frame: pd.DataFrame = self.timing  # type: ignore[assignment]
        on = frame["toggle_on"].to_numpy(dtype=bool)
        starts = pd.DatetimeIndex(frame.index)[on & ~np.concatenate(([False], on[:-1]))]
        if len(starts) < 2:  # noqa: PLR2004
            msg = f"upgrade {self.name!r}: the toggle frame has fewer than two on runs, so it states no cycle"
            raise ValueError(msg)
        return pd.Timedelta(pd.Series(starts).diff().median())

    def datum(self, *, first_record: pd.Timestamp) -> pd.Timestamp:
        """Return a boundary of a toggle's cycle grid: declared, else where the toggle starts.

        :param first_record: where a toggle schedule without a start begins
        """
        if self.toggle_datum is not None:
            return pd.Timestamp(self.toggle_datum)
        start = self.signal_span()[0]
        return first_record if start is None else start

    @property
    def mode(self) -> Literal["prepost", "toggle"]:
        """Return ``"toggle"`` for a schedule or a toggle frame, ``"prepost"`` for a permanent upgrade."""
        return "toggle" if is_toggle(self.timing) else "prepost"

    @property
    def reach_d(self) -> float | None:
        """Return the declared reach of the wake change in rotor diameters, None when neither field is given."""
        if self.wake_reach_d is not None:
            return self.wake_reach_d
        return None if self.expected_uplift is None else reach_from_uplift(self.expected_uplift)

    def changeover(self, turbine: str) -> Window:
        """Return a permanently upgraded turbine's changeover ``[start, end)``: off before it, on from its end.

        The envelope of its installation windows when the upgrade has no date; otherwise the date,
        widened to the installation windows that hold it.
        """
        if self.mode == "toggle" or turbine not in self.turbines:
            msg = f"upgrade {self.name!r} has no changeover for {turbine!r}"
            raise ValueError(msg)
        windows = self.installation_windows.get(turbine, ())
        if self.timing is None:
            return min(a for a, _ in windows), max(b for _, b in windows)
        date = self.timing[turbine] if isinstance(self.timing, Mapping) else pd.Timestamp(self.timing)  # type: ignore[arg-type]
        holding = [(a, b) for a, b in windows if a <= date <= b]
        return (min(a for a, _ in holding), max(b for _, b in holding)) if holding else (date, date)

    def change_windows(self, turbine: str, *, first_record: pd.Timestamp) -> list[Window]:
        """Return the windows ``[start, end)`` over which ``turbine`` changed under this upgrade.

        A permanent upgrade's changeover and its other installation windows; a toggle's start, as a
        window of no length; and each override to on or off. An override to neither is not a change
        window: the turbine counts as changed only while it lasts. None for a turbine the upgrade
        leaves alone.

        :param first_record: where a toggle schedule without a start begins
        """
        windows: list[Window] = []
        if turbine in self.turbines:
            if self.mode == "toggle":
                start = self.signal_span()[0]
                start = first_record if start is None else start
                windows.append((start, start))
            else:
                windows.append(self.changeover(turbine))
            windows.extend(self.installation_windows.get(turbine, ()))
        windows.extend(
            (a, b) for who, a, b, state in self.upgrade_state_overrides if who in (None, turbine) and state != NEITHER
        )
        return sorted(set(windows))

    def neither_windows(self, turbine: str) -> list[Window]:
        """Return the windows over which an override puts ``turbine`` in neither state."""
        return [
            (a, b) for who, a, b, state in self.upgrade_state_overrides if who in (None, turbine) and state == NEITHER
        ]

    def signal_span(self) -> Period:
        """Return a toggle's span, from its first record in the on or off state to its last; None is open."""
        if isinstance(self.timing, ToggleSchedule):
            return self.timing.start, None
        if isinstance(self.timing, pd.DataFrame):
            flagged = self.timing.index[(self.timing["toggle_on"] | self.timing["toggle_off"]).to_numpy(dtype=bool)]
            return (flagged.min(), flagged.max()) if len(flagged) else (None, None)
        msg = f"upgrade {self.name!r} is not a toggle"
        raise ValueError(msg)

    def states(self, index: pd.DatetimeIndex, *, turbine: str) -> npt.NDArray[np.object_]:
        """Return ``turbine``'s upgrade state at each timestamp of ``index``.

        A turbine that did not receive the upgrade is off. A permanent upgrade is off before its
        date and on from it. A toggle is off before its signal starts, then follows the signal, and
        is in neither state where the signal is in neither, missing, or settling. Installation
        windows are in neither state. The overrides are applied last.
        """
        index = pd.DatetimeIndex(index)
        state = np.full(len(index), OFF, dtype=object)
        if turbine in self.turbines:
            if self.mode == "toggle":
                state = self._toggle_states(index)
            else:
                first, last = self.changeover(turbine)
                state[np.asarray(index >= last)] = ON
                if self.timing is None:
                    state[np.asarray((index >= first) & (index < last))] = NEITHER
            for start, end in self.installation_windows.get(turbine, ()):
                state[np.asarray((index >= start) & (index < end))] = NEITHER
        for who, start, end, named in self.upgrade_state_overrides:
            if who is None or who == turbine:
                state[np.asarray((index >= start) & (index < end))] = named
        return state

    def _toggle_states(self, index: pd.DatetimeIndex) -> npt.NDArray[np.object_]:
        """Return the toggle's state at each timestamp of ``index``."""
        if isinstance(self.timing, ToggleSchedule):
            return self._schedule_states(index, self.timing)
        frame: pd.DataFrame = self.timing  # type: ignore[assignment]
        on = frame["toggle_on"].to_numpy(dtype=bool)
        off = frame["toggle_off"].to_numpy(dtype=bool)
        own = np.select([on, off], [ON, OFF], default=NEITHER).astype(object)
        times = pd.DatetimeIndex(frame.index)
        if self.settling_time > NO_TIME:
            changed = (on[1:] & off[:-1]) | (off[1:] & on[:-1])
            own[_settling(times, changes=times[1:][changed], settling_time=self.settling_time)] = NEITHER
        state = pd.Series(own, index=times).reindex(index).to_numpy(dtype=object)
        state[pd.isna(state)] = NEITHER
        flagged = times[on | off]
        if len(flagged):
            state[np.asarray(index < flagged.min())] = OFF
        return state

    def _schedule_states(self, index: pd.DatetimeIndex, schedule: ToggleSchedule) -> npt.NDArray[np.object_]:
        """Return a periodic toggle's state at each timestamp of ``index``."""
        state = np.where(treated_mask(index, schedule), ON, OFF).astype(object)
        if self.settling_time > NO_TIME and len(index):
            origin = schedule.start if schedule.start is not None else index.min()
            half = schedule.period / 2
            since_origin = index - origin
            settling = np.asarray((since_origin >= half) & ((since_origin % half) < self.settling_time))
            state[settling] = NEITHER
        return state


def _settling(
    times: pd.DatetimeIndex, *, changes: pd.DatetimeIndex, settling_time: pd.Timedelta
) -> npt.NDArray[np.bool_]:
    """Return True at each of ``times`` from one of ``changes`` until ``settling_time`` after it."""
    if not len(changes):
        return np.zeros(len(times), dtype=bool)
    last = np.searchsorted(changes.to_numpy(), times.to_numpy(), side="right") - 1
    since = times - changes[np.clip(last, 0, None)]
    return (last >= 0) & np.asarray(since < settling_time)


@dataclass(frozen=True)
class NonTurbineMeasurement:
    """A mast or LiDAR: where it stands, what it measured and what its columns mean.

    :param name: its name, distinct from every turbine's
    :param latitude: degrees north
    :param longitude: degrees east
    :param records: its measurements on a time index
    :param columns: the role of each column a method may read, role to column name
    """

    name: str
    latitude: float
    longitude: float
    records: pd.DataFrame
    columns: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Reanalysis:
    """One reanalysis node: where it is and its records.

    :param name: the node's name
    :param latitude: degrees north
    :param longitude: degrees east
    :param records: the reanalysis records on a time index
    """

    name: str
    latitude: float
    longitude: float
    records: pd.DataFrame


@dataclass(frozen=True)
class WindFarmFacts:
    """What is true of a wind farm and what was done to it.

    :param records: long-format SCADA as measured, on a time index: every year supplied, nothing
        dropped or blanked
    :param columns: what the columns of ``records`` mean
    :param timebase: the records' period
    :param turbines: one row per turbine: ``name``, ``latitude``, ``longitude`` and
        ``rated_power_kw``; ``rotor_diameter_m`` and ``hub_height_m`` when a method needs them
    :param upgrades: the upgrades, at least one
    :param operating_states: the site's operating states; the generic rules alone when omitted
    :param exclusions: ``(turbine or None, start, end)``: the data is wrong, its records are missing
    :param operating_state_overrides: ``(turbine or None, start, end, state)``: the data is right
        and the operating state was ``state``
    :param labels: the analyst's own labels
    :param north_offsets: the declared northing, ``(turbine, from, offset in degrees)``
    :param non_turbine_measurements: masts and LiDARs
    :param reanalysis: reanalysis nodes
    """

    records: pd.DataFrame
    columns: ColumnSchema
    timebase: pd.Timedelta
    turbines: pd.DataFrame
    upgrades: Sequence[Upgrade]
    operating_states: OperatingStateConfig = field(default_factory=OperatingStateConfig)
    exclusions: Sequence[Exclusion] = ()
    operating_state_overrides: Sequence[StateOverride] = ()
    labels: Sequence[Label] = ()
    north_offsets: Sequence[NorthOffset] = ()
    non_turbine_measurements: Sequence[NonTurbineMeasurement] = ()
    reanalysis: Sequence[Reanalysis] = ()

    def __post_init__(self) -> None:
        """Freeze the collections and reject facts that contradict each other."""
        for name in (
            "upgrades",
            "exclusions",
            "operating_state_overrides",
            "labels",
            "north_offsets",
            "non_turbine_measurements",
            "reanalysis",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        self._check_records()
        self._check_turbines()
        self._check_upgrades()
        self._check_declarations()

    def _check_records(self) -> None:
        """Reject records without a time index, a required or named column, or a label not true or false."""
        if not isinstance(self.records.index, pd.DatetimeIndex):
            msg = "the records need a time index"
            raise TypeError(msg)
        if self.timebase <= NO_TIME:
            msg = f"the timebase must be positive, got {self.timebase}"
            raise ValueError(msg)
        self.columns.require_roles(["turbine", "active_power"])
        named = {
            "turbine": self.columns.turbine,
            "active_power": self.columns.active_power,
            **({} if self.columns.availability is None else {"availability": self.columns.availability}),
            **{f"label {label.name!r}": label.column for label in self.labels},
        }
        absent = {what: column for what, column in named.items() if column not in self.records.columns}
        if absent:
            msg = f"the records lack the column(s) {absent}"
            raise ValueError(msg)
        names = [label.name for label in self.labels]
        if len(set(names)) != len(names):
            msg = f"each label needs its own name; got {names}"
            raise ValueError(msg)
        for label in self.labels:
            values = self.records[label.column]
            if values.isna().any() or not values.isin([True, False]).all():
                msg = f"label {label.name!r}: column {label.column!r} must be true or false on every record, never NaN"
                raise ValueError(msg)

    def _check_turbines(self) -> None:
        """Reject a turbine table without its required columns, or records of a turbine it does not hold."""
        absent = [c for c in REQUIRED_TURBINE_COLUMNS if c not in self.turbines.columns]
        if absent:
            msg = f"the turbine table lacks {absent}; it carries {list(self.turbines.columns)}"
            raise ValueError(msg)
        names = self.turbine_names
        if len(set(names)) != len(names):
            msg = f"the turbine table names a turbine more than once: {names}"
            raise ValueError(msg)
        rated = self.turbines[RATED_POWER_COL].to_numpy(dtype=float)
        if not (np.isfinite(rated) & (rated > 0)).all():
            msg = "every turbine needs a rated power that is positive and finite"
            raise ValueError(msg)
        unknown = sorted({str(t) for t in self.records[self.columns.turbine].unique()} - set(names))
        if unknown:
            msg = f"the records hold {unknown}, which the turbine table does not"
            raise ValueError(msg)

    def _check_upgrades(self) -> None:
        """Reject no upgrade, two upgrades of one name, and an upgrade of an unknown turbine."""
        if not self.upgrades:
            msg = "the facts need at least one upgrade"
            raise ValueError(msg)
        names = [u.name for u in self.upgrades]
        if len(set(names)) != len(names):
            msg = f"each upgrade needs its own name; got {names}"
            raise ValueError(msg)
        known = set(self.turbine_names)
        for upgrade in self.upgrades:
            overridden = {who for who, *_ in upgrade.upgrade_state_overrides if who is not None}
            unknown = sorted(({*upgrade.turbines} | overridden) - known)
            if unknown:
                msg = f"upgrade {upgrade.name!r} names {unknown}, which the turbine table does not hold"
                raise ValueError(msg)

    def _check_declarations(self) -> None:
        """Reject a declaration naming an unknown turbine, and a measurement named like a turbine."""
        known = set(self.turbine_names)
        declared: dict[str, set[str | None]] = {
            "exclusions": {who for who, *_ in self.exclusions},
            "operating state overrides": {who for who, *_ in self.operating_state_overrides},
            "north offsets": {who for who, *_ in self.north_offsets},
        }
        for what, turbines in declared.items():
            unknown = sorted({str(t) for t in turbines if t is not None} - known)
            if unknown:
                msg = f"the {what} name {unknown}, which the turbine table does not hold"
                raise ValueError(msg)
        clash = sorted({m.name for m in self.non_turbine_measurements} & known)
        if clash:
            msg = f"the non-turbine measurements {clash} carry a turbine's name"
            raise ValueError(msg)

    @property
    def turbine_names(self) -> list[str]:
        """Return every turbine's name, in the turbine table's order."""
        return [str(n) for n in self.turbines[NAME_COL]]

    @property
    def layout(self) -> Layout:
        """Return the farm layout. Raises when the turbine table has no rotor diameter."""
        if ROTOR_DIAMETER_COL not in self.turbines.columns:
            msg = f"the turbine table has no {ROTOR_DIAMETER_COL!r} column, which a layout needs"
            raise ValueError(msg)
        return Layout.from_frame(self.turbines)

    def rated_power_kw(self, turbine: str) -> float:
        """Return ``turbine``'s rated power in kW."""
        return float(self._turbine_row(turbine)[RATED_POWER_COL])

    def upgrade(self, name: str) -> Upgrade:
        """Return the upgrade called ``name``."""
        for upgrade in self.upgrades:
            if upgrade.name == name:
                return upgrade
        msg = f"no upgrade is called {name!r}; the facts hold {[u.name for u in self.upgrades]}"
        raise KeyError(msg)

    def warnings(self) -> tuple[AnalysisWarning, ...]:
        """Return what the facts cannot support, for the warnings register."""
        if self.columns.availability is not None:
            return ()
        message = "the records have no availability counter, so downtime could not be labelled"
        return (AnalysisWarning(severity="medium", step=LABELS_STEP, turbine="all", message=message),)

    def record_labels(self) -> pd.DataFrame:
        """Return the labels of every turbine's record at every timestamp, ordered by timestamp then turbine.

        One row per turbine and timestamp of the full grid, a record never supplied reading as
        missing. Columns: the turbine, the operating state and its three uses, the state under each
        upgrade (:func:`~wind_up.labels.upgrade_state_column`), each analyst label under its name,
        and why a record is not valid for uplift or for northing. A record is valid for a use when
        its operating state allows it and no label on it withholds it.

        Computed on each call; a caller that needs it more than once keeps the frame.
        """
        turbine_col = self.columns.turbine
        wanted = [turbine_col, *signal_columns(self.records, columns=self.columns)]
        wanted += [self.operating_states.site_state_column or turbine_col, *(label.column for label in self.labels)]
        grid = fill_grid(self.records[list(dict.fromkeys(wanted))], turbine_col=turbine_col, timebase=self.timebase)
        labelled = apply_declarations(
            self._generic_states(grid),
            turbine_col=turbine_col,
            config=self.operating_states,
            exclusions=self.exclusions,
            overrides=self.operating_state_overrides,
        )
        out = labelled[[turbine_col, *STATE_COLUMNS]].copy()
        index = pd.DatetimeIndex(out.index)
        turbines = out[turbine_col].astype(str).to_numpy()
        members = {name: turbines == name for name in pd.unique(turbines)}
        for upgrade in self.upgrades:
            state = np.empty(len(out), dtype=object)
            for name, mine in members.items():
                state[mine] = upgrade.states(index[mine], turbine=name)
            out[upgrade_state_column(upgrade.name)] = state
        held = {label.name: grid[label.column].eq(True).to_numpy(dtype=bool) for label in self.labels}  # noqa: FBT003
        for name, values in held.items():
            out[name] = values
        for use, valid_col, reason_col in (
            (UPLIFT, VALID_UPLIFT_COL, UPLIFT_REASON_COL),
            (NORTHING, VALID_NORTHING_COL, NORTHING_REASON_COL),
        ):
            withheld_by = {label.name: held[label.name] for label in self.labels if use in label.withholds}
            valid, reason = _validity(
                state=out[STATE_COL].to_numpy(dtype=object),
                state_valid=out[valid_col].to_numpy(dtype=bool),
                withheld_by=withheld_by,
            )
            out[valid_col] = valid
            out[reason_col] = reason
        return out

    def comparison_sides(self, analysis: AnalysisSpec, *, record_labels: pd.DataFrame) -> pd.Series:
        """Return the side of ``analysis``'s comparison each record is on: on, off or neither.

        A subject's record is on the side of its upgrade state when it lies within the analysis's
        period, within the signal's span for a toggle, holds every include label and no exclude
        label. Every other record is on neither side. Validity for uplift is not applied.

        :param record_labels: the frame :meth:`record_labels` returned
        """
        upgrade = self.upgrade(analysis.upgrade)
        self.check(analysis)
        index = pd.DatetimeIndex(record_labels.index)
        turbines = record_labels[self.columns.turbine].astype(str)
        compared = turbines.isin(self.subjects_of(analysis)).to_numpy(dtype=bool).copy()
        compared &= _within(index, analysis.period, end_inclusive=False)
        if upgrade.mode == "toggle":
            compared &= _within(index, upgrade.signal_span(), end_inclusive=True)
        for name in analysis.include_labels:
            compared &= record_labels[name].to_numpy(dtype=bool)
        for name in analysis.exclude_labels:
            compared &= ~record_labels[name].to_numpy(dtype=bool)
        state = record_labels[upgrade_state_column(upgrade.name)].to_numpy(dtype=object)
        return pd.Series(np.where(compared, state, NEITHER), index=record_labels.index, name=SIDE_COL)

    def check(self, analysis: AnalysisSpec) -> None:
        """Raise when ``analysis`` names an upgrade, a turbine or a label the facts do not hold."""
        upgrade = self.upgrade(analysis.upgrade)
        known = set(self.turbine_names)
        named = [*(analysis.subjects or ()), *(analysis.references or ())]
        unknown = sorted(set(named) - known)
        if unknown:
            msg = f"analysis {analysis.name!r} names {unknown}, which the turbine table does not hold"
            raise ValueError(msg)
        outside = sorted(set(analysis.subjects or ()) - set(upgrade.turbines))
        if outside:
            msg = f"analysis {analysis.name!r} has subjects {outside}, which did not receive {upgrade.name!r}"
            raise ValueError(msg)
        both = sorted(set(analysis.subjects or upgrade.turbines) & set(analysis.references or ()))
        if both:
            msg = f"analysis {analysis.name!r} names {both} as both subject and reference"
            raise ValueError(msg)
        labels = {label.name for label in self.labels}
        stray = sorted({*analysis.include_labels, *analysis.exclude_labels} - labels)
        if stray:
            msg = f"analysis {analysis.name!r} names the labels {stray}; the facts hold {sorted(labels)}"
            raise ValueError(msg)

    def subjects_of(self, analysis: AnalysisSpec) -> list[str]:
        """Return the analysis's subjects: those it names, or every turbine that received its upgrade."""
        return list(analysis.subjects if analysis.subjects is not None else self.upgrade(analysis.upgrade).turbines)

    def references_of(self, analysis: AnalysisSpec) -> list[str]:
        """Return the analysis's candidate references: those it names, or every turbine unchanged over its period.

        A turbine is unchanged when, over the period, its state under every upgrade is one constant
        state of on or off. A turbine upgraded before the period starts is unchanged.
        """
        if analysis.references is not None:
            return list(analysis.references)
        subjects = set(self.subjects_of(analysis))
        index = pd.DatetimeIndex(self.records.index.unique()).sort_values()
        index = index[_within(index, analysis.period, end_inclusive=False)]
        return [t for t in self.turbine_names if t not in subjects and self._unchanged(t, index=index)]

    def changed_turbines(
        self,
        analysis: AnalysisSpec,
        *,
        subject: str,
        span: Window,
        near: pd.Timedelta = NEAR_CHANGEOVER,
    ) -> ChangedTimes:
        """Return the times each turbine counts as changed while ``subject`` is measured over ``span``.

        The subject counts as changed throughout. Every other turbine's change windows under every
        upgrade follow :func:`~wind_up.record_selection.changed_times` against the subject's own
        change under the analysis's upgrade; a turbine in neither state by an override counts as
        changed while it is. A change counts only upwind of the turbines within its upgrade's reach.

        :param analysis: the analysis the subject belongs to
        :param subject: the turbine measured
        :param span: ``[start, end)`` of the records the measurement uses
        :param near: how close to the subject's change another change counts as changed throughout
        """
        start, end = span
        first_record = pd.DatetimeIndex(self.records.index).min()
        upgrade = self.upgrade(analysis.upgrade)
        own = upgrade.change_windows(subject, first_record=first_record)
        if subject not in upgrade.turbines or not own:
            msg = f"{subject!r} did not receive {upgrade.name!r}, so it is not a subject of {analysis.name!r}"
            raise ValueError(msg)
        changeover = own[0] if upgrade.mode == "toggle" else upgrade.changeover(subject)
        reaches = {u.name: MAX_REACH_D if u.reach_d is None else u.reach_d for u in self.upgrades}
        works: dict[str, list[WorksWindow]] = {}
        for other in self.upgrades:
            for turbine in self.turbine_names:
                windows = other.change_windows(turbine, first_record=first_record)
                works.setdefault(turbine, []).extend((a, b, reaches[other.name]) for a, b in windows)
        layout = self.layout if min(reaches.values()) < MAX_REACH_D else None
        changed = changed_times(
            test_wtg=subject,
            works={t: sorted(set(w)) for t, w in works.items() if w},
            changeover=changeover,
            start=start,
            end=end,
            near=near,
            test_reach_d=reaches[upgrade.name],
            layout=layout,
        )
        for other in self.upgrades:
            for turbine in self.turbine_names:
                intervals = tuple(
                    ChangedInterval(a, b, turbines_within_reach(layout, waker=turbine, reach_d=reaches[other.name]))
                    for a, b in other.neither_windows(turbine)
                    if a < end and b > start and turbine != subject
                )
                if intervals:
                    changed[turbine] = (*changed.get(turbine, ()), *intervals)
        return changed

    def _unchanged(self, turbine: str, *, index: pd.DatetimeIndex) -> bool:
        """Return whether ``turbine`` holds one on or off state under every upgrade over ``index``."""
        for upgrade in self.upgrades:
            states = set(upgrade.states(index, turbine=turbine))
            if len(states) > 1 or NEITHER in states:
                return False
        return True

    def _turbine_row(self, turbine: str) -> pd.Series:
        """Return ``turbine``'s row of the turbine table."""
        rows = self.turbines[self.turbines[NAME_COL].astype(str) == turbine]
        if rows.empty:
            msg = f"the turbine table does not hold {turbine!r}"
            raise KeyError(msg)
        return rows.iloc[0]

    def _generic_states(self, grid: pd.DataFrame) -> pd.DataFrame:
        """Return ``grid`` with each record's operating state by wind-up's rules and the site's states."""
        turbines = grid[self.columns.turbine].astype(str)
        rated = turbines.map(dict(zip(self.turbine_names, self.turbines[RATED_POWER_COL].astype(float), strict=True)))
        ratings = pd.unique(rated.to_numpy())
        if len(ratings) <= 1:
            return self._label(grid, rated_power_kw=float(ratings[0]) if len(ratings) else 1.0)
        out = grid.copy()
        for column in STATE_COLUMNS:
            out[column] = None
        for rating in ratings:
            mine = (rated == rating).to_numpy()
            part = self._label(grid[mine], rated_power_kw=float(rating))
            for column in STATE_COLUMNS:
                out.loc[mine, column] = part[column].to_numpy()
        return out.astype({VALID_NORTHING_COL: bool, VALID_UPLIFT_COL: bool})

    def _label(self, records: pd.DataFrame, *, rated_power_kw: float) -> pd.DataFrame:
        """Return ``records`` labelled with their operating states at one rated power."""
        return label_operating_states(
            records,
            columns=self.columns,
            config=self.operating_states,
            timebase=self.timebase,
            rated_power_kw=rated_power_kw,
        )


def _within(index: pd.DatetimeIndex, period: Period, *, end_inclusive: bool) -> npt.NDArray[np.bool_]:
    """Return True at each timestamp of ``index`` inside ``period``, a side of None being open."""
    start, end = period
    inside = np.ones(len(index), dtype=bool)
    if start is not None:
        inside &= np.asarray(index >= start)
    if end is not None:
        inside &= np.asarray(index <= end) if end_inclusive else np.asarray(index < end)
    return inside


def _validity(
    *,
    state: npt.NDArray[np.object_],
    state_valid: npt.NDArray[np.bool_],
    withheld_by: Mapping[str, npt.NDArray[np.bool_]],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.object_]]:
    """Return each record's validity for a use and, where it is not valid, the state and labels that say so."""
    valid = state_valid.copy()
    reason = np.where(state_valid, "", state).astype(object)
    for name, withheld in withheld_by.items():
        valid &= ~withheld
        joined = np.where(reason == "", name, reason + REASON_SEPARATOR + name)
        reason = np.where(withheld, joined, reason).astype(object)
    return valid, reason
