"""What a campaign declaration declares: its upgrades, its analyses and its non-turbine measurements.

:func:`campaign_spec` turns one declared analysis into the :class:`CampaignSpec` the composed run
measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.campaigns.declaration import CampaignSpec, layout_coords
from wind_up.analysis import REPORTS
from wind_up.facts import NO_TIME, NonTurbineMeasurement

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from benchmarking.campaigns.declaration import Exclusion, StateOverride, Window
    from wind_up.layout import Layout
    from wind_up.toggle import ToggleSchedule


@dataclass(frozen=True)
class DeclaredUpgrade:
    """One upgrade as declared.

    :param name: the upgrade's name, as analyses refer to it
    :param turbines: the turbines that received it
    :param timing: the shared changeover, the toggle schedule, or None when each turbine's changeover is
        the end of its installation window
    :param installation_windows: per turbine, the windows ``[start, end)`` when it was being fitted
    :param wake_reach_d: how far its wake change reaches, in rotor diameters
    :param window_reach_d: per turbine, the installation windows with a reach of their own
    :param expected_uplift: the uplift expected, a fraction, when declared
    :param settling_time: toggle only: records within this long after a change are not valid for uplift or northing
    """

    name: str
    turbines: tuple[str, ...]
    timing: pd.Timestamp | ToggleSchedule | None
    installation_windows: Mapping[str, Sequence[Window]]
    wake_reach_d: float
    window_reach_d: Mapping[str, Mapping[Window, float]] = field(default_factory=dict)
    expected_uplift: float | None = None
    settling_time: pd.Timedelta = NO_TIME

    def reach_of(self, turbine: str, window: Window) -> float:
        """Return the reach of ``turbine``'s installation ``window``, in rotor diameters."""
        return self.window_reach_d.get(turbine, {}).get(window, self.wake_reach_d)


@dataclass(frozen=True)
class DeclaredAnalysis:
    """One analysis as declared.

    :param name: the analysis's name, which names its output folder when a declaration has several
    :param upgrade: the name of the upgrade measured
    :param subjects: the turbines measured; the upgrade's turbines when None
    :param references: the candidate references; every turbine neither measured nor excluded when None
    :param period: one ``(start, end)`` span, a span per subject, or None for wind-up to choose each one's
    :param include_labels: only a subject's records holding all of these labels are compared
    :param exclude_labels: no subject's record holding any of these labels is compared
    :param report: the uplifts wanted
    """

    name: str
    upgrade: str
    subjects: tuple[str, ...] | None = None
    references: tuple[str, ...] | None = None
    period: Window | dict[str, Window] | None = None
    include_labels: tuple[str, ...] = ()
    exclude_labels: tuple[str, ...] = ()
    report: tuple[str, ...] = REPORTS


@dataclass(frozen=True)
class DeclaredMeasurement:
    """A mast or LiDAR as declared: where it is and the file of its records.

    :param name: its name, distinct from every turbine's
    :param latitude: degrees north
    :param longitude: degrees east
    :param records_path: a parquet file of its records on a time index
    :param columns: the role of each column a method may read, role to column name
    """

    name: str
    latitude: float
    longitude: float
    records_path: Path
    columns: Mapping[str, str] = field(default_factory=dict)

    def load(self) -> NonTurbineMeasurement:
        """Read its records and return the measurement."""
        records = pd.read_parquet(self.records_path)
        if not isinstance(records.index, pd.DatetimeIndex):
            msg = f"the records of {self.name!r} in {self.records_path.name} need a time index"
            raise TypeError(msg)
        return NonTurbineMeasurement(
            name=self.name,
            latitude=self.latitude,
            longitude=self.longitude,
            records=records,
            columns=dict(self.columns),
        )


def campaign_spec(
    analysis: DeclaredAnalysis,
    *,
    upgrades: Sequence[DeclaredUpgrade],
    layout: Layout,
    excluded: Sequence[str],
    rated_power_kw: float,
    turbine_col: str,
    north_offsets: list[tuple[str, pd.Timestamp, float]] | None,
    exclusions: Sequence[Exclusion] = (),
    operating_state_overrides: Sequence[StateOverride] = (),
) -> CampaignSpec:
    """Return the campaign the composed run measures for ``analysis``.

    Its upgraded turbines are the analysis's subjects, with the analysed upgrade's timing and reach.
    Every upgrade's installation windows are the campaign's works, each at its own upgrade's reach.

    Raises when the analysis names labels, which the composed run does not read, and when the
    subjects or references are not turbines the campaign can give those roles.
    """
    if analysis.include_labels or analysis.exclude_labels:
        msg = f"the composed run does not read labels; analysis {analysis.name!r} names some"
        raise ValueError(msg)
    by_name = {u.name: u for u in upgrades}
    if analysis.upgrade not in by_name:
        msg = f"analysis {analysis.name!r} measures upgrade {analysis.upgrade!r}; the upgrades are {list(by_name)}"
        raise ValueError(msg)
    upgrade = by_name[analysis.upgrade]
    subjects = list(analysis.subjects) if analysis.subjects is not None else list(upgrade.turbines)
    outside = sorted(set(subjects) - set(upgrade.turbines))
    if outside:
        msg = f"analysis {analysis.name!r} measures {outside}, which did not receive {upgrade.name!r}"
        raise ValueError(msg)
    coords = layout_coords(layout)
    references = (
        list(analysis.references)
        if analysis.references
        else [w for w in coords if w not in set(subjects) | set(excluded)]
    )
    check_roles(upgraded=subjects, references=references, excluded=list(excluded), coords=coords)

    works: dict[str, list[Window]] = {}
    works_reach_d: dict[str, dict[Window, float]] = {}
    for declared in upgrades:
        for turbine, windows in declared.installation_windows.items():
            works.setdefault(turbine, []).extend(windows)
            for window in windows:
                if declared is not upgrade or window in declared.window_reach_d.get(turbine, {}):
                    works_reach_d.setdefault(turbine, {})[window] = declared.reach_of(turbine, window)
    works = {turbine: sorted(windows) for turbine, windows in works.items()}
    if isinstance(upgrade.timing, pd.Timestamp):
        check_works_clear_of(upgrade.timing, works={w: works.get(w, []) for w in subjects})
    elif upgrade.timing is None and len(upgrades) > 1:
        several = sorted(w for w in subjects if len(works.get(w, [])) > 1)
        if several:
            msg = (
                f"analysis {analysis.name!r}: {several} have other installation windows besides their "
                f"{upgrade.name!r} one; the composed run takes a subject's changeover from its only window, "
                f"so give {upgrade.name!r} a shared changeover"
            )
            raise ValueError(msg)
    return CampaignSpec(
        upgraded_turbines=subjects,
        upgrade_timing=upgrade.timing,
        candidate_references=references,
        excluded_turbines=list(excluded),
        layout=layout,
        north_offsets=north_offsets,
        rated_power_kw=rated_power_kw,
        analysis_period=dict(analysis.period) if isinstance(analysis.period, dict) else analysis.period,
        turbine_col=turbine_col,
        works=works,
        exclusions=list(exclusions),
        operating_state_overrides=list(operating_state_overrides),
        references_declared=bool(analysis.references),
        wake_reach_d=upgrade.wake_reach_d,
        works_reach_d=works_reach_d,
        settling_time=upgrade.settling_time,
    )


def check_roles(
    *, upgraded: list[str], references: list[str], excluded: list[str], coords: Mapping[str, tuple[float, float]]
) -> None:
    """Check every declared turbine has coordinates and holds exactly one role."""
    if not upgraded:
        msg = "the declaration lists no upgraded turbines, so there is nothing to estimate"
        raise ValueError(msg)
    declared = [*upgraded, *references, *excluded]
    unknown = sorted({w for w in declared if w not in coords})
    if unknown:
        msg = f"the turbines file has no row for {unknown}, so they have no coordinates"
        raise ValueError(msg)
    doubled = sorted({w for w in declared if declared.count(w) > 1})
    if doubled:
        msg = f"{doubled} appear in more than one turbine role; each turbine holds exactly one"
        raise ValueError(msg)


def check_works_clear_of(changeover: pd.Timestamp, *, works: Mapping[str, Sequence[Window]]) -> None:
    """Raise naming the upgraded turbines with a works window over the shared changeover."""
    over = sorted(w for w, windows in works.items() if any(s < changeover < e for s, e in windows))
    if over:
        msg = (
            f"{over} have a works window over timing.changeover; with a shared changeover the works "
            f"table holds a turbine's other works"
        )
        raise ValueError(msg)
