"""Read a campaign declaration: one YAML file of campaign facts, plus a turbines sidecar.

Campaign facts only. Method configuration stays on the method, the deliberate break from v0's
``WindUpConfig``, which mixes both into one file.

A declaration either lists its ``upgrades`` and ``analyses``, or declares one upgrade and one
analysis with the single-upgrade keys (``turbines.upgraded``, ``timing``, ``upgrade``,
``analysis_period`` and ``data.installation_windows``).
"""

from __future__ import annotations

import math
import numbers
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import yaml

from benchmarking.campaigns.declaration import CampaignSpec, layout_coords
from benchmarking.campaigns.declared import (
    DeclaredAnalysis,
    DeclaredMeasurement,
    DeclaredUpgrade,
    campaign_spec,
    check_roles,
    check_works_clear_of,
)
from benchmarking.harness.reanalysis import TimestampConvention
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule
from benchmarking.synthetic.sources.greenbyte import GREENBYTE_COLUMNS
from wind_up.analysis import REPORTS
from wind_up.facts import NO_TIME
from wind_up.labels import Label
from wind_up.layout import Layout, centroid
from wind_up.operating_state import OperatingStateConfig, StateValidity
from wind_up.wake_reach import MAX_REACH_D, reach_from_uplift

if TYPE_CHECKING:
    from benchmarking.campaigns.declaration import Exclusion, StateOverride, Window
    from benchmarking.synthetic import ColumnSchema

# The schemas a declaration may name. Inline schema definition is not supported: a source's
# column names belong to its adapter, not to a campaign.
SCHEMAS: dict[str, ColumnSchema] = {"hill_of_towie": HOT_COLUMNS, "greenbyte": GREENBYTE_COLUMNS}

# How a declaration names a farm-wide exclusion or operating state override.
ALL_TURBINES = "ALL"
PER_TURBINE_PERIOD = "chosen per upgraded turbine"

MODES = ("prepost", "toggle")

# The name of the one upgrade a single-upgrade declaration declares.
SHORTHAND_UPGRADE = "upgrade"

# How a declaration writes a state's northing and uplift validity.
VALID = "valid"
NOT_VALID = "not valid"
_VALIDITY = {VALID: True, NOT_VALID: False}
_OPERATING_STATE_KEYS = (
    "site_state_column",
    "parked_pitch_above_deg",
    "parked_pitch_below_deg",
    "site_states",
    "label_column",
    "labels",
)
_STATE_VALIDITY_KEYS = ("northing", "waking", "uplift")
_TIMESTAMP_KEYS = ("convention", "time_zone")
_SITE_KEYS = ("offshore",)
_UPGRADE_KEYS = ("expected_uplift", "wake_reach_d", "settling_time")
_UPGRADES_ENTRY_KEYS = (
    "name",
    "turbines",
    "timing",
    "installation_windows",
    "expected_uplift",
    "wake_reach_d",
    "settling_time",
)
_ANALYSIS_KEYS = ("name", "upgrade", "subjects", "references", "period", "include_labels", "exclude_labels", "report")
_LABEL_KEYS = ("name", "column", "withholds")
_MEASUREMENT_KEYS = ("name", "latitude", "longitude", "records", "columns")
_EXCLUSION_KEYS = ("turbine", "start", "end")
_OVERRIDE_KEYS = ("turbine", "start", "end", "state")
# The keys a declaration that lists its upgrades and analyses does not use.
_SHORTHAND_KEYS = ("timing", "upgrade", "analysis_period")
_SHORTHAND_ROLES = ("upgraded", "references")
# Keys renamed, new name to old; the old name is still read.
_RENAMED = {
    "operating_state_overrides": "label_overrides",
    "installation_windows": "works",
    "site_state_column": "label_column",
    "site_states": "labels",
}
# A works table time with a time of day; any other spelling pandas reads is a date alone.
_TIME_OF_DAY = re.compile(r"\d{1,2}:\d{2}")
# The works table's optional per-row reach columns, matched by prefix.
EXPECTED_UPLIFT_PREFIX = "Expected uplift"
WAKE_REACH_PREFIX = "Wake reach"

# The reanalysis centroid is rounded to this many decimals, and taken over the whole turbines
# file rather than the declared roles, so every campaign on one site shares a cache entry and
# changing the reference set does not move a model input.


class _RawTimestampLoader(yaml.SafeLoader):
    """SafeLoader with the implicit timestamp resolver removed, so times arrive as strings.

    What PyYAML makes of a timestamp varies by version -- a naive datetime, an aware one, or a
    bare date. Taking the raw scalar and coercing it here makes the timezone rule this module's
    own rather than the YAML library's.
    """


_RawTimestampLoader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


@dataclass(frozen=True)
class Declaration:
    """A campaign declared in YAML, resolved into what is needed to run it.

    :param name: names the run's output subdirectory, so a declaration is self-identifying
    :param columns: the source-native schema the SCADA is keyed by
    :param scada_path: the SCADA parquet, resolved relative to the declaration
    :param centroid: the site's ``(latitude, longitude)`` centroid over the whole turbines file,
        rounded, which reanalysis is self-served from
    :param upgrades: the upgrades declared
    :param analyses: the analyses declared, in order
    :param specs: the campaign the composed run measures for each analysis, by name; an analysis
        naming labels has none
    :param operating_state: the site's operating states and parked-pitch rule; generic states only
        when none is declared
    :param timestamps: the SCADA timestamp convention and time zone; period start, UTC by default
    :param offshore: whether reanalysis is drawn from a sea grid cell rather than a land one
    :param labels: the analyst's own labels, columns of the SCADA
    :param non_turbine_measurements: the masts and LiDARs declared
    :param shorthand: declared with the single-upgrade keys
    """

    name: str
    columns: ColumnSchema
    scada_path: Path
    centroid: tuple[float, float]
    upgrades: tuple[DeclaredUpgrade, ...]
    analyses: tuple[DeclaredAnalysis, ...]
    specs: dict[str, CampaignSpec]
    operating_state: OperatingStateConfig = field(default_factory=OperatingStateConfig)
    timestamps: TimestampConvention = field(default_factory=TimestampConvention)
    offshore: bool = False
    labels: tuple[Label, ...] = ()
    non_turbine_measurements: tuple[DeclaredMeasurement, ...] = ()
    shorthand: bool = True

    @property
    def spec(self) -> CampaignSpec:
        """The campaign of the declaration's one analysis."""
        if len(self.analyses) != 1:
            msg = f"{self.name!r} declares {len(self.analyses)} analyses; name one with spec_for"
            raise ValueError(msg)
        return self.spec_for(self.analyses[0].name)

    def spec_for(self, analysis: str) -> CampaignSpec:
        """Return the campaign the composed run measures for the analysis named ``analysis``."""
        if analysis not in self.specs:
            names = [a.name for a in self.analyses]
            if analysis in names:
                msg = f"analysis {analysis!r} names labels, which the composed run does not read"
                raise ValueError(msg)
            msg = f"{self.name!r} declares no analysis {analysis!r}; it declares {names}"
            raise KeyError(msg)
        return self.specs[analysis]

    @property
    def expected_uplift(self) -> float | None:
        """The expected uplift of the upgrade the declaration's one analysis measures."""
        (analysis,) = self.analyses
        return next(u.expected_uplift for u in self.upgrades if u.name == analysis.upgrade)

    @property
    def cell_selection(self) -> Literal["land", "sea"]:
        """Return the Open-Meteo grid-cell selection for this site."""
        return "sea" if self.offshore else "land"

    def resolved(self) -> dict[str, Any]:
        """Return the resolved campaign facts, for echoing into the run output.

        Timestamps appear as the UTC values the declaration actually resolved to, so a
        mis-declared timezone is visible rather than silent.
        """
        resolved: dict[str, Any] = {
            "name": self.name,
            "scada": str(self.scada_path),
            "schema": {v: k for k, v in SCHEMAS.items()}.get(self.columns, "custom"),
            "timestamps": {"convention": self.timestamps.convention, "time_zone": self.timestamps.time_zone},
            "site": {"offshore": self.offshore},
        }
        first = next(iter(self.specs.values()), None)
        resolved |= self._resolved_shorthand() if self.shorthand else self._resolved_upgrades()
        if first is not None:
            resolved["northing"] = {
                "discover": first.north_offsets is None,
                "table": [] if first.north_offsets is None else [[w, str(t), o] for w, t, o in first.north_offsets],
            }
        resolved["reanalysis"] = {
            "centroid": list(self.centroid),
            "cell_selection": self.cell_selection,
            "window": "the whole SCADA record, in whole calendar years",
        }
        resolved["operating_state"] = _resolved_operating_state(self.operating_state)
        if self.shorthand:
            spec = self.spec
            resolved["upgrade"] = {
                "expected_uplift": self.expected_uplift,
                "wake_reach_d": spec.wake_reach_d,
                "settling_time": str(spec.settling_time),
            }
            if spec.works:
                resolved["installation_windows"] = {
                    t: [[str(s), str(e), spec.reach_of(t, (s, e))] for s, e in windows]
                    for t, windows in spec.works.items()
                }
        if first is not None and first.exclusions:
            resolved["exclusions"] = [
                {"turbine": ALL_TURBINES if who is None else who, "start": str(s), "end": str(e)}
                for who, s, e in first.exclusions
            ]
        if first is not None and first.operating_state_overrides:
            resolved["operating_state_overrides"] = [
                {"turbine": ALL_TURBINES if who is None else who, "start": str(s), "end": str(e), "state": state}
                for who, s, e, state in first.operating_state_overrides
            ]
        if self.labels:
            resolved["labels"] = [
                {"name": label.name, "column": label.column, "withholds": sorted(label.withholds)}
                for label in self.labels
            ]
        if self.non_turbine_measurements:
            resolved["non_turbine_measurements"] = [
                {
                    "name": m.name,
                    "latitude": m.latitude,
                    "longitude": m.longitude,
                    "records": str(m.records_path),
                    "columns": dict(m.columns),
                }
                for m in self.non_turbine_measurements
            ]
        return resolved

    def _resolved_shorthand(self) -> dict[str, Any]:
        """Return the turbines, timing and analysis period of a single-upgrade declaration."""
        spec = self.spec
        timing: dict[str, Any] = {"mode": spec.mode}
        if isinstance(spec.upgrade_timing, ToggleSchedule):
            timing |= _resolved_schedule(spec.upgrade_timing)
        elif spec.upgrade_timing is None:
            timing["changeover"] = {t: str(spec.timing_for(t)) for t in spec.upgraded_turbines}
        else:
            timing["changeover"] = str(spec.upgrade_timing)
        return {
            "turbines": {
                "upgraded": list(spec.upgraded_turbines),
                "references": list(spec.candidate_references),
                "excluded": list(spec.excluded_turbines),
                "rated_power_kw": spec.rated_power_kw,
            },
            "timing": timing,
            "analysis_period": _resolved_period(spec.analysis_period),
        }

    def _resolved_upgrades(self) -> dict[str, Any]:
        """Return the turbines, upgrades and analyses of a declaration that lists them."""
        first = next(iter(self.specs.values()), None)
        upgrades = []
        for upgrade in self.upgrades:
            entry: dict[str, Any] = {"name": upgrade.name, "turbines": list(upgrade.turbines)}
            if isinstance(upgrade.timing, ToggleSchedule):
                entry["timing"] = {"mode": "toggle", **_resolved_schedule(upgrade.timing)}
                entry["settling_time"] = str(upgrade.settling_time)
            elif upgrade.timing is not None:
                entry["timing"] = {"mode": "prepost", "changeover": str(upgrade.timing)}
            entry["expected_uplift"] = upgrade.expected_uplift
            entry["wake_reach_d"] = upgrade.wake_reach_d
            if upgrade.installation_windows:
                entry["installation_windows"] = {
                    t: [[str(s), str(e), upgrade.reach_of(t, (s, e))] for s, e in windows]
                    for t, windows in upgrade.installation_windows.items()
                }
            upgrades.append(entry)
        analyses = []
        for analysis in self.analyses:
            spec = self.specs.get(analysis.name)
            entry = {"name": analysis.name, "upgrade": analysis.upgrade}
            if spec is not None:
                entry["subjects"] = list(spec.upgraded_turbines)
                entry["references"] = list(spec.candidate_references)
                entry["period"] = _resolved_period(spec.analysis_period)
            else:
                entry["subjects"] = None if analysis.subjects is None else list(analysis.subjects)
                entry["references"] = None if analysis.references is None else list(analysis.references)
                entry["period"] = _resolved_period(analysis.period)
            entry["include_labels"] = list(analysis.include_labels)
            entry["exclude_labels"] = list(analysis.exclude_labels)
            entry["report"] = list(analysis.report)
            analyses.append(entry)
        return {
            "turbines": {
                "excluded": [] if first is None else list(first.excluded_turbines),
                "rated_power_kw": None if first is None else first.rated_power_kw,
            },
            "upgrades": upgrades,
            "analyses": analyses,
        }


def load_declaration(path: str | Path) -> Declaration:
    """Read the campaign declaration at ``path`` and resolve it.

    Paths named in the declaration are resolved relative to the declaration's own directory, so a
    campaign folder can be moved or copied whole.
    """
    path = Path(path)
    with path.open() as stream:
        raw = yaml.load(stream, Loader=_RawTimestampLoader)  # noqa: S506 - subclass of SafeLoader
    root = path.parent

    data = _section(raw, "data")
    columns = _schema(str(data["schema"]))
    scada_path = _resolve(root, str(data["scada"]), what="scada")
    layout = _read_turbines(_resolve(root, str(data["turbines"]), what="turbines"))
    coords = layout_coords(layout)
    roles = _section(raw, "turbines")
    excluded = [str(w) for w in roles.get("excluded") or []]

    if "wake_unchanged" in raw:
        msg = (
            "wake_unchanged is no longer declared: a change that leaves the wakes as they were declares "
            "upgrade.wake_reach_d: 0"
        )
        raise ValueError(msg)
    offshore = _offshore(raw.get("site"))
    exclusions = _exclusions(raw.get("exclusions") or [])
    _check_known({w: None for w, _, _ in exclusions if w is not None}, coords=coords, what="the exclusions")
    operating_state = operating_state_config(raw.get("operating_state"))
    overrides = _state_overrides(
        _renamed(raw, "operating_state_overrides", where="the declaration") or [],
        states=set(operating_state.validity()),
    )
    _check_known({w: None for w, *_ in overrides if w is not None}, coords=coords, what="the operating state overrides")
    labels = _labels(raw.get("labels") or [])
    measurements = _measurements(raw.get("non_turbine_measurements") or [], root=root, coords=coords)

    shorthand = "upgrades" not in raw
    upgrades: tuple[DeclaredUpgrade, ...]
    analyses: tuple[DeclaredAnalysis, ...]
    if shorthand:
        if "analyses" in raw:
            msg = "analyses are declared with the upgrades they measure; the declaration has no upgrades"
            raise ValueError(msg)
        upgrade, analysis = _shorthand(raw, roles=roles, data=data, root=root, coords=coords, offshore=offshore)
        upgrades, analyses = (upgrade,), (analysis,)
    else:
        _check_no_shorthand(raw, roles=roles, data=data)
        upgrades = _upgrades(raw["upgrades"], root=root, coords=coords, offshore=offshore)
        analyses = _analyses(raw.get("analyses"), upgrades=upgrades, labels=labels)

    north_offsets = _north_offsets(raw.get("northing"))
    specs = {
        analysis.name: campaign_spec(
            analysis,
            upgrades=upgrades,
            layout=layout,
            excluded=excluded,
            rated_power_kw=float(roles["rated_power_kw"]),
            turbine_col=columns.turbine,
            north_offsets=north_offsets,
            exclusions=exclusions,
            operating_state_overrides=overrides,
        )
        for analysis in analyses
        if not analysis.include_labels and not analysis.exclude_labels
    }
    return Declaration(
        name=_path_component(str(raw["name"]), what="campaign name"),
        columns=columns,
        scada_path=scada_path,
        centroid=centroid(coords),
        upgrades=upgrades,
        analyses=analyses,
        specs=specs,
        operating_state=operating_state,
        timestamps=_timestamps(data.get("timestamps")),
        offshore=offshore,
        labels=labels,
        non_turbine_measurements=measurements,
        shorthand=shorthand,
    )


def _shorthand(
    raw: dict, *, roles: dict, data: dict, root: Path, coords: dict[str, tuple[float, float]], offshore: bool
) -> tuple[DeclaredUpgrade, DeclaredAnalysis]:
    """Read a single-upgrade declaration's one upgrade and one analysis."""
    upgraded = [str(w) for w in roles.get("upgraded", [])]
    excluded = [str(w) for w in roles.get("excluded") or []]
    # Both default to the obvious campaign: nothing excluded, and every other turbine in the
    # turbines file offered as a reference. Declaring either narrows it.
    declared_references = roles.get("references")
    references = (
        [str(w) for w in declared_references]
        if declared_references
        else [w for w in coords if w not in set(upgraded) | set(excluded)]
    )
    check_roles(upgraded=upgraded, references=references, excluded=excluded, coords=coords)
    expected_uplift, wake_reach_d, settling = _upgrade(raw.get("upgrade"), offshore=offshore)
    table = _renamed(data, "installation_windows", where="data")
    works, works_reach_d = (
        read_works(_resolve(root, str(table), what="installation windows"), offshore=offshore) if table else ({}, {})
    )
    _check_known(works, coords=coords, what="the installation windows table")
    timing = _timing(_section(raw, "timing"), has_works=bool(works))
    if isinstance(timing, pd.Timestamp):
        check_works_clear_of(timing, works={w: works.get(w, []) for w in upgraded})
    settling_time = _settling_time(settling, timing=timing, what="the upgrade")
    upgrade = DeclaredUpgrade(
        name=SHORTHAND_UPGRADE,
        turbines=tuple(upgraded),
        timing=timing,
        installation_windows=works,
        wake_reach_d=wake_reach_d,
        window_reach_d=works_reach_d,
        expected_uplift=expected_uplift,
        settling_time=settling_time,
    )
    analysis = DeclaredAnalysis(
        name=str(raw["name"]),
        upgrade=SHORTHAND_UPGRADE,
        subjects=tuple(upgraded),
        references=tuple(str(w) for w in declared_references) if declared_references else None,
        period=_analysis_period(raw.get("analysis_period"), upgraded=upgraded),
    )
    return upgrade, analysis


def _check_no_shorthand(raw: dict, *, roles: dict, data: dict) -> None:
    """Raise naming the single-upgrade keys a declaration that lists its upgrades also uses."""
    used = [key for key in _SHORTHAND_KEYS if key in raw]
    used += [f"turbines.{key}" for key in _SHORTHAND_ROLES if key in roles]
    used += [f"data.{key}" for key in ("installation_windows", "works") if key in data]
    if used:
        msg = (
            f"a declaration that lists its upgrades declares timings, windows, subjects, references and "
            f"periods on its upgrades and analyses, not with {used}"
        )
        raise ValueError(msg)


def _upgrades(
    entries: object, *, root: Path, coords: dict[str, tuple[float, float]], offshore: bool
) -> tuple[DeclaredUpgrade, ...]:
    """Read the ``upgrades`` list."""
    if not isinstance(entries, list) or not entries:
        msg = "upgrades is a list of at least one upgrade"
        raise ValueError(msg)
    upgrades = tuple(_upgrade_entry(entry, root=root, coords=coords, offshore=offshore) for entry in entries)
    names = [u.name for u in upgrades]
    doubled = sorted({n for n in names if names.count(n) > 1})
    if doubled:
        msg = f"each upgrade needs its own name; {doubled} name more than one"
        raise ValueError(msg)
    return upgrades


def _upgrade_entry(
    entry: object, *, root: Path, coords: dict[str, tuple[float, float]], offshore: bool
) -> DeclaredUpgrade:
    """Read one entry of the ``upgrades`` list."""
    if not isinstance(entry, dict) or "name" not in entry:
        msg = f"an upgrade is a mapping with a name, got {entry!r}"
        raise ValueError(msg)
    name = str(entry["name"])
    stray = sorted(set(map(str, entry)) - set(_UPGRADES_ENTRY_KEYS))
    if stray:
        msg = f"upgrade {name!r} has unknown keys {stray}; known keys are {list(_UPGRADES_ENTRY_KEYS)}"
        raise ValueError(msg)
    windows, window_reach = _installation_windows(
        entry.get("installation_windows"), root=root, offshore=offshore, upgrade=name
    )
    _check_known(windows, coords=coords, what=f"the installation windows of {name!r}")
    declared_turbines = entry.get("turbines")
    turbines = [str(w) for w in declared_turbines] if declared_turbines else list(windows)
    if not turbines:
        msg = f"upgrade {name!r} needs its turbines, or installation windows naming them"
        raise ValueError(msg)
    _check_known(dict.fromkeys(turbines), coords=coords, what=f"upgrade {name!r}")
    others = sorted(set(windows) - set(turbines))
    if others:
        msg = f"the installation windows of {name!r} name {others}, which are not among its turbines"
        raise ValueError(msg)
    timing = _timing(entry["timing"], has_works=bool(windows)) if "timing" in entry else None
    if timing is None and not windows:
        msg = f"upgrade {name!r} needs a timing or installation windows"
        raise ValueError(msg)
    if timing is None:
        bare = sorted(set(turbines) - set(windows))
        if bare:
            msg = (
                f"upgrade {name!r} has no timing, so each of its turbines needs an installation window; "
                f"{bare} have none"
            )
            raise ValueError(msg)
    if "expected_uplift" in entry and "wake_reach_d" in entry:
        msg = f"upgrade {name!r} declares both expected_uplift and wake_reach_d; declare at most one"
        raise ValueError(msg)
    expected_uplift = (
        _number(entry["expected_uplift"], what=f"expected_uplift of {name!r}") if "expected_uplift" in entry else None
    )
    if "wake_reach_d" in entry:
        reach = _reach(entry["wake_reach_d"], what=f"wake_reach_d of {name!r}")
    elif expected_uplift is not None:
        reach = reach_from_uplift(expected_uplift, offshore=offshore)
    else:
        reach = MAX_REACH_D
    return DeclaredUpgrade(
        name=name,
        turbines=tuple(turbines),
        timing=timing,
        installation_windows=windows,
        wake_reach_d=reach,
        window_reach_d=window_reach,
        expected_uplift=expected_uplift,
        settling_time=_settling_time(entry.get("settling_time"), timing=timing, what=f"upgrade {name!r}"),
    )


def _installation_windows(
    value: object, *, root: Path, offshore: bool, upgrade: str
) -> tuple[dict[str, list[Window]], dict[str, dict[Window, float]]]:
    """Read an upgrade's installation windows: a table file, or a list of ``{turbine, start, end}``."""
    if value is None:
        return {}, {}
    if isinstance(value, str):
        return read_works(_resolve(root, value, what=f"installation windows of {upgrade!r}"), offshore=offshore)
    if not isinstance(value, list):
        msg = f"the installation windows of {upgrade!r} are a table file or a list of {{turbine, start, end}}"
        raise ValueError(msg)  # noqa: TRY004 - a declaration error, reported like the others
    windows: dict[str, list[Window]] = {}
    for entry in value:
        if not isinstance(entry, dict) or set(map(str, entry)) != set(_EXCLUSION_KEYS):
            msg = f"an installation window of {upgrade!r} has exactly the keys {list(_EXCLUSION_KEYS)}, got {entry!r}"
            raise ValueError(msg)
        turbine, start, end = _window_entry(entry, what=f"installation window of {upgrade!r}")
        if turbine is None:
            msg = f"an installation window of {upgrade!r} names one turbine, not {ALL_TURBINES}"
            raise ValueError(msg)
        windows.setdefault(turbine, []).append((start, end))
    return {turbine: sorted(spans) for turbine, spans in windows.items()}, {}


def _settling_time(value: object, *, timing: object, what: str) -> pd.Timedelta:
    """Read a settling time: a toggle upgrade's, zero when not declared."""
    if value is None:
        return NO_TIME
    if not isinstance(timing, ToggleSchedule):
        msg = f"{what} declares a settling time, which is for a toggle upgrade"
        raise ValueError(msg)  # noqa: TRY004 - a declaration error, reported like the others
    settling = pd.Timedelta(str(value))
    if settling < NO_TIME:
        msg = f"the settling time of {what} must not be negative, got {settling}"
        raise ValueError(msg)
    return settling


def _analyses(
    entries: object, *, upgrades: tuple[DeclaredUpgrade, ...], labels: tuple[Label, ...]
) -> tuple[DeclaredAnalysis, ...]:
    """Read the ``analyses`` list."""
    if not isinstance(entries, list) or not entries:
        msg = "a declaration that lists its upgrades lists its analyses: analyses is a list of at least one analysis"
        raise ValueError(msg)
    analyses = tuple(_analysis_entry(entry, upgrades=upgrades, labels=labels) for entry in entries)
    names = [a.name for a in analyses]
    doubled = sorted({n for n in names if names.count(n) > 1})
    if doubled:
        msg = f"each analysis needs its own name; {doubled} name more than one"
        raise ValueError(msg)
    return analyses


def _analysis_entry(
    entry: object, *, upgrades: tuple[DeclaredUpgrade, ...], labels: tuple[Label, ...]
) -> DeclaredAnalysis:
    """Read one entry of the ``analyses`` list."""
    if not isinstance(entry, dict) or "name" not in entry or "upgrade" not in entry:
        msg = f"an analysis is a mapping with a name and the upgrade it measures, got {entry!r}"
        raise ValueError(msg)
    name = _path_component(str(entry["name"]), what="analysis name")
    stray = sorted(set(map(str, entry)) - set(_ANALYSIS_KEYS))
    if stray:
        msg = f"analysis {name!r} has unknown keys {stray}; known keys are {list(_ANALYSIS_KEYS)}"
        raise ValueError(msg)
    by_name = {u.name: u for u in upgrades}
    upgrade = str(entry["upgrade"])
    if upgrade not in by_name:
        msg = f"analysis {name!r} measures upgrade {upgrade!r}; the upgrades are {list(by_name)}"
        raise ValueError(msg)
    subjects = _names(entry.get("subjects"))
    known = {label.name for label in labels}
    include, exclude = _names(entry.get("include_labels")) or (), _names(entry.get("exclude_labels")) or ()
    unknown = sorted({*include, *exclude} - known)
    if unknown:
        msg = f"analysis {name!r} names the labels {unknown}; the declared labels are {sorted(known)}"
        raise ValueError(msg)
    report = entry.get("report", list(REPORTS))
    report = (str(report),) if isinstance(report, str) else tuple(str(r) for r in report)
    wrong = sorted(set(report) - set(REPORTS)) if report else ["nothing"]
    if wrong:
        msg = f"analysis {name!r} reports {wrong}; it reports one or both of {list(REPORTS)}"
        raise ValueError(msg)
    return DeclaredAnalysis(
        name=name,
        upgrade=upgrade,
        subjects=subjects,
        references=_names(entry.get("references")),
        period=_analysis_period(
            entry.get("period"), upgraded=list(subjects or by_name[upgrade].turbines), what=f"period of {name!r}"
        ),
        include_labels=include,
        exclude_labels=exclude,
        report=report,
    )


def _names(value: object) -> tuple[str, ...] | None:
    """Return a declared list of names, or None when not declared."""
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, list):
        msg = f"expected a list of names, got {value!r}"
        raise ValueError(msg)  # noqa: TRY004 - a declaration error, reported like the others
    return tuple(str(v) for v in value)


def _labels(entries: object) -> tuple[Label, ...]:
    """Read the analyst's ``labels``."""
    if not isinstance(entries, list):
        msg = f"labels is a list of {{name, column, withholds}}, got {entries!r}"
        raise ValueError(msg)  # noqa: TRY004 - a declaration error, reported like the others
    labels = []
    for entry in entries:
        if not isinstance(entry, dict) or not {"name", "column"} <= set(map(str, entry)):
            msg = f"a label has a name and a column, got {entry!r}"
            raise ValueError(msg)
        stray = sorted(set(map(str, entry)) - set(_LABEL_KEYS))
        if stray:
            msg = f"label {entry['name']!r} has unknown keys {stray}; known keys are {list(_LABEL_KEYS)}"
            raise ValueError(msg)
        withholds = entry.get("withholds") or []
        labels.append(
            Label(
                name=str(entry["name"]),
                column=str(entry["column"]),
                withholds=frozenset([str(withholds)] if isinstance(withholds, str) else map(str, withholds)),
            )
        )
    names = [label.name for label in labels]
    doubled = sorted({n for n in names if names.count(n) > 1})
    if doubled:
        msg = f"each label needs its own name; {doubled} name more than one"
        raise ValueError(msg)
    return tuple(labels)


def _measurements(
    entries: object, *, root: Path, coords: dict[str, tuple[float, float]]
) -> tuple[DeclaredMeasurement, ...]:
    """Read the ``non_turbine_measurements``: name, position, records file and column roles."""
    if not isinstance(entries, list):
        msg = f"non_turbine_measurements is a list, got {entries!r}"
        raise ValueError(msg)  # noqa: TRY004 - a declaration error, reported like the others
    measurements = []
    for entry in entries:
        if not isinstance(entry, dict) or not {"name", "latitude", "longitude", "records"} <= set(map(str, entry)):
            msg = f"a non-turbine measurement has a name, latitude, longitude and records file, got {entry!r}"
            raise ValueError(msg)
        name = str(entry["name"])
        stray = sorted(set(map(str, entry)) - set(_MEASUREMENT_KEYS))
        if stray:
            msg = f"non-turbine measurement {name!r} has unknown keys {stray}; known keys are {list(_MEASUREMENT_KEYS)}"
            raise ValueError(msg)
        if name in coords:
            msg = f"the non-turbine measurement {name!r} has a turbine's name"
            raise ValueError(msg)
        columns = entry.get("columns") or {}
        if not isinstance(columns, dict):
            msg = f"the columns of {name!r} map a role to a column name, got {columns!r}"
            raise ValueError(msg)  # noqa: TRY004 - a declaration error, reported like the others
        measurements.append(
            DeclaredMeasurement(
                name=name,
                latitude=_number(entry["latitude"], what=f"latitude of {name!r}"),
                longitude=_number(entry["longitude"], what=f"longitude of {name!r}"),
                records_path=_resolve(root, str(entry["records"]), what=f"records of {name!r}"),
                columns={str(role): str(column) for role, column in columns.items()},
            )
        )
    names = [m.name for m in measurements]
    doubled = sorted({n for n in names if names.count(n) > 1})
    if doubled:
        msg = f"each non-turbine measurement needs its own name; {doubled} name more than one"
        raise ValueError(msg)
    return tuple(measurements)


def _renamed(block: dict, key: str, *, where: str) -> Any:  # noqa: ANN401 - whatever the declaration holds
    """Return ``block``'s value under ``key`` or, read the same way, under the key's old name."""
    old = _RENAMED[key]
    if key in block and old in block:
        msg = f"{where} declares both {key} and its old name {old}; keep {key}"
        raise ValueError(msg)
    return block.get(key, block.get(old))


def _resolved_schedule(schedule: ToggleSchedule) -> dict[str, Any]:
    """Return a toggle schedule as the resolved echo shows it."""
    return {"start": str(schedule.start), "period": str(schedule.period), "start_on": schedule.start_on}


def read_works(path: Path, *, offshore: bool = False) -> tuple[dict[str, list[Window]], dict[str, dict[Window, float]]]:
    """Read a works table: ``Turbine`` plus the first ``First date...`` and ``Last date...`` columns.

    Each row is one window. A date alone is a whole day: the window runs from the first date's 00:00 to
    the day after the last date, UTC. A date with a time of day is that instant, the last one exclusive;
    without a zone it is UTC. A turbine may have several rows. Returns the windows, and per turbine the
    windows with a reach of their own: from the optional ``Wake reach...`` column (rotor diameters), else
    from the optional ``Expected uplift...`` column (a fraction) through the reach table.
    """
    header = pd.read_csv(path, nrows=0).columns
    frame = pd.read_csv(path, dtype={c: str for c in header if str(c).startswith(("First date", "Last date"))})
    uplift_col = next((c for c in frame.columns if str(c).startswith(EXPECTED_UPLIFT_PREFIX)), None)
    reach_col = next((c for c in frame.columns if str(c).startswith(WAKE_REACH_PREFIX)), None)
    first = next((c for c in frame.columns if str(c).startswith("First date")), None)
    last = next((c for c in frame.columns if str(c).startswith("Last date")), None)
    if "Turbine" not in frame.columns or first is None or last is None:
        msg = (
            f"the works table {path.name} needs a Turbine column and columns starting 'First date' and "
            f"'Last date'; it has {list(frame.columns)}"
        )
        raise ValueError(msg)
    works: dict[str, list[Window]] = {}
    reaches: dict[str, dict[Window, float]] = {}
    for row, (turbine, first_day, last_day) in enumerate(
        zip(frame["Turbine"].astype(str), frame[first], frame[last], strict=True)
    ):
        if pd.isna(first_day) or pd.isna(last_day):
            msg = f"the works table {path.name} gives {turbine} a blank first or last date"
            raise ValueError(msg)
        start = _works_time(first_day, end=False)
        end = _works_time(last_day, end=True)
        if end <= start:
            msg = f"the works table {path.name} gives {turbine} a window ending at or before its start"
            raise ValueError(msg)
        works.setdefault(turbine, []).append((start, end))
        reach = _row_reach(frame, row=row, uplift_col=uplift_col, reach_col=reach_col, offshore=offshore)
        if reach is not None:
            reaches.setdefault(turbine, {})[(start, end)] = reach
    return {turbine: sorted(windows) for turbine, windows in works.items()}, reaches


def _works_time(value: str, *, end: bool) -> pd.Timestamp:
    """Return a works table time in UTC: a date alone is its 00:00, or the next day's for an ``end``."""
    text = str(value).strip()
    if not _TIME_OF_DAY.search(text):
        day = _timestamp(pd.Timestamp(text))
        return day + pd.Timedelta(days=1) if end else day
    return _timestamp(text)


def _row_reach(frame: pd.DataFrame, *, row: int, uplift_col: object, reach_col: object, offshore: bool) -> float | None:
    """Return a works table row's own reach, or None when it takes the campaign's."""
    if reach_col is not None and pd.notna(frame[reach_col].iloc[row]):
        return _reach(frame[reach_col].iloc[row], what=f"works table row {row + 1} {reach_col}")
    if uplift_col is not None and pd.notna(frame[uplift_col].iloc[row]):
        uplift = _number(frame[uplift_col].iloc[row], what=f"works table row {row + 1} {uplift_col}")
        return reach_from_uplift(uplift, offshore=offshore)
    return None


def _upgrade(block: dict | None, *, offshore: bool) -> tuple[float, float, object]:
    """Read the ``upgrade`` block: the expected uplift, the reach (declared or from the table), the settling time."""
    if block is None or "expected_uplift" not in block:
        msg = (
            "the declaration needs upgrade.expected_uplift: the AEP uplift expected of the change, a fraction "
            "(e.g. 0.05), which sets how far its wake change reaches"
        )
        raise ValueError(msg)
    stray = sorted(set(map(str, block)) - set(_UPGRADE_KEYS))
    if stray:
        msg = f"upgrade has unknown keys {stray}; known keys are {list(_UPGRADE_KEYS)}"
        raise ValueError(msg)
    expected_uplift = _number(block["expected_uplift"], what="upgrade.expected_uplift")
    declared = block.get("wake_reach_d")
    settling = block.get("settling_time")
    if declared is not None:
        return expected_uplift, _reach(declared, what="upgrade.wake_reach_d"), settling
    return expected_uplift, reach_from_uplift(expected_uplift, offshore=offshore), settling


def _number(value: object, *, what: str) -> float:
    """Return a declared finite number, raising on anything else."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        msg = f"{what} must be a finite number, not {value!r}"
        raise ValueError(msg)
    return float(value)


def _reach(value: object, *, what: str) -> float:
    """Return a declared reach in rotor diameters, raising unless it is a finite number at least 0."""
    reach = _number(value, what=what)
    if reach < 0:
        msg = f"{what} must be at least 0 rotor diameters, not {reach}"
        raise ValueError(msg)
    return reach


def _check_known(named: dict[str, Any], *, coords: dict[str, tuple[float, float]], what: str) -> None:
    """Raise naming the turbines ``what`` names that the turbines file does not."""
    unknown = sorted(set(named) - set(coords))
    if unknown:
        msg = f"{what} names {unknown}, which the turbines file has no row for"
        raise ValueError(msg)


def _exclusions(entries: list) -> list[Exclusion]:
    """Read the declared exclusions; ``ALL`` is farm-wide."""
    exclusions: list[Exclusion] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(map(str, entry)) != set(_EXCLUSION_KEYS):
            msg = (
                f"an exclusion has exactly the keys {list(_EXCLUSION_KEYS)}, got {entry!r}; a period whose data "
                f"is right but whose state the labels miss is declared under operating_state_overrides"
            )
            raise ValueError(msg)
        turbine, start, end = _window_entry(entry, what="exclusion")
        exclusions.append((turbine, start, end))
    return exclusions


def _state_overrides(entries: list, *, states: set[str]) -> list[StateOverride]:
    """Read the declared operating state overrides; ``ALL`` is farm-wide.

    Raises on a state ``states`` does not define, and on two overrides overlapping on one turbine.
    """
    overrides: list[StateOverride] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(map(str, entry)) != set(_OVERRIDE_KEYS):
            msg = f"a operating state override has exactly the keys {list(_OVERRIDE_KEYS)}, got {entry!r}"
            raise ValueError(msg)
        state = str(entry["state"])
        if state not in states:
            msg = (
                f"the operating state override state {state!r} is not defined; use a generic state or a label declared "
                f"under operating_state.site_states: {sorted(states)}"
            )
            raise ValueError(msg)
        turbine, start, end = _window_entry(entry, what="operating state override")
        overrides.append((turbine, start, end, state))
    for i, (who, start, end, _) in enumerate(overrides):
        for other, other_start, other_end, _ in overrides[i + 1 :]:
            if (who is None or other is None or who == other) and start < other_end and other_start < end:
                names = ", ".join(ALL_TURBINES if w is None else w for w in dict.fromkeys([who, other]))
                msg = (
                    f"the operating state overrides of {names} overlap "
                    f"({start}..{end} and {other_start}..{other_end}); "
                    f"two overrides of one turbine may not cover the same record"
                )
                raise ValueError(msg)
    return overrides


def _window_entry(entry: dict, *, what: str) -> tuple[str | None, pd.Timestamp, pd.Timestamp]:
    """Read a declared ``{turbine, start, end}`` window, end exclusive; ``ALL`` is farm-wide."""
    turbine = str(entry["turbine"])
    start, end = _timestamp(entry["start"]), _timestamp(entry["end"])
    if end <= start:
        msg = f"the {what} of {turbine} ends at {end}, not after its start {start}"
        raise ValueError(msg)
    return None if turbine == ALL_TURBINES else turbine, start, end


def _span(block: dict, *, what: str) -> Window:
    """Read one ``{start, end}`` span, end exclusive."""
    start, end = _timestamp(block["start"]), _timestamp(block["end"])
    if end <= start:
        msg = f"{what} end {end} is not after start {start}; the campaign would cover no records"
        raise ValueError(msg)
    return start, end


def _analysis_period(
    block: dict | None, *, upgraded: list[str], what: str = "analysis_period"
) -> Window | dict[str, Window] | None:
    """Read the optional analysis period: one span, a span per upgraded turbine, or none."""
    if block is None:
        return None
    if set(block) == {"start", "end"}:
        return _span(block, what=what)
    stray = sorted(set(map(str, block)) - set(upgraded))
    if stray:
        msg = f"{what} names {stray}, which are not upgraded turbines; a per-turbine span is for those"
        raise ValueError(msg)
    return {str(t): _span(span, what=f"{what} of {t}") for t, span in block.items()}


def _resolved_period(period: Window | dict[str, Window] | None) -> dict[str, Any] | str:
    """Return the analysis period as the resolved echo shows it."""
    if period is None:
        return PER_TURBINE_PERIOD
    if isinstance(period, dict):
        return {t: {"start": str(s), "end": str(e)} for t, (s, e) in period.items()}
    start, end = period
    return {"start": str(start), "end": str(end)}


def operating_state_config(block: dict | None) -> OperatingStateConfig:
    """Read an operating-state block as a declaration writes it; none gives the generic states only."""
    if block is None:
        return OperatingStateConfig()
    stray = sorted(set(map(str, block)) - set(_OPERATING_STATE_KEYS))
    if stray:
        msg = f"operating_state has unknown keys {stray}; known keys are {list(_OPERATING_STATE_KEYS)}"
        raise ValueError(msg)
    states = _renamed(block, "site_states", where="operating_state") or {}
    labels = {str(name): _state_validity(str(name), entry) for name, entry in states.items()}
    label_column = _renamed(block, "site_state_column", where="operating_state")
    above, below = block.get("parked_pitch_above_deg"), block.get("parked_pitch_below_deg")
    return OperatingStateConfig(
        site_state_column=None if label_column is None else str(label_column),
        site_states=labels,
        parked_pitch_above_deg=None if above is None else float(above),
        parked_pitch_below_deg=None if below is None else float(below),
    )


def _timestamps(block: dict | None) -> TimestampConvention:
    """Read the optional ``data.timestamps`` block; none gives period start, UTC."""
    if block is None:
        return TimestampConvention()
    stray = sorted(set(map(str, block)) - set(_TIMESTAMP_KEYS))
    if stray:
        msg = f"data.timestamps has unknown keys {stray}; known keys are {list(_TIMESTAMP_KEYS)}"
        raise ValueError(msg)
    default = TimestampConvention()
    return TimestampConvention(
        convention=str(block.get("convention", default.convention)),  # type: ignore[arg-type]
        time_zone=str(block.get("time_zone", default.time_zone)),
    )


def _offshore(block: dict | None) -> bool:
    """Read the optional ``site`` block's offshore flag; onshore by default."""
    if block is None:
        return False
    stray = sorted(set(map(str, block)) - set(_SITE_KEYS))
    if stray:
        msg = f"site has unknown keys {stray}; known keys are {list(_SITE_KEYS)}"
        raise ValueError(msg)
    offshore = block.get("offshore", False)
    if not isinstance(offshore, bool):
        msg = f"site.offshore must be true or false, not {offshore!r}"
        raise ValueError(msg)  # noqa: TRY004 - a declaration error, reported like the others
    return offshore


def _state_validity(name: str, entry: dict) -> StateValidity:
    """Read one label's ``{northing, waking, uplift}`` validity."""
    if not isinstance(entry, dict) or set(entry) != set(_STATE_VALIDITY_KEYS):
        msg = f"operating_state site state {name!r} needs exactly the keys {list(_STATE_VALIDITY_KEYS)}, got {entry!r}"
        raise ValueError(msg)
    for key in ("northing", "uplift"):
        if entry[key] not in _VALIDITY:
            msg = f"operating_state site state {name!r} has {key} {entry[key]!r}; use {VALID!r} or {NOT_VALID!r}"
            raise ValueError(msg)
    return StateValidity(
        northing=_VALIDITY[entry["northing"]], waking=str(entry["waking"]), uplift=_VALIDITY[entry["uplift"]]
    )


def _resolved_operating_state(config: OperatingStateConfig) -> dict[str, Any]:
    """Return the operating-state block as the resolved echo shows it."""
    resolved: dict[str, Any] = {"site_state_column": config.site_state_column}
    if config.parked_pitch_below_deg is not None:
        resolved["parked_pitch_below_deg"] = config.parked_pitch_below_deg
    else:
        resolved["parked_pitch_above_deg"] = config.parked_pitch_above_deg
    resolved["site_states"] = {
        name: {
            "northing": VALID if v.northing else NOT_VALID,
            "waking": v.waking,
            "uplift": VALID if v.uplift else NOT_VALID,
        }
        for name, v in config.site_states.items()
    }
    return resolved


def _section(raw: dict, name: str) -> dict:
    """Return a required top-level mapping of the declaration."""
    if name not in raw:
        msg = f"the declaration has no {name!r} section"
        raise ValueError(msg)
    return dict(raw[name])


def _schema(name: str) -> ColumnSchema:
    """Return the named column schema, or raise listing the names there are."""
    if name not in SCHEMAS:
        msg = f"unknown data.schema {name!r}; known schemas are {sorted(SCHEMAS)}"
        raise ValueError(msg)
    return SCHEMAS[name]


def _resolve(root: Path, name: str, *, what: str) -> Path:
    """Resolve a declared path against the declaration's directory, checking it exists."""
    path = root / name
    if not path.exists():
        msg = f"the declaration's {what} file {name!r} is not at {path}"
        raise FileNotFoundError(msg)
    return path


def _read_turbines(path: Path) -> Layout:
    """Read the turbines sidecar -- name, latitude, longitude, rotor diameter -- into a layout.

    The header may be cased any way. ``rotor_diameter_m`` is required.

    Rows without a name are skipped, so a campaign-design layout can serve as the sidecar. A name
    given more than once is rejected.
    """
    frame = pd.read_csv(path)
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    missing = [c for c in ("name", "latitude", "longitude", "rotor_diameter_m") if c not in lookup]
    if missing:
        msg = f"the turbines file {path.name} has no {missing} column(s); it carries {list(frame.columns)}"
        raise ValueError(msg)
    named = frame[frame[lookup["name"]].notna() & (frame[lookup["name"]].astype(str).str.strip() != "")]
    names = named[lookup["name"]].astype(str)
    doubled = sorted(set(names[names.duplicated()]))
    if doubled:
        msg = f"the turbines file {path.name} names {doubled} more than once"
        raise ValueError(msg)
    for name in names:
        _path_component(name, what=f"turbine name in {path.name}")
    return Layout.from_frame(named.reset_index(drop=True))


def _path_component(value: str, *, what: str) -> str:
    """Check a name a run writes a directory for is one path component, and return it."""
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        msg = (
            f"the {what} {value!r} names an output directory, so it cannot be empty, '.', '..', "
            f"or carry a path separator"
        )
        raise ValueError(msg)
    return value


def _timing(block: dict, *, has_works: bool) -> pd.Timestamp | ToggleSchedule | None:
    """Build the campaign's timing from its tagged block; None when a works table gives it."""
    mode = str(block.get("mode", ""))
    if mode == "prepost":
        changeover = block.get("changeover")
        if changeover is None and not has_works:
            msg = "a prepost declaration needs timing.changeover or a works table (data.works)"
            raise ValueError(msg)
        return None if changeover is None else _timestamp(changeover)
    if mode == "toggle":
        if block.get("start") is None:
            # ToggleSchedule allows no start, taking the first timestamp as origin with no
            # baseline. A declaration must say instead, or a declared baseline silently toggles.
            msg = "a toggle declaration needs timing.start: when toggling began"
            raise ValueError(msg)
        period = pd.Timedelta(block["period"])
        if period <= pd.Timedelta(0):
            msg = f"timing.period must be positive, got {period}"
            raise ValueError(msg)
        return ToggleSchedule(
            period=period,
            start=_timestamp(block["start"]),
            start_on=bool(block.get("start_on", False)),
        )
    msg = f"unknown timing.mode {mode!r}; known modes are {list(MODES)}"
    raise ValueError(msg)


def _north_offsets(block: dict | None) -> list[tuple[str, pd.Timestamp, float]] | None:
    """Read the northing tri-state: discover, apply a declared table, or apply nothing."""
    if block is None or block.get("discover", True):
        return None
    return [(str(w), _timestamp(t), float(o)) for w, t, o in block.get("table", [])]


def _timestamp(value: object) -> pd.Timestamp:
    """Coerce a declared time to UTC: tz-aware is converted, naive is read as UTC."""
    stamp = pd.Timestamp(value)  # type: ignore[arg-type]
    return stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")


def era5_window(start: pd.Timestamp, end: pd.Timestamp, *, today: pd.Timestamp | None = None) -> tuple[str, str]:
    """Return the reanalysis fetch window, rounded out to whole calendar years and ending no later than today.

    The reanalysis cache is keyed by its arguments, so exact analysis windows would re-download
    for every campaign on the same site. ``end`` is exclusive, so the last year needed is the one
    holding the final instant the campaign can read. The end is clamped by :func:`clamp_to_today`.

    :param today: the date to clamp to, defaulting to today in UTC
    """
    last = end - pd.Timedelta(nanoseconds=1)
    return f"{start.year}-01-01", clamp_to_today(f"{last.year}-12-31", today=today)


def clamp_to_today(date: str, *, today: pd.Timestamp | None = None) -> str:
    """Return ``date`` (``YYYY-MM-DD``), or today in UTC if that is earlier.

    The reanalysis archive refuses an end date after today, so a campaign whose SCADA runs to the
    present fetches to today; the archive's last few days are not yet published and come back empty.

    :param today: the date to clamp to, defaulting to today in UTC
    """
    latest = (pd.Timestamp.now(tz="UTC") if today is None else today).strftime("%Y-%m-%d")
    return min(date, latest)
