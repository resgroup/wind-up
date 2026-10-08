"""Read a campaign declaration: one YAML file of campaign facts, plus a turbines sidecar.

Campaign facts only. Method configuration stays on the method, the deliberate break from v0's
``WindUpConfig``, which mixes both into one file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from benchmarking.campaigns.declaration import CampaignSpec, layout_coords
from benchmarking.harness.operating_state import OperatingStateConfig, StateValidity
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule
from benchmarking.synthetic.sources.greenbyte import GREENBYTE_COLUMNS
from wind_up.layout import Layout

if TYPE_CHECKING:
    from benchmarking.campaigns.declaration import Exclusion, Window
    from benchmarking.synthetic import ColumnSchema

# The schemas a declaration may name. Inline schema definition is not supported: a source's
# column names belong to its adapter, not to a campaign.
SCHEMAS: dict[str, ColumnSchema] = {"hill_of_towie": HOT_COLUMNS, "greenbyte": GREENBYTE_COLUMNS}

# How a declaration names a farm-wide exclusion.
ALL_TURBINES = "ALL"
PER_TURBINE_PERIOD = "chosen per upgraded turbine"

MODES = ("prepost", "toggle")

# How a declaration writes a state's northing and uplift validity.
VALID = "valid"
NOT_VALID = "not valid"
_VALIDITY = {VALID: True, NOT_VALID: False}
_OPERATING_STATE_KEYS = ("label_column", "parked_pitch_above_deg", "parked_pitch_below_deg", "labels")
_STATE_VALIDITY_KEYS = ("northing", "waking", "uplift")

# The reanalysis centroid is rounded to this many decimals, and taken over the whole turbines
# file rather than the declared roles, so every campaign on one site shares a cache entry and
# changing the reference set does not move a model input.
CENTROID_DECIMALS = 2


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
    :param spec: the public campaign facts, the only thing a method sees
    :param columns: the source-native schema the SCADA is keyed by
    :param scada_path: the SCADA parquet, resolved relative to the declaration
    :param centroid: the site's ``(latitude, longitude)`` centroid over the whole turbines file,
        rounded, which reanalysis is self-served from
    :param era5_window: ``(start_date, end_date)`` for the reanalysis fetch, rounded out to whole
        calendar years so campaigns on one site share a cache entry. ``None`` when no analysis
        period is declared; the run then takes it from the data.
    :param operating_state: the site's operating-state labels and parked-pitch rule; generic states
        only when none is declared
    """

    name: str
    spec: CampaignSpec
    columns: ColumnSchema
    scada_path: Path
    centroid: tuple[float, float]
    era5_window: tuple[str, str] | None
    operating_state: OperatingStateConfig = field(default_factory=OperatingStateConfig)

    def resolved(self) -> dict[str, Any]:
        """Return the resolved campaign facts, for echoing into the run output.

        Timestamps appear as the UTC values the declaration actually resolved to, so a
        mis-declared timezone is visible rather than silent.
        """
        spec = self.spec
        timing: dict[str, Any] = {"mode": spec.mode}
        if isinstance(spec.upgrade_timing, ToggleSchedule):
            timing["start"] = str(spec.upgrade_timing.start)
            timing["period"] = str(spec.upgrade_timing.period)
            timing["start_on"] = spec.upgrade_timing.start_on
        elif spec.upgrade_timing is None:
            timing["changeover"] = {t: str(spec.timing_for(t)) for t in spec.upgraded_turbines}
        else:
            timing["changeover"] = str(spec.upgrade_timing)
        resolved: dict[str, Any] = {
            "name": self.name,
            "scada": str(self.scada_path),
            "schema": {v: k for k, v in SCHEMAS.items()}.get(self.columns, "custom"),
            "turbines": {
                "upgraded": list(spec.upgraded_turbines),
                "references": list(spec.candidate_references),
                "excluded": list(spec.excluded_turbines),
                "rated_power_kw": spec.rated_power_kw,
            },
            "timing": timing,
            "analysis_period": _resolved_period(spec.analysis_period),
            "northing": {
                "discover": spec.north_offsets is None,
                "table": [] if spec.north_offsets is None else [[w, str(t), o] for w, t, o in spec.north_offsets],
            },
            "reanalysis": {
                "centroid": list(self.centroid),
                "window": list(self.era5_window) if self.era5_window is not None else "from the data",
            },
            "operating_state": _resolved_operating_state(self.operating_state),
        }
        if spec.works:
            resolved["works"] = {t: [[str(s), str(e)] for s, e in windows] for t, windows in spec.works.items()}
        if spec.exclusions:
            resolved["exclusions"] = [
                {"turbine": ALL_TURBINES if who is None else who, "start": str(s), "end": str(e)}
                for who, s, e in spec.exclusions
            ]
        return resolved


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
    upgraded = [str(w) for w in roles.get("upgraded", [])]
    # Both default to the obvious campaign: nothing excluded, and every other turbine in the
    # turbines file offered as a reference. Declaring either narrows it.
    excluded = [str(w) for w in roles.get("excluded") or []]
    declared_references = roles.get("references")
    references = (
        [str(w) for w in declared_references]
        if declared_references
        else [w for w in coords if w not in set(upgraded) | set(excluded)]
    )
    _check_roles(upgraded=upgraded, references=references, excluded=excluded, coords=coords)

    works = read_works(_resolve(root, str(data["works"]), what="works")) if data.get("works") else {}
    _check_known(works, coords=coords, what="the works table")
    exclusions = _exclusions(raw.get("exclusions") or [])
    _check_known({w: None for w, _, _ in exclusions if w is not None}, coords=coords, what="the exclusions")
    timing = _timing(_section(raw, "timing"), has_works=bool(works))
    period = _analysis_period(raw.get("analysis_period"), upgraded=upgraded)

    spec = CampaignSpec(
        upgraded_turbines=upgraded,
        upgrade_timing=timing,
        candidate_references=references,
        excluded_turbines=excluded,
        layout=layout,
        north_offsets=_north_offsets(raw.get("northing")),
        rated_power_kw=float(roles["rated_power_kw"]),
        analysis_period=period,
        turbine_col=columns.turbine,
        works=works,
        exclusions=exclusions,
        references_declared=bool(declared_references),
    )
    bounds = spec.period_bounds()
    return Declaration(
        name=_path_component(str(raw["name"]), what="campaign name"),
        spec=spec,
        columns=columns,
        scada_path=scada_path,
        centroid=centroid(coords),
        era5_window=era5_window(*bounds) if bounds is not None else None,
        operating_state=_operating_state(raw.get("operating_state")),
    )


def read_works(path: Path) -> dict[str, list[Window]]:
    """Read a works table: ``Turbine`` plus the first ``First date...`` and ``Last date...`` columns.

    Each row is one window of whole days, ``[first 00:00, last + 1 day 00:00)`` UTC. A turbine may
    have several rows.
    """
    frame = pd.read_csv(path)
    first = next((c for c in frame.columns if str(c).startswith("First date")), None)
    last = next((c for c in frame.columns if str(c).startswith("Last date")), None)
    if "Turbine" not in frame.columns or first is None or last is None:
        msg = (
            f"the works table {path.name} needs a Turbine column and columns starting 'First date' and "
            f"'Last date'; it has {list(frame.columns)}"
        )
        raise ValueError(msg)
    works: dict[str, list[Window]] = {}
    for turbine, first_day, last_day in zip(frame["Turbine"].astype(str), frame[first], frame[last], strict=True):
        start = _timestamp(pd.Timestamp(first_day).normalize())
        end = _timestamp(pd.Timestamp(last_day).normalize()) + pd.Timedelta(days=1)
        if end <= start:
            msg = f"the works table {path.name} gives {turbine} a last date before its first"
            raise ValueError(msg)
        works.setdefault(turbine, []).append((start, end))
    return {turbine: sorted(windows) for turbine, windows in works.items()}


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
        turbine = str(entry["turbine"])
        start, end = _timestamp(entry["start"]), _timestamp(entry["end"])
        if end <= start:
            msg = f"the exclusion of {turbine} ends at {end}, not after its start {start}"
            raise ValueError(msg)
        exclusions.append((None if turbine == ALL_TURBINES else turbine, start, end))
    return exclusions


def _span(block: dict, *, what: str) -> Window:
    """Read one ``{start, end}`` span, end exclusive."""
    start, end = _timestamp(block["start"]), _timestamp(block["end"])
    if end <= start:
        msg = f"{what} end {end} is not after start {start}; the campaign would cover no records"
        raise ValueError(msg)
    return start, end


def _analysis_period(block: dict | None, *, upgraded: list[str]) -> Window | dict[str, Window] | None:
    """Read the optional analysis period: one span, a span per upgraded turbine, or none."""
    if block is None:
        return None
    if set(block) == {"start", "end"}:
        return _span(block, what="analysis_period")
    stray = sorted(set(map(str, block)) - set(upgraded))
    if stray:
        msg = f"analysis_period names {stray}, which are not upgraded turbines; a per-turbine span is for those"
        raise ValueError(msg)
    return {str(t): _span(span, what=f"analysis_period of {t}") for t, span in block.items()}


def _resolved_period(period: Window | dict[str, Window] | None) -> dict[str, Any] | str:
    """Return the analysis period as the resolved echo shows it."""
    if period is None:
        return PER_TURBINE_PERIOD
    if isinstance(period, dict):
        return {t: {"start": str(s), "end": str(e)} for t, (s, e) in period.items()}
    start, end = period
    return {"start": str(start), "end": str(end)}


def _operating_state(block: dict | None) -> OperatingStateConfig:
    """Read the optional operating-state block; none gives the generic states only."""
    if block is None:
        return OperatingStateConfig()
    stray = sorted(set(map(str, block)) - set(_OPERATING_STATE_KEYS))
    if stray:
        msg = f"operating_state has unknown keys {stray}; known keys are {list(_OPERATING_STATE_KEYS)}"
        raise ValueError(msg)
    labels = {str(name): _state_validity(str(name), entry) for name, entry in (block.get("labels") or {}).items()}
    label_column = block.get("label_column")
    above, below = block.get("parked_pitch_above_deg"), block.get("parked_pitch_below_deg")
    return OperatingStateConfig(
        label_column=None if label_column is None else str(label_column),
        labels=labels,
        parked_pitch_above_deg=None if above is None else float(above),
        parked_pitch_below_deg=None if below is None else float(below),
    )


def _state_validity(name: str, entry: dict) -> StateValidity:
    """Read one label's ``{northing, waking, uplift}`` validity."""
    if not isinstance(entry, dict) or set(entry) != set(_STATE_VALIDITY_KEYS):
        msg = f"operating_state label {name!r} needs exactly the keys {list(_STATE_VALIDITY_KEYS)}, got {entry!r}"
        raise ValueError(msg)
    for key in ("northing", "uplift"):
        if entry[key] not in _VALIDITY:
            msg = f"operating_state label {name!r} has {key} {entry[key]!r}; use {VALID!r} or {NOT_VALID!r}"
            raise ValueError(msg)
    return StateValidity(
        northing=_VALIDITY[entry["northing"]], waking=str(entry["waking"]), uplift=_VALIDITY[entry["uplift"]]
    )


def _resolved_operating_state(config: OperatingStateConfig) -> dict[str, Any]:
    """Return the operating-state block as the resolved echo shows it."""
    resolved: dict[str, Any] = {"label_column": config.label_column}
    if config.parked_pitch_below_deg is not None:
        resolved["parked_pitch_below_deg"] = config.parked_pitch_below_deg
    else:
        resolved["parked_pitch_above_deg"] = config.parked_pitch_above_deg
    resolved["labels"] = {
        name: {
            "northing": VALID if v.northing else NOT_VALID,
            "waking": v.waking,
            "uplift": VALID if v.uplift else NOT_VALID,
        }
        for name, v in config.labels.items()
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


def _check_roles(
    *, upgraded: list[str], references: list[str], excluded: list[str], coords: dict[str, tuple[float, float]]
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


def _timing(block: dict, *, has_works: bool) -> pd.Timestamp | ToggleSchedule | None:
    """Build the campaign's timing from its tagged block; None when a works table gives it."""
    mode = str(block.get("mode", ""))
    if mode == "prepost":
        changeover = block.get("changeover")
        if changeover is not None and has_works:
            msg = "declare either timing.changeover or a works table (data.works), not both"
            raise ValueError(msg)
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


def centroid(coords: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """Return the site's mean latitude and longitude, rounded.

    Taken over every turbine in the file, not the declared roles: reanalysis is a model input, so
    a reference-set sensitivity run must not move it.
    """
    points = list(coords.values())
    return (
        round(sum(lat for lat, _ in points) / len(points), CENTROID_DECIMALS),
        round(sum(lon for _, lon in points) / len(points), CENTROID_DECIMALS),
    )


def era5_window(start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, str]:
    """Return the reanalysis fetch window, rounded out to whole calendar years.

    The reanalysis cache is keyed by its arguments, so exact analysis windows would re-download
    for every campaign on the same site. ``end`` is exclusive, so the last year needed is the one
    holding the final instant the campaign can read.
    """
    last = end - pd.Timedelta(nanoseconds=1)
    return f"{start.year}-01-01", f"{last.year}-12-31"
