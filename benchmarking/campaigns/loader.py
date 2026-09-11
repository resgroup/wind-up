"""Read a campaign declaration: one YAML file of campaign facts, plus a turbines sidecar.

Campaign facts only. Method configuration stays on the method, the deliberate break from v0's
``WindUpConfig``, which mixes both into one file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from benchmarking.campaigns.declaration import CampaignSpec
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule

if TYPE_CHECKING:
    from benchmarking.synthetic import ColumnSchema

# The schemas a declaration may name. Inline schema definition is not supported: a source's
# column names belong to its adapter, not to a campaign.
SCHEMAS: dict[str, ColumnSchema] = {"hill_of_towie": HOT_COLUMNS}

MODES = ("prepost", "toggle")

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
        calendar years so campaigns on one site share a cache entry
    """

    name: str
    spec: CampaignSpec
    columns: ColumnSchema
    scada_path: Path
    centroid: tuple[float, float]
    era5_window: tuple[str, str]

    def resolved(self) -> dict[str, Any]:
        """Return the resolved campaign facts, for echoing into the run output.

        Timestamps appear as the UTC values the declaration actually resolved to, so a
        mis-declared timezone is visible rather than silent.
        """
        spec = self.spec
        start, end = spec.analysis_period
        timing: dict[str, Any] = {"mode": spec.mode}
        if isinstance(spec.upgrade_timing, ToggleSchedule):
            timing["start"] = str(spec.upgrade_timing.start)
            timing["period"] = str(spec.upgrade_timing.period)
            timing["start_on"] = spec.upgrade_timing.start_on
        else:
            timing["changeover"] = str(spec.upgrade_timing)
        return {
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
            "analysis_period": {"start": str(start), "end": str(end)},
            "northing": {
                "discover": spec.north_offsets is None,
                "table": [] if spec.north_offsets is None else [[w, str(t), o] for w, t, o in spec.north_offsets],
            },
            "reanalysis": {"centroid": list(self.centroid), "window": list(self.era5_window)},
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
    coords = _read_turbines(_resolve(root, str(data["turbines"]), what="turbines"))

    roles = _section(raw, "turbines")
    upgraded = [str(w) for w in roles.get("upgraded", [])]
    references = [str(w) for w in roles.get("references", [])]
    excluded = [str(w) for w in roles.get("excluded", [])]
    _check_roles(upgraded=upgraded, references=references, excluded=excluded, coords=coords)

    period = _section(raw, "analysis_period")
    start, end = _timestamp(period["start"]), _timestamp(period["end"])
    if end <= start:
        msg = f"analysis_period end {end} is not after start {start}; the campaign would cover no records"
        raise ValueError(msg)

    return Declaration(
        name=str(raw["name"]),
        spec=CampaignSpec(
            upgraded_turbines=upgraded,
            upgrade_timing=_timing(_section(raw, "timing")),
            candidate_references=references,
            excluded_turbines=excluded,
            coords={w: coords[w] for w in [*upgraded, *references, *excluded]},
            north_offsets=_north_offsets(raw.get("northing")),
            rated_power_kw=float(roles["rated_power_kw"]),
            analysis_period=(start, end),
            turbine_col=columns.turbine,
        ),
        columns=columns,
        scada_path=scada_path,
        centroid=_centroid(coords),
        era5_window=_era5_window(start, end),
    )


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


def _read_turbines(path: Path) -> dict[str, tuple[float, float]]:
    """Read the turbines sidecar: name, latitude, longitude, however the header is cased.

    Rows without a name are skipped, so a campaign-design layout can serve as the sidecar.
    """
    frame = pd.read_csv(path)
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    missing = [c for c in ("name", "latitude", "longitude") if c not in lookup]
    if missing:
        msg = f"the turbines file {path.name} has no {missing} column(s); it carries {list(frame.columns)}"
        raise ValueError(msg)
    named = frame[frame[lookup["name"]].notna() & (frame[lookup["name"]].astype(str).str.strip() != "")]
    return {
        str(row[lookup["name"]]): (float(row[lookup["latitude"]]), float(row[lookup["longitude"]]))
        for _, row in named.iterrows()
    }


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


def _timing(block: dict) -> pd.Timestamp | ToggleSchedule:
    """Build the campaign's timing from its tagged block."""
    mode = str(block.get("mode", ""))
    if mode == "prepost":
        return _timestamp(block["changeover"])
    if mode == "toggle":
        return ToggleSchedule(
            period=pd.Timedelta(block["period"]),
            start=_timestamp(block["start"]) if block.get("start") is not None else None,
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


def _centroid(coords: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """Return the site's mean latitude and longitude, rounded.

    Taken over every turbine in the file, not the declared roles: reanalysis is a model input, so
    a reference-set sensitivity run must not move it.
    """
    points = list(coords.values())
    return (
        round(sum(lat for lat, _ in points) / len(points), CENTROID_DECIMALS),
        round(sum(lon for _, lon in points) / len(points), CENTROID_DECIMALS),
    )


def _era5_window(start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, str]:
    """Return the reanalysis fetch window, rounded out to whole calendar years.

    The reanalysis cache is keyed by its arguments, so exact analysis windows would re-download
    for every campaign on the same site. ``end`` is exclusive, so the last year needed is the one
    holding the final instant the campaign can read.
    """
    last = end - pd.Timedelta(nanoseconds=1)
    return f"{start.year}-01-01", f"{last.year}-12-31"
