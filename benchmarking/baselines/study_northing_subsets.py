"""Consistency of the northing method across subsets of a farm and of its record.

A method that reports different corrections depending on which turbines or which years it was
handed is not trustworthy, however good its answer on the full record.

This runs the shipped two-pass northing over a grid of turbine groups and time windows on real Hill of Towie data, then
reports where a subset finds a changepoint the full run does not.

The subsets are declared in :data:`WINDOWS` and :data:`GROUPS` so the exercise re-runs unchanged
after any change to the estimator, and two runs can be compared directly.

Run it::

    uv run python -m benchmarking.baselines.study_northing_subsets

The first run builds a compact input frame (one row per turbine per record, yaw + reanalysis
direction + power + availability) year by year and caches it, which takes some minutes and needs
the Hill of Towie SCADA downloaded. Later runs reuse the cache. Outputs land under
``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``northing_subsets``/.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

from benchmarking.baselines.hot_context import NORTHING_YAML, build_hot_v0_context
from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada
from wind_up.circular_math import circ_diff
from wind_up.northing import DEFAULT_NORTHING, north_farm, yaw_usable

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

TIMEBASE_S = 600.0
RATED_KW = 2300.0
YEARS = range(2016, 2025)

ALL = tuple(f"T{n:02d}" for n in range(1, 22))
WEST = tuple(f"T{n:02d}" for n in range(1, 16))
EAST = tuple(f"T{n:02d}" for n in range(16, 22))
GROUPS = {"all": ALL, "west": WEST, "east": EAST}


def _window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")


# Spanning three months to nine years, straddling the known recalibrations (spring/summer 2017)
# and the known farm outages (November 2019, June 2020), and placing those events at the start,
# middle and end of a window so an edge effect has somewhere to show.
WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "full_2016_2024": _window("2016-01-01", "2025-01-01"),
    **{f"year_{y}": _window(f"{y}-01-01", f"{y + 1}-01-01") for y in YEARS},
    **{f"2y_{y}": _window(f"{y}-01-01", f"{y + 2}-01-01") for y in range(2016, 2024)},
    **{f"3y_{y}": _window(f"{y}-01-01", f"{y + 3}-01-01") for y in (2016, 2018, 2020, 2022)},
    "half_2017H1": _window("2017-01-01", "2017-07-01"),
    "half_2017H2": _window("2017-07-01", "2018-01-01"),
    "half_2019H2": _window("2019-07-01", "2020-01-01"),
    "half_2020H1": _window("2020-01-01", "2020-07-01"),
    "q_2017Q2_recals": _window("2017-04-01", "2017-07-01"),
    "q_2017Q3": _window("2017-07-01", "2017-10-01"),
    "q_2019Q4_outage": _window("2019-10-01", "2020-01-01"),
    "q_2020Q2_outage": _window("2020-04-01", "2020-07-01"),
    "q_2018Q4_edge": _window("2018-10-01", "2019-01-01"),
    "edge_after_t16_recal": _window("2016-06-01", "2017-06-01"),
    "edge_after_t05_recal": _window("2017-06-01", "2018-05-01"),
}

REFERENCE_CASE = "all__full_2016_2024"


def default_output_root() -> Path:
    """Return the directory this driver writes under (``WIND_UP_BENCHMARKING_OUTPUT_DIR`` overrides)."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "northing_subsets"


def build_inputs(out_dir: Path, *, years: Sequence[int] = tuple(YEARS)) -> pd.DataFrame:
    """Build (or reuse) the compact northing-input frame, caching one parquet per year.

    Loading nine years of 21-turbine SCADA at once needs far more memory than the northing itself,
    so each year is reduced to the handful of columns the estimator reads and cached.
    """
    cache = out_dir / "years"
    cache.mkdir(parents=True, exist_ok=True)
    era5 = build_hot_v0_context(wtg_names=list(ALL)).reanalysis_datasets[0].data["wind_direction_100m"]
    frames = []
    for year in years:
        cached = cache / f"{year}.parquet"
        if cached.exists():
            frames.append(pd.read_parquet(cached))
            continue
        scada, _ = load_hot_scada(
            start_dt=pd.Timestamp(f"{year}-01-01", tz="UTC"),
            end_dt_excl=pd.Timestamp(f"{year + 1}-01-01", tz="UTC"),
            wtg_numbers=list(range(1, 22)),
            wtg_names=list(ALL),
        )
        index = pd.DatetimeIndex(scada.index)
        # the SCADA index repeats each timestamp once per turbine, so carry ERA5 onto the unique
        # timestamps and let the lookup broadcast it back
        unique = pd.DatetimeIndex(index.unique()).sort_values()
        wd = era5.reindex(era5.index.union(unique)).ffill(limit=6).reindex(unique).reindex(index)
        year_frame = pd.DataFrame(
            {
                "turbine": scada[HOT_COLUMNS.turbine].astype(str).to_numpy(),
                "timestamp": index,
                "yaw_deg": scada[HOT_COLUMNS.nacelle_position].to_numpy(np.float32),
                "power_kw": scada[HOT_COLUMNS.active_power].to_numpy(np.float32),
                "availability_s": scada[HOT_COLUMNS.availability].to_numpy(np.float32),
                "era5_wd_deg": wd.to_numpy(np.float32),
            }
        )
        year_frame.to_parquet(cached, index=False, compression="zstd")
        frames.append(year_frame)
        logger.info("built %d: %d rows", year, len(year_frame))
        del scada, year_frame
    frame = pd.concat(frames, ignore_index=True)
    frame["turbine"] = frame["turbine"].astype("category")
    return frame.sort_values("timestamp")


def run_case(frame: pd.DataFrame, turbines: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """North one (turbines, window) and return its changepoints as ``turbine``/``date``/``step_deg``."""
    rows = frame[(frame["turbine"].isin(turbines)) & (frame["timestamp"] >= start) & (frame["timestamp"] < end)]
    if rows.empty:
        return pd.DataFrame(columns=["turbine", "date", "step_deg"])
    index = pd.DatetimeIndex(sorted(rows["timestamp"].unique()))
    direction: dict[str, np.ndarray] = {}
    usable: dict[str, np.ndarray] = {}
    reanalysis: np.ndarray = np.full(len(index), np.nan)
    for turbine in sorted(rows["turbine"].unique()):
        one = rows[rows["turbine"] == turbine].drop_duplicates("timestamp").set_index("timestamp").reindex(index)
        wd = one["era5_wd_deg"].to_numpy(dtype=float)
        reanalysis = np.where(np.isfinite(reanalysis), reanalysis, wd)
        direction[str(turbine)] = one["yaw_deg"].to_numpy(dtype=float)
        usable[str(turbine)] = yaw_usable(
            power=one["power_kw"].to_numpy(dtype=float),
            downtime_s=TIMEBASE_S - np.nan_to_num(one["availability_s"].to_numpy(dtype=float), nan=0.0),
            reference_deg=wd,
            rated_power=RATED_KW,
            timebase_s=TIMEBASE_S,
        )
    if len(direction) < 3:  # noqa: PLR2004 - north_farm needs a farm
        return pd.DataFrame(columns=["turbine", "date", "step_deg"])
    tables = north_farm(
        index, direction_deg=direction, usable=usable, reanalysis_deg=reanalysis, settings=DEFAULT_NORTHING
    )
    found: list[dict[str, object]] = []
    for name, table in sorted(tables.items()):
        offsets = table["north_offset"].to_numpy(dtype=float)
        found.extend(
            {
                "turbine": name,
                "date": table["timestamp"].iloc[i],
                "step_deg": round(float(circ_diff(offsets[i], offsets[i - 1])), 2),
            }
            for i in range(1, len(table))
        )
    return pd.DataFrame(found, columns=["turbine", "date", "step_deg"])


def _match(expected: pd.DataFrame, found: pd.DataFrame, *, days: float = 5.0) -> int:
    """How many of ``expected`` have a counterpart in ``found`` within ``days``, matched one to one."""
    used: set = set()
    matched = 0
    for row in expected.itertuples():
        candidates = found[
            (found["turbine"] == row.turbine) & ((found["date"] - row.date).abs() <= pd.Timedelta(days=days))
        ]
        candidates = candidates[~candidates.index.isin(used)]
        if len(candidates):
            used.add((candidates["date"] - row.date).abs().idxmin())
            matched += 1
    return matched


def run_study(*, out_root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every subset and compare each with the full run over the same window.

    :return: ``(changepoints, consistency)`` -- one row per found changepoint, and one row per case
        with how many it found, matched, invented (``extra``) and lost (``missing``)
    """
    out_dir = out_root or default_output_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = build_inputs(out_dir)
    logger.info("frame: %d rows, %s..%s", len(frame), frame["timestamp"].min(), frame["timestamp"].max())

    cases = [
        {"case": f"{group}__{name}", "group": group, "window": name, "turbines": turbines, "span": span}
        for group, turbines in GROUPS.items()
        for name, span in WINDOWS.items()
    ]
    changepoints, summary = [], []
    for n, case in enumerate(cases, start=1):
        began = time.perf_counter()
        found = run_case(frame, case["turbines"], *case["span"])
        found.insert(0, "case", case["case"])
        changepoints.append(found)
        summary.append(
            {
                "case": case["case"],
                "group": case["group"],
                "window": case["window"],
                "start": case["span"][0].date(),
                "end": case["span"][1].date(),
                "n_found": len(found),
                "seconds": round(time.perf_counter() - began, 1),
            }
        )
        logger.info("[%3d/%d] %-32s %4d cp %6.1fs", n, len(cases), case["case"], len(found), summary[-1]["seconds"])

    found_all = pd.concat(changepoints, ignore_index=True)
    reference = found_all[found_all["case"] == REFERENCE_CASE]
    rows: list[dict[str, object]] = []
    for entry in summary:
        if entry["case"] == REFERENCE_CASE:
            continue
        case_found = found_all[found_all["case"] == entry["case"]]
        start, end = pd.Timestamp(entry["start"], tz="UTC"), pd.Timestamp(entry["end"], tz="UTC")
        expected = reference[
            (reference["date"] >= start)
            & (reference["date"] < end)
            & (reference["turbine"].isin(GROUPS[entry["group"]]))
        ]
        matched = _match(expected, case_found)
        rows.append(
            {
                **entry,
                "n_expected": len(expected),
                "matched": matched,
                "extra": len(case_found) - matched,
                "missing": len(expected) - matched,
            }
        )
    consistency = pd.DataFrame(rows).sort_values("extra", ascending=False)
    found_all.to_csv(out_dir / "changepoints.csv", index=False)
    consistency.to_csv(out_dir / "consistency.csv", index=False)
    return found_all, consistency


def v0_published_changepoints() -> pd.DataFrame:
    """Return v0's published Hill of Towie changepoints, to compare a run against."""
    published = pd.DataFrame(
        [(str(a), pd.Timestamp(b, tz="UTC"), float(c)) for a, b, c in yaml.safe_load(NORTHING_YAML.read_text())],
        columns=["turbine", "date", "north_offset"],
    ).sort_values(["turbine", "date"])
    rows: list[dict[str, object]] = []
    for turbine, group in published.groupby("turbine"):
        offsets = group["north_offset"].to_numpy()
        rows.extend(
            {
                "turbine": turbine,
                "date": group["date"].iloc[i],
                "step_deg": round(float(circ_diff(offsets[i], offsets[i - 1])), 2),
            }
            for i in range(1, len(group))
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    all_found, how_consistent = run_study()
    full = all_found[all_found["case"] == REFERENCE_CASE]
    v0 = v0_published_changepoints()
    print(f"\nfull run: {len(full)} changepoints; v0's published table: {len(v0)}")  # noqa: T201
    print(  # noqa: T201
        f"cases inventing changepoints the full run does not see: "
        f"{int((how_consistent['extra'] > 0).sum())}/{len(how_consistent)}; "
        f"total extra {int(how_consistent['extra'].sum())}, total missing {int(how_consistent['missing'].sum())}"
    )
    print(how_consistent.head(15).to_string(index=False))  # noqa: T201
