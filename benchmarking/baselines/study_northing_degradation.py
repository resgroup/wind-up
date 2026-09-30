"""Exploration study: how v1 northing degrades as its input data is impoverished.

Gathers evidence for R5 Stage 3 (graceful degradation, and the changepoints-v-reanalysis
changepoint floor). For each open farm it runs the full ``north_farm`` pipeline under a
matrix of input degradations and scores the result against that farm's recorded golden
table (the full-data answer):

- **few turbines** -- contiguous spatial subsets full -> ... -> 1, so a wake pair survives;
- **low data** -- the record truncated to the most recent 365 / 182 / 90 / 30 / 14 / 7 days;
- **few turbines x low data** -- the corner cases that must degrade, not crash;
- **gaps** -- random per-turbine missingness and a contiguous multi-month outage.

It also **sweeps the reanalysis changepoint floor** (``min_step_deg``) on Hill of Towie
turbines northed one at a time against ERA5, scored against the published table, to inform
the ``against_reanalysis`` step floor for changepoints-v-reanalysis.

Read-only on the repo and on the pipeline (it does not build or change changepoints-v-reanalysis). Every case is
wrapped so a crash is recorded as a finding rather than aborting the run; results stream to a
CSV and a human log under the output dir, flushed per case, so an interrupted run keeps its
evidence.

    .venv/bin/python -m benchmarking.baselines.study_northing_degradation
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import yaml

from benchmarking.baselines.study_wake_nadir_golden import (
    greenbyte_inputs,
    hot_inputs,
)
from benchmarking.synthetic.sources import greenbyte
from wind_up.circular_math import circ_diff
from wind_up.geodesy import local_east_north
from wind_up.layout import LATITUDE_COL, LONGITUDE_COL, NAME_COL, Layout
from wind_up.northing import (
    DEFAULT_NORTHING,
    NorthingSettings,
    against_reanalysis,
    estimate_north_table,
    north_farm,
)

logger = logging.getLogger("northing_degradation")

REPO = Path(__file__).resolve().parents[2]
NORTHING_DIR = REPO / "tests" / "test_data" / "hot" / "northing"
DAY = pd.Timedelta(days=1)
MATCH_TOL_DAYS = 14  # a recovered changepoint counts as a published one within this many days (ERA5 timing is coarse)


def output_root() -> Path:
    """Directory the study writes its CSV and log under (``WIND_UP_BENCHMARKING_OUTPUT_DIR`` overrides)."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "northing_degradation"


# ---------------------------------------------------------------------------
# reference tables (golden per farm; published for the floor sweep)
# ---------------------------------------------------------------------------
def load_north_table_yaml(path: Path) -> dict[str, pd.DataFrame]:
    """Load a ``[name, timestamp, offset]`` YAML into one sorted (timestamp, north_offset) table per turbine."""
    rows = yaml.safe_load(path.read_text())
    by_turbine: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for name, when, offset in rows:
        ts = pd.Timestamp(when)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        by_turbine.setdefault(str(name), []).append((ts, float(offset)))
    tables = {}
    for name, items in by_turbine.items():
        items.sort(key=lambda p: p[0])
        tables[name] = pd.DataFrame({"timestamp": [t for t, _ in items], "north_offset": [o for _, o in items]})
    return tables


def offset_at(table: pd.DataFrame, times: pd.DatetimeIndex) -> np.ndarray:
    """Evaluate a north table (a step function of time) at ``times``; before the first row uses the first offset."""
    ts = pd.DatetimeIndex(table["timestamp"]).tz_convert("UTC").tz_localize(None).to_numpy()
    off = table["north_offset"].to_numpy(dtype=float)
    q = pd.DatetimeIndex(times).tz_convert("UTC").tz_localize(None).to_numpy()
    idx = np.clip(np.searchsorted(ts, q, side="right") - 1, 0, len(off) - 1)
    return off[idx]


def table_distance(
    tables: dict[str, pd.DataFrame], golden: dict[str, pd.DataFrame], times: pd.DatetimeIndex
) -> dict[str, float]:
    """Absolute-offset error (deg) of ``tables`` vs ``golden`` sampled at ``times``.

    Per turbine present in both, the median |circ_diff| over ``times``; summarised across turbines
    as the median, 90th percentile and max of those per-turbine errors. Captures both a wrong level
    and a mistimed/missed changepoint in one number.
    """
    per_turbine = []
    for name, table in tables.items():
        if name not in golden:
            continue
        err = np.abs(circ_diff(offset_at(table, times), offset_at(golden[name], times)))
        per_turbine.append(float(np.nanmedian(err)))
    if not per_turbine:
        return {"n_scored": 0, "err_median": float("nan"), "err_p90": float("nan"), "err_max": float("nan")}
    arr = np.array(per_turbine)
    return {
        "n_scored": len(arr),
        "err_median": float(np.median(arr)),
        "err_p90": float(np.percentile(arr, 90)),
        "err_max": float(arr.max()),
    }


def month_grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return a monthly sampling grid across ``index`` (a handful of even points for a short window)."""
    min_grid_points = 4
    grid = pd.date_range(index.min().ceil("D"), index.max(), freq="MS", tz="UTC")
    if len(grid) >= min_grid_points:
        return grid
    return pd.DatetimeIndex(np.linspace(index.min().value, index.max().value, 6).astype("datetime64[ns]")).tz_localize(
        "UTC"
    )


# ---------------------------------------------------------------------------
# running one degraded case
# ---------------------------------------------------------------------------
def contiguous_order(layout: Layout) -> list[str]:
    """Order turbines so that every prefix is a geographically contiguous cluster.

    Region-grown by single linkage on the layout's WGS84-geodesic distances: seed from the
    westernmost turbine, then repeatedly append whichever remaining turbine is nearest to any
    turbine already in the cluster. So the first ``k`` names are always a connected spatial blob
    (nested as ``k`` grows), which keeps a wake pair alive for wake-nadir-shift even at small ``k`` -- unlike
    a plain west-to-east strip, which need not be compact on a two-dimensional layout.
    """
    east, _ = local_east_north(latitudes=layout.frame[LATITUDE_COL], longitudes=layout.frame[LONGITUDE_COL])
    names = list(layout.frame[NAME_COL])
    seed = names[int(np.argmin(east))]
    order = [seed]
    remaining = [n for n in names if n != seed]
    while remaining:
        nearest = min(
            remaining,
            key=lambda r: min(float(layout.distance_m[layout.index_of(r), layout.index_of(m)]) for m in order),
        )
        order.append(nearest)
        remaining.remove(nearest)
    return order


def sub_layout(layout: Layout, turbines: list[str]) -> Layout:
    """Return the layout restricted to ``turbines`` (a genuine small-farm layout, not the whole farm)."""
    frame = layout.frame[layout.frame[NAME_COL].isin(turbines)]
    return Layout.from_frame(frame.rename(columns={NAME_COL: "name"}))


def run_north(
    layout: Layout,
    index: pd.DatetimeIndex,
    direction: dict[str, np.ndarray],
    usable: dict[str, np.ndarray],
    reference: np.ndarray,
    power: dict[str, np.ndarray],
    wind_speed: dict[str, np.ndarray],
) -> dict[str, pd.DataFrame]:
    """Run the full pipeline (every step, the wake-nadir shift included) on already-degraded inputs."""
    return north_farm(
        index,
        direction_deg=direction,
        usable=usable,
        reanalysis_deg=reference,
        layout=layout,
        power=power,
        wind_speed=wind_speed,
    )


def slice_inputs(
    inputs: dict, turbines: list[str], mask: np.ndarray
) -> tuple[pd.DatetimeIndex, dict, dict, np.ndarray, dict, dict]:
    """Restrict the farm inputs to ``turbines`` and the rows selected by ``mask``."""
    index = inputs["index"][mask]

    def take(field: str) -> dict[str, np.ndarray]:
        return {t: np.asarray(inputs[field][t], dtype=float)[mask] for t in turbines}

    ref = np.asarray(inputs["reference"], dtype=float)[mask]
    return (
        index,
        take("direction"),
        {t: np.asarray(inputs["usable"][t], dtype=bool)[mask] for t in turbines},
        ref,
        take("power"),
        take("wind_speed"),
    )


# ---------------------------------------------------------------------------
# the degradation matrix
# ---------------------------------------------------------------------------
class Recorder:
    """Streams one row per case to a CSV (flushed each write) and echoes a summary to the log."""

    FIELDS: ClassVar[list[str]] = [
        "farm",
        "family",
        "label",
        "n_turbines",
        "n_days",
        "n_rows",
        "status",
        "n_scored",
        "err_median",
        "err_p90",
        "err_max",
        "seconds",
        "detail",
    ]

    def __init__(self, path: Path) -> None:
        """Open ``path`` for writing and emit the CSV header."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._fh.flush()

    def write(self, **row: object) -> None:
        """Write one case row to the CSV (flushed) and echo a one-line summary to the log."""
        self._writer.writerow({k: row.get(k, "") for k in self.FIELDS})
        self._fh.flush()
        logger.info(
            "  [%s/%s] %s: status=%s scored=%s err(med/p90/max)=%s/%s/%s (%ss) %s",
            row.get("farm"),
            row.get("family"),
            row.get("label"),
            row.get("status"),
            row.get("n_scored"),
            _fmt(row.get("err_median")),
            _fmt(row.get("err_p90")),
            _fmt(row.get("err_max")),
            row.get("seconds"),
            row.get("detail", ""),
        )

    def close(self) -> None:
        """Close the underlying CSV file."""
        self._fh.close()


def _fmt(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, float) and np.isfinite(value) else str(value)


def score_case(
    rec: Recorder,
    *,
    farm: str,
    family: str,
    label: str,
    layout: Layout,
    index: pd.DatetimeIndex,
    direction: dict[str, np.ndarray],
    usable: dict[str, np.ndarray],
    reference: np.ndarray,
    power: dict[str, np.ndarray],
    wind_speed: dict[str, np.ndarray],
    golden: dict[str, pd.DataFrame],
) -> None:
    """Run one degraded case, score it against the golden table, and record the outcome."""
    n_days = round((index.max() - index.min()) / DAY) if len(index) else 0
    started = time.monotonic()
    try:
        tables = run_north(layout, index, direction, usable, reference, power, wind_speed)
        dist = table_distance(tables, golden, month_grid(index))
        status, detail = "ok", ""
    except Exception as exc:  # noqa: BLE001 - a crash is the finding we are hunting for
        dist = {"n_scored": 0, "err_median": float("nan"), "err_p90": float("nan"), "err_max": float("nan")}
        status, detail = "ERROR", f"{type(exc).__name__}: {exc}"[:200]
        logger.warning("      %s raised: %s\n%s", label, detail, traceback.format_exc())
    rec.write(
        farm=farm,
        family=family,
        label=label,
        n_turbines=len(direction),
        n_days=n_days,
        n_rows=len(index),
        status=status,
        seconds=round(time.monotonic() - started, 1),
        detail=detail,
        **dist,
    )


def few_turbine_sizes(n: int) -> list[int]:
    """Contiguous-subset sizes from the full farm down to a lone turbine."""
    return [k for k in (n, 15, 10, 6, 3, 2, 1) if k <= n and k >= 1]


def window_lengths_days() -> list[int | None]:
    """Record truncations (most-recent N days); ``None`` is the full record."""
    return [None, 365, 182, 90, 30, 14, 7]


def mask_last_days(index: pd.DatetimeIndex, days: int | None) -> np.ndarray:
    """Boolean mask selecting the most recent ``days`` of the record (all rows when ``days is None``)."""
    if days is None:
        return np.ones(len(index), dtype=bool)
    cutoff = index.max() - pd.Timedelta(days=days)
    return np.asarray(index > cutoff)


def run_farm(rec: Recorder, *, farm: str, layout: Layout, inputs: dict, golden: dict[str, pd.DataFrame]) -> None:
    """Run the full degradation matrix for one farm."""
    ordered = contiguous_order(layout)
    n = len(ordered)
    logger.info(
        "%s: %d turbines, %d rows, window %s..%s",
        farm,
        n,
        len(inputs["index"]),
        inputs["index"].min().date(),
        inputs["index"].max().date(),
    )

    # few turbines (full window)
    full = np.ones(len(inputs["index"]), dtype=bool)
    for k in few_turbine_sizes(n):
        turbines = ordered[:k]
        index, direction, usable, reference, power, wind_speed = slice_inputs(inputs, turbines, full)
        score_case(
            rec,
            farm=farm,
            family="few_turbines",
            label=f"N={k}",
            layout=sub_layout(layout, turbines),
            index=index,
            direction=direction,
            usable=usable,
            reference=reference,
            power=power,
            wind_speed=wind_speed,
            golden=golden,
        )

    # low data (full farm)
    for days in window_lengths_days():
        mask = mask_last_days(inputs["index"], days)
        index, direction, usable, reference, power, wind_speed = slice_inputs(inputs, ordered, mask)
        score_case(
            rec,
            farm=farm,
            family="low_data",
            label=f"{days or 'all'}d",
            layout=layout,
            index=index,
            direction=direction,
            usable=usable,
            reference=reference,
            power=power,
            wind_speed=wind_speed,
            golden=golden,
        )

    # few turbines x low data corners (must degrade, not crash)
    for k, days in [(3, 90), (2, 30), (1, 14), (1, 7)]:
        if k > n:
            continue
        turbines = ordered[:k]
        mask = mask_last_days(inputs["index"], days)
        index, direction, usable, reference, power, wind_speed = slice_inputs(inputs, turbines, mask)
        score_case(
            rec,
            farm=farm,
            family="cross",
            label=f"N={k},{days}d",
            layout=sub_layout(layout, turbines),
            index=index,
            direction=direction,
            usable=usable,
            reference=reference,
            power=power,
            wind_speed=wind_speed,
            golden=golden,
        )

    # gaps (full farm & window, rows knocked out)
    run_gap_cases(rec, farm=farm, layout=layout, inputs=inputs, golden=golden, turbines=ordered)


def run_gap_cases(
    rec: Recorder, *, farm: str, layout: Layout, inputs: dict, golden: dict[str, pd.DataFrame], turbines: list[str]
) -> None:
    """Missing-data cases: random per-turbine dropout, and a contiguous multi-month outage."""
    full = np.ones(len(inputs["index"]), dtype=bool)
    index, direction0, usable0, reference, power0, wind_speed0 = slice_inputs(inputs, turbines, full)
    rng = np.random.default_rng(0)

    for frac in (0.5, 0.75):
        direction, usable, power, wind_speed = _drop_random(direction0, usable0, power0, wind_speed0, frac, rng)
        score_case(
            rec,
            farm=farm,
            family="gaps",
            label=f"drop{int(frac * 100)}%",
            layout=layout,
            index=index,
            direction=direction,
            usable=usable,
            reference=reference,
            power=power,
            wind_speed=wind_speed,
            golden=golden,
        )

    # a contiguous ~6-month blackout in the middle of the record
    lo, hi = index.min() + pd.Timedelta(days=180), index.min() + pd.Timedelta(days=360)
    black = np.asarray((index >= lo) & (index < hi))
    direction, usable, power, wind_speed = _blackout(direction0, usable0, power0, wind_speed0, black)
    score_case(
        rec,
        farm=farm,
        family="gaps",
        label="6mo_outage",
        layout=layout,
        index=index,
        direction=direction,
        usable=usable,
        reference=reference,
        power=power,
        wind_speed=wind_speed,
        golden=golden,
    )


def _drop_random(
    direction: dict, usable: dict, power: dict, wind_speed: dict, frac: float, rng: np.random.Generator
) -> tuple[dict, dict, dict, dict]:
    """Independently mark a fraction of each turbine's rows missing (NaN signals, unusable)."""
    d, u, p, w = {}, {}, {}, {}
    for t, series in direction.items():
        drop = rng.random(len(series)) < frac
        d[t] = np.where(drop, np.nan, series)
        p[t] = np.where(drop, np.nan, power[t])
        w[t] = np.where(drop, np.nan, wind_speed[t])
        u[t] = usable[t] & ~drop
    return d, u, p, w


def _blackout(
    direction: dict, usable: dict, power: dict, wind_speed: dict, black: np.ndarray
) -> tuple[dict, dict, dict, dict]:
    """Mark the same contiguous block of rows missing for every turbine."""
    d, u, p, w = {}, {}, {}, {}
    for t, series in direction.items():
        d[t] = np.where(black, np.nan, series)
        p[t] = np.where(black, np.nan, power[t])
        w[t] = np.where(black, np.nan, wind_speed[t])
        u[t] = usable[t] & ~black
    return d, u, p, w


# ---------------------------------------------------------------------------
# reanalysis floor sweep (changepoints-v-reanalysis evidence, Hill of Towie)
# ---------------------------------------------------------------------------
def published_changepoints() -> dict[str, list[pd.Timestamp]]:
    """Per-turbine changepoint timestamps from the published HoT table (rows after each turbine's first)."""
    tables = load_north_table_yaml(NORTHING_DIR / "optimized_northing_corrections.yaml")
    return {name: list(pd.DatetimeIndex(tbl["timestamp"])[1:]) for name, tbl in tables.items()}


def floor_sweep(rec: Recorder, *, inputs: dict, floors: list[float]) -> None:
    """Sweep ``min_step_deg`` per HoT turbine northed against ERA5 alone, scored vs the published table.

    For each floor, each turbine is northed one at a time (as changepoints-v-reanalysis does, via
    ``against_reanalysis``)
    and its recovered changepoints are matched to the published table's within ``MATCH_TOL_DAYS`` --
    so the log shows recall (published steps found) against spurious extras as the floor is lowered.
    """
    published = published_changepoints()
    index = inputs["index"]
    tol = pd.Timedelta(days=MATCH_TOL_DAYS)
    # Each floor is min_step-only (the diagnostic showing floor alone cannot tame ERA5); "tuned" is
    # the shipped changepoints-v-reanalysis config (against_reanalysis: min_step + min_segment), the row that matters.
    configs: list[tuple[str, NorthingSettings]] = [
        (f"floor={floor:g}", replace(DEFAULT_NORTHING, min_step_deg=floor)) for floor in floors
    ]
    configs.append(("tuned", against_reanalysis(DEFAULT_NORTHING)))
    for name, settings in configs:
        totals = {"published": 0, "recovered": 0, "matched": 0, "spurious": 0}
        for turbine in inputs["direction"]:
            started = time.monotonic()
            try:
                table = estimate_north_table(
                    index,
                    inputs["direction"][turbine],
                    reference_deg=inputs["reference"],
                    usable=inputs["usable"][turbine],
                    settings=settings,
                )
                found = list(pd.DatetimeIndex(table["timestamp"])[1:])
                truth = published.get(turbine, [])
                matched = sum(any(abs(f - c) <= tol for f in found) for c in truth)
                spurious = sum(not any(abs(f - c) <= tol for c in truth) for f in found)
                for key, val in (
                    ("published", len(truth)),
                    ("recovered", len(found)),
                    ("matched", matched),
                    ("spurious", spurious),
                ):
                    totals[key] += val
                rec.write(
                    farm="hill_of_towie",
                    family="floor_sweep",
                    label=f"{name},{turbine}",
                    n_turbines=1,
                    n_days=round((index.max() - index.min()) / DAY),
                    n_rows=len(index),
                    status="ok",
                    n_scored=len(truth),
                    err_median=len(found),
                    err_p90=matched,
                    err_max=spurious,
                    seconds=round(time.monotonic() - started, 1),
                    detail=f"published={len(truth)} found={len(found)} matched={matched} spurious={spurious}",
                )
            except Exception as exc:  # noqa: BLE001
                rec.write(
                    farm="hill_of_towie",
                    family="floor_sweep",
                    label=f"{name},{turbine}",
                    n_turbines=1,
                    n_rows=len(index),
                    status="ERROR",
                    seconds=round(time.monotonic() - started, 1),
                    detail=f"{type(exc).__name__}: {exc}"[:200],
                )
        logger.info(
            "%s: recall %d/%d published steps, %d spurious (%d recovered total)",
            name,
            totals["matched"],
            totals["published"],
            totals["spurious"],
            totals["recovered"],
        )


# ---------------------------------------------------------------------------
# digestible summary
# ---------------------------------------------------------------------------
def summarise(csv_path: Path) -> None:
    """Write a compact per-family summary of the results CSV (worst-case error per farm; floor curve)."""
    frame = pd.read_csv(csv_path)
    degradation = frame[frame["family"] != "floor_sweep"].copy()
    farms = sorted(degradation["farm"].unique())
    out: list[str] = ["", "=" * 78]
    out.append("DEGRADATION SUMMARY  --  worst-turbine offset error vs golden (deg), by farm")
    out.append("(a farm absent from a row has fewer turbines than that subset needs)")
    out.append("=" * 78)
    header = f"{'family':13s} {'level':11s} " + " ".join(f"{f[:11]:>11s}" for f in farms)
    out.append(header)
    out.append("-" * len(header))
    order = {"few_turbines": 0, "low_data": 1, "cross": 2, "gaps": 3}
    for (family, label), rows in sorted(
        degradation.groupby(["family", "label"]), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1])
    ):
        cells = [
            (f"{hit['err_max'].max():11.2f}" if len(hit := rows[rows["farm"] == farm]) else f"{'-':>11s}")
            for farm in farms
        ]
        out.append(f"{family:13s} {str(label)[:11]:11s} " + " ".join(cells))

    crashes = int((frame["status"] != "ok").sum())
    out.append(f"\nhard failures (crashes / NaN tables) across all cases: {crashes}")

    floor = frame[frame["family"] == "floor_sweep"].copy()
    if len(floor):
        floor["config"] = floor["label"].str.rsplit(",", n=1).str[0]

        def config_key(name: str) -> tuple[int, float]:
            return (1, 0.0) if name == "tuned" else (0, float(name.removeprefix("floor=")))

        out.extend(
            [
                "",
                "=" * 78,
                f"CHANGEPOINTS-V-REANALYSIS CONFIG (Hill of Towie, published-step match +/-{MATCH_TOL_DAYS}d)",
                "floor=* is min_step-only (diagnostic); "
                "'tuned' is the shipped changepoints-v-reanalysis (min_step + min_segment)",
                "=" * 78,
                f"{'config':>16s} {'recall':>12s} {'spurious':>10s} {'found':>7s}",
            ]
        )
        for name in sorted(floor["config"].unique(), key=config_key):
            rows = floor[floor["config"] == name]
            published = int(rows["n_scored"].sum())
            matched = int(rows["err_p90"].sum())  # floor_sweep stashes matched in err_p90
            spurious = int(rows["err_max"].sum())  # and spurious in err_max
            found = int(rows["err_median"].sum())  # and recovered count in err_median
            out.append(f"{name:>16s} {f'{matched}/{published}':>12s} {spurious:>10d} {found:>7d}")
    sys.stdout.write("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the whole degradation matrix across the three open farms and the HoT floor sweep."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout, force=True)
    root = output_root()
    rec = Recorder(root / "degradation_results.csv")
    logger.info("writing results under %s", root)
    started = time.monotonic()

    farms = [
        ("hill_of_towie", lambda: hot_inputs(start="2016-01-01", end="2021-01-01")),
        (
            "kelmarsh",
            lambda: greenbyte_inputs(greenbyte.KELMARSH, rotor_diameter_m=92.0, start="2017-01-01", end="2019-01-01"),
        ),
        (
            "penmanshiel",
            lambda: greenbyte_inputs(
                greenbyte.PENMANSHIEL, rotor_diameter_m=82.0, start="2017-01-01", end="2019-01-01"
            ),
        ),
    ]
    hot_cache: dict = {}
    for farm, build in farms:
        logger.info("=== %s: loading inputs ===", farm)
        try:
            layout, inputs = build()
        except Exception:
            logger.exception("could not load %s; skipping", farm)
            continue
        if farm == "hill_of_towie":
            hot_cache = {"layout": layout, "inputs": inputs}
        golden = load_north_table_yaml(NORTHING_DIR / f"golden_northing_corrections_{farm}.yaml")
        run_farm(rec, farm=farm, layout=layout, inputs=inputs, golden=golden)

    if hot_cache:
        logger.info("=== reanalysis floor sweep (Hill of Towie, changepoints-v-reanalysis evidence) ===")
        logger.info("current reanalysis step floor = %.1f", against_reanalysis(DEFAULT_NORTHING).min_step_deg)
        floor_sweep(rec, inputs=hot_cache["inputs"], floors=[5.0, 7.0, 8.0, 10.0, 12.0, 15.0])

    rec.close()
    logger.info("done in %.0f s; results at %s", time.monotonic() - started, root / "degradation_results.csv")
    summarise(root / "degradation_results.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summarise(output_root() / "degradation_results.csv")
    else:
        main()
