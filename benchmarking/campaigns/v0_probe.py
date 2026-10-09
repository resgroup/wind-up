"""The v0 probe: wind-up v0 and the power model side by side on the Hill of Towie placebo arm.

CF26 and the level probe put the power model's placebo level at +0.1 to +1 pp on contiguous
prepost periods, and the first like-for-like reading (T05, four nearest references) had v0 about
1 pp below the power model on every pair. This probe runs that comparison for the whole farm.

Each test turbine gets its own small campaign: the turbine plus its ``n`` nearest neighbours as
the only candidate references, on the whole-farm reference arm's window (12 months of baseline
into a 6-month campaign changing over 2018-09-01, nothing injected, so every reading has a truth
of 0). Both methods see exactly the same rows. The power model runs with its reference screen off
and its per-reference readings on; v0 runs as :class:`~benchmarking.baselines.v0_binned.V0BinnedMethod`
wraps it (season-matched pre period, one-year detrend, long-term distribution off), and its
per-pair results are harvested so each reference's reading against the test turbine, the
power-only variant and the reversed estimate come out beside the power model's.

Run it::

    uv run python -m benchmarking.campaigns.v0_probe hot [--turbines T05 T16 ...] [--n-refs 4] [--smoke]
    uv run python -m benchmarking.campaigns.v0_probe hot --resume <run_dir>     # finish a stopped run
    uv run python -m benchmarking.campaigns.v0_probe summarise <run_dir>

Settings come from the environment or the repository's ``.env`` (see :mod:`benchmarking.env`).
A turbine that fails is logged and the next one still runs; ``--resume`` skips the turbines a run
has already written, so a run stopped by a time limit picks up where it left off.

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``v0_probe``/``hot_<timestamp>/``:
``estimates.csv`` (one row per test turbine and method, appended as each turbine finishes),
``pairs.csv`` (every per-reference reading of either method), ``summary.csv`` (the two methods
side by side with their difference), ``probe.json`` and ``probe.log``; each turbine's method
output goes under ``<turbine>/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.campaigns.composed import output_root
from benchmarking.campaigns.level_probe import PROBE_JSON, PROBE_LOG, _log_to
from benchmarking.env import load_env
from wind_up.geodesy import distance_and_bearing

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

OUTPUT_DIRNAME = "v0_probe"
CHANGEOVER = pd.Timestamp("2018-09-01", tz="UTC")
BASELINE_MONTHS = 12
CAMPAIGN_MONTHS = 6
DEFAULT_N_REFS = 4
# v0's pre period is the campaign shifted back a year, so a smoke run keeps the full baseline and shortens the campaign.
SMOKE_TURBINES = ("T05",)
SMOKE_N_REFS = 2
SMOKE_CAMPAIGN_MONTHS = 1
POWER_MODEL = "power_model"
V0 = "v0_binned"
ESTIMATES_CSV = "estimates.csv"
PAIRS_CSV = "pairs.csv"
SUMMARY_CSV = "summary.csv"
ESTIMATE_COLUMNS = ("test", "method", "estimate_pp", "references")
PAIR_COLUMNS = ("test", "read_turbine", "against", "method", "variant", "reading_pp", "unc_one_sigma_pp")
# v0's per-pair columns and the variant each becomes.
_V0_VARIANTS = {"uplift_frc": "pair", "poweronly_uplift_frc": "pair_power_only", "reversed_uplift_frc": "pair_reversed"}
_V0_FINAL_GLOB = "*_results_per_test_ref_*.csv"
_V0_INTERIM = "results_interim.csv"


# --- pure parts ------------------------------------------------------------------------------------


def nearest_references(coords: Mapping[str, tuple[float, float]], turbine: str, *, n: int) -> list[str]:
    """Return the ``n`` turbines nearest to ``turbine`` by great-circle distance, nearest first."""
    distances = {
        other: distance_and_bearing((coords[turbine][0], coords[turbine][1]), (position[0], position[1]))[0]
        for other, position in coords.items()
        if other != turbine
    }
    return sorted(distances, key=distances.__getitem__)[:n]


def neighbourhoods(
    turbines: Sequence[str], *, coords: Mapping[str, tuple[float, float]], n_refs: int
) -> dict[str, list[str]]:
    """Return each turbine's references: its ``n_refs`` nearest neighbours."""
    return {t: nearest_references(coords, t, n=n_refs) for t in turbines}


def v0_pair_readings(scratch_dir: Path) -> pd.DataFrame:
    """Harvest v0's per-pair results from its scratch directory, in pp, without a reference read against itself.

    The final ``*_results_per_test_ref_*.csv`` is preferred; a run that was stopped leaves only
    ``results_interim.csv``, which is read instead. One row per pair and variant: the headline
    pair reading, its power-only variant and the reversed estimate.
    """
    candidates = sorted(scratch_dir.rglob(_V0_FINAL_GLOB)) or sorted(scratch_dir.rglob(_V0_INTERIM))
    if not candidates:
        return pd.DataFrame(columns=["read_turbine", "against", "variant", "reading_pp", "unc_one_sigma_pp"])
    trdf = pd.read_csv(candidates[-1])
    trdf = trdf[trdf["test_wtg"] != trdf["ref"]]
    frames = [
        pd.DataFrame(
            {
                "read_turbine": trdf["test_wtg"].to_numpy(),
                "against": trdf["ref"].to_numpy(),
                "variant": variant,
                "reading_pp": 100 * trdf[column].to_numpy(dtype=float),
                "unc_one_sigma_pp": 100 * trdf["unc_one_sigma_frc"].to_numpy(dtype=float),
            }
        )
        for column, variant in _V0_VARIANTS.items()
        if column in trdf.columns
    ]
    return pd.concat(frames, ignore_index=True)


def summarise(estimates: pd.DataFrame) -> pd.DataFrame:
    """Return one row per test turbine with each method's estimate, their difference, and a mean row."""
    table = estimates.pivot_table(index="test", columns="method", values="estimate_pp", aggfunc="first")
    table.columns = [str(c) for c in table.columns]
    if POWER_MODEL in table.columns and V0 in table.columns:
        table["power_model_minus_v0_pp"] = table[POWER_MODEL] - table[V0]
    table["references"] = estimates.groupby("test")["references"].first()
    mean = table.drop(columns="references").mean(numeric_only=True)
    mean["references"] = ""
    table.loc["mean"] = mean
    return table


# --- the run ---------------------------------------------------------------------------------------


def run_turbine(
    test: str,
    refs: list[str],
    *,
    scada_df: pd.DataFrame,
    out_dir: Path,
    changeover: pd.Timestamp = CHANGEOVER,
    baseline_months: int = BASELINE_MONTHS,
    campaign_months: int = CAMPAIGN_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both methods on ``test`` against ``refs`` alone; return its estimate rows and pair rows.

    The campaign is the placebo arm restricted to the turbine and its references, so the pool each
    method sees is identical. The power model's per-reference readings (each reference read against
    the rest of the pool) and v0's per-pair readings both land in the pair rows.
    """
    from benchmarking.baselines.hot_context import build_hot_v0_context  # noqa: PLC0415
    from benchmarking.baselines.v0_binned import V0BinnedMethod  # noqa: PLC0415
    from benchmarking.campaigns.methods import carried_forward_methods  # noqa: PLC0415
    from benchmarking.campaigns.runner import CampaignRunner, per_turbine_table  # noqa: PLC0415
    from benchmarking.campaigns.shift_probe import probe_campaign  # noqa: PLC0415
    from benchmarking.diagnostics.context import era5_source_label  # noqa: PLC0415
    from benchmarking.harness.northing import era5_direction  # noqa: PLC0415
    from benchmarking.synthetic.sources.hill_of_towie import HOT_LAT, HOT_LON  # noqa: PLC0415

    participating = [test, *refs]
    context = build_hot_v0_context(wtg_names=participating)
    era5_df = context.reanalysis_datasets[0].data
    campaign = probe_campaign(
        changeover,
        turbines=participating,
        upgraded=[test],
        baseline_months=baseline_months,
        campaign_months=campaign_months,
    )
    dataset = campaign.generate(scada_df[scada_df["TurbineName"].isin(participating)])
    spec = campaign.spec()
    index = pd.DatetimeIndex(dataset.synthetic_df.index.unique()).sort_values()
    label = era5_source_label(HOT_LAT, HOT_LON)
    result = CampaignRunner(
        spec,
        dataset,
        build_methods=lambda wtg: [
            *carried_forward_methods(
                spec,
                out_dir=out_dir / wtg,
                era5_hourly_df=era5_df,
                era5_label=label,
                reference_screen=False,
                report_reference_uplifts=True,
            ),
            V0BinnedMethod(context, scratch_dir=out_dir / wtg / V0),
        ],
        era5_wd=era5_direction(era5_df, index),
        northing_out_dir=out_dir / "northing",
        northing_plots=False,
    ).run()
    headline = per_turbine_table(result)
    headline = headline[headline["method"].isin([POWER_MODEL, V0])]
    estimates = pd.DataFrame(
        {
            "method": headline["method"].to_numpy(),
            "estimate_pp": 100 * headline["estimate"].to_numpy(dtype=float),
            "references": ",".join(refs),
        }
    )
    stability = result.report.reference_stability
    stability = stability[stability["method"] == POWER_MODEL]
    pairs = [
        pd.DataFrame(
            {
                "read_turbine": stability["turbine"].to_numpy(),
                "against": "pool",
                "method": POWER_MODEL,
                "variant": "reference_vs_pool",
                "reading_pp": 100 * stability["uplift"].to_numpy(dtype=float),
                "unc_one_sigma_pp": float("nan"),
            }
        ),
        v0_pair_readings(out_dir / test / V0).assign(method=V0),
    ]
    return estimates, pd.concat(pairs, ignore_index=True)


def _load_hot(turbines: Sequence[str], *, baseline_months: int, campaign_months: int) -> pd.DataFrame:
    from benchmarking.campaigns.shift_probe import analysis_period  # noqa: PLC0415
    from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada  # noqa: PLC0415

    start, end = analysis_period(CHANGEOVER, baseline_months=baseline_months, campaign_months=campaign_months)
    logger.info("loading Hill of Towie SCADA %s..%s for %s", start, end, list(turbines))
    scada_df, _ = load_hot_scada(
        start_dt=start, end_dt_excl=end, wtg_numbers=[int(w[1:]) for w in turbines], wtg_names=list(turbines)
    )
    return scada_df


def _write_meta(run_dir: Path, **meta: object) -> None:
    from benchmarking.baselines.prepost_matrix.study import git_state  # noqa: PLC0415

    path = run_dir / PROBE_JSON
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    commit, dirty = git_state()
    payload = {
        **existing,
        **meta,
        "commit": commit,
        "dirty": dirty,
        "updated": f"{pd.Timestamp.now():%Y-%m-%dT%H:%M:%S}",
    }
    payload.setdefault("started", payload["updated"])
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("v0 probe %s", payload)


def _read_or_empty(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=list(columns))


def _append(table: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Append ``rows`` to ``table`` without concatenating an empty frame, which pandas warns about."""
    if not len(table):
        return rows.reset_index(drop=True)
    if not len(rows):
        return table
    return pd.concat([table, rows], ignore_index=True)


def run_hot(
    *,
    turbines: Sequence[str] | None = None,
    n_refs: int = DEFAULT_N_REFS,
    smoke: bool = False,
    resume: Path | None = None,
    coords: Mapping[str, tuple[float, float]] | None = None,
    estimate: Callable[[str, list[str]], tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    out_root: Path | None = None,
) -> Path:
    """Run both methods on every test turbine with its nearest references; return the run directory.

    :param turbines: the test turbines; the whole placebo farm when ``None``
    :param n_refs: how many nearest neighbours each test turbine is read against
    :param smoke: one turbine, two references, a one-month campaign
    :param resume: an existing run directory to finish; its finished turbines are skipped
    :param coords: turbine positions; the published Hill of Towie layout when ``None``
    :param estimate: what runs one turbine, ``(test, refs) -> (estimate rows, pair rows)``; the
        real two-method run when ``None``. A test passes a fake.
    :param out_root: where ``v0_probe/`` runs are written; ``WIND_UP_BENCHMARKING_OUTPUT_DIR`` when ``None``
    """
    from benchmarking.campaigns.placebo import PLACEBO_TURBINES  # noqa: PLC0415
    from benchmarking.synthetic.sources.hill_of_towie import HOT_COORDINATES  # noqa: PLC0415

    tests = list(SMOKE_TURBINES if smoke else (turbines or PLACEBO_TURBINES))
    n_refs = SMOKE_N_REFS if smoke else n_refs
    campaign_months = SMOKE_CAMPAIGN_MONTHS if smoke else CAMPAIGN_MONTHS
    hoods = neighbourhoods(tests, coords=coords if coords is not None else HOT_COORDINATES, n_refs=n_refs)
    root = out_root if out_root is not None else output_root_dir()
    run_dir = (
        resume if resume is not None else root / f"hot{'_smoke' if smoke else ''}_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _log_to(run_dir / PROBE_LOG)
    estimates = _read_or_empty(run_dir / ESTIMATES_CSV, ESTIMATE_COLUMNS)
    pairs = _read_or_empty(run_dir / PAIRS_CSV, PAIR_COLUMNS)
    done = set(estimates["test"].astype(str))
    _write_meta(
        run_dir,
        case="hot",
        changeover=CHANGEOVER,
        baseline_months=BASELINE_MONTHS,
        campaign_months=campaign_months,
        n_refs=n_refs,
        neighbourhoods=hoods,
        resumed_from=sorted(done),
    )
    runner = (
        estimate
        if estimate is not None
        else _real_estimator(hoods, done=done, run_dir=run_dir, campaign_months=campaign_months)
    )
    failed: list[str] = []
    for test, refs in hoods.items():
        if test in done:
            logger.info("%s already done, skipping", test)
            continue
        logger.info("=== %s against %s", test, refs)
        try:
            new_estimates, new_pairs = runner(test, refs)
        except Exception:
            logger.exception("%s failed; carrying on", test)
            failed.append(test)
            continue
        estimates = _append(estimates, new_estimates.assign(test=test)[list(ESTIMATE_COLUMNS)])
        pairs = _append(pairs, new_pairs.assign(test=test)[list(PAIR_COLUMNS)])
        estimates.to_csv(run_dir / ESTIMATES_CSV, index=False)
        pairs.to_csv(run_dir / PAIRS_CSV, index=False)
        for row in new_estimates.itertuples():
            logger.info("%s %s: %+.3f pp", test, row.method, row.estimate_pp)
    _write_meta(run_dir, failed=failed)
    if len(estimates):
        summary = summarise(estimates)
        summary.to_csv(run_dir / SUMMARY_CSV)
        logger.info("v0 probe summary:\n%s", summary.round(3).to_string())
    return run_dir


def _real_estimator(
    hoods: Mapping[str, list[str]], *, done: set[str], run_dir: Path, campaign_months: int
) -> Callable[[str, list[str]], tuple[pd.DataFrame, pd.DataFrame]]:
    """Load the SCADA the unfinished turbines need once and return the two-method runner over it."""
    needed = sorted({t for test, refs in hoods.items() if test not in done for t in (test, *refs)})
    scada_df = _load_hot(needed, baseline_months=BASELINE_MONTHS, campaign_months=campaign_months)

    def estimate(test: str, refs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        return run_turbine(test, refs, scada_df=scada_df, out_dir=run_dir / test, campaign_months=campaign_months)

    return estimate


def output_root_dir() -> Path:
    """Return where probe runs are written: ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``v0_probe``."""
    return output_root() / OUTPUT_DIRNAME


# --- command line ----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Run the comparison, finish a stopped run, or re-summarise an existing one."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    hot = sub.add_parser("hot", help="v0 and the power model on the Hill of Towie placebo arm")
    hot.add_argument("--turbines", nargs="+", default=None)
    hot.add_argument("--n-refs", type=int, default=DEFAULT_N_REFS)
    hot.add_argument("--smoke", action="store_true", help="one turbine, two references, a one-month campaign")
    hot.add_argument("--resume", type=Path, default=None, help="finish this run directory")
    summ = sub.add_parser("summarise", help="re-summarise an existing run")
    summ.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Before anything resolves a path: where data, outputs and the reanalysis cache live.
    load_env()
    if args.command == "hot":
        run_hot(turbines=args.turbines, n_refs=args.n_refs, smoke=args.smoke, resume=args.resume)
    else:
        summary = summarise(pd.read_csv(args.run_dir / ESTIMATES_CSV))
        summary.to_csv(args.run_dir / SUMMARY_CSV)
        print(summary.round(3).to_string())  # noqa: T201


if __name__ == "__main__":
    main()
