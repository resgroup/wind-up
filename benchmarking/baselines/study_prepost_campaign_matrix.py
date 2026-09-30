"""The prepost campaign matrix: shipped ``wind-up`` over the real HoT T13 campaign and many synthetic ones.

Each cell is one full ``wind-up`` run on one (campaign, multiplier, K, post length). A study is run at
a size: ``small``, a smoke test of every arm for a laptop, or ``big``, sized to about 11.5 h on 16
workers of the HPC. It is one resumable invocation, meant for tmux on a shared box::

    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix plan --size big
    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix download --size big
    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix run --size big
    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix run --size big --limit 32  # timing trial
    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix merge STUDY_DIR
    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix compare STUDY_DIR [--accept-candidate]
    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix run-cell STUDY_DIR CELL_ID
    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix list

Settings come from the environment or a ``.env`` file at the repository root (copy ``.env.example``):
where Zenodo data is downloaded, where studies write, and the reanalysis cache.

``plan`` projects a size's cells, CPU-hours, wall time on its workers and peak memory, from costs
measured on the HPC, or from a finished study's ``cells.csv`` with ``--measured STUDY_DIR``.

``download`` fetches every site's open data from Zenodo and caches its reanalysis. ``run`` does the
same first, so ``download`` is optional; then it writes the SCADA each site needs into the study's
detail directory, runs every cell without a record, then merges the cells into
``candidate_baseline.json`` and compares it with the size's committed
``study_prepost_campaign_matrix_baseline_<size>.json``. Re-running the same command resumes; a failed
cell is kept as a result unless ``--retry-failed``. Progress is one line per cell in the study's
``run.log``.

The study directory, ``prepost_matrix__<commit7>[-dirty]__<size>`` under
``WIND_UP_BENCHMARKING_OUTPUT_DIR``, is the one to download: merge and compare need only it. Its
``__detail`` sibling, holding the SCADA and every cell's log and working files, can stay behind.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

from benchmarking.baselines.prepost_matrix.cells import SIZES, matrix_cells
from benchmarking.baselines.prepost_matrix.execute import download_sources, prefetch
from benchmarking.baselines.prepost_matrix.plan import plan_study
from benchmarking.baselines.prepost_matrix.study import (
    accept_candidate,
    compare_study,
    detail_dir,
    list_studies,
    merge_study,
    run_cell,
    run_study,
    single_thread_env,
    study_settings,
)
from benchmarking.env import load_env

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarking.baselines.study_prepost_campaign_matrix",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sizes = sorted(SIZES)
    plan = sub.add_parser("plan", help="project a size's cells, hours and memory")
    plan.add_argument("--size", required=True, choices=sizes)
    plan.add_argument("--workers", type=int, default=None, help="override the size's worker count")
    plan.add_argument("--measured", type=Path, default=None, help="take cell costs from this study's cells.csv")
    download = sub.add_parser("download", help="download the open data and reanalysis every cell needs")
    download.add_argument("--size", required=True, choices=sizes)
    run = sub.add_parser("run", help="run (or resume) the study, then merge and compare")
    run.add_argument("--size", required=True, choices=sizes, help="small: a laptop smoke test; big: the HPC")
    run.add_argument("--workers", type=int, default=None, help="override the size's worker count")
    run.add_argument("--limit", type=int, default=None, help="run only the first N cells (a timing trial)")
    run.add_argument("--root", type=Path, default=None, help="where the study directory goes")
    run.add_argument("--retry-failed", action="store_true", help="run failed cells again too")
    merge = sub.add_parser("merge", help="merge the finished cells into the candidate baseline")
    merge.add_argument("study_dir", type=Path)
    compare = sub.add_parser("compare", help="compare the candidate with the committed baseline")
    compare.add_argument("study_dir", type=Path)
    compare.add_argument("--accept-candidate", action="store_true", help="promote the candidate over the baseline")
    one = sub.add_parser("run-cell", help="run one cell in the foreground")
    one.add_argument("study_dir", type=Path)
    one.add_argument("cell_id")
    listing = sub.add_parser("list", help="list the study directories")
    listing.add_argument("--root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the command."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "plan":
        size = SIZES[args.size]
        workers = size.workers if args.workers is None else args.workers
        print(plan_study(size.settings, workers=workers, measured=args.measured).describe())  # noqa: T201
        return
    load_env()
    if args.command == "download":
        download_sources(matrix_cells(SIZES[args.size].settings))
    elif args.command == "run":
        size = SIZES[args.size]
        run_study(
            size.settings,
            size=args.size,
            root=args.root,
            workers=size.workers if args.workers is None else args.workers,
            limit=args.limit,
            retry_failed=args.retry_failed,
        )
    elif args.command == "merge":
        merge_study(args.study_dir)
    elif args.command == "compare":
        merge_study(args.study_dir)
        compare_study(args.study_dir)
        if args.accept_candidate:
            accept_candidate(args.study_dir)
    elif args.command == "run-cell":
        settings = study_settings(args.study_dir)
        cells = {c.cell_id: c for c in matrix_cells(settings)}
        if args.cell_id not in cells:
            sys.exit(f"no cell {args.cell_id!r} in {args.study_dir}")
        cell = cells[args.cell_id]
        single_thread_env()
        prefetch(detail_dir(args.study_dir), [cell])
        record = run_cell(args.study_dir, cell, settings=settings)
        logger.info(
            "%s %s in %.0f s, peak RSS %.0f MB",
            cell.cell_id,
            record["status"],
            record["wall_time_s"],
            record["peak_rss_mb"],
        )
    elif args.command == "list":
        print(list_studies(args.root).to_string(index=False))  # noqa: T201


if __name__ == "__main__":
    main()
