"""The prepost campaign matrix: shipped ``wind-up`` over the real HoT T13 campaign and many synthetic ones.

Each cell is one full ``wind-up`` run on one (campaign, multiplier, K, post length). The study is one
resumable invocation, meant for tmux on a shared box::

    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix download
    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix run --workers 16
    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix run --workers 16 --limit 32  # timing trial
    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix merge STUDY_DIR
    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix compare STUDY_DIR [--accept-candidate]
    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix run-cell STUDY_DIR CELL_ID
    uv run python -m benchmarking.baselines.study_prepost_campaign_matrix list

Settings come from the environment or a ``.env`` file at the repository root (copy ``.env.example``):
where Zenodo data is downloaded, where studies write, and the reanalysis cache.

``download`` fetches every site's open data from Zenodo and caches its reanalysis. ``run`` does the
same first, so ``download`` is optional; then it writes the SCADA each site needs into the study,
runs every cell not yet ``ok``, then merges the cells into ``candidate_baseline.json`` and compares
it with the committed ``study_prepost_campaign_matrix_baseline.json``. Re-running the same command
resumes. Progress is one line per cell in the study's ``run.log``. The study directory sits under
``WIND_UP_BENCHMARKING_OUTPUT_DIR``.

``--limit N`` runs the first N cells, which are the longest: wall time is about the number of cells
times the mean cell wall time over the workers. Check that, and peak RSS times workers against the
box's memory, before a full run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

from benchmarking.baselines.prepost_matrix.cells import MatrixSettings, matrix_cells
from benchmarking.baselines.prepost_matrix.execute import download_sources, prefetch
from benchmarking.baselines.prepost_matrix.study import (
    BASELINE_PATH,
    DEFAULT_WORKERS,
    accept_candidate,
    compare_study,
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
    sub.add_parser("download", help="download the open data and reanalysis every cell needs")
    run = sub.add_parser("run", help="run (or resume) the study, then merge and compare")
    run.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="cells run at once, one core each")
    run.add_argument("--limit", type=int, default=None, help="run only the first N cells (a timing trial)")
    run.add_argument("--root", type=Path, default=None, help="where the study directory goes")
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
    load_env()
    if args.command == "download":
        download_sources(matrix_cells(MatrixSettings()))
    elif args.command == "run":
        run_study(MatrixSettings(), root=args.root, workers=args.workers, limit=args.limit)
    elif args.command == "merge":
        merge_study(args.study_dir)
    elif args.command == "compare":
        merge_study(args.study_dir)
        compare_study(args.study_dir)
        if args.accept_candidate:
            accept_candidate(args.study_dir, baseline_path=BASELINE_PATH)
    elif args.command == "run-cell":
        settings = study_settings(args.study_dir)
        cells = {c.cell_id: c for c in matrix_cells(settings)}
        if args.cell_id not in cells:
            sys.exit(f"no cell {args.cell_id!r} in {args.study_dir}")
        cell = cells[args.cell_id]
        single_thread_env()
        prefetch(args.study_dir, [cell])
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
