"""Shared setup for the overnight prepost/toggle studies: a fresh per-run output dir and log.

Each run gets its own timestamped output directory plus a ``run.log`` recording provenance (git
commit, study config, timings), so results from different runs never overwrite or
silently intermix (the older studies dumped everything into a single ``prepost``/``toggle`` dir,
leaving stale ``leaderboard_all_profiles.csv`` and accumulating per-run diagnostic CSVs), and every
output is traceable to the exact code that produced it.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarking.harness import StudyConfig

logger = logging.getLogger(__name__)

_REPO_DIR = Path(__file__).resolve().parent


def _git_commit() -> str:
    """Return the current commit (with a ``-dirty`` suffix if the tree has uncommitted changes)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=_REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            cwd=_REPO_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def start_overnight_run(mode: str, study: StudyConfig, output_root: Path) -> Path:
    """Create a fresh timestamped output dir for one overnight run and wire logging into it.

    :param mode: ``"prepost"`` or ``"toggle"`` (the per-mode subfolder)
    :param study: the study configuration (logged for provenance)
    :param output_root: the base ``WIND_UP_BENCHMARKING_OUTPUT_DIR`` (the studies' ``default_output_root``
        already appends ``mode``; pass its ``.parent`` so we can add the timestamp under ``mode``)
    :return: the per-run directory to pass as ``out_root`` to the study runner
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / mode / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "run.log")],
        force=True,
    )
    logger.info("Starting %s overnight run", mode)
    logger.info("git commit: %s", _git_commit())
    logger.info("output dir: %s", out_dir)
    logger.info("study config: %s", study)
    return out_dir
