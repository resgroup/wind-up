"""One-off: split the single v2 toggle benchmark into a portable + a per-platform v3 pair (F30).

The v2 file was recorded on a Windows laptop and holds both methods. ``power_model``'s cells are
machine-specific, so they can only ever be re-derived on that machine — hence a split rather than a
re-record. ``toggle_specialist``'s cells are portable (verified at ~5e-07 pp on Linux), so they
become the shared baseline unchanged.

Delete this script once both platform baselines are recorded and committed.

    uv run python -m benchmarking.baselines.migrate_toggle_baseline_v2_to_v3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarking.baselines.study_toggle_methods_compare import (
    _BASELINE_DIR,
    _BASELINE_SCHEMA,
    _portable_methods,
    baseline_paths,
)

logger = logging.getLogger(__name__)

_V2_SCHEMA = "toggle_methods_compare_baseline_v2"
_V2_NAME = "study_toggle_methods_compare_baseline.json"
# The v2 file predates the machine fingerprint. Its platform is known (it was recorded on the Windows
# laptop); the rest cannot be recovered and are written as null, which _warn_on_fingerprint_mismatch
# reads as "no claim" rather than guessing.
_RECORDED_PLATFORM = "win32"
_UNRECOVERABLE: dict[str, Any] = {"cpu_count": None, "python_version": None, "lightgbm_version": None}


def _split(doc: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split v2 cells into (portable, machine-specific) by method."""
    cells = pd.DataFrame(doc["cells"])
    portable_names = _portable_methods()
    return cells[cells["method"].isin(portable_names)], cells[~cells["method"].isin(portable_names)]


def _write(path: Path, *, cells: pd.DataFrame, doc: dict[str, Any], platform: str) -> None:
    """Write one v3 file, carrying the v2 provenance forward rather than restamping it."""
    out = {
        "schema": _BASELINE_SCHEMA,
        "recorded_utc": doc["recorded_utc"],
        "git_commit": doc["git_commit"],
        "platform": platform,
        **_UNRECOVERABLE,
        "n_replicates": doc["n_replicates"],
        "seed": doc["seed"],
        "campaign_weeks": doc["campaign_weeks"],
        "profiles": doc["profiles"],
        "methods": sorted(cells["method"].unique()),
        "cells": cells.to_dict(orient="records"),
    }
    path.write_text(json.dumps(out, indent=2) + "\n")
    logger.info("Wrote %s (%d cells, methods %s)", path.name, len(cells), out["methods"])


def migrate(baseline_dir: Path | None = None) -> None:
    """Split the committed v2 file into the v3 portable + win32 pair."""
    directory = _BASELINE_DIR if baseline_dir is None else baseline_dir
    v2_path = directory / _V2_NAME
    doc = json.loads(v2_path.read_text())
    if doc.get("schema") != _V2_SCHEMA:
        msg = f"{v2_path} has schema {doc.get('schema')!r}, expected {_V2_SCHEMA!r}; nothing to migrate"
        raise ValueError(msg)

    portable, machine_specific = _split(doc)
    if portable.empty or machine_specific.empty:
        msg = (
            f"expected both portable and machine-specific cells in {v2_path}; "
            f"got {len(portable)} and {len(machine_specific)}"
        )
        raise ValueError(msg)

    portable_path, _ = baseline_paths(directory)
    _, win32_path = baseline_paths(directory, platform=_RECORDED_PLATFORM)
    _write(portable_path, cells=portable, doc=doc, platform=_RECORDED_PLATFORM)
    _write(win32_path, cells=machine_specific, doc=doc, platform=_RECORDED_PLATFORM)
    logger.info("Migration done. `git rm %s`, then record this machine's own platform baseline.", _V2_NAME)


def main() -> None:
    """Run the migration."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-dir", type=Path, default=None, help="where the baseline JSONs live")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    migrate(args.baseline_dir.expanduser() if args.baseline_dir else None)


if __name__ == "__main__":
    main()
