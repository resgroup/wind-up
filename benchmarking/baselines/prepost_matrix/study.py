"""The prepost campaign matrix as a resumable study: cells run in parallel, merged and compared.

A study is two sibling directories under the output root. ``prepost_matrix__<commit7>[-dirty]__<size>``
is the one to download; merging and comparing need nothing else::

    run_meta.json          matrix settings, size, master seed, commit, dirty flag, host, start time
    git_diff.patch         only when dirty
    run.log                driver log
    cells/<cell_id>.json   each cell's record, written last
    cells.csv              one row per cell
    candidate_baseline.json
    leak_check.csv
    comparison/            the compare output

Its ``__detail`` sibling stays where the study ran::

    sources/               the SCADA the cells read
    cells/<cell_id>/       cell.log, the northing found, and a failed cell's method diagnostics
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from benchmarking.baselines.prepost_matrix.cells import Cell, MatrixSettings, matrix_cells
from benchmarking.baselines.prepost_matrix.execute import METHOD_DIRNAME, describe_cell, execute_cell, prefetch
from benchmarking.baselines.prepost_matrix.metrics import TABLE_KEYS, baseline_tables, compare_tables, leak_check
from benchmarking.campaigns.composed import output_root

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

STUDY_PREFIX = "prepost_matrix__"
DETAIL_SUFFIX = "__detail"
BASELINE_DIR = Path(__file__).resolve().parents[1]
BASELINE_SCHEMA = "prepost_campaign_matrix_baseline_v1"
RUN_META = "run_meta.json"
CELLS_DIRNAME = "cells"
CELL_LOG = "cell.log"
CANDIDATE = "candidate_baseline.json"
# Rebuild the worker pool at most this many times after a worker dies (for instance, out of memory).
MAX_POOL_RESTARTS = 3
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_RSS_SAMPLE_S = 0.5
_BYTES_PER_MB = 1024 * 1024
_ROUND = 10

Execute = Callable[..., dict[str, Any]]
Prefetch = Callable[[Path, list[Cell]], None]


# --- the study directory -----------------------------------------------------------------------


def git_state() -> tuple[str, bool]:
    """Return HEAD's short commit and whether tracked files are modified; ``unknown`` outside git."""
    repo = Path(__file__).resolve().parent
    try:
        commit = _git("rev-parse", "--short=7", "HEAD", cwd=repo).strip()
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no", cwd=repo).strip())
    except (subprocess.SubprocessError, OSError):
        return "unknown", False
    return commit, dirty


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout  # noqa: S603, S607


def study_dir_for(root: Path, *, commit: str, dirty: bool, size: str) -> Path:
    """Return the study directory of a ``size`` run from ``commit``."""
    return root / f"{STUDY_PREFIX}{commit}{'-dirty' if dirty else ''}__{size}"


def detail_dir(study_dir: Path) -> Path:
    """Return the sibling holding the study's sources and each cell's working files; it need not be downloaded."""
    return study_dir.with_name(study_dir.name + DETAIL_SUFFIX)


def baseline_path_for(size: str) -> Path:
    """Return the committed baseline a ``size`` study is compared with."""
    return BASELINE_DIR / f"study_prepost_campaign_matrix_baseline_{size}.json"


def open_study(study_dir: Path, settings: MatrixSettings, *, size: str, commit: str, dirty: bool) -> None:
    """Create the study directory, or reopen it after checking it ran the same matrix.

    :raises ValueError: if the directory's ``run_meta.json`` has other settings or another master seed
    """
    meta_path = study_dir / RUN_META
    if meta_path.exists():
        recorded = json.loads(meta_path.read_text())["settings"]
        if recorded != settings.to_json():
            msg = (
                f"{study_dir} ran another matrix: {recorded} there, {settings.to_json()} here. "
                f"Use another output root, or delete the study."
            )
            raise ValueError(msg)
        return
    study_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "settings": settings.to_json(),
        "size": size,
        "master_seed": settings.master_seed,
        "commit": commit,
        "dirty": dirty,
        "host": socket.gethostname(),
        "platform": sys.platform,
        "started_utc": _now(),
    }
    if dirty:
        (study_dir / "git_diff.patch").write_text(_git("diff", "HEAD", cwd=Path(__file__).resolve().parent))
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def study_settings(study_dir: Path) -> MatrixSettings:
    """Return the matrix settings a study was run with."""
    return MatrixSettings.from_json(json.loads((study_dir / RUN_META).read_text())["settings"])


def study_size(study_dir: Path) -> str:
    """Return the size a study was run at."""
    return str(json.loads((study_dir / RUN_META).read_text())["size"])


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cell_dir(study_dir: Path, cell: Cell) -> Path:
    """Return where ``cell`` does its work: its log, northing and method diagnostics, in the detail sibling."""
    return detail_dir(study_dir) / CELLS_DIRNAME / cell.cell_id


def record_path(study_dir: Path, cell: Cell) -> Path:
    """Return where ``cell``'s record is written, in the study itself."""
    return study_dir / CELLS_DIRNAME / f"{cell.cell_id}.json"


def read_cell(study_dir: Path, cell: Cell) -> dict[str, Any] | None:
    """Return ``cell``'s record, or ``None`` if it has not finished."""
    path = record_path(study_dir, cell)
    return json.loads(path.read_text()) if path.exists() else None


# --- one cell ----------------------------------------------------------------------------------


def run_cell(
    study_dir: Path, cell: Cell, *, settings: MatrixSettings, execute: Execute = execute_cell
) -> dict[str, Any]:
    """Run ``cell``, write its record last, and return it. Never raises.

    An exception is recorded as ``status="failed"`` with its traceback. The cell's logging goes to
    its own ``cell.log``. A successful cell's method diagnostics are removed.
    """
    directory = cell_dir(study_dir, cell)
    record_file = record_path(study_dir, cell)
    record_file.unlink(missing_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    record_file.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "cell_id": cell.cell_id,
        "arm": cell.arm,
        "site": cell.site,
        "seed_index": cell.seed_index,
        "multiplier": cell.multiplier,
        "k": cell.k,
        "post_months": cell.post_months,
    }
    with _cell_log(directory / CELL_LOG), _PeakRss() as rss:
        start = time.perf_counter()
        try:
            record |= describe_cell(cell, settings=settings)
            record |= execute(cell, detail_dir=detail_dir(study_dir), cell_dir=directory, settings=settings)
            record["status"] = "ok"
        except Exception as exc:
            logger.exception("cell %s failed", cell.cell_id)
            record |= {"status": "failed", "error_type": type(exc).__name__, "traceback": traceback.format_exc()}
        record["wall_time_s"] = time.perf_counter() - start
    record["peak_rss_mb"] = rss.peak_mb
    if record["status"] == "ok":
        shutil.rmtree(directory / METHOD_DIRNAME, ignore_errors=True)
    partial = record_file.with_suffix(".partial")
    partial.write_text(json.dumps(record, indent=2, default=str) + "\n")
    partial.replace(record_file)
    return record


@contextmanager
def _cell_log(path: Path) -> Iterator[None]:
    """Route every log record to ``path`` for the duration."""
    root = logging.getLogger()
    handler = logging.FileHandler(path, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    level = root.level
    root.addHandler(handler)
    root.setLevel(min(level, logging.INFO) if level else logging.INFO)
    try:
        yield
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
        handler.close()


class _PeakRss:
    """Sample this process's resident memory on a thread; ``peak_mb`` is the largest sample."""

    def __init__(self) -> None:
        self.peak_mb = float("nan")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> _PeakRss:  # noqa: PYI034 - never subclassed
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def _sample(self) -> None:
        import psutil  # noqa: PLC0415 - only the study's workers need it

        process = psutil.Process()
        peak = 0
        while True:
            peak = max(peak, process.memory_info().rss)
            if self._stop.wait(_RSS_SAMPLE_S):
                break
        self.peak_mb = max(peak, process.memory_info().rss) / _BYTES_PER_MB


def _worker(study_dir: Path, cell: Cell, settings: MatrixSettings, execute: Execute) -> dict[str, Any]:
    return run_cell(study_dir, cell, settings=settings, execute=execute)


def _init_worker() -> None:
    logging.captureWarnings(capture=True)


# --- the whole study ---------------------------------------------------------------------------


def run_study(
    settings: MatrixSettings,
    *,
    size: str,
    root: Path | None = None,
    workers: int,
    limit: int | None = None,
    retry_failed: bool = False,
    execute: Execute = execute_cell,
    prefetch_sources: Prefetch = prefetch,
    baseline_path: Path | None = None,
) -> Path:
    """Run every cell without a record, then merge and compare. Re-running resumes; returns the study directory.

    :param settings: the matrix
    :param size: the size's name, which names the study and its committed baseline
    :param root: where the study directory goes; the benchmarking output root by default
    :param workers: cells run at once, each on one core; 1 runs them in this process
    :param limit: run only the first ``limit`` cells, as a timing trial
    :param retry_failed: run failed cells again too; a failure is otherwise a result, kept on resuming
    :param execute: runs one cell; :func:`~benchmarking.baselines.prepost_matrix.execute.execute_cell`
    :param prefetch_sources: fills the detail directory's ``sources/`` and the reanalysis cache first
    :param baseline_path: the committed baseline compared against; the size's by default
    """
    commit, dirty = git_state()
    root = root if root is not None else output_root()
    study_dir = study_dir_for(root, commit=commit, dirty=dirty, size=size)
    open_study(study_dir, settings, size=size, commit=commit, dirty=dirty)
    detail_dir(study_dir).mkdir(exist_ok=True)
    handler = _log_to(study_dir / "run.log")
    try:
        cells = matrix_cells(settings)[:limit]
        logger.info("Study %s: %d cells", study_dir, len(cells))
        prefetch_sources(detail_dir(study_dir), cells)
        todo = [c for c in cells if _wants_run(read_cell(study_dir, c), retry_failed=retry_failed)]
        logger.info("%d cells already done; running %d on %d worker(s)", len(cells) - len(todo), len(todo), workers)
        _run_cells(study_dir, todo, settings=settings, workers=workers, execute=execute, n_total=len(cells))
        merge_study(study_dir)
        compare_study(study_dir, baseline_path=baseline_path)
        logger.info("Download %s; %s can stay here", study_dir, detail_dir(study_dir))
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return study_dir


def _wants_run(record: dict[str, Any] | None, *, retry_failed: bool) -> bool:
    return record is None or (retry_failed and record.get("status") == "failed")


def _log_to(path: Path) -> logging.Handler:
    handler = logging.FileHandler(path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    return handler


def single_thread_env() -> None:
    """Set the BLAS and OpenMP thread counts to 1, for this process and any it spawns."""
    for name in _THREAD_VARS:
        os.environ[name] = "1"


def _run_cells(
    study_dir: Path,
    cells: list[Cell],
    *,
    settings: MatrixSettings,
    workers: int,
    execute: Execute,
    n_total: int,
) -> None:
    done = n_total - len(cells)

    def finished(record: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        if record["status"] == "failed":
            reason = record["traceback"].strip().splitlines()[-1]
            logger.info("%s failed (%d/%d): %s", record["cell_id"], done, n_total, reason)
        else:
            logger.info("%s %s (%d/%d)", record["cell_id"], record["status"], done, n_total)

    if workers <= 1:
        for cell in cells:
            finished(run_cell(study_dir, cell, settings=settings, execute=execute))
        return

    single_thread_env()
    context = multiprocessing.get_context("spawn")
    pending = list(cells)
    for attempt in range(MAX_POOL_RESTARTS + 1):
        lost: list[Cell] = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
            futures = {pool.submit(_worker, study_dir, cell, settings, execute): cell for cell in pending}
            for future in as_completed(futures):
                try:
                    finished(future.result())
                except BrokenProcessPool:  # noqa: PERF203 - a worker died; its cell is run again
                    lost.append(futures[future])
        if not lost:
            return
        logger.warning(
            "a worker died; %d cells unfinished (restart %d of %d)", len(lost), attempt + 1, MAX_POOL_RESTARTS
        )
        pending = sorted(lost, key=cells.index)
    logger.error("stopped after %d pool restarts; re-run to resume", MAX_POOL_RESTARTS)


# --- merge, compare, accept, list --------------------------------------------------------------


def merge_study(study_dir: Path) -> dict[str, Any]:
    """Merge the finished cells into ``cells.csv``, ``leak_check.csv`` and ``candidate_baseline.json``.

    Reads only the cell files, so it can run on a partial study; the candidate then says it is incomplete.
    """
    meta = json.loads((study_dir / RUN_META).read_text())
    cells = matrix_cells(study_settings(study_dir))
    records = [r for c in cells if (r := read_cell(study_dir, c)) is not None]
    by_id = {r["cell_id"]: r for r in records}
    pd.DataFrame(
        [
            {
                "cell_id": c.cell_id,
                "arm": c.arm,
                "site": c.site,
                "seed_index": c.seed_index,
                "multiplier": c.multiplier,
                "k": c.k,
                "post_months": c.post_months,
                "status": by_id.get(c.cell_id, {}).get("status", "pending"),
                "error_type": by_id.get(c.cell_id, {}).get("error_type", ""),
                "wall_time_s": by_id.get(c.cell_id, {}).get("wall_time_s"),
                "peak_rss_mb": by_id.get(c.cell_id, {}).get("peak_rss_mb"),
            }
            for c in cells
        ]
    ).astype({"seed_index": "Int64", "multiplier": "Int64"}).to_csv(study_dir / "cells.csv", index=False)

    n_ok = sum(r["status"] == "ok" for r in records)
    n_failed = sum(r["status"] == "failed" for r in records)
    tables = baseline_tables(records)
    doc = {
        "schema": BASELINE_SCHEMA,
        "units": "uplift fractions",
        "git_commit": f"{meta['commit']}{'-dirty' if meta['dirty'] else ''}",
        "recorded_utc": _now(),
        "platform": meta["platform"],
        "matrix": meta["settings"],
        "n_cells": len(cells),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "complete": n_ok == len(cells),
        **{name: _records(table) for name, table in tables.items()},
    }
    (study_dir / CANDIDATE).write_text(json.dumps(doc, indent=2) + "\n")

    leaks = leak_check(records)
    leaks.to_csv(study_dir / "leak_check.csv", index=False)
    if not leaks.empty:
        logger.info(
            "Leak check: the largest move of one reference reading across multipliers is %.3f pp "
            "(%s); median %.3f pp over %d readings",
            100 * leaks["leak"].iloc[0],
            ", ".join(f"{k}={leaks[k].iloc[0]}" for k in ("arm", "site", "seed_index", "k", "post_months", "turbine")),
            100 * leaks["leak"].median(),
            len(leaks),
        )
    logger.info("Merged %d ok and %d failed of %d cells into %s", n_ok, n_failed, len(cells), study_dir / CANDIDATE)
    return doc


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: _plain(v) for k, v in row.items()} for row in table.to_dict(orient="records")]


def _plain(value: object) -> object:
    """Return ``value`` ready for JSON: NaN as null, floats rounded, numpy scalars as Python ones."""
    if hasattr(value, "item"):
        value = value.item()  # type: ignore[union-attr]
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, _ROUND)
    return value


def load_tables(doc: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Return a baseline document's tables as frames."""
    return {
        name: pd.DataFrame(doc.get(name, []), columns=None if doc.get(name) else keys)
        for name, keys in TABLE_KEYS.items()
    }


def compare_study(study_dir: Path, *, baseline_path: Path | None = None) -> pd.DataFrame | None:
    """Compare the study's candidate with the committed baseline, its size's by default; write ``comparison/``.

    :return: the comparison, or ``None`` when no baseline is committed yet
    """
    baseline_path = baseline_path if baseline_path is not None else baseline_path_for(study_size(study_dir))
    candidate = json.loads((study_dir / CANDIDATE).read_text())
    if not baseline_path.exists():
        logger.info("No committed baseline at %s yet; nothing to compare", baseline_path)
        return None
    baseline = json.loads(baseline_path.read_text())
    comparison = compare_tables(load_tables(candidate), load_tables(baseline))
    out = study_dir / "comparison"
    out.mkdir(exist_ok=True)
    comparison.to_csv(out / "compare.csv", index=False)
    moved = comparison[comparison["moved"]]
    moved.to_csv(out / "moved.csv", index=False)
    logger.info(
        "Compared with %s (commit %s): %d of %d values moved%s",
        baseline_path.name,
        baseline.get("git_commit"),
        len(moved),
        len(comparison),
        "" if candidate["complete"] else " -- the candidate is INCOMPLETE",
    )
    return comparison


def accept_candidate(study_dir: Path, *, baseline_path: Path | None = None) -> None:
    """Promote the study's candidate over the committed baseline, its size's by default.

    :raises ValueError: if the study is incomplete, has a failed cell, or was run from a dirty tree
    """
    baseline_path = baseline_path if baseline_path is not None else baseline_path_for(study_size(study_dir))
    doc = json.loads((study_dir / CANDIDATE).read_text())
    if doc["n_failed"]:
        msg = f"refusing to accept {study_dir}: {doc['n_failed']} cell(s) failed"
        raise ValueError(msg)
    if not doc["complete"]:
        msg = f"refusing to accept {study_dir}: only {doc['n_ok']} of {doc['n_cells']} cells are ok"
        raise ValueError(msg)
    if str(doc["git_commit"]).endswith("-dirty"):
        msg = f"refusing to accept {study_dir}: it ran from a dirty tree ({doc['git_commit']})"
        raise ValueError(msg)
    shutil.copyfile(study_dir / CANDIDATE, baseline_path)
    logger.info("Promoted %s over %s; commit the new JSON", study_dir / CANDIDATE, baseline_path)


def list_studies(root: Path | None = None) -> pd.DataFrame:
    """Return one row per study: size, commit, whether HEAD contains it, run date, cells, and disk use.

    ``download_mb`` is the study directory; ``detail_mb`` its sibling, which need not be downloaded.
    """
    root = root if root is not None else output_root()
    rows = []
    for study_dir in sorted(root.glob(f"{STUDY_PREFIX}*")):
        meta_path = study_dir / RUN_META
        if study_dir.name.endswith(DETAIL_SUFFIX) or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        statuses = [json.loads(p.read_text()).get("status") for p in study_dir.glob(f"{CELLS_DIRNAME}/*.json")]
        rows.append(
            {
                "study": study_dir.name,
                "size": meta.get("size", ""),
                "commit": meta["commit"],
                "dirty": meta["dirty"],
                "in_head": _is_ancestor(meta["commit"]),
                "started_utc": meta["started_utc"],
                "n_ok": statuses.count("ok"),
                "n_failed": statuses.count("failed"),
                "download_mb": _disk_mb(study_dir),
                "detail_mb": _disk_mb(detail_dir(study_dir)),
            }
        )
    columns = ["study", "size", "commit", "dirty", "in_head", "started_utc", "n_ok", "n_failed"]
    return pd.DataFrame(rows, columns=[*columns, "download_mb", "detail_mb"])


def _is_ancestor(commit: str) -> bool:
    try:
        _git("merge-base", "--is-ancestor", commit, "HEAD", cwd=Path(__file__).resolve().parent)
    except (subprocess.SubprocessError, OSError):
        return False
    return True


def _disk_mb(directory: Path) -> float:
    return round(sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) / _BYTES_PER_MB, 3)
