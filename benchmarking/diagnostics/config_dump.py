"""Write a human-readable record of what a diagnostic run actually received.

v0 drops a ``config_*.json`` per run and it is very handy for confirming the inputs after the
fact (feedback 2026-06-26). The benchmarking methods get the same, as YAML: the method and its
parameters, the run's window / turbines / timebase, the columns in play, and the git commit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    from benchmarking.diagnostics.context import DiagnosticContext

_REPO_DIR = Path(__file__).resolve().parent


def _git_commit() -> str:
    """Return the current commit (``-dirty`` suffix if the tree has uncommitted changes), or ``unknown``."""
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


def _diagnostic_columns(ctx: DiagnosticContext) -> dict[str, str | None]:
    """Return the source-native column names in play (so a reader knows what each plot read)."""
    cols = ctx.columns
    return {
        "turbine": cols.turbine,
        "active_power": cols.active_power,
        "wind_speed": cols.wind_speed,
        "wind_speed_sd": cols.wind_speed_sd,
        "gen_rpm": cols.gen_rpm,
        "pitch": cols.pitch,
        "reactive_power": cols.reactive_power,
        "nacelle_position": cols.nacelle_position,
        "ambient_temp": cols.ambient_temp,
        "availability": cols.availability,
    }


def write_run_config(
    ctx: DiagnosticContext,
    *,
    method_name: str,
    method_params: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``<run_dir>/config_<run_dir name>.yaml`` describing the run; return its path.

    :param method_name: the method's leaderboard name
    :param method_params: the method's configuration (plain, YAML-serialisable values)
    :param extra: optional method-specific extras to record (e.g. ERA5 lag, n_folds)
    """
    index = ctx.index
    used = np.asarray(ctx.used_ts, dtype=bool)
    treated = np.asarray(ctx.treated_ts, dtype=bool)
    record: dict[str, Any] = {
        "method": method_name,
        "method_params": {k: _yamlable(v) for k, v in method_params.items()},
        "mode": ctx.mode,
        "test_wtg": ctx.test_wtg,
        "references": ctx.references(),
        "n_turbines": int(ctx.scada_df[ctx.turbine_col].nunique()),
        "first_timestamp": _yamlable(index.min()) if len(index) else None,
        "last_timestamp": _yamlable(index.max()) if len(index) else None,
        "timebase_seconds": ctx.timebase.total_seconds(),
        "n_timestamps": len(index),
        "n_used_timestamps": int(used.sum()),
        "n_used_upgraded": int((used & treated).sum()),
        "n_used_baseline": int((used & ~treated).sum()),
        "has_era5": ctx.era5_df is not None,
        "columns": _diagnostic_columns(ctx),
        "git_commit": _git_commit(),
    }
    if extra:
        record["extra"] = {k: _yamlable(v) for k, v in extra.items()}

    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.run_dir / f"config_{ctx.run_dir.name}.yaml"
    path.write_text(yaml.safe_dump(record, sort_keys=False, default_flow_style=False))
    return path


def _yamlable(value: Any) -> Any:  # noqa: ANN401
    """Convert numpy/pandas scalars and containers to plain Python so ``yaml.safe_dump`` accepts them."""
    if isinstance(value, dict):
        return {k: _yamlable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yamlable(v) for v in value]
    return _yamlable_scalar(value)


def _yamlable_scalar(value: Any) -> Any:  # noqa: ANN401
    """Convert a single numpy/pandas scalar (or Path) to a plain YAML-serialisable value."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
