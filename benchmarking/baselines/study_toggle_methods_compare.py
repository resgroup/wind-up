"""Score the toggle-capable methods on Hill of Towie and diff them against a committed benchmark.

The regression harness for the **toggle** methods — ``toggle_specialist`` and ``power_model`` — on a
known-stable real dataset. Its job is to answer one question objectively: *did a change to a method
move its numbers, and in which direction?*

The cases are deliberately small-signal and short:

- **Profiles:** a placebo (``cp_0pct``) plus a symmetric +/-2% Cp pair. Symmetric magnitudes let a
  sign error show up as an asymmetry between the pair, and 2% is the regime a real toggle campaign
  actually lives in — far more informative here than a +/-10% signal any method can find.
- **Campaign grid:** 1/2/4/8 **weeks**. A real toggle campaign runs for weeks, and the short end is
  where these methods are hardest pressed.

Each run diffs the fresh bias/spread/score per ``(method, profile, campaign_weeks)`` against the
committed benchmark (``study_toggle_methods_compare_baseline.json``, next to this script) and logs
the deltas plus a per-cell ``benchmark_comparison.csv``.

**Deltas are reported raw, and an unchanged method must read exactly 0.0.** Ground truth is
deterministic in the study config + seed, so an unchanged method re-run on the same commit produces
identical numbers — not merely similar ones. This script is therefore a strict regression detector:
any non-zero delta means the change under test moved the method, and the size of the move is the
thing to judge. (Contrast ``study_power_model_compare``, whose neutral band exists to absorb
cross-run noise it cannot avoid.)

Run from the repo root::

    uv run python -m benchmarking.baselines.study_toggle_methods_compare

Restrict to one profile for fast feedback with ``--profiles cp_0pct``. Record the benchmark with
``--update-baseline`` (deliberately — only when a change is accepted), then commit the JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarking.baselines.example_prepost_study import (
    DEFAULT_END_DT_EXCL,
    DEFAULT_START_DT,
    DEFAULT_TREATMENT_START_RANGE,
    DEFAULT_TURBINE_SUBSET,
    DEFAULT_WTG_NUMBERS,
    MIN_PRE_MONTHS,
)
from benchmarking.baselines.example_toggle_study import DEFAULT_TOGGLE_PERIOD
from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.power_model import PowerModelMethod
from benchmarking.baselines.toggle_specialist import ToggleSpecialistMethod
from benchmarking.harness import StudyConfig, leaderboard, plot_campaign_curves, score_study
from benchmarking.synthetic import HOT_COLUMNS, HOT_RATED_POWER_KW, ConstantCpChange
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

logger = logging.getLogger(__name__)

# The toggle methods under regression. power_model is the slow one, so it runs last (see
# `on_method_complete` in score_study — order fastest-first for the earliest feedback).
COMPARE_METHODS = ["toggle_specialist", "power_model"]

# Local to this script rather than added to `overnight_profiles()`: adding profiles there would leave
# study_power_model_compare's committed baseline missing cells (its --update-baseline demands the full
# profile set), disturbing the very benchmark this script exists to protect.
TOGGLE_PROFILES: dict[str, list] = {
    "cp_0pct": [ConstantCpChange(delta=0.0)],
    "cp_plus_2pct": [ConstantCpChange(delta=0.02)],
    "cp_minus_2pct": [ConstantCpChange(delta=-0.02)],
}

CAMPAIGN_WEEKS = [1, 2, 4, 8]
N_REPLICATES = 4
SEED = 0

_LENGTH_COL = "campaign_weeks"
_DEFAULT_OUTPUT_DIR = Path.home() / "temp" / "wind-up-benchmarking" / "toggle_methods_compare"
_BASELINE_PATH = Path(__file__).resolve().parent / "study_toggle_methods_compare_baseline.json"
_BASELINE_SCHEMA = "toggle_methods_compare_baseline_v1"
# Per-cell metrics recorded and diffed. spread/score: lower is better; bias: |bias| nearer 0 is better.
_METRIC_COLS = ["bias", "spread", "score"]
_MERGE_KEYS = ["method", "profile", _LENGTH_COL]
_PP = 100.0  # fraction -> percentage points


def toggle_study() -> StudyConfig:
    """Return the study every run scores: toggle mode over the weeks grid."""
    return StudyConfig(
        mode="toggle",
        turbine_subset=DEFAULT_TURBINE_SUBSET,
        treatment_start_range=DEFAULT_TREATMENT_START_RANGE,
        min_pre_months=MIN_PRE_MONTHS,
        campaign_weeks=CAMPAIGN_WEEKS,
        toggle_period=DEFAULT_TOGGLE_PERIOD,
        n_replicates=N_REPLICATES,
        seed=SEED,
    )


def _select_profiles(requested: list[str] | None) -> dict[str, list]:
    """Return the profiles to score: all when ``requested`` is ``None``, else the named subset.

    Unknown names fail loudly rather than silently scoring less.
    """
    if requested is None:
        return TOGGLE_PROFILES
    unknown = [name for name in requested if name not in TOGGLE_PROFILES]
    if unknown:
        msg = f"unknown profile(s) {unknown}; available: {sorted(TOGGLE_PROFILES)}"
        raise ValueError(msg)
    return {name: TOGGLE_PROFILES[name] for name in requested}


def _build_methods(out_dir: Path, *, era5_hourly_df: pd.DataFrame) -> list:
    """Construct the HoT-configured toggle methods, fastest first."""
    return [
        ToggleSpecialistMethod(
            columns=HOT_COLUMNS,
            out_dir=out_dir / "toggle_specialist_runs",
        ),
        PowerModelMethod(
            columns=HOT_COLUMNS,
            baseline_rated_power_kw=HOT_RATED_POWER_KW,
            era5_hourly_df=era5_hourly_df,
            out_dir=out_dir / "power_model_runs",
        ),
    ]


def run_study(out_dir: Path, *, profiles: list[str] | None = None) -> pd.DataFrame:
    """Score both toggle methods over the profiles, writing a per-profile ``results_*.csv``.

    Returns the concatenated tidy results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scada_df, _ = load_hot_scada(
        start_dt=DEFAULT_START_DT,
        end_dt_excl=DEFAULT_END_DT_EXCL,
        wtg_numbers=DEFAULT_WTG_NUMBERS,
        wtg_names=DEFAULT_TURBINE_SUBSET,
    )
    context = build_hot_v0_context(wtg_names=DEFAULT_TURBINE_SUBSET)
    study = toggle_study()

    all_results = []
    for profile_name, profile in _select_profiles(profiles).items():
        # Per-profile subfolder: a method's run dir is <method>_<wtg>_<start>_<end> (no profile), so
        # profiles sharing a (wtg, window) would otherwise overwrite each other's diagnostics.
        methods = _build_methods(out_dir / profile_name, era5_hourly_df=context.reanalysis_datasets[0].data)
        logger.info("Scoring profile %s with %s", profile_name, ", ".join(COMPARE_METHODS))
        results = score_study(
            scada_df,
            profile=profile,
            methods=methods,
            study=study,
            profile_name=profile_name,
        )
        results.to_csv(out_dir / f"results_{profile_name}.csv", index=False)
        all_results.append(results)
    return pd.concat(all_results, ignore_index=True)


def _git_commit() -> str:
    """Return the short HEAD commit (``-dirty`` if the tree is modified), or ``unknown``."""
    repo = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def methods_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, profile, campaign_weeks) of bias/spread/score over the overall rows."""
    lb = leaderboard(results, length_col=_LENGTH_COL)
    return lb.sort_values(_MERGE_KEYS).reset_index(drop=True)


def record_baseline(lb: pd.DataFrame, *, study: StudyConfig, path: Path) -> None:
    """Write the benchmark: every scored cell plus the provenance needed to interpret it."""
    commit = _git_commit()
    doc = {
        "schema": _BASELINE_SCHEMA,
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit,
        "n_replicates": study.n_replicates,
        "seed": study.seed,
        "campaign_weeks": list(study.campaign_lengths),
        "profiles": sorted(lb["profile"].unique()),
        "methods": sorted(lb["method"].unique()),
        "cells": lb.round(8).to_dict(orient="records"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    logger.info("Recorded toggle-methods benchmark at %s (commit %s). Commit the JSON.", path, commit)


def _load_baseline(path: Path) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Load the recorded benchmark cells (+ provenance), or ``None`` if absent/stale."""
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if doc.get("schema") != _BASELINE_SCHEMA:
        logger.warning(
            "Baseline %s has schema %r, expected %r — run --update-baseline to regenerate.",
            path,
            doc.get("schema"),
            _BASELINE_SCHEMA,
        )
        return None
    return pd.DataFrame(doc["cells"]), doc


def compare_to_benchmark(lb: pd.DataFrame, *, baseline_path: Path, comparison_dir: Path) -> pd.DataFrame:
    """Diff the fresh cells against the committed benchmark; write the per-cell CSV and log the deltas.

    Returns the merged frame (empty if no benchmark is recorded yet). Deltas are raw: an unchanged
    method must read exactly 0.0, so any non-zero value is a real move to judge, not noise.
    """
    loaded = _load_baseline(baseline_path)
    if loaded is None:
        logger.warning("No benchmark recorded at %s yet — run with --update-baseline to set it.", baseline_path)
        return pd.DataFrame()
    base, prov = loaded
    base = base[base["profile"].isin(lb["profile"].unique())]  # scope to the profiles actually run
    merged = lb.merge(base, on=_MERGE_KEYS, how="outer", suffixes=("", "_base"))
    for col in _METRIC_COLS:
        merged[f"d_{col}"] = merged[col] - merged[f"{col}_base"]
    merged = merged.sort_values(_MERGE_KEYS)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(comparison_dir / "benchmark_comparison.csv", index=False)

    show = pd.DataFrame(
        {
            "method": merged["method"],
            "profile": merged["profile"],
            _LENGTH_COL: merged[_LENGTH_COL],
            "bias": merged["bias"] * _PP,
            "d_bias": merged["d_bias"] * _PP,
            "spread": merged["spread"] * _PP,
            "d_spread": merged["d_spread"] * _PP,
            "score": merged["score"] * _PP,
            "d_score": merged["d_score"] * _PP,
        }
    ).round(6)
    logger.info(
        "Toggle methods vs benchmark (recorded %s, commit %s) [pp]; an unchanged method reads d_*=0.0 exactly:\n%s",
        prov.get("recorded_utc", "?"),
        prov.get("git_commit", "?"),
        show.to_string(index=False),
    )
    _log_unchanged_verdict(merged)
    return merged


def _log_unchanged_verdict(merged: pd.DataFrame) -> None:
    """Log, per method, whether every diffed cell is bit-identical to the benchmark."""
    delta_cols = [f"d_{col}" for col in _METRIC_COLS]
    for method, group in merged.groupby("method"):
        comparable = group.dropna(subset=delta_cols)
        if comparable.empty:
            logger.info("%s: no cells line up with the benchmark (new method or new cells).", method)
            continue
        moved = comparable[(comparable[delta_cols] != 0.0).any(axis=1)]
        if moved.empty:
            logger.info("%s: UNCHANGED — all %d cells identical to the benchmark.", method, len(comparable))
        else:
            logger.warning(
                "%s: MOVED — %d of %d cells differ from the benchmark:\n%s",
                method,
                len(moved),
                len(comparable),
                moved[[*_MERGE_KEYS, *delta_cols]].to_string(index=False),
            )


def plot_results(lb: pd.DataFrame, comparison_dir: Path) -> None:
    """Write one campaign-length curve per profile (both methods overlaid)."""
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for profile in sorted(lb["profile"].unique()):
        plot_campaign_curves(
            lb[lb["profile"] == profile],
            save_path=comparison_dir / f"campaign_curves_{profile}.png",
            title=f"toggle - {profile} (toggle_specialist vs power_model)",
            length_col=_LENGTH_COL,
        )


def main() -> None:
    """Score the toggle methods over the profiles, then diff against (or record) the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="where method runs, results and the comparison are written",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(TOGGLE_PROFILES),
        default=None,
        help="restrict to a subset of profiles for fast feedback (default: all three). "
        "Cannot be combined with --update-baseline.",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=_BASELINE_PATH,
        help="the committed benchmark JSON to diff against (and to --update-baseline)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="overwrite the recorded benchmark with this run (do this deliberately, only when a change "
        "is accepted); without it, the run is diffed against the benchmark",
    )
    args = parser.parse_args()
    if args.profiles is not None and args.update_baseline:
        # record_baseline rewrites the cells wholesale, so a subset run would drop the other profiles
        # from the committed benchmark. Refuse rather than silently corrupt it.
        parser.error("--update-baseline needs the full profile set; do not combine it with --profiles")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    output_dir = args.output_dir.expanduser()
    baseline_path = args.baseline_path.expanduser()

    results = run_study(output_dir, profiles=args.profiles)
    lb = methods_leaderboard(results)
    lb.to_csv(output_dir / "leaderboard.csv", index=False)
    logger.info("Leaderboard:\n%s", lb[[*_MERGE_KEYS, *_METRIC_COLS]].to_string(index=False))

    comparison_dir = output_dir / "comparison"
    plot_results(lb, comparison_dir)
    if args.update_baseline:
        record_baseline(lb, study=toggle_study(), path=baseline_path)
    else:
        compare_to_benchmark(lb, baseline_path=baseline_path, comparison_dir=comparison_dir)
    logger.info("All done. Outputs under %s", output_dir)


if __name__ == "__main__":
    main()
