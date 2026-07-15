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

**Deltas are reported raw, and "unchanged" is judged per method** (see :data:`_UNCHANGED_ATOL`), because
the two methods' reproducibility differs by four orders of magnitude — a fact measured on two runs of
identical code, not assumed:

- ``toggle_specialist`` reproduces **exactly** (max |delta| 0.0 across every cell), so it is held to an
  effectively bit-exact band. That makes it a genuinely strict regression detector.
- ``power_model`` does **not**, despite its ``seed``: LightGBM's threaded float reduction order varies,
  and the seed governs sampling rather than that. Measured: **0.05 pp at campaign_weeks=1, exactly 0.0
  at 2 and 8 weeks** — the noise is sparsity-driven, since with a week of data the model sits near a
  split boundary and a tiny float difference flips a tree.

The max observed delta is logged every run, so a cell sitting just inside its band stays visible.

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
from benchmarking.harness import (
    StudyConfig,
    conditional_leaderboard,
    leaderboard,
    plot_campaign_curves,
    score_study,
)
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
# v2 adds the per-power-bin cells alongside the headline, so every cell gained condition /
# condition_bin. A v1 file cannot be diffed against a v2 run (its cells carry no condition), so the
# bump makes an old baseline read as stale rather than silently mis-merge.
_BASELINE_SCHEMA = "toggle_methods_compare_baseline_v2"
# Per-cell metrics recorded and **diffed**. spread/score: lower is better; bias: |bias| nearer 0 is better.
_METRIC_COLS = ["bias", "spread", "score"]
# Recorded per cell but never diffed: wall time is machine- and load-dependent, so diffing it would
# trip the unchanged verdict on every run. Kept because a change that makes a method dramatically
# slower is a regression worth seeing, and because it is the record of what a method actually costs.
_WALL_TIME_COLS = ["wall_time_s_sum", "wall_time_s_mean"]
_CELL_COLS = [*_METRIC_COLS, "mean_estimate", "mean_truth", "n_replicates"]
_MERGE_KEYS = ["method", "profile", _LENGTH_COL, "condition", "condition_bin"]
_PP = 100.0  # fraction -> percentage points
# How close a re-run must land to the benchmark to read "unchanged" (fraction). **Per method, because
# the two methods' reproducibility differs by four orders of magnitude — measured, not assumed:**
#
# - `toggle_specialist` is pure arithmetic. Two runs of identical code reproduce **exactly** (max
#   |delta| = 0.0 across every cell). Its band only has to clear `record_baseline`'s `round(8)`
#   residual (~1e-9), so 1e-7 holds it to what is effectively a bit-exact standard.
# - `power_model` is **not** reproducible run to run despite `seed`: LightGBM's threaded float
#   reduction order varies, and the seed governs sampling rather than that. Measured on two runs of
#   identical code: **5e-4 (0.05 pp) at campaign_weeks=1, and exactly 0.0 at 2 and 8 weeks.** The
#   noise is sparsity-driven — with a week of data the model sits near a split boundary, so a tiny
#   float difference flips a tree and moves the estimate. 1e-3 sits at 2x that measured floor, and
#   matches study_power_model_compare's 0.1 pp band (which this now explains rather than copies).
#
# Holding toggle_specialist to power_model's band would discard a much stronger regression detector,
# which is the whole reason these are separate. The max observed delta is always logged, so a cell
# sitting just inside its band stays visible instead of hiding behind the verdict.
_UNCHANGED_ATOL: dict[str, float] = {"toggle_specialist": 1e-7, "power_model": 1e-3}
_DEFAULT_UNCHANGED_ATOL = 1e-3  # a method not named above gets the conservative band


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
    """Construct the HoT-configured toggle methods, fastest first.

    Both report the **power** axis and only that, so the benchmark tracks per-bin behaviour (a change
    can leave the headline untouched and still wreck a bin) and so the two methods are compared on a
    common axis — ``power`` is the only one ``toggle_specialist`` can offer. ``power_model``'s ws/TI
    conditional is deliberately not duplicated here: ``study_power_model_compare`` already tracks it,
    and recording it for one method only would make this study's table asymmetric for no gain.
    """
    return [
        ToggleSpecialistMethod(
            columns=HOT_COLUMNS,
            conditions=("power",),
            rated_power_kw=HOT_RATED_POWER_KW,
            out_dir=out_dir / "toggle_specialist_runs",
        ),
        PowerModelMethod(
            columns=HOT_COLUMNS,
            baseline_rated_power_kw=HOT_RATED_POWER_KW,
            conditions=("power",),
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
    """One row per (method, profile, campaign_weeks, condition, bin) of bias/spread/score.

    Both the headline (``condition == "overall"``) and the per-power-bin cells are recorded. Tracking
    only the headline would miss the failure mode this study most needs to catch: a change can leave
    the overall number untouched and still wreck an individual bin.
    """
    overall = leaderboard(results, length_col=_LENGTH_COL).assign(condition="overall", condition_bin="overall")
    conditional = conditional_leaderboard(results, length_col=_LENGTH_COL)
    stacked = pd.concat([overall[[*_MERGE_KEYS, *_CELL_COLS]], conditional[[*_MERGE_KEYS, *_CELL_COLS]]])
    # Wall time is per *estimate*, so only the headline rows carry it (a per-bin row has no fit of its
    # own and gets NaN). It is recorded but never diffed — see _METRIC_COLS — because it is machine-
    # and load-dependent; it exists so a change that makes a method dramatically slower is visible.
    stacked = stacked.merge(overall[[*_MERGE_KEYS, *_WALL_TIME_COLS]], on=_MERGE_KEYS, how="left")
    return stacked.sort_values(_MERGE_KEYS).reset_index(drop=True)


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

    Returns the merged frame (empty if no benchmark is recorded yet). Deltas are raw; the
    unchanged/moved verdict applies :data:`_UNCHANGED_ATOL`, whose size is set by ``power_model``'s
    run-to-run nondeterminism rather than chosen.
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


def _log_unchanged_verdict(merged: pd.DataFrame, *, atol: dict[str, float] | None = None) -> None:
    """Log, per method, whether every diffed cell matches the benchmark within that method's band.

    The max observed delta is always reported, so "unchanged" never hides a cell sitting just inside
    the band — the margin against :data:`_UNCHANGED_ATOL` is the reader's evidence, not the verdict.
    """
    bands = _UNCHANGED_ATOL if atol is None else atol
    delta_cols = [f"d_{col}" for col in _METRIC_COLS]
    for method, group in merged.groupby("method"):
        comparable = group.dropna(subset=delta_cols)
        if comparable.empty:
            logger.info("%s: no cells line up with the benchmark (new method or new cells).", method)
            continue
        band = bands.get(str(method), _DEFAULT_UNCHANGED_ATOL)
        worst = float(comparable[delta_cols].abs().to_numpy().max())
        moved = comparable[(comparable[delta_cols].abs() > band).any(axis=1)]
        if moved.empty:
            logger.info(
                "%s: UNCHANGED — all %d cells within +/-%.3g pp of the benchmark (max delta %.3g pp).",
                method,
                len(comparable),
                band * _PP,
                worst * _PP,
            )
        else:
            logger.warning(
                "%s: MOVED — %d of %d cells differ by more than +/-%.3g pp (max delta %.3g pp):\n%s",
                method,
                len(moved),
                len(comparable),
                band * _PP,
                worst * _PP,
                moved[[*_MERGE_KEYS, *delta_cols]].to_string(index=False),
            )


def plot_results(lb: pd.DataFrame, comparison_dir: Path) -> None:
    """Write one campaign-length curve per profile (both methods overlaid), from the headline rows.

    Restricted to ``condition == "overall"``: the leaderboard now also carries per-bin rows, and a
    campaign curve drawn over both would silently average the headline together with six power bins.
    """
    comparison_dir.mkdir(parents=True, exist_ok=True)
    headline = lb[lb["condition"] == "overall"]
    for profile in sorted(headline["profile"].unique()):
        plot_campaign_curves(
            headline[headline["profile"] == profile],
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
    # The headline goes to the log; the per-bin rows are ~6x more numerous and would bury it. They are
    # all in leaderboard.csv, and the benchmark diff reports any that move.
    headline = lb[lb["condition"] == "overall"]
    logger.info(
        "Leaderboard (headline; %d per-bin rows also recorded, see leaderboard.csv):\n%s",
        len(lb) - len(headline),
        headline[["method", "profile", _LENGTH_COL, *_METRIC_COLS]].to_string(index=False),
    )

    comparison_dir = output_dir / "comparison"
    plot_results(lb, comparison_dir)
    if args.update_baseline:
        record_baseline(lb, study=toggle_study(), path=baseline_path)
    else:
        compare_to_benchmark(lb, baseline_path=baseline_path, comparison_dir=comparison_dir)
    logger.info("All done. Outputs under %s", output_dir)


if __name__ == "__main__":
    main()
