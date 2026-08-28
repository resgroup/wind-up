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

Each run diffs the fresh bias/spread/score per ``(method, profile, campaign_weeks, condition, bin)``
against the committed benchmark and logs the deltas plus a per-cell ``benchmark_comparison.csv``.

**Deltas are raw, and "unchanged" is judged per method** (:data:`_REPRODUCIBILITY`), because the two
methods' reproducibility differs by four orders of magnitude — measured, not assumed.
``toggle_specialist`` is pure arithmetic and reproduces exactly; ``power_model`` does not, despite
its ``seed``, because LightGBM's threaded float reduction order varies (~0.05 pp same-machine). The
max observed delta is logged every run, so a cell just inside its band stays visible.

**The benchmark is split across files because that reproducibility is also machine-dependent.**
LightGBM's reduction order depends on the machine, so ``power_model`` scores ~0.7 pp against a
benchmark recorded elsewhere — 14x its same-machine noise, and a permanent false MOVED. Its cells
therefore live in a per-platform file (``..._baseline_<sys.platform>.json``), while
``toggle_specialist``, which is portable (~5e-07 pp across machines), lives in the shared
``..._baseline_portable.json``. A run diffs the two merged.

Run from the repo root::

    uv run python -m benchmarking.baselines.study_toggle_methods_compare

Restrict to one profile for fast feedback with ``--profiles cp_0pct``. Record with
``--update-baseline`` (deliberately — only when a change is accepted), then commit the JSON(s); it
writes this machine's platform file and, only when they actually change, the portable one.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform as platform_module
import subprocess
import sys
from dataclasses import dataclass
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
_BASELINE_DIR = Path(__file__).resolve().parent
_BASELINE_STEM = "study_toggle_methods_compare_baseline"
# v3 splits the single file into a portable baseline plus one per platform.
_BASELINE_SCHEMA = "toggle_methods_compare_baseline_v3"
# Per-cell metrics recorded and **diffed**. spread/score: lower is better; bias: |bias| nearer 0 is better.
_METRIC_COLS = ["bias", "spread", "score"]
# Recorded per cell but never diffed: wall time is machine- and load-dependent, so diffing it would
# trip the unchanged verdict on every run. Kept because a change that makes a method dramatically
# slower is a regression worth seeing, and because it is the record of what a method actually costs.
_WALL_TIME_COLS = ["wall_time_s_sum", "wall_time_s_mean"]
_CELL_COLS = [*_METRIC_COLS, "mean_estimate", "mean_truth", "n_replicates"]
_MERGE_KEYS = ["method", "profile", _LENGTH_COL, "condition", "condition_bin"]
_PP = 100.0  # fraction -> percentage points


@dataclass(frozen=True)
class MethodReproducibility:
    """How reproducible a method is, and under what conditions.

    :param band: how close a re-run must land to the benchmark to read "unchanged" (fraction)
    :param portable: whether its numbers survive a change of machine, and so whether its cells live
        in the shared baseline or in a per-platform one
    """

    band: float
    portable: bool


# Measured, not assumed. `toggle_specialist` is pure arithmetic: two runs reproduce exactly, and it
# matches a baseline recorded on another machine to ~5e-07 pp, so 1e-7 holds it to an effectively
# bit-exact standard. `power_model` is not reproducible even run to run (LightGBM's threaded float
# reduction order; the seed governs sampling, not that): ~0.05 pp same-machine, but ~0.7 pp against a
# baseline from another machine — hence portable=False.
_REPRODUCIBILITY: dict[str, MethodReproducibility] = {
    "toggle_specialist": MethodReproducibility(band=1e-7, portable=True),
    "power_model": MethodReproducibility(band=1e-3, portable=False),
}
# An unclassified method is assumed machine-specific: the safe side, since wrongly calling one
# portable produces a permanent, confusing failure on the other machine.
_DEFAULT_REPRODUCIBILITY = MethodReproducibility(band=1e-3, portable=False)


def _reproducibility(method: str) -> MethodReproducibility:
    """Return ``method``'s reproducibility facts, defaulting to the conservative assumption."""
    return _REPRODUCIBILITY.get(method, _DEFAULT_REPRODUCIBILITY)


def _portable_methods() -> set[str]:
    """Return the methods whose cells belong in the shared, cross-machine baseline."""
    return {name for name, repro in _REPRODUCIBILITY.items() if repro.portable}


def baseline_paths(baseline_dir: Path | None = None, platform: str | None = None) -> tuple[Path, Path]:
    """Return ``(portable_path, platform_path)`` for this machine.

    ``sys.platform`` is a proxy for *the machine*, which is only sound while there is one machine per
    platform; a second box on the same platform would silently share a file. The recorded fingerprint
    (see :func:`_provenance`) is what would expose that.
    """
    directory = _BASELINE_DIR if baseline_dir is None else baseline_dir
    key = sys.platform if platform is None else platform
    return directory / f"{_BASELINE_STEM}_portable.json", directory / f"{_BASELINE_STEM}_{key}.json"


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
    """Return the short HEAD commit (``-dirty`` if *tracked* files are modified), or ``unknown``.

    ``--untracked-files=no`` is deliberate: only tracked modifications make a run irreproducible from
    its commit. An untracked file (a scratch script, an editor artifact, a local CLAUDE.md) has no
    bearing on what ``git checkout <commit>`` would run, and counting it would make ``--update-baseline``
    unusable for anyone with a stray file in their working copy.
    """
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
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
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


def _provenance(study: StudyConfig, lb: pd.DataFrame, *, git_commit: str) -> dict[str, Any]:
    """Return the context needed to interpret a recorded baseline, including which machine made it.

    The machine fingerprint exists because a ``power_model`` MOVED against a baseline from another
    machine is expected rather than a regression, and without this the file cannot say so.
    """
    return {
        "schema": _BASELINE_SCHEMA,
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit,
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "python_version": platform_module.python_version(),
        "lightgbm_version": _lightgbm_version(),
        "n_replicates": study.n_replicates,
        "seed": study.seed,
        "campaign_weeks": list(study.campaign_lengths),
        "profiles": sorted(lb["profile"].unique()),
    }


def _lightgbm_version() -> str | None:
    """LightGBM's version, or ``None`` when it is not installed (it is an optional dependency)."""
    try:
        import lightgbm  # noqa: PLC0415
    except ImportError:
        return None
    return str(lightgbm.__version__)


def _write_baseline(path: Path, *, cells: pd.DataFrame, provenance: dict[str, Any]) -> None:
    """Write one baseline file: its cells plus provenance.

    A no-op when there are no cells, rather than writing an empty file: a run that scored none of this
    file's methods knows nothing about them, and overwriting a good baseline with zero cells would
    silently destroy it.
    """
    if cells.empty:
        logger.info("No cells for %s in this run — leaving it alone.", path.name)
        return
    doc = {**provenance, "methods": sorted(cells["method"].unique()), "cells": cells.round(8).to_dict(orient="records")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    logger.info("Recorded %s (%d cells, methods %s).", path.name, len(cells), doc["methods"])


def record_baselines(
    lb: pd.DataFrame, *, study: StudyConfig, git_commit: str, baseline_dir: Path | None = None
) -> None:
    """Record this machine's baselines: the portable cells (shared) and the rest (per platform).

    ``git_commit`` is captured *before* the run, so a commit landing mid-sweep cannot stamp the
    baseline with code that never produced it.

    The portable file is only rewritten when it has to be. Its cells are the same on every machine by
    definition, so an unchanged recording leaves the file — and its provenance — untouched, which
    keeps the two laptops from fighting over it. Its ``git_commit`` therefore records when those
    numbers were last *established*, not who last ran a recording.
    """
    portable_path, platform_path = baseline_paths(baseline_dir)
    provenance = _provenance(study, lb, git_commit=git_commit)
    portable_names = _portable_methods()
    portable = lb[lb["method"].isin(portable_names)]
    machine_specific = lb[~lb["method"].isin(portable_names)]

    _check_portable_or_raise(portable, path=portable_path, git_commit=git_commit)
    if _portable_cells_match(portable, path=portable_path):
        logger.info("Portable baseline unchanged at %s — portability confirmed, not rewritten.", portable_path)
    else:
        _write_baseline(portable_path, cells=portable, provenance=provenance)
    _write_baseline(platform_path, cells=machine_specific, provenance=provenance)
    logger.info("Recorded benchmark(s) at commit %s on %s. Commit the JSON(s).", git_commit, sys.platform)


def _portable_cells_match(portable: pd.DataFrame, *, path: Path) -> bool:
    """Whether the fresh portable cells match the committed ones **within each method's band**.

    Not a bit-exact comparison, for two reasons. Portability is a claim at the band's precision
    (measured ~5e-07 pp), and `round(8)`'s 1e-8 resolution is close enough to that to flip a last
    digit. And the recorded cells carry wall time, which differs every run by construction — an exact
    comparison would rewrite the shared file on every recording and hand the two laptops a conflict.
    """
    loaded = _load_baseline(path)
    if loaded is None:
        return False
    base, _ = loaded
    merged = portable.merge(base, on=_MERGE_KEYS, how="outer", suffixes=("", "_base"), indicator=True)
    if (merged["_merge"] != "both").any():
        return False  # a cell appeared or vanished: not the same set of numbers
    for method, group in merged.groupby("method"):
        band = _reproducibility(str(method)).band
        for col in _METRIC_COLS:
            fresh, old = group[col], group[f"{col}_base"]
            delta = (fresh - old).abs()
            both_nan = fresh.isna() & old.isna()  # an empty bin is NaN in both and matches
            if ((delta > band) | (~both_nan & delta.isna())).any():
                return False
    return True


def _check_portable_or_raise(portable: pd.DataFrame, *, path: Path, git_commit: str) -> None:
    """Refuse to record when a portable method's cells moved **at the same commit**.

    From one machine "the numbers moved" is ambiguous: it means either the method changed (which is
    what --update-baseline is for) or portability broke. The commit disambiguates — same code
    producing different numbers on a different machine is a portability break, and since
    ``--update-baseline`` refuses a dirty tree the commit is trustworthy enough to lean on.
    """
    loaded = _load_baseline(path)
    if loaded is None or portable.empty:
        return
    _, prov = loaded
    if prov.get("git_commit") != git_commit or _portable_cells_match(portable, path=path):
        return
    msg = (
        f"portable baseline {path.name} was recorded at this same commit ({git_commit}) on "
        f"{prov.get('platform')}, but this machine ({sys.platform}) produces different cells for "
        f"{sorted(portable['method'].unique())}. Same code, different numbers, different machine: either a "
        f"method marked portable=True is not (check _REPRODUCIBILITY), or that file's commit is wrong. "
        f"Refusing to overwrite — this is the check the portable/per-platform split exists to make."
    )
    raise ValueError(msg)


def _load_baseline(path: Path) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Load one baseline's cells (+ provenance), or ``None`` if absent/stale."""
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


def _warn_on_fingerprint_mismatch(prov: dict[str, Any], *, path: Path) -> None:
    """Warn when the **platform** baseline was recorded somewhere unlike this machine. Never fatal.

    Only meaningful for the platform file, whose methods are machine-specific by definition. The
    portable file is *expected* to come from the other laptop — that is the point of it — so warning
    there would be noise on every run. Fields recorded as ``None`` (unrecoverable when the v2 file was
    migrated) make no claim and are skipped.
    """
    current = {"platform": sys.platform, "cpu_count": os.cpu_count(), "lightgbm_version": _lightgbm_version()}
    differing = {k: (prov.get(k), v) for k, v in current.items() if prov.get(k) is not None and prov.get(k) != v}
    if differing:
        detail = ", ".join(f"{k}: recorded {was!r}, now {now!r}" for k, (was, now) in differing.items())
        logger.warning(
            "%s holds machine-specific cells but was recorded on a machine unlike this one (%s). A MOVED "
            "verdict may be that rather than your change.",
            path.name,
            detail,
        )


def load_merged_baseline(baseline_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Load the portable + this-platform baselines merged, or ``None`` when neither is recorded.

    Either half missing is a warning, not an error: a fresh machine with no platform file still gets
    its portable regression check for free.
    """
    portable_path, platform_path = baseline_paths(baseline_dir)
    frames, provenance = [], {}
    for path in (portable_path, platform_path):
        loaded = _load_baseline(path)
        if loaded is None:
            logger.warning(
                "No usable benchmark at %s — %s. Run --update-baseline on this machine to record it.",
                path.name,
                "portable cells will not be diffed"
                if path == portable_path
                else f"{sys.platform} cells will not be diffed",
            )
            continue
        cells, prov = loaded
        if path == platform_path:  # the portable file is meant to come from the other machine
            _warn_on_fingerprint_mismatch(prov, path=path)
        frames.append(cells)
        provenance[path.name] = prov
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True), provenance


def compare_to_benchmark(lb: pd.DataFrame, *, comparison_dir: Path, baseline_dir: Path | None = None) -> pd.DataFrame:
    """Diff the fresh cells against the committed benchmarks; write the per-cell CSV and log the deltas.

    Returns the merged frame (empty if nothing is recorded yet). Deltas are raw; the unchanged/moved
    verdict applies each method's band from :data:`_REPRODUCIBILITY`.
    """
    loaded = load_merged_baseline(baseline_dir)
    if loaded is None:
        logger.warning("No benchmark recorded for this machine yet — run with --update-baseline to set it.")
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
        "Toggle methods vs benchmark (%s) [pp]; unchanged = within each method's band (%s):\n%s",
        "; ".join(
            f"{name} recorded {p.get('recorded_utc', '?')} at {p.get('git_commit', '?')}" for name, p in prov.items()
        ),
        ", ".join(f"{m} {r.band * _PP:g} pp" for m, r in _REPRODUCIBILITY.items()),
        show.to_string(index=False),
    )
    _log_unchanged_verdict(merged)
    return merged


def _log_unchanged_verdict(merged: pd.DataFrame, *, atol: dict[str, float] | None = None) -> None:
    """Log, per method, whether every diffed cell matches the benchmark within that method's band.

    The max observed delta is always reported, so "unchanged" never hides a cell sitting just inside
    its band.
    """
    delta_cols = [f"d_{col}" for col in _METRIC_COLS]
    for method, group in merged.groupby("method"):
        comparable = group.dropna(subset=delta_cols)
        if comparable.empty:
            logger.info("%s: no cells line up with the benchmark (new method or new cells).", method)
            continue
        band = (
            _reproducibility(str(method)).band if atol is None else atol.get(str(method), _DEFAULT_REPRODUCIBILITY.band)
        )
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
        "--baseline-dir",
        type=Path,
        default=None,
        help="directory holding the committed benchmark JSONs (default: next to this script)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="overwrite this machine's recorded benchmark with this run (do this deliberately, only when "
        "a change is accepted); without it, the run is diffed against the benchmark",
    )
    args = parser.parse_args()
    if args.profiles is not None and args.update_baseline:
        # A recording rewrites the cells wholesale, so a subset run would drop the other profiles from
        # the committed benchmark. Refuse rather than silently corrupt it.
        parser.error("--update-baseline needs the full profile set; do not combine it with --profiles")

    # Captured *before* the sweep: the run takes ~15 min, so reading HEAD afterwards would stamp the
    # baseline with whatever was committed meanwhile rather than the code that actually ran.
    git_commit = _git_commit()
    if args.update_baseline and git_commit.endswith("-dirty"):
        # The committed benchmark is only worth anything if a reader can check out the commit and
        # reproduce it. Recording from a dirty tree bakes in changes that commit does not contain, so
        # refuse rather than write an untraceable baseline. Commit first, then record, then commit the
        # JSON. (Without --update-baseline a dirty tree is fine — that run only reports.)
        parser.error(
            f"refusing to --update-baseline from a dirty tree (commit {git_commit}): the committed "
            f"benchmark must be reproducible from its commit. Commit your changes first, then re-run."
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    output_dir = args.output_dir.expanduser()
    baseline_dir = args.baseline_dir.expanduser() if args.baseline_dir is not None else None

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
        record_baselines(lb, study=toggle_study(), git_commit=git_commit, baseline_dir=baseline_dir)
    else:
        compare_to_benchmark(lb, comparison_dir=comparison_dir, baseline_dir=baseline_dir)
    logger.info("All done. Outputs under %s", output_dir)


if __name__ == "__main__":
    main()
