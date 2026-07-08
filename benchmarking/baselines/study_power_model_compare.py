"""Re-run ``power_model`` over the overnight cases and compare it against the existing v0 + naive.

Iterative-development helper. The overnight v0 runs are very slow, so they are computed once
(:mod:`benchmarking.baselines.study_overnight_prepost` / ``study_overnight_toggle`` with
``include_v0=True``) and kept on disk. As ``power_model`` is improved you only want to re-run the
cheap methods and re-draw the comparison plots, reusing the frozen v0 numbers. That is what this
script does:

1. **Run** ``power_model`` **and** ``naive_ratio`` over the *exact* current overnight prepost + toggle
   configs — same seven :func:`~benchmarking.baselines.overnight_profiles.overnight_profiles`, same
   campaign grids (Issue 14: 1/2/3/6/12 months both modes), ``n_replicates=4``, ``seed=0``. naive is
   very cheap (no wind_up pipeline) so it is recomputed fresh here rather than pulled from the reference
   run — which is what lets the grid extend to 1/2 months, where the frozen reference has no naive
   either. Only v0 is *not* recomputed (it is very slow and already frozen in the reference run; it is
   simply absent below 3 months).
2. **Merge** each fresh ``power_model`` + ``naive_ratio`` result with the ``v0_binned`` rows pulled
   from the reference overnight directory.
3. **Plot** a three-method campaign-length comparison (``naive_ratio``, ``v0_binned``,
   ``power_model``) per profile, and write merged per-profile + all-profiles leaderboards.

An alignment guard checks that the method-independent ground ``truth`` of the fresh cases equals the
reference run's truth on the campaign lengths they share; a mismatch there means the configs have
drifted and the merge would be comparing different cases, so it fails loudly. Fresh cases the
reference never covered (1/2 months) are expected and skipped by the guard.

**Benchmark regression tracking.** ``power_model``'s bias/spread/score per
``(mode, profile, campaign_months)`` at a known-good commit is frozen in a committed JSON benchmark
(``study_power_model_compare_baseline.json``, next to this script). Every run diffs the fresh
``power_model`` against it and logs a mean-over-profiles table of the deltas (spread/score: a
negative delta is an improvement; bias: a smaller ``|bias|`` is better) plus a per-cell
``benchmark_comparison_<mode>.csv``, so an attempt to improve ``power_model`` is scored objectively
against where it stands today.

**Accepting an improvement without a second sweep.** Every full sweep also drops a *candidate*
baseline (``<output-dir>/candidate_baseline.json``, seeded from the committed baseline so unrun modes
are preserved) — i.e. exactly what the committed file would become if you accept this run. If the run
looks good, promote it near-instantly with ``--accept-candidate`` (copies the candidate over the
committed JSON and exits, no re-run); then commit the JSON. ``--update-baseline`` still records
directly from a fresh sweep, but with the candidate flow you rarely need it.

**Per-bin before/after view.** Alongside the overall diff, the run also surfaces the *conditional*
before/after for the change under test on the two condition-dependent hard cases plus the placebo
(:data:`COVERED_PROFILES`): a per-bin ``|bias|`` table with a ``better``/``worse``/``~`` verdict
(``conditional_benchmark_comparison_<mode>.csv`` + log) and one overlay per ``(profile, condition)``
plotting truth vs the benchmark vs the current run
(``conditional_before_after_<profile>_<condition>.png``). The benchmark's per-bin curve is
reconstructed from its stored per-bin bias, so no benchmark-JSON change is needed. Both the overall
tally and the per-bin verdict use one materiality band (:data:`_MATERIAL_PP`).

Run from the repo root::

    uv run python -m benchmarking.baselines.study_power_model_compare \
        --reference-dir "~/temp/wind-up-benchmarking/badass overnight runs 30 June"

For fast feedback on a power_model change, restrict to one mode and one profile — e.g.
``--modes prepost --profiles cp_0pct`` fits a single case in ~minutes (vs ~30 for the full sweep) and
still emits its overall + per-bin before/after view. Use ``--skip-run`` to only re-merge/re-plot (and
re-diff the benchmark) from a previous ``power_model`` run under ``--output-dir`` (e.g. to tweak
plotting without re-fitting). Use ``--update-baseline`` to re-record the benchmark from the current
run (needs the full profile set — it cannot be combined with ``--profiles``), or ``--accept-candidate``
to promote the candidate a previous full sweep already wrote (no re-run).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarking.baselines.example_prepost_study import (
    DEFAULT_END_DT_EXCL,
    DEFAULT_START_DT,
    DEFAULT_TREATMENT_START_RANGE,
    DEFAULT_TURBINE_SUBSET,
    DEFAULT_WTG_NUMBERS,
    MIN_PRE_MONTHS,
    save_per_method_curve,
)
from benchmarking.baselines.example_toggle_study import DEFAULT_TOGGLE_PERIOD
from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.naive_ratio import NaiveRatioMethod
from benchmarking.baselines.overnight_profiles import overnight_profiles
from benchmarking.baselines.power_model import PowerModelMethod
from benchmarking.harness import (
    StudyConfig,
    conditional_leaderboard,
    leaderboard,
    plot_campaign_curves,
    plot_conditional_uplift,
    score_study,
)
from benchmarking.synthetic import HOT_COLUMNS, HOT_RATED_POWER_KW
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

logger = logging.getLogger(__name__)

# The three methods compared in the merged plots (oracle is dropped: it is only a truth anchor).
COMPARE_METHODS = ["naive_ratio", "v0_binned", "power_model"]
REUSED_METHODS = ["v0_binned"]  # only v0 is reused from the reference run; naive is recomputed fresh

# Issue 14: the out-of-the-box method is scored across the full 1-12-month range in both modes, so the
# short-campaign regime (F16) is part of every A/B. v0 is only frozen in the reference run at 3/6/(9)/12,
# so it is simply absent below 3 months — naive (recomputed fresh here) is the bar to beat there.
PREPOST_CAMPAIGN_MONTHS = [1, 2, 3, 6, 12]
TOGGLE_CAMPAIGN_MONTHS = [1, 2, 3, 6, 12]
N_REPLICATES = 4
SEED = 0

# Keys that identify one scored case independently of the method.
_CASE_KEYS = ["profile", "test_wtg", "campaign_months", "treatment_start"]
_DEFAULT_REFERENCE_DIR = Path.home() / "temp" / "wind-up-benchmarking" / "badass overnight runs 30 June"
_DEFAULT_OUTPUT_DIR = Path.home() / "temp" / "wind-up-benchmarking" / "power_model_compare"

# The committed power_model benchmark: its bias/spread/score per (mode, profile, campaign) frozen at
# a known-good commit, so future power_model changes are scored against it. Lives next to this script
# (tracked) — update it deliberately with --update-baseline when an improvement is accepted.
_BASELINE_PATH = Path(__file__).resolve().parent / "study_power_model_compare_baseline.json"
_BASELINE_SCHEMA = "power_model_compare_baseline_v2"
# Per-cell metrics recorded and diffed. spread/score: lower is better; bias: |bias| nearer 0 is better.
_METRIC_COLS = ["bias", "spread", "score"]
# Materiality band (percentage points) for the "did this change help/hurt/neutral" verdicts, shared by
# the overall tally() and the per-bin conditional table so the report speaks one language. A move whose
# magnitude is <= this reads neutral ("~"). 0.1 pp is well above floating-point noise (so an identical
# deterministic re-run still reads all-neutral) yet small enough to catch any change worth judging;
# hard-case per-bin biases run to tens of pp. tally() works on fractional deltas, so it uses the
# fractional form _MATERIAL_PP / 100.
_MATERIAL_PP = 0.1
_PP = 100.0  # fraction -> percentage points
# Profiles that get the per-bin before/after conditional view: the two condition-dependent hard cases
# plus the placebo (true uplift 0 in every bin — confirms a change adds no per-bin bias). The overall
# benchmark diff already covers all profiles; the other homogeneous cp_* have flat per-bin truth.
COVERED_PROFILES = ("cp_0pct", "ti_dependent_cp", "ws_dependent_cp")


def _prepost_study() -> StudyConfig:
    return StudyConfig(
        mode="prepost",
        turbine_subset=DEFAULT_TURBINE_SUBSET,
        treatment_start_range=DEFAULT_TREATMENT_START_RANGE,
        min_pre_months=MIN_PRE_MONTHS,
        campaign_months=PREPOST_CAMPAIGN_MONTHS,
        n_replicates=N_REPLICATES,
        seed=SEED,
    )


def _toggle_study() -> StudyConfig:
    return StudyConfig(
        mode="toggle",
        turbine_subset=DEFAULT_TURBINE_SUBSET,
        treatment_start_range=DEFAULT_TREATMENT_START_RANGE,
        min_pre_months=MIN_PRE_MONTHS,
        campaign_months=TOGGLE_CAMPAIGN_MONTHS,
        toggle_period=DEFAULT_TOGGLE_PERIOD,
        n_replicates=N_REPLICATES,
        seed=SEED,
    )


def _select_profiles(requested: list[str] | None) -> dict[str, list]:
    """Return the overnight profiles to score: all when ``requested`` is ``None``, else the named subset.

    A subset (e.g. ``["cp_0pct"]``) gives fast iteration on a power_model change — one case in ~minutes
    rather than the whole ~30-minute sweep. Unknown names fail loudly rather than silently scoring less.
    """
    all_profiles = overnight_profiles()
    if requested is None:
        return all_profiles
    unknown = [name for name in requested if name not in all_profiles]
    if unknown:
        msg = f"unknown profile(s) {unknown}; available: {sorted(all_profiles)}"
        raise ValueError(msg)
    return {name: all_profiles[name] for name in requested}


def _make_power_model(
    out_dir: Path, *, era5_hourly_df: pd.DataFrame, overrides: dict[str, Any] | None = None
) -> PowerModelMethod:
    """Construct the HoT-configured ``power_model`` method (its default: conditional uplift on).

    ``overrides`` are ``PowerModelMethod`` field overrides for candidate A/B runs (the Issue 9/10/11
    protocol), e.g. ``{"era5_derivations": ["gust_ratio"]}``; unknown field names fail loudly and
    JSON lists are coerced to the tuples the dataclass fields expect.
    """
    # Only data-schema description is passed here now: the accepted behaviour defaults (F13
    # availability_feature/era5_exclude, F14 min_child_samples) live on the PowerModelMethod class
    # (Issue 14), so a bare method already *is* the benchmarked config. reference_stat_cols stays
    # driver-level because it names a source-specific SCADA tag (wtc_ActPower_min) — schema
    # description, not tuning — so it belongs with the other HoT column names, not on the class.
    kwargs: dict[str, Any] = {
        "active_power_col": HOT_COLUMNS.active_power,
        "wind_speed_col": HOT_COLUMNS.wind_speed,
        "availability_col": HOT_COLUMNS.availability,
        "wind_speed_sd_col": HOT_COLUMNS.wind_speed_sd,
        "baseline_rated_power_kw": HOT_RATED_POWER_KW,
        "era5_hourly_df": era5_hourly_df,
        "reference_stat_cols": ("wtc_ActPower_min",),  # Issue 11 accepted feature (F12); a HoT SCADA tag
        "out_dir": out_dir / "power_model_runs",
    }
    if overrides:
        known = {f.name for f in dataclasses.fields(PowerModelMethod)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            msg = f"unknown PowerModelMethod field(s) in --method-overrides: {unknown}"
            raise ValueError(msg)
        kwargs.update({k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()})
    return PowerModelMethod(**kwargs)


def run_power_model(
    mode: str,
    out_dir: Path,
    *,
    profiles: list[str] | None = None,
    method_overrides: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Score ``power_model`` **and** ``naive_ratio`` over the overnight cases for one mode (no v0/oracle).

    ``profiles`` restricts to a subset of :func:`overnight_profiles` (default: all) for fast feedback on
    a power_model change. power_model runs at its default (conditional uplift on), so every case also
    emits the per-bin conditional cells the benchmark tracks. naive is recomputed fresh here (it is very
    cheap and has no wind_up pipeline) so the merge no longer depends on the reference run's naive rows —
    which matters at 1/2 months, where the frozen reference run has no naive either. Each fresh row still
    carries the harness's method-independent ground ``truth`` (so the merge's alignment guard has its
    cross-check against v0). Writes a per-profile ``results_*.csv`` and per-profile method curves under
    ``out_dir``, and returns the concatenated tidy results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scada_df, _ = load_hot_scada(
        start_dt=DEFAULT_START_DT,
        end_dt_excl=DEFAULT_END_DT_EXCL,
        wtg_numbers=DEFAULT_WTG_NUMBERS,
        wtg_names=DEFAULT_TURBINE_SUBSET,
    )
    context = build_hot_v0_context(wtg_names=DEFAULT_TURBINE_SUBSET)
    study = _prepost_study() if mode == "prepost" else _toggle_study()

    all_results = []
    for profile_name, profile in _select_profiles(profiles).items():
        # Per-profile subfolder: the method's run dir is power_model_<wtg>_<start>_<end> (no profile), so
        # profiles sharing a (wtg, window) would otherwise overwrite each other's non-timestamped diagnostics.
        method = _make_power_model(
            out_dir / profile_name,
            era5_hourly_df=context.reanalysis_datasets[0].data,
            overrides=method_overrides,
        )
        naive = NaiveRatioMethod(
            active_power_col=HOT_COLUMNS.active_power,
            availability_col=HOT_COLUMNS.availability,
            out_dir=out_dir / profile_name / "naive_runs",
        )
        logger.info("Scoring %s profile %s with power_model + naive_ratio", mode, profile_name)
        results = score_study(
            scada_df,
            profile=profile,
            methods=[naive, method],
            study=study,
            profile_name=profile_name,
            on_method_complete=partial(save_per_method_curve, out_dir, profile_name),
        )
        results.to_csv(out_dir / f"results_{profile_name}.csv", index=False)
        all_results.append(results)
    return pd.concat(all_results, ignore_index=True)


def _load_fresh_results(mode_out_dir: Path) -> pd.DataFrame:
    """Concatenate the per-profile ``results_*.csv`` a previous run wrote (for ``--skip-run``)."""
    files = sorted(mode_out_dir.glob("results_*.csv"))
    if not files:
        msg = f"no results_*.csv under {mode_out_dir}; run without --skip-run first."
        raise FileNotFoundError(msg)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _load_reference_methods(reference_mode_dir: Path, profiles: list[str], methods: list[str]) -> pd.DataFrame:
    """Load the requested methods' rows for the given profiles from the reference run directory."""
    frames = []
    for profile in profiles:
        path = reference_mode_dir / f"results_{profile}.csv"
        if not path.exists():
            msg = f"reference results missing for profile {profile!r}: {path}"
            raise FileNotFoundError(msg)
        df = pd.read_csv(path)
        frames.append(df[df["method"].isin(methods)])
    return pd.concat(frames, ignore_index=True)


def _case_key(df: pd.DataFrame) -> pd.Series:
    """Build a stable per-case string key (treatment_start normalised so str/Timestamp compare equal)."""
    ts = pd.to_datetime(df["treatment_start"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S%z")
    return (
        df["profile"].astype(str)
        + "|"
        + df["test_wtg"].astype(str)
        + "|"
        + df["campaign_months"].astype(int).astype(str)
        + "|"
        + ts
    )


def _check_alignment(fresh: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Fail loudly if the fresh and reference runs disagree on ground truth where they overlap.

    Ground ``truth`` is method-independent and deterministic in the study config + seed, so equal
    keys must carry equal truth. Since Issue 14 the fresh grid (1-12 months) extends past the frozen
    reference run (3/6/(9)/12), so fresh cases absent from the reference are expected (e.g. 1/2 months)
    and not an error — only the fresh ∩ reference intersection is truth-checked; a mismatch *there*
    still means the configs drifted and the merge would compare different cases.
    """
    f_overall = fresh[fresh["condition"] == "overall"].copy()
    r_overall = reference[reference["condition"] == "overall"].copy()
    f_truth = f_overall.assign(key=_case_key(f_overall)).groupby("key")["truth"].first()
    r_truth = r_overall.assign(key=_case_key(r_overall)).groupby("key")["truth"].first()

    common = f_truth.index.intersection(r_truth.index)
    bad = ~np.isclose(f_truth.loc[common].to_numpy(), r_truth.loc[common].to_numpy(), rtol=1e-6, atol=1e-9)
    if bad.any():
        example = common[bad][0]
        msg = (
            f"{int(bad.sum())} case(s) disagree on ground truth between the fresh power_model run and the "
            f"reference run (e.g. {example}: fresh={f_truth[example]:.6g} vs ref={r_truth[example]:.6g}). "
            f"The overnight config has drifted from the reference run; the merge would compare different cases."
        )
        raise ValueError(msg)
    logger.info("Alignment OK: %d cases match the reference run on ground truth.", len(common))


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


def power_model_leaderboard(fresh: pd.DataFrame) -> pd.DataFrame:
    """One row per (profile, campaign, condition, bin) of power_model bias/spread/score (overall incl.)."""
    pm = fresh[fresh["method"] == "power_model"]
    overall = leaderboard(pm).assign(condition="overall", condition_bin="overall")
    conditional = conditional_leaderboard(pm)
    cols = ["profile", "campaign_months", "condition", "condition_bin", *_METRIC_COLS]
    stacked = pd.concat([overall[cols], conditional[cols]], ignore_index=True)
    return stacked.sort_values(["profile", "campaign_months", "condition", "condition_bin"])


def record_baseline(
    lb_by_mode: dict[str, pd.DataFrame],
    study_by_mode: dict[str, StudyConfig],
    path: Path,
    *,
    seed_path: Path | None = None,
) -> None:
    """Write/refresh the benchmark for the given modes (other modes in the file are kept).

    Sibling modes not in ``lb_by_mode`` are inherited from ``seed_path`` (default: ``path`` itself). A
    *candidate* baseline is written by pointing ``path`` at the candidate file while seeding from the
    committed baseline, so the candidate equals what accepting this run would make the committed file.
    """
    seed = seed_path if seed_path is not None else path
    doc: dict[str, Any] = {"schema": _BASELINE_SCHEMA, "modes": {}}
    if seed.exists():
        loaded = json.loads(seed.read_text())
        if loaded.get("schema") == _BASELINE_SCHEMA:
            doc = loaded
            doc.setdefault("modes", {})
    commit = _git_commit()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for mode, lb in lb_by_mode.items():
        study = study_by_mode[mode]
        doc["modes"][mode] = {
            "recorded_utc": now,
            "git_commit": commit,
            "n_replicates": study.n_replicates,
            "seed": study.seed,
            "campaign_months": list(study.campaign_months),
            "profiles": sorted(lb["profile"].unique()),
            "cells": lb.round(8).to_dict(orient="records"),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    logger.info("Recorded power_model benchmark for %s at %s (commit %s)", list(lb_by_mode), path, commit)


def _candidate_path(output_dir: Path) -> Path:
    """Where a full sweep drops its candidate baseline (promote it with ``--accept-candidate``)."""
    return output_dir / "candidate_baseline.json"


def accept_candidate(candidate_path: Path, baseline_path: Path) -> None:
    """Promote a candidate baseline to the committed benchmark — a near-instant, no-sweep action.

    Every full sweep writes a candidate (see :func:`record_baseline`); this copies it over
    ``baseline_path`` so accepting an improvement never means re-running the ~30-minute sweep.
    """
    if not candidate_path.exists():
        msg = f"no candidate baseline at {candidate_path}; run a full sweep (all profiles) first to produce one."
        raise FileNotFoundError(msg)
    text = candidate_path.read_text()
    schema = json.loads(text).get("schema")
    if schema != _BASELINE_SCHEMA:
        msg = f"candidate {candidate_path} has schema {schema!r}, expected {_BASELINE_SCHEMA!r}; regenerate it."
        raise ValueError(msg)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(text)
    logger.info(
        "Promoted candidate %s -> committed baseline %s (modes %s). Commit the new JSON.",
        candidate_path,
        baseline_path,
        sorted(json.loads(text).get("modes", {})),
    )


def _load_baseline_cells(mode: str, path: Path) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Load the recorded benchmark cells (+ provenance) for ``mode``, or ``None`` if not recorded."""
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
    entry = doc.get("modes", {}).get(mode)
    if entry is None:
        return None
    return pd.DataFrame(entry["cells"]), entry


def _tally(delta: pd.Series, n_cells: int, *, threshold: float) -> str:
    """Count cells that moved beyond the materiality band: ``better`` (down) vs ``worse`` (up).

    ``delta`` is a fractional metric change; ``threshold`` is the band in the same fractional units.
    """
    better = int((delta < -threshold).sum())
    worse = int((delta > threshold).sum())
    return f"{better} better / {worse} worse (of {n_cells})"


def _verdict(d_abs_bias_pp: float, material_pp: float) -> str:
    """``better`` / ``worse`` / ``~`` for a per-bin |bias| change (pp) against the materiality band."""
    if d_abs_bias_pp < -material_pp:
        return "better"
    if d_abs_bias_pp > material_pp:
        return "worse"
    return "~"


_MERGE_KEYS = ["profile", "campaign_months", "condition", "condition_bin"]


def _covered_longest(fresh_cond_lb: pd.DataFrame) -> pd.DataFrame:
    """Fresh conditional rows restricted to the covered profiles' ws/ti bins at their longest campaign."""
    fresh = fresh_cond_lb[
        fresh_cond_lb["profile"].isin(COVERED_PROFILES) & fresh_cond_lb["condition"].isin(["ws", "ti"])
    ].copy()
    if fresh.empty:
        return fresh
    longest = fresh.groupby("profile")["campaign_months"].transform("max")
    return fresh[fresh["campaign_months"] == longest]


def conditional_before_after_table(
    fresh_cond_lb: pd.DataFrame, baseline_cells: pd.DataFrame, *, material_pp: float = _MATERIAL_PP
) -> pd.DataFrame:
    """Per-bin before/after table for the covered profiles at their longest campaign (values in pp).

    ``fresh_cond_lb`` is a fresh :func:`~benchmarking.harness.conditional_leaderboard` (carrying
    ``mean_estimate`` / ``mean_truth`` / ``bias`` per bin); ``baseline_cells`` is the recorded benchmark
    (per-bin ``bias`` only). The benchmark's per-bin estimate is reconstructed exactly as
    ``mean_truth + bias`` (ground truth is deterministic and alignment-guarded), so no benchmark-JSON
    schema change is needed. Returns one row per ``(profile, condition, condition_bin)`` with the
    reconstructed ``est_before``, fresh ``est_after``, ``|bias|`` before/after, their pp delta, and a
    ``better``/``worse``/``~`` verdict against ``material_pp``.
    """
    columns = [
        "profile", "condition", "condition_bin", "campaign_months",
        "mean_truth", "est_before", "est_after", "abs_bias_before", "abs_bias_after", "d_abs_bias", "verdict",
    ]  # fmt: skip
    fresh = _covered_longest(fresh_cond_lb)
    if fresh.empty:
        return pd.DataFrame(columns=columns)

    base = baseline_cells[[*_MERGE_KEYS, "bias"]].rename(columns={"bias": "bias_before"})
    merged = fresh.merge(base, on=_MERGE_KEYS, how="inner")

    out = pd.DataFrame(
        {
            "profile": merged["profile"],
            "condition": merged["condition"],
            "condition_bin": merged["condition_bin"],
            "campaign_months": merged["campaign_months"],
            "mean_truth": merged["mean_truth"] * _PP,
            "est_before": (merged["mean_truth"] + merged["bias_before"]) * _PP,
            "est_after": merged["mean_estimate"] * _PP,
            "abs_bias_before": merged["bias_before"].abs() * _PP,
            "abs_bias_after": merged["bias"].abs() * _PP,
        }
    )
    out["d_abs_bias"] = out["abs_bias_after"] - out["abs_bias_before"]
    out["verdict"] = out["d_abs_bias"].apply(lambda d: _verdict(d, material_pp))
    return out.sort_values(["profile", "condition", "condition_bin"]).reset_index(drop=True)


def _overlay_frame(fresh_cond_lb: pd.DataFrame, baseline_cells: pd.DataFrame) -> pd.DataFrame:
    """Two-"method" frame (benchmark + current) for :func:`plot_conditional_uplift`, in fractional units.

    The benchmark series' ``mean_estimate`` is reconstructed as ``mean_truth + bias`` and carries the
    benchmark's own per-bin ``spread``; the current series is the fresh estimate/spread. Both share the
    fresh ``mean_truth`` (the truth line). Empty if no covered-profile cells line up.
    """
    fresh = _covered_longest(fresh_cond_lb)
    plot_cols = ["profile", "condition", "condition_bin", "method", "mean_truth", "mean_estimate", "spread"]
    if fresh.empty:
        return pd.DataFrame(columns=plot_cols)
    base = baseline_cells[[*_MERGE_KEYS, "bias", "spread"]].rename(
        columns={"bias": "bias_before", "spread": "spread_before"}
    )
    merged = fresh.merge(base, on=_MERGE_KEYS, how="inner")
    shared = {c: merged[c] for c in ["profile", "condition", "condition_bin", "mean_truth"]}
    benchmark = pd.DataFrame(
        {**shared, "method": "power_model (benchmark)",
         "mean_estimate": merged["mean_truth"] + merged["bias_before"], "spread": merged["spread_before"]}
    )  # fmt: skip
    current = pd.DataFrame(
        {**shared, "method": "power_model (current)",
         "mean_estimate": merged["mean_estimate"], "spread": merged["spread"]}
    )  # fmt: skip
    return pd.concat([benchmark, current], ignore_index=True)[plot_cols]


def conditional_before_after(mode: str, fresh: pd.DataFrame, baseline_path: Path, comparison_dir: Path) -> None:
    """Per-bin before/after view for the covered profiles: a delta table (CSV + log) and overlay plots.

    Reads the recorded benchmark, computes the fresh power_model conditional leaderboard, and writes
    ``conditional_benchmark_comparison_<mode>.csv`` plus one
    ``conditional_before_after_<profile>_<condition>.png`` overlay (truth vs benchmark vs current) per
    covered ``(profile, condition)``. A no-op (with a warning) when no benchmark is recorded for ``mode``.
    """
    loaded = _load_baseline_cells(mode, baseline_path)
    if loaded is None:
        logger.warning(
            "No power_model benchmark for %s in %s — skipping the per-bin before/after view.",
            mode,
            baseline_path,
        )
        return
    base, prov = loaded
    cond_lb = conditional_leaderboard(fresh[fresh["method"] == "power_model"])

    table = conditional_before_after_table(cond_lb, base)
    if table.empty:
        logger.info("%s: no covered-profile conditional cells line up with the benchmark; nothing to show.", mode)
        return
    table.round(4).to_csv(comparison_dir / f"conditional_benchmark_comparison_{mode}.csv", index=False)
    logger.info(
        "%s power_model per-bin |bias| vs benchmark (recorded %s, commit %s) [pp], verdict band +/-%.3g pp:\n%s",
        mode,
        prov.get("recorded_utc", "?"),
        prov.get("git_commit", "?"),
        _MATERIAL_PP,
        table.round(3).to_string(index=False),
    )

    overlay = _overlay_frame(cond_lb, base)
    commit = prov.get("git_commit", "?")
    for (profile, condition), subset in overlay.groupby(["profile", "condition"]):
        plot_conditional_uplift(
            subset,
            condition=condition,
            save_path=comparison_dir / f"conditional_before_after_{profile}_{condition}.png",
            title=f"{mode} - {profile} power_model benchmark vs current vs {condition} (benchmark {commit})",
        )


def compare_to_benchmark(mode: str, lb: pd.DataFrame, baseline_path: Path, comparison_dir: Path) -> None:
    """Diff the fresh power_model bias/spread/score against the committed benchmark and report it.

    Writes a per-cell ``benchmark_comparison_<mode>.csv`` and logs a mean-over-profiles table with the
    deltas. spread/score: a negative delta is an improvement; bias: a smaller ``|bias|`` is better.
    """
    loaded = _load_baseline_cells(mode, baseline_path)
    if loaded is None:
        logger.warning(
            "No power_model benchmark recorded for %s in %s yet — run with --update-baseline to set it.",
            mode,
            baseline_path,
        )
        return
    base, prov = loaded
    base = base[base["profile"].isin(lb["profile"].unique())]  # scope to the profiles actually run
    merge_keys = ["profile", "campaign_months", "condition", "condition_bin"]
    merged = lb.merge(base, on=merge_keys, how="outer", suffixes=("", "_base"))
    for col in _METRIC_COLS:
        merged[f"d_{col}"] = merged[col] - merged[f"{col}_base"]
    merged["d_abs_bias"] = merged["bias"].abs() - merged["bias_base"].abs()
    merged.sort_values(["profile", "campaign_months", "condition", "condition_bin"]).to_csv(
        comparison_dir / f"benchmark_comparison_{mode}.csv", index=False
    )

    report = merged.groupby(["campaign_months", "condition"]).mean(numeric_only=True)
    overall = merged.mean(numeric_only=True).to_frame().T
    overall.index = ["ALL"]
    table = pd.concat([report, overall])
    show = pd.DataFrame(
        {
            "bias": table["bias"] * _PP,
            "Δbias": table["d_bias"] * _PP,
            "spread": table["spread"] * _PP,
            "Δspread": table["d_spread"] * _PP,
            "score": table["score"] * _PP,
            "Δscore": table["d_score"] * _PP,
        }
    ).round(3)
    # Only rows where both fresh and benchmark metrics are present can contribute to the tallies
    # (outer-merge leaves NaN deltas otherwise), so count those to keep "(of N)" consistent.
    metric_cols = _METRIC_COLS + [f"{col}_base" for col in _METRIC_COLS]
    n_cells = int(merged[metric_cols].notna().all(axis=1).sum())
    threshold = _MATERIAL_PP / _PP  # tally works on fractional deltas; the band is defined in pp.

    logger.info(
        "%s power_model vs benchmark (recorded %s, commit %s); mean over profiles [pp], "
        "Δ<0 = better for spread/score (|Δ| <= %.3g pp reads neutral):\n%s\n"
        "cells: spread %s; score %s; |bias| %s",
        mode,
        prov.get("recorded_utc", "?"),
        prov.get("git_commit", "?"),
        _MATERIAL_PP,
        show.to_string(),
        _tally(merged["d_spread"], n_cells, threshold=threshold),
        _tally(merged["d_score"], n_cells, threshold=threshold),
        _tally(merged["d_abs_bias"], n_cells, threshold=threshold),
    )


def _conditional_plot_subset(cond_lb: pd.DataFrame, profile: str, condition: str) -> pd.DataFrame:
    """Rows for one (profile, condition) at the longest campaign length only.

    ``conditional_leaderboard`` keeps one row per ``(method, profile, campaign_months, condition,
    condition_bin)``, but :func:`~benchmarking.harness.plots.plot_conditional_uplift` expects a
    single row per ``condition_bin`` per method. Collapse to the longest campaign (most data, so the
    cleanest per-bin estimate) rather than mixing campaign lengths into one line.
    """
    subset = cond_lb[(cond_lb["profile"] == profile) & (cond_lb["condition"] == condition)]
    if subset.empty:
        return subset
    longest = int(subset["campaign_months"].max())
    return subset[subset["campaign_months"] == longest]


def merge_and_plot(mode: str, fresh: pd.DataFrame, reference_mode_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Merge fresh power_model with reference v0/naive, write merged tables + per-profile plots."""
    comparison_dir = out_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    profiles = sorted(fresh["profile"].unique())
    reference = _load_reference_methods(reference_mode_dir, profiles, REUSED_METHODS)
    _check_alignment(fresh, reference)

    # v0 comes from the frozen reference run (slow, reference-only, absent below 3 months); power_model
    # and the freshly-recomputed naive come from this run. Concatenating both gives the three-method
    # comparison, with v0 simply missing at the campaign lengths the reference never covered.
    fresh_compare = fresh[fresh["method"].isin(COMPARE_METHODS) & (fresh["method"] != "v0_binned")]
    merged = pd.concat([reference, fresh_compare], ignore_index=True)
    merged = merged[merged["method"].isin(COMPARE_METHODS)]

    merged.to_csv(comparison_dir / f"merged_results_{mode}.csv", index=False)
    for profile in profiles:
        prof_rows = merged[merged["profile"] == profile]
        summary = leaderboard(prof_rows)
        summary.to_csv(comparison_dir / f"leaderboard_{profile}.csv", index=False)
        plot_campaign_curves(
            summary,
            save_path=comparison_dir / f"campaign_curves_{profile}.png",
            title=f"{mode} - {profile} (naive vs v0 vs power_model)",
        )

    all_summary = leaderboard(merged)
    all_summary.to_csv(comparison_dir / f"leaderboard_all_profiles_{mode}.csv", index=False)
    logger.info(
        "%s comparison (all profiles):\n%s",
        mode,
        all_summary[["method", "profile", "campaign_months", "bias", "spread", "score"]].to_string(index=False),
    )

    # Conditional comparison plots for power_model (v0/naive emit no conditional rows — expected).
    pm_rows = merged[merged["method"] == "power_model"]
    if not pm_rows.empty and "condition" in pm_rows.columns:
        cond_lb = conditional_leaderboard(pm_rows)
        if not cond_lb.empty:
            for profile in profiles:
                for condition in cond_lb["condition"].unique():
                    subset = _conditional_plot_subset(cond_lb, profile, condition)
                    if subset.empty:
                        continue
                    longest = int(subset["campaign_months"].iloc[0])
                    plot_conditional_uplift(
                        subset,
                        condition=condition,
                        save_path=comparison_dir / f"conditional_{profile}_{condition}.png",
                        title=f"{mode} - {profile} power_model vs {condition} ({longest}mo)",
                    )

    logger.info("Wrote %s comparison outputs to %s", mode, comparison_dir)
    return merged


def main() -> None:
    """Run power_model over the overnight cases for each mode, then merge + plot vs v0/naive."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=_DEFAULT_REFERENCE_DIR,
        help="overnight run dir holding prepost/ and toggle/ with the frozen v0 + naive results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="where fresh power_model runs and the merged comparison are written",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["prepost", "toggle"],
        default=["prepost", "toggle"],
        help="which mode(s) to run (default: both)",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(overnight_profiles()),
        default=None,
        help="restrict to a subset of overnight profiles for fast feedback (default: all seven). "
        "e.g. --profiles cp_0pct runs just the placebo. Cannot be combined with --update-baseline.",
    )
    parser.add_argument(
        "--method-overrides",
        type=str,
        default=None,
        help="JSON dict of PowerModelMethod field overrides for a candidate A/B run, e.g. "
        '\'{"era5_derivations": ["gust_ratio"]}\'. The run is diffed against the benchmark as usual '
        "but no candidate baseline is written (the overridden config is not the committed default).",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="reuse a previous power_model run under --output-dir; only re-merge and re-plot",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=_BASELINE_PATH,
        help="the committed power_model benchmark JSON to diff against (and to --update-baseline)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="overwrite the recorded benchmark for the run modes with this run (do this deliberately, "
        "only when an improvement is accepted); without it, the run is diffed against the benchmark",
    )
    parser.add_argument(
        "--accept-candidate",
        action="store_true",
        help="promote the candidate baseline written by the last full sweep (under --output-dir) to the "
        "committed --baseline-path and exit — a near-instant accept with no re-run. Then commit the JSON.",
    )
    args = parser.parse_args()
    if args.profiles is not None and args.update_baseline:
        # record_baseline rewrites a mode's cells wholesale, so a subset run would drop the other
        # profiles from the committed benchmark. Refuse rather than silently corrupt it.
        parser.error("--update-baseline needs the full profile set; do not combine it with --profiles")
    method_overrides: dict[str, Any] | None = json.loads(args.method_overrides) if args.method_overrides else None
    if method_overrides is not None and args.update_baseline:
        # the benchmark freezes the *committed default* config; an overridden run must not be recorded as it
        parser.error("--update-baseline records the default config; do not combine it with --method-overrides")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    reference_dir = args.reference_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    baseline_path = args.baseline_path.expanduser()

    if args.accept_candidate:
        accept_candidate(_candidate_path(output_dir), baseline_path)
        return

    lb_by_mode: dict[str, pd.DataFrame] = {}
    study_by_mode: dict[str, StudyConfig] = {}
    for mode in args.modes:
        logger.info("=== %s ===", mode.upper())
        mode_out_dir = output_dir / mode
        if args.skip_run:
            fresh = _load_fresh_results(mode_out_dir)
            if args.profiles is not None:
                fresh = fresh[fresh["profile"].isin(args.profiles)]
            logger.info("Reusing %d fresh rows from %s", len(fresh), mode_out_dir)
        else:
            fresh = run_power_model(mode, mode_out_dir, profiles=args.profiles, method_overrides=method_overrides)
            if method_overrides is not None:  # provenance: which A/B candidate this run was
                (mode_out_dir / "method_overrides.json").write_text(json.dumps(method_overrides, indent=2) + "\n")
        merge_and_plot(mode, fresh, reference_dir / mode, mode_out_dir)
        lb_by_mode[mode] = power_model_leaderboard(fresh)
        study_by_mode[mode] = _prepost_study() if mode == "prepost" else _toggle_study()
        if not args.update_baseline:
            compare_to_benchmark(mode, lb_by_mode[mode], baseline_path, mode_out_dir / "comparison")
            conditional_before_after(mode, fresh, baseline_path, mode_out_dir / "comparison")

    if args.update_baseline:
        record_baseline(lb_by_mode, study_by_mode, baseline_path)
    elif method_overrides is not None:
        logger.info("Overridden run (--method-overrides) — no candidate baseline written (not the default config).")
    elif args.profiles is None:
        # A full sweep: drop a candidate baseline (seeded from committed) so an accepted improvement is a
        # near-instant `--accept-candidate` away, with no second sweep. Subset runs are incomplete -> skip.
        candidate = _candidate_path(output_dir)
        record_baseline(lb_by_mode, study_by_mode, candidate, seed_path=baseline_path)
        logger.info("Candidate baseline written to %s — accept it with --accept-candidate (no re-run).", candidate)
    else:
        logger.info("Subset run (--profiles) — no candidate baseline written (it would be incomplete).")

    logger.info("All done. Comparison plots under %s/<mode>/comparison/", output_dir)


if __name__ == "__main__":
    main()
