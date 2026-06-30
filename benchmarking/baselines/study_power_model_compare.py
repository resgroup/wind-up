"""Re-run ``power_model`` over the overnight cases and compare it against the existing v0 + naive.

Iterative-development helper. The overnight v0 runs are very slow, so they are computed once
(:mod:`benchmarking.baselines.study_overnight_prepost` / ``study_overnight_toggle`` with
``include_v0=True``) and kept on disk. As ``power_model`` is improved you only want to re-run the
cheap method and re-draw the comparison plots, reusing the frozen v0 (and naive) numbers. That is
what this script does:

1. **Run** ``power_model`` **only** over the *exact* current overnight prepost + toggle configs —
   same seven :func:`~benchmarking.baselines.overnight_profiles.overnight_profiles`, same campaign
   grids, ``n_replicates=4``, ``seed=0`` — so every ``(profile, turbine, treatment-start, campaign)``
   case matches the frozen overnight run. v0 and naive are *not* recomputed (v0 is very slow and
   both are already frozen in the reference run).
2. **Merge** each fresh ``power_model`` result with the ``v0_binned`` and ``naive_ratio`` rows
   pulled from the reference overnight directory.
3. **Plot** a three-method campaign-length comparison (``naive_ratio``, ``v0_binned``,
   ``power_model``) per profile, and write merged per-profile + all-profiles leaderboards.

An alignment guard checks that the method-independent ground ``truth`` of every fresh case equals
the reference run's truth for the same key; a mismatch means the configs have drifted and the
merge would be comparing different cases, so it fails loudly rather than plotting nonsense. (The
``truth`` column the harness attaches to every ``power_model`` row is itself the cross-check, so
re-running the oracle/naive anchors purely to validate the line-up is unnecessary.)

Run from the repo root::

    uv run python -m benchmarking.baselines.study_power_model_compare \
        --reference-dir "~/temp/wind-up-benchmarking/badass overnight runs 30 June"

Use ``--skip-run`` to only re-merge/re-plot from a previous ``power_model`` run under
``--output-dir`` (e.g. to tweak plotting without re-fitting). Use ``--modes prepost`` /
``--modes toggle`` to restrict to one mode.
"""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path

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
from benchmarking.baselines.overnight_profiles import overnight_profiles
from benchmarking.baselines.power_model import PowerModelMethod
from benchmarking.harness import StudyConfig, leaderboard, plot_campaign_curves, score_study
from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

logger = logging.getLogger(__name__)

# The three methods compared in the merged plots (oracle is dropped: it is only a truth anchor).
COMPARE_METHODS = ["naive_ratio", "v0_binned", "power_model"]
REUSED_METHODS = ["naive_ratio", "v0_binned"]  # taken from the reference run, not recomputed

# Match study_overnight_prepost.py / study_overnight_toggle.py exactly.
PREPOST_CAMPAIGN_MONTHS = [3, 6, 12]
TOGGLE_CAMPAIGN_MONTHS = [3, 6, 9, 12]
N_REPLICATES = 4
SEED = 0

# Keys that identify one scored case independently of the method.
_CASE_KEYS = ["profile", "test_wtg", "campaign_months", "treatment_start"]
_DEFAULT_REFERENCE_DIR = Path.home() / "temp" / "wind-up-benchmarking" / "badass overnight runs 30 June"
_DEFAULT_OUTPUT_DIR = Path.home() / "temp" / "wind-up-benchmarking" / "power_model_compare"


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


def run_power_model(mode: str, out_dir: Path) -> pd.DataFrame:
    """Score **only** ``power_model`` over the overnight cases for one mode (no v0/naive/oracle).

    Each fresh row still carries the harness's method-independent ground ``truth`` (so the merge's
    alignment guard has its cross-check without recomputing any anchor). Writes a per-profile
    ``results_*.csv`` and per-profile ``power_model`` curve under ``out_dir`` (the latter lets a long
    run be sanity-checked profile-by-profile), and returns the concatenated tidy results.
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
    for profile_name, profile in overnight_profiles().items():
        method = PowerModelMethod(
            active_power_col=HOT_COLUMNS.active_power,
            wind_speed_col=HOT_COLUMNS.wind_speed,
            availability_col=HOT_COLUMNS.availability,
            era5_hourly_df=context.reanalysis_datasets[0].data,
            out_dir=out_dir / "power_model_runs",
        )
        logger.info("Scoring %s profile %s with power_model", mode, profile_name)
        results = score_study(
            scada_df,
            profile=profile,
            methods=[method],
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
    """Fail loudly if the fresh cases do not line up case-for-case with the reference run.

    Ground ``truth`` is method-independent and deterministic in the study config + seed, so equal
    keys must carry equal truth. Mismatched keys or truths mean the configs drifted.
    """
    f_overall = fresh[fresh["condition"] == "overall"].copy()
    r_overall = reference[reference["condition"] == "overall"].copy()
    f_truth = f_overall.assign(key=_case_key(f_overall)).groupby("key")["truth"].first()
    r_truth = r_overall.assign(key=_case_key(r_overall)).groupby("key")["truth"].first()

    missing = sorted(set(f_truth.index) - set(r_truth.index))
    if missing:
        msg = f"{len(missing)} fresh case(s) have no reference match, e.g. {missing[:3]}"
        raise ValueError(msg)
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


def merge_and_plot(mode: str, fresh: pd.DataFrame, reference_mode_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Merge fresh power_model with reference v0/naive, write merged tables + per-profile plots."""
    comparison_dir = out_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    profiles = sorted(fresh["profile"].unique())
    reference = _load_reference_methods(reference_mode_dir, profiles, REUSED_METHODS)
    _check_alignment(fresh, reference)

    power_model = fresh[fresh["method"] == "power_model"]
    merged = pd.concat([reference, power_model], ignore_index=True)
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
        "--skip-run",
        action="store_true",
        help="reuse a previous power_model run under --output-dir; only re-merge and re-plot",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    reference_dir = args.reference_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    for mode in args.modes:
        logger.info("=== %s ===", mode.upper())
        mode_out_dir = output_dir / mode
        if args.skip_run:
            fresh = _load_fresh_results(mode_out_dir)
            logger.info("Reusing %d fresh rows from %s", len(fresh), mode_out_dir)
        else:
            fresh = run_power_model(mode, mode_out_dir)
        merge_and_plot(mode, fresh, reference_dir / mode, mode_out_dir)

    logger.info("All done. Comparison plots under %s/<mode>/comparison/", output_dir)


if __name__ == "__main__":
    main()
