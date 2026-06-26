"""Inspect one hard PREPOST case across every method, with plots on.

A focused investigation driver (findings F1): replay the **exact** overnight prepost study
draws (same ``StudyConfig``/seed/profile), pin a single hard ``(test_wtg, campaign_months)``
run, and execute naive + R-learner (+ v0) on the **identical** ``MethodInput`` with
``save_plots=True``, each into its own subfolder of one timestamped run dir. The default case is
``cp_0pct`` (placebo) on ``T07`` at 6 months — where the R-learner is badly biased (~-14%) while
naive (~+2%) and v0 (~0%) are fine — so the per-method diagnostics can be eyeballed side by side
to understand *why*.

Because the draws are a deterministic function of ``(StudyConfig, seed)`` and the harness builds
one ``MethodInput`` per ``(replicate, window)``, every method here sees the same data the
overnight run scored — only the estimate differs.

Run it::

    uv run python -m benchmarking.baselines.inspect_prepost_hard_case

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``inspect_hard_case``/``<timestamp>/``
(``naive/``, ``rlearner/``, ``v0/`` run folders, a ``comparison_summary.csv`` and ``run.log``).
The first run downloads + caches the Hill of Towie SCADA (Zenodo) and ERA5 (Open-Meteo, ``era5``
group); the ``ml`` group is needed for the R-learner and v0 needs the wind_up pipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.baselines.example_prepost_study import (
    DEFAULT_END_DT_EXCL,
    DEFAULT_START_DT,
    DEFAULT_TREATMENT_START_RANGE,
    DEFAULT_TURBINE_SUBSET,
    DEFAULT_WTG_NUMBERS,
    MIN_PRE_MONTHS,
    default_output_root,
)
from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.naive_ratio import NaiveRatioMethod
from benchmarking.baselines.overnight_common import start_overnight_run
from benchmarking.baselines.overnight_profiles import overnight_profiles
from benchmarking.baselines.rlearner import RLearnerMethod
from benchmarking.baselines.v0_binned import V0BinnedMethod
from benchmarking.harness import (
    Method,
    MethodInput,
    StudyConfig,
    build_replicates,
    campaign_windows,
    treated_activity_mask,
    window_row_mask,
)
from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

if TYPE_CHECKING:
    from benchmarking.harness.replicates import Replicate

logger = logging.getLogger(__name__)

# Match the overnight prepost study exactly so the pinned run is the one its leaderboard scored.
N_REPLICATES = 4
CAMPAIGN_MONTHS = [3, 6, 12]

# The default hard case (see module docstring / findings F1).
DEFAULT_PROFILE = "cp_0pct"
DEFAULT_TEST_WTG = "T07"
DEFAULT_CAMPAIGN_MONTHS = 6


def _overnight_study() -> StudyConfig:
    """Return the prepost ``StudyConfig`` the overnight run used (so draws are bit-identical)."""
    return StudyConfig(
        mode="prepost",
        turbine_subset=DEFAULT_TURBINE_SUBSET,
        treatment_start_range=DEFAULT_TREATMENT_START_RANGE,
        min_pre_months=MIN_PRE_MONTHS,
        campaign_months=CAMPAIGN_MONTHS,
        n_replicates=N_REPLICATES,
        seed=0,
    )


def _select_replicate(replicates: list[Replicate], test_wtg: str) -> Replicate:
    """Return the first replicate drawn for ``test_wtg`` (logging the full draw for transparency)."""
    for rep in replicates:
        logger.info(
            "replicate %d: test_wtg=%s treatment_start=%s",
            rep.replicate_id,
            rep.test_wtg,
            pd.Timestamp(rep.treatment_start).date(),
        )
    matches = [rep for rep in replicates if rep.test_wtg == test_wtg]
    if not matches:
        drawn = sorted({rep.test_wtg for rep in replicates})
        msg = f"no replicate drawn for test_wtg {test_wtg!r}; drawn turbines were {drawn}"
        raise ValueError(msg)
    return matches[0]


def _pin_case(
    scada_df: pd.DataFrame, *, study: StudyConfig, profile_name: str, test_wtg: str, campaign_months: int
) -> tuple[Replicate, MethodInput, float]:
    """Build the pinned replicate, its shared ``MethodInput`` and the ground-truth uplift."""
    profile = overnight_profiles()[profile_name]
    replicates = build_replicates(scada_df, profile=profile, study=study)
    rep = _select_replicate(replicates, test_wtg)

    windows = campaign_windows(
        rep.treatment_start,
        min_pre_months=study.min_pre_months,
        campaign_months=study.campaign_months,
        data_start=scada_df.index.min(),
        data_end=scada_df.index.max(),
    )
    matching = [w for w in windows if w.months == campaign_months]
    if not matching:
        available = [w.months for w in windows]
        msg = f"campaign_months={campaign_months} not feasible for replicate {rep.replicate_id}; available {available}"
        raise ValueError(msg)
    window = matching[0]

    syn = rep.synthetic_df
    mi = MethodInput(
        scada_df=syn.loc[window_row_mask(syn.index, window)],
        test_wtg=rep.test_wtg,
        upgrade_timing=rep.upgrade_timing,
        turbine_col=HOT_COLUMNS.turbine,
    )
    test_index = syn.loc[syn[HOT_COLUMNS.turbine] == rep.test_wtg].index
    truth = rep.true_uplift(mask=treated_activity_mask(test_index, rep.upgrade_timing, window=window)).overall
    logger.info(
        "pinned case: profile=%s test_wtg=%s treatment_start=%s campaign_months=%d window=[%s, %s) truth=%+.3f%%",
        profile_name,
        rep.test_wtg,
        pd.Timestamp(rep.treatment_start).date(),
        window.months,
        window.baseline_start,
        window.activity_end,
        100 * truth,
    )
    return rep, mi, truth


def _build_methods(out_dir: Path, *, include_v0: bool) -> list[Method]:
    """Return the methods to inspect, each writing diagnostics (plots on) into its own subfolder."""
    context = build_hot_v0_context(wtg_names=DEFAULT_TURBINE_SUBSET)
    methods: list[Method] = [
        NaiveRatioMethod(
            active_power_col=HOT_COLUMNS.active_power,
            availability_col=HOT_COLUMNS.availability,
            out_dir=out_dir / "naive",
            save_plots=True,
        ),
        RLearnerMethod(
            active_power_col=HOT_COLUMNS.active_power,
            wind_speed_col=HOT_COLUMNS.wind_speed,
            availability_col=HOT_COLUMNS.availability,
            era5_hourly_df=context.reanalysis_datasets[0].data,
            out_dir=out_dir / "rlearner",
            save_plots=True,
        ),
    ]
    if include_v0:
        methods.append(V0BinnedMethod(context, scratch_dir=out_dir / "v0", save_plots=True))
    return methods


def _run_methods(methods: list[Method], *, mi: MethodInput, truth: float) -> pd.DataFrame:
    """Run every method on the identical input; return a tidy estimate/error/wall-time comparison."""
    rows = []
    for method in methods:
        start = time.perf_counter()
        estimate = method.estimate(mi).p50_overall
        wall_time_s = time.perf_counter() - start
        logger.info(
            "%-12s estimate %+.3f%%  truth %+.3f%%  error %+.3f%%  (%.1fs)",
            method.name,
            100 * estimate,
            100 * truth,
            100 * (estimate - truth),
            wall_time_s,
        )
        rows.append(
            {
                "method": method.name,
                "estimate": estimate,
                "truth": truth,
                "signed_error": estimate - truth,
                "wall_time_s": wall_time_s,
            }
        )
    return pd.DataFrame(rows)


def inspect_prepost_hard_case(
    *,
    profile_name: str = DEFAULT_PROFILE,
    test_wtg: str = DEFAULT_TEST_WTG,
    campaign_months: int = DEFAULT_CAMPAIGN_MONTHS,
    out_root: str | Path | None = None,
    include_v0: bool = True,
) -> pd.DataFrame:
    """Run every method on one pinned hard prepost case with plots on; return the comparison frame.

    :param profile_name: an :func:`overnight_profiles` key (default ``cp_0pct``, the placebo)
    :param test_wtg: the test turbine to pin (must be one of the overnight draws)
    :param campaign_months: which campaign length of the pinned replicate to inspect
    :param out_root: base output dir; defaults to :func:`default_output_root`'s parent
    :param include_v0: also run the slow v0 baseline (a full wind_up run for the campaign)
    :return: per-method ``estimate``/``truth``/``signed_error``/``wall_time_s``
    """
    study = _overnight_study()
    output_root = Path(out_root) if out_root is not None else default_output_root().parent
    out_dir = start_overnight_run("inspect_hard_case", study, output_root=output_root)

    logger.info("Loading Hill of Towie SCADA %s..%s", DEFAULT_START_DT, DEFAULT_END_DT_EXCL)
    scada_df, _ = load_hot_scada(
        start_dt=DEFAULT_START_DT,
        end_dt_excl=DEFAULT_END_DT_EXCL,
        wtg_numbers=DEFAULT_WTG_NUMBERS,
        wtg_names=DEFAULT_TURBINE_SUBSET,
    )

    _rep, mi, truth = _pin_case(
        scada_df, study=study, profile_name=profile_name, test_wtg=test_wtg, campaign_months=campaign_months
    )
    methods = _build_methods(out_dir, include_v0=include_v0)
    summary = _run_methods(methods, mi=mi, truth=truth)

    summary_path = out_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %s\n%s", summary_path, summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    inspect_prepost_hard_case(include_v0=False)
