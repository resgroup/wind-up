"""Manual inspection driver: run a few v0 replicates with wind_up plots saved to file.

A runnable companion to the ``slow`` end-to-end test
(``tests/benchmarking/baselines/test_v0_end_to_end.py``): same real-data setup, but instead of
asserting it runs a handful of replicates with ``save_plots=True``, each in its **own output
directory**, so the full set of wind_up diagnostic plots (power curves, detrend, data coverage,
pre/post comparisons, ...) can be eyeballed to confirm the baseline is wired correctly. It also
writes an ``inspection_summary.csv`` of recovered P50 vs injected truth per replicate.

Each replicate is built and scored exactly as :func:`benchmarking.harness.score_study` would
(same windowing and ground truth), so what you inspect is the real scored path.

Run it::

    uv run python -m benchmarking.baselines.inspect_v0_run

First run downloads and caches the Hill of Towie v2 SCADA (Zenodo) and ERA5 (Open-Meteo; needs
the ``era5`` optional dependency group). Each replicate's plots land under
``<out_root>/inspection/replicate_<id>_<wtg>/v0_<wtg>_<dates>/plots``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from benchmarking.baselines.example_v0_study import default_output_root
from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.v0_binned import V0BinnedMethod
from benchmarking.harness import (
    MethodInput,
    StudyConfig,
    build_replicates,
    campaign_windows,
    treated_activity_mask,
    window_row_mask,
)
from benchmarking.synthetic import ConstantCpChange
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada
from wind_up.constants import DataColumns

logger = logging.getLogger(__name__)

# The full 2016-2020 stable window, same as the driver: treatment drawn per replicate from
# 2018-01-01..2020-01-01, with a 24-month baseline (so the earliest upgrade's baseline starts at
# the 2016-01-01 data start and v0's detrend is fully covered).
DEFAULT_START_DT = pd.Timestamp("2016-01-01", tz="UTC")
DEFAULT_END_DT_EXCL = pd.Timestamp("2021-01-01", tz="UTC")
DEFAULT_WTG_NUMBERS = [1, 3, 4, 7]
DEFAULT_TURBINE_SUBSET = [f"T{x:02d}" for x in DEFAULT_WTG_NUMBERS]
DEFAULT_TREATMENT_START_RANGE = (pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2019-12-31 23:50", tz="UTC"))
MIN_PRE_MONTHS = 24


def inspect_v0_run(
    *,
    out_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    n_replicates: int = 3,
    delta: float = 0.05,
    campaign_months: list[int] | None = None,
) -> pd.DataFrame:
    """Run ``n_replicates`` v0 estimates with plots saved per replicate; return a summary frame.

    :param out_root: output directory; defaults to :func:`default_output_root` / ``inspection``
    :param data_dir: Hill of Towie data/cache dir; defaults to the source package default
    :param n_replicates: how many replicate datasets to run (each in its own out dir)
    :param delta: injected constant-Cp uplift fraction
    :param campaign_months: campaign length(s); the longest is used for each replicate's plots
    :return: per-replicate frame of test_wtg, treatment_start, campaign_months, estimate, truth, signed_error
    """
    out_dir = (Path(out_root) if out_root is not None else default_output_root()) / "inspection"
    out_dir.mkdir(parents=True, exist_ok=True)
    campaign_months = campaign_months if campaign_months is not None else [6]

    logger.info("Loading Hill of Towie SCADA %s..%s", DEFAULT_START_DT, DEFAULT_END_DT_EXCL)
    context = build_hot_v0_context(wtg_names=DEFAULT_TURBINE_SUBSET)
    scada_df, _metadata_df = load_hot_scada(
        start_dt=DEFAULT_START_DT,
        end_dt_excl=DEFAULT_END_DT_EXCL,
        wtg_numbers=DEFAULT_WTG_NUMBERS,
        wtg_names=DEFAULT_TURBINE_SUBSET,
        data_dir=Path(data_dir) if data_dir is not None else None,
    )
    study = StudyConfig(
        mode="prepost",
        turbine_subset=DEFAULT_TURBINE_SUBSET,
        treatment_start_range=DEFAULT_TREATMENT_START_RANGE,
        min_pre_months=MIN_PRE_MONTHS,
        campaign_months=campaign_months,
        n_replicates=n_replicates,
        seed=0,
    )
    replicates = build_replicates(scada_df, profile=[ConstantCpChange(delta=delta)], study=study)

    rows = []
    for rep in replicates:
        windows = campaign_windows(
            rep.treatment_start,
            min_pre_months=study.min_pre_months,
            campaign_months=study.campaign_months,
            data_start=scada_df.index.min(),
            data_end=scada_df.index.max(),
        )
        if not windows:
            logger.warning("replicate %d (%s): no feasible campaign window, skipping", rep.replicate_id, rep.test_wtg)
            continue
        window = windows[-1]  # the longest campaign -> richest plots

        syn = rep.synthetic_df
        mi = MethodInput(
            scada_df=syn.loc[window_row_mask(syn.index, window)],
            test_wtg=rep.test_wtg,
            upgrade_timing=rep.upgrade_timing,
        )
        rep_dir = out_dir / f"replicate_{rep.replicate_id:02d}_{rep.test_wtg}"
        method = V0BinnedMethod(context, scratch_dir=rep_dir, save_plots=True)
        estimate = method.estimate(mi).p50_overall

        test_index = syn.loc[syn[DataColumns.turbine_name] == rep.test_wtg].index
        truth = rep.true_uplift(mask=treated_activity_mask(test_index, rep.upgrade_timing, window=window)).overall

        logger.info(
            "replicate %d (%s, start %s, %d mo): estimate %+.2f%%, truth %+.2f%%, error %+.2f%% -> %s",
            rep.replicate_id,
            rep.test_wtg,
            pd.Timestamp(rep.treatment_start).date(),
            window.months,
            100 * estimate,
            100 * truth,
            100 * (estimate - truth),
            rep_dir,
        )
        rows.append(
            {
                "replicate_id": rep.replicate_id,
                "test_wtg": rep.test_wtg,
                "treatment_start": rep.treatment_start,
                "campaign_months": window.months,
                "estimate": estimate,
                "truth": truth,
                "signed_error": estimate - truth,
                "out_dir": str(rep_dir),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "inspection_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %s\n%s", summary_path, summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    inspect_v0_run()
