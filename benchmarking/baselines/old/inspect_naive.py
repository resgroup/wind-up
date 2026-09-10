"""Manual inspection driver: run a few naive-ratio replicates (prepost and toggle) with plots on.

A runnable companion to :mod:`benchmarking.baselines.old.inspect_v0_run`, but for
:class:`benchmarking.baselines.naive_ratio.NaiveRatioMethod`. It runs a handful of replicates in
each mode with ``save_plots=True``, each in its **own output directory**, so the naive method's
diagnostics (the per-run data-stats / results CSVs and the scatter, ratio-timeseries and
used-data-coverage plots) can
be eyeballed to confirm it received and interpreted the data correctly. The naive method has no
wind_up dependency, so no v0 context / metadata is needed.

Each replicate is built and scored exactly as :func:`benchmarking.harness.score_study` would
(same windowing and ground truth), so what you inspect is the real scored path.

Run it::

    uv run python -m benchmarking.baselines.old.inspect_naive

First run downloads and caches the Hill of Towie v2 SCADA (Zenodo). Each replicate's outputs land
under ``<out_root>/inspection_naive/<mode>/replicate_<id>_<wtg>/naive_<wtg>_<dates>/`` (with a
``plots`` subfolder).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from benchmarking.baselines.example_prepost_study import default_output_root
from benchmarking.baselines.naive_ratio import NaiveRatioMethod
from benchmarking.harness import (
    MethodInput,
    StudyConfig,
    build_replicates,
    campaign_windows,
    treated_activity_mask,
    window_row_mask,
)
from benchmarking.synthetic import HOT_COLUMNS, ConstantCpChange
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

logger = logging.getLogger(__name__)

# Same 2016-2020 stable window as inspect_v0_run: treatment drawn per replicate from
# 2018-01-01..2019-12-31, with a 24-month baseline.
DEFAULT_START_DT = pd.Timestamp("2016-01-01", tz="UTC")
DEFAULT_END_DT_EXCL = pd.Timestamp("2021-01-01", tz="UTC")
DEFAULT_WTG_NUMBERS = [1, 3, 4, 7]
DEFAULT_TURBINE_SUBSET = [f"T{x:02d}" for x in DEFAULT_WTG_NUMBERS]
DEFAULT_TREATMENT_START_RANGE = (pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2019-12-31 23:50", tz="UTC"))
MIN_PRE_MONTHS = 24
# 20 minutes on, 20 minutes off -> a 40-minute on/off cycle.
DEFAULT_TOGGLE_PERIOD = pd.Timedelta(minutes=40)


def _inspect_mode(scada_df: pd.DataFrame, *, study: StudyConfig, out_dir: Path, delta: float) -> list[dict]:
    """Run every replicate of ``study`` with the naive method (plots on) and return summary rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
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
        window = windows[-1]  # the longest campaign -> most data to eyeball

        syn = rep.synthetic_df
        mi = MethodInput(
            scada_df=syn.loc[window_row_mask(syn.index, window)],
            test_wtg=rep.test_wtg,
            upgrade_timing=rep.upgrade_timing,
            turbine_col=HOT_COLUMNS.turbine,
        )
        rep_dir = out_dir / f"replicate_{rep.replicate_id:02d}_{rep.test_wtg}"
        estimate = (
            NaiveRatioMethod(
                columns=HOT_COLUMNS,
                out_dir=rep_dir,
                save_plots=True,
            )
            .estimate(mi)
            .p50_overall
        )

        test_index = syn.loc[syn[HOT_COLUMNS.turbine] == rep.test_wtg].index
        truth = rep.true_uplift(mask=treated_activity_mask(test_index, rep.upgrade_timing, window=window)).overall

        logger.info(
            "[%s] replicate %d (%s, start %s, %d mo): estimate %+.2f%%, truth %+.2f%%, error %+.2f%% -> %s",
            study.mode,
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
                "mode": study.mode,
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
    return rows


def inspect_naive_run(
    *,
    out_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    n_replicates: int = 3,
    delta: float = 0.05,
    campaign_months: list[int] | None = None,
    toggle_period: pd.Timedelta = DEFAULT_TOGGLE_PERIOD,
) -> pd.DataFrame:
    """Run a few naive-ratio prepost and toggle replicates with plots on; return a summary frame.

    :param out_root: output directory; defaults to :func:`default_output_root` / ``inspection_naive``
    :param data_dir: Hill of Towie data/cache dir; defaults to the source package default
    :param n_replicates: how many replicate datasets to run per mode (each in its own out dir)
    :param delta: injected constant-Cp uplift fraction
    :param campaign_months: campaign length(s); the longest is used for each replicate
    :param toggle_period: the toggle on/off cycle length (20 on + 20 off = 40 min by default)
    :return: per-replicate frame across both modes (mode, test_wtg, estimate, truth, signed_error, ...)
    """
    out_dir = (Path(out_root) if out_root is not None else default_output_root()) / "inspection_naive"
    out_dir.mkdir(parents=True, exist_ok=True)
    campaign_months = campaign_months if campaign_months is not None else [6]

    logger.info("Loading Hill of Towie SCADA %s..%s", DEFAULT_START_DT, DEFAULT_END_DT_EXCL)
    scada_df, _metadata_df = load_hot_scada(
        start_dt=DEFAULT_START_DT,
        end_dt_excl=DEFAULT_END_DT_EXCL,
        wtg_numbers=DEFAULT_WTG_NUMBERS,
        wtg_names=DEFAULT_TURBINE_SUBSET,
        data_dir=Path(data_dir) if data_dir is not None else None,
    )

    prepost_study = StudyConfig(
        mode="prepost",
        turbine_subset=DEFAULT_TURBINE_SUBSET,
        treatment_start_range=DEFAULT_TREATMENT_START_RANGE,
        min_pre_months=MIN_PRE_MONTHS,
        campaign_months=campaign_months,
        n_replicates=n_replicates,
        seed=0,
    )
    toggle_study = StudyConfig(
        mode="toggle",
        turbine_subset=DEFAULT_TURBINE_SUBSET,
        treatment_start_range=DEFAULT_TREATMENT_START_RANGE,
        min_pre_months=MIN_PRE_MONTHS,
        campaign_months=campaign_months,
        n_replicates=n_replicates,
        toggle_period=toggle_period,
        seed=0,
    )

    rows = _inspect_mode(scada_df, study=prepost_study, out_dir=out_dir / "prepost", delta=delta)
    rows += _inspect_mode(scada_df, study=toggle_study, out_dir=out_dir / "toggle", delta=delta)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "inspection_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %s\n%s", summary_path, summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    inspect_naive_run()
