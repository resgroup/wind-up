"""The whole-farm inspection report: per-turbine and farm tables, plus diagnostic plots."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from benchmarking.campaigns.runner import per_turbine_table
from benchmarking.harness import CONDITIONS, condition_bins, conditional_truth_vs_estimate, plot_conditional_uplift
from benchmarking.synthetic import treated_mask

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.campaigns.runner import CampaignResult
    from benchmarking.synthetic import SyntheticDataset

logger = logging.getLogger(__name__)


def write_campaign_report(result: CampaignResult, dataset: SyntheticDataset, *, out_dir: Path) -> Path:
    """Write the campaign's tables and plots under ``out_dir`` and return it.

    Writes ``per_turbine.csv``, ``farm_uplift.csv``, ``farm_uplift_detail.csv`` and ``scores.csv``,
    plus one conditional-uplift plot per condition under ``conditional/`` for each method that
    reports per-condition estimates.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_turbine = per_turbine_table(result)
    per_turbine.to_csv(out_dir / "per_turbine.csv", index=False)
    result.farm.to_csv(out_dir / "farm_uplift.csv", index=False)
    result.scores.to_csv(out_dir / "scores.csv", index=False)

    detail = pd.concat(
        [frame.turbines.assign(method=name) for name, frame in result.farm_uplifts.items()], ignore_index=True
    )
    detail.to_csv(out_dir / "farm_uplift_detail.csv", index=False)

    label = result.spec.change_label()
    logger.info("Per-turbine uplift for %s:\n%s", label, per_turbine.to_string(index=False))
    logger.info("Farm uplift for %s:\n%s", label, result.farm.to_string(index=False))
    guarded = detail[detail["guard"] != ""]
    if not guarded.empty:
        logger.warning("Guards fired:\n%s", guarded.to_string(index=False))

    _write_conditional_plots(result, dataset, out_dir=out_dir / "conditional")
    return out_dir


def _write_conditional_plots(result: CampaignResult, dataset: SyntheticDataset, *, out_dir: Path) -> None:
    """One conditional-uplift plot per condition, for every method that reports per-condition rows."""
    spec = result.spec
    for (method_name, wtg), output in result.outputs.items():
        if output.p50_by_condition is None:
            continue
        rows = dataset.synthetic_df[dataset.synthetic_df[spec.turbine_col] == wtg]
        mask = treated_mask(pd.DatetimeIndex(rows.index), spec.timing_for(wtg))
        truth_by_condition = {
            condition: dataset.true_uplift(
                test_wtg=wtg,
                mask=mask,
                by=condition,
                bins=condition_bins(condition, rated_power_kw=spec.rated_power_kw),
            ).by_condition
            for condition in CONDITIONS
        }
        clean = {c: frame for c, frame in truth_by_condition.items() if frame is not None}
        if not clean:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        frame = conditional_truth_vs_estimate(output, clean, method_name=method_name)
        for condition in clean:
            fig = plot_conditional_uplift(
                frame,
                condition=condition,
                save_path=out_dir / f"conditional_uplift_{condition}_{wtg}_{method_name}.png",
                title=f"Conditional uplift ({condition}) - {wtg}, {method_name} vs truth",
            )
            plt.close(fig)
