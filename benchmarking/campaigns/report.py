"""The campaign reports: the analyst-facing one, and the benchmark's truth overlay on it.

:func:`write_report` writes what a real campaign can produce -- estimates, the farm headline and
the reference-stability check -- and reads no ground truth. :func:`write_campaign_report` writes
the same directory with everything truth adds.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from benchmarking.campaigns.runner import per_turbine_table
from benchmarking.harness import CONDITIONS, condition_bins, conditional_truth_vs_estimate, plot_conditional_uplift
from benchmarking.harness.plots import conditional_estimates
from benchmarking.synthetic import treated_mask

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.campaigns.run import CampaignReport
    from benchmarking.campaigns.runner import CampaignResult
    from benchmarking.synthetic import SyntheticDataset

logger = logging.getLogger(__name__)


def write_report(report: CampaignReport, *, out_dir: Path) -> Path:
    """Write the analyst-facing campaign report under ``out_dir`` and return it.

    Writes ``per_turbine.csv``, ``farm_uplift.csv``, ``farm_uplift_detail.csv`` and
    ``reference_stability.csv`` -- each candidate reference estimated as if it were a test
    turbine, which a healthy campaign reads near 0%. A method reporting per-condition estimates
    also gets ``conditional.csv`` and one plot per condition under ``conditional/``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report.per_turbine.to_csv(out_dir / "per_turbine.csv", index=False)
    report.farm.to_csv(out_dir / "farm_uplift.csv", index=False)
    report.reference_stability.to_csv(out_dir / "reference_stability.csv", index=False)
    _write_detail(report.farm_uplifts, out_dir=out_dir)

    label = report.spec.change_label()
    logger.info("Per-turbine uplift for %s:\n%s", label, report.per_turbine.to_string(index=False))
    logger.info("Farm uplift for %s:\n%s", label, report.farm.to_string(index=False))
    _log_guards(report.farm_uplifts)
    _log_reference_stability(report.reference_stability)

    if not report.conditional.empty:
        report.conditional.to_csv(out_dir / "conditional.csv", index=False)
        _write_estimate_plots(report, out_dir=out_dir / "conditional")
    return out_dir


def write_campaign_report(result: CampaignResult, dataset: SyntheticDataset, *, out_dir: Path) -> Path:
    """Write the benchmark report under ``out_dir`` and return it.

    The analyst report plus what truth adds: ``per_turbine.csv`` and ``farm_uplift.csv`` gain
    their truth and signed-error columns, ``scores.csv`` carries the tidy harness rows, and each
    conditional plot gains its true-uplift series.
    """
    write_report(result.report, out_dir=out_dir)
    per_turbine = per_turbine_table(result)
    per_turbine.to_csv(out_dir / "per_turbine.csv", index=False)
    result.farm.to_csv(out_dir / "farm_uplift.csv", index=False)
    result.scores.to_csv(out_dir / "scores.csv", index=False)

    label = result.spec.change_label()
    logger.info("Per-turbine uplift vs truth for %s:\n%s", label, per_turbine.to_string(index=False))
    logger.info("Farm uplift vs truth for %s:\n%s", label, result.farm.to_string(index=False))

    _write_conditional_plots(result, dataset, out_dir=out_dir / "conditional")
    return out_dir


def _write_detail(farm_uplifts: dict, *, out_dir: Path) -> None:
    """Write the per-turbine detail behind every method's farm headline."""
    if not farm_uplifts:
        return
    detail = pd.concat([frame.turbines.assign(method=name) for name, frame in farm_uplifts.items()], ignore_index=True)
    detail.to_csv(out_dir / "farm_uplift_detail.csv", index=False)


def _log_guards(farm_uplifts: dict) -> None:
    """Warn about any turbine a guard dropped from a method's farm headline."""
    if not farm_uplifts:
        return
    detail = pd.concat([frame.turbines.assign(method=name) for name, frame in farm_uplifts.items()], ignore_index=True)
    guarded = detail[detail["guard"] != ""]
    if not guarded.empty:
        logger.warning("Guards fired:\n%s", guarded.to_string(index=False))


def _log_reference_stability(stability: pd.DataFrame) -> None:
    """Report each method's reference-turbine self-uplift, the campaign judging its own references."""
    if stability.empty:
        return
    logger.info(
        "Reference stability (each reference estimated as if it were a test turbine):\n%s",
        stability.to_string(index=False),
    )


def _write_estimate_plots(report: CampaignReport, *, out_dir: Path) -> None:
    """One estimate-only conditional-uplift plot per method, turbine and condition reported."""
    for (method_name, wtg), output in report.outputs.items():
        if output.p50_by_condition is None:
            continue
        frame = conditional_estimates(output, method_name=method_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        for condition in sorted(set(frame["condition"])):
            fig = plot_conditional_uplift(
                frame,
                condition=condition,
                save_path=out_dir / f"conditional_uplift_{condition}_{wtg}_{method_name}.png",
                title=f"Conditional uplift ({condition}) - {wtg}, {method_name}",
            )
            plt.close(fig)


def _write_conditional_plots(result: CampaignResult, dataset: SyntheticDataset, *, out_dir: Path) -> None:
    """One conditional-uplift plot per condition, for every method that reports per-condition rows."""
    spec = result.spec
    for (method_name, wtg), output in result.outputs.items():
        if output.p50_by_condition is None:
            continue
        # only the conditions the method actually reported: plotting the others would draw a
        # method-vs-truth chart whose method series is entirely NaN
        reported = set(output.p50_by_condition["condition"].astype(str))
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
            if condition in reported
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
