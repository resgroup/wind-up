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
from benchmarking.campaigns.uplift_plots import write_uplift_plots
from benchmarking.diagnostics.context import infer_timebase
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
    write_uplift_plots(
        out_dir,
        per_turbine=report.per_turbine,
        stability=report.reference_stability,
        rated_power_kw=report.spec.rated_power_kw,
        farm=report.farm,
    )

    label = report.spec.change_label()
    logger.info(
        "Measured uplift per turbine for %s:\n%s", label, _for_reading(report.per_turbine).to_string(index=False)
    )
    logger.info("Measured farm uplift for %s:\n%s", label, _for_reading(report.farm).to_string(index=False))
    _log_guards(report.farm_uplifts)
    _log_reference_stability(
        report.reference_stability,
        timebase=infer_timebase(pd.DatetimeIndex(report.scada_df.index.unique()).sort_values()),
    )

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


# Uplift columns are reported as percentages, named for what they are rather than for the estimator.
_PERCENT_COLUMNS = {"estimate": "measured uplift [%]", "uplift": "measured uplift [%]", "uplift_spread": "spread [%]"}
# actual_energy is a sum of mean power, not energy; the log converts it and says which it is.
_ENERGY_COLUMN = "actual_energy"
_ENERGY_SHOWN = "treated energy [MWh]"
KW_PER_MW = 1000.0
# A reference read this differently by two test turbines is a real difference, not arithmetic noise.
_STABILITY_TOLERANCE_PCT = 0.1


def _for_reading(frame: pd.DataFrame, *, timebase: pd.Timedelta | None = None) -> pd.DataFrame:
    """Return ``frame`` as a human reads it: uplifts in percent, energy in MWh, counts as counts.

    ``actual_energy`` is a sum of mean power over the treated records, so it becomes energy only
    once the timebase is known; without one the column is left out rather than printed as a number
    whose unit the reader would have to guess.
    """
    shown = frame.copy()
    for column in _PERCENT_COLUMNS:
        if column in shown.columns:
            shown[column] = (shown[column].astype(float) * 100).round(3)
    if _ENERGY_COLUMN in shown.columns:
        if timebase is None:
            shown = shown.drop(columns=[_ENERGY_COLUMN])
        else:
            hours = timebase / pd.Timedelta(hours=1)
            shown[_ENERGY_COLUMN] = (shown[_ENERGY_COLUMN].astype(float) * hours / KW_PER_MW).round(0).astype(int)
    if "n_records" in shown.columns:
        shown["n_records"] = shown["n_records"].astype(int)
    return shown.rename(columns={**_PERCENT_COLUMNS, _ENERGY_COLUMN: _ENERGY_SHOWN})


def _log_reference_stability(stability: pd.DataFrame, *, timebase: pd.Timedelta | None = None) -> None:
    """Report each reference's self-uplift: one row per reference, the campaign judging its pool.

    Every test turbine estimates each reference against the same pool over the same contrast, so
    the readings are one answer repeated and the table collapses to it. A reference the test
    turbines disagree about is the exception -- it means their contrasts differ -- so it is
    reported per turbine instead, with a warning.
    """
    if stability.empty:
        return
    spread = stability.groupby(["method", "turbine"])["uplift"].agg(lambda u: (u.max() - u.min()) * 100)
    disputed = spread[spread > _STABILITY_TOLERANCE_PCT]
    if not disputed.empty:
        logger.warning(
            "The test turbines do not agree on %d reference(s), so each is reported separately. Spread [%%]:\n%s",
            len(disputed),
            disputed.round(3).to_string(),
        )
        logger.info(
            "Reference stability (each reference estimated as if it were a test turbine):\n%s",
            _for_reading(stability, timebase=timebase).to_string(index=False),
        )
        return
    collapsed = (
        stability.groupby(["method", "turbine"], as_index=False)
        .agg(
            test_wtg=("test_wtg", lambda names: "ALL" if len(set(names)) > 1 else str(names.iloc[0])),
            uplift=("uplift", "median"),
            actual_energy=("actual_energy", "median"),
            n_records=("n_records", "median"),
            screened=("screened", "any"),
        )
        .loc[:, ["method", "test_wtg", "turbine", "uplift", "actual_energy", "n_records", "screened"]]
    )
    logger.info(
        "Reference stability (each reference estimated as if it were a test turbine):\n%s",
        _for_reading(collapsed, timebase=timebase).to_string(index=False),
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
