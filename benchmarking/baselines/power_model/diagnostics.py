"""Per-run diagnostics for the power model: CSVs, feature importance, and plots.

A human reviewer must be able to confirm the right data was received and interpreted and —
critically — spot feature leakage (a feature that trivially predicts power rather than carrying
weather/wake information; design note §3). Hence the feature-importance table and plot are
first-class outputs and the top features are logged. The headline uplift is re-derivable from the
results CSV as ``sum_actual / sum_counterfactual - 1``.

Everything here is pure reporting; the estimate itself is computed in :mod:`method`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from benchmarking.baselines.power_model.features import QUALIFIER
from benchmarking.diagnostics import stages
from benchmarking.diagnostics.density import density_scatter
from benchmarking.diagnostics.style import apply_grid, save_fig

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_SEGMENTS = ("all", "baseline", "upgraded")
_TOP_FEATURES_LOGGED = 12
_MIN_CORR_PAIRS = 2


@dataclass
class DiagnosticData:
    """Everything the diagnostics need; assembled by :class:`~.method.PowerModelMethod`."""

    test_wtg: str
    mode: str
    index: pd.DatetimeIndex  # all unique timestamps
    treated_all: np.ndarray  # upgrade flag (0/1) over all timestamps
    selected_all: np.ndarray  # bool: normally-operating rows with finite power
    y_all: np.ndarray  # test power over all timestamps
    timebase: pd.Timedelta
    # counterfactual on the upgraded selected rows
    upgraded_ts: pd.DatetimeIndex
    y_upgraded: np.ndarray
    pred_upgraded: np.ndarray  # counterfactual expected power had there been no upgrade
    # held-out baseline fit quality
    y_baseline_valid: np.ndarray
    pred_baseline_valid: np.ndarray
    # feature catalogue / importance
    feature_names: list[str]
    feature_values: pd.DataFrame  # selected feature matrix X (baseline+upgraded selected rows)
    y_selected: np.ndarray  # test power over those selected rows (for |corr|)
    outcome_model: Any
    overall_uplift: float
    sum_actual_kw: float
    sum_counterfactual_kw: float
    n_refs: int
    era5_lag_rows: int | None
    era5_corr: float | None
    era5_sweep: pd.DataFrame | None


def feature_importance_long(data: DiagnosticData) -> pd.DataFrame:
    """Long table of LightGBM gain/split importance for the (single) outcome power model."""
    booster = data.outcome_model.booster_
    return pd.DataFrame(
        {
            "feature": data.feature_names,
            "gain": booster.feature_importance(importance_type="gain"),
            "split_count": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False, ignore_index=True)


def log_top_features(importance: pd.DataFrame) -> None:
    """Log the model's top features so a human can spot a leaking (too-good) predictor."""
    top = importance.head(_TOP_FEATURES_LOGGED)
    pairs = ", ".join(f"{r.feature} (gain={r.gain:.0f})" for r in top.itertuples())
    logger.info("power_model top features by gain: %s", pairs)
    logger.info(
        "Review the above: weather + wake tags are expected; a feature that trivially predicts power is a flag."
    )


def segment_stats(data: DiagnosticData) -> pd.DataFrame:
    """Per-segment (all/baseline/upgraded) counts and energy, for a human data sanity check."""
    timebase_hours = data.timebase / pd.Timedelta(hours=1)
    treated = data.treated_all.astype(bool)
    masks = {"all": np.ones(len(data.index), dtype=bool), "baseline": ~treated, "upgraded": treated}
    rows = []
    for segment in _SEGMENTS:
        seg = masks[segment]
        seg_sel = seg & data.selected_all
        seg_power = data.y_all[seg_sel]
        finite = seg_power[np.isfinite(seg_power)]
        rows.append(
            {
                "segment": segment,
                "first_timestamp": data.index[seg].min() if seg.any() else pd.NaT,
                "last_timestamp": data.index[seg].max() if seg.any() else pd.NaT,
                "n_timestamps": int(seg.sum()),
                "n_selected": int(seg_sel.sum()),
                "selected_fraction": float(seg_sel.sum() / seg.sum()) if seg.any() else np.nan,
                "test_mean_power_kw": float(finite.mean()) if len(finite) else np.nan,
                "test_mwh": float(finite.sum()) * timebase_hours / 1000.0 if len(finite) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def results_row(data: DiagnosticData) -> pd.DataFrame:
    """Single-row headline results: uplift, energy totals, baseline fit quality, ERA5 sync."""
    timebase_hours = data.timebase / pd.Timedelta(hours=1)
    r2, mae = _r2_mae(data.y_baseline_valid, data.pred_baseline_valid)
    return pd.DataFrame(
        [
            {
                "test_wtg": data.test_wtg,
                "mode": data.mode,
                "n_refs": data.n_refs,
                "n_timestamps": len(data.index),
                "n_selected": int(data.selected_all.sum()),
                "n_selected_upgraded": len(data.y_upgraded),
                "n_features": len(data.feature_names),
                "uplift_frc": data.overall_uplift,
                "sum_actual_mwh": data.sum_actual_kw * timebase_hours / 1000.0,
                "sum_counterfactual_mwh": data.sum_counterfactual_kw * timebase_hours / 1000.0,
                "baseline_holdout_r2": r2,
                "baseline_holdout_mae_kw": mae,
                "era5_lag_rows": data.era5_lag_rows,
                "era5_corr": data.era5_corr,
                "time_calculated": pd.Timestamp.utcnow(),
            }
        ]
    )


def feature_catalogue(data: DiagnosticData) -> pd.DataFrame:
    """One row per feature: source tag/turbine, coverage, basic stats, gain, |corr| with power."""
    importance = feature_importance_long(data).set_index("feature")
    rows = []
    for feature in data.feature_names:
        col = data.feature_values[feature].to_numpy(dtype=float)
        finite = np.isfinite(col)
        tag, _, turbine = feature.partition(QUALIFIER)
        rows.append(
            {
                "feature": feature,
                "source_tag": tag,
                "turbine": turbine or "ERA5/derived",
                "coverage_pct": float(100.0 * finite.mean()) if len(col) else np.nan,
                "mean": float(np.nanmean(col)) if finite.any() else np.nan,
                "std": float(np.nanstd(col)) if finite.any() else np.nan,
                "min": float(np.nanmin(col)) if finite.any() else np.nan,
                "max": float(np.nanmax(col)) if finite.any() else np.nan,
                "gain": float(importance["gain"].get(feature, 0.0)),
                "abs_corr_with_power": _abs_corr(col, data.y_selected),
            }
        )
    return pd.DataFrame(rows).sort_values("gain", ascending=False, ignore_index=True)


def _abs_corr(col: np.ndarray, y: np.ndarray) -> float:
    """Absolute Pearson correlation of a feature with the outcome over their finite pairs."""
    pair = np.isfinite(col) & np.isfinite(y)
    if pair.sum() < _MIN_CORR_PAIRS or np.std(col[pair]) == 0 or np.std(y[pair]) == 0:
        return float("nan")
    return float(abs(np.corrcoef(col[pair], y[pair])[0, 1]))


def _r2_mae(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Return (R², MAE) over the finite pairs of ``actual`` / ``predicted``."""
    finite = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[finite], predicted[finite]
    if len(actual) == 0:
        return float("nan"), float("nan")
    resid = actual - predicted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return r2, float(np.mean(np.abs(resid)))


def write_csvs(run_dir: Path, run_name: str, ts: str, data: DiagnosticData) -> pd.DataFrame:
    """Write the data-stats, results, feature-importance and feature-catalogue CSVs; return importance."""
    segment_stats(data).to_csv(run_dir / f"{run_name}_data_stats_{ts}.csv", index=False)
    results_row(data).to_csv(run_dir / f"{run_name}_results_{ts}.csv", index=False)
    importance = feature_importance_long(data)
    importance.to_csv(run_dir / f"{run_name}_feature_importance_{ts}.csv", index=False)
    feature_catalogue(data).to_csv(run_dir / f"{run_name}_feature_catalogue_{ts}.csv", index=False)
    return importance


def save_plots(plots_dir: Path, data: DiagnosticData, importance: pd.DataFrame) -> None:
    """Write the power-model diagnostic plots into their analysis-stage subfolders."""
    model_dir = plots_dir / stages.UPLIFT_MODELLING
    model_dir.mkdir(parents=True, exist_ok=True)
    _plot_importance(model_dir, importance, test_wtg=data.test_wtg)
    _plot_predicted_vs_actual(model_dir, data)
    _plot_residual_vs_mean(model_dir, data)

    results_dir = plots_dir / stages.UPLIFT_RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)
    _plot_actual_vs_counterfactual_timeseries(results_dir, data)

    inputs_dir = plots_dir / stages.UPLIFT_INPUTS
    inputs_dir.mkdir(parents=True, exist_ok=True)
    _plot_feature_overview(inputs_dir, feature_catalogue(data))
    _save_feature_histograms(inputs_dir / "feature_histograms", data)

    if data.era5_sweep is not None:
        feat_dir = plots_dir / stages.FEATURE_ENG
        feat_dir.mkdir(parents=True, exist_ok=True)
        _plot_era5_sweep(feat_dir, data)


def _plot_importance(plots_dir: Path, importance: pd.DataFrame, *, test_wtg: str) -> None:
    """Top features by gain for the single power model — the leakage / feature-thesis check."""
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(6.0, 0.4 * len(top))))
    ax.barh(top["feature"], top["gain"], color="C0")
    ax.set_xlabel("gain")
    ax.set_title(f"{test_wtg}: power-model feature importance — expect weather + wake tags to dominate")
    apply_grid(ax)
    save_fig(fig, plots_dir / "feature_importance.png")


def _plot_feature_overview(plots_dir: Path, catalogue: pd.DataFrame) -> None:
    """All features as a horizontal bar of gain, coloured by coverage — the add/remove overview."""
    df = catalogue.sort_values("gain", ascending=True)
    norm = Normalize(vmin=0.0, vmax=100.0)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(12, max(6.0, 0.3 * len(df))))
    ax.barh(df["feature"], df["gain"], color=cmap(norm(df["coverage_pct"].to_numpy())))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="coverage [%]")
    ax.set_xlabel("outcome-model gain")
    ax.set_title("all features: importance (bar) and coverage (colour) — short + cold = drop candidate")
    apply_grid(ax)
    save_fig(fig, plots_dir / "feature_overview.png")


def _save_feature_histograms(out_dir: Path, data: DiagnosticData) -> None:
    """One baseline-vs-upgraded density histogram per model input feature, using the real tag name.

    Every column the uplift model actually sees gets its own plot (named by its real source tag /
    ERA5 column) so a reviewer can check the baseline and upgraded distributions overlap for *each*
    input — the per-feature view of the overlap/positivity story, complementing the curated shared
    ``condition_histograms.png``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = np.asarray(data.selected_all, dtype=bool)
    treated_sel = np.asarray(data.treated_all, dtype=bool)[sel]  # aligned to feature_values rows
    baseline_sel = ~treated_sel
    for feature in data.feature_names:
        vals = data.feature_values[feature].to_numpy(dtype=float)
        bins = _robust_bins(vals)
        fig, ax = plt.subplots(figsize=(7, 5))
        for seg_label, seg, color in (("baseline", baseline_sel, "C0"), ("upgraded", treated_sel, "C1")):
            seg_vals = vals[seg & np.isfinite(vals)]
            if seg_vals.size:
                ax.hist(
                    seg_vals, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=color, label=seg_label
                )
        ax.set_xlabel(feature)
        ax.set_ylabel("density")
        ax.set_title(f"{data.test_wtg}: {feature}\n(used rows, baseline vs upgraded)")
        apply_grid(ax)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        save_fig(fig, out_dir / f"{_safe_filename(feature)}.png")


def _robust_bins(values: np.ndarray, *, bins: int = 30) -> list[float] | int:
    """Bin edges over the 1st-99th percentile of the finite values, so outliers don't dominate."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return bins
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        return bins
    return np.linspace(lo, hi, bins + 1).tolist()


def _safe_filename(name: str) -> str:
    """Turn a feature name (which may contain ``@``, spaces, ``/``) into a safe file stem."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")


def _plot_predicted_vs_actual(plots_dir: Path, data: DiagnosticData) -> None:
    """Two panels: held-out baseline fit (should hug 1:1) and the upgraded counterfactual gap.

    On the upgraded panel the points sit *above* the 1:1 line by roughly the uplift: actual upgraded
    power exceeds the counterfactual the model predicts from the (upgrade-blind) references.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    yb, pb = data.y_baseline_valid, data.pred_baseline_valid
    lim_b = [0.0, float(np.nanmax(yb))] if len(yb) else [0.0, 1.0]
    density_scatter(yb, pb, ax=axes[0], s=6, colorbar=True)
    axes[0].plot(lim_b, lim_b, color="red", linewidth=1.2, label="1:1")
    r2, mae = _r2_mae(yb, pb)
    axes[0].set_title(f"baseline (held-out)  R²={r2:.3f}, MAE={mae:.0f} kW, n={len(yb)}")
    axes[0].set_xlabel("actual power [kW]")
    axes[0].set_ylabel("predicted power [kW]")
    axes[0].legend(loc="upper left")
    apply_grid(axes[0])

    yu, pu = data.y_upgraded, data.pred_upgraded
    lim_u = [0.0, float(np.nanmax(yu))] if len(yu) else [0.0, 1.0]
    density_scatter(yu, pu, ax=axes[1], s=6, colorbar=True)
    axes[1].plot(lim_u, lim_u, color="red", linewidth=1.2, label="1:1 (no uplift)")
    axes[1].set_title(f"upgraded: actual vs counterfactual  uplift={100 * data.overall_uplift:+.2f}%, n={len(yu)}")
    axes[1].set_xlabel("actual power [kW]")
    axes[1].set_ylabel("counterfactual predicted power [kW]")
    axes[1].legend(loc="upper left")
    apply_grid(axes[1])

    fig.suptitle(f"{data.test_wtg}: power-model predicted vs actual")
    save_fig(fig, plots_dir / "predicted_vs_actual.png")


def _plot_residual_vs_mean(plots_dir: Path, data: DiagnosticData) -> None:
    """Bland-Altman: residual (actual - predicted) vs the mean of predicted and actual power.

    Two panels (held-out baseline, upgraded). The baseline residuals should sit on zero with no
    trend across the power range (a tilt would flag a conditional bias in the fit). On the upgraded
    panel the residual *is* the uplift signal, so it sits above zero by the mean uplift in kW
    (annotated) — a flat band is a clean additive uplift; a slope means the uplift varies with
    power level.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    panels = (
        ("baseline (held-out)", data.y_baseline_valid, data.pred_baseline_valid),
        ("upgraded", data.y_upgraded, data.pred_upgraded),
    )
    for ax, (label, actual, predicted) in zip(axes, panels, strict=True):
        resid = actual - predicted
        mean_power = 0.5 * (actual + predicted)
        finite = np.isfinite(resid) & np.isfinite(mean_power)
        density_scatter(mean_power[finite], resid[finite], ax=ax, s=6, colorbar=(label == "upgraded"))
        ax.axhline(0, color="k", linewidth=1)
        mean_resid = float(np.mean(resid[finite])) if finite.any() else float("nan")
        ax.axhline(
            mean_resid, color="red", linestyle="--", linewidth=1.2, label=f"mean residual = {mean_resid:+.0f} kW"
        )
        ax.set_title(f"{label}  (n={int(finite.sum())})")
        ax.set_xlabel("mean of predicted and actual power [kW]")
        ax.set_ylabel("residual (actual - predicted) [kW]")
        ax.legend(loc="upper left")
        apply_grid(ax)
    fig.suptitle(f"{data.test_wtg}: power-model residual vs mean power (Bland-Altman)")
    save_fig(fig, plots_dir / "residual_vs_mean.png")


def _plot_actual_vs_counterfactual_timeseries(plots_dir: Path, data: DiagnosticData) -> None:
    """Daily energy: actual upgraded power vs the model's counterfactual, over the upgraded window."""
    timebase_hours = data.timebase / pd.Timedelta(hours=1)
    actual = pd.Series(data.y_upgraded * timebase_hours / 1000.0, index=data.upgraded_ts)
    counter = pd.Series(data.pred_upgraded * timebase_hours / 1000.0, index=data.upgraded_ts)
    daily_actual = actual.resample("1D").sum(min_count=1)
    daily_counter = counter.resample("1D").sum(min_count=1)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily_actual.index.to_numpy(), daily_actual.to_numpy(), marker=".", color="C1", label="actual (upgraded)")
    ax.plot(
        daily_counter.index.to_numpy(),
        daily_counter.to_numpy(),
        marker=".",
        color="C0",
        label="counterfactual (model)",
    )
    ax.set_xlabel("date")
    ax.set_ylabel("daily energy [MWh]")
    ax.set_title(f"{data.test_wtg}: actual vs counterfactual daily energy (gap = uplift)")
    apply_grid(ax)
    ax.legend()
    save_fig(fig, plots_dir / "actual_vs_counterfactual_timeseries.png")


def _plot_era5_sweep(plots_dir: Path, data: DiagnosticData) -> None:
    """ERA5 correlation-vs-lag sweep, with the chosen optimal shift annotated."""
    sweep = data.era5_sweep
    if sweep is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sweep["shift_rows"], sweep["corr"], marker=".")
    if data.era5_lag_rows is not None:
        corr_text = f"{data.era5_corr:.3f}" if data.era5_corr is not None else "n/a"
        ax.axvline(
            data.era5_lag_rows,
            color="k",
            linestyle="--",
            label=f"best shift = {data.era5_lag_rows} rows (corr = {corr_text})",
        )
        ax.legend()
    ax.set_xlabel("ERA5 shift [rows]")
    ax.set_ylabel("wind-speed correlation")
    ax.set_title(f"{data.test_wtg}: ERA5-SCADA correlation vs lag")
    apply_grid(ax)
    save_fig(fig, plots_dir / "era5_sync.png")
