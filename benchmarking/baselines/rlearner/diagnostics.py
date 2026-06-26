"""Per-run diagnostics for the R-learner: CSVs, feature importance, and plots.

A human reviewer must be able to confirm the right data was received and interpreted, and —
critically — spot feature leakage (a feature that trivially predicts power, e.g. a
post-treatment nacelle wind speed or a series-wired voltage; design note §3). Hence the
feature-importance table and plot are first-class outputs and the top features are logged.

Everything here is pure reporting; the estimate itself is computed in :mod:`method`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from benchmarking.baselines.rlearner.features import QUALIFIER
from benchmarking.diagnostics import stages
from benchmarking.diagnostics.density import density_scatter
from benchmarking.diagnostics.style import apply_grid, save_fig

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_SEGMENTS = ("all", "baseline", "upgraded")
_MODEL_LABELS = ("outcome", "propensity", "effect")
_TOP_FEATURES_LOGGED = 10
_MIN_CORR_PAIRS = 2
# Per-segment colours used across the R-learner diagnostic plots.
_SEGMENT_COLORS = {"all": "C0", "baseline": "C0", "upgraded": "C1"}


@dataclass
class DiagnosticData:
    """Everything the diagnostics need; assembled by :class:`~.method.RLearnerMethod`."""

    test_wtg: str
    mode: str
    index: pd.DatetimeIndex  # all unique timestamps
    treated_all: np.ndarray  # upgrade flag (0/1) over all timestamps
    selected_all: np.ndarray  # bool: rows used in the fit
    y_all: np.ndarray  # test power over all timestamps
    timebase: pd.Timedelta
    tau: np.ndarray  # per selected row
    m_hat: np.ndarray
    e_hat: np.ndarray
    mu0: np.ndarray
    y_selected: np.ndarray
    condition_ws: np.ndarray | None  # test turbine's own ws over selected rows (for tau-vs-ws)
    condition_ws_label: str | None  # the original column name of condition_ws (for axis labels)
    feature_names: list[str]
    feature_values: pd.DataFrame  # the selected feature matrix X (for the feature catalogue)
    outcome_model: Any
    propensity_model: Any
    effect_model: Any
    overall_uplift: float
    n_refs: int
    era5_lag_rows: int | None
    era5_corr: float | None
    era5_sweep: pd.DataFrame | None


def feature_importance_long(data: DiagnosticData) -> pd.DataFrame:
    """Long table of LightGBM gain/split importance for the outcome, propensity and effect models."""
    blocks = []
    models = (
        ("outcome", data.outcome_model),
        ("propensity", data.propensity_model),
        ("effect", data.effect_model),
    )
    for label, model in models:
        booster = model.booster_
        blocks.append(
            pd.DataFrame(
                {
                    "model": label,
                    "feature": data.feature_names,
                    "gain": booster.feature_importance(importance_type="gain"),
                    "split_count": booster.feature_importance(importance_type="split"),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True).sort_values(["model", "gain"], ascending=[True, False])


def log_top_features(importance: pd.DataFrame) -> None:
    """Log the outcome model's top features so a human can spot a leaking (too-good) predictor."""
    top = importance[importance["model"] == "outcome"].head(_TOP_FEATURES_LOGGED)
    pairs = ", ".join(f"{r.feature} (gain={r.gain:.0f})" for r in top.itertuples())
    logger.info("R-learner outcome-model top features by gain: %s", pairs)
    logger.info("Review the above for leakage: a feature that trivially predicts power is a red flag.")


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
    """Single-row headline results: uplift, sizes, nuisance fit quality, ERA5 sync."""
    treated_sel = data.treated_all[data.selected_all].astype(bool)
    resid = data.y_selected - data.m_hat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((data.y_selected - data.y_selected.mean()) ** 2))
    return pd.DataFrame(
        [
            {
                "test_wtg": data.test_wtg,
                "mode": data.mode,
                "n_refs": data.n_refs,
                "n_timestamps": len(data.index),
                "n_selected": int(data.selected_all.sum()),
                "n_selected_upgraded": int(treated_sel.sum()),
                "n_features": len(data.feature_names),
                "uplift_frc": data.overall_uplift,
                "outcome_mae": float(np.mean(np.abs(resid))),
                "outcome_r2": 1.0 - ss_res / ss_tot if ss_tot else np.nan,
                "propensity_mean": float(np.mean(data.e_hat)),
                "propensity_std": float(np.std(data.e_hat)),
                "era5_lag_rows": data.era5_lag_rows,
                "era5_corr": data.era5_corr,
                "time_calculated": pd.Timestamp.utcnow(),
            }
        ]
    )


def feature_catalogue(data: DiagnosticData) -> pd.DataFrame:
    """One row per ML feature: source tag/turbine, coverage, basic stats, per-model gain, |corr| with power.

    The overview a human scans to decide whether a feature should be dropped (low coverage and/or
    no importance) or whether something is missing — with dozens of columns a sortable table is the
    practical medium, complemented by :func:`save_plots`'s overview scatter.
    """
    importance = feature_importance_long(data)
    gain = {model: importance[importance["model"] == model].set_index("feature")["gain"] for model in _MODEL_LABELS}
    y = data.y_selected
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
                "gain_outcome": float(gain["outcome"].get(feature, 0.0)),
                "gain_propensity": float(gain["propensity"].get(feature, 0.0)),
                "gain_effect": float(gain["effect"].get(feature, 0.0)),
                "abs_corr_with_power": _abs_corr(col, y),
            }
        )
    return pd.DataFrame(rows).sort_values("gain_outcome", ascending=False, ignore_index=True)


def _abs_corr(col: np.ndarray, y: np.ndarray) -> float:
    """Absolute Pearson correlation of a feature with the outcome over their finite pairs."""
    pair = np.isfinite(col) & np.isfinite(y)
    if pair.sum() < _MIN_CORR_PAIRS or np.std(col[pair]) == 0 or np.std(y[pair]) == 0:
        return float("nan")
    return float(abs(np.corrcoef(col[pair], y[pair])[0, 1]))


def write_csvs(run_dir: Path, run_name: str, ts: str, data: DiagnosticData) -> pd.DataFrame:
    """Write the data-stats, results, feature-importance and feature-catalogue CSVs; return importance."""
    segment_stats(data).to_csv(run_dir / f"{run_name}_data_stats_{ts}.csv", index=False)
    results_row(data).to_csv(run_dir / f"{run_name}_results_{ts}.csv", index=False)
    importance = feature_importance_long(data)
    importance.to_csv(run_dir / f"{run_name}_feature_importance_{ts}.csv", index=False)
    feature_catalogue(data).to_csv(run_dir / f"{run_name}_feature_catalogue_{ts}.csv", index=False)
    return importance


def save_plots(plots_dir: Path, data: DiagnosticData, importance: pd.DataFrame) -> None:
    """Write the R-learner modelling plots into their analysis-stage subfolders."""
    model_dir = plots_dir / stages.UPLIFT_MODELLING
    model_dir.mkdir(parents=True, exist_ok=True)
    _plot_importance(model_dir, importance)
    _plot_residual_vs_prediction(model_dir, data)
    _plot_propensity(model_dir, data)
    _plot_predicted_vs_actual(model_dir, data)
    _plot_tau(model_dir, data)

    inputs_dir = plots_dir / stages.UPLIFT_INPUTS
    inputs_dir.mkdir(parents=True, exist_ok=True)
    _plot_feature_overview(inputs_dir, feature_catalogue(data))

    if data.era5_sweep is not None:
        feat_dir = plots_dir / stages.FEATURE_ENG
        feat_dir.mkdir(parents=True, exist_ok=True)
        _plot_era5_sweep(feat_dir, data)


def _plot_feature_overview(plots_dir: Path, catalogue: pd.DataFrame) -> None:
    """All features as a horizontal bar of outcome gain, coloured by coverage — the add/remove overview."""
    df = catalogue.sort_values("gain_outcome", ascending=True)
    norm = Normalize(vmin=0.0, vmax=100.0)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(12, max(6.0, 0.25 * len(df))))
    ax.barh(df["feature"], df["gain_outcome"], color=cmap(norm(df["coverage_pct"].to_numpy())))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="coverage [%]")
    ax.set_xlabel("outcome-model gain")
    ax.set_title("all ML features: importance (bar) and coverage (colour) — short + cold = drop candidate")
    apply_grid(ax)
    save_fig(fig, plots_dir / "feature_overview.png")


def _segment_masks(data: DiagnosticData) -> dict[str, np.ndarray]:
    """Boolean masks over the *selected* rows for the all/baseline/upgraded segments."""
    treated_sel = data.treated_all[data.selected_all].astype(bool)
    return {"all": np.ones_like(treated_sel, dtype=bool), "baseline": ~treated_sel, "upgraded": treated_sel}


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


def _plot_importance(plots_dir: Path, importance: pd.DataFrame) -> None:
    """Top features by gain for the three models, stacked vertically so long tag names never overlap."""
    panels = (
        ("outcome", "outcome m(x) = E[Y|X]  (predicts power)", "C0"),
        ("propensity", "propensity e(x) = E[T|X]  (predicts upgrade flag)", "C2"),
        ("effect", "effect tau(x)  (predicts per-record uplift)", "C1"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 16))
    for ax, (label, title, color) in zip(axes, panels, strict=True):
        top = importance[importance["model"] == label].head(15).iloc[::-1]
        ax.barh(top["feature"], top["gain"], color=color)
        ax.set_title(title)
        ax.set_xlabel("gain")
        apply_grid(ax)
    fig.suptitle("R-learner feature importance — review for leakage (a feature that trivially predicts power)")
    save_fig(fig, plots_dir / "feature_importance.png")


def _plot_residual_vs_prediction(plots_dir: Path, data: DiagnosticData) -> None:
    """Outcome residual vs prediction, density-coloured, per all/baseline/upgraded segment."""
    masks = _segment_masks(data)
    resid = data.y_selected - data.m_hat
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=True)
    for i, (ax, segment) in enumerate(zip(axes, _SEGMENTS, strict=True)):
        seg = masks[segment]
        density_scatter(data.m_hat[seg], resid[seg], ax=ax, s=6, colorbar=(i == len(_SEGMENTS) - 1))
        ax.axhline(0, color="k", linewidth=1)
        r2, mae = _r2_mae(data.y_selected[seg], data.m_hat[seg])
        ax.set_title(f"{segment}  (R²={r2:.3f}, MAE={mae:.0f} kW, n={int(seg.sum())})")
        ax.set_xlabel("predicted power [kW]")
        ax.set_ylabel("residual (actual - predicted) [kW]")
        apply_grid(ax)
    fig.suptitle(f"{data.test_wtg}: outcome residual vs prediction (conditional-bias check)")
    save_fig(fig, plots_dir / "residual_vs_prediction.png")


def _plot_propensity(plots_dir: Path, data: DiagnosticData) -> None:
    """Propensity overlap histogram, y-axis in hours of data (feedback 7)."""
    hours_per_row = data.timebase / pd.Timedelta(hours=1)
    weights = np.full(len(data.e_hat), hours_per_row)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data.e_hat, bins=40, range=(0, 1), weights=weights, color="C2")
    ax.set_xlabel("propensity e(x) = P(upgraded | X)")
    ax.set_ylabel("hours of data")
    ax.set_title(f"{data.test_wtg}: propensity overlap (~0.5 for toggle, spread for before/after)")
    apply_grid(ax)
    save_fig(fig, plots_dir / "propensity_hist.png")


def _plot_predicted_vs_actual(plots_dir: Path, data: DiagnosticData) -> None:
    """Predicted vs actual power, density-coloured, per all/baseline/upgraded with R²/MAE and a 1:1 line."""
    masks = _segment_masks(data)
    lim = [0.0, float(np.nanmax(data.y_selected))] if len(data.y_selected) else [0.0, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), sharex=True, sharey=True)
    for i, (ax, segment) in enumerate(zip(axes, _SEGMENTS, strict=True)):
        seg = masks[segment]
        density_scatter(data.y_selected[seg], data.m_hat[seg], ax=ax, s=6, colorbar=(i == len(_SEGMENTS) - 1))
        ax.plot(lim, lim, color="red", linewidth=1.2, label="1:1")
        r2, mae = _r2_mae(data.y_selected[seg], data.m_hat[seg])
        ax.set_title(f"{segment}  (R²={r2:.3f}, MAE={mae:.0f} kW, n={int(seg.sum())})")
        ax.set_xlabel("actual power [kW]")
        ax.set_ylabel("predicted power [kW]")
        ax.legend(loc="upper left")
        apply_grid(ax)
    fig.suptitle(f"{data.test_wtg}: outcome model predicted vs actual")
    save_fig(fig, plots_dir / "predicted_vs_actual.png")


def _plot_tau(plots_dir: Path, data: DiagnosticData) -> None:
    """Per-record uplift in kW and as % of expected power, with the wind-speed dependence.

    The kW values can be implausibly wide in prepost — that spread is the F1 overlap/extrapolation
    symptom (the effect model extrapolating where ``t_res ≈ 0``), not a unit error; the mean is
    annotated and the % view normalises by the baseline expected power ``mu0``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_pct = np.where(np.abs(data.mu0) > 0, 100.0 * data.tau / data.mu0, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].hist(data.tau, bins=40, color="C3")
    axes[0].axvline(float(np.nanmean(data.tau)), color="k", linestyle="--", label=f"mean={np.nanmean(data.tau):.1f} kW")
    axes[0].set_xlabel("tau(x): absolute uplift [kW]")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"{data.test_wtg}: per-record uplift [kW]")
    axes[0].legend()
    apply_grid(axes[0])

    finite_pct = tau_pct[np.isfinite(tau_pct)]
    axes[1].hist(finite_pct, bins=40, range=_robust_range(finite_pct), color="C3")
    axes[1].set_xlabel("tau(x) / mu0: uplift [% of expected power]")
    axes[1].set_ylabel("count")
    axes[1].set_title("per-record uplift [%]")
    apply_grid(axes[1])

    if data.condition_ws is not None:
        density_scatter(data.condition_ws, data.tau, ax=axes[2], s=6, colorbar=True)
        axes[2].set_xlabel(data.condition_ws_label or "wind speed [m/s]")
        axes[2].set_ylabel("tau(x) [kW]")
        axes[2].set_title("uplift vs wind speed")
        apply_grid(axes[2])
    else:
        axes[2].set_visible(False)
    save_fig(fig, plots_dir / "tau.png")


def _robust_range(values: np.ndarray) -> tuple[float, float] | None:
    """Return a 1st-99th percentile range so a few extreme tau% values do not flatten the histogram."""
    if values.size == 0:
        return None
    lo, hi = np.percentile(values, [1, 99])
    return (float(lo), float(hi)) if hi > lo else None


def _plot_era5_sweep(plots_dir: Path, data: DiagnosticData) -> None:
    """ERA5 correlation-vs-lag sweep, grid on, with the chosen optimal shift annotated (feedback 2)."""
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
