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

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_SEGMENTS = ("all", "baseline", "upgraded")
_TOP_FEATURES_LOGGED = 10


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
    condition_ws: np.ndarray | None  # reference mean ws over selected rows (for tau-vs-ws)
    feature_names: list[str]
    outcome_model: Any
    effect_model: Any
    overall_uplift: float
    n_refs: int
    era5_lag_rows: int | None
    era5_corr: float | None
    era5_sweep: pd.DataFrame | None


def feature_importance_long(data: DiagnosticData) -> pd.DataFrame:
    """Long table of LightGBM gain/split importance for the outcome and effect models."""
    blocks = []
    for label, model in (("outcome", data.outcome_model), ("effect", data.effect_model)):
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


def write_csvs(run_dir: Path, run_name: str, ts: str, data: DiagnosticData) -> pd.DataFrame:
    """Write the data-stats, results and feature-importance CSVs; return the importance table."""
    segment_stats(data).to_csv(run_dir / f"{run_name}_data_stats_{ts}.csv", index=False)
    results_row(data).to_csv(run_dir / f"{run_name}_results_{ts}.csv", index=False)
    importance = feature_importance_long(data)
    importance.to_csv(run_dir / f"{run_name}_feature_importance_{ts}.csv", index=False)
    return importance


def save_plots(plots_dir: Path, data: DiagnosticData, importance: pd.DataFrame) -> None:
    """Write the diagnostic plots (feature importance is the headline leakage diagnostic)."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_importance(plots_dir, importance)
    _plot_residual_vs_prediction(plots_dir, data)
    _plot_propensity(plots_dir, data)
    _plot_predicted_vs_actual(plots_dir, data)
    _plot_tau(plots_dir, data)
    if data.era5_sweep is not None:
        _plot_era5_sweep(plots_dir, data)


def _plot_importance(plots_dir: Path, importance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, label in zip(axes, ("outcome", "effect"), strict=True):
        top = importance[importance["model"] == label].head(15).iloc[::-1]
        ax.barh(top["feature"], top["gain"], color="C0" if label == "outcome" else "C1")
        ax.set_title(f"{label} model: top features by gain")
        ax.set_xlabel("gain")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


def _plot_residual_vs_prediction(plots_dir: Path, data: DiagnosticData) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data.m_hat, data.y_selected - data.m_hat, s=6, alpha=0.3)
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xlabel("predicted power [kW]")
    ax.set_ylabel("residual (actual - predicted) [kW]")
    ax.set_title(f"{data.test_wtg}: outcome residual vs prediction (conditional-bias check)")
    fig.tight_layout()
    fig.savefig(plots_dir / "residual_vs_prediction.png", dpi=150)
    plt.close(fig)


def _plot_propensity(plots_dir: Path, data: DiagnosticData) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data.e_hat, bins=40, range=(0, 1), color="C2")
    ax.set_xlabel("propensity e(x) = P(upgraded | X)")
    ax.set_ylabel("count")
    ax.set_title(f"{data.test_wtg}: propensity overlap (~0.5 for toggle, spread for before/after)")
    fig.tight_layout()
    fig.savefig(plots_dir / "propensity_hist.png", dpi=150)
    plt.close(fig)


def _plot_predicted_vs_actual(plots_dir: Path, data: DiagnosticData) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(data.y_selected, data.m_hat, s=6, alpha=0.3)
    lim = [0, float(np.nanmax(data.y_selected))]
    ax.plot(lim, lim, color="k", linewidth=1)
    ax.set_xlabel("actual power [kW]")
    ax.set_ylabel("predicted power [kW]")
    ax.set_title(f"{data.test_wtg}: outcome model predicted vs actual")
    fig.tight_layout()
    fig.savefig(plots_dir / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def _plot_tau(plots_dir: Path, data: DiagnosticData) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(data.tau, bins=40, color="C3")
    axes[0].set_xlabel("tau(x): absolute uplift [kW]")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"{data.test_wtg}: per-record uplift distribution")
    if data.condition_ws is not None:
        axes[1].scatter(data.condition_ws, data.tau, s=6, alpha=0.3, color="C3")
        axes[1].set_xlabel("reference mean wind speed [m/s]")
        axes[1].set_ylabel("tau(x) [kW]")
        axes[1].set_title("uplift vs wind speed")
    fig.tight_layout()
    fig.savefig(plots_dir / "tau.png", dpi=150)
    plt.close(fig)


def _plot_era5_sweep(plots_dir: Path, data: DiagnosticData) -> None:
    sweep = data.era5_sweep
    if sweep is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sweep["shift_rows"], sweep["corr"], marker=".")
    if data.era5_lag_rows is not None:
        ax.axvline(data.era5_lag_rows, color="k", linestyle="--", label=f"best lag = {data.era5_lag_rows}")
        ax.legend()
    ax.set_xlabel("ERA5 shift [rows]")
    ax.set_ylabel("wind-speed correlation")
    ax.set_title(f"{data.test_wtg}: ERA5-SCADA correlation vs lag")
    fig.tight_layout()
    fig.savefig(plots_dir / "era5_sync.png", dpi=150)
    plt.close(fig)
