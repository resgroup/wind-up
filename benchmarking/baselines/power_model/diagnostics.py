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

from benchmarking.baselines.power_model.features import QUALIFIER, SITE_FEATURE_UNITS
from benchmarking.diagnostics import stages
from benchmarking.diagnostics.density import density_scatter
from benchmarking.diagnostics.features import category_shares_figure
from benchmarking.diagnostics.style import apply_grid, save_fig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from benchmarking.baselines.power_model.shape import ShapeResult

logger = logging.getLogger(__name__)

_SEGMENTS = ("all", "baseline", "upgraded")
_TOP_FEATURES_LOGGED = 12
# Feature histograms of a signal below this count stay at the top level rather than in a folder.
_MIN_FOR_OWN_FOLDER = 2
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
    n_refs: int  # candidate references in the pool, screened ones included
    constant_features: tuple[str, ...] = ()  # left out of the model: one value over the selected rows


def feature_importance_long(data: DiagnosticData) -> pd.DataFrame:
    """Long table of LightGBM gain/split importance for the (single) outcome power model.

    An alternative learner injected via the model-factory seam has no ``booster_``; the table then
    carries NaN importances (the feature *catalogue* still works) rather than failing the run.
    """
    names = list(data.feature_names)
    booster = getattr(data.outcome_model, "booster_", None)
    if booster is None:
        return pd.DataFrame({"feature": names, "gain": np.nan, "split_count": np.nan})
    return pd.DataFrame(
        {
            "feature": names,
            "gain": booster.feature_importance(importance_type="gain"),
            "split_count": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False, ignore_index=True)


def log_top_features(importance: pd.DataFrame, *, active_power_col: str = "") -> None:
    """Say what the model leant on: a line at INFO, the table at DEBUG, a warning when it is odd.

    A neighbour's active power should lead: it is the same weather, measured. Anything else on top
    means the model is explaining this turbine with something other than its neighbours' output,
    which is worth a look before the number is believed.
    """
    if importance.empty:
        return
    top = importance.head(_TOP_FEATURES_LOGGED)
    logger.debug("power_model feature gains:\n%s", top[["feature", "gain"]].to_string(index=False, float_format="%.0f"))
    leader = str(top.iloc[0]["feature"])
    names = ", ".join(str(r.feature) for r in top.head(3).itertuples())
    logger.info("power_model leant on %s (top 3 by gain); %d features in the model", names, len(importance))
    if active_power_col and not _is_neighbour_power(leader, active_power_col=active_power_col):
        logger.warning(
            "power_model's strongest feature is %r, not a neighbour's %s. A neighbour's power is the "
            "expected leader, so check the feature importance before believing the estimate.",
            leader,
            active_power_col,
        )


def _is_neighbour_power(feature: str, *, active_power_col: str) -> bool:
    """Whether ``feature`` is another turbine's active power, the expected strongest predictor."""
    tag, separator, turbine = feature.partition(QUALIFIER)
    return bool(separator) and bool(turbine) and tag == active_power_col


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
                "time_calculated": pd.Timestamp.utcnow(),
            }
        ]
    )


def feature_catalogue(data: DiagnosticData) -> pd.DataFrame:
    """One row per feature: source tag/turbine, coverage, basic stats, gain, |corr| with power."""
    importance = feature_importance_long(data).set_index("feature")
    rows = []
    for feature in data.feature_names:
        col = _numeric(data.feature_values[feature])
        finite = np.isfinite(col)
        tag, _, source = feature.partition(QUALIFIER)
        rows.append(
            {
                "feature": feature,
                "source_tag": tag,
                "turbine": source,
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


def _numeric(values: pd.Series) -> np.ndarray:
    """Return ``values`` as floats; a categorical becomes its category codes, NaN where missing."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        codes = values.cat.codes.to_numpy(dtype=float)
        return np.where(codes < 0, np.nan, codes)
    return values.to_numpy(dtype=float)


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


def csv_name(kind: str, *, run_name: str, ts: str) -> str:
    """Return a diagnostics CSV's file name: ``<run_name>_<kind>_<ts>.csv``, or ``<kind>.csv`` with no run name."""
    return f"{run_name}_{kind}_{ts}.csv" if run_name else f"{kind}.csv"


def write_csvs(run_dir: Path, run_name: str, ts: str, data: DiagnosticData) -> pd.DataFrame:
    """Write the data-stats, results, feature-importance and feature-catalogue CSVs; return importance."""
    segment_stats(data).to_csv(run_dir / csv_name("data_stats", run_name=run_name, ts=ts), index=False)
    results_row(data).to_csv(run_dir / csv_name("results", run_name=run_name, ts=ts), index=False)
    importance = feature_importance_long(data)
    importance.to_csv(run_dir / csv_name("feature_importance", run_name=run_name, ts=ts), index=False)
    feature_catalogue(data).to_csv(run_dir / csv_name("feature_catalogue", run_name=run_name, ts=ts), index=False)
    return importance


_AXIS_UNITS = {
    "predicted_power": "predicted power (kW)",
    "freestream_ws": "free-stream wind speed (m/s)",
    "direction": "direction (deg)",
    "solar_elevation": "solar elevation (deg)",
    "waked": "waked (0 front row, 1 waked)",
}


def plot_uplift_shape(plots_dir: Path, *, shape: ShapeResult, drift: pd.DataFrame, test_wtg: str) -> None:
    """Write one plot per axis, one per subsets family, and the reference wind speed drift.

    Axis plots are ``uplift_<axis>.png``; subset plots ``uplift_subsets_<axis>.png``; the drift plot
    ``reference_wind_speed_drift.png``.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    for condition, rows in shape.frame.groupby("condition", sort=False):
        save_fig(_axis_figure(rows, condition=str(condition), test_wtg=test_wtg), plots_dir / f"uplift_{condition}.png")
    shown = shape.subsets.loc[~shape.subsets["too_thin"].astype(bool)]
    for axis, rows in shown.groupby("axis", sort=False):
        save_fig(_subsets_figure(rows, axis=str(axis), test_wtg=test_wtg), plots_dir / f"uplift_subsets_{axis}.png")
    if not drift.empty:
        save_fig(_drift_figure(drift, test_wtg=test_wtg), plots_dir / "reference_wind_speed_drift.png")


def _axis_figure(rows: pd.DataFrame, *, condition: str, test_wtg: str) -> plt.Figure:
    """Build the uplift per bin with its one-sigma bar, over the hours per bin."""
    labels = rows["condition_bin"].astype(str).to_numpy()
    x = np.arange(len(labels))
    uplift = 100 * rows["p50_uplift"].to_numpy(dtype=float)
    sigma = 100 * rows["sigma_uplift"].to_numpy(dtype=float)
    thin = rows["thin"].to_numpy(dtype=bool)
    fig, (ax, ax_hours) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, height_ratios=[3, 1])
    ax.errorbar(x[~thin], uplift[~thin], yerr=sigma[~thin], fmt="o", color="C0", capsize=3, label="uplift ± 1 sigma")
    if thin.any():
        ax.errorbar(
            x[thin],
            uplift[thin],
            yerr=sigma[thin],
            fmt="o",
            color="C0",
            mfc="white",
            capsize=3,
            label="thin bin (under the minimum hours)",
        )
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_ylabel("uplift (%)")
    ax.set_title(f"{test_wtg}: uplift against {_AXIS_UNITS.get(condition, condition)}")
    ax.legend(loc="best")
    apply_grid(ax)
    ax_hours.bar(x, rows["hours"].to_numpy(dtype=float), color="0.6")
    ax_hours.set_ylabel("after-period hours")
    ax_hours.set_xticks(x, labels, rotation=90)
    ax_hours.set_xlabel(_AXIS_UNITS.get(condition, condition))
    apply_grid(ax_hours)
    fig.tight_layout()
    return fig


def _subsets_figure(rows: pd.DataFrame, *, axis: str, test_wtg: str) -> plt.Figure:
    """Build one panel per subset: its own power shape beside the whole after period's on the same bins."""
    subsets = list(dict.fromkeys(rows["subset"]))
    fig, axes = plt.subplots(1, len(subsets), figsize=(4.5 * len(subsets), 4.5), squeeze=False, sharey=True)
    for ax, subset in zip(axes[0], subsets, strict=True):
        part = rows[rows["subset"] == subset]
        x = np.arange(len(part))
        ax.errorbar(
            x - 0.1,
            100 * part["overall_uplift"].to_numpy(dtype=float),
            yerr=100 * part["overall_sigma"].to_numpy(dtype=float),
            fmt="s",
            color="0.5",
            capsize=3,
            label="whole after period",
        )
        ax.errorbar(
            x + 0.1,
            100 * part["p50_uplift"].to_numpy(dtype=float),
            yerr=100 * part["sigma_sampling"].to_numpy(dtype=float),
            fmt="o",
            color="C3" if bool(part["flagged"].any()) else "C0",
            capsize=3,
            label="this subset",
        )
        ax.set_xticks(x, part["power_bin"].astype(str).to_numpy(), rotation=90)
        ax.set_title(f"{subset}{' (differs)' if bool(part['flagged'].any()) else ''}")
        ax.set_xlabel("predicted power (kW)")
        apply_grid(ax)
    axes[0][0].set_ylabel("uplift, unshifted (%)")
    axes[0][0].legend(loc="best")
    fig.suptitle(f"{test_wtg}: the power shape within each {_AXIS_UNITS.get(axis, axis)} subset")
    fig.tight_layout()
    return fig


def _drift_figure(drift: pd.DataFrame, *, test_wtg: str) -> plt.Figure:
    """Build each reference's monthly wind speed over the mean of the others."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for reference, rows in drift.groupby("reference"):
        flagged = bool(rows["flagged"].iloc[0])
        ax.plot(
            pd.PeriodIndex(rows["month"], freq="M").to_timestamp(),
            rows["ratio"].to_numpy(dtype=float),
            marker="o",
            linestyle="-" if flagged else ":",
            label=f"{reference}{' (flagged)' if flagged else ''}",
        )
    ax.set_ylabel("wind speed / mean of the other references")
    ax.set_title(f"{test_wtg}: power references' anemometers against each other (not corrected)")
    ax.legend(loc="best")
    apply_grid(ax)
    fig.tight_layout()
    return fig


def save_plots(plots_dir: Path, data: DiagnosticData, importance: pd.DataFrame) -> None:
    """Write the power-model diagnostic plots into their analysis-stage subfolders."""
    model_dir = plots_dir / stages.RELATE_REFERENCES
    model_dir.mkdir(parents=True, exist_ok=True)
    _plot_importance(model_dir, importance, test_wtg=data.test_wtg, left_out=data.constant_features)
    _plot_predicted_vs_actual(model_dir, data)

    results_dir = plots_dir / stages.UPLIFT
    results_dir.mkdir(parents=True, exist_ok=True)
    _plot_actual_vs_counterfactual_timeseries(results_dir, data)

    inputs_dir = plots_dir / stages.FEATURES
    inputs_dir.mkdir(parents=True, exist_ok=True)
    _plot_feature_overview(inputs_dir, feature_catalogue(data))
    _save_feature_histograms(inputs_dir / "feature_histograms", data)


def _plot_importance(plots_dir: Path, importance: pd.DataFrame, *, test_wtg: str, left_out: Sequence[str] = ()) -> None:
    """Write the top features by gain for the single power model."""
    save_fig(_importance_figure(importance, test_wtg=test_wtg, left_out=left_out), plots_dir / "feature_importance.png")


def _importance_figure(importance: pd.DataFrame, *, test_wtg: str, left_out: Sequence[str] = ()) -> plt.Figure:
    """Draw the top 20 features by gain on a log scale; a feature of no gain has no bar.

    ``left_out`` names the features left out of the model for holding one value in its records.
    """
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(6.0, 0.4 * len(top))))
    ax.barh(top["feature"], top["gain"], color="C0")
    ax.set_xscale("log")
    ax.set_xlabel("gain (log scale)")
    ax.set_title(f"{test_wtg}: power-model feature importance")
    if left_out:
        ax.text(
            0.99,
            0.02,
            "\n".join(["left out, one value in every record used:", *left_out]),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize="x-small",
            color="0.3",
        )
    apply_grid(ax)
    return fig


def _plot_feature_overview(plots_dir: Path, catalogue: pd.DataFrame) -> None:
    """Write every feature's gain, coloured by coverage."""
    save_fig(_feature_overview_figure(catalogue), plots_dir / "feature_overview.png")


def _feature_overview_figure(catalogue: pd.DataFrame) -> plt.Figure:
    """Draw every feature as a bar of gain on a log scale, coloured by coverage; a feature of no gain has no bar."""
    df = catalogue.sort_values("gain", ascending=True)
    norm = Normalize(vmin=0.0, vmax=100.0)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(12, max(6.0, 0.3 * len(df))))
    ax.barh(df["feature"], df["gain"], color=cmap(norm(df["coverage_pct"].to_numpy())))
    ax.set_xscale("log")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="coverage [%]")
    ax.set_xlabel("outcome-model gain (log scale)")
    ax.set_title("all features: importance (bar) and coverage (colour) — short + cold = drop candidate")
    apply_grid(ax)
    return fig


def _save_feature_histograms(out_dir: Path, data: DiagnosticData) -> None:
    """One baseline-vs-upgraded distribution plot per model input feature.

    A numeric feature gets a density histogram; a categorical one gets the share of records in each
    category. A site feature is labelled with its units.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = np.asarray(data.selected_all, dtype=bool)
    treated_sel = np.asarray(data.treated_all, dtype=bool)[sel]  # aligned to feature_values rows
    groups = _histogram_groups(data.feature_names)
    for feature in data.feature_names:
        values = data.feature_values[feature]
        label = feature
        if isinstance(values.dtype, pd.CategoricalDtype):
            fig = category_shares_figure(
                values,
                segments={"baseline": ~treated_sel, "upgraded": treated_sel},
                title=f"{data.test_wtg}: {label}\n(used rows, baseline vs upgraded)",
            )
        else:
            fig = _histogram_figure(_numeric(values), treated=treated_sel, label=label, test_wtg=data.test_wtg)
        save_fig(fig, groups[feature](out_dir) / f"{_safe_filename(feature)}.png")


def _histogram_figure(vals: np.ndarray, *, treated: np.ndarray, label: str, test_wtg: str) -> plt.Figure:
    """Density histograms of ``vals`` in the baseline and upgraded rows."""
    bins = _robust_bins(vals)
    fig, ax = plt.subplots(figsize=(7, 5))
    for seg_label, seg, color in (("baseline", ~treated, "C0"), ("upgraded", treated, "C1")):
        seg_vals = vals[seg & np.isfinite(vals)]
        if seg_vals.size:
            ax.hist(seg_vals, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=color, label=seg_label)
    unit = SITE_FEATURE_UNITS.get(label)
    ax.set_xlabel(f"{label} [{unit}]" if unit else label)
    ax.set_ylabel("density")
    ax.set_title(f"{test_wtg}: {label}\n(used rows, baseline vs upgraded)")
    apply_grid(ax)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    return fig


def _signal_of(feature: str) -> str:
    """Return the signal a feature measures: its tag, without the turbine or a sin/cos companion.

    ``northed_wtc_NacelPos_mean_cos @ T01`` and its sine are both the nacelle position, and every
    turbine's copy is the same signal.
    """
    tag = feature.partition(QUALIFIER)[0]
    for companion in ("_sin", "_cos"):
        tag = tag.removesuffix(companion)
    return tag


def _histogram_groups(features: Sequence[str]) -> dict[str, Callable[[Path], Path]]:
    """Map each feature to where its histogram goes: a per-signal folder, or the root.

    A signal carried by more than one turbine gets a folder of its own; every other feature stays
    at the top level.
    """
    turbines: dict[str, set[str]] = {}
    for feature in features:
        turbines.setdefault(_signal_of(feature), set()).add(feature.partition(QUALIFIER)[2])

    def place(feature: str) -> Callable[[Path], Path]:
        signal = _signal_of(feature)
        if len(turbines[signal]) < _MIN_FOR_OWN_FOLDER:
            return lambda root: root
        return lambda root: _made(root / _safe_filename(signal))

    return {f: place(f) for f in features}


def _made(path: Path) -> Path:
    """Return ``path``, created if it is not there yet."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _robust_bins(values: np.ndarray, *, bins: int = 30) -> list[float] | int:
    """Bin edges over every finite value."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return bins
    lo, hi = float(finite.min()), float(finite.max())
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


def _plot_actual_vs_counterfactual_timeseries(plots_dir: Path, data: DiagnosticData) -> None:
    """Write the weekly actual and counterfactual energy over the upgraded window."""
    save_fig(_energy_timeseries_figure(data), plots_dir / "actual_vs_counterfactual_timeseries.png")


def _energy_timeseries_figure(data: DiagnosticData) -> plt.Figure:
    """Draw the weekly actual and counterfactual energy, and below it each week's uplift."""
    timebase_hours = data.timebase / pd.Timedelta(hours=1)
    actual = pd.Series(data.y_upgraded * timebase_hours / 1000.0, index=data.upgraded_ts)
    counter = pd.Series(data.pred_upgraded * timebase_hours / 1000.0, index=data.upgraded_ts)
    weekly_actual = actual.resample("W").sum(min_count=1)
    weekly_counter = counter.resample("W").sum(min_count=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        weekly_uplift = 100.0 * (weekly_actual / weekly_counter.where(weekly_counter > 0) - 1.0)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, layout="constrained"
    )
    weeks = weekly_actual.index.to_numpy()
    top.plot(weeks, weekly_actual.to_numpy(), marker=".", color="C1", label="actual (upgraded)")
    top.plot(weeks, weekly_counter.to_numpy(), marker=".", color="C0", label="counterfactual (model)")
    top.set_ylabel("weekly energy [MWh]")
    top.set_title(f"{data.test_wtg}: actual vs counterfactual weekly energy over the records used")
    top.legend()
    bottom.bar(weeks, weekly_uplift.to_numpy(), width=5.0, color="C2")
    bottom.axhline(100.0 * data.overall_uplift, color="k", linestyle="--", linewidth=1.0, label="overall uplift")
    bottom.axhline(0.0, color="0.5", linewidth=0.8)
    bottom.set_ylabel("weekly uplift [%]")
    bottom.set_xlabel("week ending")
    bottom.legend()
    for ax in (top, bottom):
        apply_grid(ax)
    return fig
