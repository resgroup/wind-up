"""Operating-curve and reactive-power diagnostics for the test turbine (feedback 2026-06-26).

These are SCADA normal-operation diagnostics, so they use the **turbine's own** wind speed (the
operationally meaningful signal), not a reference mean — even though own wind speed is never a
*model feature* (it is post-treatment, design-note §3). Columns are labelled by their original
source-native names.

* :func:`plot_ops_curves` — a 2x3 figure (power curve; pitch/rpm vs power; pitch/rpm vs wind
  speed) coloured kept vs removed, so it is both the operating-curve view and the filter check
  (stage: filter).
* :func:`plot_curves_by_upgrade` — pitch/rpm/power vs wind speed split baseline vs upgraded
  (stage: uplift inputs).
* :func:`plot_reactive_vs_active` / :func:`plot_power_factor` — reactive-power behaviour, per
  turbine (stage: inputs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from benchmarking.diagnostics import stages
from benchmarking.diagnostics.style import apply_grid, save_fig
from benchmarking.diagnostics.timeaxis import shade_segments

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from benchmarking.diagnostics.context import DiagnosticContext

_MIN_FINITE = 2
_PF_BUCKET = "MS"  # monthly buckets keep the power-factor / northing timelines smooth

# A plotting "segment": (legend label, boolean row mask, colour).
Segments = list[tuple[str, np.ndarray, str]]


def _own_ws(ctx: DiagnosticContext) -> np.ndarray | None:
    """Return the test turbine's own wind speed, or None if too little of it to plot a curve against."""
    ws = ctx.test_series(ctx.columns.wind_speed).to_numpy(dtype=float)
    return ws if np.isfinite(ws).sum() >= _MIN_FINITE else None


def _ws_label(ctx: DiagnosticContext) -> str:
    return f"{ctx.columns.wind_speed} @ {ctx.test_wtg}"


def _sig_label(ctx: DiagnosticContext, col: str) -> str:
    return f"{col} @ {ctx.test_wtg}"


def _segmented_panel(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, segments: Segments, *, xlabel: str, ylabel: str
) -> None:
    """Scatter ``y`` vs ``x`` once per (label, mask, colour) segment."""
    for label, mask, color in segments:
        ax.scatter(x[mask], y[mask], s=6, alpha=0.3, color=color, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    apply_grid(ax)


def _ops_pair(
    ctx: DiagnosticContext,
    fig: plt.Figure,
    grid: object,
    *,
    col: int,
    x: np.ndarray,
    x_label: str,
    segments: Segments,
    top: str | None,
    bottom: str | None,
) -> None:
    """Draw a shared-x column of two segmented panels (``top`` and ``bottom`` signals vs ``x``)."""
    ax_top = fig.add_subplot(grid[0, col])  # type: ignore[index]
    ax_bottom = fig.add_subplot(grid[1, col], sharex=ax_top)  # type: ignore[index]
    for ax, sig in ((ax_top, top), (ax_bottom, bottom)):
        if sig is None or not ctx.has_column(sig):
            ax.set_visible(False)
            continue
        y = ctx.test_series(sig).to_numpy(dtype=float)
        _segmented_panel(ax, x, y, segments, xlabel=x_label, ylabel=_sig_label(ctx, sig))
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlabel("")


def _ops_curve_figure(
    ctx: DiagnosticContext, *, segments: Segments, title: str, filename: str, stage: str
) -> Path | None:
    """Build the shared 2x3 operating-curve figure: power curve, plus pitch(top)/rpm(bottom) vs power and vs ws."""
    ws = _own_ws(ctx)
    if ws is None or not ctx.has_column(ctx.columns.active_power):
        return None
    cols = ctx.columns
    power = ctx.test_series(cols.active_power).to_numpy(dtype=float)
    ws_label, power_label = _ws_label(ctx), _sig_label(ctx, cols.active_power)

    fig = plt.figure(figsize=(18, 9))
    grid = fig.add_gridspec(2, 3)
    ax_pc = fig.add_subplot(grid[:, 0])
    _segmented_panel(ax_pc, ws, power, segments, xlabel=ws_label, ylabel=power_label)
    ax_pc.set_title("power curve")
    ax_pc.legend()
    _ops_pair(
        ctx, fig, grid, col=1, x=power, x_label=power_label, segments=segments, top=cols.pitch, bottom=cols.gen_rpm
    )
    _ops_pair(ctx, fig, grid, col=2, x=ws, x_label=ws_label, segments=segments, top=cols.pitch, bottom=cols.gen_rpm)

    fig.suptitle(f"{ctx.test_wtg}: {title}")
    path = ctx.stage_dir(stage) / filename
    save_fig(fig, path)
    return path


def plot_ops_curves(ctx: DiagnosticContext) -> Path | None:
    """Draw the operating-curve figure coloured kept vs removed (the filter check)."""
    used = np.asarray(ctx.used_ts, dtype=bool)
    segments: Segments = [("kept", used, "C0"), ("removed", ~used, "C3")]
    return _ops_curve_figure(
        ctx,
        segments=segments,
        title="operating curves (kept vs removed by the row filter)",
        filename="ops_curves.png",
        stage=stages.FILTER,
    )


def plot_ops_curves_kept(ctx: DiagnosticContext) -> Path | None:
    """Draw the operating-curve figure for the KEPT rows only (so removed points cannot mask them)."""
    used = np.asarray(ctx.used_ts, dtype=bool)
    segments: Segments = [("kept", used, "C0")]
    return _ops_curve_figure(
        ctx,
        segments=segments,
        title="operating curves (used rows only)",
        filename="ops_curves_kept_only.png",
        stage=stages.FILTER,
    )


def plot_curves_by_upgrade(ctx: DiagnosticContext) -> Path | None:
    """Draw the operating-curve figure coloured baseline vs upgraded, over the used rows."""
    used = np.asarray(ctx.used_ts, dtype=bool)
    segments: Segments = [("baseline", used & ctx.baseline_ts, "C0"), ("upgraded", used & ctx.upgraded_ts, "C1")]
    return _ops_curve_figure(
        ctx,
        segments=segments,
        title="operating curves by upgrade state (used rows)",
        filename="ops_curves_by_upgrade.png",
        stage=stages.UPLIFT_INPUTS,
    )


def plot_reactive_vs_active(ctx: DiagnosticContext) -> Path | None:
    """Reactive vs active power by upgrade state, one panel per turbine. None if no reactive tag."""
    if not ctx.has_column(ctx.columns.reactive_power):
        return None
    turbines = [ctx.test_wtg, *ctx.references()]
    ncols = min(3, len(turbines))
    nrows = int(np.ceil(len(turbines) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    flat = axes.flatten()
    for ax, turbine in zip(flat, turbines, strict=False):
        active = ctx.turbine_series(turbine, ctx.columns.active_power).to_numpy(dtype=float)
        reactive = ctx.turbine_series(turbine, ctx.columns.reactive_power).to_numpy(dtype=float)
        for seg_label, seg, color in (("baseline", ctx.baseline_ts, "C0"), ("upgraded", ctx.upgraded_ts, "C1")):
            ax.scatter(active[seg], reactive[seg], s=6, alpha=0.3, color=color, label=seg_label)
        ax.set_title(f"{turbine}{' (test)' if turbine == ctx.test_wtg else ''}")
        ax.set_xlabel(ctx.columns.active_power)
        ax.set_ylabel(ctx.columns.reactive_power)  # type: ignore[arg-type]
        apply_grid(ax)
        ax.legend()
    for ax in flat[len(turbines) :]:
        ax.set_visible(False)
    fig.suptitle("reactive vs active power by upgrade state")
    path = ctx.stage_dir(stages.INPUTS) / "reactive_vs_active.png"
    save_fig(fig, path)
    return path


def plot_power_factor(ctx: DiagnosticContext) -> Path | None:
    """Active-power-weighted monthly power factor over time, per turbine. None if no reactive tag."""
    if not ctx.has_column(ctx.columns.reactive_power):
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    shade_segments(ax, ctx)
    for turbine in [ctx.test_wtg, *ctx.references()]:
        monthly = _monthly_power_factor(ctx, turbine)
        label = f"{turbine}{' (test)' if turbine == ctx.test_wtg else ''}"
        ax.plot(monthly.index.to_numpy(), monthly.to_numpy(), linewidth=1.0, marker=".", markersize=3, label=label)
    ax.set_xlabel("date")
    ax.set_ylabel("power factor (active-power-weighted monthly mean)")
    ax.set_title("power factor over time — |P| / sqrt(P^2 + Q^2)")
    apply_grid(ax)
    ax.legend(ncol=2, fontsize="small")
    path = ctx.stage_dir(stages.INPUTS) / "power_factor.png"
    save_fig(fig, path)
    return path


def _monthly_power_factor(ctx: DiagnosticContext, turbine: str) -> pd.Series:
    """Monthly active-power-weighted mean power factor for one turbine."""
    active = ctx.turbine_series(turbine, ctx.columns.active_power)
    reactive = ctx.turbine_series(turbine, ctx.columns.reactive_power)
    apparent = np.sqrt(active**2 + reactive**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        pf = (active.abs() / apparent).where(apparent > 0)
    weight = active.abs()
    num = (pf * weight).resample(_PF_BUCKET).sum(min_count=1)
    den = weight.where(pf.notna()).resample(_PF_BUCKET).sum(min_count=1)
    return num / den
