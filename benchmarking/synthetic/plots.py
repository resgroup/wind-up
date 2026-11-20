"""Verification plots for synthetic datasets.

Three panels for the test turbine, all sharing the wind-speed x-axis:

1. *original* power curve (power vs wind speed);
2. *synthetic* power curve, on the same power y-axis as the original so the injected
   upgrade is directly comparable;
3. the per-record **kW change** (synthetic minus original) vs wind speed for the
   treated records, which makes the injected uplift shape easy to read.

Treated (post-upgrade) records are highlighted in the first two panels so you can
confirm the injection lands where expected and leaves the baseline rows untouched.

For wake-steering datasets there are three further diagnostics per steering pair:
:func:`plot_wake_steering_by_direction` (steer angle and kW change vs direction),
:func:`plot_wake_steering_heatmaps` (steer and percent power change as power-vs-direction
heat maps, zoomed to the nadir) and :func:`plot_wake_steering_stability` (percent power
change vs hour of day, month and solar altitude, to confirm the diurnal modulation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.synthetic.geometry import wrap180
from benchmarking.synthetic.ground_truth import changed_record_mask
from benchmarking.synthetic.solar import sin_solar_elevation
from benchmarking.synthetic.sources.hill_of_towie import HOT_COLUMNS
from benchmarking.synthetic.upgrades import north_calibrated_direction

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from benchmarking.synthetic.generator import SyntheticDataset
    from benchmarking.synthetic.schema import ColumnSchema

_BASELINE_STYLE = {"s": 6, "alpha": 0.4, "color": "tab:blue", "label": "baseline rows"}
_TREATED_STYLE = {"s": 6, "alpha": 0.5, "color": "tab:red", "label": "treated rows"}


def plot_power_curve_comparison(
    synthetic_df: pd.DataFrame,
    original_df: pd.DataFrame,
    *,
    test_wtg: str,
    save_path: str | Path | None = None,
    title: str | None = None,
    columns: ColumnSchema = HOT_COLUMNS,
) -> Figure:
    """Plot the test turbine's original vs synthetic power curve plus the kW change.

    The original and synthetic power-curve panels share x (wind speed) and y (power)
    limits and gridlines; a third panel shows the synthetic-minus-original power change
    against wind speed for the treated records. Records the upgrade actually changed
    (NaN-safe) are highlighted in the first two panels.

    :param synthetic_df: source-native synthetic SCADA (all turbines), keyed by ``columns``
    :param original_df: the untouched source-native original SCADA (all turbines)
    :param test_wtg: turbine to plot
    :param save_path: if given, the figure is written here (PNG)
    :param title: optional overall figure title
    :param columns: the source-native column schema the frames are keyed by
    :return: the matplotlib Figure
    """
    original = original_df[original_df[columns.turbine] == test_wtg]
    synthetic = synthetic_df[synthetic_df[columns.turbine] == test_wtg]

    ws = original[columns.wind_speed].to_numpy(dtype=float)
    original_power = original[columns.active_power].to_numpy(dtype=float)
    synthetic_power = synthetic[columns.active_power].to_numpy(dtype=float)

    # Treated = records genuinely modified by the upgrade (NaN downtime rows excluded).
    treated = changed_record_mask(synthetic_power, original_power)

    fig, (ax_orig, ax_syn, ax_delta) = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
    ax_syn.sharey(ax_orig)  # tie the two power-curve y-axes; the kW-change panel is its own

    for ax, power, panel_title in (
        (ax_orig, original_power, "Original"),
        (ax_syn, synthetic_power, "Synthetic"),
    ):
        ax.scatter(ws[~treated], power[~treated], **_BASELINE_STYLE)
        ax.scatter(ws[treated], power[treated], **_TREATED_STYLE)
        ax.set_title(panel_title)
        ax.set_xlabel("Wind speed [m/s]")
        ax.grid(visible=True, alpha=0.3)
        ax.legend(loc="lower right", markerscale=2)
    ax_orig.set_ylabel("Active power [kW]")

    delta = synthetic_power - original_power
    finite_treated = treated & np.isfinite(delta)
    ax_delta.scatter(ws[finite_treated], delta[finite_treated], s=6, alpha=0.5, color="tab:red")
    ax_delta.axhline(0.0, color="k", linewidth=0.8)
    ax_delta.set_title("Injected change (synthetic - original)")
    ax_delta.set_xlabel("Wind speed [m/s]")
    ax_delta.set_ylabel("Power change [kW]")
    ax_delta.grid(visible=True, alpha=0.3)

    fig.suptitle(title if title is not None else f"{test_wtg} power curve: original vs synthetic")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def _wake_steering_description(dataset: SyntheticDataset) -> dict:
    """Return the wake-steering upgrade's recorded description from the dataset's run metadata."""
    for upgrade in dataset.run_metadata.get("upgrades", []):
        if upgrade.get("kind") == "wake_steering":
            return upgrade
    msg = "dataset run_metadata has no wake_steering upgrade to plot"
    raise ValueError(msg)


def _pair_nadir(description: dict, *, upstream: str, downstream: str) -> float:
    """Look up the nadir bearing (deg) of one directed pair in a wake-steering description."""
    for pair in description["pairs"]:
        if pair["upstream"] == upstream and pair["downstream"] == downstream:
            return float(pair["nadir_bearing"])
    msg = f"no wake-steering pair {upstream!r}->{downstream!r} in the dataset"
    raise ValueError(msg)


def _parse_north_offsets(raw: list) -> list[tuple[str, pd.Timestamp, float]]:
    """Coerce recorded ``[turbine, timestamp, offset]`` rows to typed triples (timestamps as UTC)."""
    return [(str(t), pd.Timestamp(ts), float(off)) for (t, ts, off) in raw]


def plot_wake_steering_by_direction(
    dataset: SyntheticDataset,
    *,
    upstream: str,
    downstream: str,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """Plot a wake-steering pair's steer angle and power change against calibrated wind direction.

    Left panel (the geometry arbiter): the upstream turbine's applied nacelle yaw (compass, CW
    positive) versus its north-calibrated wind direction, so the two-way steering can be confirmed
    (negative/CCW to the right of the nadir, positive/CW to the left). Right panel: the per-record
    power change (synthetic minus original) versus calibrated direction for both turbines (upstream
    loss, downstream gain). The nadir and sector edges are marked on both.

    :param dataset: a generated wake-steering :class:`SyntheticDataset`
    :param upstream: steering turbine name (must be an upstream of a derived pair)
    :param downstream: benefitting turbine name of the same pair
    :param save_path: if given, the figure is written here (PNG)
    :param title: optional overall figure title
    :return: the matplotlib Figure
    """
    columns = dataset.columns
    description = _wake_steering_description(dataset)
    nadir = _pair_nadir(description, upstream=upstream, downstream=downstream)
    north_offsets = _parse_north_offsets(description["north_offsets"])
    half_width = float(description["wd_width"]) / 2.0

    def _calibrated(wtg: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        original = dataset.original_df[dataset.original_df[columns.turbine] == wtg]
        synthetic = dataset.synthetic_df[dataset.synthetic_df[columns.turbine] == wtg]
        index = pd.DatetimeIndex(original.index)
        orig_nacelle = original[columns.nacelle_position].to_numpy(dtype=float)
        cal = north_calibrated_direction(index, orig_nacelle, turbine=wtg, north_offsets=north_offsets)
        return (
            cal,
            original[columns.active_power].to_numpy(dtype=float),
            synthetic[columns.active_power].to_numpy(dtype=float),
        )

    def _mark_sector(ax: plt.Axes) -> None:
        ax.axvline(nadir, color="k", linewidth=0.9, linestyle="--", label="nadir")
        for edge in (nadir - half_width, nadir + half_width):
            ax.axvline(edge % 360.0, color="grey", linewidth=0.7, linestyle=":")
        ax.grid(visible=True, alpha=0.3)

    fig, (ax_steer, ax_power) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: applied steer angle vs calibrated direction (from the upstream turbine's nacelle change).
    up_original = dataset.original_df[dataset.original_df[columns.turbine] == upstream]
    up_synthetic = dataset.synthetic_df[dataset.synthetic_df[columns.turbine] == upstream]
    cal_up = north_calibrated_direction(
        pd.DatetimeIndex(up_original.index),
        up_original[columns.nacelle_position].to_numpy(dtype=float),
        turbine=upstream,
        north_offsets=north_offsets,
    )
    steer = wrap180(
        up_synthetic[columns.nacelle_position].to_numpy(dtype=float)
        - up_original[columns.nacelle_position].to_numpy(dtype=float)
    )
    ax_steer.scatter(cal_up, steer, s=6, alpha=0.4, color="tab:purple")
    ax_steer.axhline(0.0, color="k", linewidth=0.8)
    _mark_sector(ax_steer)
    ax_steer.set_xlabel("Calibrated wind direction [deg]")
    ax_steer.set_ylabel("Applied steer angle [deg] (CW +)")
    ax_steer.set_title(f"{upstream}: steer angle vs direction")
    ax_steer.legend(loc="upper right")

    # Right: power change vs calibrated direction for both turbines.
    for wtg, color, role in ((upstream, "tab:red", "steering"), (downstream, "tab:green", "benefitting")):
        cal, orig_power, syn_power = _calibrated(wtg)
        delta = syn_power - orig_power
        finite = np.isfinite(delta)
        ax_power.scatter(cal[finite], delta[finite], s=6, alpha=0.4, color=color, label=f"{wtg} ({role})")
    ax_power.axhline(0.0, color="k", linewidth=0.8)
    _mark_sector(ax_power)
    ax_power.set_xlabel("Calibrated wind direction [deg]")
    ax_power.set_ylabel("Power change [kW]")
    ax_power.set_title("Power change vs direction")
    ax_power.legend(loc="upper right")

    fig.suptitle(title if title is not None else f"Wake steering {upstream} -> {downstream}")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


# Below this steer magnitude a record is treated as "not steering" (toggle off or out of sector).
_STEER_EPS_DEG = 1e-6
_LOSS_STYLE = {"color": "tab:red", "label": "upstream (loss)"}
_GAIN_STYLE = {"color": "tab:green", "label": "downstream (gain)"}


def _pair_records(dataset: SyntheticDataset, *, upstream: str, downstream: str) -> tuple[pd.DataFrame, dict]:
    """Return the steering-active records of a pair, aligned on timestamp, plus geometry metadata.

    The frame carries, per steering-active timestamp (upstream steer angle nonzero): ``dir_off`` (the
    upstream north-calibrated direction as a signed offset from the pair's nadir, deg), ``steer`` (the
    applied nacelle yaw, deg, CW +) and each turbine's original power (``up_power``/``down_power``, kW)
    and synthetic-minus-original power change (``up_delta``/``down_delta``, kW). The metadata dict holds
    the ``nadir`` bearing, sector ``half_width`` and site ``lat``/``lon``.
    """
    columns = dataset.columns
    description = _wake_steering_description(dataset)
    nadir = _pair_nadir(description, upstream=upstream, downstream=downstream)
    north_offsets = _parse_north_offsets(description["north_offsets"])

    def _turbine(wtg: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        original = dataset.original_df[dataset.original_df[columns.turbine] == wtg]
        synthetic = dataset.synthetic_df[dataset.synthetic_df[columns.turbine] == wtg]
        return original, synthetic

    up_original, up_synthetic = _turbine(upstream)
    down_original, down_synthetic = _turbine(downstream)

    up_index = pd.DatetimeIndex(up_original.index)
    cal_up = north_calibrated_direction(
        up_index,
        up_original[columns.nacelle_position].to_numpy(dtype=float),
        turbine=upstream,
        north_offsets=north_offsets,
    )
    steer = wrap180(
        up_synthetic[columns.nacelle_position].to_numpy(dtype=float)
        - up_original[columns.nacelle_position].to_numpy(dtype=float)
    )
    up_power = up_original[columns.active_power].to_numpy(dtype=float)
    down_power = down_original[columns.active_power].to_numpy(dtype=float)
    frame = pd.DataFrame(
        {
            "dir_off": wrap180(cal_up - nadir),
            "steer": steer,
            "up_power": up_power,
            "up_delta": up_synthetic[columns.active_power].to_numpy(dtype=float) - up_power,
        },
        index=up_index,
    )
    down = pd.DataFrame(
        {
            "down_power": down_power,
            "down_delta": down_synthetic[columns.active_power].to_numpy(dtype=float) - down_power,
        },
        index=pd.DatetimeIndex(down_original.index),
    )
    frame = frame.join(down, how="inner")
    active = frame[np.abs(frame["steer"].to_numpy()) > _STEER_EPS_DEG]
    meta = {
        "nadir": nadir,
        "half_width": float(description["wd_width"]) / 2.0,
        "lat": float(description["site_lat"]),
        "lon": float(description["site_lon"]),
    }
    return active, meta


def _binned_mean_2d(
    x: np.ndarray, y: np.ndarray, values: np.ndarray, *, x_edges: np.ndarray, y_edges: np.ndarray
) -> np.ndarray:
    """Mean of ``values`` in each ``(x, y)`` cell as a ``[len(y_edges)-1, len(x_edges)-1]`` grid (NaN where empty)."""
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    counts, _, _ = np.histogram2d(x[finite], y[finite], bins=[x_edges, y_edges])
    sums, _, _ = np.histogram2d(x[finite], y[finite], bins=[x_edges, y_edges], weights=values[finite])
    with np.errstate(invalid="ignore"):
        grid = np.where(counts > 0, sums / counts, np.nan)
    return grid.T


def _binned_energy_pct_2d(
    x: np.ndarray, y: np.ndarray, delta: np.ndarray, base: np.ndarray, *, x_edges: np.ndarray, y_edges: np.ndarray
) -> np.ndarray:
    """Energy-weighted percent change ``100*sum(delta)/sum(base)`` per ``(x, y)`` cell (NaN where base <= 0)."""
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(delta) & np.isfinite(base)
    num, _, _ = np.histogram2d(x[finite], y[finite], bins=[x_edges, y_edges], weights=delta[finite])
    den, _, _ = np.histogram2d(x[finite], y[finite], bins=[x_edges, y_edges], weights=base[finite])
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(den > 0.0, 100.0 * num / den, np.nan)
    return grid.T


def _binned_energy_pct_1d(
    x: np.ndarray, delta: np.ndarray, base: np.ndarray, *, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bin centres and energy-weighted percent change (``100*sum(delta)/sum(base)``) per ``x`` bin."""
    finite = np.isfinite(x) & np.isfinite(delta) & np.isfinite(base)
    num, _ = np.histogram(x[finite], bins=edges, weights=delta[finite])
    den, _ = np.histogram(x[finite], bins=edges, weights=base[finite])
    centres = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(den > 0.0, 100.0 * num / den, np.nan)
    return centres, pct


def _diverging_cmap(name: str) -> Colormap:
    """Copy the named diverging colormap, drawing empty (masked) cells light grey."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("0.9")
    return cmap


def _draw_heatmap(
    fig: Figure,
    ax: plt.Axes,
    grid: np.ndarray,
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    cmap_name: str,
    label: str,
    half_width: float,
) -> None:
    """Draw one power-vs-direction heat map (symmetric diverging scale about zero) with the sector marked."""
    masked = np.ma.masked_invalid(grid)
    finite = grid[np.isfinite(grid)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = vmax or 1.0
    mesh = ax.pcolormesh(
        x_edges, y_edges, masked, cmap=_diverging_cmap(cmap_name), vmin=-vmax, vmax=vmax, shading="flat"
    )
    fig.colorbar(mesh, ax=ax).set_label(label)
    ax.axvline(0.0, color="k", linewidth=0.9, linestyle="--", label="nadir")
    for edge in (-half_width, half_width):
        ax.axvline(edge, color="grey", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Direction from nadir [deg]")
    ax.set_ylabel("Original power [kW]")


def plot_wake_steering_heatmaps(
    dataset: SyntheticDataset,
    *,
    upstream: str,
    downstream: str,
    dir_bin_deg: float = 3.0,
    power_bin_kw: float = 200.0,
    dir_window_deg: float | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """Plot a steering pair's applied steer and percent power change as power-vs-direction heat maps.

    Three panels, all binned by original power (y) and upstream calibrated direction relative to the
    nadir (x), over the steering-active records only (upstream steer angle nonzero): the mean applied
    steer angle, and the energy-weighted percent power change (``100*sum(delta)/sum(original)`` per cell)
    for the upstream (loss) and downstream (gain) turbines. The x-axis is zoomed to the nadir and the
    nadir and sector edges are marked; each panel uses a symmetric diverging scale about zero.

    :param dataset: a generated wake-steering :class:`SyntheticDataset`
    :param upstream: steering turbine name of a derived pair
    :param downstream: benefitting turbine name of the same pair
    :param dir_bin_deg: direction bin width (deg)
    :param power_bin_kw: power bin height (kW)
    :param dir_window_deg: half-range of the direction axis about the nadir; defaults to the sector
        half-width plus a small margin
    :param save_path: if given, the figure is written here (PNG)
    :param title: optional overall figure title
    :return: the matplotlib Figure
    """
    frame, meta = _pair_records(dataset, upstream=upstream, downstream=downstream)
    half_width = meta["half_width"]
    window = half_width + 6.0 if dir_window_deg is None else dir_window_deg
    dir_edges = np.arange(-window, window + dir_bin_deg, dir_bin_deg)
    max_power = float(np.nanmax([frame["up_power"].max(), frame["down_power"].max(), 0.0]))
    power_edges = np.arange(0.0, max_power + power_bin_kw, power_bin_kw)

    direction = frame["dir_off"].to_numpy()
    steer_grid = _binned_mean_2d(
        direction, frame["up_power"].to_numpy(), frame["steer"].to_numpy(), x_edges=dir_edges, y_edges=power_edges
    )
    up_grid = _binned_energy_pct_2d(
        direction,
        frame["up_power"].to_numpy(),
        frame["up_delta"].to_numpy(),
        frame["up_power"].to_numpy(),
        x_edges=dir_edges,
        y_edges=power_edges,
    )
    down_grid = _binned_energy_pct_2d(
        direction,
        frame["down_power"].to_numpy(),
        frame["down_delta"].to_numpy(),
        frame["down_power"].to_numpy(),
        x_edges=dir_edges,
        y_edges=power_edges,
    )

    fig, (ax_steer, ax_up, ax_down) = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    _draw_heatmap(
        fig,
        ax_steer,
        steer_grid,
        x_edges=dir_edges,
        y_edges=power_edges,
        cmap_name="coolwarm",
        label="Applied steer [deg]",
        half_width=half_width,
    )
    ax_steer.set_title(f"{upstream}: applied steer")
    _draw_heatmap(
        fig,
        ax_up,
        up_grid,
        x_edges=dir_edges,
        y_edges=power_edges,
        cmap_name="RdBu_r",
        label="Power change [%]",
        half_width=half_width,
    )
    ax_up.set_title(f"{upstream} (loss): power change")
    _draw_heatmap(
        fig,
        ax_down,
        down_grid,
        x_edges=dir_edges,
        y_edges=power_edges,
        cmap_name="RdBu_r",
        label="Power change [%]",
        half_width=half_width,
    )
    ax_down.set_title(f"{downstream} (gain): power change")
    ax_steer.legend(loc="upper right")

    fig.suptitle(title if title is not None else f"Wake steering {upstream} -> {downstream}: heat maps")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def _stability_panel(
    ax: plt.Axes,
    x: np.ndarray,
    frame: pd.DataFrame,
    *,
    edges: np.ndarray,
    xlabel: str,
    title: str,
) -> None:
    """Draw one panel of energy-weighted percent power change vs ``x`` (binned) for both pair turbines."""
    for delta_col, base_col, style in (
        ("up_delta", "up_power", _LOSS_STYLE),
        ("down_delta", "down_power", _GAIN_STYLE),
    ):
        centres, pct = _binned_energy_pct_1d(x, frame[delta_col].to_numpy(), frame[base_col].to_numpy(), edges=edges)
        ax.plot(centres, pct, marker="o", markersize=4, color=style["color"], label=style["label"])
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(visible=True, alpha=0.3)


def plot_wake_steering_stability(
    dataset: SyntheticDataset,
    *,
    upstream: str,
    downstream: str,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """Plot a steering pair's percent power change vs hour of day, month and solar altitude.

    Three panels, each showing the energy-weighted percent power change (``100*sum(delta)/sum(original)``
    per bin) over the steering-active records for both pair turbines (upstream loss, downstream gain).
    The panels give a visual check of the intended diurnal (atmospheric-stability) modulation: the effect
    magnitude should grow at night (low/negative solar altitude) and shrink by day.

    :param dataset: a generated wake-steering :class:`SyntheticDataset`
    :param upstream: steering turbine name of a derived pair
    :param downstream: benefitting turbine name of the same pair
    :param save_path: if given, the figure is written here (PNG)
    :param title: optional overall figure title
    :return: the matplotlib Figure
    """
    frame, meta = _pair_records(dataset, upstream=upstream, downstream=downstream)
    index = pd.DatetimeIndex(frame.index)
    hour = index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60.0
    month = index.month.to_numpy(dtype=float)
    sin_elev = sin_solar_elevation(index, lat=meta["lat"], lon=meta["lon"])
    altitude = np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))

    alt_lo = float(np.floor(altitude.min() / 5.0) * 5.0)
    alt_hi = float(np.ceil(altitude.max() / 5.0) * 5.0)

    fig, (ax_hour, ax_month, ax_alt) = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    _stability_panel(
        ax_hour,
        hour,
        frame,
        edges=np.arange(0.0, 25.0, 1.0),
        xlabel="Hour of day [UTC]",
        title="Power change vs hour of day",
    )
    _stability_panel(
        ax_month, month, frame, edges=np.arange(0.5, 13.5, 1.0), xlabel="Month of year", title="Power change vs month"
    )
    _stability_panel(
        ax_alt,
        altitude,
        frame,
        edges=np.arange(alt_lo, alt_hi + 5.0, 5.0),
        xlabel="Solar altitude [deg]",
        title="Power change vs solar altitude",
    )
    ax_alt.axvline(0.0, color="grey", linewidth=0.7, linestyle=":", label="horizon")
    ax_hour.set_ylabel("Energy-weighted power change [%]")
    ax_hour.legend(loc="upper right")

    fig.suptitle(title if title is not None else f"Wake steering {upstream} -> {downstream}: stability modulation")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
