"""Verification plots for synthetic datasets.

Three panels for the test turbine, all sharing the wind-speed x-axis:

1. *original* power curve (power vs wind speed);
2. *synthetic* power curve, on the same power y-axis as the original so the injected
   upgrade is directly comparable;
3. the per-record **kW change** (synthetic minus original) vs wind speed for the
   treated records, which makes the injected uplift shape easy to read.

Treated (post-upgrade) records are highlighted in the first two panels so you can
confirm the injection lands where expected and leaves the baseline rows untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.synthetic.geometry import wrap180
from benchmarking.synthetic.ground_truth import changed_record_mask
from benchmarking.synthetic.sources.hill_of_towie import HOT_COLUMNS
from benchmarking.synthetic.upgrades import north_calibrated_direction

if TYPE_CHECKING:
    from pathlib import Path

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
