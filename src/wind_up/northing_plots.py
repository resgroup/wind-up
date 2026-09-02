"""Show what the northing estimator did, so a user can judge it rather than trust it.

Two views per device, because a northing error and site veer look alike in either one alone:

* **over time** -- the residual against the reference, time-averaged so veer is smeared out,
  before and after correction, with the fitted step function and its changepoints drawn on. This
  is the view that answers "is the corrected direction believable to a degree?".
* **against direction** -- the same residual binned by the reference direction. What is left
  after correction is site veer: the wind direction genuinely differs across a site, and no
  north offset can remove it. A tilt here is expected; a vertical shift is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_up.circular_math import circ_diff, circ_median
from wind_up.northing import NORTH_OFFSET_COL, TIMESTAMP_COL, apply_north_table

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

# The accuracy a corrected direction is judged against, drawn as a band around zero.
BELIEVABLE_DEG = 1.0
_DEFAULT_AVERAGE = pd.Timedelta(days=14)
_DEFAULT_SECTOR_DEG = 30.0
_MIN_ROWS_PER_POINT = 20


def _binned_median(values: pd.Series, *, by: pd.Series, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Circular median of ``values`` in each bin of ``by``, and the bin centres."""
    which = np.digitize(by.to_numpy(dtype=float), bins) - 1
    centres = (bins[:-1] + bins[1:]) / 2
    out = np.full(len(centres), np.nan)
    for b in range(len(centres)):
        rows = values.to_numpy(dtype=float)[which == b]
        rows = rows[np.isfinite(rows)]
        if len(rows) >= _MIN_ROWS_PER_POINT:
            out[b] = circ_median(rows, range_360=False)
    return centres, out


def _time_averaged(residual: pd.Series, *, average: pd.Timedelta) -> pd.Series:
    """Circular median of the residual in each ``average``-long bin, indexed by bin start."""
    grouped = residual.dropna().groupby(pd.Grouper(freq=average))
    return grouped.apply(
        lambda x: circ_median(x.to_numpy(), range_360=False) if len(x) >= _MIN_ROWS_PER_POINT else np.nan
    )


def _step_series(index: pd.DatetimeIndex, north_table: pd.DataFrame) -> np.ndarray:
    """Return the correction the table applies at each timestamp, as a signed angle."""
    return np.asarray(circ_diff(apply_north_table(index, np.zeros(len(index)), north_table=north_table), 0.0))


def plot_northing(
    index: pd.DatetimeIndex,
    direction_deg: np.ndarray,
    *,
    reference_deg: np.ndarray,
    usable: np.ndarray,
    north_table: pd.DataFrame,
    device: str,
    reference_name: str = "reference",
    average: pd.Timedelta = _DEFAULT_AVERAGE,
    sector_deg: float = _DEFAULT_SECTOR_DEG,
    out_dir: Path | None = None,
) -> Figure:
    """Plot one device's northing: the residual over time, and against direction.

    :param index: timestamps of every array
    :param direction_deg: the **raw** direction signal, before correction
    :param reference_deg: the direction it was northed against
    :param usable: the rows the estimate was allowed to use
    :param north_table: the estimated table, as returned by
        :func:`~wind_up.northing.estimate_north_table`
    :param device: name used in the title and filename
    :param reference_name: what the reference is, for the axis labels
    :param average: time-averaging window; longer smears out more veer
    :param sector_deg: direction sector width for the lower panel
    :param out_dir: when given, the figure is saved as ``<device>_northing.png``
    :return: the figure, so a caller can further adjust or close it
    """
    index = pd.DatetimeIndex(index)
    ok = np.asarray(usable, dtype=bool)
    corrected = apply_north_table(index, np.asarray(direction_deg, dtype=float), north_table=north_table)
    before = pd.Series(np.where(ok, circ_diff(direction_deg, reference_deg), np.nan), index=index)
    after = pd.Series(np.where(ok, circ_diff(corrected, reference_deg), np.nan), index=index)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[3, 2])

    smoothed_before = _time_averaged(before, average=average)
    smoothed_after = _time_averaged(after, average=average)
    top.plot(
        before.dropna().index,
        before.dropna().to_numpy(),
        ".",
        color="0.85",
        markersize=1,
        zorder=1,
        label="every record (uncorrected)",
    )
    top.plot(
        smoothed_before.index,
        smoothed_before.to_numpy(),
        color="tab:red",
        linewidth=1.5,
        zorder=3,
        label=f"uncorrected, {_describe(average)} median",
    )
    top.plot(
        smoothed_after.index,
        smoothed_after.to_numpy(),
        color="tab:blue",
        linewidth=1.8,
        zorder=4,
        label=f"corrected, {_describe(average)} median",
    )
    top.plot(
        index,
        -_step_series(index, north_table),
        color="black",
        linewidth=1.2,
        linestyle="--",
        zorder=5,
        label="fitted north offset (negated)",
    )
    top.axhspan(-BELIEVABLE_DEG, BELIEVABLE_DEG, color="tab:blue", alpha=0.12, zorder=0)
    top.axhline(0.0, color="k", linewidth=0.8, zorder=2)
    for changepoint in pd.DatetimeIndex(north_table[TIMESTAMP_COL])[1:]:
        top.axvline(changepoint, color="tab:orange", linewidth=1.2, linestyle=":", zorder=6)
    _annotate_steps(top, north_table)

    span = np.nanpercentile(np.abs(smoothed_before.to_numpy()), 99) if smoothed_before.notna().any() else 10.0
    top.set_ylim(-max(span * 1.4, 12.0), max(span * 1.4, 12.0))
    top.set_ylabel(f"yaw - {reference_name} [deg]")
    top.set_title(
        f"{device}: northing against {reference_name} "
        f"({len(north_table) - 1} changepoint{'s' if len(north_table) != 2 else ''}); "  # noqa: PLR2004
        f"shaded band is +/-{BELIEVABLE_DEG:.0f} deg"
    )
    top.grid(alpha=0.3)
    top.legend(ncol=2, fontsize="small", loc="upper left")

    bins = np.arange(0.0, 360.0 + sector_deg, sector_deg)
    reference = pd.Series(np.where(ok, reference_deg, np.nan), index=index)
    centres, before_by_dir = _binned_median(before, by=reference, bins=bins)
    _, after_by_dir = _binned_median(after, by=reference, bins=bins)
    bottom.plot(centres, before_by_dir, "o-", color="tab:red", label="uncorrected")
    bottom.plot(centres, after_by_dir, "o-", color="tab:blue", label="corrected")
    bottom.axhspan(-BELIEVABLE_DEG, BELIEVABLE_DEG, color="tab:blue", alpha=0.12)
    bottom.axhline(0.0, color="k", linewidth=0.8)
    bottom.set_xlim(0, 360)
    bottom.set_xticks(np.arange(0, 361, 45))
    bottom.set_xlabel(f"{reference_name} [deg]")
    bottom.set_ylabel("median residual [deg]")
    bottom.set_title("residual by direction sector: what remains after correction is site veer")
    bottom.grid(alpha=0.3)
    bottom.legend(fontsize="small")

    fig.tight_layout()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{device}_northing.png", dpi=130)
    return fig


def _describe(average: pd.Timedelta) -> str:
    """Return a short human label for an averaging window."""
    days = average / pd.Timedelta(days=1)
    return f"{days:.0f}-day" if days >= 1 else f"{average / pd.Timedelta(hours=1):.0f}-hour"


def _annotate_steps(ax: plt.Axes, north_table: pd.DataFrame) -> None:
    """Label each changepoint with the size of the step it applies."""
    offsets = north_table[NORTH_OFFSET_COL].to_numpy(dtype=float)
    times = pd.DatetimeIndex(north_table[TIMESTAMP_COL])
    for i in range(1, len(offsets)):
        step = float(circ_diff(offsets[i], offsets[i - 1]))
        ax.annotate(
            f"{step:+.1f}°",
            xy=(times[i], 0.0),
            xytext=(4, 6),
            textcoords="offset points",
            color="tab:orange",
            fontsize="small",
            fontweight="bold",
        )


def circular_spread(values_deg: np.ndarray) -> float:
    """Circular standard deviation (deg) of an angle sample: ``sqrt(-2 ln R)``, R the resultant."""
    finite = np.asarray(values_deg, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:  # noqa: PLR2004 - a spread needs two samples
        return float("nan")
    rad = np.deg2rad(finite)
    resultant = np.hypot(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    if resultant <= 0.0:
        return float("inf")
    return float(np.degrees(np.sqrt(max(-2.0 * np.log(min(resultant, 1.0)), 0.0))))


def _circular_mean(values_deg: np.ndarray) -> float:
    """Circular mean (deg, wrapped to +/-180) of an angle sample."""
    finite = np.asarray(values_deg, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan")
    rad = np.deg2rad(finite)
    return float((np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) + 180.0) % 360.0 - 180.0)


def _by_bin(
    residual: np.ndarray, driver: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Circular mean, circular spread and record count of ``residual`` in each bin of ``driver``."""
    which = np.digitize(driver, edges) - 1
    centres = (edges[:-1] + edges[1:]) / 2
    mean = np.full(len(centres), np.nan)
    spread = np.full(len(centres), np.nan)
    count = np.zeros(len(centres))
    for b in range(len(centres)):
        rows = residual[which == b]
        rows = rows[np.isfinite(rows)]
        count[b] = len(rows)
        if len(rows) >= _MIN_ROWS_PER_POINT:
            mean[b] = _circular_mean(rows)
            spread[b] = circular_spread(rows)
    return centres, mean, spread, count


def plot_residual_conditions(
    residual_deg: np.ndarray,
    *,
    reference_deg: np.ndarray,
    wind_speed: np.ndarray,
    power: np.ndarray,
    rated_power: float,
    title: str,
    sector_deg: float = _DEFAULT_SECTOR_DEG,
    out_dir: Path | None = None,
    filename: str = "residual_conditions.png",
) -> Figure:
    """Mean and spread of the northing residual against direction, wind speed and power.

    The question these answer is whether the residual should be **weighted**: if its spread
    blows up at low power or low wind speed, those records tell you less about where north is
    and should count for less. A flat spread says an unweighted estimate is fine.

    Pass the residual **after** northing, over the rows the estimate was allowed to use.
    """
    fraction = np.asarray(power, dtype=float) / rated_power
    panels = (
        (
            "wind direction [deg]",
            np.asarray(reference_deg, dtype=float),
            np.arange(0.0, 360.0 + sector_deg, sector_deg),
        ),
        ("wind speed [m/s]", np.asarray(wind_speed, dtype=float), np.arange(0.0, 26.0, 1.0)),
        ("power [fraction of rated]", fraction, np.arange(0.0, 1.05, 0.05)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (label, driver, edges) in zip(axes, panels, strict=True):
        centres, mean, spread, count = _by_bin(np.asarray(residual_deg, dtype=float), driver, edges)
        ax.fill_between(centres, mean - spread, mean + spread, color="tab:blue", alpha=0.2, label="+/-1 circular SD")
        ax.plot(centres, mean, "o-", color="tab:blue", markersize=4, label="circular mean")
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.axhspan(-BELIEVABLE_DEG, BELIEVABLE_DEG, color="tab:green", alpha=0.12)
        ax.set_xlabel(label)
        ax.set_ylabel("residual [deg]")
        ax.grid(alpha=0.3)
        counts = ax.twinx()
        counts.bar(centres, count, width=(edges[1] - edges[0]) * 0.85, color="0.8", zorder=0, alpha=0.5)
        counts.set_ylabel("records", color="0.5")
        counts.tick_params(axis="y", colors="0.5")
        counts.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)
    axes[0].legend(fontsize="small", loc="upper left")
    fig.suptitle(f"{title}: northing residual mean and spread by condition")
    fig.tight_layout()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / filename, dpi=130)
    return fig


def plot_northing_farm(
    index: pd.DatetimeIndex,
    *,
    direction_deg: dict[str, np.ndarray],
    reference_deg: np.ndarray,
    usable: dict[str, np.ndarray],
    north_tables: dict[str, pd.DataFrame],
    average: pd.Timedelta = _DEFAULT_AVERAGE,
    reference_name: str = "farm direction",
    out_dir: Path | None = None,
) -> Figure:
    """One panel per device: the time-averaged residual before and after, across the whole farm.

    The overview that answers "did anything move that should not have?" at a glance -- a device
    whose corrected trace leaves the band, or whose changepoints do not line up with a visible
    step, is the one to open :func:`plot_northing` on.
    """
    index = pd.DatetimeIndex(index)
    devices = sorted(direction_deg)
    columns = 3
    rows = int(np.ceil(len(devices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 2.2 * rows), sharex=True, squeeze=False)
    for ax, device in zip(axes.ravel(), devices, strict=False):
        ok = np.asarray(usable[device], dtype=bool)
        raw = np.asarray(direction_deg[device], dtype=float)
        corrected = apply_north_table(index, raw, north_table=north_tables[device])
        before = pd.Series(np.where(ok, circ_diff(raw, reference_deg), np.nan), index=index)
        after = pd.Series(np.where(ok, circ_diff(corrected, reference_deg), np.nan), index=index)
        smoothed_before = _time_averaged(before, average=average)
        smoothed_after = _time_averaged(after, average=average)
        ax.plot(smoothed_before.index, smoothed_before.to_numpy(), color="tab:red", linewidth=1.0)
        ax.plot(smoothed_after.index, smoothed_after.to_numpy(), color="tab:blue", linewidth=1.4)
        ax.axhspan(-BELIEVABLE_DEG, BELIEVABLE_DEG, color="tab:blue", alpha=0.12)
        ax.axhline(0.0, color="k", linewidth=0.6)
        for changepoint in pd.DatetimeIndex(north_tables[device][TIMESTAMP_COL])[1:]:
            ax.axvline(changepoint, color="tab:orange", linewidth=1.0, linestyle=":")
        ax.set_ylim(-15, 15)
        ax.set_title(f"{device} ({len(north_tables[device]) - 1} cp)", fontsize="small")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize="x-small")
    for ax in axes.ravel()[len(devices) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"Northing across the farm vs {reference_name}: {_describe(average)} median residual, "
        f"red before / blue after, band +/-{BELIEVABLE_DEG:.0f} deg"
    )
    fig.tight_layout()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "farm_northing.png", dpi=120)
    return fig
