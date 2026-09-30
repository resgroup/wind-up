"""What the waking feature carries, for the turbines reduced to it.

A screened reference or another changed turbine contributes one column: whether it was producing
enough to wake its neighbours. Two plots say whether that column is doing what it should:

* :func:`plot_waking_layout` -- where those turbines sit, so a reader can see which of the test
  turbine's neighbours are power-free rather than reading a list of names;
* :func:`plot_waking_fractions` -- how often each turbine was waking in the baseline and in the
  treated period. A turbine whose waking fraction moves sharply between the two was running
  differently across the changeover, which a single wake column cannot express.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.diagnostics import stages
from benchmarking.diagnostics.style import apply_grid, save_fig
from wind_up.geodesy import local_east_north

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# A waking fraction moving by more than this between the periods is called out on the plot.
NOTABLE_SHIFT = 0.10


def waking_fractions(
    scada: pd.DataFrame,
    *,
    turbine_col: str,
    active_power_col: str,
    threshold_kw: float,
    treated: pd.Series,
) -> pd.DataFrame:
    """Return each turbine's waking fraction in the baseline and treated periods.

    :param scada: long-format SCADA
    :param turbine_col: its turbine-identifier column
    :param active_power_col: the power the waking threshold is applied to
    :param threshold_kw: power at or above which a turbine wakes its neighbours
    :param treated: per-timestamp treatment flag, covering ``scada``'s index
    :return: one row per turbine with ``baseline``, ``treated`` and their difference, NaN where a
        period has no finite power for that turbine
    """
    rows = []
    for turbine, frame in scada.groupby(turbine_col, sort=True):
        power = pd.Series(frame[active_power_col].to_numpy(dtype=float), index=pd.DatetimeIndex(frame.index))
        is_treated = treated.reindex(power.index).to_numpy(dtype=bool)
        waking = (power >= threshold_kw).to_numpy()
        finite = np.isfinite(power.to_numpy())
        fractions = {}
        for label, period in (("baseline", ~is_treated), ("treated", is_treated)):
            usable = period & finite
            fractions[label] = float(waking[usable].mean()) if usable.any() else float("nan")
        rows.append({"turbine": str(turbine), **fractions, "shift": fractions["treated"] - fractions["baseline"]})
    return pd.DataFrame(rows)


def plot_waking_fractions(
    out_dir: Path,
    *,
    fractions: pd.DataFrame,
    test_wtg: str,
    power_free: Sequence[str],
    threshold_kw: float,
) -> Path:
    """Bar the baseline and treated waking fraction per turbine, flagging those that moved."""
    free = set(power_free)
    frame = fractions.sort_values("turbine", ignore_index=True)
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(max(9.0, 0.55 * len(frame) + 3.0), 5.5))
    ax.bar(x - 0.2, frame["baseline"], width=0.4, color="C0", label="baseline")
    ax.bar(x + 0.2, frame["treated"], width=0.4, color="C1", label="treated")
    for i, row in frame.iterrows():
        if np.isfinite(row["shift"]) and abs(row["shift"]) >= NOTABLE_SHIFT:
            top = float(np.nanmax([row["baseline"], row["treated"]]))
            ax.annotate(f"{row['shift']:+.0%}", (i, top), ha="center", va="bottom", fontsize="small", color="C3")
    labels = [f"{t}{' (test)' if t == test_wtg else ''}{' *' if t in free else ''}" for t in frame["turbine"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize="small")
    ax.set_ylabel(f"fraction of records at or above {threshold_kw:.0f} kW")
    ax.set_title(
        f"{test_wtg}: how often each turbine was waking its neighbours\n"
        f"* contributes this column and nothing else; a shift of {NOTABLE_SHIFT:.0%} or more is labelled"
    )
    ax.set_ylim(0, 1)
    apply_grid(ax)
    ax.legend()
    path = out_dir / "waking_fractions.png"
    save_fig(fig, path)
    return path


def plot_waking_layout(
    out_dir: Path,
    *,
    coords: dict[str, tuple[float, float]],
    test_wtg: str,
    references: Sequence[str],
    power_free: Sequence[str],
) -> Path | None:
    """Map the farm, marking the turbines reduced to their waking column. None without coordinates."""
    named = [t for t in coords if t in {test_wtg, *references, *power_free}]
    if not named:
        return None
    east, north = local_east_north(
        latitudes=pd.Series([coords[t][0] for t in named]), longitudes=pd.Series([coords[t][1] for t in named])
    )
    free = set(power_free)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, turbine in enumerate(named):
        if turbine == test_wtg:
            color, marker, size, label = "C3", "*", 260.0, "test turbine"
        elif turbine in free:
            color, marker, size, label = "C1", "s", 90.0, "waking column only"
        else:
            color, marker, size, label = "C0", "o", 60.0, "reference"
        drawn_already = label in ax.get_legend_handles_labels()[1]
        ax.scatter(
            east[i],
            north[i],
            color=color,
            marker=marker,
            s=size,
            zorder=3,
            label=None if drawn_already else label,
        )
        ax.annotate(turbine, (east[i], north[i]), textcoords="offset points", xytext=(6, 4), fontsize="small")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{test_wtg}: which turbines contribute their wake alone")
    apply_grid(ax)
    ax.legend(loc="best", fontsize="small")
    path = out_dir / "waking_layout.png"
    save_fig(fig, path)
    return path


def write_waking_diagnostics(
    run_dir: Path,
    *,
    scada: pd.DataFrame,
    turbine_col: str,
    active_power_col: str,
    threshold_kw: float,
    treated: pd.Series,
    test_wtg: str,
    references: Sequence[str],
    power_free: Sequence[str],
    coords: dict[str, tuple[float, float]] | None,
) -> list[Path]:
    """Write both waking plots into the feature-engineering stage; returns what was written."""
    out_dir = run_dir / "plots" / stages.FEATURE_ENG
    out_dir.mkdir(parents=True, exist_ok=True)
    fractions = waking_fractions(
        scada,
        turbine_col=turbine_col,
        active_power_col=active_power_col,
        threshold_kw=threshold_kw,
        treated=treated,
    )
    written = [
        plot_waking_fractions(
            out_dir, fractions=fractions, test_wtg=test_wtg, power_free=power_free, threshold_kw=threshold_kw
        )
    ]
    if coords:
        drawn = plot_waking_layout(
            out_dir, coords=coords, test_wtg=test_wtg, references=references, power_free=power_free
        )
        if drawn is not None:
            written.append(drawn)
    fractions.to_csv(out_dir / "waking_fractions.csv", index=False)
    return written
