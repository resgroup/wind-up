"""Campaign-level uplift plots: the test turbines against the references that did not change.

The report's numbers answer "how big"; these answer "is it distinguishable". References are
turbines nothing happened to, so their readings are what this campaign can resolve: a test turbine
inside that spread is not distinguishable from zero, however large its number.

Screened references are drawn but kept out of the spread they would otherwise inflate -- the screen
ruled them out of the estimate, and a reading it rejected is not a measure of campaign noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.diagnostics.style import apply_grid, save_fig
from wind_up.farm import TurbineUplift, farm_uplift

if TYPE_CHECKING:
    from pathlib import Path

PERCENT = 100.0


def per_reference_uplifts(stability: pd.DataFrame) -> pd.DataFrame:
    """Return one row per reference from the per-test-turbine reference table.

    Each reference is estimated once per test turbine against the same pool over the same contrast,
    so the readings agree; the median is taken rather than assumed, and a reference screened for
    any test turbine counts as screened.
    """
    if stability.empty:
        # Typed, not merely empty: an object-dtype `screened` makes ~mask read as column selection.
        return pd.DataFrame(
            {
                "turbine": pd.Series(dtype="object"),
                "uplift": pd.Series(dtype="float64"),
                "actual_energy": pd.Series(dtype="float64"),
                "n_records": pd.Series(dtype="int64"),
                "screened": pd.Series(dtype="bool"),
            }
        )
    grouped = stability.groupby("turbine", sort=True)
    return pd.DataFrame(
        {
            "turbine": [str(name) for name, _ in grouped],
            "uplift": grouped["uplift"].median().to_numpy(),
            "actual_energy": grouped["actual_energy"].median().to_numpy(),
            "n_records": grouped["n_records"].median().astype(int).to_numpy(),
            "screened": grouped["screened"].any().to_numpy(),
        }
    )


def plot_uplift_distributions(out_dir: Path, *, per_turbine: pd.DataFrame, references: pd.DataFrame) -> Path:
    """Violin the test-turbine uplifts beside the reference readings, on one axis."""
    tests = per_turbine["estimate"].to_numpy(dtype=float) * PERCENT
    kept = references[~references["screened"]]
    refs = kept["uplift"].to_numpy(dtype=float) * PERCENT

    fig, ax = plt.subplots(figsize=(8, 6))
    bodies = [values for values in (tests, refs) if len(values) > 1]
    positions = [pos for pos, values in zip((1, 2), (tests, refs), strict=True) if len(values) > 1]
    if bodies:
        drawn = ax.violinplot(bodies, positions=positions, widths=0.7, showextrema=False)
        # matplotlib types the dict's values as one Collection artist; it is a list of them
        shapes = cast("list[Any]", drawn["bodies"])
        for body, position in zip(shapes, positions, strict=True):
            body.set_facecolor("C0" if position == 1 else "C1")
            body.set_alpha(0.35)
    _strip(ax, 1, tests, per_turbine["test_wtg"], colour="C0")
    _strip(ax, 2, refs, kept["turbine"], colour="C1")
    screened = references[references["screened"]]
    if not screened.empty:
        _strip(ax, 2, screened["uplift"].to_numpy(dtype=float) * PERCENT, screened["turbine"], colour="C3", marker="x")
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"test turbines ({len(tests)})", f"references ({len(refs)} kept)"])
    ax.set_ylabel("uplift [%]")
    ax.set_title(
        "uplift of the turbines that changed, against the turbines that did not\n"
        "references read the campaign's noise; screened ones (x) are outside the spread"
    )
    apply_grid(ax)
    path = out_dir / "uplift_distributions.png"
    save_fig(fig, path)
    return path


def plot_per_turbine_uplifts(out_dir: Path, *, per_turbine: pd.DataFrame, references: pd.DataFrame) -> Path:
    """Bar every turbine's uplift: the test turbines first, then the references."""
    tests = per_turbine.sort_values("test_wtg")
    refs = references.sort_values("turbine")
    names = [*tests["test_wtg"], *refs["turbine"]]
    parts = [tests["estimate"].to_numpy(dtype=float)]
    if len(refs):
        parts.append(refs["uplift"].to_numpy(dtype=float))
    values = np.concatenate(parts) * PERCENT
    colours = ["C0"] * len(tests) + ["C3" if screened else "C1" for screened in refs["screened"]]

    fig, ax = plt.subplots(figsize=(max(9.0, 0.5 * len(names) + 3.0), 5.5))
    ax.bar(np.arange(len(names)), values, color=colours)
    if len(tests):
        ax.axvline(len(tests) - 0.5, color="k", linewidth=1, linestyle=":")
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize="small")
    ax.set_ylabel("uplift [%]")
    ax.set_title("per-turbine uplift: test turbines (blue), references (orange), screened references (red)")
    apply_grid(ax)
    path = out_dir / "per_turbine_uplift.png"
    save_fig(fig, path)
    return path


def plot_farm_uplift(
    out_dir: Path, *, per_turbine: pd.DataFrame, references: pd.DataFrame, rated_power_kw: float, farm_estimate: float
) -> Path:
    """Bar the farm uplift against the same aggregate over the references."""
    kept = references[~references["screened"]]
    reference_farm = _aggregate(kept, rated_power_kw=rated_power_kw)
    values = [farm_estimate * PERCENT, reference_farm * PERCENT]
    labels = [f"test turbines ({len(per_turbine)})", f"references ({len(kept)} kept)"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar([0, 1], values, color=["C0", "C1"], width=0.6)
    for x, value in zip([0, 1], values, strict=True):
        if np.isfinite(value):
            ax.annotate(
                f"{value:+.2f}%", (x, value), ha="center", va="bottom" if value >= 0 else "top", fontsize="large"
            )
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("energy-weighted uplift [%]")
    ax.set_title("farm uplift, and the same aggregate over the references\nthe reference bar is what zero looks like")
    apply_grid(ax)
    path = out_dir / "farm_uplift.png"
    save_fig(fig, path)
    return path


@dataclass(frozen=True)
class MethodUpliftInputs:
    """One method's slice of a campaign report, ready to plot.

    :param method: the method these came from, which names its plot folder
    :param per_turbine: its per-test-turbine estimates
    :param references: its per-reference readings, one row per reference
    :param farm_estimate: its farm headline
    """

    method: str
    per_turbine: pd.DataFrame
    references: pd.DataFrame
    farm_estimate: float


def per_method_inputs(
    per_turbine: pd.DataFrame, *, stability: pd.DataFrame, farm: pd.DataFrame
) -> list[MethodUpliftInputs]:
    """Split a campaign report into one plottable slice per method, in method order."""
    estimates = dict(zip(farm["method"], farm["estimate"].astype(float), strict=True)) if not farm.empty else {}
    inputs = []
    for method, rows in per_turbine.groupby("method", sort=True):
        mine = stability[stability["method"] == method] if not stability.empty else stability
        inputs.append(
            MethodUpliftInputs(
                method=str(method),
                per_turbine=rows,
                references=per_reference_uplifts(mine),
                farm_estimate=estimates.get(method, float("nan")),
            )
        )
    return inputs


def write_uplift_plots(
    out_dir: Path, *, per_turbine: pd.DataFrame, stability: pd.DataFrame, rated_power_kw: float, farm: pd.DataFrame
) -> list[Path]:
    """Write each method's three campaign-level uplift plots under ``out_dir``/its name.

    Returns what was written.
    """
    if per_turbine.empty:
        return []
    written = []
    for inputs in per_method_inputs(per_turbine, stability=stability, farm=farm):
        method_dir = out_dir / inputs.method
        method_dir.mkdir(parents=True, exist_ok=True)
        written += [
            plot_uplift_distributions(method_dir, per_turbine=inputs.per_turbine, references=inputs.references),
            plot_per_turbine_uplifts(method_dir, per_turbine=inputs.per_turbine, references=inputs.references),
            plot_farm_uplift(
                method_dir,
                per_turbine=inputs.per_turbine,
                references=inputs.references,
                rated_power_kw=rated_power_kw,
                farm_estimate=inputs.farm_estimate,
            ),
        ]
    return written


def _strip(
    ax: plt.Axes, position: float, values: np.ndarray, names: pd.Series, *, colour: str, marker: str = "o"
) -> None:
    """Scatter one group's readings beside its violin, each labelled with its turbine."""
    if not len(values):
        return
    jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(1)
    ax.scatter(position + jitter, values, color=colour, marker=marker, s=36, zorder=3)
    for x, y, name in zip(position + jitter, values, names, strict=True):
        ax.annotate(str(name), (x, y), textcoords="offset points", xytext=(5, 2), fontsize="x-small")


def _aggregate(references: pd.DataFrame, *, rated_power_kw: float) -> float:
    """Return the energy-weighted uplift over ``references``, computed as the farm result is."""
    if references.empty:
        return float("nan")
    rows = [
        TurbineUplift(
            turbine=str(row.turbine),
            uplift=float(row.uplift),
            actual_energy=float(row.actual_energy),
            n_records=int(row.n_records),
            rated_power_kw=rated_power_kw,
        )
        for row in references.itertuples()
    ]
    return float(farm_uplift(rows).uplift)
