"""Step 6 outputs: the engineered features over the whole record.

:func:`write_feature_outputs` writes, into one folder:

* ``era5/<feature>.png``: each ERA5 feature's distribution over the whole record and its monthly mean
  over time, labelled with its source and units, changeovers dashed;
* ``era5_features.csv``: each ERA5 feature's coverage and summary statistics;
* ``waking_scenarios/waking_scenario_<turbine>.png``: the share of each turbine's hours in each
  waking scenario.

:func:`category_shares_figure` draws the share of records in each category of a categorical, for any
named subsets of the records.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.baselines.power_model.features import ERA5_UNITS, QUALIFIER
from benchmarking.diagnostics.style import apply_grid, save_fig
from benchmarking.harness.waking import WAKING_SCENARIO_COL

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

MAX_CATEGORIES_SHOWN = 20
MISSING_CATEGORY = "(missing)"
_COLORS = ("C0", "C1", "C2", "C3")


def era5_label_with_units(feature: str, *, era5_label: str) -> str:
    """Return ``"<feature> @ <era5_label> [<units>]"``, without units where none are known."""
    unit = ERA5_UNITS.get(feature)
    return f"{feature}{QUALIFIER}{era5_label}" + (f" [{unit}]" if unit else "")


def category_shares_figure(values: pd.Series, *, segments: Mapping[str, np.ndarray], title: str) -> plt.Figure:
    """Horizontal bars of the share of each segment's records in each category, most common first.

    Categories beyond the :data:`MAX_CATEGORIES_SHOWN` most common are pooled, and missing values are
    a category of their own.

    :param values: one value per record
    :param segments: boolean masks over the records, by name; each gets its own bars
    :param title: the figure title
    """
    names = values.astype(object).where(values.notna(), MISSING_CATEGORY).astype(str).to_numpy()
    order = pd.Series(names).value_counts().index.tolist()
    if len(order) > MAX_CATEGORIES_SHOWN:
        rest = order[MAX_CATEGORIES_SHOWN - 1 :]
        other = f"{len(rest)} other categories"
        names = np.where(np.isin(names, rest), other, names)
        order = [*order[: MAX_CATEGORIES_SHOWN - 1], other]
    y = np.arange(len(order))
    longest = max((len(o) for o in order), default=10)
    fig, ax = plt.subplots(figsize=(7 + 0.07 * longest, 1.5 + 0.45 * len(order)))
    height = 0.8 / max(len(segments), 1)
    for k, (label, mask) in enumerate(segments.items()):
        shares = pd.Series(names[mask]).value_counts(normalize=True).reindex(order, fill_value=0.0)
        offset = (k - (len(segments) - 1) / 2) * height
        ax.barh(y + offset, shares.to_numpy(), height=height, color=_COLORS[k % len(_COLORS)], alpha=0.8, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize="small")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("share of records (log scale)")
    ax.set_title(title)
    apply_grid(ax)
    ax.legend()
    return fig


def era5_feature_figure(values: pd.Series, *, label: str, changeovers: Sequence[pd.Timestamp] = ()) -> plt.Figure:
    """Distribution over the whole record beside the monthly mean over time; changeovers dashed."""
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    fig, (hist_ax, time_ax) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 2]})
    if len(finite):
        lo, hi = np.percentile(finite, [0.5, 99.5])
        hist_ax.hist(finite.clip(lo, hi), bins=50, density=True, color="C0", alpha=0.8)
        monthly = finite.resample("MS").mean()
        time_ax.plot(monthly.index, monthly.to_numpy(), color="C0", marker="o", markersize=3)
    for changeover in sorted(set(changeovers)):
        time_ax.axvline(changeover, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    hist_ax.set_xlabel(label)
    hist_ax.set_ylabel("density")
    time_ax.set_ylabel(f"monthly mean of {label}")
    for ax in (hist_ax, time_ax):
        apply_grid(ax)
    dashed = "; dashed: changeovers" if changeovers else ""
    fig.suptitle(f"{label}: whole record ({len(finite)} records, 0.5-99.5th percentile shown{dashed})")
    return fig


def write_feature_outputs(
    scada_df: pd.DataFrame,
    *,
    era5_features: pd.DataFrame,
    era5_label: str,
    turbine_col: str,
    changeovers: Mapping[str, Sequence[pd.Timestamp]] | None = None,
    out_dir: Path,
) -> list[Path]:
    """Write the step 6 plots and table over the whole record; returns the files written.

    :param scada_df: long SCADA with the step 5 waking scenario column; turbines without it are skipped
    :param era5_features: the ERA5 features on the SCADA timestamps
    :param era5_label: where the ERA5 came from, as named in labels
    :param turbine_col: the turbine-identifier column
    :param changeovers: each turbine's changeover dates, all drawn on the ERA5 plots
    :param out_dir: the folder written into
    """
    dates = [d for days in (changeovers or {}).values() for d in days]
    written: list[Path] = []
    era5_dir = out_dir / "era5"
    era5_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for feature in era5_features.columns:
        values = era5_features[feature].astype(float)
        label = era5_label_with_units(feature, era5_label=era5_label)
        path = era5_dir / f"{feature}.png"
        save_fig(era5_feature_figure(values, label=label, changeovers=dates), path)
        written.append(path)
        rows.append(
            {
                "feature": f"{feature}{QUALIFIER}{era5_label}",
                "unit": ERA5_UNITS.get(feature, ""),
                "coverage_pct": 100.0 * float(values.notna().mean()),
                **values.describe().drop("count").to_dict(),
            }
        )
    table = out_dir / "era5_features.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    written.append(table)
    if WAKING_SCENARIO_COL not in scada_df.columns:
        logger.warning("no waking scenario plots: scada_df has no %r column", WAKING_SCENARIO_COL)
        return written
    scenario_dir = out_dir / "waking_scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for turbine, rows_of in scada_df.groupby(turbine_col, sort=True):
        values = rows_of[WAKING_SCENARIO_COL]
        fig = category_shares_figure(
            values,
            segments={"whole record": np.ones(len(values), dtype=bool)},
            title=f"{turbine}: waking scenario ({len(values)} records, whole record)",
        )
        path = scenario_dir / f"waking_scenario_{turbine}.png"
        save_fig(fig, path)
        written.append(path)
    logger.info("Wrote %d step 6 feature files to %s", len(written), out_dir)
    return written
