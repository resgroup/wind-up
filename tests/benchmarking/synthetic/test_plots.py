"""Tests for the synthetic-dataset verification plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless: no display needed for tests

import numpy as np
import pandas as pd

from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.generator import generate_dataset
from benchmarking.synthetic.plots import plot_power_curve_comparison
from benchmarking.synthetic.upgrades import ConstantCpChange
from wind_up_v0.constants import TIMESTAMP_COL

if TYPE_CHECKING:
    from pathlib import Path


def _swept_dataset() -> object:
    periods = 288
    index = pd.date_range("2020-01-01", periods=periods, freq="10min", tz="UTC")
    ws = np.linspace(3.0, 14.0, periods)
    power = np.clip(2300.0 * ((ws - 3.0) / 11.0) ** 3, 0.0, 2300.0)
    frames = [
        pd.DataFrame(
            {
                HOT_COLUMNS.turbine: turbine,
                HOT_COLUMNS.active_power: power,
                HOT_COLUMNS.wind_speed: ws,
                HOT_COLUMNS.wind_speed_sd: 0.1 * ws,
                HOT_COLUMNS.gen_rpm: 1400.0,
            },
            index=index,
        )
        for turbine in ("T01", "T02")
    ]
    wf_df = pd.concat(frames)
    wf_df.index.name = TIMESTAMP_COL
    return generate_dataset(
        scada_df=wf_df,
        test_wtgs=["T01"],
        upgrades=[ConstantCpChange(delta=0.05)],
        mode="prepost",
        upgrade_timing=index[periods // 2],
    )


def _paired_dfs(orig_power: list[float], syn_power: list[float], ws: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(orig_power)
    index = pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC")

    def frame(power: list[float]) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                HOT_COLUMNS.turbine: "T01",
                HOT_COLUMNS.active_power: np.array(power, dtype=float),
                HOT_COLUMNS.wind_speed: np.array(ws, dtype=float),
            },
            index=index,
        )
        df.index.name = TIMESTAMP_COL
        return df

    return frame(syn_power), frame(orig_power)


def test_power_curve_comparison_has_three_panels_with_aligned_axes() -> None:
    """Three panels; the two power-curve panels share x and y, all three share x."""
    dataset = _swept_dataset()
    fig = plot_power_curve_comparison(dataset.synthetic_df, dataset.original_df, test_wtg="T01")

    assert len(fig.axes) == 3
    ax_orig, ax_syn, ax_delta = fig.axes
    assert ax_orig.get_xlim() == ax_syn.get_xlim() == ax_delta.get_xlim()
    assert ax_orig.get_ylim() == ax_syn.get_ylim()


def test_third_panel_shows_kw_change() -> None:
    """The third panel plots the synthetic-minus-original power change."""
    dataset = _swept_dataset()
    fig = plot_power_curve_comparison(dataset.synthetic_df, dataset.original_df, test_wtg="T01")
    ax_delta = fig.axes[2]
    assert "change" in ax_delta.get_ylabel().lower()
    assert any(coll.get_offsets().size > 0 for coll in ax_delta.collections)


def test_power_curve_comparison_enables_grid() -> None:
    """All panels show gridlines (per the verification request)."""
    dataset = _swept_dataset()
    fig = plot_power_curve_comparison(dataset.synthetic_df, dataset.original_df, test_wtg="T01")
    for ax in fig.axes:
        assert any(line.get_visible() for line in ax.get_xgridlines())
        assert any(line.get_visible() for line in ax.get_ygridlines())


def test_kw_change_panel_excludes_nan_downtime_rows() -> None:
    """NaN-power (downtime) rows are not treated as changed in the kW-change panel."""
    # row 0 changed; row 1 both-NaN (downtime); row 2 unchanged; row 3 changed
    synthetic, original = _paired_dfs(
        orig_power=[1000.0, np.nan, 1000.0, 1000.0],
        syn_power=[1100.0, np.nan, 1000.0, 1100.0],
        ws=[8.0, 9.0, 10.0, 11.0],
    )
    fig = plot_power_curve_comparison(synthetic, original, test_wtg="T01")
    ax_delta = fig.axes[2]
    plotted = np.concatenate([coll.get_offsets().data for coll in ax_delta.collections if coll.get_offsets().size])
    # exactly the two genuinely changed finite rows (ws 8 and 11), not the NaN row
    assert sorted(plotted[:, 0].tolist()) == [8.0, 11.0]


def test_power_curve_comparison_saves_file(tmp_path: Path) -> None:
    """A PNG is written when ``save_path`` is given."""
    dataset = _swept_dataset()
    save_path = tmp_path / "power_curve_comparison.png"
    plot_power_curve_comparison(dataset.synthetic_df, dataset.original_df, test_wtg="T01", save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
