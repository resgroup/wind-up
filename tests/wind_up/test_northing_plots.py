"""Smoke tests for the northing plots: they must draw, save, and survive thin input."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_up.northing import estimate_north_table
from wind_up.northing_plots import plot_northing, plot_northing_farm

if TYPE_CHECKING:
    from pathlib import Path

TIMEBASE_S = 600


def _index(days: float = 400.0) -> pd.DatetimeIndex:
    periods = round(days * 24 * 3600 / TIMEBASE_S)
    return pd.date_range(start="2017-01-01", periods=periods, freq=f"{TIMEBASE_S}s", tz="UTC")


def _device(index: pd.DatetimeIndex, *, seed: int, step_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(reported, reference)`` for a device that steps by ``step_deg`` halfway through."""
    rng = np.random.default_rng(seed)
    reference = np.cumsum(rng.normal(0.0, 2.0, size=len(index))) % 360.0
    offset = np.where(index >= index.min() + (index.max() - index.min()) / 2, step_deg, 0.0)
    reported = (reference + rng.normal(0.0, 6.0, size=len(index)) - offset) % 360.0
    return reported, reference


class TestPlotNorthing:
    def test_draws_and_saves_a_single_device(self, tmp_path: Path) -> None:
        index = _index()
        reported, reference = _device(index, seed=0, step_deg=30.0)
        usable = np.ones(len(index), dtype=bool)
        table = estimate_north_table(index, reported, reference_deg=reference, usable=usable)

        figure = plot_northing(
            index,
            reported,
            reference_deg=reference,
            usable=usable,
            north_table=table,
            device="T01",
            out_dir=tmp_path,
        )
        plt.close(figure)

        assert (tmp_path / "T01_northing.png").is_file()

    def test_survives_a_device_with_almost_no_usable_rows(self, tmp_path: Path) -> None:
        """A near-empty residual must not raise; the panels are simply blank."""
        index = _index(days=40)
        reported, reference = _device(index, seed=1, step_deg=0.0)
        usable = np.zeros(len(index), dtype=bool)
        usable[:5] = True
        table = estimate_north_table(index, reported, reference_deg=reference, usable=usable)

        figure = plot_northing(
            index, reported, reference_deg=reference, usable=usable, north_table=table, device="T02", out_dir=tmp_path
        )
        plt.close(figure)

        assert (tmp_path / "T02_northing.png").is_file()


class TestPlotNorthingFarm:
    def test_draws_one_panel_per_device_and_saves(self, tmp_path: Path) -> None:
        index = _index()
        names = ("T01", "T02", "T03", "T04")
        reported, reference = {}, None
        for i, name in enumerate(names):
            reported[name], reference = _device(index, seed=i, step_deg=10.0 * i)
        usable = {name: np.ones(len(index), dtype=bool) for name in names}
        tables = {
            name: estimate_north_table(index, reported[name], reference_deg=reference, usable=usable[name])
            for name in names
        }

        figure = plot_northing_farm(
            index,
            direction_deg=reported,
            reference_deg=reference,
            usable=usable,
            north_tables=tables,
            out_dir=tmp_path,
        )
        visible = [ax for ax in figure.axes if ax.get_visible()]
        plt.close(figure)

        assert (tmp_path / "farm_northing.png").is_file()
        assert len(visible) == len(names)
