"""Smoke tests for the northing plots: they must draw, save, and survive thin input."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_up.layout import Layout
from wind_up.northing import estimate_north_table
from wind_up.northing_plots import plot_northing, plot_northing_farm, plot_wake_nadir_farm, plot_wake_nadir_pair
from wind_up.wake_nadir import WakePairCurves

if TYPE_CHECKING:
    from pathlib import Path


def _index(days: float = 400.0) -> pd.DatetimeIndex:
    timebase_s = 600
    periods = round(days * 24 * 3600 / timebase_s)
    return pd.date_range(start="2017-01-01", periods=periods, freq=f"{timebase_s}s", tz="UTC")


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


def _bubble_layout(names: tuple[str, ...]) -> Layout:
    frame = pd.DataFrame(
        {
            "name": list(names),
            "latitude": [55.0 + 0.004 * i for i in range(len(names))],
            "longitude": [0.0 + 0.006 * i for i in range(len(names))],
            "rotor_diameter_m": [82.0] * len(names),
        }
    )
    return Layout.from_frame(frame)


class TestPlotWakeNadirFarm:
    def test_draws_a_bubble_per_turbine_and_saves(self, tmp_path: Path) -> None:
        names = ("T01", "T02", "T03", "T04")
        layout = _bubble_layout(names)
        corrections = {"T01": 4.2, "T02": -3.1, "T03": 0.0, "T04": 8.5}

        figure = plot_wake_nadir_farm(layout, corrections=corrections, out_dir=tmp_path)
        offsets = figure.axes[0].collections[0].get_offsets()
        plt.close(figure)

        assert (tmp_path / "wake_nadir_bubble.png").is_file()
        assert len(offsets) == len(names)

    def test_a_turbine_missing_a_correction_is_drawn_at_zero(self, tmp_path: Path) -> None:
        names = ("T01", "T02", "T03")
        layout = _bubble_layout(names)

        figure = plot_wake_nadir_farm(layout, corrections={"T01": 5.0}, out_dir=tmp_path)
        offsets = figure.axes[0].collections[0].get_offsets()
        plt.close(figure)

        assert len(offsets) == len(names)


def _pair_curves(*, nadir_deg: float | None, wind_speed: bool) -> WakePairCurves:
    offset = np.arange(30, dtype=float) - 14.5
    ratio = 1.0 - 0.3 * np.exp(-0.5 * ((offset - (nadir_deg or 0.0)) / 3.0) ** 2)
    return WakePairCurves(
        upstream="T01",
        downstream="T02",
        offset_deg=offset,
        power_ratio=ratio,
        wind_speed_ratio=ratio if wind_speed else None,
        nadir_deg=nadir_deg,
    )


class TestPlotWakeNadirPair:
    def test_draws_power_and_wind_speed_and_saves(self, tmp_path: Path) -> None:
        before = _pair_curves(nadir_deg=5.0, wind_speed=True)
        after = _pair_curves(nadir_deg=0.2, wind_speed=True)

        figure = plot_wake_nadir_pair(before, after=after, out_dir=tmp_path)
        panels = len(figure.axes)
        plt.close(figure)

        assert (tmp_path / "wake_nadir_pair_T01_T02.png").is_file()
        assert panels == 2

    def test_power_only_and_an_unresolved_nadir_still_draw(self) -> None:
        before = _pair_curves(nadir_deg=None, wind_speed=False)
        after = _pair_curves(nadir_deg=None, wind_speed=False)

        figure = plot_wake_nadir_pair(before, after=after)
        panels = len(figure.axes)
        plt.close(figure)

        assert panels == 1
