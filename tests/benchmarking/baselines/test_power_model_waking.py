"""Tests for the waking diagnostics: what the turbines reduced to one column actually carry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.baselines.power_model.waking import (
    plot_waking_layout,
    waking_fractions,
    write_waking_diagnostics,
)

if TYPE_CHECKING:
    from pathlib import Path

_TURBINE = "TurbineName"
_POWER = "power"
_THRESHOLD_KW = 100.0


def _scada(index: pd.DatetimeIndex, powers: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.concat([pd.DataFrame({_TURBINE: name, _POWER: values}, index=index) for name, values in powers.items()])


def _case() -> tuple[pd.DataFrame, pd.Series]:
    """T1 wakes throughout; T2 stops waking after the changeover; T3 has no data at all."""
    index = pd.date_range("2019-01-01", periods=100, freq="10min", tz="UTC")
    treated = pd.Series(np.arange(len(index)) >= 50, index=index)
    always = np.full(len(index), 500.0)
    stops = np.where(treated.to_numpy(), 10.0, 500.0)
    return _scada(index, {"T1": always, "T2": stops, "T3": np.full(len(index), np.nan)}), treated


class TestWakingFractions:
    def _frame(self) -> pd.DataFrame:
        scada, treated = _case()
        return waking_fractions(
            scada, turbine_col=_TURBINE, active_power_col=_POWER, threshold_kw=_THRESHOLD_KW, treated=treated
        ).set_index("turbine")

    def test_a_turbine_waking_throughout_reads_one_in_both_periods(self) -> None:
        row = self._frame().loc["T1"]
        assert row["baseline"] == 1.0
        assert row["treated"] == 1.0
        assert row["shift"] == 0.0

    def test_a_turbine_that_stops_waking_shows_the_whole_shift(self) -> None:
        row = self._frame().loc["T2"]
        assert row["baseline"] == 1.0
        assert row["treated"] == 0.0
        assert row["shift"] == -1.0

    def test_a_turbine_with_no_finite_power_is_nan_rather_than_zero(self) -> None:
        # never waking and never measured are different claims
        row = self._frame().loc["T3"]
        assert np.isnan(row["baseline"])
        assert np.isnan(row["treated"])


class TestWhatIsWritten:
    def _write(self, tmp_path: Path, *, coords: dict[str, tuple[float, float]] | None) -> list[str]:
        scada, treated = _case()
        written = write_waking_diagnostics(
            tmp_path,
            scada=scada,
            turbine_col=_TURBINE,
            active_power_col=_POWER,
            threshold_kw=_THRESHOLD_KW,
            treated=treated,
            test_wtg="T1",
            references=["T2"],
            power_free=["T3"],
            coords=coords,
        )
        return [p.name for p in written]

    def test_both_plots_land_in_the_feature_engineering_stage(self, tmp_path: Path) -> None:
        coords = {"T1": (57.5, -3.25), "T2": (57.51, -3.25), "T3": (57.52, -3.25)}
        names = self._write(tmp_path, coords=coords)
        assert names == ["waking_fractions.png", "waking_layout.png"]
        stage = tmp_path / "plots" / "3_feature_eng"
        assert (stage / "waking_fractions.png").exists()
        assert (stage / "waking_layout.png").exists()

    def test_the_fractions_are_written_as_a_table_too(self, tmp_path: Path) -> None:
        self._write(tmp_path, coords=None)
        table = pd.read_csv(tmp_path / "plots" / "3_feature_eng" / "waking_fractions.csv")
        assert set(table.columns) == {"turbine", "baseline", "treated", "shift"}

    def test_the_layout_is_skipped_when_the_campaign_has_no_coordinates(self, tmp_path: Path) -> None:
        # a method run outside a campaign knows the turbines but not where they are
        assert self._write(tmp_path, coords=None) == ["waking_fractions.png"]

    def test_the_layout_is_skipped_when_no_named_turbine_has_coordinates(self, tmp_path: Path) -> None:
        drawn = plot_waking_layout(
            tmp_path, coords={"other": (57.5, -3.25)}, test_wtg="T1", references=["T2"], power_free=["T3"]
        )
        assert drawn is None
