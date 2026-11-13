from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from wind_up_v0.constants import RAW_POWER_COL, REANALYSIS_WD_COL, WINDFARM_YAWDIR_COL
from wind_up_v0.models import PlotConfig, WindUpConfig
from wind_up_v0.plots import optimize_northing_plots

if TYPE_CHECKING:
    from pathlib import Path


def _diff_df(north_ref_wd_col: str) -> pd.DataFrame:
    ts = pd.date_range("2020-01-01", periods=20, freq="10min")
    return pd.DataFrame(
        {
            f"yaw_diff_to_{north_ref_wd_col}": 1.0,
            f"filt_diff_to_{north_ref_wd_col}": 1.0,
            f"short_rolling_diff_to_{north_ref_wd_col}": 1.0,
            f"long_rolling_diff_to_{north_ref_wd_col}": 1.0,
            RAW_POWER_COL: 500.0,
        },
        index=ts,
    )


class TestPlotDiffToNorthRefWd:
    @staticmethod
    @pytest.mark.parametrize("save_plots", [False, True])
    def test_honors_save_plots(tmp_path: Path, save_plots: bool) -> None:  # noqa: FBT001
        plots_dir = tmp_path / "plots"
        wtg_name = "T01"
        optimize_northing_plots.plot_diff_to_north_ref_wd(
            _diff_df(REANALYSIS_WD_COL),
            wtg_name=wtg_name,
            north_ref_wd_col=REANALYSIS_WD_COL,
            loop_count=0,
            plot_cfg=PlotConfig(save_plots=save_plots, show_plots=False, plots_dir=plots_dir),
        )
        any_saved = plots_dir.exists() and any(plots_dir.rglob("*.png"))
        assert any_saved is save_plots


class TestPlotYawDiffVsPower:
    @staticmethod
    @pytest.mark.parametrize("save_plots", [False, True])
    def test_honors_save_plots(tmp_path: Path, save_plots: bool) -> None:  # noqa: FBT001
        plots_dir = tmp_path / "plots"
        optimize_northing_plots.plot_yaw_diff_vs_power(
            _diff_df(REANALYSIS_WD_COL),
            wtg_name="T01",
            north_ref_wd_col=REANALYSIS_WD_COL,
            plot_cfg=PlotConfig(save_plots=save_plots, show_plots=False, plots_dir=plots_dir),
        )
        any_saved = plots_dir.exists() and any(plots_dir.rglob("*.png"))
        assert any_saved is save_plots


class TestPlotWfYawdirAndReanalysisTimeseries:
    @staticmethod
    @pytest.mark.parametrize("save_plots", [False, True])
    def test_honors_save_plots(test_homer_config: WindUpConfig, tmp_path: Path, save_plots: bool) -> None:  # noqa: FBT001
        plots_dir = tmp_path / "plots"
        wtg_name = test_homer_config.asset.wtgs[0].name
        ts = pd.date_range("2020-01-01", periods=20, freq="10min")
        wf_df = pd.DataFrame(
            {WINDFARM_YAWDIR_COL: 100.0, REANALYSIS_WD_COL: 100.0},
            index=pd.MultiIndex.from_product([[wtg_name], ts], names=["TurbineName", "TimeStamp_StartFormat"]),
        )
        optimize_northing_plots.plot_wf_yawdir_and_reanalysis_timeseries(
            wf_df,
            cfg=test_homer_config,
            plot_cfg=PlotConfig(save_plots=save_plots, show_plots=False, plots_dir=plots_dir),
        )
        any_saved = plots_dir.exists() and any(plots_dir.rglob("*.png"))
        assert any_saved is save_plots
