from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from wind_up.constants import (
    RAW_DOWNTIME_S_COL,
    RAW_POWER_COL,
    RAW_YAWDIR_COL,
    REANALYSIS_WD_COL,
)
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.northing import calc_northed_col_name
from wind_up.plots import northing_plots

if TYPE_CHECKING:
    from pathlib import Path


def _make_wf_df(wtg_name: str, changepoint: pd.Timestamp, northed_col: str) -> pd.DataFrame:
    ts = pd.date_range(changepoint - pd.Timedelta(days=1), changepoint + pd.Timedelta(days=1), freq="10min")
    return pd.DataFrame(
        {
            RAW_YAWDIR_COL: 100.0,
            northed_col: 100.0,
            REANALYSIS_WD_COL: 100.0,
            RAW_POWER_COL: 500.0,
            RAW_DOWNTIME_S_COL: 0.0,
        },
        index=pd.MultiIndex.from_product([[wtg_name], ts], names=["TurbineName", "TimeStamp_StartFormat"]),
    )


class TestPlotNorthingChangepoint:
    @staticmethod
    def test_does_not_save_when_save_plots_false(test_homer_config: WindUpConfig, tmp_path: Path) -> None:
        wtg_name = test_homer_config.asset.wtgs[0].name
        changepoint = pd.Timestamp("2018-07-01 00:00:00", tz="UTC")
        northed_col = calc_northed_col_name(REANALYSIS_WD_COL)
        wf_df = _make_wf_df(wtg_name, changepoint, northed_col)
        plots_dir = tmp_path / "plots"

        northing_plots.plot_northing_changepoint(
            wf_df,
            northing_turbine=wtg_name,
            northed_col=northed_col,
            north_ref_wd_col=REANALYSIS_WD_COL,
            northing_datetime_utc=changepoint,
            cfg=test_homer_config,
            plot_cfg=PlotConfig(save_plots=False, show_plots=False, plots_dir=plots_dir),
        )

        assert not plots_dir.exists()

    @staticmethod
    def test_saves_when_save_plots_true(test_homer_config: WindUpConfig, tmp_path: Path) -> None:
        wtg_name = test_homer_config.asset.wtgs[0].name
        changepoint = pd.Timestamp("2018-07-01 00:00:00", tz="UTC")
        northed_col = calc_northed_col_name(REANALYSIS_WD_COL)
        wf_df = _make_wf_df(wtg_name, changepoint, northed_col)
        plots_dir = tmp_path / "plots"

        northing_plots.plot_northing_changepoint(
            wf_df,
            northing_turbine=wtg_name,
            northed_col=northed_col,
            north_ref_wd_col=REANALYSIS_WD_COL,
            northing_datetime_utc=changepoint,
            cfg=test_homer_config,
            plot_cfg=PlotConfig(save_plots=True, show_plots=False, plots_dir=plots_dir),
        )

        expected = (
            plots_dir
            / wtg_name
            / f"{wtg_name} north_ref_wd_col={REANALYSIS_WD_COL} {changepoint.strftime('%Y-%m-%d')}.png"
        )
        assert expected.is_file()
