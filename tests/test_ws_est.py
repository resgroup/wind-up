from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from wind_up.models import WindUpConfig
from wind_up.scada_power_curve import calc_pc_and_rated_ws
from wind_up.ws_est import add_ws_est


def test_ws_est(test_homer_config: WindUpConfig) -> None:
    cfg = test_homer_config
    wf_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/Homer Wind Farm_July2023_scada_improved.parquet")

    # usually x_bin_width would be cfg.ws_bin_width / 2 but that gives invalid power curve with so little data
    pc_per_ttype, rated_ws_per_ttype = calc_pc_and_rated_ws(
        cfg=cfg,
        wf_df=wf_df,
        x_col="WindSpeedMean",
        y_col="pw_clipped",
        x_bin_width=cfg.ws_bin_width * 2.01,
        plot_cfg=None,
    )
    adf = add_ws_est(cfg=cfg, wf_df=wf_df, pc_per_ttype=pc_per_ttype, plot_cfg=None)
    edf = pd.read_parquet(Path(__file__).parents[0] / "test_data/test_add_ws_est.parquet")
    assert_frame_equal(adf, edf)
