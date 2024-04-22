import pickle
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wind_up.models import WindUpConfig
from wind_up.scada_power_curve import calc_pc_and_rated_ws


def test_calc_pc_and_rated_ws(test_homer_config: WindUpConfig) -> None:
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
    with Path.open(Path(__file__).parents[0] / "test_data/pc_per_ttype.pickle", "rb") as handle:
        e_pc_per_ttype = pickle.load(handle)
    with Path.open(Path(__file__).parents[0] / "test_data/rated_ws_per_ttype.pickle", "rb") as handle:
        e_rated_ws_per_ttype = pickle.load(handle)

    for ttype in pc_per_ttype:
        assert_frame_equal(pc_per_ttype[ttype], e_pc_per_ttype[ttype])
        assert rated_ws_per_ttype[ttype] == pytest.approx(e_rated_ws_per_ttype[ttype])
