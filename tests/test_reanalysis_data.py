from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import TEST_DATA_FLD
from wind_up.models import WindUpConfig
from wind_up.reanalysis_data import (
    ReanalysisDataset,
    add_reanalysis_data,
    get_dsid_and_dates_from_filename,
)


def test_get_dsid_and_dates_from_filename() -> None:
    assert get_dsid_and_dates_from_filename("ERA5T_47.50N_-3.25E_100m_1hr_19900101_20231031.parquet") == (
        "ERA5T_47.50N_-3.25E_100m_1hr",
        pd.Timestamp("1990-01-01", tz="UTC"),
        pd.Timestamp("2023-10-31", tz="UTC"),
    )


def test_add_reanalysis_data(test_homer_config: WindUpConfig) -> None:
    cfg = test_homer_config
    cfg.lt_first_dt_utc_start = pd.Timestamp("2023-07-01 00:00:00", tz="UTC")
    cfg.analysis_last_dt_utc_start = pd.Timestamp("2023-07-31 23:50:00", tz="UTC")
    wf_df = pd.read_parquet(Path(__file__).parents[0] / "test_data/Homer Wind Farm_July2023_scada_improved.parquet")
    reanalysis_datasets = [
        ReanalysisDataset(id=fp.stem, data=pd.read_parquet(fp))
        for fp in (TEST_DATA_FLD / "reanalysis" / "Homer Wind Farm").glob("*.parquet")
    ]
    wf_df_after = add_reanalysis_data(wf_df, cfg=cfg, plot_cfg=None, reanalysis_datasets=reanalysis_datasets)
    assert len(wf_df_after.dropna(subset=["reanalysis_ws", "reanalysis_wd"])) == 8928
    assert wf_df_after["reanalysis_ws"].mean() == pytest.approx(7.095553315412186)
    assert wf_df_after["reanalysis_ws"].min() == pytest.approx(0.19)
    assert wf_df_after["reanalysis_ws"].max() == pytest.approx(16.35)
    assert wf_df_after["reanalysis_wd"].mean() == pytest.approx(217.18956093189962)
    assert wf_df_after["reanalysis_wd"].min() == pytest.approx(4.4)
    assert wf_df_after["reanalysis_wd"].max() == pytest.approx(359.6)
