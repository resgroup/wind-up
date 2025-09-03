"""Data loading and configuration for Hill of Towie (HoT) wind farm analysis.

Note: This module requires Python 3.10 or above.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from tests.test_data.hot.unpack import unpack_local_meta_data, unpack_local_scada_data
from wind_up.interface import PrePostSplitter
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.reanalysis_data import ReanalysisDataset

logger = logging.getLogger(__name__)

ANALYSIS_DIR = Path(__file__).parent / Path(__file__).stem
ANALYSIS_CACHE_DIR = ANALYSIS_DIR / "cache"
ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = Path(__file__).parent


class HoTData(NamedTuple):
    metadata_df: pd.DataFrame
    scada_df: pd.DataFrame


def get_meta_and_scada_data() -> HoTData:
    data_dir = Path(__file__).parent / "scada"
    logger.warning("Loading Hill of Towie open data from %s", data_dir)
    return HoTData(
        metadata_df=unpack_local_meta_data(data_dir=data_dir),
        scada_df=unpack_local_scada_data(
            data_dir=data_dir,
            start_dt=pd.Timestamp(year=2019, month=1, day=1, tz="UTC"),
            end_dt_excl=pd.Timestamp(year=2022, month=9, day=13, tz="UTC"),
        ),
    )


class WindUpComponents(NamedTuple):
    scada_df: pd.DataFrame
    metadata_df: pd.DataFrame
    wind_up_config: WindUpConfig
    plot_cfg: PlotConfig
    reanalysis_datasets: list[ReanalysisDataset]
    pre_post_splitter: PrePostSplitter


def construct_hot_windup_components(scada_df: pd.DataFrame, metadata_df: pd.DataFrame) -> WindUpComponents:
    logger.info("Loading reference reanalysis data")
    reanalysis_file_path = Path(__file__).parent / "reanalysis_data/ERA5T_57.50N_-3.25E_100m_1hr_20241231.parquet"
    reanalysis_dataset = ReanalysisDataset(
        id="ERA5T_57.50N_-3.25E_100m_1hr",
        data=pd.read_parquet(reanalysis_file_path),
    )

    logger.info("Defining Assessment Configuration")
    cfg = WindUpConfig.from_yaml(CONFIG_DIR / "HoT_AeroUp_T13.yaml")
    cfg.out_dir = ANALYSIS_DIR / cfg.assessment_name
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir = ANALYSIS_DIR / cfg.assessment_name
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    plot_cfg = PlotConfig(
        show_plots=False, save_plots=False, plots_dir=cfg.out_dir / "plots", skip_per_turbine_plots=True
    )

    (ANALYSIS_CACHE_DIR / cfg.assessment_name).mkdir(parents=True, exist_ok=True)

    splitter = PrePostSplitter(cfg=cfg)

    # return components rather than AssessmentInputs to avoid running the slow preprocessing (speeds up tests)
    return WindUpComponents(
        wind_up_config=cfg,
        plot_cfg=plot_cfg,
        scada_df=scada_df,
        metadata_df=metadata_df,
        reanalysis_datasets=[reanalysis_dataset],
        pre_post_splitter=splitter,
    )
