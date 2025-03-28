from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from examples.smarteole_example import (
    SmarteoleData,
    _construct_assessment_inputs,
    unpack_smarteole_metadata,
    unpack_smarteole_scada,
    unpack_smarteole_toggle_data,
)
from wind_up.models import WindUpConfig

if TYPE_CHECKING:
    from wind_up.interface import AssessmentInputs

TEST_DATA_FLD = Path(__file__).parent / "test_data"
TEST_CONFIG_DIR = TEST_DATA_FLD / "config"

SMARTEOLE_DATA_DIR = TEST_DATA_FLD / "smarteole"
SMARTEOLE_CACHE_DIR = Path(__file__).parent / "timebase_600"
SMARTEOLE_OUTPUT_DIR = Path(__file__).parent / "output"

logger = logging.getLogger(__name__)


@pytest.fixture
def test_lsa_t13_config() -> WindUpConfig:
    return WindUpConfig.from_yaml(TEST_CONFIG_DIR / "test_LSA_T13.yaml")


@pytest.fixture
def test_brt_t16_pitch() -> WindUpConfig:
    return WindUpConfig.from_yaml(TEST_CONFIG_DIR / "test_BRT_T16_pitch.yaml")


@pytest.fixture
def test_marge_config() -> WindUpConfig:
    return WindUpConfig.from_yaml(TEST_CONFIG_DIR / "test_Marge.yaml")


@pytest.fixture
def test_homer_config() -> WindUpConfig:
    cfg = WindUpConfig.from_yaml(TEST_CONFIG_DIR / "test_Homer.yaml")
    t1lat = -58.60364145072843
    t1long = 103.6841410289202052
    t2lat = -58.60261454305449
    t2long = 103.688364968451364
    cfg.asset.wtgs[0].latitude = t1lat
    cfg.asset.wtgs[0].longitude = t1long
    cfg.asset.wtgs[1].latitude = t2lat
    cfg.asset.wtgs[1].longitude = t2long
    return cfg


@pytest.fixture
def test_homer_with_t00_config() -> WindUpConfig:
    cfg = WindUpConfig.from_yaml(TEST_CONFIG_DIR / "test_Homer.yaml")
    t1lat = -58.60364145072843
    t1long = 103.6841410289202052
    t2lat = -58.60261454305449
    t2long = 103.688364968451364
    cfg.asset.wtgs[0].latitude = t1lat
    cfg.asset.wtgs[0].longitude = t1long
    cfg.asset.wtgs[1].latitude = t2lat
    cfg.asset.wtgs[1].longitude = t2long
    cfg.asset.wtgs.append(cfg.asset.wtgs[0].model_copy())
    cfg.asset.wtgs[-1].name = "HMR_T00"
    cfg.asset.wtgs[-1].latitude = -58.601587635380
    cfg.asset.wtgs[-1].longitude = 103.692588907983
    return cfg


@pytest.fixture(scope="session")
def smarteole_assessment_inputs() -> tuple[AssessmentInputs, SmarteoleData]:
    timebase_s = 600
    output_dir = SMARTEOLE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_subdir = SMARTEOLE_CACHE_DIR
    cache_subdir.mkdir(parents=True, exist_ok=True)

    scada_df = unpack_smarteole_scada(
        timebase_s=timebase_s, scada_data_file=SMARTEOLE_DATA_DIR / "SMARTEOLE_WakeSteering_SCADA_1minData.csv"
    )
    metadata_df = unpack_smarteole_metadata(
        timebase_s=timebase_s, metadata_file=SMARTEOLE_DATA_DIR / "SMARTEOLE_WakeSteering_Coordinates_staticData.csv"
    )
    toggle_df = unpack_smarteole_toggle_data(
        timebase_s=timebase_s, toggle_file=SMARTEOLE_DATA_DIR / "SMARTEOLE_WakeSteering_ControlLog_1minData.csv"
    )
    smarteole_data = SmarteoleData(scada_df=scada_df, metadata_df=metadata_df, toggle_df=toggle_df)

    return _construct_assessment_inputs(
        smarteole_data=smarteole_data,
        logger=logger,
        reanalysis_file_path=SMARTEOLE_DATA_DIR / "ERA5T_50.00N_2.75E_100m_1hr_20200201_20200531.parquet",
        analysis_timebase_s=timebase_s,
        analysis_output_dir=output_dir,
        cache_sub_dir=cache_subdir,
    ), smarteole_data


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ARG001,ANN001
    # remove directories created during tests
    plot_testing = [
        Path(__file__).parents[1] / "result_images",  # if pytest run from root dir
    ]
    for d in [SMARTEOLE_OUTPUT_DIR, SMARTEOLE_CACHE_DIR, *plot_testing]:
        if d.is_dir():
            shutil.rmtree(d)
