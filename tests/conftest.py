from pathlib import Path

import pytest

from wind_up.models import WindUpConfig

TEST_DATA_FLD = Path(__file__).parent / "test_data"
TEST_CONFIG_DIR = TEST_DATA_FLD / "config"


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
