from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib as mpl
import pytest

from wind_up.constants import PROJECTROOT_DIR
from wind_up.models import WindUpConfig

mpl.use("Agg")

TEST_DATA_FLD = Path(__file__).parent / "test_data"
TEST_CONFIG_DIR = TEST_DATA_FLD / "config"
CACHE_DIR = PROJECTROOT_DIR / "cache"


def _set_legacy_datetimes(cfg: WindUpConfig) -> None:
    """Set the datetimes as they were prior to v0.3."""
    cfg.prepost.pre_first_dt_utc_start = cfg.upgrade_first_dt_utc_start - dt.timedelta(
        days=(cfg.years_offset_for_pre_period * 365.25)
    )
    cfg.prepost.pre_last_dt_utc_start = cfg.analysis_last_dt_utc_start - dt.timedelta(
        days=(cfg.years_offset_for_pre_period * 365.25)
    )
    cfg.lt_last_dt_utc_start = cfg.prepost.pre_last_dt_utc_start
    cfg.lt_first_dt_utc_start = cfg.lt_last_dt_utc_start - dt.timedelta(days=(cfg.years_for_lt_distribution * 365.25))
    cfg.detrend_last_dt_utc_start = cfg.lt_last_dt_utc_start
    cfg.detrend_first_dt_utc_start = cfg.detrend_last_dt_utc_start - dt.timedelta(days=(cfg.years_for_detrend * 365.25))
    return cfg


@pytest.fixture
def test_lsa_t13_config() -> WindUpConfig:
    return _set_legacy_datetimes(WindUpConfig.from_yaml(TEST_CONFIG_DIR / "test_LSA_T13.yaml"))


@pytest.fixture
def test_brt_t16_pitch() -> WindUpConfig:
    return _set_legacy_datetimes(WindUpConfig.from_yaml(TEST_CONFIG_DIR / "test_BRT_T16_pitch.yaml"))


@pytest.fixture
def test_marge_config() -> WindUpConfig:
    return _set_legacy_datetimes(WindUpConfig.from_yaml(TEST_CONFIG_DIR / "test_Marge.yaml"))


@pytest.fixture
def test_homer_config() -> WindUpConfig:
    cfg = _set_legacy_datetimes(WindUpConfig.from_yaml(TEST_CONFIG_DIR / "test_Homer.yaml"))
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
    cfg = _set_legacy_datetimes(WindUpConfig.from_yaml(TEST_CONFIG_DIR / "test_Homer.yaml"))
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
