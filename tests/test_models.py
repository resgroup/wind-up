import copy
import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from wind_up.models import PrePost, WindUpConfig


def test_lsa_asset_name(test_lsa_t13_config: WindUpConfig) -> None:
    assert test_lsa_t13_config.asset.name == "Lisa Wind Farm"


def test_brt_asset_name(test_brt_t16_pitch: WindUpConfig) -> None:
    assert test_brt_t16_pitch.asset.name == "Bart Wind Farm"


def test_datetimes_lsa_t13(test_lsa_t13_config: WindUpConfig) -> None:
    cfg = test_lsa_t13_config
    assert isinstance(cfg.prepost, PrePost)
    assert cfg.prepost.post_first_dt_utc_start == pd.Timestamp("2021-09-30 00:00:00", tz="UTC")
    assert cfg.prepost.post_last_dt_utc_start == pd.Timestamp("2022-07-20 23:50:00", tz="UTC")
    assert cfg.prepost.pre_first_dt_utc_start == pd.Timestamp("2020-09-29 18:00:00", tz="UTC")
    assert cfg.prepost.pre_last_dt_utc_start == pd.Timestamp("2021-07-20 17:50:00", tz="UTC")
    assert cfg.analysis_first_dt_utc_start == pd.Timestamp("2020-09-29 18:00:00", tz="UTC")
    assert cfg.analysis_last_dt_utc_start == pd.Timestamp("2022-07-20 23:50:00", tz="UTC")
    assert cfg.lt_first_dt_utc_start == pd.Timestamp("2018-07-20 23:50:00", tz="UTC")
    assert cfg.lt_last_dt_utc_start == pd.Timestamp("2021-07-20 17:50:00", tz="UTC")


def test_datetimes_brt_t16(test_brt_t16_pitch: WindUpConfig) -> None:
    cfg = test_brt_t16_pitch
    assert cfg.analysis_first_dt_utc_start == pd.Timestamp("2022-11-03 15:30:00", tz="UTC")
    assert cfg.analysis_last_dt_utc_start == pd.Timestamp("2023-06-08 23:50:00", tz="UTC")
    assert cfg.lt_first_dt_utc_start == pd.Timestamp("2019-10-27 21:30:00", tz="UTC")
    assert cfg.lt_last_dt_utc_start == pd.Timestamp("2022-10-27 15:30:00", tz="UTC")


def test_get_max_rated_power_marge(test_marge_config: WindUpConfig) -> None:
    cfg = test_marge_config
    assert cfg.get_max_rated_power() == 1900


def test_get_max_rated_power_homer(test_homer_config: WindUpConfig) -> None:
    # Homer Wind Farm has two rated powers, 1000 and 1330
    cfg = test_homer_config
    assert cfg.get_max_rated_power() == 1330


def test_list_unique_turbine_types_marge(test_marge_config: WindUpConfig) -> None:
    cfg = test_marge_config
    assert isinstance(cfg.list_unique_turbine_types(), list)
    assert len(cfg.list_unique_turbine_types()) == 1
    assert cfg.list_unique_turbine_types()[0].turbine_type == "A90-1.9MW-90"


def test_list_unique_turbine_types_homer(test_homer_config: WindUpConfig) -> None:
    cfg = test_homer_config
    assert len(cfg.list_unique_turbine_types()) == 2
    assert cfg.list_unique_turbine_types()[0].turbine_type == "B-1.0-62"
    assert cfg.list_unique_turbine_types()[1].turbine_type == "C-1.3-62"


def test_list_turbine_ids_of_type_marge(test_marge_config: WindUpConfig) -> None:
    cfg = test_marge_config
    tt_list = cfg.list_unique_turbine_types()
    assert len(cfg.list_turbine_ids_of_type(tt_list[0])) == 9
    assert cfg.list_turbine_ids_of_type(tt_list[0])[0] == "MRG_T01"
    assert cfg.list_turbine_ids_of_type(tt_list[0])[-1] == "MRG_T09"


def test_list_turbine_ids_of_type_homer(test_homer_config: WindUpConfig) -> None:
    cfg = test_homer_config
    tt_list = cfg.list_unique_turbine_types()
    assert len(cfg.list_turbine_ids_of_type(tt_list[0])) == 1
    assert len(cfg.list_turbine_ids_of_type(tt_list[1])) == 1
    assert cfg.list_turbine_ids_of_type(tt_list[0])[0] == "HMR_T01"
    assert cfg.list_turbine_ids_of_type(tt_list[1])[0] == "HMR_T02"


def test_get_normal_operation_genrpm_range_marge(test_marge_config: WindUpConfig) -> None:
    cfg = test_marge_config
    tt_list = cfg.list_unique_turbine_types()
    assert cfg.get_normal_operation_genrpm_range(tt_list[0]) == (800, 1600)


def test_get_normal_operation_genrpm_range_homer(test_homer_config: WindUpConfig) -> None:
    cfg = test_homer_config
    tt_list = cfg.list_unique_turbine_types()
    assert cfg.get_normal_operation_genrpm_range(tt_list[0]) == (950, 1550)
    assert cfg.get_normal_operation_genrpm_range(tt_list[1]) == (950, 1550)


def test_get_normal_operation_pitch_range_marge(test_marge_config: WindUpConfig) -> None:
    cfg = test_marge_config
    tt_list = cfg.list_unique_turbine_types()
    assert cfg.get_normal_operation_pitch_range(tt_list[0]) == (-10, 40)


def test_get_normal_operation_pitch_range_homer(test_homer_config: WindUpConfig) -> None:
    cfg = test_homer_config
    tt_list = cfg.list_unique_turbine_types()
    assert cfg.get_normal_operation_pitch_range(tt_list[0]) == (-10, 5)
    assert cfg.get_normal_operation_pitch_range(tt_list[1]) == (-10, 5)


def test_turbine_get_latlongs(test_homer_config: WindUpConfig) -> None:
    cfg = test_homer_config
    assert cfg.asset.wtgs[0].get_latlongs() == [(-58.60364145072843, 103.6841410289202052)]
    assert cfg.asset.wtgs[1].get_latlongs() == [(-58.60261454305449, 103.688364968451364)]


class TestWindUpConfigSaveJson:
    def test_fp_extension_is_not_dot_json(self, test_marge_config: WindUpConfig) -> None:
        conf = copy.deepcopy(test_marge_config)
        with pytest.raises(ValueError, match="file_path must end with .json"):
            conf.save_json(file_path=Path("is_not_a_json_file_extension.txt"))

    def test_saves_as_expected(
        self, test_marge_config: WindUpConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        conf = copy.deepcopy(test_marge_config)

        # fake exclusions / northing corrections to test the serialization
        conf.yaw_data_exclusions_utc = [
            (
                "ALL",
                dt.datetime(2000, 1, 1, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2000, 1, 1, 15, 10, tzinfo=dt.timezone.utc),
            )
        ]
        conf.exclusion_periods_utc = [
            (
                "ALL",
                dt.datetime(2000, 1, 1, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2000, 1, 1, 15, 10, tzinfo=dt.timezone.utc),
            )
        ]
        conf.northing_corrections_utc = [("ALL", dt.datetime(2000, 1, 1, 0, 40, tzinfo=dt.timezone.utc), 15)]

        fp = tmp_path / "config.json"

        with caplog.at_level(logging.INFO):
            conf.save_json(file_path=fp)

        assert f"Saved WindUpConfig to {fp}" in caplog.text

        # read the file back and ensure it parses correctly into a WindUpConfig
        with fp.open() as f:
            data = json.load(f)
            WindUpConfig.model_validate(data)
