import copy
import datetime as dt
import json
import logging
import re
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from tests.conftest import TEST_CONFIG_DIR
from wind_up.models import DEFAULT_TIMEBASE_S, PrePost, WindUpConfig


def test_lsa_asset_name(test_lsa_t13_config: WindUpConfig) -> None:
    assert test_lsa_t13_config.asset.name == "Lisa Wind Farm"


def test_brt_asset_name(test_brt_t16_pitch: WindUpConfig) -> None:
    assert test_brt_t16_pitch.asset.name == "Bart Wind Farm"


def test_legacy_datetimes_lsa_t13(test_lsa_t13_config: WindUpConfig) -> None:
    """Legacy test of datetimes calculated when loading LSA T13 yaml."""
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


def test_datetimes_lsa_t13() -> None:
    """Test datetimes calculated when loading LSA T13 yaml."""
    cfg = WindUpConfig.from_yaml(TEST_CONFIG_DIR / "test_LSA_T13.yaml")
    assert isinstance(cfg.prepost, PrePost)
    assert cfg.prepost.post_first_dt_utc_start == pd.Timestamp("2021-09-30 00:00:00", tz="UTC")
    assert cfg.prepost.post_last_dt_utc_start == pd.Timestamp("2022-07-20 23:50:00", tz="UTC")
    assert cfg.prepost.pre_first_dt_utc_start == pd.Timestamp("2020-09-30 00:00:00", tz="UTC")
    assert cfg.prepost.pre_last_dt_utc_start == pd.Timestamp("2021-07-20 23:50:00", tz="UTC")
    assert cfg.analysis_first_dt_utc_start == pd.Timestamp("2020-09-30 00:00:00", tz="UTC")
    assert cfg.analysis_last_dt_utc_start == pd.Timestamp("2022-07-20 23:50:00", tz="UTC")
    assert cfg.lt_first_dt_utc_start == pd.Timestamp("2018-07-20 23:50:00", tz="UTC")
    assert cfg.lt_last_dt_utc_start == pd.Timestamp("2021-07-20 23:50:00", tz="UTC")
    assert cfg.detrend_first_dt_utc_start == pd.Timestamp("2018-07-20 23:50:00", tz="UTC")
    assert cfg.detrend_last_dt_utc_start == pd.Timestamp("2021-07-20 23:50:00", tz="UTC")


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
        with pytest.raises(ValueError, match=r"file_path must end with .json"):
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


class TestMatchingMonthsOverride:
    """Tests date ranges of pre-upgrade period based on `WindUpConfig.from_yaml` config file loading."""

    def test_without_pre_last_dt_utc_start(self) -> None:
        """Checks default matching-months logic.

        Tests constructed pre-upgrade period when `pre_last_dt_utc_start` is not specified in the config file.
        """
        cfg = WindUpConfig.from_yaml(TEST_CONFIG_DIR / "test_LSA_T13.yaml")
        assert cfg.prepost.pre_first_dt_utc_start == pd.Timestamp("2020-09-30 00:00:00+0000", tz="UTC")
        assert cfg.prepost.pre_last_dt_utc_start == pd.Timestamp("2021-07-20 23:50:00+0000", tz="UTC")
        assert cfg.upgrade_first_dt_utc_start == pd.Timestamp("2021-09-30 00:00:00+0000", tz="UTC")
        assert cfg.prepost.post_first_dt_utc_start == pd.Timestamp("2021-09-30 00:00:00+0000", tz="UTC")
        assert cfg.prepost.post_last_dt_utc_start == pd.Timestamp("2022-07-20 23:50:00+0000", tz="UTC")

    def test_with_pre_last_dt_utc_start(self) -> None:
        """Check that a pre-upgrade period (start and end date) may be specified explicitly in the config file.

        If a config file contains an entry for `pre_last_dt_utc_start`, this should override the default matching-months
        logic.
        """
        # Modify yaml file and then load it to ensure the override works as expected
        yaml_path = TEST_CONFIG_DIR / "test_LSA_T13.yaml"
        with yaml_path.open() as f:
            yaml_str = f.read()
            # Add a new line with "pre_last_dt_utc_start" value
            new_line = "\npre_last_dt_utc_start: 2021-09-29 23:50:00+0000\n"
            yaml_str += new_line
            modified_yaml_path = TEST_CONFIG_DIR / "modified_test_LSA_T13.yaml"
            with modified_yaml_path.open("w") as mf:
                mf.write(yaml_str)

        cfg = WindUpConfig.from_yaml(modified_yaml_path)

        # delete the modified yaml file after loading the config
        modified_yaml_path.unlink()

        assert cfg.prepost.pre_first_dt_utc_start == pd.Timestamp("2020-09-30 00:00:00+0000", tz="UTC")
        assert cfg.prepost.pre_last_dt_utc_start == pd.Timestamp("2021-09-29 23:50:00+0000", tz="UTC")
        assert cfg.upgrade_first_dt_utc_start == pd.Timestamp("2021-09-30 00:00:00+0000", tz="UTC")
        assert cfg.prepost.post_first_dt_utc_start == pd.Timestamp("2021-09-30 00:00:00+0000", tz="UTC")
        assert cfg.prepost.post_last_dt_utc_start == pd.Timestamp("2022-07-20 23:50:00+0000", tz="UTC")


def test_windupconfig_with_extended_post_period_length() -> None:
    """Check that the pre-period does not extend over the upgrade date.

    Check that if the `analysis_last_dt_utc_start` is >1 year post upgrade that if the `years_offset_for_pre_period` is
    1 year, then the maximum end date of the pre-period is one timebase before upgrade date.
    """
    # Modify yaml file and then load it to ensure the override works as expected
    yaml_path = TEST_CONFIG_DIR / "test_LSA_T13.yaml"
    with yaml_path.open() as f:
        yaml_str = f.read()

    # Replace the existing line containing "pre_last_dt_utc_start"
    analysis_end = "2026-01-01 23:50:00+0000"
    yaml_str = re.sub(r"analysis_last_dt_utc_start:.*", f"analysis_last_dt_utc_start: {analysis_end}", yaml_str)

    modified_yaml_path = TEST_CONFIG_DIR / "modified_test_LSA_T13.yaml"
    with modified_yaml_path.open("w") as mf:
        mf.write(yaml_str)

    cfg = WindUpConfig.from_yaml(modified_yaml_path)

    # delete the modified yaml file after loading the config
    modified_yaml_path.unlink()

    assert cfg.prepost.pre_first_dt_utc_start == pd.Timestamp("2020-09-30 00:00:00+0000", tz="UTC")
    assert cfg.prepost.pre_last_dt_utc_start == (
        cfg.upgrade_first_dt_utc_start - pd.Timedelta(seconds=DEFAULT_TIMEBASE_S)
    )  # key check
    assert cfg.upgrade_first_dt_utc_start == pd.Timestamp("2021-09-30 00:00:00+0000", tz="UTC")
    assert cfg.prepost.post_first_dt_utc_start == pd.Timestamp("2021-09-30 00:00:00+0000", tz="UTC")
    assert cfg.prepost.post_last_dt_utc_start == pd.Timestamp(analysis_end, tz="UTC")


class TestPrePostValidation:
    @pytest.fixture
    def valid_dates(self) -> dict[str, dt.datetime]:
        return {
            "pre_first_dt_utc_start": dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc),
            "pre_last_dt_utc_start": dt.datetime(2000, 1, 15, tzinfo=dt.timezone.utc),
            "post_first_dt_utc_start": dt.datetime(2000, 1, 16, tzinfo=dt.timezone.utc),
            "post_last_dt_utc_start": dt.datetime(2000, 1, 29, tzinfo=dt.timezone.utc),
        }

    def test_valid_prepost(self, valid_dates: dict[str, dt.datetime]) -> None:
        PrePost(**valid_dates)

    def test_pre_period_start_after_end_raises(self, valid_dates: dict[str, dt.datetime]) -> None:
        valid_dates["pre_first_dt_utc_start"] = dt.datetime(2000, 1, 20, tzinfo=dt.timezone.utc)
        with pytest.raises(
            ValidationError, match=re.escape("Start date of pre-period must be before the end date of pre-period.")
        ):
            PrePost(**valid_dates)

    def test_pre_period_equal_start_and_end_is_invalid(self, valid_dates: dict[str, dt.datetime]) -> None:
        valid_dates["pre_first_dt_utc_start"] = valid_dates["pre_last_dt_utc_start"]
        with pytest.raises(
            ValidationError, match=re.escape("Start date of pre-period must be before the end date of pre-period.")
        ):
            PrePost(**valid_dates)

    def test_post_period_start_after_end_raises(self, valid_dates: dict[str, dt.datetime]) -> None:
        valid_dates["post_first_dt_utc_start"] = dt.datetime(2000, 1, 30, tzinfo=dt.timezone.utc)
        with pytest.raises(
            ValidationError, match=re.escape("Start date of post-period must be before the end date of post-period.")
        ):
            PrePost(**valid_dates)

    def test_post_period_equal_start_and_end_is_invalid(self, valid_dates: dict[str, dt.datetime]) -> None:
        valid_dates["post_first_dt_utc_start"] = valid_dates["post_last_dt_utc_start"]
        with pytest.raises(
            ValidationError, match=re.escape("Start date of post-period must be before the end date of post-period.")
        ):
            PrePost(**valid_dates)

    def test_pre_last_after_post_first_raises(self, valid_dates: dict[str, dt.datetime]) -> None:
        valid_dates["pre_last_dt_utc_start"] = dt.datetime(2000, 1, 20, tzinfo=dt.timezone.utc)
        with pytest.raises(
            ValidationError, match=re.escape("End date of pre-period must be before the Start date of post-period.")
        ):
            PrePost(**valid_dates)

    def test_pre_last_equal_post_first_raises(self, valid_dates: dict[str, dt.datetime]) -> None:
        same_dt = dt.datetime(2000, 1, 16, tzinfo=dt.timezone.utc)
        valid_dates["pre_last_dt_utc_start"] = same_dt
        valid_dates["post_first_dt_utc_start"] = same_dt
        with pytest.raises(
            ValidationError, match=re.escape("End date of pre-period must be before the Start date of post-period.")
        ):
            PrePost(**valid_dates)
