import copy
import datetime as dt
import logging
import re
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.testing.decorators import image_comparison

from tests.test_smarteole import _create_config
from wind_up.constants import DataColumns
from wind_up.interface import AssessmentInputs
from wind_up.models import PrePost
from wind_up.plots.input_data import plot_input_data_timeline

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def smarteole_assessment_inputs() -> Generator[AssessmentInputs, None, None]:
    with tempfile.TemporaryDirectory() as tmpdirname:  # cannot use pytest tmp_path because of fixture scope mismatch
        yield _create_config(tmp_path=Path(tmpdirname))


class TestInputDataTimeline:
    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize("exclusion_period_attribute_name", ["yaw_data_exclusions_utc", "exclusion_periods_utc"])
    def test_data_is_present_within_an_exclusion_period(
        self, exclusion_period_attribute_name: str, smarteole_assessment_inputs: AssessmentInputs
    ) -> None:
        """Test that a ValueError is raised if any non-NaN data is present within any exclusion period."""

        assessment_inputs = copy.deepcopy(smarteole_assessment_inputs)

        setattr(
            assessment_inputs.cfg,
            exclusion_period_attribute_name,
            [
                ("SMV1", pd.Timestamp("2020-03-01T00:00:00+0000"), pd.Timestamp("2020-03-03T00:00:00+0000")),
            ],
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Data is not all NaN within exclusion period "
                f"{getattr(assessment_inputs.cfg, exclusion_period_attribute_name)[0]}"
            ),
        ):
            plot_input_data_timeline(assessment_inputs)

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore")
    @image_comparison(
        baseline_images=["input_data_timeline_fig_toggle"], remove_text=False, extensions=["png"], style="mpl20"
    )
    def test_toggle(self, smarteole_assessment_inputs: AssessmentInputs) -> None:
        """Test plotting timeline of input data on the Smarteole wind farm."""

        assessment_inputs = copy.deepcopy(smarteole_assessment_inputs)

        assessment_inputs.cfg.yaw_data_exclusions_utc = [
            ("SMV1", pd.Timestamp("2020-03-01T00:00:00+0000"), pd.Timestamp("2020-03-03T00:00:00+0000")),
            ("SMV4", pd.Timestamp("2020-04-02T00:00:00+0000"), pd.Timestamp("2020-05-20T00:00:00+0000")),
        ]

        assessment_inputs.cfg.exclusion_periods_utc = [
            ("SMV3", pd.Timestamp("2020-04-01T00:00:00+0000"), pd.Timestamp("2020-04-10T00:00:00+0000")),
            ("SMV6", pd.Timestamp("2020-03-10T00:00:00+0000"), pd.Timestamp("2020-03-12T00:00:00+0000")),
        ]

        for exclusion in assessment_inputs.cfg.exclusion_periods_utc:
            turbine_name = exclusion[0]
            start, end = exclusion[1], exclusion[2]
            assessment_inputs.wf_df.loc[pd.IndexSlice[turbine_name, start:end], "ActivePowerMean"] = np.nan

        for exclusion in assessment_inputs.cfg.yaw_data_exclusions_utc:
            turbine_name = exclusion[0]
            start, end = exclusion[1], exclusion[2]
            assessment_inputs.wf_df.loc[pd.IndexSlice[turbine_name, start:end], "YawAngleMean"] = np.nan

        plot_input_data_timeline(assessment_inputs)

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore")
    @image_comparison(
        baseline_images=["input_data_timeline_fig_prepost"], remove_text=False, extensions=["png"], style="mpl20"
    )
    def test_prepost(self, smarteole_assessment_inputs: AssessmentInputs) -> None:
        """Test plotting timeline of input data on the Smarteole wind farm."""

        assessment_inputs = copy.deepcopy(smarteole_assessment_inputs)

        # manual adjustments to the configuration for the test
        # ----------------------------------------------------
        assessment_inputs.cfg.toggle = None
        assessment_inputs.cfg.years_for_lt_distribution = 5
        assessment_inputs.cfg.years_for_detrend = 3
        assessment_inputs.cfg.years_offset_for_pre_period = 1
        assessment_inputs.cfg.upgrade_first_dt_utc_start = pd.Timestamp("2020-04-15T00:00:00+0000")
        # analysis period
        assessment_inputs.cfg.analysis_last_dt_utc_start = (
            assessment_inputs.cfg.upgrade_first_dt_utc_start + pd.DateOffset(months=8)
        )
        assessment_inputs.cfg.analysis_first_dt_utc_start = (
            assessment_inputs.cfg.analysis_last_dt_utc_start - pd.DateOffset(months=12)
        )
        # long-term period
        assessment_inputs.cfg.lt_last_dt_utc_start = assessment_inputs.cfg.upgrade_first_dt_utc_start - dt.timedelta(
            days=7
        )  # go back 1 week for buffer before toggling
        assessment_inputs.cfg.lt_first_dt_utc_start = assessment_inputs.cfg.lt_last_dt_utc_start - dt.timedelta(
            days=(assessment_inputs.cfg.years_for_lt_distribution * 365.25),
        )
        # detrend period
        assessment_inputs.cfg.detrend_last_dt_utc_start = assessment_inputs.cfg.lt_last_dt_utc_start
        assessment_inputs.cfg.detrend_first_dt_utc_start = (
            assessment_inputs.cfg.detrend_last_dt_utc_start
            - dt.timedelta(days=assessment_inputs.cfg.years_for_detrend * 365.25)
        )
        # pre-post period
        pre_post = PrePost(
            post_first_dt_utc_start=assessment_inputs.cfg.upgrade_first_dt_utc_start,
            post_last_dt_utc_start=assessment_inputs.cfg.analysis_last_dt_utc_start,
            pre_first_dt_utc_start=assessment_inputs.cfg.upgrade_first_dt_utc_start
            - dt.timedelta(days=(365.25 * assessment_inputs.cfg.years_offset_for_pre_period)),
            pre_last_dt_utc_start=assessment_inputs.cfg.analysis_last_dt_utc_start
            - dt.timedelta(days=(365.25 * assessment_inputs.cfg.years_offset_for_pre_period)),
        )
        assessment_inputs.cfg.prepost = pre_post
        # extend scada data with fake time series data
        d1 = assessment_inputs.cfg.lt_first_dt_utc_start
        d2 = assessment_inputs.cfg.analysis_last_dt_utc_start
        _df = pd.DataFrame(
            columns=assessment_inputs.wf_df.columns,
            index=pd.MultiIndex.from_product(
                [
                    list(assessment_inputs.wf_df.index.get_level_values(0).unique()),
                    pd.date_range(start=d1, end=d2, freq=pd.Timedelta(minutes=10), tz="UTC"),
                ],
                names=["TurbineName", "TimeStamp_StartFormat"],
            ),
        )
        _df["ActivePowerMean"] = 0.1
        _df["YawAngleMean"] = 0.2
        assessment_inputs.wf_df = _df

        assessment_inputs.cfg.yaw_data_exclusions_utc = [
            ("SMV1", pd.Timestamp("2020-03-01T00:00:00+0000"), pd.Timestamp("2020-03-03T00:00:00+0000")),
            ("SMV4", pd.Timestamp("2020-04-02T00:00:00+0000"), pd.Timestamp("2020-05-20T00:00:00+0000")),
        ]

        assessment_inputs.cfg.exclusion_periods_utc = [
            ("SMV3", pd.Timestamp("2020-04-01T00:00:00+0000"), pd.Timestamp("2020-04-10T00:00:00+0000")),
            ("SMV6", pd.Timestamp("2020-03-10T00:00:00+0000"), pd.Timestamp("2020-03-12T00:00:00+0000")),
        ]

        for exclusion in assessment_inputs.cfg.exclusion_periods_utc:
            turbine_name = exclusion[0]
            start, end = exclusion[1], exclusion[2]
            assessment_inputs.wf_df.loc[pd.IndexSlice[turbine_name, start:end], "ActivePowerMean"] = np.nan

        for exclusion in assessment_inputs.cfg.yaw_data_exclusions_utc:
            turbine_name = exclusion[0]
            start, end = exclusion[1], exclusion[2]
            assessment_inputs.wf_df.loc[pd.IndexSlice[turbine_name, start:end], "YawAngleMean"] = np.nan

        plot_input_data_timeline(assessment_inputs)

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore")
    def test_prepost_not_set_on_wu_cfg(self, smarteole_assessment_inputs: AssessmentInputs) -> None:
        """Test that a ValueError is raised if prepost is not set on the configuration."""

        assessment_inputs = copy.deepcopy(smarteole_assessment_inputs)
        assessment_inputs.cfg.toggle = None  # ensure the analysis type is not toggle
        assessment_inputs.cfg.prepost = None  # remove prepost from the configuration

        with pytest.raises(ValueError, match="PrePost attribute is not set on WindUpConfig."):
            plot_input_data_timeline(assessment_inputs)

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore")
    @image_comparison(
        baseline_images=["input_data_timeline_fig_figsize"], remove_text=False, extensions=["png"], style="mpl20"
    )
    def test_figsize(self, smarteole_assessment_inputs: AssessmentInputs) -> None:
        assessment_inputs = copy.deepcopy(smarteole_assessment_inputs)

        dfs = []
        any_float = 0.1
        n_turbines_to_add = 10
        for t in range(n_turbines_to_add):
            new_turbine = f"figsize_t0{t}"
            _df = pd.DataFrame(
                any_float,
                index=assessment_inputs.wf_df.index.get_level_values(1),
                columns=assessment_inputs.wf_df.columns,
            )
            _df[DataColumns.turbine_name] = new_turbine
            _df = _df.reset_index().set_index([DataColumns.turbine_name, "TimeStamp_StartFormat"])
            dfs.append(_df)

        assessment_inputs.wf_df = pd.concat([assessment_inputs.wf_df, *dfs])

        plot_input_data_timeline(assessment_inputs=assessment_inputs)
