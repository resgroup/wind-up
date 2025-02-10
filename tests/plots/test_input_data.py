import copy
import logging
import re

import numpy as np
import pandas as pd
import pytest
from matplotlib.testing.decorators import image_comparison

from examples.smarteole_example import SmarteoleData
from wind_up.interface import AssessmentInputs
from wind_up.plots.input_data import plot_input_data_timeline

logger = logging.getLogger(__name__)


class TestInputDataTimeline:
    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize("exclusion_period_attribute_name", ["yaw_data_exclusions_utc", "exclusion_periods_utc"])
    def test_data_is_present_within_an_exclusion_period(
        self, exclusion_period_attribute_name: str, smarteole_assessment_inputs: tuple[AssessmentInputs, SmarteoleData]
    ) -> None:
        """Test that a ValueError is raised if any non-NaN data is present within any exclusion period."""

        assessment_inputs = copy.deepcopy(smarteole_assessment_inputs[0])

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
    def test_toggle(self, smarteole_assessment_inputs: tuple[AssessmentInputs, SmarteoleData]) -> None:
        """Test plotting timeline of input data on the Smarteole wind farm."""

        assessment_inputs = copy.deepcopy(smarteole_assessment_inputs[0])

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
    def test_prepost(self, smarteole_assessment_inputs: tuple[AssessmentInputs, SmarteoleData]) -> None:
        """Test plotting timeline of input data on the Smarteole wind farm."""

        assessment_inputs = copy.deepcopy(smarteole_assessment_inputs[0])

        # manual adjustments to the configuration for the test
        assessment_inputs.cfg.toggle = None
        assessment_inputs.cfg.upgrade_first_dt_utc_start = pd.Timestamp("2020-03-01T00:00:00+0000")

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
