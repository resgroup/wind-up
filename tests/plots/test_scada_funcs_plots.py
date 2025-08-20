import copy
import logging
import re
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.testing.decorators import image_comparison

from build.lib.wind_up.models import PlotConfig
from wind_up.constants import DataColumns
from wind_up.plots import scada_funcs_plots


class TestAxisLabelFromFieldName:
    @staticmethod
    @pytest.mark.parametrize(
        "field_name_unsupported", ["UnsupportedFieldName", "test_UnsupportedFieldName", "raw_UnsupportedFieldName"]
    )
    def test_unsupported_field_name(field_name_unsupported: str) -> None:
        fname_unsupported_lean = re.sub(r"^.*?_", "", field_name_unsupported)
        msg = (
            f"Failed to construct axis label for field '{field_name_unsupported}' "
            f"because {fname_unsupported_lean} does not have a default unit defined"
        )

        with pytest.raises(ValueError, match=msg):
            scada_funcs_plots._axis_label_from_field_name(field_name=field_name_unsupported)  # noqa: SLF001

    @staticmethod
    @pytest.mark.parametrize(
        ("field_name", "expected"),
        [
            ("ActivePowerMean", "ActivePowerMean [kW]"),
            ("test_ActivePowerMean", "test_ActivePowerMean [kW]"),
            ("raw_ActivePowerMean", "raw_ActivePowerMean [kW]"),
            ("WindSpeedMean", "WindSpeedMean [m/s]"),
            ("YawAngleMean", "YawAngleMean [deg]"),
            ("PitchAngleMean", "PitchAngleMean [deg]"),
            ("GenRpmMean", "GenRpmMean [RPM]"),
            ("AmbientTemp", "AmbientTemp [degC]"),
        ],
    )
    def test_supported_field_name(field_name: str, expected: str) -> None:
        actual = scada_funcs_plots._axis_label_from_field_name(field_name=field_name)  # noqa: SLF001
        assert actual == expected


@pytest.mark.slow
class TestCompareActiveAndReactivePowerPrePost:
    @staticmethod
    @image_comparison(
        baseline_images=["reactive_power_fig_prepost_not_toggle"], remove_text=False, extensions=["png"], style="mpl20"
    )
    def test_not_toggle(hot_windup_components) -> None:  # noqa: ANN001
        assessment_inputs = copy.deepcopy(hot_windup_components)

        test_turbine = assessment_inputs.wind_up_config.test_wtgs[0].name
        _, pre_df, post_df = assessment_inputs.pre_post_splitter.split(
            df=assessment_inputs.scada_df, test_wtg_name=test_turbine
        )

        scada_funcs_plots.compare_active_and_reactive_power_pre_post(
            pre_df=pre_df,
            post_df=post_df,
            wtg_name=test_turbine,
            active_power_col=DataColumns.active_power_mean,
            reactive_power_col=DataColumns.reactive_power_mean,
            plot_cfg=assessment_inputs.plot_cfg,
            is_toggle_test=assessment_inputs.wind_up_config.toggle is not None,
            sub_dir=None,
        )

    @staticmethod
    @image_comparison(
        baseline_images=["reactive_power_fig_prepost_toggle"], remove_text=False, extensions=["png"], style="mpl20"
    )
    def test_toggle(hot_windup_components) -> None:  # noqa: ANN001
        assessment_inputs = copy.deepcopy(hot_windup_components)

        test_turbine = assessment_inputs.wind_up_config.test_wtgs[0].name
        _, pre_df, post_df = assessment_inputs.pre_post_splitter.split(
            df=assessment_inputs.scada_df, test_wtg_name=test_turbine
        )

        # add fake toggle data
        pre_df["test_toggle_off"] = False
        post_df["test_toggle_on"] = False
        pre_df.iloc[::2, pre_df.columns.get_loc("test_toggle_off")] = True
        post_df.iloc[1::2, post_df.columns.get_loc("test_toggle_on")] = True

        scada_funcs_plots.compare_active_and_reactive_power_pre_post(
            pre_df=pre_df,
            post_df=post_df,
            wtg_name=test_turbine,
            active_power_col=DataColumns.active_power_mean,
            reactive_power_col=DataColumns.reactive_power_mean,
            plot_cfg=assessment_inputs.plot_cfg,
            is_toggle_test=True,
            sub_dir=None,
        )

    @staticmethod
    def test_no_reactive_power_column(caplog: pytest.LogCaptureFixture) -> None:
        missing_column = DataColumns.reactive_power_mean
        turbine_name = "AnyTurbine"
        _df = pd.DataFrame(columns=[DataColumns.active_power_mean, "SomeOtherColumn"])
        with caplog.at_level(logging.WARNING):
            scada_funcs_plots.compare_active_and_reactive_power_pre_post(
                pre_df=_df,
                post_df=_df,
                wtg_name=turbine_name,
                active_power_col=DataColumns.active_power_mean,
                reactive_power_col=missing_column,
                plot_cfg=PlotConfig(save_plots=False, plots_dir=Path()),
                is_toggle_test=False,
                sub_dir=None,
            )
        expected_msg = f"Column '{missing_column}' not found, skipping reactive vs active power plot for {turbine_name}"

        log_msg = caplog.records[-1]
        assert log_msg.levelname == logging.getLevelName(logging.WARNING)
        assert expected_msg == log_msg.getMessage()
