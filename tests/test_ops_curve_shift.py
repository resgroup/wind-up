import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from wind_up.ops_curve_shift import (
    CurveConfig,
    CurveShiftInput,
    CurveShiftOutput,
    CurveTypes,
    calculate_curve_shift,
    check_for_ops_curve_shift,
)


@pytest.fixture
def fake_power_curve_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wind_speed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "power": [0, 0, np.nan, 1, 3, 6, 10, 15, 22, 30, 36, 39, 40, 40, 40],
        }
    ).set_index("power")


@pytest.fixture
def fake_gen_rpm_curve_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wind_speed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "gen_rpm": [900, 900, 850, 875, 900, 1000, 1100, 1200, 1350, 1500, 1600, 1600, 1600, 1600, 1600],
        }
    ).set_index("gen_rpm")


@pytest.fixture
def fake_pitch_curve_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wind_speed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "pitch": [4, 4, 4, 3, 2, 1, 1, 1, 2, 5, 8, 11, 13, 14, 15],
        }
    ).set_index("pitch")


class TestCurveShiftInput:
    @staticmethod
    def test_acceptable_inputs(fake_power_curve_df: pd.DataFrame) -> None:
        _input = CurveShiftInput(
            turbine_name="anything",
            pre_df=fake_power_curve_df.reset_index(),
            post_df=fake_power_curve_df.reset_index(),
            curve_config=CurveConfig(
                name=CurveTypes.POWER_CURVE.value,
                x_col="wind_speed",
                y_col="power",
                x_bin_width=1,
                warning_threshold=0.01,
            ),
        )

    @pytest.mark.parametrize("column_name", ["wind_speed", "power"])
    def test_missing_column_in_pre_df(self, column_name: str, fake_power_curve_df: pd.DataFrame) -> None:
        with pytest.raises(IndexError, match="Column name missing in dataframe"):
            CurveShiftInput(
                turbine_name="anything",
                pre_df=fake_power_curve_df.reset_index().drop(columns=column_name),
                post_df=(fake_power_curve_df + 2).reset_index(),
                curve_config=CurveConfig(
                    name=CurveTypes.POWER_CURVE.value,
                    x_col="wind_speed",
                    y_col="power",
                    x_bin_width=1,
                    warning_threshold=0.01,
                ),
            )

    @pytest.mark.parametrize("column_name", ["wind_speed", "power"])
    def test_missing_column_in_post_df(self, column_name: str, fake_power_curve_df: pd.DataFrame) -> None:
        with pytest.raises(IndexError, match="Column name missing in dataframe"):
            CurveShiftInput(
                turbine_name="anything",
                pre_df=fake_power_curve_df.reset_index(),
                post_df=(fake_power_curve_df + 2).reset_index().drop(columns=column_name),
                curve_config=CurveConfig(
                    name=CurveTypes.POWER_CURVE.value,
                    x_col="wind_speed",
                    y_col="power",
                    x_bin_width=1,
                    warning_threshold=0.01,
                ),
            )


@pytest.mark.parametrize(
    ("shift_amount", "expected"),
    [
        pytest.param(2.0, -0.21557719054241997, id="shift DOES exceed threshold"),
        pytest.param(0.05, -0.006954837573730166, id="shift DOES NOT exceed threshold"),
    ],
)
def test_calculate_power_curve_shift(shift_amount: float, expected: float, fake_power_curve_df: pd.DataFrame) -> None:
    curve_shift_input = CurveShiftInput(
        turbine_name="anything",
        pre_df=fake_power_curve_df.reset_index(),
        post_df=(fake_power_curve_df + shift_amount).reset_index(),
        curve_config=CurveConfig(name=CurveTypes.POWER_CURVE, x_col="wind_speed", y_col="power"),
    )
    # check that CurveShiftInput pydantic model has removed NaNs
    assert not curve_shift_input.pre_df.isna().to_numpy().any()
    assert not curve_shift_input.post_df.isna().to_numpy().any()
    actual = calculate_curve_shift(curve_shift_input=curve_shift_input)

    np.testing.assert_almost_equal(actual=actual.value, desired=expected)


@pytest.mark.parametrize(
    ("shift_amount", "expected"),
    [
        pytest.param(0.2, -0.00712694877505593, id="shift DOES exceed threshold"),
        pytest.param(0.1, -0.0033534540576795058, id="shift DOES NOT exceed threshold"),
    ],
)
def test_calculate_rpm_curve_shift(
    shift_amount: float, expected: float, fake_gen_rpm_curve_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        actual = calculate_curve_shift(
            curve_shift_input=CurveShiftInput(
                turbine_name="anything",
                pre_df=fake_gen_rpm_curve_df.reset_index(),
                post_df=(fake_gen_rpm_curve_df + shift_amount).reset_index(),
                curve_config=CurveConfig(name=CurveTypes.RPM, x_col="wind_speed", y_col="gen_rpm"),
            )
        )

    np.testing.assert_almost_equal(actual=actual.value, desired=expected)


@pytest.mark.parametrize(
    ("shift_amount", "expected"),
    [
        pytest.param(0.14, -0.1026666666666678, id="shift DOES exceed threshold"),
        pytest.param(0.13, -0.09533333333333438, id="shift DOES NOT exceed threshold"),
    ],
)
def test_calculate_pitch_curve_shift(shift_amount: float, expected: float, fake_pitch_curve_df: pd.DataFrame) -> None:
    actual = calculate_curve_shift(
        curve_shift_input=CurveShiftInput(
            turbine_name="anything",
            pre_df=fake_pitch_curve_df.reset_index(),
            post_df=(fake_pitch_curve_df + shift_amount).reset_index(),
            curve_config=CurveConfig(name=CurveTypes.PITCH, x_col="wind_speed", y_col="pitch"),
        )
    )

    np.testing.assert_almost_equal(actual=actual.value, desired=expected)


@pytest.mark.parametrize(
    ("shift_amount", "expected"),
    [
        pytest.param(2.0, 0.21296296296296302, id="shift DOES exceed threshold"),
        pytest.param(0.05, -0.03981481481481486, id="shift DOES NOT exceed threshold"),
    ],
)
def test_calculate_wind_speed_curve_shift(
    shift_amount: float, expected: float, fake_power_curve_df: pd.DataFrame
) -> None:
    actual = calculate_curve_shift(
        curve_shift_input=CurveShiftInput(
            turbine_name="anything",
            pre_df=fake_power_curve_df.reset_index(),
            post_df=(fake_power_curve_df + shift_amount).reset_index(),
            curve_config=CurveConfig(name=CurveTypes.WIND_SPEED, x_col="power", y_col="wind_speed"),
        )
    )

    np.testing.assert_almost_equal(actual=actual.value, desired=expected)


class TestCheckForOpsCurveShift:
    @pytest.mark.parametrize(
        ("pre_df_or_post_df", "missing_column"),
        [
            ("pre", "wind_speed"),
            ("pre", "power"),
            ("pre", "gen_rpm"),
            ("pre", "pitch"),
            ("post", "wind_speed"),
            ("post", "power"),
            ("post", "gen_rpm"),
            ("post", "pitch"),
        ],
    )
    def test_missing_required_column(
        self,
        pre_df_or_post_df: str,
        missing_column: str,
        fake_power_curve_df: pd.DataFrame,
        fake_gen_rpm_curve_df: pd.DataFrame,
        fake_pitch_curve_df: pd.DataFrame,
    ) -> None:
        _df = pd.concat(
            [
                fake_power_curve_df.reset_index().set_index("wind_speed"),
                fake_gen_rpm_curve_df.reset_index().set_index("wind_speed"),
                fake_pitch_curve_df.reset_index().set_index("wind_speed"),
            ],
            axis=1,
        ).reset_index()

        pre_df = _df.drop(columns=missing_column) if pre_df_or_post_df == "pre" else _df
        post_df = _df.drop(columns=missing_column) if pre_df_or_post_df == "post" else _df

        actual = check_for_ops_curve_shift(
            pre_df=pre_df,
            post_df=post_df,
            wtg_name="anything",
            scada_ws_col="wind_speed",
            pw_col="power",
            rpm_col="gen_rpm",
            pt_col="pitch",
            cfg=Mock(),
            plot_cfg=Mock(),
            plot=False,
        )

        expected = {
            f"{CurveTypes.POWER_CURVE.value}_shift": np.nan,
            f"{CurveTypes.RPM.value}_shift": np.nan,
            f"{CurveTypes.PITCH.value}_shift": np.nan,
            f"{CurveTypes.WIND_SPEED.value}_shift": np.nan,
        }

        assert actual == expected

    def test_calls_funcs_as_intended(
        self, fake_power_curve_df: pd.DataFrame, fake_gen_rpm_curve_df: pd.DataFrame, fake_pitch_curve_df: pd.DataFrame
    ) -> None:
        _df = pd.concat(
            [
                fake_power_curve_df.reset_index().set_index("wind_speed"),
                fake_gen_rpm_curve_df.reset_index().set_index("wind_speed"),
                fake_pitch_curve_df.reset_index().set_index("wind_speed"),
            ],
            axis=1,
        ).reset_index()

        wtg_name = "anything"

        with (
            patch(
                "wind_up.ops_curve_shift.calculate_curve_shift",
                return_value=CurveShiftOutput(value=np.nan, warning_msg=None),
            ) as mock_curve_shift,
            patch("wind_up.ops_curve_shift.compare_ops_curves_pre_post", return_value=None) as mock_plot_func,
        ):
            mock_wind_up_conf = Mock()
            mock_wind_up_conf.toggle = True
            mock_plot_conf = Mock()

            actual = check_for_ops_curve_shift(
                pre_df=_df,
                post_df=_df,
                wtg_name=wtg_name,
                scada_ws_col="wind_speed",
                pw_col="power",
                rpm_col="gen_rpm",
                pt_col="pitch",
                cfg=mock_wind_up_conf,
                plot_cfg=mock_plot_conf,
            )

        # define expected call inputs
        curve_input_power = CurveShiftInput(
            turbine_name=wtg_name,
            pre_df=_df,
            post_df=_df,
            curve_config=CurveConfig(name=CurveTypes.POWER_CURVE, x_col="wind_speed", y_col="power"),
        )
        curve_input_rpm = CurveShiftInput(
            turbine_name=wtg_name,
            pre_df=_df,
            post_df=_df,
            curve_config=CurveConfig(name=CurveTypes.RPM, x_col="power", y_col="gen_rpm"),
        )
        curve_input_pitch = CurveShiftInput(
            turbine_name=wtg_name,
            pre_df=_df,
            post_df=_df,
            curve_config=CurveConfig(name=CurveTypes.PITCH, x_col="wind_speed", y_col="pitch"),
        )
        curve_input_wind_speed = CurveShiftInput(
            turbine_name=wtg_name,
            pre_df=_df,
            post_df=_df,
            curve_config=CurveConfig(name=CurveTypes.WIND_SPEED, x_col="power", y_col="wind_speed"),
        )
        _call_inputs_list = [curve_input_power, curve_input_rpm, curve_input_pitch, curve_input_wind_speed]

        # check calls are made with expected inputs
        for _call, _input in zip(mock_curve_shift.mock_calls, _call_inputs_list):
            pd.testing.assert_frame_equal(_call.kwargs["curve_shift_input"].pre_df, _input.pre_df)
            pd.testing.assert_frame_equal(_call.kwargs["curve_shift_input"].post_df, _input.post_df)
            assert _call.kwargs["curve_shift_input"].model_dump(exclude=["pre_df", "post_df"]) == _input.model_dump(
                exclude=["pre_df", "post_df"]
            )

        mock_plot_func.assert_called_once_with(
            pre_df=_df,
            post_df=_df,
            wtg_name=wtg_name,
            ws_col="wind_speed",
            pw_col="power",
            pt_col="pitch",
            rpm_col="gen_rpm",
            plot_cfg=mock_plot_conf,
            is_toggle_test=mock_wind_up_conf.toggle is not None,
            sub_dir=None,
        )

        expected = {
            f"{CurveTypes.POWER_CURVE.value}_shift": np.nan,
            f"{CurveTypes.RPM.value}_shift": np.nan,
            f"{CurveTypes.PITCH.value}_shift": np.nan,
            f"{CurveTypes.WIND_SPEED.value}_shift": np.nan,
        }

        assert actual == expected
