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
    OpsCurveRequiredColumns,
    calculate_curve_shift,
    check_for_ops_curve_shift,
)


@pytest.fixture
def fake_required_columns() -> OpsCurveRequiredColumns:
    return OpsCurveRequiredColumns(wind_speed="wind_speed", power="active_power", rpm="gen_rpm", pitch="pitch_angle")


@pytest.fixture
def fake_curve_df(fake_required_columns: OpsCurveRequiredColumns) -> pd.DataFrame:
    return pd.DataFrame(
        {
            fake_required_columns.wind_speed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            fake_required_columns.power: [0, 0, np.nan, 1, 3, 6, 10, 15, 22, 30, 36, 39, 40, 40, 40],
            fake_required_columns.rpm: [
                900,
                900,
                850,
                875,
                900,
                1000,
                1100,
                1200,
                1350,
                1500,
                1600,
                1600,
                1600,
                1600,
                1600,
            ],
            fake_required_columns.pitch: [4, 4, 4, 3, 2, 1, 1, 1, 2, 5, 8, 11, 13, 14, 15],
        }
    )


class TestCurveShiftInput:
    @staticmethod
    def test_acceptable_inputs(fake_curve_df: pd.DataFrame, fake_required_columns: OpsCurveRequiredColumns) -> None:
        _input = CurveShiftInput(
            turbine_name="anything",
            pre_df=fake_curve_df,
            post_df=fake_curve_df,
            curve_config=CurveConfig(
                name=CurveTypes.POWER_CURVE.value,
                x_col=fake_required_columns.wind_speed,
                y_col=fake_required_columns.power,
                x_bin_width=1,
                warning_threshold=0.01,
            ),
            ops_curve_required_columns=fake_required_columns,
        )

    @pytest.mark.parametrize("column_name", ["wind_speed", "active_power"])
    def test_missing_column_in_pre_df(
        self, column_name: str, fake_curve_df: pd.DataFrame, fake_required_columns: OpsCurveRequiredColumns
    ) -> None:
        with pytest.raises(IndexError, match=f"'{column_name}' column name missing in pre-dataframe"):
            CurveShiftInput(
                turbine_name="anything",
                pre_df=fake_curve_df.drop(columns=column_name),
                post_df=(fake_curve_df + 2),
                curve_config=CurveConfig(
                    name=CurveTypes.POWER_CURVE.value,
                    x_col=fake_required_columns.wind_speed,
                    y_col=fake_required_columns.power,
                    x_bin_width=1,
                    warning_threshold=0.01,
                ),
                ops_curve_required_columns=fake_required_columns,
            )

    @pytest.mark.parametrize("column_name", ["wind_speed", "active_power"])
    def test_missing_column_in_post_df(
        self, column_name: str, fake_curve_df: pd.DataFrame, fake_required_columns: OpsCurveRequiredColumns
    ) -> None:
        with pytest.raises(IndexError, match=f"'{column_name}' column name missing in post-dataframe"):
            CurveShiftInput(
                turbine_name="anything",
                pre_df=fake_curve_df,
                post_df=(fake_curve_df + 2).drop(columns=column_name),
                curve_config=CurveConfig(
                    name=CurveTypes.POWER_CURVE.value,
                    x_col=fake_required_columns.wind_speed,
                    y_col=fake_required_columns.power,
                    x_bin_width=1,
                    warning_threshold=0.01,
                ),
                ops_curve_required_columns=fake_required_columns,
            )


@pytest.mark.parametrize(
    ("shift_amount", "expected"),
    [
        pytest.param(0.0, 0.0, id="zero"),
        pytest.param(2.0, -0.1376912378303199, id="shift DOES exceed threshold"),
        pytest.param(0.05, -0.004489831851395176, id="shift DOES NOT exceed threshold"),
    ],
)
def test_calculate_power_curve_shift(
    shift_amount: float, expected: float, fake_curve_df: pd.DataFrame, fake_required_columns: OpsCurveRequiredColumns
) -> None:
    curve_shift_input = CurveShiftInput(
        turbine_name="anything",
        pre_df=fake_curve_df,
        post_df=(fake_curve_df + shift_amount),
        curve_config=CurveConfig(
            name=CurveTypes.POWER_CURVE, x_col=fake_required_columns.wind_speed, y_col=fake_required_columns.power
        ),
        ops_curve_required_columns=fake_required_columns,
    )
    # check that CurveShiftInput pydantic model has removed NaNs
    assert not curve_shift_input.pre_df.isna().to_numpy().any()
    assert not curve_shift_input.post_df.isna().to_numpy().any()
    actual = calculate_curve_shift(curve_shift_input=curve_shift_input)

    np.testing.assert_almost_equal(actual=actual.value, desired=expected)


@pytest.mark.parametrize(
    ("shift_amount", "expected"),
    [
        pytest.param(0.2, -0.00865091569970633, id="shift DOES exceed threshold"),
        pytest.param(0.1, -0.004926790475744736, id="shift DOES NOT exceed threshold"),
    ],
)
def test_calculate_rpm_curve_shift(
    shift_amount: float,
    expected: float,
    fake_curve_df: pd.DataFrame,
    fake_required_columns: OpsCurveRequiredColumns,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        actual = calculate_curve_shift(
            curve_shift_input=CurveShiftInput(
                turbine_name="anything",
                pre_df=fake_curve_df,
                post_df=(fake_curve_df + shift_amount),
                curve_config=CurveConfig(name=CurveTypes.RPM, x_col="wind_speed", y_col="gen_rpm"),
                ops_curve_required_columns=fake_required_columns,
            ),
        )

    np.testing.assert_almost_equal(actual=actual.value, desired=expected)


@pytest.mark.parametrize(
    ("shift_amount", "expected"),
    [
        pytest.param(0.0, 0.0, id="zero"),
        pytest.param(0.6, 0.10714285714285765, id="shift DOES exceed threshold"),
        pytest.param(0.5, 0.08928571428571441, id="shift DOES NOT exceed threshold"),
    ],
)
def test_calculate_pitch_curve_shift(
    shift_amount: float, expected: float, fake_curve_df: pd.DataFrame, fake_required_columns: OpsCurveRequiredColumns
) -> None:
    actual = calculate_curve_shift(
        curve_shift_input=CurveShiftInput(
            turbine_name="anything",
            pre_df=fake_curve_df,
            post_df=(fake_curve_df + shift_amount),
            curve_config=CurveConfig(
                name=CurveTypes.PITCH, x_col=fake_required_columns.wind_speed, y_col=fake_required_columns.pitch
            ),
            ops_curve_required_columns=fake_required_columns,
        )
    )

    np.testing.assert_almost_equal(actual=actual.value, desired=expected)


@pytest.mark.parametrize(
    ("shift_amount", "expected"),
    [
        pytest.param(2.0, 0.13811720414537776, id="shift DOES exceed threshold"),
        pytest.param(0.0, -0.04629629629629639, id="shift DOES NOT exceed threshold"),
    ],
)
def test_calculate_wind_speed_curve_shift(
    shift_amount: float, expected: float, fake_curve_df: pd.DataFrame, fake_required_columns: OpsCurveRequiredColumns
) -> None:
    _df = fake_curve_df.copy()
    actual = calculate_curve_shift(
        curve_shift_input=CurveShiftInput(
            turbine_name="anything",
            pre_df=_df,
            post_df=(_df + shift_amount),
            curve_config=CurveConfig(
                name=CurveTypes.WIND_SPEED, x_col=fake_required_columns.power, y_col=fake_required_columns.wind_speed
            ),
            ops_curve_required_columns=fake_required_columns,
        )
    )

    np.testing.assert_almost_equal(actual=actual.value, desired=expected)


class TestCheckForOpsCurveShift:
    @pytest.mark.parametrize(
        ("pre_df_or_post_df", "missing_column"),
        [
            ("pre", "wind_speed"),
            ("pre", "active_power"),
            ("pre", "gen_rpm"),
            ("pre", "pitch_angle"),
            ("post", "wind_speed"),
            ("post", "active_power"),
            ("post", "gen_rpm"),
            ("post", "pitch_angle"),
        ],
    )
    def test_missing_required_column(
        self,
        pre_df_or_post_df: str,
        missing_column: str,
        fake_curve_df: pd.DataFrame,
    ) -> None:
        _df = fake_curve_df.copy()

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
        self, fake_curve_df: pd.DataFrame, fake_required_columns: OpsCurveRequiredColumns
    ) -> None:
        _df = fake_curve_df.copy()

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
                scada_ws_col=fake_required_columns.wind_speed,
                pw_col=fake_required_columns.power,
                rpm_col=fake_required_columns.rpm,
                pt_col=fake_required_columns.pitch,
                cfg=mock_wind_up_conf,
                plot_cfg=mock_plot_conf,
            )

        # define expected call inputs
        curve_input_power = CurveShiftInput(
            turbine_name=wtg_name,
            pre_df=_df,
            post_df=_df,
            ops_curve_required_columns=fake_required_columns,
            curve_config=CurveConfig(
                name=CurveTypes.POWER_CURVE, x_col=fake_required_columns.wind_speed, y_col=fake_required_columns.power
            ),
        )
        curve_input_rpm = CurveShiftInput(
            turbine_name=wtg_name,
            pre_df=_df,
            post_df=_df,
            ops_curve_required_columns=fake_required_columns,
            curve_config=CurveConfig(
                name=CurveTypes.RPM, x_col=fake_required_columns.power, y_col=fake_required_columns.rpm
            ),
        )
        curve_input_pitch = CurveShiftInput(
            turbine_name=wtg_name,
            pre_df=_df,
            post_df=_df,
            ops_curve_required_columns=fake_required_columns,
            curve_config=CurveConfig(
                name=CurveTypes.PITCH, x_col=fake_required_columns.wind_speed, y_col=fake_required_columns.pitch
            ),
        )
        curve_input_wind_speed = CurveShiftInput(
            turbine_name=wtg_name,
            pre_df=_df,
            post_df=_df,
            ops_curve_required_columns=fake_required_columns,
            curve_config=CurveConfig(
                name=CurveTypes.WIND_SPEED, x_col=fake_required_columns.power, y_col=fake_required_columns.wind_speed
            ),
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
            ws_col=fake_required_columns.wind_speed,
            pw_col=fake_required_columns.power,
            pt_col=fake_required_columns.pitch,
            rpm_col=fake_required_columns.rpm,
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
