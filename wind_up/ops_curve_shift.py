from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from wind_up.plots.scada_funcs_plots import compare_ops_curves_pre_post
from wind_up.result_manager import result_manager

if TYPE_CHECKING:
    from wind_up.models import PlotConfig, WindUpConfig


class CurveThresholds(Enum):
    POWER_CURVE = 0.01
    RPM = 0.005
    PITCH = 0.1


class CurveTypes(str, Enum):
    POWER_CURVE = "powercurve"
    RPM = "rpm"
    PITCH = "pitch"


class CurveConfig(BaseModel):
    name: CurveTypes
    x_col: str
    y_col: str
    x_bin_width: int
    warning_threshold: float


class CurveShiftInput(BaseModel):
    turbine_name: str
    pre_df: pd.DataFrame
    post_df: pd.DataFrame
    curve_config: CurveConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_dataframes(self) -> CurveShiftInput:
        # check column names
        required_cols = {self.curve_config.x_col, self.curve_config.y_col}
        columns_missing_in_pre_df = required_cols - set(self.pre_df.columns)
        columns_missing_in_post_df = required_cols - set(self.post_df.columns)
        if columns_missing_in_pre_df or columns_missing_in_post_df:
            err_msg = "Column name missing in dataframe"
            raise IndexError(err_msg)

        # remove NA
        self.pre_df = self.pre_df.dropna(subset=list(required_cols)).copy()
        self.post_df = self.post_df.dropna(subset=list(required_cols)).copy()

        return self


class OpsCurveRequiredColumns(NamedTuple):
    wind_speed: str
    power: str
    pitch: str
    rpm: str


def check_for_ops_curve_shift(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    *,
    wtg_name: str,
    scada_ws_col: str,
    pw_col: str,
    rpm_col: str,
    pt_col: str,
    cfg: WindUpConfig,
    plot_cfg: PlotConfig,
    sub_dir: str | None = None,
    plot: bool = True,
) -> dict[str, float]:
    results_dict = {
        f"{CurveTypes.POWER_CURVE.value}_shift": np.nan,
        f"{CurveTypes.RPM.value}_shift": np.nan,
        f"{CurveTypes.PITCH.value}_shift": np.nan,
    }

    required_cols = OpsCurveRequiredColumns(wind_speed=scada_ws_col, power=pw_col, pitch=pt_col, rpm=rpm_col)

    if not _required_cols_are_present(
        pre_df=pre_df, post_df=post_df, turbine_name=wtg_name, required_ops_curve_columns=required_cols
    ):
        return results_dict

    results_dict[f"{CurveTypes.POWER_CURVE.value}_shift"] = calculate_power_curve_shift(
        turbine_name=wtg_name, pre_df=pre_df, post_df=post_df, x_col=scada_ws_col, y_col=pw_col
    )

    results_dict[f"{CurveTypes.RPM.value}_shift"] = calculate_rpm_curve_shift(
        turbine_name=wtg_name, pre_df=pre_df, post_df=post_df, x_col=pw_col, y_col=rpm_col
    )

    results_dict[f"{CurveTypes.PITCH.value}_shift"] = calculate_pitch_curve_shift(
        turbine_name=wtg_name, pre_df=pre_df, post_df=post_df, x_col=scada_ws_col, y_col=pt_col
    )

    if plot:
        compare_ops_curves_pre_post(
            pre_df=pre_df,
            post_df=post_df,
            wtg_name=wtg_name,
            ws_col=scada_ws_col,
            pw_col=pw_col,
            pt_col=pt_col,
            rpm_col=rpm_col,
            plot_cfg=plot_cfg,
            is_toggle_test=(cfg.toggle is not None),
            sub_dir=sub_dir,
        )

    return results_dict


def calculate_power_curve_shift(
    turbine_name: str, pre_df: pd.DataFrame, post_df: pd.DataFrame, x_col: str, y_col: str
) -> float:
    curve_config = CurveConfig(
        name=CurveTypes.POWER_CURVE.value,
        x_col=x_col,
        y_col=y_col,
        x_bin_width=1,
        warning_threshold=CurveThresholds.POWER_CURVE.value,
    )

    curve_shift_input = CurveShiftInput(
        turbine_name=turbine_name, pre_df=pre_df, post_df=post_df, curve_config=curve_config
    )

    return _calculate_curve_shift(curve_shift_input=curve_shift_input)


def calculate_rpm_curve_shift(
    turbine_name: str, pre_df: pd.DataFrame, post_df: pd.DataFrame, x_col: str, y_col: str
) -> float:
    curve_config = CurveConfig(
        name=CurveTypes.RPM.value, x_col=x_col, y_col=y_col, x_bin_width=0, warning_threshold=CurveThresholds.RPM.value
    )

    curve_shift_input = CurveShiftInput(
        turbine_name=turbine_name, pre_df=pre_df, post_df=post_df, curve_config=curve_config
    )

    return _calculate_curve_shift(curve_shift_input=curve_shift_input)


def calculate_pitch_curve_shift(
    turbine_name: str, pre_df: pd.DataFrame, post_df: pd.DataFrame, x_col: str, y_col: str
) -> float:
    curve_config = CurveConfig(
        name=CurveTypes.PITCH.value,
        x_col=x_col,
        y_col=y_col,
        x_bin_width=1,
        warning_threshold=CurveThresholds.PITCH.value,
    )

    curve_shift_input = CurveShiftInput(
        turbine_name=turbine_name, pre_df=pre_df, post_df=post_df, curve_config=curve_config
    )

    return _calculate_curve_shift(curve_shift_input=curve_shift_input)


def _required_cols_are_present(
    pre_df: pd.DataFrame, post_df: pd.DataFrame, turbine_name: str, required_ops_curve_columns: OpsCurveRequiredColumns
) -> bool:
    # check if all required columns are present
    required_cols = list(required_ops_curve_columns)
    for req_col in required_cols:
        if req_col not in pre_df.columns:
            msg = f"check_for_ops_curve_shift {turbine_name} pre_df missing required column {req_col}"
            result_manager.warning(msg)
            return False
        if req_col not in post_df.columns:
            msg = f"check_for_ops_curve_shift {turbine_name} post_df missing required column {req_col}"
            result_manager.warning(msg)
            return False
    return True


def _calculate_curve_shift(curve_shift_input: CurveShiftInput) -> float:
    conf = curve_shift_input.curve_config
    pre_df = curve_shift_input.pre_df
    post_df = curve_shift_input.post_df
    wtg_name = curve_shift_input.turbine_name

    bins = np.arange(0, pre_df[conf.x_col].max() + conf.x_bin_width, conf.x_bin_width) if conf.x_bin_width > 0 else 10

    mean_curve = pre_df.groupby(pd.cut(pre_df[conf.x_col], bins=bins, retbins=False), observed=True).agg(
        x_mean=pd.NamedAgg(column=conf.x_col, aggfunc="mean"),
        y_mean=pd.NamedAgg(column=conf.y_col, aggfunc="mean"),
    )
    post_df["expected_y"] = np.interp(post_df[conf.x_col], mean_curve["x_mean"], mean_curve["y_mean"])
    mean_df = post_df.mean()

    if conf.y_col == CurveTypes.PITCH.value:
        result = mean_df[conf.y_col] - mean_df["expected_y"]
    else:
        result = (mean_df[conf.y_col] / mean_df["expected_y"] - 1).clip(-1, 1)

    # log warning
    if abs(result) > conf.warning_threshold:
        warning_msg = (
            f"{wtg_name} Ops Curve Shift warning: abs({conf.name}) > {conf.warning_threshold}: {abs(result):.3f}"
        )
        result_manager.warning(warning_msg)

    return result
