"""Plot timeline of input data with key milestones and data exclusions."""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from wind_up.constants import DataColumns

if TYPE_CHECKING:
    import datetime as dt
    from pathlib import Path

    import pandas as pd

    from wind_up.interface import AssessmentInputs


logger = logging.getLogger(__name__)


class DateRangeColors(str, Enum):
    """Colors for date ranges."""

    PRE = "#59a89c"
    POST = "#7E4794"
    ANALYSIS = "#0b81a2"
    DETREND = "#9d2c00"
    LONG_TERM = "#c8c8c8"


def _validate_data_within_exclusions(
    exclusions: list[tuple[str, dt.datetime, dt.datetime]], wf_series: pd.DataFrame
) -> None:
    _series = wf_series.copy()
    for exclusion in exclusions:
        mask = _series.index.get_level_values(DataColumns.turbine_name) == exclusion[0]
        _data = _series.loc[mask].droplevel(level=DataColumns.turbine_name, axis=0).sort_index()
        if _data[exclusion[1] : exclusion[2]].notna().any():  # type: ignore[misc]
            _msg = f"Data is not all NaN within exclusion period {exclusion}"
            logger.warning(_msg)


def _plot_exclusion(
    *,
    y_value: int,
    y_values: list[int],
    turbine_name: str,
    name_for_legend: str,
    exclusions: list[tuple[str, dt.datetime, dt.datetime]],
    trace_format: dict,
    ax: plt.Axes,
) -> None:
    for _count, exclusion in enumerate(exclusions):
        _name_for_legend = {"label": name_for_legend} if _count == 0 else {}
        left, right = exclusion[1], exclusion[2]
        if exclusion[0] == turbine_name:
            ax.barh(y_value, left=left, width=right - left, **trace_format, **_name_for_legend)  # type: ignore[arg-type]
        elif exclusion[0].lower() == "all":
            for y in y_values:
                ax.barh(y, left=left, width=right - left, **trace_format, **_name_for_legend)  # type: ignore[arg-type]


def _plot_data_coverage(
    *, turbine_series: pd.Series, ax: plt.Axes, y_value: float, color: str, label: str | None = None
) -> None:
    mask = turbine_series.notna()
    column_data = turbine_series.copy()
    column_data.loc[mask] = y_value
    _label = {"label": label} if label else {}
    ax.plot(column_data.index, column_data, color=color, linewidth=1, **_label)  # type: ignore[arg-type]


def plot_input_data_timeline(
    assessment_inputs: AssessmentInputs,
    *,
    figsize: tuple[int, int] | None = None,
    height_ratios: tuple[int, int] | None = None,
    save_to_folder: Path | None = None,
    show_plots: bool = True,
    scada_data_column_for_power: str = DataColumns.active_power_mean,
    scada_data_column_for_yaw_angle: str = DataColumns.yaw_angle_mean,
) -> plt.Figure:
    """Plot timeline of input data with key milestones and data exclusions.

    This function does not do any data filtering itself, but instead only displays the data as it is provided.

    :param assessment_inputs: wind-up configuration and time series data for the assessment
    :param figsize: size of the plot figure, if `None` it will be auto-sized based on the number of turbines
    :param height_ratios: ratios for the two subplots, if `None` it will be auto-sized based on the number of turbines
    :param save_to_folder: directory in which to save the plot
    :param show_plots: whether to show the interactive plot or not
    :param scada_data_column_for_power: column name in the wind farm DataFrame to use for power data coverage plotting
    :param scada_data_column_for_yaw_angle: column name in the wind farm DataFrame to use for yaw data coverage plotting
    :return: figure object
    """

    _wu_cfg = assessment_inputs.cfg
    _df = assessment_inputs.wf_df.copy()
    pwr_col = scada_data_column_for_power
    yaw_col = scada_data_column_for_yaw_angle

    _validate_data_within_exclusions([*_wu_cfg.exclusion_periods_utc], _df[pwr_col])
    _validate_data_within_exclusions([*_wu_cfg.yaw_data_exclusions_utc], _df[yaw_col])

    turbines = list(_df.index.get_level_values(DataColumns.turbine_name).unique())
    n_turbines = len(turbines)
    y_values = list(range(1, n_turbines + 1))

    if figsize is None:
        logger.debug("Auto-sizing figure based on number of turbines")
        figsize = (15, n_turbines)
        if height_ratios is None:
            height_ratios = (max(int(np.floor(n_turbines / 3)), 2), 1)

    fig, (ax_turbines, ax_wf) = plt.subplots(
        ncols=1,
        nrows=2,
        sharex=True,
        figsize=figsize,
        gridspec_kw={"height_ratios": list(height_ratios)},  # type:ignore[arg-type]
    )

    for y_value_count, t in enumerate(turbines):
        y_value = y_value_count + 1

        # turbine power data coverage
        pwr_label = f"{pwr_col} is not NaN" if y_value_count == 0 else None
        turbine_series_power = (
            _df.query(f"{DataColumns.turbine_name} == '{t}'")
            .droplevel(level=DataColumns.turbine_name, axis=0)
            .sort_index()
        )[pwr_col]
        _plot_data_coverage(
            turbine_series=turbine_series_power,
            ax=ax_turbines,
            y_value=y_value - 0.1,  # to offset the power data from the yaw data on the axis
            color="darkblue",
            label=pwr_label,
        )

        # turbine yaw data coverage
        yaw_label = f"{yaw_col} is not NaN" if y_value_count == 0 else None
        turbine_series_yaw = (
            _df.query(f"{DataColumns.turbine_name} == '{t}'")
            .droplevel(level=DataColumns.turbine_name, axis=0)
            .sort_index()
        )[yaw_col]
        _plot_data_coverage(
            turbine_series=turbine_series_yaw,
            ax=ax_turbines,
            y_value=y_value + 0.1,  # to offset the yaw data from the power data on the axis
            color="lightskyblue",
            label=yaw_label,
        )

        # yaw exclusions
        trace_fmt_yaw = {"height": 0.5, "color": "black", "alpha": 0.5}
        _plot_exclusion(
            y_value=y_value,
            y_values=y_values,
            turbine_name=t,
            name_for_legend="Yaw Exclusion Period",
            exclusions=_wu_cfg.yaw_data_exclusions_utc,
            trace_format=trace_fmt_yaw,
            ax=ax_turbines,
        )

        # general exclusions
        trace_fmt_general = {"height": 0.5, "color": "red", "alpha": 0.5}
        _plot_exclusion(
            y_value=y_value,
            y_values=y_values,
            turbine_name=t,
            name_for_legend="Exclusion Period",
            exclusions=_wu_cfg.exclusion_periods_utc,
            trace_format=trace_fmt_general,
            ax=ax_turbines,
        )

    # plot wind farm
    # --------------

    _key_dates = [
        _wu_cfg.analysis_first_dt_utc_start,
        _wu_cfg.analysis_last_dt_utc_start,
        _wu_cfg.detrend_first_dt_utc_start,
        _wu_cfg.detrend_last_dt_utc_start,
        _wu_cfg.lt_first_dt_utc_start,
        _wu_cfg.lt_last_dt_utc_start,
    ]

    # extend x-axis to show key dates
    x_range_extension = (max(_key_dates) - min(_key_dates)) * 0.05
    x_min = min(_key_dates) - x_range_extension
    x_max = max(_key_dates) + x_range_extension

    key_dates_styles = [
        {
            "trace_fmt": {"height": 0.5, "color": DateRangeColors.DETREND},
            "label": "Detrend Period",
            "left": _wu_cfg.detrend_first_dt_utc_start,
            "right": _wu_cfg.detrend_last_dt_utc_start,
        },
        {
            "trace_fmt": {"height": 0.5, "color": DateRangeColors.LONG_TERM},
            "label": "Long Term Period",
            "left": _wu_cfg.lt_first_dt_utc_start,
            "right": _wu_cfg.lt_last_dt_utc_start,
        },
    ]

    if _wu_cfg.toggle is None:
        if _wu_cfg.prepost is None:
            _msg = "PrePost attribute is not set on WindUpConfig."
            raise ValueError(_msg)
        pre_style = {
            "trace_fmt": {"height": 0.5, "color": DateRangeColors.PRE},
            "label": "Pre-Upgrade Period",
            "left": _wu_cfg.prepost.pre_first_dt_utc_start,
            "right": _wu_cfg.prepost.pre_last_dt_utc_start,
        }
        post_style = {
            "trace_fmt": {"height": 0.5, "color": DateRangeColors.POST},
            "label": "Post-Upgrade Period",
            "left": _wu_cfg.prepost.post_first_dt_utc_start,
            "right": _wu_cfg.prepost.post_last_dt_utc_start,
        }

        key_dates_styles = [post_style, pre_style, *key_dates_styles]
    else:
        key_dates_styles = [
            {
                "trace_fmt": {"height": 0.5, "color": DateRangeColors.PRE},
                "label": "Toggle Period",
                "left": _wu_cfg.upgrade_first_dt_utc_start,
                "right": _wu_cfg.analysis_last_dt_utc_start,
            },
            *key_dates_styles,
        ]

    for y_value, trace_metadata in enumerate(key_dates_styles):
        ax_wf.barh(
            y_value,
            left=trace_metadata["left"],
            width=trace_metadata["right"] - trace_metadata["left"],  # type: ignore[operator]
            label=trace_metadata["label"],
            **trace_metadata["trace_fmt"],
        )

    upgrade_date = _wu_cfg.upgrade_first_dt_utc_start
    ax_wf.axvline(
        x=upgrade_date,  # type: ignore[arg-type]
        label="Upgrade Date",
        color="orange",
        linewidth=2,
    )
    ax_wf.set_xlim(x_min, x_max)  # type: ignore[arg-type]

    ax_turbines.set_title("Turbine Level")
    ax_turbines.set_ylabel("Turbine")
    ax_turbines.set_yticks(y_values)
    ax_turbines.set_yticklabels(turbines)
    ax_turbines.set_ylim(0, len(turbines) + 1)

    ax_wf.set_title("Wind Farm Level")
    ax_wf.set_ylim(-1, len(key_dates_styles))
    ax_wf.set_yticks(range(len(key_dates_styles)))
    ax_wf.set_yticklabels([d["label"] for d in key_dates_styles])
    ax_wf.set_xlabel("TimeStamp")

    fig.suptitle("Wind-Up Assessment Timeline", fontsize=14)

    # legends
    for a in [ax_turbines, ax_wf]:
        # Shrink current axis's width by 20%
        box = a.get_position()
        a.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # type: ignore[arg-type]
        handles, labels = a.get_legend_handles_labels()
        a.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.04, 1), borderaxespad=0)

    if save_to_folder is not None:
        if not save_to_folder.is_dir():
            save_to_folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to_folder / "input_data_timeline_fig.png")

    if not show_plots:
        plt.close(fig)

    return fig
