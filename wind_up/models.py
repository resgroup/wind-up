import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

from wind_up.constants import OUTPUT_DIR
from wind_up.yaml_loader import Loader, construct_include

logger = logging.getLogger(__name__)


class PlotConfig(BaseModel):
    show_plots: bool = Field(default=False, description="Show plots in interactive window")
    save_plots: bool = Field(default=True, description="Save plots to file")
    skip_per_turbine_plots: bool = Field(default=False, description="If True skip per turbine plots")
    plots_dir: Path = Field(description="Directory to save plots to")

    @model_validator(mode="after")
    def make_plots_dir(self: "PlotConfig") -> "PlotConfig":
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        return self


class TurbineType(BaseModel):
    turbine_type: str = Field(
        description="Turbine type description",
        min_length=2,
    )
    rotor_diameter_m: float = Field(description="Rotor diameter in meters", gt=0, examples=[100])
    rated_power_kw: float = Field(description="Rated power in kW", gt=0, examples=[3000])
    cutout_ws_mps: float = Field(default=25, description="Cutout wind speed in m/s", gt=0, examples=[25])
    normal_operation_pitch_range: tuple[float, float] = Field(
        description="Normal operation pitch range in degrees",
        examples=[[-10, 40]],
    )
    normal_operation_genrpm_range: tuple[float, float] = Field(
        description="Normal operation generator rpm range",
        examples=[[800, 1600]],
    )
    rpm_v_pw_margin_factor: float = Field(
        default=0.05,
        description="Factor to multiply max RPM by to get rpm margin for rpm vs power filtering",
        examples=[0.05, 0.1],
    )
    pitch_to_stall: bool = Field(
        default=False,
        description="If False then pitch filter will eliminate data below the threshold rather than above",
    )


class Turbine(BaseModel):
    name: str = Field(description="Turbine name", min_length=2)
    turbine_type: TurbineType
    latitude: float = Field(default=np.nan, ge=-90, le=90)
    longitude: float = Field(default=np.nan, ge=-180, le=180)

    def get_latlongs(self: "Turbine") -> list[tuple[float, float]]:
        return [(self.latitude, self.longitude)]


class MastOrLidar(BaseModel):
    name: str = Field(description="Object name", min_length=2)
    latitude: float = Field(default=np.nan, ge=-90, le=90)
    longitude: float = Field(default=np.nan, ge=-180, le=180)
    data_file_name: str = Field(
        min_length=2,
        description="Name of data timeseries file",
    )
    wind_speed_column: str = Field(min_length=2, description="Name of wind speed column in data timeseries file")
    wind_direction_column: str = Field(
        min_length=2,
        description="Name of wind direction column in data timeseries file",
    )


class Asset(BaseModel):
    name: str = Field(description="Asset Name", min_length=2, examples=["Kelmarsh"])
    wtgs: list[Turbine]
    masts_and_lidars: list[MastOrLidar] = Field(description="list of mast and LiDAR objects, if any", default=[])


class Toggle(BaseModel):
    name: str = Field(description="Name of toggle signal to use in plots", min_length=2, default="toggle")
    toggle_file_per_turbine: bool = Field(
        description="Is there one toggle timeseries file per turbine, or one for the asset?",
    )
    toggle_filename: str = Field(
        min_length=2,
        description="Name of toggle timeseries file, can use {wtg} if toggle_file_per_turbine is True",
    )
    detrend_data_selection: str = Field(
        description="Method for selecting directional detrending data",
        default="do_not_use_toggle_test",
        examples=["do_not_use_toggle_test", "use_toggle_off_data"],
    )
    pairing_filter_method: str = Field(
        description="Method to use for pairing filter",
        examples=["none", "one_to_one", "any_within_timedelta"],
        default="none",
    )
    pairing_filter_timedelta_seconds: int = Field(
        description="Time delta in seconds to use for pairing filter",
        default=50 * 60,
    )
    toggle_change_settling_filter_seconds: int = Field(
        description="Time delta in seconds filter out after a toggle state change",
        default=10 * 60,
    )


class PrePost(BaseModel):
    pre_first_dt_utc_start: dt.datetime = Field(
        description="First time to use in pre-upgrade analysis, UTC Start format",
    )
    pre_last_dt_utc_start: dt.datetime = Field(
        description="Last time to use in pre-upgrade analysis, UTC Start format",
    )
    post_first_dt_utc_start: dt.datetime = Field(
        description="First time to use in post-upgrade analysis, UTC Start format",
    )
    post_last_dt_utc_start: dt.datetime = Field(
        description="Last time to use in post-upgrade analysis, UTC Start format",
    )


class WindUpConfig(BaseModel):
    assessment_name: str = Field(
        min_length=2,
        description="Name used for assessment output folder",
    )
    timebase_s: int = Field(
        default=10 * 60,
        description="Timebase in seconds for SCADA data, other data is converted to this timebase",
    )
    ignore_turbine_anemometer_data: bool = Field(
        default=False,
        description="If true do not use turbine anemometer data for anything",
    )
    require_test_wake_free: bool = Field(
        default=False,
        description="Remove data where the test turbine has any upwind turbines",
    )
    require_ref_wake_free: bool = Field(
        default=False,
        description="Remove data where the reference has any upwind turbines",
    )
    detrend_min_hours: int = Field(
        default=24,
        description="Minimum number of hours to use in directional detrending",
    )
    ref_wd_filter: list[float] | None = Field(
        default=None,
        description="Wind direction filter for reference data; only data within this range is used",
        examples=[[195, 241]],
    )
    ref_hod_filter: list[float] | None = Field(
        default=None,
        description="Hour of day filter for reference data; only data within this range is used",
        examples=[[18, 6]],
    )
    filter_all_test_wtgs_together: bool = Field(
        description="If True any row filtered for one test wtg is filtered for all test wtgs. "
        "Recommended for wake steering analysis.",
        default=False,
    )
    use_lt_distribution: bool = Field(
        description="If True the long term distribution is calculated and used for uplift calculation", default=True
    )
    use_test_wtg_lt_distribution: bool = Field(
        description="If True the test turbine's data is used for the long term distribution, "
        "otherwise the whole wind farm is used",
        default=True,
    )
    out_dir: Path = Field(description="Directory to save assessment output to")
    test_wtgs: list[Turbine] = Field(description="List of test Turbine ids")
    ref_wtgs: list[Turbine] = Field(default=[], description="List of reference Turbine ids")
    ref_super_wtgs: list[str] = Field(
        default=[],
        description="List of reference super Turbine ids",
    )
    non_wtg_ref_names: list[str] = Field(
        default=[],
        description="List of non turbine references",
    )
    upgrade_first_dt_utc_start: dt.datetime = Field(
        description="First time when test turbine is upgraded, UTC Start format",
    )
    analysis_last_dt_utc_start: dt.datetime = Field(
        description="Last time to use in upgrade analysis, UTC Start format",
    )
    analysis_first_dt_utc_start: dt.datetime = Field(
        description="First time to use in upgrade analysis, UTC Start format",
    )
    lt_first_dt_utc_start: dt.datetime = Field(
        description="First time to use in long term analysis, UTC Start format",
    )
    lt_last_dt_utc_start: dt.datetime = Field(
        description="Last time to use in long term analysis, UTC Start format",
    )
    detrend_first_dt_utc_start: dt.datetime = Field(
        description="First time to use in detrend analysis, UTC Start format",
    )
    detrend_last_dt_utc_start: dt.datetime = Field(
        description="Last time to use in detrend analysis, UTC Start format",
    )
    years_offset_for_pre_period: int | None = Field(
        description="How many years to go back in time to begin pre upgrade dataset",
        ge=0,
        examples=[None, 1],
        default=None,
    )
    years_for_lt_distribution: int = Field(
        description="How many years to use in long term analysis",
        ge=0,
        examples=[3, 5],
    )
    years_for_detrend: int = Field(
        description="How many years to use in directional detrending",
        ge=0,
        examples=[1, 2],
    )
    ws_bin_width: float = Field(description="Wind speed bin width in m/s", gt=0, examples=[0.5, 1])
    bootstrap_runs_override: int | None = Field(
        description="Number of bootstrap runs to use, if None then calculated",
        ge=0,
        examples=[None, 1000],
        default=None,
    )
    reanalysis_method: str = Field(
        default="node_with_best_ws_corr",
        description="Method to use for reanalysis (e.g. ERA5) selection",
        examples=["node_with_best_ws_corr"],
    )
    missing_scada_data_fields: list[str] = Field(
        default=[],
        description="List of SCADA fields not available for this asset",
        examples=[["YawAngleMin", "YawAngleMax"]],
    )
    asset: Asset
    exclusion_periods_utc: list[tuple[str, dt.datetime, dt.datetime]] = Field(
        default=[],
        description="Turbine id or 'ALL', first UTC timestamp to exclude, last UTC timestamp to exclude",
        examples=[
            "ALL",
            dt.datetime(2021, 9, 21, 12, 40, tzinfo=dt.timezone.utc),
            dt.datetime(2021, 9, 29, 18, 10, tzinfo=dt.timezone.utc),
        ],
    )
    yaw_data_exclusions_utc: list[tuple[str, dt.datetime, dt.datetime]] = Field(
        default=[],
        description="Turbine id or 'ALL', first UTC start format timestamp to exclude yaw data,"
        " last UTC timestamp to exclude yaw data",
        examples=[
            "ALL",
            dt.datetime(2021, 9, 21, 12, 40, tzinfo=dt.timezone.utc),
            dt.datetime(2021, 9, 29, 18, 10, tzinfo=dt.timezone.utc),
        ],
    )
    optimize_northing_corrections: bool = Field(
        default=False,
        description="run northing correction optimization",
    )
    northing_corrections_utc: list[tuple[str, dt.datetime, float]] = Field(
        default=[],
        description="Turbine id, first UTC timestamp to correct, correction to add to yaw data",
        examples=["SMV1", dt.datetime(2021, 9, 21, 12, 40, tzinfo=dt.timezone.utc), 15],
    )
    toggle: Toggle | None = None
    prepost: PrePost | None = None

    @model_validator(mode="after")
    def check_years_offset_for_pre_period(self: "WindUpConfig") -> "WindUpConfig":
        if self.toggle is None and self.years_offset_for_pre_period is None:
            msg = "toggle is None and years_offset_for_pre_period is None"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_first_datetime_before_last(self: "WindUpConfig") -> "WindUpConfig":
        if self.upgrade_first_dt_utc_start >= self.analysis_last_dt_utc_start:
            msg = "upgrade_first_datetime must be before last_useable_datetime"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_non_wtg_ref_names(self: "WindUpConfig") -> "WindUpConfig":
        for non_wtg_ref in self.non_wtg_ref_names:
            if non_wtg_ref == "reanalysis":
                if len(self.reanalysis_method) < 1:
                    msg = "reanalysis is in non_wtg_ref_names but reanalysis_method is not valid"
                    raise ValueError(msg)
            elif len([x for x in self.asset.masts_and_lidars if x.name == non_wtg_ref]) != 1:
                msg = f"there is not exactly 1 non_wtg_ref {non_wtg_ref} in cfg.asset.masts_and_lidars"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def print_summary(self: "WindUpConfig") -> "WindUpConfig":
        logger.info(f"loaded WindUpConfig assessment_name: {self.assessment_name}")
        dt_fmt = "%Y-%m-%d %H:%M"
        if self.toggle is not None:
            dt_rng_start = self.analysis_first_dt_utc_start.strftime(dt_fmt)
            dt_rng_end = (self.analysis_last_dt_utc_start + pd.Timedelta(seconds=self.timebase_s)).strftime(dt_fmt)
            logger.info(
                f"toggle analysis period (UTC): {dt_rng_start} to {dt_rng_end}",
            )
        elif self.prepost is not None:
            dt_rng_start = self.prepost.pre_first_dt_utc_start.strftime(dt_fmt)
            dt_rng_end = (self.prepost.pre_last_dt_utc_start + pd.Timedelta(seconds=self.timebase_s)).strftime(dt_fmt)
            logger.info(
                f"pre analysis period (UTC): {dt_rng_start} to {dt_rng_end}",
            )
            dt_rng_start = self.prepost.post_first_dt_utc_start.strftime(dt_fmt)
            dt_rng_end = (self.prepost.post_last_dt_utc_start + pd.Timedelta(seconds=self.timebase_s)).strftime(dt_fmt)
            logger.info(
                f"post analysis period (UTC): {dt_rng_start} to {dt_rng_end}",
            )
        else:
            msg = "toggle and prepost are both set to None"
            raise RuntimeError(msg)
        dt_rng_start = self.lt_first_dt_utc_start.strftime(dt_fmt)
        dt_rng_end = (self.lt_last_dt_utc_start + pd.Timedelta(seconds=self.timebase_s)).strftime(dt_fmt)
        logger.info(f"long term period (UTC): {dt_rng_start} to {dt_rng_end}")
        dt_rng_start = self.detrend_first_dt_utc_start.strftime(dt_fmt)
        dt_rng_end = (self.detrend_last_dt_utc_start + pd.Timedelta(seconds=self.timebase_s)).strftime(dt_fmt)
        logger.info(f"detrend period (UTC): {dt_rng_start} to {dt_rng_end}")
        return self

    @classmethod
    def from_yaml(cls, file_path: Path) -> "WindUpConfig":  # noqa ANN102
        yaml.add_constructor("!include", construct_include, Loader)
        with Path.open(file_path) as f:
            cfg_dct = yaml.load(f, Loader)  # noqa S506

        cfg_dct["out_dir"] = OUTPUT_DIR / cfg_dct["assessment_name"]

        # ensure datetimes are UTC aware
        cfg_dct["upgrade_first_dt_utc_start"] = pd.Timestamp(cfg_dct["upgrade_first_dt_utc_start"], tz="UTC")
        cfg_dct["analysis_last_dt_utc_start"] = pd.Timestamp(cfg_dct["analysis_last_dt_utc_start"], tz="UTC")
        if cfg_dct.get("exclusion_periods_utc", None) is not None:
            for x in cfg_dct["exclusion_periods_utc"]:
                x[1] = pd.Timestamp(x[1], tz="UTC")
                x[2] = pd.Timestamp(x[2], tz="UTC")
        if cfg_dct.get("yaw_data_exclusions_utc", None) is not None:
            for x in cfg_dct["yaw_data_exclusions_utc"]:
                x[1] = pd.Timestamp(x[1], tz="UTC")
                x[2] = pd.Timestamp(x[2], tz="UTC")
        if cfg_dct.get("northing_corrections_utc", None) is not None:
            for x in cfg_dct["northing_corrections_utc"]:
                x[1] = pd.Timestamp(x[1], tz="UTC")

        # calculate analysis and long term datetimes
        test_is_toggle = cfg_dct.get("toggle", None) is not None
        if test_is_toggle:
            cfg_dct["analysis_first_dt_utc_start"] = cfg_dct["upgrade_first_dt_utc_start"]
        else:
            pre_post_dict = {
                "post_first_dt_utc_start": cfg_dct["upgrade_first_dt_utc_start"],
                "post_last_dt_utc_start": cfg_dct["analysis_last_dt_utc_start"],
                "pre_first_dt_utc_start": cfg_dct["upgrade_first_dt_utc_start"]
                - dt.timedelta(days=(365.25 * cfg_dct["years_offset_for_pre_period"])),
                "pre_last_dt_utc_start": cfg_dct["analysis_last_dt_utc_start"]
                - dt.timedelta(days=(365.25 * cfg_dct["years_offset_for_pre_period"])),
            }
            cfg_dct["prepost"] = PrePost.model_validate(pre_post_dict)
            cfg_dct["analysis_first_dt_utc_start"] = cfg_dct["prepost"].pre_first_dt_utc_start
        cfg_dct["lt_last_dt_utc_start"] = (
            cfg_dct["upgrade_first_dt_utc_start"] - dt.timedelta(days=7)  # go back 1 week for buffer before toggling
            if test_is_toggle
            else cfg_dct["prepost"].pre_last_dt_utc_start
        )
        cfg_dct["lt_first_dt_utc_start"] = cfg_dct["lt_last_dt_utc_start"] - dt.timedelta(
            days=(cfg_dct["years_for_lt_distribution"] * 365.25),
        )
        cfg_dct["detrend_last_dt_utc_start"] = cfg_dct["lt_last_dt_utc_start"]
        if test_is_toggle and cfg_dct["toggle"].get("detrend_data_selection", None) == "use_toggle_off_data":
            cfg_dct["detrend_last_dt_utc_start"] = cfg_dct["analysis_last_dt_utc_start"]
        cfg_dct["detrend_first_dt_utc_start"] = cfg_dct["detrend_last_dt_utc_start"] - dt.timedelta(
            days=(cfg_dct["years_for_detrend"] * 365.25),
        )

        # check each test_wtg exists in asset.wtgs
        for x in cfg_dct["test_wtgs"]:
            if x not in cfg_dct["asset"]["wtgs"]:
                msg = f"test_wtg {x} does not exist in asset.wtgs"
                raise ValueError(msg)

        # if ref_wtgs exist, check each ref_wtg exists in asset.wtgs
        if cfg_dct.get("ref_wtgs", None) is not None:
            for x in cfg_dct["ref_wtgs"]:
                if x not in cfg_dct["asset"]["wtgs"]:
                    msg = f"ref_wtg {x} does not exist in asset.wtgs"
                    raise ValueError(msg)

        # change lists of strings into list of turbines
        if len(cfg_dct["asset"]["turbine_types"]) == 1:
            cfg_dct["asset"]["turbine_types"] = cfg_dct["asset"]["turbine_types"] * len(
                cfg_dct["asset"]["wtgs"],
            )
        tt_list = [TurbineType.model_validate(tt) for tt in cfg_dct["asset"]["turbine_types"]]
        cfg_dct["asset"]["wtgs"] = [
            Turbine(name=x, turbine_type=tt) for x, tt in zip(cfg_dct["asset"]["wtgs"], tt_list, strict=True)
        ]
        test_wtg_list = []
        for x in cfg_dct["test_wtgs"]:
            for wtg in cfg_dct["asset"]["wtgs"]:
                if x == wtg.name:
                    test_wtg_list.append(wtg)
                    continue
        cfg_dct["test_wtgs"] = test_wtg_list
        if cfg_dct.get("ref_wtgs", None) is not None:
            ref_wtg_list = []
            for x in cfg_dct["ref_wtgs"]:
                for wtg in cfg_dct["asset"]["wtgs"]:
                    if x == wtg.name:
                        ref_wtg_list.append(wtg)
                        continue
            cfg_dct["ref_wtgs"] = ref_wtg_list
        elif cfg_dct.get("non_wtg_ref_names", None) is not None:
            if len(cfg_dct["non_wtg_ref_names"]) < 1:
                msg = "ref_wtgs or non_wtg_ref_names must have more than one element"
                raise ValueError(msg)
        else:
            msg = "ref_wtgs or non_wtg_ref_names must be specified"
            raise ValueError(msg)
        return WindUpConfig.model_validate(cfg_dct)

    def get_max_rated_power(self: "WindUpConfig") -> float:
        return max(x.turbine_type.rated_power_kw for x in self.asset.wtgs)

    def list_unique_turbine_types(self: "WindUpConfig") -> list["TurbineType"]:
        unique_names = sorted({x.turbine_type.turbine_type for x in self.asset.wtgs})
        unique_turbine_types = []
        for name in unique_names:
            unique_turbine_types.append(
                next(x.turbine_type for x in self.asset.wtgs if x.turbine_type.turbine_type == name),
            )
        return unique_turbine_types

    def list_turbine_ids_of_type(self: "WindUpConfig", ttype: "TurbineType") -> list[str]:
        return [x.name for x in self.asset.wtgs if x.turbine_type == ttype]

    def get_normal_operation_genrpm_range(self: "WindUpConfig", ttype: "TurbineType") -> tuple[float, float]:
        return next(x.turbine_type.normal_operation_genrpm_range for x in self.asset.wtgs if x.turbine_type == ttype)

    def get_normal_operation_pitch_range(self: "WindUpConfig", ttype: "TurbineType") -> tuple[float, float]:
        return next(x.turbine_type.normal_operation_pitch_range for x in self.asset.wtgs if x.turbine_type == ttype)
