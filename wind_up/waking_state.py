"""Functions for calculating waking state of turbines and waking scenarios."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
from tabulate import tabulate

from wind_up.circular_math import circ_diff
from wind_up.constants import (
    DEFAULT_AIR_DENSITY,
    RAW_DOWNTIME_S_COL,
    RAW_POWER_COL,
    RAW_WINDSPEED_COL,
    TIMESTAMP_COL,
)
from wind_up.plots.waking_state_plots import plot_waking_state_one_ttype_or_wtg
from wind_up.wind_funcs import calc_cp

if TYPE_CHECKING:
    from wind_up.models import PlotConfig, TurbineType, WindUpConfig

logger = logging.getLogger(__name__)


def add_waking_state_one_ttype(
    wf_df: pd.DataFrame, *, ttype: TurbineType, timebase_s: int, plot_cfg: PlotConfig | None
) -> pd.DataFrame:
    """Add waking state columns to `wf_df` for one turbine type.

    :param wf_df: wind farm time series data for one turbine type
    :param ttype: turbine type
    :param timebase_s: time series index frequency in seconds
    :param plot_cfg: plot configuration
    :return: `wf_df` with additional columns "waking", "not_waking", and "unknown_waking"
    """
    wf_df = wf_df.copy()
    rated_power = ttype.rated_power_kw
    wf_df["waking"] = ~wf_df["ActivePowerMean"].isna()
    margin_for_waking_rated_power = 0.8
    wf_df["waking"] = wf_df["waking"] | (wf_df[RAW_POWER_COL] > margin_for_waking_rated_power * rated_power)
    margin_for_cp_calc = 0.3
    df_for_median_cp = wf_df[
        wf_df["waking"]
        & (wf_df[RAW_POWER_COL] > margin_for_cp_calc * rated_power)
        & (wf_df[RAW_POWER_COL] < (1 - margin_for_cp_calc) * rated_power)
    ]
    median_waking_cp = calc_cp(  # type: ignore[union-attr]
        df_for_median_cp["ActivePowerMean"],
        df_for_median_cp["WindSpeedMean"].clip(lower=1),
        DEFAULT_AIR_DENSITY,
        ttype.rotor_diameter_m,
    ).median()
    factor_for_waking_cp = 0.7
    wf_df["waking"] = wf_df["waking"] | (
        calc_cp(
            wf_df[RAW_POWER_COL],
            wf_df[RAW_WINDSPEED_COL].clip(lower=1),
            DEFAULT_AIR_DENSITY,
            ttype.rotor_diameter_m,
        )
        > factor_for_waking_cp * median_waking_cp
    )

    margin_for_not_waking_rated_power = 0.01
    wf_df["not_waking"] = wf_df[RAW_POWER_COL] < margin_for_not_waking_rated_power * rated_power
    wf_df["not_waking"] = wf_df["not_waking"] | (wf_df[RAW_DOWNTIME_S_COL] > timebase_s * factor_for_waking_cp)

    wf_df["not_waking"] = wf_df["not_waking"] & (~wf_df["waking"])
    wf_df["unknown_waking"] = (~wf_df["waking"]) & (~wf_df["not_waking"])

    if plot_cfg is not None:
        waking_frc = wf_df["waking"].sum() / len(wf_df)
        logger.info(f"{ttype.turbine_type} {waking_frc * 100:.1f}% of rows are waking")
        not_waking_frc = wf_df["not_waking"].sum() / len(wf_df)
        logger.info(
            f"{ttype.turbine_type} {not_waking_frc * 100:.1f}% of rows are not waking",
        )
        unknown_waking_frc = wf_df["unknown_waking"].sum() / len(wf_df)
        logger.info(
            f"{ttype.turbine_type} {unknown_waking_frc * 100:.1f}% of rows have unknown or partial waking",
        )
        plot_waking_state_one_ttype_or_wtg(wf_df=wf_df, ttype_or_wtg=ttype.turbine_type, plot_cfg=plot_cfg)
        if not plot_cfg.skip_per_turbine_plots:
            for wtg_name in wf_df.index.unique(level="TurbineName"):
                df_wtg = wf_df.loc[wtg_name]
                plot_waking_state_one_ttype_or_wtg(wf_df=df_wtg, ttype_or_wtg=wtg_name, plot_cfg=plot_cfg)
    return wf_df


def add_waking_state(cfg: WindUpConfig, wf_df: pd.DataFrame, plot_cfg: PlotConfig | None) -> pd.DataFrame:
    """Add waking state columns to `wf_df`.

    :param cfg: wind up configuration
    :param wf_df: wind farm time series data
    :param plot_cfg: plot configuration
    :return: `wf_df` with additional columns "waking", "not_waking", and "unknown_waking"
    """
    df_input = wf_df.copy()
    wf_df = pd.DataFrame()
    for ttype in cfg.list_unique_turbine_types():
        wtgs = cfg.list_turbine_ids_of_type(ttype)
        df_ttype = df_input.loc[wtgs]
        wf_df_ = add_waking_state_one_ttype(
            wf_df=df_ttype,
            ttype=ttype,
            timebase_s=cfg.timebase_s,
            plot_cfg=plot_cfg,
        )
        wf_df = pd.concat([wf_df, wf_df_])
    return wf_df.sort_index()


def calc_bearing(*, lat1: float, long1: float, lat2: float, long2: float) -> float:
    """Calculate bearing between two points.

    :param lat1: latitude of point 1
    :param long1: longitude of point 1
    :param lat2: latitude of point 2
    :param long2: longitude of point 2
    :return: bearing in degrees
    """
    bearing_deg = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)["azi1"]
    return bearing_deg % 360


def calc_distance(*, lat1: float, long1: float, lat2: float, long2: float) -> float:
    """Calculate distance between two points.

    :param lat1: latitude of point 1
    :param long1: longitude of point 1
    :param lat2: latitude of point 2
    :param long2: longitude of point 2
    :return: distance in meters
    """
    return Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)["s12"]


distance_and_bearing_cache: dict[tuple[float, float, float, float], tuple[float, float]] = {}


def get_distance_and_bearing(*, lat1: float, long1: float, lat2: float, long2: float) -> tuple[float, float]:
    """Get distance and bearing between two points.

    :param lat1: latitude of point 1
    :param long1: longitude of point 1
    :param lat2: latitude of point 2
    :param long2: longitude of point 2
    :return: distance in meters and bearing in degrees
    """
    if (lat1, long1, lat2, long2) in distance_and_bearing_cache:
        distance_m, bearing_deg = distance_and_bearing_cache[(lat1, long1, lat2, long2)]
    else:
        distance_m = calc_distance(lat1=lat1, long1=long1, lat2=lat2, long2=long2)
        bearing_deg = calc_bearing(lat1=lat1, long1=long1, lat2=lat2, long2=long2)
        distance_and_bearing_cache[(lat1, long1, lat2, long2)] = (distance_m, bearing_deg)
    return distance_m, bearing_deg


def calc_iec_upwind_turbines(*, lat: float, long: float, wind_direction: float, cfg: WindUpConfig) -> list[str]:
    """Find all turbines in cfg which are upwind of location (`lat`, `long`).

    :param lat: latitude
    :param long: longitude
    :param wind_direction: wind direction in degrees
    :param cfg: wind up configuration
    :return: list of upwind turbines
    """
    wind_direction = wind_direction % 360
    wtg_names = []
    bearings = []
    distances = []
    wtg_d_norms = []
    for wtg in cfg.asset.wtgs:
        wtg_lat = wtg.latitude
        wtg_long = wtg.longitude
        wtg_names.append(wtg.name)
        wtg_distance, wtg_bearing = get_distance_and_bearing(lat1=lat, long1=long, lat2=wtg_lat, long2=wtg_long)
        bearings.append(wtg_bearing)
        distances.append(wtg_distance)
        wtg_d_norms.append(wtg_distance / wtg.turbine_type.rotor_diameter_m)
    upwind_df = pd.DataFrame(
        data={
            "wtg_name": wtg_names,
            "bearing": bearings,
            "distance_m": distances,
            "distance_diameters": wtg_d_norms,
        },
    )
    upwind_df["disturbed_sector"] = 180.0
    ge_2_diameters = upwind_df["distance_diameters"] >= 2  # noqa PLR2004
    upwind_df.loc[ge_2_diameters, "disturbed_sector"] = (
        1.3 * np.rad2deg(np.arctan(2.5 / upwind_df.loc[ge_2_diameters, "distance_diameters"] + 0.15)) + 10
    )
    gt_20_diameters = upwind_df["distance_diameters"] > 20  # noqa PLR2004
    upwind_df.loc[gt_20_diameters, "disturbed_sector"] = 0

    upwind_df["relative_bearing"] = circ_diff(upwind_df["bearing"], wind_direction)
    upwind_df["iec_upwind"] = abs(upwind_df["relative_bearing"]) < upwind_df["disturbed_sector"] / 2

    upwind_turbine_list = list(upwind_df.query("iec_upwind")["wtg_name"].sort_values().values)
    if wind_direction % 90 == 0:
        logger.info(
            f"calc_iec_upwind_turbines lat={lat:.2f} long={long:.2f} "
            f"wind_dir={wind_direction:.0f} {upwind_turbine_list}",
        )
    return upwind_turbine_list


upwind_wtgs_cache: dict[tuple[float, float, float, str | None], list[str]] = {}


def lat_long_is_valid(lat: float, long: float) -> bool:
    """Validate latitude and longitude.

    :return: True if latitude and longitude are valid, False otherwise
    """
    return (-90 <= lat <= 90) and (-180 <= long <= 180)  # noqa PLR2004


def get_iec_upwind_turbines_one_latlong(
    *,
    lat: float,
    long: float,
    wind_direction: float,
    cfg: WindUpConfig,
    object_name: str | None = None,
) -> list[str]:
    """Get a list of upwind turbines for one latitude and longitude.

    :param lat: latitude
    :param long: longitude
    :param wind_direction: wind direction in degrees
    :param cfg: wind up configuration
    :param object_name: name of the turbine that is being evaluated
    :return: list of upwind turbines
    """
    if not lat_long_is_valid(lat, long):
        msg = f"lat={lat} long={long} is not a valid lat long"
        raise ValueError(msg)

    if (lat, long, wind_direction, object_name) in upwind_wtgs_cache:
        upwind_wtgs = upwind_wtgs_cache[(lat, long, wind_direction, object_name)]
    else:
        upwind_wtgs = calc_iec_upwind_turbines(lat=lat, long=long, wind_direction=wind_direction, cfg=cfg)
        if object_name is not None:
            upwind_wtgs = [x for x in upwind_wtgs if x.lower() != object_name.lower()]
        upwind_wtgs_cache[(lat, long, wind_direction, object_name)] = upwind_wtgs
    return upwind_wtgs


def get_iec_upwind_turbines(
    *,
    latlongs: list[tuple[float, float]],
    wind_direction: float,
    cfg: WindUpConfig,
    object_name: str | None = None,
) -> list[str]:
    """Get a list of upwind turbines for a list of latitudes and longitudes.

    :param latlongs: list of latitudes and longitudes, e.g. [(lat1, long1), (lat2, long2)]
    :param wind_direction: wind direction in degrees
    :param cfg: wind up configuration
    :param object_name: name of the turbine that is being evaluated
    :return: list of upwind turbines
    """
    upwind_wtgs = []
    for lat, long in latlongs:
        upwind_wtgs += get_iec_upwind_turbines_one_latlong(
            lat=lat,
            long=long,
            wind_direction=wind_direction,
            cfg=cfg,
            object_name=object_name,
        )
    return sorted(set(upwind_wtgs))


def calc_scen_name_from_wtgs_not_waking(wtgs_not_waking: list[str]) -> str:
    """Calculate a waking scenario name from a list of turbines that are not waking.

    :param wtgs_not_waking: turbines that are not waking
    :return: waking scenario name
    """
    if len(wtgs_not_waking) < 1:
        msg = "wtgs_not_waking must have at least one element"
        raise ValueError(msg)
    return " ".join(wtgs_not_waking) + " offline"


def list_wtgs_offline_in_scen(waking_scenario: str) -> list[str]:
    """List turbines that are offline in the waking scenario.

    :param waking_scenario: waking scenario
    :return: list of turbines that are offline in the waking scenario
    """
    return [x for x in waking_scenario.split(" ") if x != "offline"]


def add_waking_scen(
    *,
    test_name: str,
    ref_name: str,
    test_ref_df: pd.DataFrame,
    cfg: WindUpConfig,
    wf_df: pd.DataFrame,
    ref_wd_col: str,
    ref_lat: float,
    ref_long: float,
) -> pd.DataFrame:
    """Calculate a waking scenario for each row in `test_ref_df` based on the wind direction of the reference.

    :param test_name: TurbineName of the test turbine
    :param ref_name: Name of the reference
    :param test_ref_df: time series data of test turbine and reference
    :param cfg: wind up configuration
    :param wf_df: wind farm time series data
    :param ref_wd_col: column name for wind direction of the reference in `test_ref_df`
    :param ref_lat: latitude of the reference
    :param ref_long: longitude of the reference
    :return: `test_ref_df` with a new column "waking_scenario" containing the calculated waking scenario
    """
    test_ref_df = test_ref_df.copy()
    test_ref_df["waking_scenario"] = "not calculated"
    try:
        test_wtg = next(x for x in cfg.asset.wtgs if x.name == test_name)
    except StopIteration as exc:
        msg = f"{test_name} not found in cfg.asset.wtgs"
        raise ValueError(msg) from exc

    all_turbines_waking = wf_df.groupby(TIMESTAMP_COL)["waking"].all().to_frame(name="all_turbines_waking")
    test_ref_df = test_ref_df.merge(all_turbines_waking, how="left", left_index=True, right_index=True)
    test_ref_df.loc[test_ref_df["all_turbines_waking"], "waking_scenario"] = "none offline"

    test_ref_df["rounded_wd"] = test_ref_df[ref_wd_col].round(0).mod(360)
    waking_df = wf_df[["waking", "not_waking", "unknown_waking"]].pivot_table(
        index=TIMESTAMP_COL,
        columns="TurbineName",
        aggfunc="max",
        observed=False,
    )

    for wd in sorted(test_ref_df["rounded_wd"].unique()):
        if wd >= 0:
            test_upwind_wtgs = get_iec_upwind_turbines(
                latlongs=test_wtg.get_latlongs(),
                wind_direction=wd,
                cfg=cfg,
                object_name=test_name,
            )
            if lat_long_is_valid(ref_lat, ref_long):
                ref_upwind_wtgs = get_iec_upwind_turbines(
                    latlongs=[(ref_lat, ref_long)],
                    wind_direction=wd,
                    cfg=cfg,
                    object_name=ref_name,
                )
            else:
                ref_upwind_wtgs = []
            upwind_wtgs = sorted(set(test_upwind_wtgs + ref_upwind_wtgs))
            if len(upwind_wtgs) == 0:
                test_ref_df.loc[test_ref_df["rounded_wd"] == wd, "waking_scenario"] = "none offline"
            else:
                for idx in test_ref_df[
                    (test_ref_df["rounded_wd"] == wd) & (test_ref_df["waking_scenario"] == "not calculated")
                ].index:
                    if waking_df.loc[idx, "unknown_waking"].loc[upwind_wtgs].any():
                        test_ref_df.loc[idx, "waking_scenario"] = "unknown"
                    elif waking_df.loc[idx, "waking"].loc[upwind_wtgs].all():
                        test_ref_df.loc[idx, "waking_scenario"] = "none offline"
                    else:
                        not_waking_series = waking_df.loc[idx, "not_waking"].loc[upwind_wtgs]
                        test_ref_df.loc[idx, "waking_scenario"] = calc_scen_name_from_wtgs_not_waking(
                            list(not_waking_series[not_waking_series].index),
                        )

    if (test_ref_df.dropna(subset=ref_wd_col)["waking_scenario"] == "not calculated").any():
        msg = "some rows with defined ref_wd_col have waking scenario = 'not calculated'"
        raise RuntimeError(msg)

    top_scens = test_ref_df.dropna(subset=ref_wd_col)["waking_scenario"].value_counts()[0:5]
    logger.info(f"top {len(top_scens)} {test_name} {ref_name} waking scenarios [%]:")
    _table = tabulate(
        (top_scens / len(test_ref_df.dropna(subset=ref_wd_col)) * 100).to_frame(),
        tablefmt="outline",
        floatfmt=".1f",
    )
    logger.info(f"{_table}")

    return test_ref_df.drop(columns=["rounded_wd", "all_turbines_waking"])
