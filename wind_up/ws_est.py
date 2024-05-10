import logging

import numpy as np
import pandas as pd

from wind_up.constants import DEFAULT_AIR_DENSITY
from wind_up.models import PlotConfig, TurbineType, WindUpConfig
from wind_up.plots.ws_est_plots import plot_ws_est_gain_xs_one_ttype, plot_ws_est_one_ttype_or_wtg
from wind_up.wind_funcs import calc_cp

logger = logging.getLogger(__name__)


def calc_pc_low_high_one_ttype(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_bin_width: float,
    low_q_pct: float,
    high_q_pct: float,
) -> pd.DataFrame:
    x_bin_edges = np.arange(0, df[x_col].max() + x_bin_width, x_bin_width)
    return df.groupby(by=pd.cut(df[x_col], bins=x_bin_edges, retbins=False), observed=True).agg(
        x_mean=pd.NamedAgg(column=x_col, aggfunc=lambda x: x.mean()),
        y_low=pd.NamedAgg(column=y_col, aggfunc=lambda x: np.nanpercentile(x, low_q_pct)),
        y_high=pd.NamedAgg(column=y_col, aggfunc=lambda x: np.nanpercentile(x, high_q_pct)),
    )


def add_ws_est_one_ttype(
    cfg: WindUpConfig,
    df: pd.DataFrame,
    ttype: TurbineType,
    pc: pd.DataFrame,
    plot_cfg: PlotConfig | None,
) -> pd.DataFrame:
    df_input = df.copy()

    # at some point would be good to use an air density time series
    df_input["cp"] = calc_cp(
        power_kw=df_input["pw_clipped"],
        ws_ms=df_input["WindSpeedMean"],
        air_density_kgpm3=DEFAULT_AIR_DENSITY,
        rotor_diameter_m=ttype.rotor_diameter_m,
    )
    df_input["cp"] = df_input["cp"].clip(upper=2)
    ws_half_rated = np.interp(ttype.rated_power_kw / 2, pc["pw_clipped"].values, pc["WindSpeedMean"].values)
    cp_calc_ws_range = 1
    target_cp = df_input["cp"][(df_input["WindSpeedMean"] - ws_half_rated).abs() < cp_calc_ws_range / 2].mean()

    for wtg in df_input.index.unique(level="TurbineName"):
        df_wtg = df_input.loc[wtg]
        mean_cp_wtg = df_wtg.loc[(df_wtg["WindSpeedMean"] - ws_half_rated).abs() < cp_calc_ws_range / 2, "cp"].mean()
        cp_correction_factor = mean_cp_wtg ** (1 / 3) / target_cp ** (1 / 3)
        logger.info(f"{wtg} cp correction factor = {cp_correction_factor:.2f}")
        df_input.loc[wtg, "ws_cp_corrected"] = cp_correction_factor * df_input.loc[wtg, "WindSpeedMean"].to_numpy()

    # find the four wind speeds used for gain
    low_q_pct = 0.01
    pc_low_high = calc_pc_low_high_one_ttype(
        df=df_input,
        x_col="ws_cp_corrected",
        y_col="pw_clipped",
        x_bin_width=cfg.ws_bin_width / 2,
        low_q_pct=low_q_pct,
        high_q_pct=100 - low_q_pct,
    )
    if pc_low_high["y_high"].min() < (ttype.rated_power_kw * 0.01):
        ws0 = float(np.interp(ttype.rated_power_kw * 0.01, pc_low_high["y_high"], pc_low_high["x_mean"]))
    else:
        ws0 = 0
    ws1 = float(np.interp(ttype.rated_power_kw * 0.01, pc_low_high["y_low"], pc_low_high["x_mean"]))
    ws1 = max(ws0 + 1, ws1)
    high_power_ws = min(ttype.cutout_ws_mps - 3, 17)
    high_power_threshold = pc_low_high["y_low"][pc_low_high["x_mean"] >= high_power_ws].min() * 0.99
    ws2 = float(np.interp(high_power_threshold, pc_low_high["y_high"], pc_low_high["x_mean"]))
    ws3 = float(np.interp(high_power_threshold, pc_low_high["y_low"], pc_low_high["x_mean"]))
    ws3 = max(ws2 + 1, ws3)
    if plot_cfg is not None:
        plot_ws_est_gain_xs_one_ttype(
            pc_low_high=pc_low_high,
            ttype=ttype.turbine_type,
            rated_power_kw=ttype.rated_power_kw,
            x0=ws0,
            x1=ws1,
            x2=ws2,
            x3=ws3,
            plot_cfg=plot_cfg,
        )
    # gain 1 uses wind speed as x axis
    ws_est_gain1_x = [ws0, ws1, ws2, ws3]
    ws_est_gain1_y = [0, 1, 1, -1]
    if not np.all(np.diff(ws_est_gain1_x) > 0):
        msg = "x values for gain 1 must be increasing"
        raise RuntimeError(msg)
    ws_est_gain1 = np.interp(df_input["ws_cp_corrected"].values, ws_est_gain1_x, ws_est_gain1_y)

    # gain 2 uses power as x axis
    ws_est_gain2_x = [0, 0.1 * ttype.rated_power_kw, 0.9 * ttype.rated_power_kw, ttype.rated_power_kw]
    ws_est_gain2_y = [0, 1, 1, 0]
    if not np.all(np.diff(ws_est_gain2_x) > 0):
        msg = "x values for gain 2 must be increasing"
        raise RuntimeError(msg)
    ws_est_gain2 = np.interp(df_input["pw_clipped"].values, ws_est_gain2_x, ws_est_gain2_y)

    # combine the two gains as a simple average
    df_input["ws_est_gain"] = (ws_est_gain1 + ws_est_gain2) / 2
    df_input["ws_est_gain"] = df_input["ws_est_gain"].clip(lower=0, upper=1)

    pc_transposed = df_input.groupby(
        by=pd.qcut(df_input["pw_clipped"], q=50, retbins=False, duplicates="drop"),
        observed=True,
    ).agg(
        p_bin=pd.NamedAgg(column="pw_clipped", aggfunc=lambda x: x.mean()),
        ws_cp_corrected=pd.NamedAgg(column="ws_cp_corrected", aggfunc=lambda x: x.mean()),
    )
    pc_transposed = pc_transposed.set_index("p_bin")

    df_input["ws_est_from_power_only"] = np.interp(
        df_input["pw_clipped"].values,
        pc_transposed.index.values,
        pc_transposed["ws_cp_corrected"].values,
    )
    df_input["ws_est_blend"] = (
        df_input["ws_est_gain"] * df_input["ws_est_from_power_only"]
        + (1 - df_input["ws_est_gain"]) * df_input["ws_cp_corrected"]
    )
    if plot_cfg is not None:
        plot_ws_est_one_ttype_or_wtg(
            df=df_input,
            ttype_or_wtg=ttype.turbine_type,
            pc_transposed=pc_transposed,
            plot_cfg=plot_cfg,
        )
        if not plot_cfg.skip_per_turbine_plots:
            for wtg in df_input.index.unique(level="TurbineName"):
                df_wtg = df_input.loc[wtg]
                plot_ws_est_one_ttype_or_wtg(
                    df=df_wtg, ttype_or_wtg=wtg, pc_transposed=pc_transposed, plot_cfg=plot_cfg
                )

    cols_to_return = ["ws_est_from_power_only", "ws_est_blend"]
    df[cols_to_return] = df_input[cols_to_return]
    return df


def add_ws_est(cfg: WindUpConfig, wf_df: pd.DataFrame, pc_per_ttype: dict, plot_cfg: PlotConfig | None) -> pd.DataFrame:
    _msg = "#" * 78 + "\n# estimate wind speed from power\n" + "#" * 78
    logger.info(_msg)
    df_input = wf_df.copy()
    wf_df = pd.DataFrame()
    for ttype in cfg.list_unique_turbine_types():
        wtgs = cfg.list_turbine_ids_of_type(ttype)
        df_ttype = df_input.loc[wtgs]
        df_ = add_ws_est_one_ttype(
            cfg=cfg,
            df=df_ttype,
            ttype=ttype,
            pc=pc_per_ttype[ttype.turbine_type],
            plot_cfg=plot_cfg,
        )
        wf_df = pd.concat([wf_df, df_])
    return wf_df.sort_index()
