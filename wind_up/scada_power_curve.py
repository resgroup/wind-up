import logging

import numpy as np
import pandas as pd

from wind_up.models import PlotConfig, WindUpConfig
from wind_up.plots.scada_power_curve_plots import plot_pc_per_ttype, plot_removed_data_per_ttype_and_wtg

logger = logging.getLogger(__name__)


def calc_pc_and_rated_ws_one_ttype(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_bin_width: float,
) -> tuple[pd.DataFrame, float]:
    x_bin_edges = np.arange(0, df[x_col].max() + x_bin_width, x_bin_width)
    pc = df.groupby(by=pd.cut(df[x_col], bins=x_bin_edges, retbins=False), observed=False).agg(
        x_mean=pd.NamedAgg(column=x_col, aggfunc=lambda x: x.mean()),
        y_mean=pd.NamedAgg(column=y_col, aggfunc=lambda x: x.mean()),
    )
    pc["bin_mid"] = [x.mid for x in pc.index]
    # this rated_ws calculation is not very robust if we are lacking data
    rated_ws_threshold = 0.995
    rated_ws = pc["x_mean"][pc["y_mean"] / pc["y_mean"].max() > rated_ws_threshold].iloc[0]
    logger.info(f"estimated rated wind speed = {rated_ws:.1f} m/s")
    below_rated = pc["x_mean"].fillna(0) < rated_ws
    low_pw_threshold = 0.005
    low_power = pc["y_mean"] / pc["y_mean"].max() < low_pw_threshold
    if (below_rated & low_power).any():
        cutin_ws = pc["x_mean"][below_rated & (pc["y_mean"].fillna(0) / pc["y_mean"].max() < low_pw_threshold)].iloc[-1]
    else:
        cutin_ws = pc["x_mean"].dropna().iloc[0]
    logger.info(f"estimated cut-in wind speed = {cutin_ws:.1f} m/s")

    pc[x_col] = pc["x_mean"].fillna(pc["bin_mid"])
    pc[y_col] = pc["y_mean"]
    pc.loc[pc["bin_mid"] < cutin_ws - x_bin_width / 2, y_col] = 0
    pc.loc[pc["bin_mid"] > rated_ws + x_bin_width / 2, y_col] = pc[y_col].max()

    if pc[y_col].diff().min() < 0:
        msg = f"power curve must be monotonically increasing (x_col = {x_col}, y_col = {y_col})"
        raise RuntimeError(msg)

    return pc, rated_ws


def calc_pc_and_rated_ws(
    cfg: WindUpConfig,
    wf_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_bin_width: float,
    plot_cfg: PlotConfig | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    pc_per_ttype: dict[str, pd.DataFrame] = {}
    rated_ws_per_ttype: dict[str, float] = {}
    for ttype in cfg.list_unique_turbine_types():
        wtgs = cfg.list_turbine_ids_of_type(ttype)
        df_ttype = wf_df.loc[wtgs]
        ttype_str = ttype.turbine_type
        original_x_bin_width = x_bin_width
        success = False
        while x_bin_width < (original_x_bin_width * 3):
            try:
                pc_per_ttype[ttype_str], rated_ws_per_ttype[ttype_str] = calc_pc_and_rated_ws_one_ttype(
                    df=df_ttype,
                    x_col=x_col,
                    y_col=y_col,
                    x_bin_width=x_bin_width,
                )
                success = True
                break
            except RuntimeError:
                x_bin_width *= 1.1
                logger.info(f"power curve calculation failed, trying again with larger bin width {x_bin_width:.2f}")
        if not success:
            msg = "power curve calculation failed for all bin widths"
            raise RuntimeError(msg)
    if plot_cfg is not None:
        plot_pc_per_ttype(cfg=cfg, pc_per_ttype=pc_per_ttype, plot_cfg=plot_cfg)
        plot_removed_data_per_ttype_and_wtg(cfg=cfg, wf_df=wf_df, pc_per_ttype=pc_per_ttype, plot_cfg=plot_cfg)
    return pc_per_ttype, rated_ws_per_ttype
