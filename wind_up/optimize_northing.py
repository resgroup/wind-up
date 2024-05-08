import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

from wind_up.constants import (
    RAW_POWER_COL,
    RAW_YAWDIR_COL,
    REANALYSIS_WD_COL,
    TIMESTAMP_COL,
    WINDFARM_YAWDIR_COL,
)
from wind_up.math_funcs import circ_diff
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.northing import (
    add_wf_yawdir,
    apply_northing_corrections,
    calc_northed_col_name,
    check_wtg_northing,
)
from wind_up.northing_utils import YAW_OK_PW_FRACTION, add_ok_yaw_col
from wind_up.plots.optimize_northing_plots import (
    plot_diff_to_north_ref_wd,
    plot_wf_yawdir_and_reanalysis_timeseries,
    plot_yaw_diff_vs_power,
)

logger = logging.getLogger(__name__)

DECAY_FRACTION = 0.4


def northing_score_changepoint_component(changepoint_count: int) -> float:
    return float(changepoint_count)


def northing_score(
    wtg_df: pd.DataFrame, *, north_ref_wd_col: str, changepoint_count: int, rated_power: float, timebase_s: int
) -> float:
    # this component penalizes long-ish, large north errors
    max_component = max(0, wtg_df[f"long_rolling_diff_to_{north_ref_wd_col}"].abs().max() - 4) ** 2

    # this component penalizes the median north error of filtered data being far from 0
    median_component = max(0, abs(wtg_df[f"filt_diff_to_{north_ref_wd_col}"].median()) - 0.1) ** 2

    # this component penalizes raw data having any north errors
    max_weight = rated_power * YAW_OK_PW_FRACTION
    min_weight = max_weight / 1000 * (min(600, timebase_s) / 600)
    raw_wmean_component = (
        wtg_df[f"yaw_diff_to_{north_ref_wd_col}"].clip(lower=-30, upper=30)
        * wtg_df[RAW_POWER_COL].clip(lower=min_weight, upper=max_weight)
    ).abs().mean() / max_weight

    # this component encourages the correction list to be as short as possible
    changepoint_component = northing_score_changepoint_component(changepoint_count)

    return max_component + median_component + raw_wmean_component + changepoint_component


def add_northing_ok_and_diff_cols(wtg_df: pd.DataFrame, *, north_ref_wd_col: str, northed_col: str) -> pd.DataFrame:
    wtg_df = wtg_df.copy()

    wtg_df[f"yaw_diff_to_{north_ref_wd_col}"] = circ_diff(wtg_df[northed_col], wtg_df[north_ref_wd_col])
    wtg_df[f"filt_diff_to_{north_ref_wd_col}"] = wtg_df[f"yaw_diff_to_{north_ref_wd_col}"]
    wtg_df.loc[~wtg_df[f"ok_for_{north_ref_wd_col}_northing"], f"filt_diff_to_{north_ref_wd_col}"] = pd.NA
    return wtg_df


def add_northed_ok_diff_and_rolling_cols(
    wtg_df: pd.DataFrame,
    *,
    north_ref_wd_col: str,
    timebase_s: int,
    north_offset: float | None = None,
    north_offset_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    wtg_df = wtg_df.copy()
    northed_col = calc_northed_col_name(north_ref_wd_col)
    if north_offset is not None:
        wtg_df[northed_col] = wtg_df[RAW_YAWDIR_COL] + north_offset
        wtg_df[northed_col] = wtg_df[northed_col] % 360
    elif north_offset_df is not None:
        for ts, no in zip(
            north_offset_df[TIMESTAMP_COL].to_list(),
            north_offset_df["north_offset"].to_list(),
            strict=True,
        ):
            wtg_df.loc[wtg_df.index >= ts, northed_col] = wtg_df.loc[wtg_df.index >= ts, RAW_YAWDIR_COL] + no
        wtg_df[northed_col] = wtg_df[northed_col] % 360

    wtg_df = add_northing_ok_and_diff_cols(wtg_df, north_ref_wd_col=north_ref_wd_col, northed_col=northed_col)
    rolling_hours = 6
    rows_per_hour = 3600 / timebase_s
    wtg_df[f"short_rolling_diff_to_{north_ref_wd_col}"] = (
        wtg_df[f"filt_diff_to_{north_ref_wd_col}"]
        .rolling(
            center=True,
            window=round(rolling_hours * rows_per_hour),
            min_periods=round(rolling_hours * rows_per_hour // 3),
        )
        .median()
    )
    rolling_hours = 15 * 24
    wtg_df[f"long_rolling_diff_to_{north_ref_wd_col}"] = (
        wtg_df[f"filt_diff_to_{north_ref_wd_col}"]
        .rolling(
            center=True,
            window=round(rolling_hours * rows_per_hour),
            min_periods=round(rolling_hours * rows_per_hour // 3),
        )
        .median()
    )
    return wtg_df


def calc_good_north_offset(section_df: pd.DataFrame, north_ref_wd_col: str) -> float:
    return -section_df[f"filt_diff_to_{north_ref_wd_col}"].median()


def calc_north_offset_col(
    wtg_north_table: pd.DataFrame,
    *,
    wtg_df: pd.DataFrame,
    north_ref_wd_col: str,
    timebase_s: int,
) -> pd.DataFrame:
    wtg_df = wtg_df.copy()
    wtg_df = add_northing_ok_and_diff_cols(wtg_df, north_ref_wd_col=north_ref_wd_col, northed_col=RAW_YAWDIR_COL)

    wtg_north_table = wtg_north_table.copy()
    wtg_north_table = wtg_north_table.sort_values(by=[TIMESTAMP_COL], ascending=True)
    end_dt = wtg_df.index.max() + pd.Timedelta(seconds=timebase_s)
    for start_dt in reversed(wtg_north_table[TIMESTAMP_COL].to_list()):
        section_df = wtg_df[(wtg_df.index >= start_dt) & (wtg_df.index < end_dt)]
        wtg_north_table.loc[
            wtg_north_table[TIMESTAMP_COL] == start_dt,
            "north_offset",
        ] = calc_good_north_offset(
            section_df,
            north_ref_wd_col=north_ref_wd_col,
        )
        end_dt = start_dt
    return wtg_north_table


def north_table_is_valid(
    wtg_north_table: pd.DataFrame,
    *,
    wtg_df: pd.DataFrame,
    check_north_offset: bool = False,
) -> bool:
    is_valid = True
    if wtg_north_table[TIMESTAMP_COL].isna().any():
        is_valid = False
    if wtg_north_table[TIMESTAMP_COL].duplicated().any():
        is_valid = False
    if not wtg_north_table[TIMESTAMP_COL].is_monotonic_increasing:
        is_valid = False
    if wtg_north_table[TIMESTAMP_COL].min() != wtg_df.index.min():
        is_valid = False
    if wtg_north_table[TIMESTAMP_COL].max() > wtg_df.index.max():
        is_valid = False
    if check_north_offset:
        if wtg_north_table["north_offset"].isna().any():
            is_valid = False
        max_abs_north_offset = 180
        if not all(wtg_north_table["north_offset"].abs() <= max_abs_north_offset):
            is_valid = False

    return is_valid


def score_wtg_north_table(
    *,
    wtg_north_table: pd.DataFrame,
    wtg_df: pd.DataFrame,
    improve_north_offset_col: bool,
    north_ref_wd_col: str,
    rated_power: float,
    timebase_s: int,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    if improve_north_offset_col:
        output_north_table = calc_north_offset_col(
            wtg_north_table,
            wtg_df=wtg_df,
            north_ref_wd_col=north_ref_wd_col,
            timebase_s=timebase_s,
        )
    else:
        output_north_table = wtg_north_table.copy()
    output_wtg_df = add_northed_ok_diff_and_rolling_cols(
        wtg_df.copy(),
        north_ref_wd_col=north_ref_wd_col,
        north_offset_df=output_north_table,
        timebase_s=timebase_s,
    )

    score = northing_score(
        output_wtg_df,
        north_ref_wd_col=north_ref_wd_col,
        changepoint_count=len(output_north_table),
        rated_power=rated_power,
        timebase_s=timebase_s,
    )

    return output_north_table, score, output_wtg_df


def calc_max_changepoints_to_add(changepoint_count: int, *, score: float) -> int:
    max_changepoints_to_add = 0
    headroom = score - northing_score_changepoint_component(changepoint_count + 1)
    while headroom > 0:
        max_changepoints_to_add += 1
        headroom = score - northing_score_changepoint_component(max_changepoints_to_add + 1)
    return max_changepoints_to_add


def list_possible_moves(
    prev_best_north_table: pd.DataFrame,
    *,
    do_changepoint_moves: bool,
    max_changepoints_to_add: int,
) -> list[str]:
    moves = []
    moves.extend([f"shift_changepoint_{x}_forward" for x in prev_best_north_table.index[1:]])
    moves.extend([f"shift_changepoint_{x}_back" for x in prev_best_north_table.index[1:]])
    if do_changepoint_moves:
        moves.append("add_1_changepoint")
        moves.extend([f"add_{x}_changepoints" for x in range(2, max_changepoints_to_add + 1)])
    return moves


def get_changepoint_objects(
    *,
    prev_best_wtg_df: pd.DataFrame,
    north_ref_wd_col: str,
) -> tuple[rpt.base.BaseEstimator, np.ndarray]:
    col = f"filt_diff_to_{north_ref_wd_col}"
    model = "l1"
    dropna_df = prev_best_wtg_df.dropna(subset=[col])
    signal = dropna_df[col].to_numpy()
    timestamps = dropna_df.index.to_numpy()
    algo = rpt.BottomUp(model=model).fit(signal)
    return algo, timestamps


def make_move(
    move: str,
    *,
    prev_best_north_table: pd.DataFrame,
    shift_step_size: int,
    do_changepoint_moves: bool,
    algo: rpt.base.BaseEstimator,
    timestamps: np.ndarray,
    timebase_s: int,
) -> pd.DataFrame:
    if move.startswith("shift_changepoint_"):
        cp_idx = int(move.split("_")[-2])
        sign = 1 if move.endswith("forward") else -1
        this_north_table = prev_best_north_table.copy()
        this_north_table.loc[cp_idx, TIMESTAMP_COL] = this_north_table.loc[
            cp_idx,
            TIMESTAMP_COL,
        ] + pd.Timedelta(seconds=sign * timebase_s * shift_step_size)
    elif do_changepoint_moves and move.startswith("add_"):
        n_changepoints_to_add = int(move.split("_")[1])
        bkp_idxs = algo.predict(n_bkps=n_changepoints_to_add)[:-1]
        if len(bkp_idxs) != n_changepoints_to_add:
            msg = f"found {len(bkp_idxs)} bkp_idxs, expected {n_changepoints_to_add}"
            raise RuntimeError(msg)
        dt_list = list(timestamps[bkp_idxs])

        this_north_offset_df_dtidx = prev_best_north_table.set_index(TIMESTAMP_COL)
        this_index = this_north_offset_df_dtidx.index.append(pd.Index(dt_list))
        this_north_table = this_north_offset_df_dtidx.reindex(this_index).sort_index().reset_index()
        this_north_table = this_north_table.rename(columns={"index": TIMESTAMP_COL})
    else:
        msg = f"invalid move {move}"
        raise RuntimeError(msg)
    return this_north_table


def make_move_and_score_wtg_north_table(
    move: str,
    *,
    prev_best_north_table: pd.DataFrame,
    wtg_df: pd.DataFrame,
    shift_step_size: int,
    north_ref_wd_col: str,
    rated_power: float,
    timebase_s: int,
    do_changepoint_moves: bool,
    algo: rpt.base.BaseEstimator,
    timestamps: np.ndarray,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    this_north_table = make_move(
        move,
        prev_best_north_table=prev_best_north_table,
        shift_step_size=shift_step_size,
        do_changepoint_moves=do_changepoint_moves,
        algo=algo,
        timestamps=timestamps,
        timebase_s=timebase_s,
    )
    if north_table_is_valid(this_north_table, wtg_df=wtg_df):
        min_step_size_for_run_optimize = 100
        run_optimize_north_offset = (
            (len(this_north_table) > len(prev_best_north_table))
            or (shift_step_size > min_step_size_for_run_optimize)
            or (shift_step_size == 1)
        )
    else:
        this_north_table = prev_best_north_table
        run_optimize_north_offset = False
    this_north_table, this_score, this_wtg_df = score_wtg_north_table(
        wtg_north_table=this_north_table,
        wtg_df=wtg_df,
        improve_north_offset_col=run_optimize_north_offset,
        north_ref_wd_col=north_ref_wd_col,
        rated_power=rated_power,
        timebase_s=timebase_s,
    )
    return this_north_table, this_score, this_wtg_df


def clip_wtg_north_table(initial_wtg_north_table: pd.DataFrame, *, wtg_df: pd.DataFrame) -> pd.DataFrame:
    clipped_wtg_north_table = initial_wtg_north_table.copy()

    if clipped_wtg_north_table[TIMESTAMP_COL].min() < wtg_df.index.min():
        first_row_before_wf_df = (
            clipped_wtg_north_table.loc[clipped_wtg_north_table[TIMESTAMP_COL] <= wtg_df.index.min()]
            .sort_values(by=[TIMESTAMP_COL], ascending=False)
            .iloc[:1]
        )
        clipped_wtg_north_table = pd.concat(
            [
                first_row_before_wf_df,
                clipped_wtg_north_table[clipped_wtg_north_table[TIMESTAMP_COL] > wtg_df.index.min()],
            ],
        ).reset_index(drop=True)
        clipped_wtg_north_table.loc[0, TIMESTAMP_COL] = wtg_df.index.min()

    if clipped_wtg_north_table[TIMESTAMP_COL].min() > wtg_df.index.min():
        clipped_wtg_north_table.loc[0, TIMESTAMP_COL] = wtg_df.index.min()

    return clipped_wtg_north_table


def prep_for_optimize_wtg_north_table(
    wtg_df: pd.DataFrame,
    *,
    wtg_name: str,
    north_ref_wd_col: str,
    rated_power: float,
    timebase_s: int,
    plot_cfg: PlotConfig | None,
    initial_wtg_north_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wtg_df = wtg_df.copy()
    wtg_df = add_ok_yaw_col(
        wtg_df,
        new_col_name=f"ok_for_{north_ref_wd_col}_northing",
        wd_col=north_ref_wd_col,
        rated_power=rated_power,
        timebase_s=timebase_s,
    )
    wtg_df = add_northed_ok_diff_and_rolling_cols(
        wtg_df, north_ref_wd_col=north_ref_wd_col, timebase_s=timebase_s, north_offset=0
    )

    initial_score = northing_score(
        wtg_df,
        north_ref_wd_col=north_ref_wd_col,
        changepoint_count=0,
        rated_power=rated_power,
        timebase_s=timebase_s,
    )
    logger.info(f"\n\nwtg_name={wtg_name}, north_ref_wd_col={north_ref_wd_col}, initial_score={initial_score:.2f}")

    if plot_cfg is not None:
        plot_yaw_diff_vs_power(wtg_df, wtg_name=wtg_name, north_ref_wd_col=north_ref_wd_col, plot_cfg=plot_cfg)

    if initial_wtg_north_table is None or len(initial_wtg_north_table) == 0:
        wtg_north_table = pd.DataFrame(
            data={TIMESTAMP_COL: wtg_df.index.min(), "north_offset": 0.0},
            index=[0],
        )
    else:
        initial_wtg_north_table = clip_wtg_north_table(
            initial_wtg_north_table,
            wtg_df=wtg_df,
        )

        if not north_table_is_valid(initial_wtg_north_table, wtg_df=wtg_df, check_north_offset=True):
            msg = "initial_wtg_north_table is not valid"
            raise ValueError(msg)
        wtg_north_table = initial_wtg_north_table.copy()

    if not north_table_is_valid(wtg_north_table, wtg_df=wtg_df, check_north_offset=True):
        msg = "wtg_north_table is not valid"
        raise RuntimeError(msg)

    return wtg_df, wtg_north_table


def optimize_wtg_north_table(
    *,
    wtg_df: pd.DataFrame,
    wtg_name: str,
    rated_power: float,
    north_ref_wd_col: str,
    timebase_s: int,
    plot_cfg: PlotConfig | None,
    initial_wtg_north_table: pd.DataFrame | None = None,
    best_score_margin: float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    wtg_df = wtg_df.copy()

    wtg_df, wtg_north_table = prep_for_optimize_wtg_north_table(
        wtg_df,
        wtg_name=wtg_name,
        north_ref_wd_col=north_ref_wd_col,
        rated_power=rated_power,
        timebase_s=timebase_s,
        plot_cfg=plot_cfg,
        initial_wtg_north_table=initial_wtg_north_table,
    )

    loop_count = 0
    best_north_table, initial_score, best_wtg_df = score_wtg_north_table(
        wtg_north_table=wtg_north_table,
        wtg_df=wtg_df,
        improve_north_offset_col=True,
        north_ref_wd_col=north_ref_wd_col,
        rated_power=rated_power,
        timebase_s=timebase_s,
    )
    best_score = initial_score
    logger.info(f"best_score={best_score:.2f} before optimization")

    if plot_cfg is not None:
        plot_diff_to_north_ref_wd(
            best_wtg_df,
            wtg_name=wtg_name,
            north_ref_wd_col=north_ref_wd_col,
            loop_count=loop_count,
            plot_cfg=plot_cfg,
        )

    done_optimizing = False
    max_changepoints_to_add = min(5, calc_max_changepoints_to_add(len(best_north_table), score=best_score))
    initial_step_size: int = 100
    shift_step_size: int = initial_step_size
    tries_left: int = 1
    while not done_optimizing:
        loop_count += 1
        best_move_found_this_loop = False
        prev_best_north_table = best_north_table.copy()
        prev_best_wtg_df = best_wtg_df.copy()

        do_changepoint_moves = max_changepoints_to_add > 0
        moves = list_possible_moves(
            prev_best_north_table,
            do_changepoint_moves=do_changepoint_moves,
            max_changepoints_to_add=max_changepoints_to_add,
        )
        if do_changepoint_moves:
            algo, timestamps = get_changepoint_objects(
                prev_best_wtg_df=prev_best_wtg_df,
                north_ref_wd_col=north_ref_wd_col,
            )

        for move in moves:
            this_north_table, this_score, this_wtg_df = make_move_and_score_wtg_north_table(
                move,
                prev_best_north_table=prev_best_north_table,
                wtg_df=wtg_df,
                shift_step_size=shift_step_size,
                north_ref_wd_col=north_ref_wd_col,
                rated_power=rated_power,
                timebase_s=timebase_s,
                do_changepoint_moves=do_changepoint_moves,
                algo=algo,
                timestamps=timestamps,
            )

            if this_score < (best_score - best_score_margin):
                best_north_table = this_north_table.copy()
                best_score = this_score
                best_wtg_df = this_wtg_df.copy()
                best_move_found_this_loop = True
                logger.info(
                    f"wtg_name={wtg_name}, best_score={this_score:.3f}, loop_count={loop_count}, "
                    f"shift_step_size={shift_step_size}, len(best_north_table)={len(best_north_table)}, "
                    f"move={move}",
                )
        if len(best_north_table) == len(prev_best_north_table):
            max_changepoints_to_add = 0
        else:
            tries_left += 1
            logger.info(f"tries_left increased to {tries_left}")
            max_changepoints_to_add = min(2, calc_max_changepoints_to_add(len(best_north_table), score=best_score))
        if not best_move_found_this_loop:
            shift_step_size = min(shift_step_size - 1, round(shift_step_size * DECAY_FRACTION))
            if shift_step_size < 1:
                max_changepoints_to_add = min(
                    2,
                    calc_max_changepoints_to_add(len(best_north_table), score=best_score),
                )
                shift_step_size = initial_step_size + 1 + (1 / (DECAY_FRACTION ** (math.pi * (tries_left + 1))) % 10)
                shift_step_size = round(shift_step_size)
                tries_left -= 1
                logger.info(f"tries_left decreased to {tries_left}")
        if tries_left == 0:
            done_optimizing = True

    wtg_north_table = best_north_table.copy()
    wtg_df = add_northed_ok_diff_and_rolling_cols(
        wtg_df,
        north_ref_wd_col=north_ref_wd_col,
        timebase_s=timebase_s,
        north_offset_df=wtg_north_table,
    )
    if plot_cfg is not None:
        plot_diff_to_north_ref_wd(
            wtg_df,
            wtg_name=wtg_name,
            north_ref_wd_col=north_ref_wd_col,
            loop_count=loop_count,
            plot_cfg=plot_cfg,
        )

    return wtg_north_table, wtg_df, initial_score, best_score


def optimize_wf_north_table(
    wf_df: pd.DataFrame,
    *,
    north_ref_wd_col: str,
    cfg: WindUpConfig,
    plot_cfg: PlotConfig | None,
    best_score_margin: float = 0,
) -> pd.DataFrame:
    optimized_wf_north_table = pd.DataFrame()
    initial_wf_north_table = pd.DataFrame(
        data=cfg.northing_corrections_utc,
        columns=["TurbineName", TIMESTAMP_COL, "north_offset"],
    )
    for wtg_name in sorted(wf_df.index.unique(level="TurbineName").to_list()):
        wtg_obj = next(x for x in cfg.asset.wtgs if x.name == wtg_name)
        rated_power = wtg_obj.turbine_type.rated_power_kw

        wtg_df = wf_df.loc[wtg_name].copy()
        max_northing_error_before = check_wtg_northing(
            wtg_df,
            wtg_name=wtg_name,
            north_ref_wd_col=north_ref_wd_col,
            timebase_s=cfg.timebase_s,
            plot_cfg=None,
        )

        initial_wtg_north_table = initial_wf_north_table.loc[initial_wf_north_table["TurbineName"] == wtg_name]
        changepoints_before = max(1, len(initial_wtg_north_table))
        wtg_north_table, optimized_wtg_df, score_before, score_after = optimize_wtg_north_table(
            wtg_df=wtg_df,
            wtg_name=wtg_name,
            rated_power=rated_power,
            north_ref_wd_col=north_ref_wd_col,
            timebase_s=cfg.timebase_s,
            plot_cfg=plot_cfg,
            initial_wtg_north_table=initial_wtg_north_table,
            best_score_margin=best_score_margin,
        )

        northed_col = calc_northed_col_name(north_ref_wd_col)
        optimized_wtg_df["YawAngleMean"] = optimized_wtg_df[northed_col]
        max_northing_error_after = check_wtg_northing(
            optimized_wtg_df,
            wtg_name=wtg_name,
            north_ref_wd_col=north_ref_wd_col,
            timebase_s=cfg.timebase_s,
            plot_cfg=plot_cfg,
        )

        changepoints_after = len(wtg_north_table)
        logger.info(
            f"changepoints changed from {changepoints_before} to {changepoints_after} "
            f"[{changepoints_after - changepoints_before}]",
        )
        logger.info(
            f"northing score changed from {score_before:.1f} to {score_after:.1f} [{score_after - score_before:.1f}]"
        )
        logger.info(
            f"max_northing_error changed from {max_northing_error_before:.1f} to {max_northing_error_after:.1f} "
            f"[{max_northing_error_after - max_northing_error_before:.1f}]",
        )

        wtg_north_table["TurbineName"] = wtg_name
        optimized_wf_north_table = (
            pd.concat([optimized_wf_north_table, wtg_north_table])
            .sort_values(by=["TurbineName", TIMESTAMP_COL])
            .reset_index(drop=True)
        )
    return optimized_wf_north_table


def write_northing_yaml(wf_north_table: pd.DataFrame, *, fpath: Path) -> None:
    north_table_for_yaml = wf_north_table.copy()
    north_table_for_yaml[TIMESTAMP_COL] = north_table_for_yaml[TIMESTAMP_COL].dt.strftime(
        "%Y-%m-%d %H:%M:%S",
    )
    yaml_strings = []
    for _, row in north_table_for_yaml.iterrows():
        yaml_strings.append(f"    - ['{row['TurbineName']}', {row[TIMESTAMP_COL]}, {row['north_offset']}]")
    yaml_content = "\n".join(yaml_strings)
    with fpath.open(mode="w") as yaml_file:
        yaml_file.write(yaml_content)


def auto_northing_corrections(
    wf_df: pd.DataFrame,
    *,
    cfg: WindUpConfig,
    plot_cfg: PlotConfig | None,
) -> pd.DataFrame:
    wf_df = wf_df.copy()

    reanalysis_wf_north_table = optimize_wf_north_table(
        wf_df,
        north_ref_wd_col=REANALYSIS_WD_COL,
        cfg=cfg,
        plot_cfg=plot_cfg,
        best_score_margin=0.5,
    )
    if plot_cfg is not None:
        reanalysis_wf_north_table.to_csv(cfg.out_dir / "reanalysis_wf_north_table.csv")
        write_northing_yaml(reanalysis_wf_north_table, fpath=cfg.out_dir / "reanalysis_wf_north_table.yaml")

    wf_df = apply_northing_corrections(
        wf_df,
        wf_north_table=reanalysis_wf_north_table,
        north_ref_wd_col=REANALYSIS_WD_COL,
        cfg=cfg,
        plot_cfg=plot_cfg,
    )

    wf_df = add_wf_yawdir(wf_df, cfg=cfg)

    if plot_cfg is not None:
        plot_wf_yawdir_and_reanalysis_timeseries(wf_df, cfg=cfg, plot_cfg=plot_cfg)

    optimized_northing_corrections = optimize_wf_north_table(
        wf_df,
        north_ref_wd_col=WINDFARM_YAWDIR_COL,
        cfg=cfg,
        plot_cfg=plot_cfg,
    )
    if plot_cfg is not None:
        optimized_northing_corrections.to_csv(cfg.out_dir / "optimized_northing_corrections.csv")
        write_northing_yaml(optimized_northing_corrections, fpath=cfg.out_dir / "optimized_northing_corrections.yaml")

    return apply_northing_corrections(
        wf_df,
        wf_north_table=optimized_northing_corrections,
        north_ref_wd_col=WINDFARM_YAWDIR_COL,
        cfg=cfg,
        plot_cfg=plot_cfg,
    )
