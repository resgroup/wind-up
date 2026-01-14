"""Pre-Post Analysis Module."""

from __future__ import annotations

import contextlib
import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm, t
from tqdm.auto import tqdm

from wind_up.constants import TIMESTAMP_COL
from wind_up.plots.pp_analysis_plots import plot_pp_data_coverage, plot_pre_post_pp_analysis
from wind_up.result_manager import result_manager

if TYPE_CHECKING:
    from wind_up.models import PlotConfig, Turbine, WindUpConfig
logger = logging.getLogger(__name__)


def _pp_raw_df_pandas(
    pre_or_post_df: pd.DataFrame,
    pre_or_post: str,
    *,
    ws_col: str,
    ws_bin_edges: np.ndarray,
    pw_col: str,
    timebase_s: int,
) -> pd.DataFrame:
    return (
        pre_or_post_df.loc[:, [pw_col, ws_col]]
        .dropna()
        .groupby(
            by=pd.cut(pre_or_post_df[ws_col], bins=ws_bin_edges, retbins=False),
            observed=False,
        )
        .agg(
            count=pd.NamedAgg(column=pw_col, aggfunc=len),
            ws_mean=pd.NamedAgg(column=ws_col, aggfunc="mean"),
            ws_std=pd.NamedAgg(column=ws_col, aggfunc="std"),
            pw_mean=pd.NamedAgg(column=pw_col, aggfunc="mean"),
            pw_std=pd.NamedAgg(column=pw_col, aggfunc="std"),
        )
        .assign(
            ws_std=lambda x: x["ws_std"].fillna(0),
            pw_std=lambda x: x["pw_std"].fillna(0),
            hours=lambda x: x["count"] * timebase_s / 3600,
            ws_sem=lambda x: x["ws_std"] / np.sqrt(x["count"].clip(lower=1)),
            pw_sem=lambda x: x["pw_std"] / np.sqrt(x["count"].clip(lower=1)),
        )
        .pipe(lambda d: d.set_axis(d.columns.map(lambda x: f"{x}_{pre_or_post}"), axis="columns"))
        .assign(
            bin_left=lambda x: [i.left for i in x.index],
            bin_mid=lambda x: [i.mid for i in x.index],
            bin_right=lambda x: [i.right for i in x.index],
            bin_closed_right=lambda x: [i.closed_right for i in x.index],
        )
        .set_index("bin_mid", drop=False, verify_integrity=True)
        .rename_axis(f"{ws_col}_bin_mid", axis=0)
    )


def _pp_raw_df_polars(
    pre_or_post_df: pl.DataFrame,
    pre_or_post: str,
    *,
    ws_col: str,
    ws_bin_edges: np.ndarray,
    pw_col: str,
    timebase_s: int,
) -> pl.DataFrame:
    # Create a lookup dataframe for bin edges (convert to lazy)
    bin_edges_df = pl.DataFrame(
        {
            "bin_right": ws_bin_edges[1:],
            "bin_left": ws_bin_edges[:-1],
            "bin_mid": (ws_bin_edges[:-1] + ws_bin_edges[1:]) / 2,
        }
    ).lazy()

    aggregated = (
        pre_or_post_df.lazy()
        .select([pw_col, ws_col])
        .drop_nulls()
        .with_columns(
            pl.col(ws_col)
            .cut(breaks=ws_bin_edges.tolist(), include_breaks=True)
            .struct.field("breakpoint")
            .alias("bin_right")
        )
        .join(bin_edges_df, on="bin_right", how="left")
        .group_by("bin_right", maintain_order=True)
        .agg(
            [
                pl.len().alias("count"),
                pl.col(ws_col).mean().alias("ws_mean"),
                pl.col(ws_col).std().alias("ws_std"),
                pl.col(pw_col).mean().alias("pw_mean"),
                pl.col(pw_col).std().alias("pw_std"),
            ]
        )
    )

    bin_closed_right = True

    # Join with all bins to include zero-count bins
    return (
        (
            bin_edges_df.join(aggregated, on="bin_right", how="left")
            .with_columns(
                [
                    pl.coalesce(pl.col("count"), pl.lit(0)).alias("count"),
                    pl.coalesce(pl.col("ws_std"), pl.lit(0.0)).alias("ws_std"),
                    pl.coalesce(pl.col("pw_std"), pl.lit(0.0)).alias("pw_std"),
                ]
            )
            .with_columns(
                [
                    (pl.col("count") * timebase_s / 3600).alias("hours"),
                    (pl.col("ws_std") / pl.col("count").clip(lower_bound=1).sqrt()).alias("ws_sem"),
                    (pl.col("pw_std") / pl.col("count").clip(lower_bound=1).sqrt()).alias("pw_sem"),
                    pl.lit(bin_closed_right).alias("bin_closed_right"),
                ]
            )
            .select(
                [
                    pl.col("bin_mid").alias(f"{ws_col}_bin_mid"),
                    *[
                        pl.col(c).alias(f"{c}_{pre_or_post}")
                        for c in ["count", "ws_mean", "ws_std", "pw_mean", "pw_std", "hours", "ws_sem", "pw_sem"]
                    ],
                    pl.col("bin_left"),
                    pl.col("bin_mid"),
                    pl.col("bin_right"),
                    pl.col("bin_closed_right"),
                ]
            )
        )
        .sort(by=f"{ws_col}_bin_mid")
        .collect()
    )


def _calc_rated_ws_pandas(*, pp_df: pd.DataFrame, pw_col: str, rated_power: float) -> float:
    return pp_df.loc[pp_df[pw_col] >= rated_power * 0.995, "bin_mid"].min() + 1


def _calc_rated_ws_polars(*, pp_df: pl.DataFrame, pw_col: str, rated_power: float) -> float:
    return pp_df.filter(pl.col(pw_col) >= rated_power * 0.995).select(pl.col("bin_mid").min() + 1).item()


def _cook_pp_pandas(
    pp_df: pd.DataFrame, *, pre_or_post: str, ws_bin_width: float, rated_power: float, clip_to_rated: bool
) -> pd.DataFrame:
    pp_df = pp_df.copy()

    valid_col = f"{pre_or_post}_valid"
    raw_pw_col = f"pw_mean_{pre_or_post}_raw"
    raw_hours_col = f"hours_{pre_or_post}_raw"
    pw_col = f"pw_mean_{pre_or_post}"
    pw_sem_col = f"pw_sem_{pre_or_post}"
    hours_col = f"hours_{pre_or_post}"
    ws_col = f"ws_mean_{pre_or_post}"
    pw_at_mid_col = f"pw_at_mid_{pre_or_post}"
    pw_sem_at_mid_col = f"pw_sem_at_mid_{pre_or_post}"

    pp_df[raw_pw_col] = pp_df[pw_col]
    pp_df[raw_hours_col] = pp_df[hours_col]

    # IEC minimum would be 1 hrs_per_mps but using 3 b/c typically higher noise with turbine side-by-side
    hrs_per_mps = 3
    pp_df[valid_col] = pp_df[f"hours_{pre_or_post}"] > (ws_bin_width * hrs_per_mps)
    pp_df.loc[~pp_df[valid_col], [pw_col, hours_col, ws_col, pw_sem_col]] = np.nan
    pp_df[hours_col] = pp_df[hours_col].fillna(0)
    pp_df[ws_col] = pp_df[ws_col].fillna(pp_df["bin_mid"])

    pp_df[pw_col] = pp_df[pw_col].clip(lower=0)
    if clip_to_rated:
        pp_df[pw_col] = pp_df[pw_col].clip(upper=rated_power)

    # data which would have been at rated can be gap filled
    rated_ws = _calc_rated_ws_pandas(pp_df=pp_df, pw_col=raw_pw_col, rated_power=rated_power)
    empty_rated_bins_fill_value = rated_power
    if not clip_to_rated:
        with contextlib.suppress(IndexError):
            empty_rated_bins_fill_value = pp_df.loc[
                (pp_df["bin_mid"] >= rated_ws) & ~pp_df[pw_col].isna(), pw_col
            ].iloc[-1]
    pp_df.loc[(pp_df["bin_mid"] >= rated_ws) & pp_df[pw_col].isna(), pw_col] = empty_rated_bins_fill_value
    pp_df[pw_sem_col] = pp_df[pw_sem_col].ffill()

    # missing data at low wind speed can be filled with 0
    pp_df.loc[pp_df.index[pp_df[pw_col].isna().cummin()], [pw_col, pw_sem_col]] = 0

    # revisit in future, an alternative is filling in with SCADA power curve
    pp_df[pw_col] = pp_df[pw_col].interpolate()

    # interpolate power and ci to bin mid
    pp_df[pw_at_mid_col] = np.interp(pp_df["bin_mid"], pp_df[ws_col], pp_df[pw_col])
    pp_df[pw_at_mid_col] = pp_df[pw_at_mid_col].clip(lower=0)
    if clip_to_rated:
        pp_df[pw_at_mid_col] = pp_df[pw_at_mid_col].clip(upper=rated_power)
    pp_df[pw_sem_at_mid_col] = pp_df[pw_sem_col] / pp_df[pw_col] * pp_df[pw_at_mid_col]
    pp_df[pw_sem_at_mid_col] = pp_df[pw_sem_at_mid_col].fillna(0)
    pp_df[pw_sem_at_mid_col] = pp_df[pw_sem_at_mid_col].clip(lower=0, upper=pp_df[pw_sem_col])

    if pp_df[[col for col in pp_df.columns if col is not raw_pw_col]].isna().any().any():
        msg = "pp_df has missing values"
        result_manager.warning(msg)

    return pp_df


def _cook_pp_polars(
    pp_df: pl.DataFrame, *, pre_or_post: str, ws_bin_width: float, rated_power: float, clip_to_rated: bool
) -> pl.DataFrame:
    valid_col = f"{pre_or_post}_valid"
    raw_pw_col = f"pw_mean_{pre_or_post}_raw"
    raw_hours_col = f"hours_{pre_or_post}_raw"
    pw_col = f"pw_mean_{pre_or_post}"
    pw_sem_col = f"pw_sem_{pre_or_post}"
    hours_col = f"hours_{pre_or_post}"
    ws_col = f"ws_mean_{pre_or_post}"
    pw_at_mid_col = f"pw_at_mid_{pre_or_post}"
    pw_sem_at_mid_col = f"pw_sem_at_mid_{pre_or_post}"

    # IEC minimum would be 1 hrs_per_mps but using 3 b/c typically higher noise with turbine side-by-side
    hrs_per_mps = 3

    pp_df = (
        pp_df.lazy()
        .with_columns(
            [
                pl.col(pw_col).alias(raw_pw_col),
                pl.col(hours_col).alias(raw_hours_col),
                (pl.col(hours_col) > (ws_bin_width * hrs_per_mps)).alias(valid_col),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col(valid_col)).then(pl.col(pw_col)).otherwise(None).alias(pw_col),
                pl.when(pl.col(valid_col)).then(pl.col(hours_col)).otherwise(None).alias(hours_col),
                pl.when(pl.col(valid_col)).then(pl.col(ws_col)).otherwise(None).alias(ws_col),
                pl.when(pl.col(valid_col)).then(pl.col(pw_sem_col)).otherwise(None).alias(pw_sem_col),
            ]
        )
        .with_columns(
            [
                pl.col(hours_col).fill_null(0),
                pl.col(ws_col).fill_null(pl.col("bin_mid")),
            ]
        )
        .with_columns(
            [
                pl.col(pw_col).clip(lower_bound=0),
            ]
        )
        .collect()
    )

    if clip_to_rated:
        pp_df = pp_df.with_columns([pl.col(pw_col).clip(upper_bound=rated_power).alias(pw_col)])

    # Calculate rated wind speed
    rated_ws = _calc_rated_ws_polars(pp_df=pp_df, pw_col=raw_pw_col, rated_power=rated_power)

    # Gap fill rated bins
    empty_rated_bins_fill_value = rated_power
    if not clip_to_rated:
        try:
            temp = pp_df.filter((pl.col("bin_mid") >= rated_ws) & pl.col(pw_col).is_not_null())
            if len(temp) > 0:
                empty_rated_bins_fill_value = temp[pw_col][-1]
        except IndexError:
            pass

    pp_df = (
        pp_df.lazy()
        .with_columns(
            [
                pl.when((pl.col("bin_mid") >= rated_ws) & pl.col(pw_col).is_null())
                .then(pl.lit(empty_rated_bins_fill_value))
                .otherwise(pl.col(pw_col))
                .alias(pw_col)
            ]
        )
        .with_columns([pl.col(pw_sem_col).forward_fill()])
        .with_columns([pl.col(pw_col).is_null().cum_min().alias("_null_mask")])
        .with_columns(
            [
                pl.when(pl.col("_null_mask")).then(0).otherwise(pl.col(pw_col)).alias(pw_col),
                pl.when(pl.col("_null_mask")).then(0).otherwise(pl.col(pw_sem_col)).alias(pw_sem_col),
            ]
        )
        .drop("_null_mask")
        .with_columns([pl.col(pw_col).interpolate()])
        .with_columns([pl.col(pw_col).forward_fill()])
        .collect()
    )

    # Interpolate power to bin mid using numpy
    bin_mid_arr = pp_df["bin_mid"].to_numpy()
    ws_col_arr = pp_df[ws_col].to_numpy()
    pw_col_arr = pp_df[pw_col].to_numpy()

    pw_at_mid_values = np.interp(bin_mid_arr, ws_col_arr, pw_col_arr)

    pp_df = (
        pp_df.lazy()
        .with_columns([pl.lit(pw_at_mid_values).alias(pw_at_mid_col)])
        .with_columns([pl.col(pw_at_mid_col).clip(lower_bound=0)])
        .collect()
    )

    if clip_to_rated:
        pp_df = pp_df.with_columns([pl.col(pw_at_mid_col).clip(upper_bound=rated_power).alias(pw_at_mid_col)])

    pp_df = (
        pp_df.lazy()
        .with_columns(
            [
                pl.when((pl.col(pw_col).is_null()) | (pl.col(pw_col) == 0))
                .then(0)
                .otherwise(pl.col(pw_sem_col) / pl.col(pw_col) * pl.col(pw_at_mid_col))
                .alias(pw_sem_at_mid_col)
            ]
        )
        .with_columns([pl.col(pw_sem_at_mid_col).fill_null(0).fill_nan(0)])
        .with_columns([pl.col(pw_sem_at_mid_col).clip(lower_bound=0, upper_bound=pl.col(pw_sem_col))])
        .collect()
    )

    # Check for missing values (excluding raw_pw_col)
    check_cols = [col for col in pp_df.columns if col != raw_pw_col]
    if pp_df.select(check_cols).null_count().sum_horizontal()[0] > 0:
        msg = "pp_df has missing values"
        result_manager.warning(msg)

    return pp_df


def _add_uplift_cols_to_pp_df_pandas(
    pp_df: pd.DataFrame, *, p_low: float, p_high: float, t_values: np.ndarray
) -> pd.DataFrame:
    new_pp_df = pp_df.copy()
    # calculations needed for uplift vs wind speed plot
    new_pp_df["uplift_kw"] = new_pp_df["pw_at_mid_post"] - new_pp_df["pw_at_mid_expected"]
    new_pp_df["uplift_kw_se"] = np.sqrt(new_pp_df["pw_sem_at_mid_post"] ** 2 + new_pp_df["pw_sem_at_mid_expected"] ** 2)
    new_pp_df[f"uplift_p{p_low * 100:.0f}_kw"] = new_pp_df["uplift_kw"] + new_pp_df["uplift_kw_se"] * t_values
    new_pp_df[f"uplift_p{p_high * 100:.0f}_kw"] = new_pp_df["uplift_kw"] - new_pp_df["uplift_kw_se"] * t_values

    # calculations needed for relative Cp plots
    new_pp_df["relative_cp_baseline"] = new_pp_df["pw_at_mid_expected"] / new_pp_df["bin_mid"] ** 3
    max_baseline_cp = new_pp_df["relative_cp_baseline"].max()
    new_pp_df["relative_cp_baseline"] = new_pp_df["relative_cp_baseline"] / max_baseline_cp
    new_pp_df["relative_cp_post"] = new_pp_df["pw_at_mid_post"] / new_pp_df["bin_mid"] ** 3 / max_baseline_cp
    new_pp_df["relative_cp_sem_at_mid_expected"] = (
        new_pp_df["pw_sem_at_mid_expected"] / new_pp_df["bin_mid"] ** 3 / max_baseline_cp
    )
    new_pp_df["relative_cp_sem_at_mid_post"] = (
        new_pp_df["pw_sem_at_mid_post"] / new_pp_df["bin_mid"] ** 3 / max_baseline_cp
    )

    return new_pp_df


def _add_uplift_cols_to_pp_df_polars(
    pp_df: pl.DataFrame, *, p_low: float, p_high: float, t_values: np.ndarray
) -> pl.DataFrame:
    # calculations needed for uplift vs wind speed plot
    uplift_kw = pl.col("pw_at_mid_post") - pl.col("pw_at_mid_expected")
    uplift_kw_se = (pl.col("pw_sem_at_mid_post") ** 2 + pl.col("pw_sem_at_mid_expected") ** 2).sqrt()

    # calculations needed for relative Cp plots
    relative_cp_baseline = pl.col("pw_at_mid_expected") / (pl.col("bin_mid") ** 3)
    max_baseline_cp = pp_df.select(relative_cp_baseline.max()).item()

    return (
        pp_df.with_columns(pl.Series("t_values", t_values))  # Add t_values as a column
        .with_columns(
            [
                uplift_kw.alias("uplift_kw"),
                uplift_kw_se.alias("uplift_kw_se"),
                (uplift_kw + uplift_kw_se * pl.col("t_values")).alias(f"uplift_p{p_low * 100:.0f}_kw"),
                (uplift_kw - uplift_kw_se * pl.col("t_values")).alias(f"uplift_p{p_high * 100:.0f}_kw"),
                (relative_cp_baseline / max_baseline_cp).alias("relative_cp_baseline"),
                (pl.col("pw_at_mid_post") / (pl.col("bin_mid") ** 3) / max_baseline_cp).alias("relative_cp_post"),
                (pl.col("pw_sem_at_mid_expected") / (pl.col("bin_mid") ** 3) / max_baseline_cp).alias(
                    "relative_cp_sem_at_mid_expected"
                ),
                (pl.col("pw_sem_at_mid_post") / (pl.col("bin_mid") ** 3) / max_baseline_cp).alias(
                    "relative_cp_sem_at_mid_post"
                ),
            ]
        )
        .drop("t_values")  # Remove the temporary column
    )


def _pre_post_pp_analysis_pandas(
    *,
    cfg: WindUpConfig,
    test_wtg: Turbine,
    ref_name: str,
    lt_df: pd.DataFrame | None,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    ws_col: str,
    pw_col: str,
    wd_col: str,
    plot_cfg: PlotConfig | None,
    confidence_level: float = 0.9,
    test_df: pd.DataFrame | None = None,
    reverse: bool = False,
) -> tuple[dict, pd.DataFrame]:
    wtg_for_turbine_type = test_wtg
    test_name = test_wtg.name
    if reverse:
        with contextlib.suppress(StopIteration):
            wtg_for_turbine_type = next(x for x in cfg.asset.wtgs if x.name == ref_name)
        test_name, ref_name = ref_name, test_name
        pre_df, post_df = post_df, pre_df
        ws_col = ws_col.replace("ref", "test")
        pw_col = pw_col.replace("test", "ref")
    cutout_ws = wtg_for_turbine_type.turbine_type.cutout_ws_mps
    rated_power = wtg_for_turbine_type.turbine_type.rated_power_kw

    ws_bin_edges = np.arange(0, cutout_ws + cfg.ws_bin_width, cfg.ws_bin_width)

    pre_pp_df = _pp_raw_df_pandas(
        pre_df, "pre", ws_col=ws_col, ws_bin_edges=ws_bin_edges, pw_col=pw_col, timebase_s=cfg.timebase_s
    )
    post_pp_df = _pp_raw_df_pandas(
        post_df, "post", ws_col=ws_col, ws_bin_edges=ws_bin_edges, pw_col=pw_col, timebase_s=cfg.timebase_s
    )

    pre_pp_df = _cook_pp_pandas(
        pp_df=pre_pp_df,
        pre_or_post="pre",
        ws_bin_width=cfg.ws_bin_width,
        rated_power=rated_power,
        clip_to_rated=cfg.clip_rated_power_pp,
    )
    post_pp_df = _cook_pp_pandas(
        pp_df=post_pp_df,
        pre_or_post="post",
        ws_bin_width=cfg.ws_bin_width,
        rated_power=rated_power,
        clip_to_rated=cfg.clip_rated_power_pp,
    )
    pp_df = pre_pp_df.merge(
        post_pp_df[[x for x in post_pp_df.columns if x not in pre_pp_df.columns]],
        on=pre_pp_df.index.name,
        how="inner",
    )
    pp_df["is_invalid_bin"] = (~pp_df["pre_valid"]) | (~pp_df["post_valid"])
    if lt_df is not None:
        pp_df = pp_df.merge(lt_df[["hours_per_year", "bin_mid"]], on="bin_mid", how="left")
        pp_df = pp_df.set_index("bin_mid", drop=False, verify_integrity=True)
        pp_df.index.name = f"{ws_col}_bin_mid"
        pp_df["hours_for_mwh_calc"] = pp_df["hours_per_year"].fillna(0)
    else:
        pp_df["hours_for_mwh_calc"] = pp_df["hours_post_raw"].fillna(0)
    pp_df["f"] = pp_df["hours_for_mwh_calc"] / pp_df["hours_for_mwh_calc"].sum()

    if lt_df is not None:
        pp_df_invalid = pp_df[pp_df["is_invalid_bin"]]
        mwh_invalid_bins = (
            pp_df_invalid["hours_for_mwh_calc"].sum()
            * (pp_df_invalid["f"] * pp_df_invalid["pw_at_mid_pre"]).sum()
            / 1000
        )
        pre_mwh = pp_df["hours_for_mwh_calc"].sum() * (pp_df["f"] * pp_df["pw_at_mid_pre"]).sum() / 1000
        missing_bins_unc_scale_factor = 1 / (1 - mwh_invalid_bins / pre_mwh)
    else:
        missing_bins_unc_scale_factor = 1

    pp_daterange = (
        cfg.analysis_last_dt_utc_start + pd.Timedelta(seconds=cfg.timebase_s)
    ) - cfg.analysis_first_dt_utc_start
    pp_possible_hours = pp_daterange.total_seconds() / 3600

    pp_df["pw_at_mid_expected"] = pp_df["pw_at_mid_post"]
    pp_df["pw_sem_at_mid_expected"] = pp_df["pw_sem_at_mid_post"]
    use_pre_for_expected = ~pp_df["is_invalid_bin"]
    if cfg.use_rated_invalid_bins:
        rated_ws = _calc_rated_ws_pandas(pp_df=pp_df, pw_col="pw_at_mid_pre", rated_power=rated_power)
        use_pre_for_expected = use_pre_for_expected | (pp_df["bin_mid"] >= rated_ws)
    pp_df.loc[use_pre_for_expected, "pw_at_mid_expected"] = pp_df.loc[use_pre_for_expected, "pw_at_mid_pre"]
    pp_df.loc[use_pre_for_expected, "pw_sem_at_mid_expected"] = pp_df.loc[
        use_pre_for_expected,
        "pw_sem_at_mid_pre",
    ]

    expected_post_mwh = (pp_df["pw_at_mid_expected"] * pp_df["hours_for_mwh_calc"]).sum() / 1000
    expected_post_se_mwh = (
        pp_df["hours_for_mwh_calc"].sum() * np.sqrt(((pp_df["f"] * pp_df["pw_sem_at_mid_expected"]) ** 2).sum()) / 1000
    )

    post_mwh = (pp_df["pw_at_mid_post"] * pp_df["hours_for_mwh_calc"]).sum() / 1000
    post_se_mwh = (
        pp_df["hours_for_mwh_calc"].sum() * np.sqrt(((pp_df["f"] * pp_df["pw_sem_at_mid_post"]) ** 2).sum()) / 1000
    )

    uplift_mwh = post_mwh - expected_post_mwh

    uplift_se_mwh = np.sqrt(expected_post_se_mwh**2 + post_se_mwh**2)

    valid_count = round(
        np.minimum(pp_df["hours_pre"].sum(), pp_df["hours_post"].sum()) * 3600 / cfg.timebase_s,
    )
    p_low = (1 - confidence_level) / 2
    p_high = 1 - p_low
    t_value_one_sigma = t.ppf(
        norm.cdf(1),
        valid_count - 1,
    )
    t_values = t.ppf(
        p_high,
        np.minimum(pp_df["hours_pre"].clip(lower=2), pp_df["hours_post"].clip(lower=2)) * 3600 / cfg.timebase_s - 1,
    )
    unc_one_sigma_mwh = uplift_se_mwh * t_value_one_sigma

    pp_df = _add_uplift_cols_to_pp_df_pandas(pp_df, p_low=p_low, p_high=p_high, t_values=t_values)

    pp_valid_hours_pre = pp_df["hours_pre"].sum()
    pp_valid_hours_post = pp_df["hours_post"].sum()
    pp_valid_hours = pp_valid_hours_pre + pp_valid_hours_post

    pp_results = {
        "time_calculated": pd.Timestamp.now("UTC"),
        "uplift_frc": uplift_mwh / expected_post_mwh,
        "unc_one_sigma_frc": unc_one_sigma_mwh / expected_post_mwh * missing_bins_unc_scale_factor,
        "t_value_one_sigma": t_value_one_sigma,
        "missing_bins_unc_scale_factor": missing_bins_unc_scale_factor,
        "pp_valid_hours_pre": pp_valid_hours_pre,
        "pp_valid_hours_post": pp_valid_hours_post,
        "pp_valid_hours": pp_valid_hours,
        "pp_data_coverage": pp_valid_hours / pp_possible_hours,
        "pp_invalid_bin_count": pp_df["is_invalid_bin"].sum(),
    }

    if plot_cfg is not None:
        plot_pre_post_pp_analysis(
            test_name=test_name,
            ref_name=ref_name,
            pp_df=pp_df,
            pre_df=pre_df,
            post_df=post_df,
            ws_col=ws_col,
            pw_col=pw_col,
            wd_col=wd_col,
            timebase_s=cfg.timebase_s,
            plot_cfg=plot_cfg,
            confidence_level=confidence_level,
        )

        if test_df is not None:
            test_df_pp_period = test_df[
                cfg.analysis_first_dt_utc_start : cfg.analysis_last_dt_utc_start  # type: ignore[misc]
            ]
            plot_pp_data_coverage(
                test_name=test_name,
                ref_name=ref_name,
                pp_df=pp_df,
                test_df_pp_period=test_df_pp_period,
                ws_bin_width=cfg.ws_bin_width,
                timebase_s=cfg.timebase_s,
                plot_cfg=plot_cfg,
            )

    return pp_results, pp_df


def _pre_post_pp_analysis_polars(
    *,
    cfg: WindUpConfig,
    test_wtg: Turbine,
    ref_name: str,
    lt_df: pl.DataFrame | None,
    pre_df: pl.DataFrame,
    post_df: pl.DataFrame,
    ws_col: str,
    pw_col: str,
    wd_col: str,
    plot_cfg: PlotConfig | None,
    confidence_level: float = 0.9,
    test_df: pl.DataFrame | None = None,
    reverse: bool = False,
) -> tuple[dict, pl.DataFrame]:
    wtg_for_turbine_type = test_wtg
    test_name = test_wtg.name
    if reverse:
        with contextlib.suppress(StopIteration):
            wtg_for_turbine_type = next(x for x in cfg.asset.wtgs if x.name == ref_name)
        test_name, ref_name = ref_name, test_name
        pre_df, post_df = post_df, pre_df
        ws_col = ws_col.replace("ref", "test")
        pw_col = pw_col.replace("test", "ref")
    cutout_ws = wtg_for_turbine_type.turbine_type.cutout_ws_mps
    rated_power = wtg_for_turbine_type.turbine_type.rated_power_kw

    ws_bin_edges = np.arange(0, cutout_ws + cfg.ws_bin_width, cfg.ws_bin_width)

    pre_pp_df = _pp_raw_df_polars(
        pre_df, "pre", ws_col=ws_col, ws_bin_edges=ws_bin_edges, pw_col=pw_col, timebase_s=cfg.timebase_s
    )
    post_pp_df = _pp_raw_df_polars(
        post_df, "post", ws_col=ws_col, ws_bin_edges=ws_bin_edges, pw_col=pw_col, timebase_s=cfg.timebase_s
    )

    pre_pp_df = _cook_pp_polars(
        pp_df=pre_pp_df,
        pre_or_post="pre",
        ws_bin_width=cfg.ws_bin_width,
        rated_power=rated_power,
        clip_to_rated=cfg.clip_rated_power_pp,
    )
    post_pp_df = _cook_pp_polars(
        pp_df=post_pp_df,
        pre_or_post="post",
        ws_bin_width=cfg.ws_bin_width,
        rated_power=rated_power,
        clip_to_rated=cfg.clip_rated_power_pp,
    )

    index_col = f"{ws_col}_bin_mid"

    pp_df = (
        pre_pp_df.lazy()
        .join(
            post_pp_df.lazy().select([x for x in post_pp_df.columns if x not in pre_pp_df.columns] + [index_col]),
            on=index_col,
            how="inner",
        )
        .with_columns(is_invalid_bin=(~pl.col("pre_valid")) | (~pl.col("post_valid")))
        .collect()
    )

    if lt_df is not None:
        pp_df = pp_df.join(lt_df.select(["hours_per_year", "bin_mid"]), on="bin_mid", how="left").with_columns(
            hours_for_mwh_calc=pl.col("hours_per_year").fill_null(0)
        )
    else:
        pp_df = pp_df.with_columns(hours_for_mwh_calc=pl.col("hours_post_raw").fill_null(0))

    pp_df = pp_df.with_columns(f=pl.col("hours_for_mwh_calc") / pl.col("hours_for_mwh_calc").sum())

    if lt_df is not None:
        pp_df_invalid = pp_df.filter(pl.col("is_invalid_bin"))
        mwh_invalid_bins = (
            pp_df_invalid["hours_for_mwh_calc"].sum()
            * (pp_df_invalid.select((pl.col("f") * pl.col("pw_at_mid_pre")).sum())).item()
            / 1000
        )
        pre_mwh = (
            pp_df["hours_for_mwh_calc"].sum()
            * (pp_df.select((pl.col("f") * pl.col("pw_at_mid_pre")).sum())).item()
            / 1000
        )
        missing_bins_unc_scale_factor = 1 / (1 - mwh_invalid_bins / pre_mwh)
    else:
        missing_bins_unc_scale_factor = 1

    pp_daterange = (
        cfg.analysis_last_dt_utc_start + pd.Timedelta(seconds=cfg.timebase_s)
    ) - cfg.analysis_first_dt_utc_start
    pp_possible_hours = pp_daterange.total_seconds() / 3600

    # Initialize expected columns with post values, then conditionally use pre values

    # Build the condition as a Polars expression
    use_pre_for_expected = ~pl.col("is_invalid_bin")
    if cfg.use_rated_invalid_bins:
        rated_ws = _calc_rated_ws_polars(pp_df=pp_df, pw_col="pw_at_mid_pre", rated_power=rated_power)
        use_pre_for_expected = use_pre_for_expected | (pl.col("bin_mid") >= rated_ws)

    pp_df = pp_df.with_columns(
        [
            pl.when(use_pre_for_expected)
            .then(pl.col("pw_at_mid_pre"))
            .otherwise(pl.col("pw_at_mid_post"))
            .alias("pw_at_mid_expected"),
            pl.when(use_pre_for_expected)
            .then(pl.col("pw_sem_at_mid_pre"))
            .otherwise(pl.col("pw_sem_at_mid_post"))
            .alias("pw_sem_at_mid_expected"),
        ]
    )

    expected_post_mwh = pp_df.select((pl.col("pw_at_mid_expected") * pl.col("hours_for_mwh_calc")).sum()).item() / 1000

    expected_post_se_mwh = (
        pp_df["hours_for_mwh_calc"].sum()
        * np.sqrt(pp_df.select(((pl.col("f") * pl.col("pw_sem_at_mid_expected")) ** 2).sum()).item())
        / 1000
    )

    post_mwh = pp_df.select((pl.col("pw_at_mid_post") * pl.col("hours_for_mwh_calc")).sum()).item() / 1000
    post_se_mwh = (
        pp_df["hours_for_mwh_calc"].sum()
        * np.sqrt(pp_df.select(((pl.col("f") * pl.col("pw_sem_at_mid_post")) ** 2).sum()).item())
        / 1000
    )

    uplift_mwh = post_mwh - expected_post_mwh
    uplift_se_mwh = np.sqrt(expected_post_se_mwh**2 + post_se_mwh**2)

    valid_count = round(
        min(pp_df["hours_pre"].sum(), pp_df["hours_post"].sum()) * 3600 / cfg.timebase_s,
    )
    p_low = (1 - confidence_level) / 2
    p_high = 1 - p_low
    t_value_one_sigma = t.ppf(norm.cdf(1), valid_count - 1)

    # For t_values with element-wise minimum
    t_values = t.ppf(
        p_high,
        pp_df.select(
            pl.min_horizontal(pl.col("hours_pre").clip(lower_bound=2), pl.col("hours_post").clip(lower_bound=2))
            * 3600
            / cfg.timebase_s
            - 1
        )
        .to_numpy()
        .flatten(),
    )

    unc_one_sigma_mwh = uplift_se_mwh * t_value_one_sigma

    pp_df = _add_uplift_cols_to_pp_df_polars(pp_df, p_low=p_low, p_high=p_high, t_values=t_values)

    pp_valid_hours_pre = pp_df["hours_pre"].sum()
    pp_valid_hours_post = pp_df["hours_post"].sum()
    pp_valid_hours = pp_valid_hours_pre + pp_valid_hours_post

    pp_results = {
        "time_calculated": pd.Timestamp.now("UTC"),
        "uplift_frc": uplift_mwh / expected_post_mwh,
        "unc_one_sigma_frc": unc_one_sigma_mwh / expected_post_mwh * missing_bins_unc_scale_factor,
        "t_value_one_sigma": t_value_one_sigma,
        "missing_bins_unc_scale_factor": missing_bins_unc_scale_factor,
        "pp_valid_hours_pre": pp_valid_hours_pre,
        "pp_valid_hours_post": pp_valid_hours_post,
        "pp_valid_hours": pp_valid_hours,
        "pp_data_coverage": pp_valid_hours / pp_possible_hours,
        "pp_invalid_bin_count": pp_df["is_invalid_bin"].sum(),
    }

    if plot_cfg is not None:
        plot_pre_post_pp_analysis(
            test_name=test_name,
            ref_name=ref_name,
            pp_df=pp_df,
            pre_df=pre_df,
            post_df=post_df,
            ws_col=ws_col,
            pw_col=pw_col,
            wd_col=wd_col,
            timebase_s=cfg.timebase_s,
            plot_cfg=plot_cfg,
            confidence_level=confidence_level,
        )

        if test_df is not None:
            test_df_pp_period = test_df.filter(
                (pl.col(TIMESTAMP_COL) >= cfg.analysis_first_dt_utc_start)
                & (pl.col(TIMESTAMP_COL) <= cfg.analysis_last_dt_utc_start)
            )
            plot_pp_data_coverage(
                test_name=test_name,
                ref_name=ref_name,
                pp_df=pp_df.to_pandas().set_index(index_col),
                test_df_pp_period=test_df_pp_period.to_pandas().set_index(TIMESTAMP_COL),
                ws_bin_width=cfg.ws_bin_width,
                timebase_s=cfg.timebase_s,
                plot_cfg=plot_cfg,
            )

    return pp_results, pp_df


def _calc_power_only_and_reversed_uplifts_pandas(
    *,
    cfg: WindUpConfig,
    test_wtg: Turbine,
    ref_name: str,
    lt_df: pd.DataFrame,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    pw_col: str,
    wd_col: str,
    confidence_level: float = 0.9,
) -> tuple[float, float]:
    # calculate power only forward result
    pre_power_only = pre_df.copy()
    pre_power_only["ref_ws_power_only_detrended"] = (
        pre_power_only["ref_ws_est_from_power_only"] * pre_power_only["ws_rom"]
    )
    post_power_only = post_df.copy()
    post_power_only["ref_ws_power_only_detrended"] = (
        post_power_only["ref_ws_est_from_power_only"] * post_power_only["ws_rom"]
    )
    power_only_results, _ = _pre_post_pp_analysis_pandas(
        cfg=cfg,
        test_wtg=test_wtg,
        ref_name=ref_name,
        lt_df=lt_df,
        pre_df=pre_power_only,
        post_df=post_power_only,
        ws_col="ref_ws_power_only_detrended",
        pw_col=pw_col,
        wd_col=wd_col,
        plot_cfg=None,
        confidence_level=confidence_level,
    )

    # need to predict the reference wind speed using the test wind speed for reverse analysis
    pre_power_only["test_ws_power_only_detrended"] = (
        pre_power_only["test_ws_est_from_power_only"] / pre_power_only["ws_rom"]
    )
    post_power_only["test_ws_power_only_detrended"] = (
        post_power_only["test_ws_est_from_power_only"] / post_power_only["ws_rom"]
    )
    reversed_results, _ = _pre_post_pp_analysis_pandas(
        cfg=cfg,
        test_wtg=test_wtg,
        ref_name=ref_name,
        lt_df=lt_df,
        pre_df=pre_power_only,
        post_df=post_power_only,
        ws_col="ref_ws_power_only_detrended",
        pw_col=pw_col,
        wd_col=wd_col,
        plot_cfg=None,
        confidence_level=confidence_level,
        reverse=True,
    )

    poweronly_uplift_frc = power_only_results["uplift_frc"]
    reversed_uplift_frc = reversed_results["uplift_frc"]
    return poweronly_uplift_frc, reversed_uplift_frc


def _calc_power_only_and_reversed_uplifts_polars(
    *,
    cfg: WindUpConfig,
    test_wtg: Turbine,
    ref_name: str,
    lt_df: pl.DataFrame | None,
    pre_df: pl.DataFrame,
    post_df: pl.DataFrame,
    pw_col: str,
    wd_col: str,
    confidence_level: float = 0.9,
) -> tuple[float, float]:
    # calculate power only forward result
    pre_power_only = pre_df.with_columns(
        (pl.col("ref_ws_est_from_power_only") * pl.col("ws_rom")).alias("ref_ws_power_only_detrended")
    )

    post_power_only = post_df.with_columns(
        (pl.col("ref_ws_est_from_power_only") * pl.col("ws_rom")).alias("ref_ws_power_only_detrended")
    )

    power_only_results, _ = _pre_post_pp_analysis_polars(
        cfg=cfg,
        test_wtg=test_wtg,
        ref_name=ref_name,
        lt_df=lt_df,
        pre_df=pre_power_only,
        post_df=post_power_only,
        ws_col="ref_ws_power_only_detrended",
        pw_col=pw_col,
        wd_col=wd_col,
        plot_cfg=None,
        confidence_level=confidence_level,
    )

    # need to predict the reference wind speed using the test wind speed for reverse analysis
    pre_power_only = pre_power_only.with_columns(
        (pl.col("test_ws_est_from_power_only") / pl.col("ws_rom")).alias("test_ws_power_only_detrended")
    )

    post_power_only = post_power_only.with_columns(
        (pl.col("test_ws_est_from_power_only") / pl.col("ws_rom")).alias("test_ws_power_only_detrended")
    )

    reversed_results, _ = _pre_post_pp_analysis_polars(
        cfg=cfg,
        test_wtg=test_wtg,
        ref_name=ref_name,
        lt_df=lt_df,
        pre_df=pre_power_only,
        post_df=post_power_only,
        ws_col="ref_ws_power_only_detrended",
        pw_col=pw_col,
        wd_col=wd_col,
        plot_cfg=None,
        confidence_level=confidence_level,
        reverse=True,
    )

    return power_only_results["uplift_frc"], reversed_results["uplift_frc"]


def _pre_post_pp_analysis_with_reversal_pandas(
    *,
    cfg: WindUpConfig,
    test_wtg: Turbine,
    ref_name: str,
    lt_df: pd.DataFrame | None,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    ws_col: str,
    pw_col: str,
    wd_col: str,
    plot_cfg: PlotConfig | None,
    confidence_level: float = 0.9,
    test_df: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    pp_results, pp_df = _pre_post_pp_analysis_pandas(
        cfg=cfg,
        test_wtg=test_wtg,
        ref_name=ref_name,
        lt_df=lt_df,
        pre_df=pre_df,
        post_df=post_df,
        ws_col=ws_col,
        pw_col=pw_col,
        wd_col=wd_col,
        plot_cfg=plot_cfg,
        confidence_level=confidence_level,
        test_df=test_df,
    )

    if test_wtg.name == ref_name:
        poweronly_uplift_frc = np.nan
        reversed_uplift_frc = np.nan
        reversal_error = 0.0
    else:
        poweronly_uplift_frc, reversed_uplift_frc = _calc_power_only_and_reversed_uplifts_pandas(
            cfg=cfg,
            test_wtg=test_wtg,
            ref_name=ref_name,
            lt_df=lt_df,
            pre_df=pre_df,
            post_df=post_df,
            pw_col=pw_col,
            wd_col=wd_col,
            confidence_level=confidence_level,
        )
        reversal_error = reversed_uplift_frc - poweronly_uplift_frc
    if plot_cfg is not None:
        logger.info(f"\nresults for test={test_wtg.name} ref={ref_name}:\n")
        logger.info(f"hours pre = {pp_results['pp_valid_hours_pre']:.1f}")
        logger.info(f"hours post = {pp_results['pp_valid_hours_post']:.1f}")
        logger.info(f"\nuplift estimate before adjustments = {100 * pp_results['uplift_frc']:.1f} %")

        logger.info(f"\npower only uplift estimate = {100 * poweronly_uplift_frc:.1f} %")
        logger.info(f"reversed (power only) uplift estimate = {100 * reversed_uplift_frc:.1f} %\n")

    pp_results["uplift_noadj_frc"] = pp_results["uplift_frc"]
    pp_results["unc_one_sigma_noadj_frc"] = pp_results["unc_one_sigma_frc"]

    pp_results["poweronly_uplift_frc"] = poweronly_uplift_frc
    pp_results["reversed_uplift_frc"] = reversed_uplift_frc
    pp_results["reversal_error"] = reversal_error
    pp_results["uplift_frc"] = pp_results["uplift_noadj_frc"] + reversal_error / 2
    pp_results["unc_one_sigma_lowerbound_frc"] = abs(reversal_error) / 2
    pp_results["unc_one_sigma_frc"] = max(
        pp_results["unc_one_sigma_noadj_frc"],
        pp_results["unc_one_sigma_lowerbound_frc"],
    )

    return pp_results, pp_df


def _pre_post_pp_analysis_with_reversal_polars(
    *,
    cfg: WindUpConfig,
    test_wtg: Turbine,
    ref_name: str,
    lt_df: pd.DataFrame | None,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    ws_col: str,
    pw_col: str,
    wd_col: str,
    plot_cfg: PlotConfig | None,
    confidence_level: float = 0.9,
    test_df: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    # convert dataframes from pandas to polars
    pre_df = pl.from_pandas(pre_df.reset_index())
    post_df = pl.from_pandas(post_df.reset_index())
    test_df = pl.from_pandas(test_df.reset_index()) if test_df is not None else None
    lt_df = pl.from_pandas(lt_df.reset_index()) if lt_df is not None else None

    pp_results, pp_df = _pre_post_pp_analysis_polars(
        cfg=cfg,
        test_wtg=test_wtg,
        ref_name=ref_name,
        lt_df=lt_df,
        pre_df=pre_df,
        post_df=post_df,
        ws_col=ws_col,
        pw_col=pw_col,
        wd_col=wd_col,
        plot_cfg=plot_cfg,
        confidence_level=confidence_level,
        test_df=test_df,
    )

    if test_wtg.name == ref_name:
        poweronly_uplift_frc = np.nan
        reversed_uplift_frc = np.nan
        reversal_error = 0.0
    else:
        poweronly_uplift_frc, reversed_uplift_frc = _calc_power_only_and_reversed_uplifts_polars(
            cfg=cfg,
            test_wtg=test_wtg,
            ref_name=ref_name,
            lt_df=lt_df,
            pre_df=pre_df,
            post_df=post_df,
            pw_col=pw_col,
            wd_col=wd_col,
            confidence_level=confidence_level,
        )
        reversal_error = reversed_uplift_frc - poweronly_uplift_frc
    if plot_cfg is not None:
        logger.info(f"\nresults for test={test_wtg.name} ref={ref_name}:\n")
        logger.info(f"hours pre = {pp_results['pp_valid_hours_pre']:.1f}")
        logger.info(f"hours post = {pp_results['pp_valid_hours_post']:.1f}")
        logger.info(f"\nuplift estimate before adjustments = {100 * pp_results['uplift_frc']:.1f} %")

        logger.info(f"\npower only uplift estimate = {100 * poweronly_uplift_frc:.1f} %")
        logger.info(f"reversed (power only) uplift estimate = {100 * reversed_uplift_frc:.1f} %\n")

    pp_results["uplift_noadj_frc"] = pp_results["uplift_frc"]
    pp_results["unc_one_sigma_noadj_frc"] = pp_results["unc_one_sigma_frc"]

    pp_results["poweronly_uplift_frc"] = poweronly_uplift_frc
    pp_results["reversed_uplift_frc"] = reversed_uplift_frc
    pp_results["reversal_error"] = reversal_error
    pp_results["uplift_frc"] = pp_results["uplift_noadj_frc"] + reversal_error / 2
    pp_results["unc_one_sigma_lowerbound_frc"] = abs(reversal_error) / 2
    pp_results["unc_one_sigma_frc"] = max(
        pp_results["unc_one_sigma_noadj_frc"],
        pp_results["unc_one_sigma_lowerbound_frc"],
    )

    return pp_results, pp_df.to_pandas().set_index(f"{ws_col}_bin_mid")


def pre_post_pp_analysis_with_reversal_and_bootstrapping(
    *,
    cfg: WindUpConfig,
    test_wtg: Turbine,
    ref_name: str,
    lt_df: pd.DataFrame | None,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    ws_col: str,
    pw_col: str,
    wd_col: str,
    plot_cfg: PlotConfig | None,
    random_seed: int,
    confidence_level: float = 0.9,
    test_df: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Perform pre-post analysis with reversal and block bootstrapping uncertainty analysis.

    :param cfg: WindUpConfig object
    :param test_wtg: Turbine object for the test turbine
    :param ref_name: name of the reference turbine
    :param lt_df: long term data DataFrame
    :param pre_df: pre period DataFrame
    :param post_df: post period DataFrame
    :param ws_col: wind speed column name
    :param pw_col: power column name
    :param wd_col: wind direction column name
    :param plot_cfg: PlotConfig object
    :param random_seed: random seed for reproducibility
    :param confidence_level: confidence level
    :param test_df: test data DataFrame
    :return: tuple of results dictionary and DataFrame
    """
    pp_results, pp_df = _pre_post_pp_analysis_with_reversal_polars(
        cfg=cfg,
        test_wtg=test_wtg,
        ref_name=ref_name,
        lt_df=lt_df,
        pre_df=pre_df,
        post_df=post_df,
        ws_col=ws_col,
        pw_col=pw_col,
        wd_col=wd_col,
        plot_cfg=plot_cfg,
        test_df=test_df,
    )

    pre_df_dropna = pre_df.dropna(subset=[ws_col, pw_col, wd_col])
    post_df_dropna = post_df.dropna(subset=[ws_col, pw_col, wd_col])

    n_samples = cfg.bootstrap_runs_override if cfg.bootstrap_runs_override else round(40 * (1 / (1 - confidence_level)))
    if plot_cfg is not None:
        logger.info(f"Running block bootstrapping uncertainty analysis n_samples = {n_samples}")
    bootstrapped_uplifts = np.empty(n_samples)
    bootstrapped_uplifts[:] = np.nan
    rng = np.random.default_rng(seed=random_seed)
    for n in tqdm(range(n_samples)):
        num_blocks = rng.choice([9, 10, 11])
        block_size_pre = math.floor(len(pre_df_dropna) / num_blocks)
        block_size_post = math.floor(len(post_df_dropna) / num_blocks)
        # randomly remove rows to make the dataframes the perfect length
        pre_target_len = num_blocks * block_size_pre
        post_target_len = num_blocks * block_size_post
        pre_rows_to_keep = rng.choice(pre_df_dropna.index, size=pre_target_len, replace=False)
        post_rows_to_keep = rng.choice(post_df_dropna.index, size=post_target_len, replace=False)
        pre_df_trim = pre_df_dropna.loc[pre_rows_to_keep].sort_index()
        post_df_trim = post_df_dropna.loc[post_rows_to_keep].sort_index()

        blocks = rng.choice(num_blocks, size=num_blocks, replace=True)

        pre_sample_ilocs = (block_size_pre * blocks[:, None] + np.arange(block_size_pre)[None, :]).flatten()
        pre_df_ = pre_df_trim.iloc[pre_sample_ilocs].sort_index()

        post_sample_ilocs = (block_size_post * blocks[:, None] + np.arange(block_size_post)[None, :]).flatten()
        post_df_ = post_df_trim.iloc[post_sample_ilocs].sort_index()

        try:
            sample_results, _ = _pre_post_pp_analysis_with_reversal_polars(
                cfg=cfg,
                test_wtg=test_wtg,
                ref_name=ref_name,
                lt_df=lt_df,
                pre_df=pre_df_.reset_index(),
                post_df=post_df_.reset_index(),
                ws_col=ws_col,
                pw_col=pw_col,
                wd_col=wd_col,
                plot_cfg=None,
            )
            bootstrapped_uplifts[n] = sample_results["uplift_frc"]
        except RuntimeError:
            result_manager.warning(f"RuntimeError on sample {n}")
            bootstrapped_uplifts[n] = np.nan

    if np.isnan(bootstrapped_uplifts).sum() < 0.5 * len(bootstrapped_uplifts):
        median = float(np.nanmedian(bootstrapped_uplifts))
        lower = float(np.nanpercentile(bootstrapped_uplifts, 100 * (1 - confidence_level) / 2))
        upper = float(np.nanpercentile(bootstrapped_uplifts, 100 * (1 - (1 - confidence_level) / 2)))
        unc_one_sigma = (upper - lower) / 2 / norm.ppf((1 + confidence_level) / 2)
    else:
        median = np.nan
        lower = np.nan
        upper = np.nan
        unc_one_sigma = np.nan

    if plot_cfg is not None:
        msg = (
            f"block bootstrapping uncertainty analysis results (conf={100 * confidence_level:.0f}%):"
            f"\n  median = {100 * median:.1f} %"
            f"\n  lower = {100 * lower:.1f} %"
            f"\n  upper = {100 * upper:.1f} %"
            f"\n  unc_one_sigma = {100 * unc_one_sigma:.1f} %"
        )
        logger.info(msg)

    pp_results["unc_one_sigma_bootstrap_frc"] = unc_one_sigma
    pp_results["unc_one_sigma_frc"] = max(
        pp_results["unc_one_sigma_frc"],
        pp_results["unc_one_sigma_bootstrap_frc"],
    )

    p_low = (1 - confidence_level) / 2
    p_high = 1 - p_low

    pp_results[f"uplift_p{p_low * 100:.0f}_frc"] = pp_results["uplift_frc"] + pp_results[
        "unc_one_sigma_frc"
    ] * norm.ppf((1 + confidence_level) / 2)
    pp_results[f"uplift_p{p_high * 100:.0f}_frc"] = pp_results["uplift_frc"] - pp_results[
        "unc_one_sigma_frc"
    ] * norm.ppf((1 + confidence_level) / 2)
    if plot_cfg is not None:
        logger.info(f"\ncat A 1 sigma unc = {100 * pp_results['unc_one_sigma_noadj_frc']:.1f} %")
        if pp_results["unc_one_sigma_lowerbound_frc"] > 0.05 / 100:
            logger.info(f"abs reversal error / 2 = {100 * pp_results['unc_one_sigma_lowerbound_frc']:.1f} %")
        else:
            logger.info(f"abs reversal error / 2 = {100 * pp_results['unc_one_sigma_lowerbound_frc']:.3f} %")
        logger.info(f"bootstrap 1 sigma unc = {100 * pp_results['unc_one_sigma_bootstrap_frc']:.1f} %")
        logger.info(f"missing bins scale factor = {pp_results['missing_bins_unc_scale_factor']:.3f}")
        logger.info(f"final 1 sigma unc = {100 * pp_results['unc_one_sigma_frc']:.1f} %\n")

        logger.info(f"final uplift estimate = {100 * pp_results['uplift_frc']:.1f} %")
        logger.info(f"final P95 uplift estimate = {100 * pp_results[f'uplift_p{p_high * 100:.0f}_frc']:.1f} %")
        logger.info(f"final P5 uplift estimate = {100 * pp_results[f'uplift_p{p_low * 100:.0f}_frc']:.1f} %")

    return pp_results, pp_df
