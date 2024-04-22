import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from wind_up.constants import PROJECTROOT_DIR, ROWS_PER_HOUR, TIMEBASE_PD_TIMEDELTA
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.plots.pp_analysis_plots import plot_pp_data_coverage, plot_pre_post_pp_analysis


def pp_raw_df(
    pre_or_post_df: pd.DataFrame,
    pre_or_post: str,
    *,
    ws_col: str,
    ws_bin_edges: np.ndarray,
    pw_col: str,
) -> pd.DataFrame:
    pp_df = (
        pre_or_post_df.dropna(subset=[pw_col, ws_col])
        .groupby(
            by=pd.cut(pre_or_post_df[ws_col], bins=ws_bin_edges, retbins=False),
            observed=False,
        )
        .agg(
            count=pd.NamedAgg(column=pw_col, aggfunc=lambda x: len(x)),
            ws_mean=pd.NamedAgg(column=ws_col, aggfunc=lambda x: x.mean()),
            ws_std=pd.NamedAgg(column=ws_col, aggfunc=lambda x: x.std()),
            pw_mean=pd.NamedAgg(column=pw_col, aggfunc=lambda x: x.mean()),
            pw_std=pd.NamedAgg(column=pw_col, aggfunc=lambda x: x.std()),
        )
    )
    pp_df["ws_std"] = pp_df["ws_std"].fillna(0)
    pp_df["pw_std"] = pp_df["pw_std"].fillna(0)
    pp_df["hours"] = pp_df["count"] / ROWS_PER_HOUR
    pp_df["ws_sem"] = pp_df["ws_std"] / np.sqrt(pp_df["count"].clip(lower=1))
    pp_df["pw_sem"] = pp_df["pw_std"] / np.sqrt(pp_df["count"].clip(lower=1))
    pp_df.columns = [x + f"_{pre_or_post}" for x in pp_df.columns]
    pp_df["bin_left"] = [x.left for x in pp_df.index]
    pp_df["bin_mid"] = [x.mid for x in pp_df.index]
    pp_df["bin_right"] = [x.right for x in pp_df.index]
    pp_df["bin_closed_right"] = [x.closed_right for x in pp_df.index]
    pp_df = pp_df.set_index("bin_mid", drop=False, verify_integrity=True)
    pp_df.index.name = f"{ws_col}_bin_mid"
    return pp_df


def cook_pp(pp_df: pd.DataFrame, pre_or_post: str, ws_bin_width: float, rated_power: float) -> pd.DataFrame:
    pp_df = pp_df.copy()

    raw_pw_col = f"pw_mean_{pre_or_post}_raw"
    pw_col = f"pw_mean_{pre_or_post}"
    pw_sem_col = f"pw_sem_{pre_or_post}"
    hours_col = f"hours_{pre_or_post}"
    ws_col = f"ws_mean_{pre_or_post}"
    pw_at_mid_col = f"pw_at_mid_{pre_or_post}"
    pw_sem_at_mid_col = f"pw_sem_at_mid_{pre_or_post}"

    pp_df[raw_pw_col] = pp_df[pw_col]

    # IEC minimum data count method
    hrs_per_mps = 1
    enough_data = pp_df[f"hours_{pre_or_post}"] > (ws_bin_width * hrs_per_mps)
    pp_df.loc[~enough_data, [pw_col, hours_col, ws_col, pw_sem_col]] = np.nan
    pp_df[hours_col] = pp_df[hours_col].fillna(0)
    pp_df[ws_col] = pp_df[ws_col].fillna(pp_df["bin_mid"])

    pp_df[pw_col] = pp_df[pw_col].clip(lower=0, upper=rated_power)

    # data which would have been at rated can be gap filled
    rated_ws = pp_df.loc[pp_df[raw_pw_col] >= rated_power * 0.995, "bin_mid"].min() + 1
    pp_df.loc[(pp_df["bin_mid"] >= rated_ws) & pp_df[pw_col].isna(), pw_col] = rated_power
    pp_df[pw_sem_col] = pp_df[pw_sem_col].ffill()

    # missing data at low wind speed can be filled with 0
    pp_df.loc[pp_df.index[pp_df[pw_col].isna().cummin()], [pw_col, pw_sem_col]] = 0

    # revisit in future, an alternative is filling in with SCADA power curve
    pp_df[pw_col] = pp_df[pw_col].interpolate()

    # interpolate power and ci to bin mid
    pp_df[pw_at_mid_col] = np.interp(pp_df["bin_mid"], pp_df[ws_col], pp_df[pw_col])
    pp_df[pw_at_mid_col] = pp_df[pw_at_mid_col].clip(lower=0, upper=rated_power)
    pp_df[pw_sem_at_mid_col] = pp_df[pw_sem_col] / pp_df[pw_col] * pp_df[pw_at_mid_col]
    pp_df[pw_sem_at_mid_col] = pp_df[pw_sem_at_mid_col].fillna(0)
    pp_df[pw_sem_at_mid_col] = pp_df[pw_sem_at_mid_col].clip(lower=0, upper=pp_df[pw_sem_col])

    if pp_df[[col for col in pp_df.columns if col is not raw_pw_col]].isna().any().any():
        msg = "pp_df has missing values"
        raise RuntimeError(msg)

    return pp_df


def pre_post_pp_analysis(
    *,
    cfg: WindUpConfig,
    test_name: str,
    ref_name: str,
    lt_df: pd.DataFrame,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    ws_col: str,
    pw_col: str,
    wd_col: str,
    plot_cfg: PlotConfig | None,
    confidence_level: float = 0.9,
    test_df: pd.DataFrame | None = None,
    use_ref_for_wtg_type: bool = False,
) -> tuple[dict, pd.DataFrame]:
    wtg_name = ref_name if use_ref_for_wtg_type else test_name
    cutout_ws = next(x.turbine_type.cutout_ws_mps for x in cfg.asset.wtgs if x.name == wtg_name)
    ws_bin_edges = np.arange(0, cutout_ws + cfg.ws_bin_width, cfg.ws_bin_width)

    pre_pp_df = pp_raw_df(pre_df, "pre", ws_col=ws_col, ws_bin_edges=ws_bin_edges, pw_col=pw_col)
    post_pp_df = pp_raw_df(post_df, "post", ws_col=ws_col, ws_bin_edges=ws_bin_edges, pw_col=pw_col)

    rated_power = next(x.turbine_type.rated_power_kw for x in cfg.asset.wtgs if x.name == wtg_name)
    pre_pp_df = cook_pp(pp_df=pre_pp_df, pre_or_post="pre", ws_bin_width=cfg.ws_bin_width, rated_power=rated_power)
    post_pp_df = cook_pp(pp_df=post_pp_df, pre_or_post="post", ws_bin_width=cfg.ws_bin_width, rated_power=rated_power)
    pp_df = pre_pp_df.merge(
        post_pp_df[[x for x in post_pp_df.columns if x not in pre_pp_df.columns]],
        on=pre_pp_df.index.name,
        how="inner",
    )
    pp_df = pp_df.merge(lt_df[["hours_per_year", "bin_mid"]], on="bin_mid", how="left")
    pp_df = pp_df.set_index("bin_mid", drop=False, verify_integrity=True)
    pp_df.index.name = f"{ws_col}_bin_mid"
    pp_df["hours_per_year"] = pp_df["hours_per_year"].fillna(0)
    pp_df["f"] = pp_df["hours_per_year"] / pp_df["hours_per_year"].sum()

    pp_df["uplift_kw"] = pp_df["pw_at_mid_post"] - pp_df["pw_at_mid_pre"]
    pp_df["uplift_se"] = np.sqrt(pp_df["pw_sem_at_mid_post"] ** 2 + pp_df["pw_sem_at_mid_pre"] ** 2)
    p_low = (1 - confidence_level) / 2
    p_high = 1 - p_low
    t_values = t.ppf(p_high, np.minimum(pp_df["count_pre"].clip(lower=2), pp_df["count_post"].clip(lower=2)) - 1)
    pp_df[f"uplift_p{p_low*100:.0f}_kw"] = pp_df["uplift_kw"] + pp_df["uplift_se"] * t_values
    pp_df[f"uplift_p{p_high*100:.0f}_kw"] = pp_df["uplift_kw"] - pp_df["uplift_se"] * t_values

    if len(pp_df) > 0:
        aep_pre_mwh = pp_df["hours_per_year"].sum() * (pp_df["f"] * pp_df["pw_at_mid_pre"]).sum() / 1000
        aep_pre_se_mwh = (
            pp_df["hours_per_year"].sum() * np.sqrt(((pp_df["f"] * pp_df["pw_sem_at_mid_pre"]) ** 2).sum()) / 1000
        )
        aep_post_mwh = pp_df["hours_per_year"].sum() * (pp_df["f"] * pp_df["pw_at_mid_post"]).sum() / 1000
        aep_post_se_mwh = (
            pp_df["hours_per_year"].sum() * np.sqrt(((pp_df["f"] * pp_df["pw_sem_at_mid_post"]) ** 2).sum()) / 1000
        )
        aep_uplift_mwh = aep_post_mwh - aep_pre_mwh
        aep_uplift_se_mwh = np.sqrt(aep_pre_se_mwh**2 + aep_post_se_mwh**2)
        t_value_one_sigma = t.ppf(norm.cdf(1), np.minimum(pp_df["count_pre"].sum(), pp_df["count_post"].sum()) - 1)
        t_value = t.ppf(p_high, np.minimum(pp_df["count_pre"].sum(), pp_df["count_post"].sum()) - 1)
        aep_unc_one_sigma_mwh = aep_uplift_se_mwh * t_value_one_sigma
        aep_uplift_se_mwh * t_value

        pp_df_invalid = pp_df[(pp_df["hours_pre"] == 0) | (pp_df["hours_post"] == 0)]
        aep_invalid_bins = (
            pp_df_invalid["hours_per_year"].sum() * (pp_df_invalid["f"] * pp_df_invalid["pw_at_mid_pre"]).sum() / 1000
        )
        if aep_invalid_bins < aep_pre_mwh:
            missing_bins_unc_scale_factor = 1 / (1 - aep_invalid_bins / aep_pre_mwh)
        else:
            missing_bins_unc_scale_factor = 1e6
    else:
        aep_uplift_mwh = np.nan

    pp_daterange = (cfg.analysis_last_dt_utc_start + TIMEBASE_PD_TIMEDELTA) - cfg.analysis_first_dt_utc_start
    pp_possible_hours = pp_daterange.total_seconds() / 3600
    pp_hours_pre = pp_df["hours_pre"].sum()
    pp_hours_post = pp_df["hours_post"].sum()
    pp_hours = pp_df["hours_pre"].sum() + pp_df["hours_post"].sum()
    pp_data_coverage = pp_hours / pp_possible_hours
    is_invalid_bin = (pp_df["hours_pre"] == 0) | (pp_df["hours_post"] == 0)
    pp_results = {
        "time_calculated": pd.Timestamp.now("UTC"),
        "aep_uplift_frc": aep_uplift_mwh / aep_pre_mwh,
        "aep_unc_one_sigma_frc": aep_unc_one_sigma_mwh / aep_pre_mwh * missing_bins_unc_scale_factor,
        "missing_bins_unc_scale_factor": missing_bins_unc_scale_factor,
        "t_value_one_sigma": t_value_one_sigma,
        f"t_value_conf{100 * confidence_level:.0f}": t_value,
        "pp_hours": pp_hours,
        "pp_hours_pre": pp_hours_pre,
        "pp_hours_post": pp_hours_post,
        "pp_data_coverage": pp_data_coverage,
        "pp_invalid_bin_count": is_invalid_bin.sum(),
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
                plot_cfg=plot_cfg,
            )

    return pp_results, pp_df


def calc_power_only_and_reversed_uplifts(
    *,
    cfg: WindUpConfig,
    test_name: str,
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
    power_only_results, _ = pre_post_pp_analysis(
        cfg=cfg,
        test_name=test_name,
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

    pre_power_only["test_ws_power_only_detrended"] = (
        pre_power_only["test_ws_est_from_power_only"] / pre_power_only["ws_rom"]
    )
    post_power_only["test_ws_power_only_detrended"] = (
        post_power_only["test_ws_est_from_power_only"] / post_power_only["ws_rom"]
    )
    reversed_results, _ = pre_post_pp_analysis(
        cfg=cfg,
        test_name=ref_name,  # swapped intentionally
        ref_name=test_name,
        lt_df=lt_df,
        pre_df=post_power_only,  # swapped intentionally
        post_df=pre_power_only,
        ws_col="test_ws_power_only_detrended",
        pw_col=pw_col.replace("test", "ref"),
        wd_col=wd_col,
        plot_cfg=None,
        confidence_level=confidence_level,
        use_ref_for_wtg_type=True,
    )

    poweronly_aep_uplift_frc = power_only_results["aep_uplift_frc"]
    reversed_aep_uplift_frc = reversed_results["aep_uplift_frc"]
    return poweronly_aep_uplift_frc, reversed_aep_uplift_frc


def pre_post_pp_analysis_with_reversal(
    *,
    cfg: WindUpConfig,
    test_name: str,
    ref_name: str,
    lt_df: pd.DataFrame,
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    ws_col: str,
    pw_col: str,
    wd_col: str,
    plot_cfg: PlotConfig | None,
    confidence_level: float = 0.9,
    test_df: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    pp_results, pp_df = pre_post_pp_analysis(
        cfg=cfg,
        test_name=test_name,
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

    if test_name == ref_name:
        poweronly_aep_uplift_frc = np.nan
        reversed_aep_uplift_frc = np.nan
        reversal_error = 0.0
    else:
        poweronly_aep_uplift_frc, reversed_aep_uplift_frc = calc_power_only_and_reversed_uplifts(
            cfg=cfg,
            test_name=test_name,
            ref_name=ref_name,
            lt_df=lt_df,
            pre_df=pre_df,
            post_df=post_df,
            pw_col=pw_col,
            wd_col=wd_col,
            confidence_level=confidence_level,
        )
        reversal_error = reversed_aep_uplift_frc - poweronly_aep_uplift_frc
    if plot_cfg is not None:
        print(f"\nAEP results for test={test_name} ref={ref_name}:\n")
        print(f"hours pre = {pp_results['pp_hours_pre']:.1f}")
        print(f"hours post = {pp_results['pp_hours_post']:.1f}")
        print(f"\nannual AEP uplift estimate before adjustments = {100*pp_results['aep_uplift_frc']:.1f} %")

        print(f"\npower only annual AEP uplift estimate = {100 * poweronly_aep_uplift_frc:.1f} %")
        print(f"reversed (power only) annual AEP uplift estimate = {100 * reversed_aep_uplift_frc:.1f} %\n")

    pp_results["aep_uplift_noadj_frc"] = pp_results["aep_uplift_frc"]
    pp_results["aep_unc_one_sigma_noadj_frc"] = pp_results["aep_unc_one_sigma_frc"]

    pp_results["poweronly_aep_uplift_frc"] = poweronly_aep_uplift_frc
    pp_results["reversed_aep_uplift_frc"] = reversed_aep_uplift_frc
    pp_results["reversal_error"] = reversal_error
    pp_results["aep_uplift_frc"] = pp_results["aep_uplift_noadj_frc"] + reversal_error / 2
    pp_results["aep_unc_one_sigma_lowerbound_frc"] = abs(reversal_error) / 2
    pp_results["aep_unc_one_sigma_frc"] = max(
        pp_results["aep_unc_one_sigma_noadj_frc"],
        pp_results["aep_unc_one_sigma_lowerbound_frc"],
    )

    return pp_results, pp_df


def pre_post_pp_analysis_with_reversal_and_bootstrapping(
    *,
    cfg: WindUpConfig,
    test_name: str,
    ref_name: str,
    lt_df: pd.DataFrame,
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
    pp_results, pp_df = pre_post_pp_analysis_with_reversal(
        cfg=cfg,
        test_name=test_name,
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

    pre_df_dropna = pre_df.dropna(subset=[ws_col, pw_col])
    post_df_dropna = post_df.dropna(subset=[ws_col, pw_col])

    n_samples = round(20 * (1 / (1 - confidence_level)))
    if plot_cfg is not None:
        print(f"Running block bootstrapping uncertainty analysis n_samples = {n_samples}")
    num_blocks = 10
    block_size_pre = math.floor(len(pre_df_dropna) / num_blocks)
    block_size_post = math.floor(len(post_df_dropna) / num_blocks)
    bootstrapped_uplifts = np.empty(n_samples)
    bootstrapped_uplifts[:] = np.nan
    rng = np.random.default_rng(seed=random_seed)
    for n in range(n_samples):
        if (n % (n_samples // 5)) == 0 and plot_cfg is not None:
            print(f"n = {n}")
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
            sample_results, _ = pre_post_pp_analysis_with_reversal(
                cfg=cfg,
                test_name=test_name,
                ref_name=ref_name,
                lt_df=lt_df,
                pre_df=pre_df_.reset_index(),
                post_df=post_df_.reset_index(),
                ws_col=ws_col,
                pw_col=pw_col,
                wd_col=wd_col,
                plot_cfg=None,
            )
            bootstrapped_uplifts[n] = sample_results["aep_uplift_frc"]
        except RuntimeError:
            print(f"WARNING: RuntimeError on sample {n}")
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
        print(f"\nblock bootstrapping uncertainty analysis results (conf={100*confidence_level:.0f}%):")
        print(f"  median = {100 * median:.1f} %")
        print(f"  lower = {100 * lower:.1f} %")
        print(f"  upper = {100 * upper:.1f} %")
        print(f"  unc_one_sigma = {100 * unc_one_sigma:.1f} %")

    pp_results["aep_unc_one_sigma_bootstrap_frc"] = unc_one_sigma
    pp_results["aep_unc_one_sigma_frc"] = max(
        pp_results["aep_unc_one_sigma_frc"],
        pp_results["aep_unc_one_sigma_bootstrap_frc"],
    )

    p_low = (1 - confidence_level) / 2
    p_high = 1 - p_low

    pp_results[f"aep_uplift_p{p_low * 100:.0f}_frc"] = pp_results["aep_uplift_frc"] + pp_results[
        "aep_unc_one_sigma_frc"
    ] * norm.ppf((1 + confidence_level) / 2)
    pp_results[f"aep_uplift_p{p_high * 100:.0f}_frc"] = pp_results["aep_uplift_frc"] - pp_results[
        "aep_unc_one_sigma_frc"
    ] * norm.ppf((1 + confidence_level) / 2)
    if plot_cfg is not None:
        print(f"\ncat A 1 sigma unc = {100 * pp_results['aep_unc_one_sigma_noadj_frc']:.1f} %")
        if pp_results["aep_unc_one_sigma_lowerbound_frc"] > 0.05 / 100:
            print(f"abs reversal error / 2 = {100 * pp_results['aep_unc_one_sigma_lowerbound_frc']:.1f} %")
        else:
            print(f"abs reversal error / 2 = {100 * pp_results['aep_unc_one_sigma_lowerbound_frc']:.3f} %")
        print(f"bootstrap 1 sigma unc = {100 * pp_results['aep_unc_one_sigma_bootstrap_frc']:.1f} %")
        print(f"missing bins scale factor = {pp_results['missing_bins_unc_scale_factor']:.3f}")
        print(f"final 1 sigma unc = {100 * pp_results['aep_unc_one_sigma_frc']:.1f} %\n")

        print(f"final annual AEP uplift estimate = {100*pp_results['aep_uplift_frc']:.1f} %")
        print(f"final P95 AEP uplift estimate = {100*pp_results[f'aep_uplift_p{p_high * 100:.0f}_frc']:.1f} %")
        print(f"final P5 AEP uplift estimate = {100*pp_results[f'aep_uplift_p{p_low * 100:.0f}_frc']:.1f} %")

    return pp_results, pp_df