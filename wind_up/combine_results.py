import datetime as dt
import itertools
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import norm

from wind_up.models import PlotConfig
from wind_up.plots.combine_results_plots import plot_combine_results

mpl.use("Agg")


def calc_sigma_ref(rdf: pd.DataFrame, ref_list: list[str]) -> float:
    # calculate the weighted absolute average reference uplift as a lower bound on uncertainty
    ref_uplifts = rdf.loc[(rdf["test_wtg"].isin(ref_list)), "p50_uplift"]
    ref_sigmas = rdf.loc[(rdf["test_wtg"].isin(ref_list)), "sigma_test"]
    weights = 1 / (ref_sigmas**2)
    return (abs(ref_uplifts) * weights).sum() / weights.sum() if weights.sum() > 0 else np.nan


def calc_tdf(trdf: pd.DataFrame, ref_list: list[str], weight_col: str = "unc_weight") -> pd.DataFrame:
    tdf = trdf.groupby("test_wtg").agg(
        p50_uplift=pd.NamedAgg(
            column="aep_uplift_frc",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        sigma_uncorr=pd.NamedAgg(
            column="aep_unc_one_sigma_frc",
            aggfunc=lambda x: np.sqrt(
                ((x * trdf.loc[x.index, weight_col] / trdf.loc[x.index, weight_col].sum()) ** 2).sum(),
            ),
        ),
        sigma_corr=pd.NamedAgg(
            column="aep_unc_one_sigma_frc",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        ref_count=pd.NamedAgg(column="aep_uplift_frc", aggfunc=len),
        is_ref=pd.NamedAgg(column="test_wtg", aggfunc=lambda x: x.isin(ref_list).any()),
    )
    tdf["sigma_test"] = (tdf["sigma_uncorr"] + tdf["sigma_corr"]) / 2
    tdf = tdf.sort_values(by=["ref_count", "test_wtg"], ascending=[False, True])
    tdf = tdf.reset_index()
    sigma_ref = calc_sigma_ref(tdf, ref_list)
    tdf["sigma_aep"] = tdf["sigma_test"].clip(lower=sigma_ref)
    tdf["p95_uplift"] = tdf["p50_uplift"] + norm.ppf(0.05) * tdf["sigma_aep"]
    tdf["p5_uplift"] = tdf["p50_uplift"] + norm.ppf(0.95) * tdf["sigma_aep"]
    return tdf


def choose_best_refs(trdf: pd.DataFrame, ref_list: list[str], min_refs: int = 3) -> list[str]:
    ref_count = len(ref_list)
    if ref_count < min_refs:
        msg = "ref_list must have at least min_refs elements"
        raise ValueError(msg)

    rdf = calc_tdf(trdf, ref_list)
    best_sigma_ref = calc_sigma_ref(rdf, ref_list)
    best_ref_list = ref_list.copy()
    ref_count -= 1
    while ref_count >= min_refs:
        for c in itertools.combinations(ref_list, ref_count):
            this_ref_list = list(c)
            this_sigma_ref = calc_sigma_ref(rdf, this_ref_list)
            if this_sigma_ref < best_sigma_ref:
                best_sigma_ref = this_sigma_ref
                best_ref_list = this_ref_list.copy()
        ref_count -= 1
    return best_ref_list


def combine_results(
    trdf: pd.DataFrame,
    *,
    auto_choose_refs: bool = True,
    exclude_refs: list[str] | None = None,
    plot_config: PlotConfig | None = None,
) -> pd.DataFrame:
    if exclude_refs is None:
        exclude_refs = []
    print("##############################################################################")
    print("#combine results per test turbine")
    print("##############################################################################")

    trdf = trdf.copy()

    if trdf.groupby(["test_wtg", "ref"]).size().max() > 1:
        msg = "trdf must have no more than one row per test-ref pair"
        raise ValueError(msg)

    # remove reference predictions of themselves
    trdf = trdf.loc[trdf["test_wtg"] != trdf["ref"], :]

    if len(exclude_refs) > 0:
        print(f"excluding refs {exclude_refs}")
        trdf = trdf.loc[~trdf["test_wtg"].isin(exclude_refs), :]
        trdf = trdf.loc[~trdf["ref"].isin(exclude_refs), :]

    if (trdf["aep_unc_one_sigma_frc"] <= 0).any() or trdf["aep_unc_one_sigma_frc"].isna().any():
        msg = "aep_unc_one_sigma_frc must be positive and non-NaN"
        raise ValueError(msg)

    weight_col = "unc_weight"
    trdf[weight_col] = 1 / (trdf["aep_unc_one_sigma_frc"] ** 2)

    ref_list = sorted(trdf["ref"].unique())

    min_refs = 3
    if auto_choose_refs:
        if len(ref_list) >= min_refs:
            best_ref_list = choose_best_refs(trdf, ref_list, min_refs=min_refs)
            refs_to_remove = [x for x in ref_list if x not in best_ref_list]
            trdf = trdf.loc[~trdf["test_wtg"].isin(refs_to_remove), :]
            trdf = trdf.loc[~trdf["ref"].isin(refs_to_remove), :]
            ref_list = sorted(trdf["ref"].unique())
        else:
            print(f"WARNING: len(ref_list) < {min_refs}, skipping auto_choose_refs")

    print(f"ref_list = {ref_list}")
    tdf = calc_tdf(trdf, ref_list, weight_col)

    # change column order for readability
    cols = list(tdf.columns)
    first_cols = ["test_wtg", "p50_uplift", "p95_uplift", "p5_uplift", "sigma_aep"]
    cols = first_cols + [x for x in cols if x not in first_cols]
    tdf = tdf[cols]

    if plot_config is not None:
        plot_combine_results(trdf=trdf, tdf=tdf, plot_cfg=plot_config)

    return tdf