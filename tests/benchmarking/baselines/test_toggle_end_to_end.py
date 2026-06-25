"""Slow end-to-end test: v0 and the naive ratio method on a HoT-derived synthetic toggle dataset.

Downloads the Hill of Towie v2 SCADA (Zenodo) and ERA5 reanalysis (Open-Meteo), injects a known
constant-Cp uplift under a fast 20-min-on/20-min-off toggle, and scores ``V0BinnedMethod`` (via
wind_up's native toggle assessment), ``NaiveRatioMethod`` and an oracle through the harness. This
is the sanity that the toggle path composes end to end; it is marked ``slow`` (network + a heavy
wind_up run) and skipped by ``-m "not slow"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.naive_ratio import NaiveRatioMethod
from benchmarking.baselines.v0_binned import V0BinnedMethod
from benchmarking.harness import StudyConfig, score_study
from benchmarking.harness.example_hot_study import OracleMethod
from benchmarking.synthetic import HOT_COLUMNS, ConstantCpChange
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada


@pytest.mark.slow
def test_naive_and_v0_recover_toggle_uplift(tmp_path) -> None:  # noqa: ANN001
    scada_df, _metadata_df = load_hot_scada(
        start_dt=pd.Timestamp("2016-01-01", tz="UTC"),
        end_dt_excl=pd.Timestamp("2018-09-01", tz="UTC"),
        wtg_numbers=[1, 3, 4, 7],
    )
    context = build_hot_v0_context(wtg_names=["T01", "T03", "T04", "T07"])
    study = StudyConfig(
        mode="toggle",
        turbine_subset=["T01", "T03", "T04", "T07"],
        treatment_start_range=(pd.Timestamp("2018-02-01", tz="UTC"), pd.Timestamp("2018-02-08", tz="UTC")),
        min_pre_months=24,
        campaign_months=[6],
        toggle_period=pd.Timedelta(minutes=40),
        n_replicates=1,
        seed=0,
    )

    results = score_study(
        scada_df,
        profile=[ConstantCpChange(delta=0.05)],
        methods=[
            V0BinnedMethod(context, scratch_dir=tmp_path / "windup_runs"),
            NaiveRatioMethod(active_power_col=HOT_COLUMNS.active_power, out_dir=tmp_path / "naive_runs"),
            OracleMethod(scada_df),
        ],
        study=study,
        profile_name="constant_cp_toggle",
    )

    oracle_row = results.loc[results["method"] == "oracle"].iloc[0]
    naive_row = results.loc[results["method"] == "naive_ratio"].iloc[0]
    v0_row = results.loc[results["method"] == "v0_binned"].iloc[0]

    # the harness toggle path is wired correctly: the oracle recovers the injected truth exactly
    assert abs(oracle_row["signed_error"]) < 1e-6
    # a real constant-Cp uplift is positive; both methods recover it to within a few pp.
    assert naive_row["truth"] > 0
    assert np.isfinite(naive_row["estimate"])
    assert abs(naive_row["signed_error"]) < 0.03
    assert np.isfinite(v0_row["estimate"])
    assert abs(v0_row["signed_error"]) < 0.03
