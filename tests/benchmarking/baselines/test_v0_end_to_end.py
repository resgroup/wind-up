"""Slow end-to-end test: a real v0 run on a HoT-derived synthetic constant-Cp dataset.

Downloads the Hill of Towie v2 SCADA (Zenodo) and ERA5 reanalysis (Open-Meteo), injects a
known constant-Cp uplift, runs the full wind_up pre/post analysis through ``V0BinnedMethod``
behind the harness, and checks the recovered P50 lands near the injected truth. This is the
sanity that the whole Issue 3 stack composes; it is marked ``slow`` (network + a heavy
wind_up run) and skipped by ``-m "not slow"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.v0_binned import V0BinnedMethod
from benchmarking.harness import StudyConfig, score_study
from benchmarking.harness.example_hot_study import OracleMethod
from benchmarking.synthetic import ConstantCpChange
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada


@pytest.mark.slow
def test_v0_recovers_constant_cp_uplift(tmp_path) -> None:  # noqa: ANN001
    # A lighter window than the driver (three Zenodo year zips): a 24-month baseline before an
    # early-2018 upgrade plus a 6-month campaign, enough to exercise the full stack on one campaign.
    scada_df, _metadata_df = load_hot_scada(
        start_dt=pd.Timestamp("2016-01-01", tz="UTC"),
        end_dt_excl=pd.Timestamp("2018-09-01", tz="UTC"),
        wtg_numbers=[1, 3, 4, 7],
    )
    context = build_hot_v0_context()
    study = StudyConfig(
        mode="prepost",
        turbine_subset=["T01", "T03", "T04", "T07"],
        treatment_start_range=(pd.Timestamp("2018-02-01", tz="UTC"), pd.Timestamp("2018-02-08", tz="UTC")),
        min_pre_months=24,
        campaign_months=[6],
        n_replicates=1,
        seed=0,
    )

    results = score_study(
        scada_df,
        profile=[ConstantCpChange(delta=0.05)],
        methods=[V0BinnedMethod(context, scratch_dir=tmp_path), OracleMethod(scada_df)],
        study=study,
        profile_name="constant_cp",
    )

    oracle_row = results.loc[results["method"] == "oracle"].iloc[0]
    v0_row = results.loc[results["method"] == "v0_binned"].iloc[0]

    # the harness is wired correctly: the oracle recovers the injected truth exactly
    assert abs(oracle_row["signed_error"]) < 1e-6
    # a real constant-Cp uplift is positive, and v0 recovers it to within a few percentage points
    assert v0_row["truth"] > 0
    assert np.isfinite(v0_row["estimate"])
    assert abs(v0_row["signed_error"]) < 0.03
