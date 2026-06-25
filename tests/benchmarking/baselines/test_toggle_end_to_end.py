"""Slow end-to-end test: the naive ratio method on a HoT-derived synthetic toggle dataset.

Downloads the Hill of Towie v2 SCADA (Zenodo), injects a known constant-Cp uplift under a fast
20-min-on/20-min-off toggle, and scores ``NaiveRatioMethod`` and an oracle through the harness on
real data. This is the sanity that the toggle path composes end to end for a lightweight method;
it is marked ``slow`` (network download) and skipped by ``-m "not slow"``.

v0 is deliberately *not* exercised here: a real wind_up run per campaign is far too slow for an
e2e test. The v0 integration has its own (env-gated) end-to-end test in ``test_v0_end_to_end``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.naive_ratio import NaiveRatioMethod
from benchmarking.harness import StudyConfig, score_study
from benchmarking.harness.example_hot_study import OracleMethod
from benchmarking.synthetic import HOT_COLUMNS, ConstantCpChange
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada


@pytest.mark.slow
def test_naive_recovers_toggle_uplift(tmp_path) -> None:  # noqa: ANN001
    scada_df, _metadata_df = load_hot_scada(
        start_dt=pd.Timestamp("2016-01-01", tz="UTC"),
        end_dt_excl=pd.Timestamp("2018-09-01", tz="UTC"),
        wtg_numbers=[1, 3, 4, 7],
    )
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
            NaiveRatioMethod(active_power_col=HOT_COLUMNS.active_power, out_dir=tmp_path / "naive_runs"),
            OracleMethod(scada_df),
        ],
        study=study,
        profile_name="constant_cp_toggle",
    )

    oracle_row = results.loc[results["method"] == "oracle"].iloc[0]
    naive_row = results.loc[results["method"] == "naive_ratio"].iloc[0]

    # the harness toggle path is wired correctly: the oracle recovers the injected truth exactly
    assert abs(oracle_row["signed_error"]) < 1e-6
    # a real constant-Cp uplift is positive; the naive method recovers it to within a few pp.
    assert naive_row["truth"] > 0
    assert np.isfinite(naive_row["estimate"])
    assert abs(naive_row["signed_error"]) < 0.03
