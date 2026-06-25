"""Slow end-to-end tests: the R-learner on HoT-derived synthetic datasets via the harness.

Downloads the Hill of Towie v2 SCADA (Zenodo) and ERA5 (Open-Meteo), injects a known
constant-Cp uplift, and scores ``RLearnerMethod`` and an oracle through the harness on real
data in both prepost and toggle modes. v0 is deliberately not exercised here (a real wind_up
run per campaign is far too slow). Marked ``slow`` (network + model fitting); skipped by
``-m "not slow"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.rlearner import RLearnerMethod
from benchmarking.harness import StudyConfig, score_study
from benchmarking.harness.example_hot_study import OracleMethod
from benchmarking.synthetic import HOT_COLUMNS, ConstantCpChange
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

_SUBSET = ["T01", "T03", "T04", "T07"]
_WTG_NUMBERS = [1, 3, 4, 7]
_FAST_MODEL = {"n_estimators": 200, "verbose": -1}


def _rlearner(tmp_path) -> RLearnerMethod:  # noqa: ANN001
    context = build_hot_v0_context(wtg_names=_SUBSET)
    return RLearnerMethod(
        active_power_col=HOT_COLUMNS.active_power,
        wind_speed_col=HOT_COLUMNS.wind_speed,
        era5_hourly_df=context.reanalysis_datasets[0].data,
        out_dir=tmp_path / "rlearner_runs",
        model_params=_FAST_MODEL,
    )


@pytest.mark.slow
def test_rlearner_recovers_prepost_uplift(tmp_path) -> None:  # noqa: ANN001
    scada_df, _ = load_hot_scada(
        start_dt=pd.Timestamp("2016-01-01", tz="UTC"),
        end_dt_excl=pd.Timestamp("2021-01-01", tz="UTC"),
        wtg_numbers=_WTG_NUMBERS,
        wtg_names=_SUBSET,
    )
    study = StudyConfig(
        mode="prepost",
        turbine_subset=_SUBSET,
        treatment_start_range=(pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2019-01-08", tz="UTC")),
        min_pre_months=24,
        campaign_months=[6],
        n_replicates=1,
        seed=0,
    )
    results = score_study(
        scada_df,
        profile=[ConstantCpChange(delta=0.05)],
        methods=[_rlearner(tmp_path), OracleMethod(scada_df)],
        study=study,
        profile_name="constant_cp_prepost",
    )
    oracle_row = results.loc[results["method"] == "oracle"].iloc[0]
    rlearner_row = results.loc[results["method"] == "rlearner"].iloc[0]
    assert abs(oracle_row["signed_error"]) < 1e-6
    assert rlearner_row["truth"] > 0
    assert np.isfinite(rlearner_row["estimate"])
    assert abs(rlearner_row["signed_error"]) < 0.03


@pytest.mark.slow
def test_rlearner_recovers_toggle_uplift(tmp_path) -> None:  # noqa: ANN001
    scada_df, _ = load_hot_scada(
        start_dt=pd.Timestamp("2016-01-01", tz="UTC"),
        end_dt_excl=pd.Timestamp("2018-09-01", tz="UTC"),
        wtg_numbers=_WTG_NUMBERS,
        wtg_names=_SUBSET,
    )
    study = StudyConfig(
        mode="toggle",
        turbine_subset=_SUBSET,
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
        methods=[_rlearner(tmp_path), OracleMethod(scada_df)],
        study=study,
        profile_name="constant_cp_toggle",
    )
    oracle_row = results.loc[results["method"] == "oracle"].iloc[0]
    rlearner_row = results.loc[results["method"] == "rlearner"].iloc[0]
    assert abs(oracle_row["signed_error"]) < 1e-6
    assert rlearner_row["truth"] > 0
    assert np.isfinite(rlearner_row["estimate"])
    assert abs(rlearner_row["signed_error"]) < 0.03
