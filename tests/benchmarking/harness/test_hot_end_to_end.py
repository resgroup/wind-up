"""Slow end-to-end test: a real Hill of Towie constant-Cp study through the full harness.

Network- and data-heavy (downloads HoT SCADA from Zenodo on first run); excluded from the
default offline suite via the ``slow`` marker. Proves the harness runs a real study end to
end: load -> inject -> replicate ensemble -> campaign sweep -> score -> leaderboard -> plot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.harness import StudyConfig, leaderboard, plot_campaign_curves, score_study
from benchmarking.synthetic import ConstantCpChange
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

from .stubs import BiasedMethod, OracleMethod

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.slow


def test_constant_cp_study_on_real_hot_data(tmp_path: Path) -> None:
    scada_df, _ = load_hot_scada(
        start_dt=pd.Timestamp("2016-01-01", tz="UTC"),
        end_dt_excl=pd.Timestamp("2017-05-01", tz="UTC"),
        wtg_numbers=[1, 3, 4, 7],  # the stable SW turbines T01, T03, T04, T07
    )

    study = StudyConfig(
        mode="prepost",
        turbine_subset=["T01", "T03", "T04", "T07"],
        treatment_start_range=(pd.Timestamp("2017-01-01", tz="UTC"), pd.Timestamp("2017-01-31", tz="UTC")),
        min_pre_months=12,
        campaign_months=[3],
        n_replicates=2,
        seed=0,
    )
    methods = [OracleMethod(scada_df), BiasedMethod(scada_df, offset=0.02)]

    results = score_study(scada_df, [ConstantCpChange(delta=0.05)], methods, study, profile_name="constant_cp_5pct")
    summary = leaderboard(results)

    oracle = summary[summary["method"] == "oracle"]
    biased = summary[summary["method"] == "biased"]
    assert np.allclose(oracle["bias"].to_numpy(), 0.0, atol=1e-6)  # recovers the injected truth
    assert np.allclose(biased["bias"].to_numpy(), 0.02, atol=1e-6)  # off by exactly the offset
    assert (summary["n_replicates"] == 2).all()

    save_path = tmp_path / "hot_campaign_curves.png"
    plot_campaign_curves(summary, save_path=save_path)
    assert save_path.exists()
