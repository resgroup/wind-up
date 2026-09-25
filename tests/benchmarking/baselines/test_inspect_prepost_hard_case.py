"""The hard-case inspector's conditional-uplift plotting helper."""

from __future__ import annotations

import pandas as pd

from benchmarking.baselines.inspect_prepost_hard_case import conditional_truth_vs_estimate
from benchmarking.harness.method import MethodOutput


def test_conditional_truth_vs_estimate_merges_estimate_and_truth() -> None:
    out = MethodOutput(
        p50_overall=0.05,
        p50_by_condition=pd.DataFrame(
            {"condition": ["ws", "ws"], "condition_bin": ["(6.0, 8.0]", "(8.0, 10.0]"], "p50_uplift": [0.06, 0.04]}
        ),
    )
    truth = pd.DataFrame({"condition_bin": ["(6.0, 8.0]", "(8.0, 10.0]"], "true_uplift": [0.05, 0.05]})
    merged = conditional_truth_vs_estimate(out, {"ws": truth}, method_name="power_model")
    row = merged[merged["condition_bin"] == "(6.0, 8.0]"].iloc[0]
    assert row["mean_estimate"] == 0.06
    assert row["mean_truth"] == 0.05
    assert row["method"] == "power_model"
