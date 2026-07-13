"""P50 evaluation harness (v1 benchmarking, WS1).

Score an uplift method's P50 estimate against the known injected truth from the synthetic
generator, reporting accuracy (bias) and precision (spread) as a function of campaign length.
"""

from __future__ import annotations

from benchmarking.harness.campaign import (
    CampaignWindow,
    campaign_windows,
    treated_activity_mask,
    window_row_mask,
)
from benchmarking.harness.conditions import (
    CONDITION_BINS,
    CONDITIONS,
    TI_BINS,
    WS_BINS,
    condition_bins,
    energy_ratio_by_bin,
)
from benchmarking.harness.leaderboard import conditional_leaderboard, leaderboard
from benchmarking.harness.method import Method, MethodInput, MethodOutput
from benchmarking.harness.metrics import ErrorSummary, summarize_errors
from benchmarking.harness.plots import plot_campaign_curves, plot_conditional_uplift
from benchmarking.harness.replicates import Replicate, StudyConfig, build_replicates
from benchmarking.harness.scoring import score_study

__all__ = [
    "CONDITIONS",
    "CONDITION_BINS",
    "TI_BINS",
    "WS_BINS",
    "CampaignWindow",
    "ErrorSummary",
    "Method",
    "MethodInput",
    "MethodOutput",
    "Replicate",
    "StudyConfig",
    "build_replicates",
    "campaign_windows",
    "condition_bins",
    "conditional_leaderboard",
    "energy_ratio_by_bin",
    "leaderboard",
    "plot_campaign_curves",
    "plot_conditional_uplift",
    "score_study",
    "summarize_errors",
    "treated_activity_mask",
    "window_row_mask",
]
