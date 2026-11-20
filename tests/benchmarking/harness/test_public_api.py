"""The harness package exposes its main entry points at the package root."""

from __future__ import annotations

from benchmarking import harness


def test_public_api_exports_core_entry_points() -> None:
    for name in (
        "StudyConfig",
        "Replicate",
        "build_replicates",
        "Method",
        "MethodInput",
        "MethodOutput",
        "score_study",
        "leaderboard",
        "summarize_errors",
        "ErrorSummary",
        "CampaignWindow",
        "campaign_windows",
        "plot_campaign_curves",
    ):
        assert hasattr(harness, name), name
