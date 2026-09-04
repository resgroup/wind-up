"""Whole-farm campaigns: declare one, run it, and report against the known truth."""

from __future__ import annotations

from benchmarking.campaigns.declaration import CampaignSpec, SyntheticCampaign
from benchmarking.campaigns.methods import carried_forward_methods
from benchmarking.campaigns.report import write_campaign_report
from benchmarking.campaigns.runner import CampaignResult, CampaignRunner, per_turbine_table

__all__ = [
    "CampaignResult",
    "CampaignRunner",
    "CampaignSpec",
    "SyntheticCampaign",
    "carried_forward_methods",
    "per_turbine_table",
    "write_campaign_report",
]
