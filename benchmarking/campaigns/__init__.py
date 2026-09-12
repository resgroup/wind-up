"""Whole-farm campaigns: declare one, run it, and report on it with or without known truth."""

from __future__ import annotations

from benchmarking.campaigns.declaration import CampaignSpec, SyntheticCampaign
from benchmarking.campaigns.methods import carried_forward_methods
from benchmarking.campaigns.report import write_campaign_report, write_report
from benchmarking.campaigns.run import CampaignReport, estimate_campaign, visible_mask, visible_scada
from benchmarking.campaigns.runner import CampaignResult, CampaignRunner, per_turbine_table

__all__ = [
    "CampaignReport",
    "CampaignResult",
    "CampaignRunner",
    "CampaignSpec",
    "SyntheticCampaign",
    "carried_forward_methods",
    "estimate_campaign",
    "per_turbine_table",
    "visible_mask",
    "visible_scada",
    "write_campaign_report",
    "write_report",
]
