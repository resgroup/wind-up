"""Resolve an analysis plan per upgraded turbine from a campaign's public facts and its SCADA."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from wind_up.analysis_period import DEFAULT_PLAN_SETTINGS, AnalysisPlan, PlanSettings, plan_analysis

if TYPE_CHECKING:
    from benchmarking.campaigns.declaration import CampaignSpec, Window
    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)


def data_extents(scada_df: pd.DataFrame, *, turbine_col: str, power_col: str) -> dict[str, Window]:
    """Each turbine's first finite power and the end of its last, one timebase after it."""
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    timebase = pd.Timedelta(np.median(np.diff(index.to_numpy()))) if len(index) > 1 else pd.Timedelta(minutes=10)
    finite = scada_df[np.isfinite(scada_df[power_col].to_numpy(dtype=float))]
    grouped = pd.Series(finite.index, index=finite[turbine_col].astype(str).to_numpy()).groupby(level=0)
    return {
        str(t): (first, last + timebase)
        for t, first, last in zip(grouped.min().index, grouped.min(), grouped.max(), strict=True)
    }


@dataclass(frozen=True)
class CampaignPlans:
    """Each upgraded turbine's plan, and every turbine the selector could not plan with its reason.

    :param plans: upgraded turbine to its analysis plan
    :param unplanned: upgraded turbine to the selector's reason for giving up on it
    """

    plans: dict[str, AnalysisPlan] = field(default_factory=dict)
    unplanned: dict[str, str] = field(default_factory=dict)


def plans_for(
    spec: CampaignSpec,
    scada_df: pd.DataFrame,
    *,
    columns: ColumnSchema,
    settings: PlanSettings = DEFAULT_PLAN_SETTINGS,
) -> CampaignPlans:
    """Return each upgraded turbine's plan, and the turbines no span could be found for.

    A turbine the selector cannot plan is reported with its reason rather than stopping the
    campaign, so the others are still analysed.

    Declared references are the candidates; otherwise every turbine not excluded is, the other
    upgraded turbines included. Under a declared span, declared references are forced in even when
    their works overlap it.

    :param spec: the campaign's public facts
    :param scada_df: long-format SCADA; data extents are read from its finite power
    :param columns: the schema ``scada_df`` is keyed by
    :param settings: how spans and references are chosen
    """
    extents = data_extents(scada_df, turbine_col=spec.turbine_col, power_col=columns.active_power)
    works = {t: list(w) for t, w in spec.works.items()}
    works.update({t: [spec.works_window(t)] for t in spec.upgraded_turbines})
    excluded = set(spec.excluded_turbines)
    if spec.references_declared:
        candidates = [r for r in spec.candidate_references if r not in excluded]
    else:
        candidates = [
            r for r in dict.fromkeys([*spec.candidate_references, *spec.upgraded_turbines]) if r not in excluded
        ]

    plans: dict[str, AnalysisPlan] = {}
    unplanned: dict[str, str] = {}
    for turbine in sorted(spec.upgraded_turbines):
        span = spec.period_for(turbine)
        try:
            plan = plan_analysis(
                spec.layout,
                turbine=turbine,
                works=works,
                exclusions=spec.exclusions,
                extents=extents,
                candidates=candidates,
                span=span,
                forced=candidates if spec.references_declared and span is not None else (),
                settings=settings,
            )
        except ValueError as exc:
            logger.warning("%s is not analysed: %s", turbine, exc)
            unplanned[turbine] = str(exc)
            continue
        for reference in plan.forced:
            logger.warning("%s: declared reference %s is used although its works overlap the span", turbine, reference)
        logger.info(
            "%s: analysis span %s..%s (%s), pre %.0f days, post %.0f days, power references %s, pool rule %s",
            turbine,
            plan.start,
            plan.end,
            "declared" if plan.declared else "chosen",
            plan.pre / pd.Timedelta(days=1),
            plan.post / pd.Timedelta(days=1),
            list(plan.power_references),
            "met" if plan.pool_rule_met else f"NOT met: {plan.pool_rule_reason}",
        )
        plans[turbine] = plan
    return CampaignPlans(plans=plans, unplanned=unplanned)
