"""Derive the method-facing campaign context from a campaign declaration.

The one place a :class:`~benchmarking.campaigns.declaration.CampaignSpec` is turned into the
:class:`~benchmarking.harness.context.CampaignContext` methods see, and so the one place to audit
that no ground truth reaches a method.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.campaigns.plans import changed_works
from benchmarking.harness.context import CampaignContext
from benchmarking.harness.record_selection import changed_times

if TYPE_CHECKING:
    from benchmarking.campaigns.declaration import CampaignSpec
    from wind_up.analysis_period import AnalysisPlan

logger = logging.getLogger(__name__)


def context_for_plan(spec: CampaignSpec, plan: AnalysisPlan, *, scada_df: pd.DataFrame) -> CampaignContext:
    """Return the context for estimating ``plan.turbine``'s uplift over its planned span.

    The plan's power references are the candidates and its reserves wait among the wake
    contributors; a turbine the frame carries no rows for is dropped from every list, with a
    warning for a power reference. Every other turbine with data contributes its wake. Each
    turbine's validity comes from the campaign's own per-turbine rule. The changed turbines are the
    test turbine and those with works inside the span, each with the times it counts as changed. The
    test turbine's works window is carried, so no measurement uses a record inside it.

    :param spec: the campaign's public facts
    :param plan: the turbine's analysis plan
    :param scada_df: the frame the context must cover, already cut to the plan's span
    """
    present = {str(t) for t in scada_df[spec.turbine_col].unique()}
    references = [r for r in plan.power_references if r in present]
    undelivered = [r for r in plan.power_references if r not in present]
    if undelivered:
        logger.warning(
            "%s: the plan's power references %s have no rows in the data, so the estimate runs on a pool of %d",
            plan.turbine,
            undelivered,
            len(references),
        )
    return CampaignContext(
        test_wtg=plan.turbine,
        timing=spec.timing_for(plan.turbine),
        turbine_col=spec.turbine_col,
        candidate_references=references,
        wake_contributors=sorted(present - {plan.turbine} - set(references)),
        valid_for_uplift=_validity(spec, scada_df=scada_df, present=present),
        coords=dict(spec.coords),
        reserve_references=[r for r in plan.reserves if r in present],
        changed_turbines=changed_times(
            test_wtg=plan.turbine,
            works=changed_works(spec),
            changeover=plan.works,
            start=plan.start,
            end=plan.end,
            test_reach_d=spec.reach_of(plan.turbine, plan.works),
            layout=spec.layout,
        ),
        works=(pd.Timestamp(plan.works[0]), pd.Timestamp(plan.works[1])),
    )


def _validity(spec: CampaignSpec, *, scada_df: pd.DataFrame, present: set[str]) -> pd.DataFrame:
    """Timestamps x turbines: may each turbine's data be used for the uplift estimate."""
    index = pd.DatetimeIndex(scada_df.index.unique()).sort_values()
    return pd.DataFrame({wtg: spec.usable_mask(wtg, index) for wtg in sorted(present)}, index=index, dtype=bool)
