"""Derive the method-facing campaign context from a campaign declaration.

The one place a :class:`~benchmarking.campaigns.declaration.CampaignSpec` is turned into the
:class:`~benchmarking.harness.context.CampaignContext` methods see, and so the one place to audit
that no ground truth reaches a method.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.harness.context import CampaignContext

if TYPE_CHECKING:
    from benchmarking.campaigns.declaration import CampaignSpec

logger = logging.getLogger(__name__)


def context_for(spec: CampaignSpec, *, turbine: str, scada_df: pd.DataFrame) -> CampaignContext:
    """Return the context for estimating ``turbine``'s uplift from ``scada_df``.

    References are the campaign's declared candidates that have data and are not excluded, so a
    turbine the campaign does not offer is never a reference however its data looks. A declared
    candidate the frame carries no rows for is dropped with a warning, since it leaves a smaller pool
    than the campaign asked for. Every other turbine with data is a wake contributor, declared or
    not. Each turbine's validity comes from the campaign's own per-turbine rule.

    :param spec: the campaign's public facts
    :param turbine: the upgraded turbine being estimated
    :param scada_df: the frame the context must cover; its timestamps set the validity index
    """
    present = {str(t) for t in scada_df[spec.turbine_col].unique()}
    offered = set(spec.candidate_references) - set(spec.excluded_turbines)
    references = sorted((offered & present) - {turbine})
    undelivered = sorted(offered - present - {turbine})
    if undelivered:
        logger.warning(
            "%s: the campaign offers %d candidate reference(s) but the data carries no rows for %s, so the "
            "estimate runs on a pool of %d. Reference count drives accuracy.",
            turbine,
            len(offered),
            undelivered,
            len(references),
        )
    wake_contributors = sorted(present - {turbine} - set(references))
    index = pd.DatetimeIndex(scada_df.index.unique()).sort_values()
    valid = pd.DataFrame(
        {wtg: spec.usable_mask(wtg, index) for wtg in sorted(present)},
        index=index,
        dtype=bool,
    )
    return CampaignContext(
        test_wtg=turbine,
        timing=spec.timing_for(turbine),
        turbine_col=spec.turbine_col,
        candidate_references=references,
        wake_contributors=wake_contributors,
        valid_for_uplift=valid,
    )
