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

    References are the campaign's declared candidates that have data, so a turbine the campaign
    does not offer is never used however its data looks. A declared candidate the frame carries no
    rows for is dropped with a warning, since it leaves a smaller pool than the campaign asked for.
    Each turbine's validity comes from the campaign's own per-turbine rule.

    :param spec: the campaign's public facts
    :param turbine: the upgraded turbine being estimated
    :param scada_df: the frame the context must cover; its timestamps set the validity index
    """
    present = {str(t) for t in scada_df[spec.turbine_col].unique()}
    references = sorted((set(spec.candidate_references) & present) - {turbine})
    undelivered = sorted(set(spec.candidate_references) - present - {turbine})
    if undelivered:
        logger.warning(
            "%s: the campaign offers %d candidate reference(s) but the data carries no rows for %s, so the "
            "estimate runs on a pool of %d. Reference count drives accuracy.",
            turbine,
            len(spec.candidate_references),
            undelivered,
            len(references),
        )
    # Validity covers every declared turbine with data, not just this estimate's references: a
    # method co-analysing several upgraded turbines keeps them via ``select(also=...)`` and their
    # rows must be screened too.
    declared = sorted((set(spec.upgraded_turbines) | set(spec.candidate_references)) & present)
    covered = sorted({turbine, *references, *declared})
    index = pd.DatetimeIndex(scada_df.index.unique()).sort_values()
    valid = pd.DataFrame(
        {wtg: spec.usable_mask(wtg, index) for wtg in covered},
        index=index,
        dtype=bool,
    )
    return CampaignContext(
        test_wtg=turbine,
        timing=spec.timing_for(turbine),
        turbine_col=spec.turbine_col,
        candidate_references=references,
        valid_for_uplift=valid,
    )
