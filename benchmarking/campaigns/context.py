"""Derive the method-facing campaign context from a campaign declaration.

The one place a :class:`~benchmarking.campaigns.declaration.CampaignSpec` is turned into the
:class:`~benchmarking.harness.context.CampaignContext` methods see, and so the one place to audit
that no ground truth reaches a method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.harness.context import CampaignContext

if TYPE_CHECKING:
    from benchmarking.campaigns.declaration import CampaignSpec


def context_for(spec: CampaignSpec, *, turbine: str, scada_df: pd.DataFrame) -> CampaignContext:
    """Return the context for estimating ``turbine``'s uplift from ``scada_df``.

    References are the campaign's declared candidates that have data, so a turbine the campaign
    does not offer is never used however its data looks. Each turbine's validity comes from the
    campaign's own per-turbine rule.

    :param spec: the campaign's public facts
    :param turbine: the upgraded turbine being estimated
    :param scada_df: the frame the context must cover; its timestamps set the validity index
    """
    present = {str(t) for t in scada_df[spec.turbine_col].unique()}
    references = sorted((set(spec.candidate_references) & present) - {turbine})
    index = pd.DatetimeIndex(scada_df.index.unique()).sort_values()
    valid = pd.DataFrame(
        {wtg: spec.usable_mask(wtg, index) for wtg in [turbine, *references]},
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
