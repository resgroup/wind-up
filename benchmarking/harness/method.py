"""The thin method seam the harness scores against.

Deliberately minimal: the harness only ever sees ``MethodInput -> MethodOutput``. The mode is
inferred from ``upgrade_timing``'s type (a timestamp is prepost; a ``ToggleSchedule`` is
toggle). Reference selection and method-specific config are baked into each ``Method``, not
carried on the input, so the seam bakes in no method assumptions. The Issue 4 data contract
later enriches this behind the same seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    from benchmarking.synthetic import ToggleSchedule


@dataclass
class MethodInput:
    """The windowed data a method sees for one campaign.

    :param scada_df: windowed synthetic SCADA (all subset turbines, baseline + activity)
    :param test_wtg: the upgraded turbine to estimate
    :param upgrade_timing: changeover timestamp (prepost) or schedule (toggle)
    """

    scada_df: pd.DataFrame
    test_wtg: str
    upgrade_timing: pd.Timestamp | ToggleSchedule


@dataclass
class MethodOutput:
    """A method's P50 uplift estimate.

    :param p50_overall: overall P50 uplift (energy-ratio fraction)
    :param p50_by_condition: optional per-condition estimates (columns ``condition_bin``,
        ``p50_uplift``); ``None`` when the method produces only an overall number
    """

    p50_overall: float
    p50_by_condition: pd.DataFrame | None = None


@runtime_checkable
class Method(Protocol):
    """A pluggable uplift estimator: a name plus ``estimate(MethodInput) -> MethodOutput``."""

    name: str

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Return a P50 uplift estimate for the campaign described by ``mi``."""
        ...
