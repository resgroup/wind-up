"""The method seam: ``Method.estimate(facts, analyses) -> list[AnalysisOutput]``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wind_up.analysis import AnalysisOutput, AnalysisSpec
    from wind_up.facts import WindFarmFacts


class MethodConfigurationError(ValueError):
    """A method refused its configuration or the facts it was given, for every subject alike.

    Distinct from the ``ValueError`` a method raises when one subject cannot be measured, which
    leaves that subject unmeasured and the others answered.
    """


@runtime_checkable
class Method(Protocol):
    """An uplift method: a name and ``estimate(facts, analyses)``, one output per analysis, in order."""

    name: str

    def estimate(self, facts: WindFarmFacts, analyses: Sequence[AnalysisSpec]) -> list[AnalysisOutput]:
        """Answer each analysis of ``facts``."""
        ...
