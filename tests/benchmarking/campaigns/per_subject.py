"""A base for test methods that measure one subject at a time."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wind_up.subject import answer_analyses

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wind_up.analysis import AnalysisOutput, AnalysisSpec
    from wind_up.facts import WindFarmFacts
    from wind_up.subject import SubjectMeasurement, SubjectRecords


class PerSubject:
    """Answers each analysis by measuring each subject with ``measure``."""

    name: str

    def estimate(self, facts: WindFarmFacts, analyses: Sequence[AnalysisSpec]) -> list[AnalysisOutput]:
        return answer_analyses(facts, analyses, name=self.name, measure=self.measure)

    def measure(self, mi: SubjectRecords) -> SubjectMeasurement:
        raise NotImplementedError
