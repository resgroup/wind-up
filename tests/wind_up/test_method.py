"""Tests for the method seam."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wind_up.analysis import AnalysisOutput, AnalysisSpec
from wind_up.method import Method

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wind_up.facts import WindFarmFacts


def test_a_conforming_class_satisfies_the_method_protocol() -> None:
    class FixedMethod:
        name = "fixed"

        def estimate(self, facts: WindFarmFacts, analyses: Sequence[AnalysisSpec]) -> list[AnalysisOutput]:  # noqa: ARG002
            return [AnalysisOutput(analysis=a, campaign_uplift=0.0) for a in analyses]

    method = FixedMethod()
    assert isinstance(method, Method)
    (out,) = method.estimate(None, [AnalysisSpec(name="x", upgrade="u")])  # type: ignore[arg-type]
    assert out.campaign_uplift == 0.0


def test_a_non_conforming_object_is_not_a_method() -> None:
    assert not isinstance(object(), Method)
