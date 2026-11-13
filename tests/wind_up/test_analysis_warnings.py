"""The analysis warnings register: severities, ranking and the summary shown at the top of a run's outputs."""

from __future__ import annotations

import pandas as pd
import pytest

from wind_up.analysis_warnings import (
    SEVERITIES,
    AnalysisWarning,
    ranked,
    summary,
    warnings_frame,
)


def _three() -> list[AnalysisWarning]:
    return [
        AnalysisWarning(severity="low", step=4, turbine="T02", message="a candidate yaw change on 2019-03-01"),
        AnalysisWarning(severity="high", step=12, turbine="T13", message="trains on 6.0 months of before data"),
        AnalysisWarning(severity="medium", step=9, turbine="T13", message="the reference screen ruled out T17"),
        AnalysisWarning(severity="medium", step=8, turbine="T12", message="T14: possible undeclared change"),
    ]


def test_the_severities_run_high_to_low() -> None:
    assert SEVERITIES == ("high", "medium", "low")


def test_an_unknown_severity_is_refused() -> None:
    with pytest.raises(ValueError, match="severity"):
        AnalysisWarning(severity="severe", step=1, turbine="T01", message="x")  # type: ignore[arg-type]


def test_ranked_puts_the_highest_severity_first_then_the_step_then_the_turbine() -> None:
    order = [(w.severity, w.step, w.turbine) for w in ranked(_three())]
    assert order == [("high", 12, "T13"), ("medium", 8, "T12"), ("medium", 9, "T13"), ("low", 4, "T02")]


def test_the_frame_is_ranked_with_one_row_per_warning() -> None:
    frame = warnings_frame(_three())
    assert list(frame.columns) == ["severity", "step", "turbine", "message"]
    assert frame["severity"].tolist() == ["high", "medium", "medium", "low"]
    assert frame["step"].tolist() == [12, 8, 9, 4]


def test_an_empty_register_gives_an_empty_frame_with_the_columns() -> None:
    frame = warnings_frame([])
    assert frame.empty
    assert list(frame.columns) == ["severity", "step", "turbine", "message"]
    pd.testing.assert_frame_equal(frame, warnings_frame(()))


def test_the_summary_counts_each_severity_at_the_top_then_lists_the_warnings_ranked() -> None:
    lines = summary(_three()).splitlines()
    assert lines[0] == "Analysis warnings: 1 high, 2 medium, 1 low"
    assert lines[1] == "- high (step 12) T13: trains on 6.0 months of before data"
    assert lines[2] == "- medium (step 8) T12: T14: possible undeclared change"
    assert lines[-1] == "- low (step 4) T02: a candidate yaw change on 2019-03-01"


def test_the_summary_of_no_warnings_says_so_in_one_line() -> None:
    assert summary([]) == "Analysis warnings: none"
