"""Tests for the analysis specification, a method's output and the combination of outputs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from wind_up.analysis import (
    AEP,
    CAMPAIGN,
    INVERSE_VARIANCE,
    NET,
    REPORTS,
    AnalysisOutput,
    AnalysisSpec,
    combine,
    combine_subjects,
)
from wind_up.analysis_warnings import AnalysisWarning
from wind_up.toggle import ToggleSchedule

from .facts_fixtures import at


def test_an_analysis_defaults_to_every_subject_an_open_period_and_both_uplifts() -> None:
    analysis = AnalysisSpec(name="blades", upgrade="blades")
    assert analysis.subjects is None
    assert analysis.references is None
    assert analysis.period == (None, None)
    assert analysis.report == REPORTS
    assert analysis.combination == INVERSE_VARIANCE


def test_an_analysis_freezes_its_collections() -> None:
    analysis = AnalysisSpec(
        name="x", upgrade="blades", subjects=["T1"], references=["T2"], include_labels=["a"], report=CAMPAIGN
    )
    assert analysis.subjects == ("T1",)
    assert analysis.references == ("T2",)
    assert analysis.include_labels == ("a",)
    assert analysis.report == (CAMPAIGN,)


@pytest.mark.parametrize(("name", "upgrade"), [(" ", "blades"), ("x", "")])
def test_an_analysis_needs_a_name_and_an_upgrade(name: str, upgrade: str) -> None:
    with pytest.raises(ValueError, match="needs a name and an upgrade"):
        AnalysisSpec(name=name, upgrade=upgrade)


@pytest.mark.parametrize("subjects", [[], ["T1", "T1"]])
def test_named_subjects_are_at_least_one_each_named_once(subjects: list[str]) -> None:
    with pytest.raises(ValueError, match="at least one subject"):
        AnalysisSpec(name="x", upgrade="blades", subjects=subjects)


@pytest.mark.parametrize("report", [(), ("p50",), (AEP, AEP)])
def test_the_report_is_one_or_both_uplifts(report: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="report is one or both"):
        AnalysisSpec(name="x", upgrade="blades", report=report)


def test_the_combination_is_inverse_variance_or_net() -> None:
    assert AnalysisSpec(name="x", upgrade="blades", combination=NET).combination == NET
    with pytest.raises(ValueError, match="combination is one of"):
        AnalysisSpec(name="x", upgrade="blades", combination="mean")


def test_a_period_ends_after_it_starts() -> None:
    with pytest.raises(ValueError, match="ends at or before its start"):
        AnalysisSpec(name="x", upgrade="blades", period=(at(2), at(2)))


def test_a_label_cannot_be_both_included_and_excluded() -> None:
    with pytest.raises(ValueError, match="both includes and excludes"):
        AnalysisSpec(name="x", upgrade="blades", include_labels=["a"], exclude_labels=["a"])


def output(
    name: str,
    *,
    energy: float | None,
    campaign: float = float("nan"),
    campaign_sigma: float = float("nan"),
    aep: float = float("nan"),
    aep_sigma: float = float("nan"),
    warnings: tuple[AnalysisWarning, ...] = (),
) -> AnalysisOutput:
    per_turbine = (
        None
        if energy is None
        else pd.DataFrame({"turbine": ["T1", "T2"], "actual_energy": [energy / 4, 3 * energy / 4]})
    )
    return AnalysisOutput(
        analysis=AnalysisSpec(name=name, upgrade="blades"),
        aep_uplift=aep,
        aep_sigma=aep_sigma,
        campaign_uplift=campaign,
        campaign_sigma=campaign_sigma,
        per_turbine=per_turbine,
        warnings=warnings,
    )


def test_an_output_is_nan_until_a_method_fills_it() -> None:
    empty = AnalysisOutput(analysis=AnalysisSpec(name="x", upgrade="blades"))
    assert all(math.isnan(v) for v in (empty.aep_uplift, empty.aep_sigma, empty.campaign_uplift, empty.campaign_sigma))
    assert empty.warnings == ()
    assert math.isnan(empty.actual_energy)


def test_an_outputs_energy_is_the_sum_over_its_subjects() -> None:
    assert output("x", energy=400.0).actual_energy == 400.0
    without = AnalysisOutput(
        analysis=AnalysisSpec(name="x", upgrade="blades"), per_turbine=pd.DataFrame({"turbine": ["T1"]})
    )
    assert math.isnan(without.actual_energy)
    unmeasured = AnalysisOutput(
        analysis=AnalysisSpec(name="x", upgrade="blades"),
        per_turbine=pd.DataFrame({"turbine": ["T1"], "actual_energy": [np.nan]}),
    )
    assert math.isnan(unmeasured.actual_energy)


def test_combine_weights_each_output_by_its_energy_without_the_upgrade() -> None:
    a = output("a", energy=110.0, campaign=0.10, campaign_sigma=0.02)
    b = output("b", energy=306.0, campaign=0.02, campaign_sigma=0.01)
    combined = combine([a, b])
    # without the upgrade: 100 and 300
    assert combined.campaign_uplift == pytest.approx(416.0 / 400.0 - 1.0)
    assert combined.campaign_sigma_correlated == pytest.approx(0.25 * 0.02 + 0.75 * 0.01)
    assert combined.campaign_sigma_independent == pytest.approx(math.hypot(0.25 * 0.02, 0.75 * 0.01))
    assert combined.campaign_sigma_independent < combined.campaign_sigma_correlated
    assert combined.outputs["analysis"].tolist() == ["a", "b"]
    assert combined.outputs["actual_energy"].tolist() == [110.0, 306.0]


def test_combine_reports_only_the_bases_every_output_has() -> None:
    a = output("a", energy=110.0, campaign=0.10, campaign_sigma=0.02, aep=0.08, aep_sigma=0.03)
    b = output("b", energy=306.0, campaign=0.02, campaign_sigma=0.01)
    combined = combine([a, b])
    assert math.isfinite(combined.campaign_uplift)
    assert math.isnan(combined.aep_uplift)
    assert math.isnan(combined.aep_sigma_independent)
    assert math.isnan(combined.aep_sigma_correlated)
    assert any("no combined aep uplift" in w.message and w.severity == "medium" for w in combined.warnings)
    assert not any("no combined campaign uplift" in w.message for w in combined.warnings)


def test_combine_gives_the_uplift_without_a_sigma_when_an_output_has_none() -> None:
    a = output("a", energy=110.0, campaign=0.10, campaign_sigma=0.02)
    b = output("b", energy=306.0, campaign=0.02)
    combined = combine([a, b])
    assert combined.campaign_uplift == pytest.approx(0.04)
    assert math.isnan(combined.campaign_sigma_independent)
    assert math.isnan(combined.campaign_sigma_correlated)


@pytest.mark.parametrize(("energy", "uplift"), [(None, 0.02), (0.0, 0.02), (-5.0, 0.02), (100.0, -1.0)])
def test_combine_gives_no_uplift_when_an_output_cannot_be_weighted(energy: float | None, uplift: float) -> None:
    a = output("a", energy=110.0, campaign=0.10, campaign_sigma=0.02)
    combined = combine([a, output("b", energy=energy, campaign=uplift, campaign_sigma=0.01)])
    assert math.isnan(combined.campaign_uplift)


def test_combine_carries_the_outputs_warnings_and_says_its_sigma_is_a_range() -> None:
    raised = AnalysisWarning(severity="high", step=12, turbine="T1", message="short training period")
    combined = combine([output("a", energy=110.0, campaign=0.10, campaign_sigma=0.02, warnings=(raised,))])
    assert combined.warnings[0] == raised
    assert combined.warnings[-1].severity == "low"
    assert "range" in combined.warnings[-1].message
    assert combined.campaign_sigma_independent == pytest.approx(0.02)
    assert combined.campaign_sigma_correlated == pytest.approx(0.02)


def test_combine_needs_an_output() -> None:
    with pytest.raises(ValueError, match="at least one output"):
        combine([])


def _group(per_turbine: pd.DataFrame | None, *, warnings: tuple[AnalysisWarning, ...] = ()) -> AnalysisOutput:
    return AnalysisOutput(
        analysis=AnalysisSpec(name="pair", upgrade="steering"), per_turbine=per_turbine, warnings=warnings
    )


def _subjects(**columns: list[float]) -> pd.DataFrame:
    nan = [float("nan")] * 2
    return pd.DataFrame(
        {
            "turbine": ["T1", "T2"],
            "aep_uplift": columns.get("aep_uplift", nan),
            "aep_sigma": columns.get("aep_sigma", nan),
            "campaign_uplift": columns.get("campaign_uplift", nan),
            "campaign_sigma": columns.get("campaign_sigma", nan),
            "actual_energy": columns.get("actual_energy", nan),
            "hours_on": [10.0, 10.0],
            "hours_off": [10.0, 10.0],
        }
    )


def test_combine_subjects_weights_each_subject_by_its_energy_without_the_upgrade() -> None:
    raised = AnalysisWarning(severity="medium", step=9, turbine="T2", message="screened out")
    per_turbine = _subjects(actual_energy=[110.0, 306.0], campaign_uplift=[0.10, 0.02], campaign_sigma=[0.02, 0.01])
    combined = combine_subjects(_group(per_turbine, warnings=(raised,)))
    assert combined.campaign_uplift == pytest.approx(416.0 / 400.0 - 1.0)
    assert combined.campaign_sigma_correlated == pytest.approx(0.25 * 0.02 + 0.75 * 0.01)
    assert combined.campaign_sigma_independent == pytest.approx(math.hypot(0.25 * 0.02, 0.75 * 0.01))
    assert math.isnan(combined.aep_uplift)
    assert combined.outputs["turbine"].tolist() == ["T1", "T2"]
    assert combined.warnings[0] == raised
    assert any("no combined aep uplift" in w.message for w in combined.warnings)
    assert "subjects' errors" in combined.warnings[-1].message


def test_combine_subjects_gives_no_uplift_when_a_subject_was_not_measured() -> None:
    per_turbine = _subjects(actual_energy=[110.0, float("nan")], campaign_uplift=[0.10, float("nan")])
    assert math.isnan(combine_subjects(_group(per_turbine)).campaign_uplift)


@pytest.mark.parametrize("per_turbine", [None, _subjects().iloc[:0]])
def test_combine_subjects_needs_a_per_turbine_table(per_turbine: pd.DataFrame | None) -> None:
    with pytest.raises(ValueError, match="no per-turbine table"):
        combine_subjects(_group(per_turbine))


def test_placebo_rows_are_left_out_of_the_subjects_combination_and_energy() -> None:
    per_turbine = _subjects(actual_energy=[110.0, 306.0], campaign_uplift=[0.10, 0.02], campaign_sigma=[0.02, 0.01])
    with_placebo = pd.concat(
        [per_turbine.assign(placebo=False), per_turbine.iloc[:1].assign(turbine="T9", placebo=True)],
        ignore_index=True,
    )
    combined = combine_subjects(_group(with_placebo))
    assert combined.campaign_uplift == pytest.approx(416.0 / 400.0 - 1.0)
    assert combined.outputs["turbine"].tolist() == ["T1", "T2"]
    assert _group(with_placebo).actual_energy == pytest.approx(416.0)


def test_placebo_timing_is_frozen_and_checked() -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=1))
    analysis = AnalysisSpec(name="a", upgrade="control", placebo_timing={"T3": schedule})
    assert analysis.placebo_timing == {"T3": schedule}
    assert hash(analysis) == hash(AnalysisSpec(name="a", upgrade="control"))
    with pytest.raises(TypeError, match="not a toggle schedule or frame"):
        AnalysisSpec(name="a", upgrade="control", placebo_timing={"T3": at(1)})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="to its subjects"):
        AnalysisSpec(name="a", upgrade="control", subjects=["T3"], placebo_timing={"T3": schedule})
