"""Tests for the truth-free campaign core: what an analyst path may produce, and what it may not."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns.run import estimate_campaign, visible_mask, visible_scada
from benchmarking.harness import MethodInput, MethodOutput
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule

from .test_declaration import CHANGEOVER, PERIOD, campaign, scada

# Every column any truth-free output is forbidden to carry.
TRUTH_COLUMNS = frozenset({"truth", "signed_error"})
EXPECTED_ROWS = 17472


def spec_and_frame() -> tuple[object, pd.DataFrame]:
    """The fixture campaign's public spec, and a plain SCADA frame with no ground truth beside it."""
    return campaign().spec(), scada()


class FixedMethod:
    """Reports a fixed uplift, so the core's aggregation is checked against a known number."""

    name = "fixed"

    def __init__(self, uplift: float = 0.02) -> None:
        self.uplift = uplift
        self.seen: dict[str, MethodInput] = {}

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Record the input and report the fixed uplift."""
        self.seen[mi.test_wtg] = mi
        return MethodOutput(p50_overall=self.uplift)


class ReferenceReportingMethod:
    """Reports a per-reference self-uplift frame, as the power model does by default."""

    name = "with_refs"

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Report zero overall, and one self-uplift row per candidate reference."""
        references = mi.context.candidate_references
        return MethodOutput(
            p50_overall=0.0,
            reference_uplifts=pd.DataFrame(
                {
                    "turbine": references,
                    "uplift": np.linspace(0.001, 0.002, len(references)),
                    "actual_energy": 1000.0,
                    "n_records": 10,
                    "screened": False,
                }
            ),
        )


class ConditionalMethod:
    """Reports a per-condition breakdown on the power axis."""

    name = "conditional"

    def estimate(self, mi: MethodInput) -> MethodOutput:  # noqa: ARG002
        """Report zero overall plus two power-bin estimates."""
        frame = pd.DataFrame(
            {"condition": "power", "condition_bin": ["(0.0, 0.5]", "(0.5, 1.0]"], "p50_uplift": [0.01, 0.03]}
        )
        return MethodOutput(p50_overall=0.02, p50_by_condition=frame)


def run(methods: list) -> object:
    """Run the truth-free core over the fixture campaign with ``methods``."""
    spec, frame = spec_and_frame()
    return estimate_campaign(spec, frame, build_methods=lambda _wtg: list(methods), columns=HOT_COLUMNS)


def test_the_core_runs_from_a_bare_scada_frame() -> None:
    # no SyntheticDataset, no original_df, no run metadata: the analyst path has none of them
    report = run([FixedMethod()])
    assert set(report.per_turbine["test_wtg"]) == {"T1", "T2"}
    assert report.per_turbine["estimate"].tolist() == [0.02, 0.02]


def test_the_report_carries_no_truth_columns() -> None:
    # the isolation guarantee: the code that writes the analyst report cannot reach the answer key
    report = run([FixedMethod(), ReferenceReportingMethod(), ConditionalMethod()])
    for name in ("per_turbine", "farm", "reference_stability", "conditional"):
        frame = getattr(report, name)
        assert TRUTH_COLUMNS.isdisjoint(frame.columns), f"{name} carries {sorted(TRUTH_COLUMNS & set(frame.columns))}"


def test_the_farm_headline_aggregates_the_per_turbine_estimates() -> None:
    report = run([FixedMethod(0.04)])
    row = report.farm.set_index("method").loc["fixed"]
    assert float(row["estimate"]) == pytest.approx(0.04)
    assert int(row["n_guarded"]) == 0
    assert float(report.farm_uplifts["fixed"].uplift) == pytest.approx(0.04)


def test_each_method_is_estimated_once_per_upgraded_turbine() -> None:
    method = FixedMethod()
    run([method])
    assert set(method.seen) == {"T1", "T2"}


def test_methods_see_only_the_analysis_period_and_the_usable_turbines() -> None:
    method = FixedMethod()
    run([method])
    frame = method.seen["T1"].scada_df
    assert len(frame) == EXPECTED_ROWS
    assert sorted(frame[HOT_COLUMNS.turbine].unique()) == ["T1", "T2", "T3", "T4"]
    assert frame.index.min() >= PERIOD[0]
    assert frame.index.max() < PERIOD[1]


def test_reference_stability_reports_each_candidate_references_own_uplift() -> None:
    report = run([ReferenceReportingMethod()])
    stability = report.reference_stability
    assert set(stability["test_wtg"]) == {"T1", "T2"}
    assert set(stability["turbine"]) == {"T3", "T4"}
    assert {"method", "test_wtg", "turbine", "uplift", "screened"} <= set(stability.columns)


def test_reference_stability_is_empty_when_no_method_reports_references() -> None:
    report = run([FixedMethod()])
    assert report.reference_stability.empty
    assert {"method", "test_wtg", "turbine", "uplift"} <= set(report.reference_stability.columns)


def test_conditional_estimates_are_carried_per_method_and_turbine() -> None:
    report = run([ConditionalMethod()])
    conditional = report.conditional
    assert set(conditional["condition"]) == {"power"}
    assert set(conditional["test_wtg"]) == {"T1", "T2"}
    assert conditional["p50_uplift"].tolist() == [0.01, 0.03, 0.01, 0.03]


def test_conditional_is_empty_when_no_method_reports_conditions() -> None:
    report = run([FixedMethod()])
    assert report.conditional.empty
    assert {"method", "test_wtg", "condition", "condition_bin", "p50_uplift"} <= set(report.conditional.columns)


def test_the_outputs_are_kept_for_every_method_and_turbine() -> None:
    report = run([FixedMethod()])
    assert set(report.outputs) == {("fixed", "T1"), ("fixed", "T2")}


def test_the_report_carries_the_frame_the_methods_were_given() -> None:
    report = run([FixedMethod()])
    assert len(report.scada_df) == EXPECTED_ROWS


class TestVisibleFrame:
    def test_visible_mask_keeps_the_analysis_period_and_drops_excluded_turbines(self) -> None:
        spec, frame = spec_and_frame()
        keep = visible_mask(spec, frame)
        kept = frame[keep]
        assert sorted(kept[HOT_COLUMNS.turbine].unique()) == ["T1", "T2", "T3", "T4"]
        assert kept.index.min() >= PERIOD[0]
        assert kept.index.max() < PERIOD[1]

    def test_visible_scada_clips_and_norths_in_one_step(self) -> None:
        spec, frame = spec_and_frame()
        visible = visible_scada(spec, frame, columns=HOT_COLUMNS)
        assert len(visible) == EXPECTED_ROWS
        assert "T5" not in set(visible[HOT_COLUMNS.turbine])


def test_a_toggle_campaign_runs_through_the_same_core() -> None:
    spec = campaign(upgrade_timing=ToggleSchedule(period=pd.Timedelta(hours=8), start=CHANGEOVER)).spec()
    report = estimate_campaign(spec, scada(), build_methods=lambda _wtg: [FixedMethod()], columns=HOT_COLUMNS)
    assert set(report.per_turbine["test_wtg"]) == {"T1", "T2"}
