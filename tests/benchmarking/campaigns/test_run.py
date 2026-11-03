"""Tests for the truth-free campaign core: what an analyst path may produce, and what it may not."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns.run import NoTurbineMeasuredError, estimate_campaign, visible_mask, visible_scada
from benchmarking.harness import MethodInput, MethodOutput
from benchmarking.harness.method import MethodConfigurationError
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule
from wind_up.analysis_warnings import AnalysisWarning

from .test_declaration import CHANGEOVER, PERIOD, campaign, scada

# Every column any truth-free output is forbidden to carry.
TRUTH_COLUMNS = frozenset({"truth", "signed_error"})
EXPECTED_ROWS = 21840


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
    assert sorted(frame[HOT_COLUMNS.turbine].unique()) == ["T1", "T2", "T3", "T4", "T5"]
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
    row = report.farm.set_index("method").loc["fixed"]
    assert np.isnan(row["reference_uplift_before_screen"])
    assert np.isnan(row["reference_uplift_after_screen"])


class TestCombinedReferenceUplift:
    """The farm table carries the references combined by energy, before and after the screen."""

    def test_each_reference_counts_once_and_nothing_screened_reads_the_same_both_ways(self) -> None:
        report = run([ReferenceReportingMethod()])
        row = report.farm.set_index("method").loc["with_refs"]
        expected = 2000.0 / (1000.0 / 1.001 + 1000.0 / 1.002) - 1.0
        assert row["reference_uplift_before_screen"] == pytest.approx(expected)
        assert row["reference_uplift_after_screen"] == pytest.approx(expected)
        assert row["reference_uplift_spread"] == pytest.approx(0.001)

    def test_a_screened_reference_counts_before_the_screen_only(self) -> None:
        class Screening(ReferenceReportingMethod):
            name = "screening"

            def estimate(self, mi: MethodInput) -> MethodOutput:
                out = super().estimate(mi)
                refs = out.reference_uplifts
                refs.loc[refs["turbine"] == "T4", ["uplift", "screened"]] = [0.5, True]
                return out

        row = run([Screening()]).farm.set_index("method").loc["screening"]
        assert row["reference_uplift_after_screen"] == pytest.approx(0.001)
        assert np.isnan(row["reference_uplift_spread"])
        assert row["reference_uplift_before_screen"] == pytest.approx(2000.0 / (1000.0 / 1.001 + 1000.0 / 1.5) - 1.0)


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
    def test_visible_mask_keeps_the_analysis_period_and_every_turbine(self) -> None:
        spec, frame = spec_and_frame()
        keep = visible_mask(spec, frame)
        kept = frame[keep]
        assert sorted(kept[HOT_COLUMNS.turbine].unique()) == ["T1", "T2", "T3", "T4", "T5"]
        assert kept.index.min() >= PERIOD[0]
        assert kept.index.max() < PERIOD[1]

    def test_visible_scada_clips_and_norths_in_one_step(self) -> None:
        spec, frame = spec_and_frame()
        visible = visible_scada(spec, frame, columns=HOT_COLUMNS)
        assert len(visible) == EXPECTED_ROWS
        assert "T5" in set(visible[HOT_COLUMNS.turbine])


class FailingMethod(FixedMethod):
    """Cannot measure the turbines named, as the power model cannot below its hour floor."""

    name = "fixed"

    def __init__(self, failing: set[str]) -> None:
        super().__init__()
        self.failing = failing

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Raise for a failing turbine; report the fixed uplift otherwise."""
        if mi.test_wtg in self.failing:
            msg = "too few usable hours"
            raise ValueError(msg)
        return super().estimate(mi)


def test_a_turbine_that_cannot_be_measured_is_reported_and_the_campaign_continues() -> None:
    report = run([FailingMethod({"T1"})])
    assert report.per_turbine["test_wtg"].tolist() == ["T2"]
    assert set(report.plans) == {"T2"}
    assert report.unplanned == {"T1": "not measured by fixed: too few usable hours"}
    assert float(report.farm.set_index("method").loc["fixed", "estimate"]) == pytest.approx(0.02)


def test_a_campaign_none_of_whose_turbines_can_be_measured_raises() -> None:
    with pytest.raises(NoTurbineMeasuredError, match="no upgraded turbine could be measured") as caught:
        run([FailingMethod({"T1", "T2"})])
    assert set(caught.value.unmeasured) == {"T1", "T2"}
    assert isinstance(caught.value, ValueError)


def test_a_toggle_campaign_runs_through_the_same_core() -> None:
    spec = campaign(upgrade_timing=ToggleSchedule(period=pd.Timedelta(hours=8), start=CHANGEOVER)).spec()
    report = estimate_campaign(spec, scada(), build_methods=lambda _wtg: [FixedMethod()], columns=HOT_COLUMNS)
    assert set(report.per_turbine["test_wtg"]) == {"T1", "T2"}


class TestAPlannedCampaign:
    """A campaign declared with works windows runs each upgraded turbine over its own planned span."""

    @staticmethod
    def planned(methods: list | None = None, **overrides: object) -> tuple:
        from tests.benchmarking.campaigns.timeline_fixtures import hourly_scada, staggered_spec  # noqa: PLC0415

        spec = staggered_spec(north_offsets=[], **overrides)
        method = FixedMethod()
        report = estimate_campaign(
            spec, hourly_scada(), build_methods=lambda _wtg: list(methods or [method]), columns=HOT_COLUMNS
        )
        return spec, report, method

    def test_a_declared_period_is_planned_as_given(self) -> None:
        report = run([FixedMethod()])
        assert set(report.plans) == {"T1", "T2"}
        assert all(plan.declared for plan in report.plans.values())
        assert report.plans["T1"].before == (PERIOD[0], CHANGEOVER)
        assert report.plans["T1"].after == (CHANGEOVER, PERIOD[1])

    def test_each_upgraded_turbine_gets_a_plan(self) -> None:
        _, report, _ = self.planned()
        assert set(report.plans) == {"T0", "T6"}

    def test_each_turbine_is_estimated_over_its_own_span(self) -> None:
        _, report, method = self.planned()
        for wtg in ("T0", "T6"):
            plan, seen = report.plans[wtg], method.seen[wtg]
            assert seen.scada_df.index.min() >= plan.start
            assert seen.scada_df.index.max() < plan.end
            assert seen.context.candidate_references == list(plan.power_references)
            assert seen.context.timing == plan.works[1]

    def test_held_back_rows_reach_the_method_marked_invalid_and_northed(self) -> None:
        _, _, method = self.planned()
        seen = method.seen["T0"]
        rows = seen.scada_df[seen.scada_df[HOT_COLUMNS.turbine] == "T4"]
        inside = (rows.index >= pd.Timestamp("2019-02-01", tz="UTC")) & (
            rows.index < pd.Timestamp("2019-02-05", tz="UTC")
        )
        assert inside.sum() == 4 * 24
        assert rows.loc[inside, HOT_COLUMNS.northed("nacelle_position")].notna().all()
        assert not seen.context.valid_over(pd.DatetimeIndex(rows.index[inside]))["T4"].any()

    def test_the_northing_step_does_not_see_held_back_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import benchmarking.campaigns.run as run_module  # noqa: PLC0415

        captured: list[pd.DataFrame] = []
        real = run_module.north_scada

        def spy(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
            captured.append(frame)
            return real(frame, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(run_module, "north_scada", spy)
        self.planned()
        t4 = captured[0][captured[0][HOT_COLUMNS.turbine] == "T4"]
        assert not (
            (t4.index >= pd.Timestamp("2019-02-01", tz="UTC")) & (t4.index < pd.Timestamp("2019-02-05", tz="UTC"))
        ).any()

    def test_the_test_turbines_works_rows_are_dropped(self) -> None:
        _, report, _ = self.planned()
        t0 = report.scada_df[report.scada_df[HOT_COLUMNS.turbine] == "T0"]
        assert not (
            (t0.index >= pd.Timestamp("2019-06-01", tz="UTC")) & (t0.index < pd.Timestamp("2019-06-06", tz="UTC"))
        ).any()


class TestTheDataScreen:
    """Each plan is screened on its data before any method sees it."""

    @staticmethod
    def screened() -> tuple:
        from tests.benchmarking.campaigns.test_data_screen import _invalid, _scada  # noqa: PLC0415
        from tests.benchmarking.campaigns.timeline_fixtures import staggered_spec  # noqa: PLC0415

        method = FixedMethod()
        report = estimate_campaign(
            staggered_spec(north_offsets=[]),
            _invalid(_scada(), "T2", before="2019-04-01"),
            build_methods=lambda _wtg: [method],
            columns=HOT_COLUMNS,
        )
        return report, method

    def test_a_dropped_candidate_is_not_a_power_reference(self) -> None:
        _, method = self.screened()
        context = method.seen["T0"].context
        assert context.candidate_references == ["T1", "T3", "T4", "T5"]
        assert "T2" in context.wake_contributors
        assert context.reserve_references == []

    def test_the_step_8_plan_keeps_the_ranking_before_the_screen(self) -> None:
        report, _ = self.screened()
        assert report.plans["T0"].power_references == ("T1", "T2", "T3", "T4")

    def test_the_report_carries_the_screen_tables(self) -> None:
        report, _ = self.screened()
        assert set(report.reference_data["test_wtg"]) == {"T0", "T6"}
        dropped = report.reference_data[report.reference_data["verdict"] == "dropped"]
        assert ("T0", "T2") in set(zip(dropped["test_wtg"], dropped["candidate"], strict=True))
        assert set(report.reference_hours["test_wtg"]) == {"T0", "T6"}

    def test_the_screen_can_be_turned_off(self) -> None:
        from tests.benchmarking.campaigns.test_data_screen import _invalid, _scada  # noqa: PLC0415
        from tests.benchmarking.campaigns.timeline_fixtures import staggered_spec  # noqa: PLC0415

        method = FixedMethod()
        report = estimate_campaign(
            staggered_spec(north_offsets=[]),
            _invalid(_scada(), "T2", before="2019-04-01"),
            build_methods=lambda _wtg: [method],
            columns=HOT_COLUMNS,
            data_screen=None,
        )
        assert method.seen["T0"].context.candidate_references == ["T1", "T2", "T3", "T4"]
        assert report.reference_data.empty


class TestAnUnplannableTurbine:
    """A turbine the selector cannot plan is reported; the rest of the campaign still runs."""

    @staticmethod
    def report(blank: tuple[str, ...]) -> object:
        from tests.benchmarking.campaigns.test_plans import without_pre  # noqa: PLC0415
        from tests.benchmarking.campaigns.timeline_fixtures import hourly_scada, staggered_spec  # noqa: PLC0415

        frame = hourly_scada()
        for turbine in blank:
            frame = without_pre(frame, turbine)
        return estimate_campaign(
            staggered_spec(north_offsets=[]), frame, build_methods=lambda _wtg: [FixedMethod()], columns=HOT_COLUMNS
        )

    def test_it_has_no_estimate_and_the_farm_is_over_the_others(self) -> None:
        report = self.report(("T6",))
        assert list(report.per_turbine["test_wtg"]) == ["T0"]
        assert set(report.plans) == {"T0"}
        assert list(report.farm_uplifts["fixed"].turbines["turbine"]) == ["T0"]
        assert report.farm.loc[0, "estimate"] == pytest.approx(0.02)

    def test_it_is_named_with_the_selectors_reason(self) -> None:
        report = self.report(("T6",))
        assert set(report.unplanned) == {"T6"}
        assert "no span gives both sides" in report.unplanned["T6"]

    def test_a_campaign_with_no_plannable_turbine_raises_naming_each(self) -> None:
        with pytest.raises(ValueError, match="no upgraded turbine could be planned") as raised:
            self.report(("T0", "T6"))
        assert "T0: no span" in str(raised.value)
        assert "T6: no span" in str(raised.value)

    def test_a_flat_campaign_reports_none_unplanned(self) -> None:
        assert run([FixedMethod()]).unplanned == {}


class WarningMethod(FixedMethod):
    """Reports a fixed uplift with a high-severity analysis warning against the test turbine."""

    name = "warning"

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Report the fixed uplift and one warning of a training period under the floor."""
        self.seen[mi.test_wtg] = mi
        warning = AnalysisWarning(severity="high", step=12, turbine=mi.test_wtg, message="trains on 6.0 months")
        return MethodOutput(p50_overall=self.uplift, warnings=(warning,))


class ScreeningMethod(ReferenceReportingMethod):
    """Reports its references with the first one ruled out by the reference screen."""

    name = "screening"

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Report zero overall, the first candidate reference screened out."""
        output = super().estimate(mi)
        assert output.reference_uplifts is not None
        screened = [i == 0 for i in range(len(output.reference_uplifts))]
        return MethodOutput(p50_overall=0.0, reference_uplifts=output.reference_uplifts.assign(screened=screened))


class TestTheWarningsRegister:
    """Every warning a run raises is gathered on the report with its severity, ranked."""

    def test_a_methods_warnings_are_gathered_against_its_turbine(self) -> None:
        report = run([WarningMethod()])
        high = [w for w in report.warnings if w.severity == "high"]
        assert {(w.step, w.turbine, w.message) for w in high} == {
            (12, "T1", "trains on 6.0 months"),
            (12, "T2", "trains on 6.0 months"),
        }

    def test_a_reference_the_screen_ruled_out_is_a_medium_warning_of_step_9(self) -> None:
        report = run([ScreeningMethod()])
        medium = [w for w in report.warnings if w.severity == "medium" and w.step == 9]
        assert len(medium) == 2
        assert all("ruled out" in w.message for w in medium)
        assert {w.turbine for w in medium} == {"T1", "T2"}

    def test_a_candidate_the_data_screen_dropped_is_a_medium_warning_of_step_9(self) -> None:
        report, _ = TestTheDataScreen.screened()
        dropped = [w for w in report.warnings if w.severity == "medium" and w.step == 9 and w.turbine == "T0"]
        assert any("T2" in w.message and "data screen" in w.message for w in dropped)

    def test_a_candidate_yaw_change_is_a_low_warning_of_step_4_and_an_undeclared_change_a_medium_of_step_8(
        self,
    ) -> None:
        from tests.benchmarking.campaigns.timeline_fixtures import hourly_scada, staggered_spec  # noqa: PLC0415

        report = estimate_campaign(
            staggered_spec(north_offsets=[]),
            hourly_scada(),
            build_methods=lambda _wtg: [FixedMethod()],
            columns=HOT_COLUMNS,
            possible_changes={"T1": [pd.Timestamp("2019-04-15", tz="UTC")]},
        )
        low = [w for w in report.warnings if w.severity == "low"]
        assert [(w.step, w.turbine) for w in low] == [(4, "T1")]
        assert "2019-04-15" in low[0].message
        medium = [w for w in report.warnings if w.severity == "medium" and w.step == 8]
        assert medium
        assert all("T1" in w.message and "undeclared change" in w.message for w in medium)

    def test_the_register_is_ranked_and_empty_without_warnings(self) -> None:
        report = run([FixedMethod()])
        assert report.warnings == ()
        ranked_report = run([WarningMethod(), ScreeningMethod()])
        severities = [w.severity for w in ranked_report.warnings]
        assert severities == sorted(severities, key=["high", "medium", "low"].index)


class MisconfiguredMethod(FixedMethod):
    """Refuses every turbine the way the power model refuses a bad option combination."""

    name = "fixed"

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Raise the method's configuration error."""
        msg = f"{mi.test_wtg}: headline_estimator='reference_corrected' needs report_reference_uplifts=True"
        raise MethodConfigurationError(msg)


def test_a_methods_configuration_error_fails_the_campaign_instead_of_leaving_its_turbines_unmeasured() -> None:
    with pytest.raises(MethodConfigurationError, match="report_reference_uplifts"):
        run([MisconfiguredMethod()])
