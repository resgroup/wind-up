"""Tests for the campaign runner: both output shapes, and a placebo reading ~0."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns import CampaignRunner, per_turbine_table, visible_mask
from benchmarking.harness import MethodInput, MethodOutput
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule

from .test_declaration import CHANGEOVER, PERIOD, campaign, scada

TOLERANCE = 1e-9
TOGGLE = ToggleSchedule(period=pd.Timedelta(hours=8), start=CHANGEOVER)


class ZeroMethod:
    """Reports exactly zero uplift, whatever it is given."""

    name = "zero"

    def estimate(self, mi: MethodInput) -> MethodOutput:  # noqa: ARG002
        """Return a zero P50."""
        return MethodOutput(p50_overall=0.0)


class OffsetMethod:
    """Reports a fixed non-zero uplift, so the farm headline is shown to follow the method."""

    name = "offset"

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def estimate(self, mi: MethodInput) -> MethodOutput:  # noqa: ARG002
        """Return the fixed offset as the P50."""
        return MethodOutput(p50_overall=self.offset)


class RecordingMethod:
    """Captures every MethodInput it is handed."""

    name = "recording"

    def __init__(self) -> None:
        self.seen: list[MethodInput] = []

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Record the input and return a zero P50."""
        self.seen.append(mi)
        return MethodOutput(p50_overall=0.0)


def run(methods: list, *, upgrade_timing: object = CHANGEOVER):  # noqa: ANN201
    """Generate the fixture campaign and run ``methods`` over it."""
    declared = campaign(upgrade_timing=upgrade_timing)
    dataset = declared.generate(scada())
    return CampaignRunner(declared.spec(), dataset, build_methods=lambda _wtg: list(methods)).run()


@pytest.mark.parametrize("timing", [CHANGEOVER, TOGGLE], ids=["prepost", "toggle"])
def test_placebo_per_turbine_estimates_are_zero(timing: object) -> None:
    table = per_turbine_table(run([ZeroMethod()], upgrade_timing=timing))
    assert set(table["test_wtg"]) == {"T1", "T2"}
    assert table["truth"].abs().max() < TOLERANCE
    assert table["signed_error"].abs().max() < TOLERANCE


@pytest.mark.parametrize("timing", [CHANGEOVER, TOGGLE], ids=["prepost", "toggle"])
def test_placebo_farm_headline_is_zero(timing: object) -> None:
    result = run([ZeroMethod()], upgrade_timing=timing)
    assert abs(result.truth_farm_uplift) < TOLERANCE
    assert abs(result.farm_uplifts["zero"].uplift) < TOLERANCE
    assert result.farm["signed_error"].abs().max() < TOLERANCE


def test_farm_headline_follows_the_method_not_the_truth() -> None:
    result = run([OffsetMethod(0.04)])
    assert result.farm_uplifts["offset"].uplift == pytest.approx(0.04)
    row = result.farm.set_index("method").loc["offset"]
    assert row["truth"] == pytest.approx(0.0, abs=TOLERANCE)
    assert row["signed_error"] == pytest.approx(0.04)


def test_scores_are_the_tidy_harness_rows() -> None:
    result = run([ZeroMethod()])
    expected = {"method", "test_wtg", "estimate", "truth", "signed_error", "treatment_start", "activity_end"}
    assert expected <= set(result.scores.columns)
    assert len(result.scores[result.scores["condition"] == "overall"]) == 2


def test_each_method_is_estimated_once_per_upgraded_turbine() -> None:
    recording = RecordingMethod()
    run([recording])
    assert len(recording.seen) == 2
    assert {mi.test_wtg for mi in recording.seen} == {"T1", "T2"}


def test_an_excluded_turbine_reaches_methods_for_its_wake_but_never_as_a_reference() -> None:
    recording = RecordingMethod()
    run([recording])
    for mi in recording.seen:
        assert {"T1", "T2", "T3", "T4", "T5"} == set(mi.scada_df[HOT_COLUMNS.turbine])
        assert "T5" not in mi.context.candidate_references
        assert "T5" in mi.context.wake_contributors


def test_methods_see_only_the_analysis_period() -> None:
    recording = RecordingMethod()
    run([recording])
    for mi in recording.seen:
        assert mi.scada_df.index.min() >= PERIOD[0]
        assert mi.scada_df.index.max() < PERIOD[1]


def test_outputs_are_kept_for_every_method_and_turbine() -> None:
    result = run([ZeroMethod()])
    assert set(result.outputs) == {("zero", "T1"), ("zero", "T2")}
    assert all(isinstance(o, MethodOutput) for o in result.outputs.values())


def test_farm_table_reports_the_spread_and_guard_count() -> None:
    result = run([ZeroMethod()])
    assert "uplift_spread" in result.farm.columns
    assert result.farm.set_index("method").loc["zero", "n_guarded"] == 0


def test_actual_energy_covers_the_records_the_truth_uses() -> None:
    result = run([ZeroMethod()])
    detail = result.farm_uplifts["zero"].turbines
    assert (detail["n_records"] > 0).all()
    assert np.isfinite(detail["actual_energy"]).all()


def test_toggle_treats_fewer_records_than_prepost_over_the_same_period() -> None:
    prepost = run([ZeroMethod()]).farm_uplifts["zero"].turbines["n_records"].sum()
    toggle = run([ZeroMethod()], upgrade_timing=TOGGLE).farm_uplifts["zero"].turbines["n_records"].sum()
    assert toggle < prepost


def test_a_campaign_with_no_methods_still_returns_an_indexable_farm_table() -> None:
    result = run([])
    assert list(result.farm.columns) == ["method", "estimate", "truth", "signed_error", "uplift_spread", "n_guarded"]
    assert result.farm.empty


class GuardedMethod:
    """Reports a usable uplift for one turbine and a non-finite one for the other."""

    name = "guarded"

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Return NaN for T1 so farm_uplift drops it, and 0 for everything else."""
        return MethodOutput(p50_overall=float("nan") if mi.test_wtg == "T1" else 0.0)


def test_a_dropped_turbine_is_excluded_from_that_methods_truth() -> None:
    # the estimate covers only the used turbines, so the truth it is compared with must too
    result = run([GuardedMethod()])
    detail = result.farm_uplifts["guarded"].turbines.set_index("turbine")
    assert not detail.loc["T1", "used"]
    assert detail.loc["T2", "used"]

    row = result.farm.set_index("method").loc["guarded"]
    assert row["n_guarded"] == 1
    # placebo truth is 0 whichever turbines are pooled, so the error stays exact rather than
    # mixing a one-turbine estimate against a two-turbine truth
    assert abs(row["signed_error"]) < TOLERANCE


def test_the_campaign_truth_still_covers_every_upgraded_turbine() -> None:
    result = run([GuardedMethod()])
    assert abs(result.truth_farm_uplift) < TOLERANCE


class TestCampaignContext:
    def test_the_method_sees_the_declared_references_not_every_turbine_present(self) -> None:
        # T2 is upgraded and in the frame, but the campaign does not offer it as a reference;
        # T5 is declared but excluded, so the runner never shows it.
        recorder = RecordingMethod()
        run([recorder])
        contexts = {mi.test_wtg: mi.context for mi in recorder.seen}
        assert contexts["T1"].candidate_references == ["T3", "T4"]
        assert contexts["T2"].candidate_references == ["T3", "T4"]

    def test_the_context_covers_the_windowed_rows_the_method_is_given(self) -> None:
        recorder = RecordingMethod()
        run([recorder])
        for mi in recorder.seen:
            given = pd.DatetimeIndex(mi.scada_df.index.unique())
            assert mi.context.valid_over(given).to_numpy().all()


class TestAPlannedCampaign:
    """Truth follows each upgraded turbine's own plan: its span, and only its usable rows."""

    EXCLUDED = (pd.Timestamp("2019-08-01", tz="UTC"), pd.Timestamp("2019-08-15", tz="UTC"))

    def result_and_expected(self) -> tuple:
        from benchmarking.campaigns import SyntheticCampaign  # noqa: PLC0415
        from benchmarking.synthetic import ConstantCpChange  # noqa: PLC0415
        from benchmarking.synthetic.ground_truth import true_uplift  # noqa: PLC0415
        from tests.benchmarking.campaigns.timeline_fixtures import hourly_scada, staggered_spec  # noqa: PLC0415

        spec = staggered_spec(north_offsets=[], exclusions=[("T0", *self.EXCLUDED)])
        declared = SyntheticCampaign(
            upgraded_turbines=spec.upgraded_turbines,
            upgrade_timing=None,
            candidate_references=spec.candidate_references,
            upgrades=[ConstantCpChange(delta=0.05)],
            layout=spec.layout,
            north_offsets=[],
            rated_power_kw=spec.rated_power_kw,
            analysis_period=None,
            works=spec.works,
            exclusions=spec.exclusions,
        )
        frame = hourly_scada()
        # T0 reads a different wind speed inside its exclusion and after its span, so a truth that
        # wrongly counted those rows would move.
        is_t0 = frame[HOT_COLUMNS.turbine] == "T0"
        odd = is_t0 & (
            ((frame.index >= self.EXCLUDED[0]) & (frame.index < self.EXCLUDED[1]))
            | (frame.index >= pd.Timestamp("2020-07-01", tz="UTC"))
        )
        frame.loc[odd, HOT_COLUMNS.wind_speed] = 5.0
        frame.loc[odd, HOT_COLUMNS.active_power] = 300.0
        dataset = declared.generate(frame)
        result = CampaignRunner(declared.spec(), dataset, build_methods=lambda _wtg: [ZeroMethod()]).run()

        plan = result.report.plans["T0"]
        synthetic = result.report.scada_df
        original = dataset.original_df[visible_mask(declared.spec(), dataset.original_df)]
        t0 = synthetic[synthetic[HOT_COLUMNS.turbine] == "T0"].index
        window = (t0 >= plan.works[1]) & (t0 < plan.end)
        inside_exclusion = (t0 >= self.EXCLUDED[0]) & (t0 < self.EXCLUDED[1])

        def truth(mask: np.ndarray) -> float:
            return true_uplift(synthetic, original, test_wtg="T0", mask=mask).overall

        return result, plan, truth(window & ~inside_exclusion), truth(window), truth(t0 >= plan.works[1])

    def test_truth_skips_the_test_turbines_excluded_rows(self) -> None:
        result, _, expected, with_excluded, _ = self.result_and_expected()
        got = per_turbine_table(result).set_index("test_wtg").loc["T0", "truth"]
        assert expected != pytest.approx(with_excluded)
        assert got == pytest.approx(expected)

    def test_truth_ends_at_the_plans_end(self) -> None:
        _, plan, expected, _, whole_post = self.result_and_expected()
        assert plan.end <= pd.Timestamp("2020-07-01", tz="UTC")
        assert expected != pytest.approx(whole_post)


def test_an_unplanned_turbine_is_left_out_of_the_farm_truth() -> None:
    from benchmarking.campaigns import SyntheticCampaign  # noqa: PLC0415
    from benchmarking.synthetic import ConstantCpChange  # noqa: PLC0415
    from tests.benchmarking.campaigns.test_plans import without_pre  # noqa: PLC0415
    from tests.benchmarking.campaigns.timeline_fixtures import hourly_scada, staggered_spec  # noqa: PLC0415

    spec = staggered_spec(north_offsets=[])
    declared = SyntheticCampaign(
        upgraded_turbines=spec.upgraded_turbines,
        upgrade_timing=None,
        candidate_references=spec.candidate_references,
        upgrades=[ConstantCpChange(delta=0.05)],
        layout=spec.layout,
        north_offsets=[],
        rated_power_kw=spec.rated_power_kw,
        analysis_period=None,
        works=spec.works,
        exclusions=spec.exclusions,
    )
    dataset = declared.generate(without_pre(hourly_scada(), "T6"))
    result = CampaignRunner(declared.spec(), dataset, build_methods=lambda _wtg: [ZeroMethod()]).run()

    assert set(result.unplanned) == {"T6"}
    per_turbine = per_turbine_table(result)
    assert list(per_turbine["test_wtg"]) == ["T0"]
    t0_truth = per_turbine.set_index("test_wtg").loc["T0", "truth"]
    assert result.truth_farm_uplift == pytest.approx(t0_truth)
    assert result.farm.loc[0, "truth"] == pytest.approx(t0_truth)
