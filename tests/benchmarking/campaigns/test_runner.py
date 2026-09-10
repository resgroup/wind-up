"""Tests for the campaign runner: both output shapes, and a placebo reading ~0."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns import CampaignRunner, per_turbine_table
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


def test_methods_never_see_an_excluded_turbine() -> None:
    recording = RecordingMethod()
    run([recording])
    for mi in recording.seen:
        assert "T5" not in set(mi.scada_df[HOT_COLUMNS.turbine])
        assert {"T1", "T2", "T3", "T4"} == set(mi.scada_df[HOT_COLUMNS.turbine])


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
