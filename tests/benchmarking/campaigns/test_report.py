"""Tests for the campaign inspection report."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns import CampaignRunner, write_campaign_report, write_report
from benchmarking.campaigns.run import estimate_campaign
from benchmarking.harness import CONDITIONS, MethodInput, MethodOutput, condition_bins
from benchmarking.synthetic import HOT_COLUMNS

from .test_declaration import CHANGEOVER, campaign, scada

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.campaigns.run import CampaignReport


class ZeroMethod:
    """Reports zero uplift overall and no per-condition breakdown."""

    name = "zero"

    def estimate(self, mi: MethodInput) -> MethodOutput:  # noqa: ARG002
        """Return a zero P50."""
        return MethodOutput(p50_overall=0.0)


class ConditionalZeroMethod:
    """Reports zero uplift overall and in every condition bin."""

    name = "cond_zero"

    def estimate(self, mi: MethodInput) -> MethodOutput:  # noqa: ARG002
        """Return a zero P50 plus zero per-bin estimates on every condition axis."""
        frames = []
        for condition in CONDITIONS:
            edges = condition_bins(condition, rated_power_kw=2300.0)
            bins = pd.IntervalIndex.from_breaks(np.asarray(edges, dtype=float))
            frames.append(
                pd.DataFrame({"condition": condition, "condition_bin": [str(b) for b in bins], "p50_uplift": 0.0})
            )
        return MethodOutput(p50_overall=0.0, p50_by_condition=pd.concat(frames, ignore_index=True))


def _result(methods: list):  # noqa: ANN202
    declared = campaign(upgrade_timing=CHANGEOVER)
    dataset = declared.generate(scada())
    result = CampaignRunner(declared.spec(), dataset, build_methods=lambda _wtg: list(methods)).run()
    return result, dataset


def test_report_writes_the_tables(tmp_path: Path) -> None:
    result, dataset = _result([ZeroMethod()])
    out = write_campaign_report(result, dataset, out_dir=tmp_path)
    assert (out / "per_turbine.csv").exists()
    assert (out / "farm_uplift.csv").exists()
    assert (out / "farm_uplift_detail.csv").exists()
    assert (out / "scores.csv").exists()


def test_farm_table_records_the_spread_and_guards(tmp_path: Path) -> None:
    result, dataset = _result([ZeroMethod()])
    farm = pd.read_csv(write_campaign_report(result, dataset, out_dir=tmp_path) / "farm_uplift.csv")
    assert {"method", "estimate", "truth", "signed_error", "uplift_spread", "n_guarded"} <= set(farm.columns)


def test_per_turbine_detail_is_written_for_each_turbine(tmp_path: Path) -> None:
    result, dataset = _result([ZeroMethod()])
    detail = pd.read_csv(write_campaign_report(result, dataset, out_dir=tmp_path) / "farm_uplift_detail.csv")
    assert set(detail["turbine"]) == {"T1", "T2"}
    assert {"guard", "used", "counterfactual_energy", "method"} <= set(detail.columns)


def test_the_results_across_turbines_go_under_the_farm_folder(tmp_path: Path) -> None:
    result, dataset = _result([ZeroMethod()])
    write_campaign_report(result, dataset, out_dir=tmp_path / "campaign", farm_dir=tmp_path / "farm")
    farm = pd.read_csv(tmp_path / "farm" / "farm_uplift.csv")
    assert "truth" in farm.columns
    assert (tmp_path / "farm" / "farm_uplift_detail.csv").exists()
    assert not (tmp_path / "campaign" / "farm_uplift.csv").exists()
    assert (tmp_path / "campaign" / "per_turbine.csv").exists()


def test_report_returns_the_directory_it_wrote_to(tmp_path: Path) -> None:
    result, dataset = _result([ZeroMethod()])
    assert write_campaign_report(result, dataset, out_dir=tmp_path) == tmp_path


def test_no_conditional_plots_when_no_method_reports_conditions(tmp_path: Path) -> None:
    result, dataset = _result([ZeroMethod()])
    write_campaign_report(result, dataset, out_dir=tmp_path)
    assert not (tmp_path / "15_uplift_distributions").exists()


class PowerOnlyMethod:
    """Reports a per-condition breakdown on the power axis only, as toggle_specialist does."""

    name = "power_only"

    def estimate(self, mi: MethodInput) -> MethodOutput:  # noqa: ARG002
        """Return zero overall and per-power-bin estimates, and nothing on ws or ti."""
        edges = condition_bins("power", rated_power_kw=2300.0)
        bins = pd.IntervalIndex.from_breaks(np.asarray(edges, dtype=float))
        frame = pd.DataFrame({"condition": "power", "condition_bin": [str(b) for b in bins], "p50_uplift": 0.0})
        return MethodOutput(p50_overall=0.0, p50_by_condition=frame)


def test_only_the_conditions_a_method_reports_are_plotted(tmp_path: Path) -> None:
    # a ws or ti plot for this method would have an all-NaN estimate series
    result, dataset = _result([PowerOnlyMethod()])
    out = write_campaign_report(result, dataset, out_dir=tmp_path)
    plots = sorted(p.name for p in (out / "15_uplift_distributions").glob("*.png"))
    assert len(plots) == 2  # one per upgraded turbine, power only
    assert all("power" in name for name in plots)
    assert not any("_ws_" in name or "_ti_" in name for name in plots)


def test_conditional_plots_are_written_per_condition_and_turbine(tmp_path: Path) -> None:
    result, dataset = _result([ConditionalZeroMethod()])
    out = write_campaign_report(result, dataset, out_dir=tmp_path)
    plots = sorted(p.name for p in (out / "15_uplift_distributions").glob("*.png"))
    assert len(plots) == len(CONDITIONS) * 2
    assert any("T1" in name for name in plots)
    assert any("T2" in name for name in plots)


# --- the analyst report: the same run, written without the answer key --------------------------


class ReferenceReportingMethod:
    """Reports a per-reference self-uplift frame, as the power model does by default."""

    name = "with_refs"

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Report zero overall, and one near-zero self-uplift row per candidate reference."""
        references = mi.context.candidate_references
        return MethodOutput(
            p50_overall=0.0,
            reference_uplifts=pd.DataFrame(
                {"turbine": references, "uplift": 0.001, "actual_energy": 1000.0, "n_records": 10, "screened": False}
            ),
        )


def _report(methods: list) -> CampaignReport:
    """Run the fixture campaign through the truth-free core."""
    spec = campaign(upgrade_timing=CHANGEOVER).spec()
    return estimate_campaign(spec, scada(), build_methods=lambda _wtg: list(methods), columns=HOT_COLUMNS)


class TestAnalystReport:
    def test_it_writes_the_analyst_tables(self, tmp_path: Path) -> None:
        out = write_report(_report([ReferenceReportingMethod()]), out_dir=tmp_path)
        assert (out / "per_turbine.csv").exists()
        assert (out / "farm_uplift.csv").exists()
        assert (out / "farm_uplift_detail.csv").exists()
        assert (out / "09_screen" / "reference_stability.csv").exists()

    def test_it_writes_no_scores_table(self, tmp_path: Path) -> None:
        # scores.csv is the harness's truth-carrying shape; the analyst path has no truth to score
        write_report(_report([ZeroMethod()]), out_dir=tmp_path)
        assert not (tmp_path / "scores.csv").exists()

    def test_no_table_it_writes_carries_a_truth_column(self, tmp_path: Path) -> None:
        write_report(_report([ReferenceReportingMethod(), ConditionalZeroMethod()]), out_dir=tmp_path)
        written = sorted(tmp_path.rglob("*.csv"))
        assert written
        for path in written:
            columns = set(pd.read_csv(path).columns)
            assert not ({"truth", "signed_error"} & columns), f"{path.name} carries truth"

    def test_reference_stability_names_every_candidate_reference(self, tmp_path: Path) -> None:
        out = write_report(_report([ReferenceReportingMethod()]), out_dir=tmp_path)
        stability = pd.read_csv(out / "09_screen" / "reference_stability.csv")
        assert set(stability["turbine"]) == {"T3", "T4"}
        assert set(stability["test_wtg"]) == {"T1", "T2"}

    def test_conditional_estimates_are_written_when_a_method_reports_them(self, tmp_path: Path) -> None:
        out = write_report(_report([ConditionalZeroMethod()]), out_dir=tmp_path)
        assert (out / "15_uplift_distributions" / "conditional.csv").exists()
        plots = sorted(p.name for p in (out / "15_uplift_distributions").glob("*.png"))
        assert len(plots) == len(CONDITIONS) * 2

    def test_nothing_conditional_is_written_when_no_method_reports_conditions(self, tmp_path: Path) -> None:
        out = write_report(_report([ZeroMethod()]), out_dir=tmp_path)
        assert not (out / "15_uplift_distributions").exists()

    def test_it_returns_the_directory_it_wrote_to(self, tmp_path: Path) -> None:
        assert write_report(_report([ZeroMethod()]), out_dir=tmp_path) == tmp_path

    def test_the_farm_results_go_to_their_own_directory_when_given_one(self, tmp_path: Path) -> None:
        campaign_dir, farm_dir = tmp_path / "campaign", tmp_path / "farm"
        write_report(_report([ReferenceReportingMethod()]), out_dir=campaign_dir, farm_dir=farm_dir)
        assert (campaign_dir / "per_turbine.csv").exists()
        assert (campaign_dir / "09_screen" / "reference_stability.csv").exists()
        assert (farm_dir / "farm_uplift.csv").exists()
        assert (farm_dir / "farm_uplift_detail.csv").exists()
        assert (farm_dir / ReferenceReportingMethod().name / "farm_uplift.png").exists()
        assert not (campaign_dir / "farm_uplift.csv").exists()


class TestPeriodsAndCandidates:
    @staticmethod
    def planned_report(scada: pd.DataFrame | None = None, **kwargs: object) -> CampaignReport:
        from tests.benchmarking.campaigns.timeline_fixtures import hourly_scada, staggered_spec  # noqa: PLC0415

        spec = staggered_spec(north_offsets=[])
        return estimate_campaign(
            spec,
            hourly_scada() if scada is None else scada,
            build_methods=lambda _wtg: [ZeroMethod()],
            columns=HOT_COLUMNS,
            **kwargs,
        )

    @staticmethod
    def without_pre_t6() -> pd.DataFrame:
        from tests.benchmarking.campaigns.test_plans import without_pre  # noqa: PLC0415
        from tests.benchmarking.campaigns.timeline_fixtures import hourly_scada  # noqa: PLC0415

        return without_pre(hourly_scada(), "T6")

    def test_each_turbines_periods_are_written(self, tmp_path: Path) -> None:
        report = self.planned_report()
        write_report(report, out_dir=tmp_path)
        periods = pd.read_csv(tmp_path / "07_periods" / "periods.csv")
        assert list(periods["test_wtg"]) == ["T0", "T6"]
        assert periods["analysed"].all()
        plan = report.plans["T0"]
        row = periods.set_index("test_wtg").loc["T0"]
        assert pd.Timestamp(row["before_start"]) == plan.start
        assert pd.Timestamp(row["before_end"]) == plan.works[0]
        assert pd.Timestamp(row["after_start"]) == plan.works[1]
        assert pd.Timestamp(row["after_end"]) == plan.end
        assert row["before_days"] == pytest.approx(plan.pre / pd.Timedelta(days=1), abs=0.01)
        assert bool(row["pool_rule_met"]) is True

    def test_the_data_screen_is_written_with_its_plots(self, tmp_path: Path) -> None:
        write_report(self.planned_report(), out_dir=tmp_path)
        screen = tmp_path / "09_screen"
        data = pd.read_csv(screen / "reference_data.csv")
        assert set(data["test_wtg"]) == {"T0", "T6"}
        assert set(pd.read_csv(screen / "reference_hours.csv")["test_wtg"]) == {"T0", "T6"}
        for name in ("data_screen", "record_losses", "reference_hours", "reference_map", "correlation_map"):
            assert (screen / f"{name}_T0.png").exists()

    def test_each_turbines_candidates_are_ranked_by_distance(self, tmp_path: Path) -> None:
        report = self.planned_report()
        write_report(report, out_dir=tmp_path)
        table = pd.read_csv(tmp_path / "08_candidates" / "candidates.csv")
        assert set(table["test_wtg"]) == {"T0", "T6"}
        assert list(table.columns) == ["test_wtg", "turbine", "rank", "distance_d", "role", "reason", "warning"]
        t0 = table[table["test_wtg"] == "T0"]
        assert list(t0["rank"]) == list(range(1, 7))
        assert list(t0["distance_d"]) == sorted(t0["distance_d"])
        assert list(t0[t0["role"] == "power_reference"]["turbine"]) == list(report.plans["T0"].power_references)

    def test_an_unplanned_turbine_is_written_with_its_reason(self, tmp_path: Path) -> None:
        report = self.planned_report(self.without_pre_t6())
        write_report(report, out_dir=tmp_path)
        periods = pd.read_csv(tmp_path / "07_periods" / "periods.csv").set_index("test_wtg")
        assert not bool(periods.loc["T6", "analysed"])
        assert periods.loc["T6", "reason"] == report.unplanned["T6"]
        assert bool(periods.loc["T0", "analysed"])
        assert "T6" not in set(pd.read_csv(tmp_path / "08_candidates" / "candidates.csv")["test_wtg"])

    def test_each_turbine_gets_a_timeline_plot_planned_or_not(self, tmp_path: Path) -> None:
        write_report(self.planned_report(self.without_pre_t6()), out_dir=tmp_path)
        assert (tmp_path / "07_periods" / "analysis_timeline_T0.png").exists()
        assert (tmp_path / "07_periods" / "analysis_timeline_T6.png").exists()

    def test_a_possible_undeclared_change_is_a_warning_against_the_turbine(self, tmp_path: Path) -> None:
        report = self.planned_report()
        plan = report.plans["T0"]
        inside = plan.start + (plan.works[0] - plan.start) / 2
        report = self.planned_report(possible_changes={"T0": [inside], plan.power_references[0]: [inside]})
        write_report(report, out_dir=tmp_path)
        periods = pd.read_csv(tmp_path / "07_periods" / "periods.csv").set_index("test_wtg")
        assert periods.loc["T0", "warning"] == f"possible undeclared change on {inside:%Y-%m-%d}"
        assert pd.isna(periods.loc["T6", "warning"])
        table = pd.read_csv(tmp_path / "08_candidates" / "candidates.csv").set_index(["test_wtg", "turbine"])
        assert table.loc[("T0", plan.power_references[0]), "warning"] == (
            f"possible undeclared change on {inside:%Y-%m-%d}"
        )

    def test_a_declared_period_is_planned_too(self, tmp_path: Path) -> None:
        report = _report([ZeroMethod()])
        write_report(report, out_dir=tmp_path)
        periods = pd.read_csv(tmp_path / "07_periods" / "periods.csv").set_index("test_wtg")
        assert periods["declared"].all()
        assert pd.Timestamp(periods.loc["T1", "after_start"]) == CHANGEOVER
        assert (tmp_path / "07_periods" / "analysis_timeline_T1.png").exists()
        assert (tmp_path / "08_candidates" / "candidates.csv").exists()
