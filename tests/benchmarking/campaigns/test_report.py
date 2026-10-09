"""Tests for the campaign inspection report."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd

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


def test_report_returns_the_directory_it_wrote_to(tmp_path: Path) -> None:
    result, dataset = _result([ZeroMethod()])
    assert write_campaign_report(result, dataset, out_dir=tmp_path) == tmp_path


def test_no_conditional_plots_when_no_method_reports_conditions(tmp_path: Path) -> None:
    result, dataset = _result([ZeroMethod()])
    write_campaign_report(result, dataset, out_dir=tmp_path)
    assert not (tmp_path / "conditional").exists()


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
    plots = sorted(p.name for p in (out / "conditional").glob("*.png"))
    assert len(plots) == 2  # one per upgraded turbine, power only
    assert all("power" in name for name in plots)
    assert not any("_ws_" in name or "_ti_" in name for name in plots)


def test_conditional_plots_are_written_per_condition_and_turbine(tmp_path: Path) -> None:
    result, dataset = _result([ConditionalZeroMethod()])
    out = write_campaign_report(result, dataset, out_dir=tmp_path)
    plots = sorted(p.name for p in (out / "conditional").glob("*.png"))
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
        assert (out / "reference_stability.csv").exists()

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
        stability = pd.read_csv(out / "reference_stability.csv")
        assert set(stability["turbine"]) == {"T3", "T4"}
        assert set(stability["test_wtg"]) == {"T1", "T2"}

    def test_conditional_estimates_are_written_when_a_method_reports_them(self, tmp_path: Path) -> None:
        out = write_report(_report([ConditionalZeroMethod()]), out_dir=tmp_path)
        assert (out / "conditional.csv").exists()
        plots = sorted(p.name for p in (out / "conditional").glob("*.png"))
        assert len(plots) == len(CONDITIONS) * 2

    def test_nothing_conditional_is_written_when_no_method_reports_conditions(self, tmp_path: Path) -> None:
        out = write_report(_report([ZeroMethod()]), out_dir=tmp_path)
        assert not (out / "conditional.csv").exists()
        assert not (out / "conditional").exists()

    def test_it_returns_the_directory_it_wrote_to(self, tmp_path: Path) -> None:
        assert write_report(_report([ZeroMethod()]), out_dir=tmp_path) == tmp_path
