"""Tests for the campaign inspection report."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd

from benchmarking.campaigns import CampaignRunner, write_campaign_report
from benchmarking.harness import CONDITIONS, MethodInput, MethodOutput, condition_bins

from .test_declaration import CHANGEOVER, campaign, scada

if TYPE_CHECKING:
    from pathlib import Path


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


def test_conditional_plots_are_written_per_condition_and_turbine(tmp_path: Path) -> None:
    result, dataset = _result([ConditionalZeroMethod()])
    out = write_campaign_report(result, dataset, out_dir=tmp_path)
    plots = sorted(p.name for p in (out / "conditional").glob("*.png"))
    assert len(plots) == len(CONDITIONS) * 2
    assert any("T1" in name for name in plots)
    assert any("T2" in name for name in plots)
