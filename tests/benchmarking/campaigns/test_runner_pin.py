"""Characterization pin for the campaign runner, held across the truth-free core refactor.

The rest of the runner suite asserts structure on a placebo, where every number is zero. These
assertions carry the numbers: an injected upgrade makes truth and estimate non-trivial, so a
refactor that changes which rows reach a method, or what the score rows say about them, fails
here rather than silently moving a benchmark.
"""

from __future__ import annotations

import pandas as pd
import pytest

from benchmarking.campaigns import CampaignRunner, per_turbine_table
from benchmarking.harness import MethodInput, MethodOutput
from benchmarking.synthetic import HOT_COLUMNS, ConstantCpChange

from .test_declaration import CHANGEOVER, PERIOD, campaign, scada

# Five turbines over 4368 hourly records; excluded T5 stays for its wake.
EXPECTED_ROWS = 21840
# The pre and post mean power of an upgraded turbine, and the truth and estimate that follow.
BASELINE_POWER_KW = 900.0
UPGRADED_POWER_KW = 944.9809758474812
TRUE_UPLIFT = 0.04997886205275681
# ``HalfOfTruthMethod`` reports half of it, so the pin covers estimate and error separately.
HALF = 0.5
EXACT = 1e-12


class HalfOfTruthMethod:
    """Reads half the mean power ratio it is handed, so its estimate tracks the frame it is given."""

    name = "half"

    def __init__(self) -> None:
        self.seen: dict[str, MethodInput] = {}

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Record the input and report half the test turbine's post/pre mean power ratio less one."""
        self.seen[mi.test_wtg] = mi
        rows = mi.scada_df[mi.scada_df[mi.turbine_col] == mi.test_wtg][HOT_COLUMNS.active_power]
        post = float(rows[rows.index >= CHANGEOVER].mean())
        pre = float(rows[rows.index < CHANGEOVER].mean())
        return MethodOutput(p50_overall=HALF * (post / pre - 1.0))


def run() -> tuple[object, HalfOfTruthMethod]:
    """Run the fixture campaign with an injected upgrade and one frame-sensitive method."""
    declared = campaign(upgrades=[ConstantCpChange(delta=0.05)])
    dataset = declared.generate(scada())
    method = HalfOfTruthMethod()
    result = CampaignRunner(declared.spec(), dataset, build_methods=lambda _wtg: [method]).run()
    return result, method


def test_the_frame_handed_to_a_method_is_unchanged() -> None:
    _, method = run()
    frame = method.seen["T1"].scada_df
    assert len(frame) == EXPECTED_ROWS
    assert list(frame.columns) == list(scada().columns)
    assert sorted(frame[HOT_COLUMNS.turbine].unique()) == ["T1", "T2", "T3", "T4", "T5"]
    assert frame.index.min() == PERIOD[0]
    assert frame.index.max() == PERIOD[1] - pd.Timedelta(hours=1)
    pd.testing.assert_frame_equal(frame, method.seen["T2"].scada_df)


def test_the_injected_upgrade_reaches_the_frame_the_method_reads() -> None:
    _, method = run()
    frame = method.seen["T1"].scada_df
    power = frame[frame[HOT_COLUMNS.turbine] == "T1"][HOT_COLUMNS.active_power]
    assert float(power[power.index < CHANGEOVER].mean()) == pytest.approx(BASELINE_POWER_KW, abs=EXACT)
    assert float(power[power.index >= CHANGEOVER].mean()) == pytest.approx(UPGRADED_POWER_KW, abs=EXACT)


def test_the_scored_numbers_are_unchanged() -> None:
    result, _ = run()
    row = per_turbine_table(result).set_index("test_wtg").loc["T1"]
    assert float(row["truth"]) == pytest.approx(TRUE_UPLIFT, abs=EXACT)
    assert float(row["estimate"]) == pytest.approx(HALF * TRUE_UPLIFT, abs=EXACT)
    assert float(row["signed_error"]) == pytest.approx(-HALF * TRUE_UPLIFT, abs=EXACT)


def test_the_farm_numbers_are_unchanged() -> None:
    result, _ = run()
    row = result.farm.set_index("method").loc["half"]
    assert float(row["truth"]) == pytest.approx(TRUE_UPLIFT, abs=EXACT)
    assert float(row["estimate"]) == pytest.approx(HALF * TRUE_UPLIFT, abs=EXACT)
    assert float(row["signed_error"]) == pytest.approx(-HALF * TRUE_UPLIFT, abs=EXACT)
    assert float(result.truth_farm_uplift) == pytest.approx(TRUE_UPLIFT, abs=EXACT)


def test_each_estimate_keeps_every_non_reference_turbine_for_its_wake() -> None:
    _, method = run()
    assert method.seen["T1"].context.wake_contributors == ["T2", "T5"]
    assert method.seen["T2"].context.wake_contributors == ["T1", "T5"]
