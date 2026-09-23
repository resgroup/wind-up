"""Validity guard on the recorded golden northing tables (Hill of Towie, Kelmarsh, Penmanshiel).

``golden_northing_corrections_<farm>.yaml`` is each farm's v1 pipeline best-estimate northing
corrections (passes 1-2 plus the pass-4 wake-nadir nudge), in the same flat ``['Txx', <timestamp>,
<offset>]`` layout as v0's ``optimized_northing_corrections.yaml``. They are recorded by
``benchmarking.baselines.study_wake_nadir_golden`` and are the reference small-N / subset / low-data
challenges are scored against. Judging them against v0 or physically is a review job the driver logs;
this test only guards that the recorded files stay well formed, so a broken regeneration cannot be
committed silently.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

NORTHING_DIR = Path(__file__).parents[1] / "test_data" / "hot" / "northing"
# farm slug -> expected turbine count (Penmanshiel's dataset has no T03, hence 14 not 15)
FARM_TURBINES = {"hill_of_towie": 21, "kelmarsh": 6, "penmanshiel": 14}


def _rows(slug: str) -> list[list]:
    return yaml.safe_load((NORTHING_DIR / f"golden_northing_corrections_{slug}.yaml").read_text())


@pytest.mark.parametrize("slug", list(FARM_TURBINES))
def test_every_row_is_a_turbine_timestamp_offset_triple(slug: str) -> None:
    for row in _rows(slug):
        assert len(row) == 3, row
        turbine, timestamp, offset = row
        assert isinstance(turbine, str), row
        assert turbine.startswith("T"), row
        assert pd.notna(pd.Timestamp(timestamp)), row
        # offsets are wrapped into [-180, 180) on the way out (write_north_table_yaml)
        assert -180.0 <= float(offset) < 180.0, row
        assert math.isfinite(float(offset)), row


@pytest.mark.parametrize(("slug", "count"), list(FARM_TURBINES.items()))
def test_the_expected_turbines_are_present(slug: str, count: int) -> None:
    assert len({row[0] for row in _rows(slug)}) == count


@pytest.mark.parametrize("slug", list(FARM_TURBINES))
def test_each_turbine_starts_at_the_window_start_and_is_time_ordered(slug: str) -> None:
    rows = _rows(slug)
    start = min(pd.Timestamp(ts) for _, ts, _ in rows)
    for turbine in {row[0] for row in rows}:
        times = [pd.Timestamp(ts) for name, ts, _ in rows if name == turbine]
        assert times[0] == start, f"{slug} {turbine}: first row {times[0]} is not the window start {start}"
        assert times == sorted(times), f"{slug} {turbine}: timestamps are not sorted"
        assert len(times) == len(set(times)), f"{slug} {turbine}: duplicate timestamps"
