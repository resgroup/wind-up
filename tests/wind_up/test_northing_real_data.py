"""Northing regression tests on two years of real Hill of Towie data, end to end through every step.

Synthetic tests pin the algorithm's contract; only real SCADA exercises what it does with site veer,
farm outages, neighbours that recalibrate on the same day and a reference derived from the farm
itself. The fixture (see :mod:`tests.wind_up.hot_northing`) is exactly the input the degradation
study (``benchmarking.baselines.study_northing_degradation``) northes -- yaw, power, nacelle wind
speed, the ``yaw_usable`` mask and ERA5 for all 21 turbines across 2017-2020 -- so each test runs
:func:`~wind_up.northing.north_farm` as a user would: with the layout, and with power for wake-nadir-shift.

Four groups:

* **the default pipeline** -- on each two-year window, the recalibrations v0's published table
  records are found, nothing else is, and the absolute offsets agree across the two windows.
* **degradation** -- a representative sample of the study's cases (few turbines, short records,
  both together, missing data), each scored against the same window's full-data answer and held to
  what the study recorded, so a change that makes northing degrade less gracefully fails.
* **the whole-farm fallback** -- ``layout=None`` still finds the same recalibrations.
* **a lone turbine against reanalysis** -- changepoints-v-reanalysis, and its guard against over-detecting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.wind_up.hot_northing import (
    ALL_TURBINES,
    EARLY,
    LATE,
    FarmInputs,
    changepoints,
    contiguous_order,
    describe,
    farm_inputs,
    fixture_available,
    hot_layout,
    load_fixture,
    monthly,
    north,
    offset_errors,
)
from wind_up.circular_math import circ_diff
from wind_up.northing import estimate_north_table

pytestmark = pytest.mark.skipif(
    not fixture_available(), reason="Hill of Towie northing fixture not available (git-lfs not pulled)"
)

WINDOWS = {"early": EARLY, "late": LATE}

# Every recalibration v0's published table records in each window, and nothing else.
EXPECTED = {
    "early": {
        "T01": [("2017-04-23", 21.1), ("2017-05-04", -19.4)],
        "T05": [("2017-05-03", 35.7), ("2018-04-21", -19.7)],
        "T16": [("2017-05-19", 98.7), ("2017-06-18", 9.0), ("2017-08-09", -7.2)],
    },
    "late": {
        "T11": [("2019-08-19", -4.4)],
        "T12": [("2020-06-18", 170.7)],
        "T19": [("2019-07-12", 98.8), ("2019-12-24", -122.5)],
    },
}
_KNOWN = [(window, turbine) for window, turbines in EXPECTED.items() for turbine in sorted(turbines)]


@pytest.fixture(scope="module")
def hot() -> pd.DataFrame:
    """The fixture, loaded once for the module."""
    return load_fixture()


_RUNS: dict[tuple, dict[str, pd.DataFrame]] = {}


def default_run(hot: pd.DataFrame, window: str, turbines: tuple[str, ...] = ALL_TURBINES) -> dict[str, pd.DataFrame]:
    """The default pipeline's tables for ``turbines`` over a window, memoised -- each run costs ~30 s."""
    key = ("default", window, turbines)
    if key not in _RUNS:
        _RUNS[key] = north(farm_inputs(hot, turbines, *WINDOWS[window]), layout=hot_layout(turbines))
    return _RUNS[key]


def _assert_matches(found: list[tuple[pd.Timestamp, float]], expected: list[tuple[str, float]]) -> None:
    assert len(found) == len(expected), describe(found)
    for (when, step), (expected_when, expected_step) in zip(found, expected, strict=True):
        assert abs(when - pd.Timestamp(expected_when, tz="UTC")) <= pd.Timedelta(days=3), describe(found)
        assert circ_diff(step, expected_step) == pytest.approx(0.0, abs=3.0), describe(found)


class TestDefaultPipeline:
    """All 21 turbines, the layout and wake-nadir-shift: v0's recalibrations, no others, consistent offsets."""

    @pytest.mark.parametrize(("window", "turbine"), _KNOWN, ids=lambda v: v)
    def test_a_turbines_published_recalibrations_are_found(self, hot: pd.DataFrame, window: str, turbine: str) -> None:
        _assert_matches(changepoints(default_run(hot, window)[turbine]), EXPECTED[window][turbine])

    @pytest.mark.parametrize("window", list(WINDOWS))
    def test_every_other_turbine_is_left_alone(self, hot: pd.DataFrame, window: str) -> None:
        """Including the neighbours of a turbine that steps: its step must not leak into them."""
        tables = default_run(hot, window)
        extra = {
            name: describe(found)
            for name, table in tables.items()
            if name not in EXPECTED[window] and (found := changepoints(table))
        }
        assert extra == {}, f"changepoints v0's table does not record: {extra}"

    def test_no_turbine_steps_during_a_farm_outage(self, hot: pd.DataFrame) -> None:
        # most of the farm is down in these windows
        outages = (("2019-11-05", "2019-11-25"), ("2020-06-08", "2020-06-17"))
        during = {
            name: [
                (w, s)
                for w, s in changepoints(table)
                if any(pd.Timestamp(lo, tz="UTC") <= w <= pd.Timestamp(hi, tz="UTC") for lo, hi in outages)
            ]
            for name, table in default_run(hot, "late").items()
        }
        offenders = {name: describe(v) for name, v in during.items() if v}
        assert offenders == {}, f"turbines stepped with the outage, not their own calibration: {offenders}"

    def test_the_two_windows_agree_where_they_meet(self, hot: pd.DataFrame) -> None:
        """Each window is northed independently, yet a turbine's offset at the end of 2018 must equal
        its offset at the start of 2019: its calibration did not change at midnight. This holds the
        absolute frame -- the reanalysis anchor and the wake-nadir shift -- and not just the steps.
        """
        early, late = default_run(hot, "early"), default_run(hot, "late")
        gaps = {
            name: float(circ_diff(late[name]["north_offset"].iloc[0], early[name]["north_offset"].iloc[-1]))
            for name in ALL_TURBINES
        }
        disagree = {name: round(gap, 1) for name, gap in gaps.items() if abs(gap) > 2.5}
        assert disagree == {}, f"offset jumps between windows (deg): {disagree}"


# ---------------------------------------------------------------------------
# degradation: the study's cases, scored against the same window's full-data answer
# ---------------------------------------------------------------------------
def _inputs_for(hot: pd.DataFrame, window: str, case: str) -> tuple[FarmInputs, tuple[str, ...]]:
    """Build a degraded case's inputs, named as the study names it ("N=3", "90d", "N=3,90d", "drop50%")."""
    order = tuple(contiguous_order(hot_layout()))
    start, end = WINDOWS[window]
    turbines, days = ALL_TURBINES, None
    for part in case.split(","):
        if part.startswith("N="):
            turbines = tuple(sorted(order[: int(part[2:])]))
        elif part.endswith("d") and part[:-1].isdigit():
            days = int(part[:-1])
    inputs = farm_inputs(hot, turbines, start, end, last_days=days)
    if case.startswith("drop"):
        rng = np.random.default_rng(0)
        frac = int(case[4:-1]) / 100
        inputs = inputs.knocked_out({t: rng.random(len(inputs.index)) < frac for t in turbines})
    elif case == "6mo_outage":
        lo = inputs.index.min() + pd.Timedelta(days=180)
        black = np.asarray((inputs.index >= lo) & (inputs.index < lo + pd.Timedelta(days=180)))
        inputs = inputs.knocked_out(dict.fromkeys(turbines, black))
    return inputs, turbines


# (case, window) -> worst turbine's offset error (deg) against the window's full-data answer, as
# recorded on the fixture with changepoints-v-consensus repeated to convergence. A case may be up to
# half again as bad, plus a degree, before it fails: enough for incidental change, not for a real
# loss of robustness.
RECORDED_WORST_ERROR = {
    ("N=6", "early"): 2.0,
    ("N=6", "late"): 1.9,
    ("N=3", "early"): 5.0,
    ("N=3", "late"): 1.9,
    ("N=2", "early"): 2.3,
    ("N=2", "late"): 2.0,
    ("N=1", "early"): 2.2,
    ("N=1", "late"): 2.7,
    ("365d", "early"): 2.3,
    ("365d", "late"): 2.1,
    ("90d", "early"): 5.3,
    ("90d", "late"): 2.4,
    ("30d", "early"): 17.5,
    ("30d", "late"): 3.7,
    ("7d", "early"): 15.7,
    ("7d", "late"): 10.9,
    ("N=3,90d", "early"): 2.2,
    ("N=3,90d", "late"): 12.5,
    ("N=2,30d", "early"): 1.4,
    ("N=2,30d", "late"): 1.9,
    ("N=1,14d", "early"): 4.7,
    ("N=1,14d", "late"): 0.7,
    ("N=1,7d", "early"): 8.0,
    ("N=1,7d", "late"): 1.4,
}
# Cases that cost a full-farm, full-window run each: CI only.
RECORDED_WORST_ERROR_SLOW = {
    ("N=10", "early"): 2.1,
    ("N=10", "late"): 2.0,
    ("drop50%", "early"): 2.2,
    ("drop50%", "late"): 1.9,
    ("drop75%", "early"): 5.0,
    ("drop75%", "late"): 3.2,
    ("6mo_outage", "early"): 2.0,
    ("6mo_outage", "late"): 1.7,
}


def _check_degraded(hot: pd.DataFrame, case: str, window: str, recorded: float) -> None:
    inputs, turbines = _inputs_for(hot, window, case)
    tables = north(inputs, layout=hot_layout(turbines))
    assert set(tables) == set(turbines)
    assert all(np.isfinite(t["north_offset"]).all() for t in tables.values()), "a degraded case returned NaN offsets"
    errors = offset_errors(tables, default_run(hot, window), monthly(inputs.index))
    worst = max(errors, key=errors.__getitem__)
    assert errors[worst] <= 1.5 * recorded + 1.0, (
        f"{case} ({window}): {worst} is {errors[worst]:.1f} deg off the full-data answer; recorded {recorded}"
    )


class TestDegradation:
    """Fewer turbines, shorter records, both, and missing data: northing degrades, it does not fail.

    Mirrors ``study_northing_degradation``: turbines are taken as a spatially contiguous cluster (so a
    wake pair survives for wake-nadir-shift), records are truncated to their most recent days, and each result
    is scored by the worst turbine's median offset error against the full-data answer.

    What the recorded errors say: to within a few degrees down to a lone turbine or a 90-day record,
    since changepoints-v-reanalysis takes over below three turbines. Two corners are genuinely weak and are held where
    they are rather than hidden: a whole-farm record of a month or less (T15 reads ~17 deg off at the
    end of 2018, where its neighbourhood data are thin), and three turbines on 90 days in 2020.
    """

    @pytest.mark.parametrize(("case", "window"), list(RECORDED_WORST_ERROR), ids=lambda v: v)
    def test_a_degraded_case_stays_near_the_full_data_answer(self, hot: pd.DataFrame, case: str, window: str) -> None:
        _check_degraded(hot, case, window, RECORDED_WORST_ERROR[case, window])

    @pytest.mark.slow
    @pytest.mark.parametrize(("case", "window"), list(RECORDED_WORST_ERROR_SLOW), ids=lambda v: v)
    def test_a_costly_degraded_case_stays_near_the_full_data_answer(
        self, hot: pd.DataFrame, case: str, window: str
    ) -> None:
        _check_degraded(hot, case, window, RECORDED_WORST_ERROR_SLOW[case, window])

    @pytest.mark.slow
    @pytest.mark.parametrize("end", ["2019-01-01", "2019-01-03", "2019-02-01"])
    def test_where_the_record_stops_does_not_invent_a_step(self, hot: pd.DataFrame, end: str) -> None:
        """T13 once had an apparent +3.5 deg step on 2018-12-20 that existed only when the record
        stopped twelve days later. A step near the end of a record is only credible if it is big.
        """
        inputs = farm_inputs(hot, ALL_TURBINES, "2017-01-01", end)
        found = changepoints(north(inputs, layout=hot_layout())["T13"])
        assert found == [], describe(found)

    @pytest.mark.slow
    @pytest.mark.parametrize("half", ["west", "east"])
    def test_half_the_farm_finds_what_the_whole_farm_finds(self, hot: pd.DataFrame, half: str) -> None:
        turbines = tuple(f"T{n:02d}" for n in (range(1, 16) if half == "west" else range(16, 22)))
        part = default_run(hot, "early", turbines)
        whole = default_run(hot, "early")
        for name in turbines:
            assert len(changepoints(part[name])) == len(changepoints(whole[name])), (
                f"{name}: {half}={describe(changepoints(part[name]))} whole={describe(changepoints(whole[name]))}"
            )


class TestWholeFarmFallback:
    """``layout=None`` -- one whole-farm consensus, no wake-nadir-shift -- finds the same recalibrations."""

    @pytest.mark.parametrize("window", list(WINDOWS))
    def test_finds_the_published_recalibrations_and_nothing_else(self, hot: pd.DataFrame, window: str) -> None:
        tables = north(farm_inputs(hot, ALL_TURBINES, *WINDOWS[window]), layout=None, wake_nadir_shift=False)
        for name in ALL_TURBINES:
            _assert_matches(changepoints(tables[name]), EXPECTED[window].get(name, []))


class TestSingleTurbineAgainstReanalysis:
    """Northing one turbine with no farm to lean on falls back to reanalysis alone."""

    def test_a_lone_turbine_still_finds_its_large_recalibration(self, hot: pd.DataFrame) -> None:
        inputs = farm_inputs(hot, ("T16",), *EARLY)
        table = estimate_north_table(
            inputs.index, inputs.direction["T16"], reference_deg=inputs.reference, usable=inputs.usable["T16"]
        )
        found = changepoints(table)
        assert any(abs(w - pd.Timestamp("2017-05-19", tz="UTC")) <= pd.Timedelta(days=3) for w, _ in found), describe(
            found
        )

    @pytest.mark.slow
    def test_changepoints_v_reanalysis_does_not_over_detect_against_reanalysis(self, hot: pd.DataFrame) -> None:
        """Below the consensus floor each turbine is northed against reanalysis with changepoints
        (changepoints-v-reanalysis). The whole farm resolves to a handful of changepoints, not one every
        few weeks.
        """
        total = 0
        for turbine in ALL_TURBINES:
            inputs = farm_inputs(hot, (turbine,), "2017-01-01", "2021-01-01")
            total += len(changepoints(north(inputs, layout=None, wake_nadir_shift=False)[turbine]))
        assert total <= 20, (
            f"changepoints-v-reanalysis over-detected against reanalysis: {total} changepoints across the farm"
        )
