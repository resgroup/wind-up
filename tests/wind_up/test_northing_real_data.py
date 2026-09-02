"""Northing regression tests on real Hill of Towie data, end to end through both passes.

Synthetic tests pin the algorithm's contract; only real SCADA exercises what it does with site
veer, farm outages and a reference derived from the farm itself. The fixture holds the raw
inputs -- each turbine's yaw and the reanalysis direction, over the rows where the turbine was
generating -- for all 21 turbines across 2017-2020, so a test runs :func:`north_farm` exactly as
a user would rather than trusting a precomputed reference.

That distinction earned itself: an earlier version of this fixture stored the farm reference,
which had been built with the very first-pass bug these tests exist to catch, so the tests could
not see it.

Three groups, and the distinction is the point:

* **known changepoints** -- recalibrations v0's published table also records. They must keep
  being found; they are what any change to the estimator must not break.
* **edge artefacts** -- changepoints that appear only because of where the record stops.
* **outage artefacts** -- farm-wide excursions during outages, which are the weather and the
  reference moving together rather than any turbine's calibration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wind_up.circular_math import circ_diff
from wind_up.northing import estimate_north_table, north_farm

FIXTURE = Path(__file__).parents[1] / "test_data" / "hot" / "northing" / "northing_inputs.parquet"
ALL_TURBINES = tuple(f"T{n:02d}" for n in range(1, 22))

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="Hill of Towie northing fixture not available (git-lfs not pulled)"
)


@pytest.fixture(scope="module")
def hot() -> pd.DataFrame:
    """The northing fixture, loaded once for the module."""
    return pd.read_parquet(FIXTURE)


def _arrays(hot: pd.DataFrame, turbines: tuple[str, ...], start: str, end: str) -> tuple:
    """Return ``(index, direction, usable, reanalysis)`` for a window, as ``north_farm`` wants them."""
    rows = hot[
        hot["turbine"].isin(turbines)
        & (hot["timestamp"] >= pd.Timestamp(start, tz="UTC"))
        & (hot["timestamp"] < pd.Timestamp(end, tz="UTC"))
    ]
    index = pd.DatetimeIndex(sorted(rows["timestamp"].unique()))
    direction, usable, reanalysis = {}, {}, None
    for turbine in sorted(rows["turbine"].unique()):
        one = rows[rows["turbine"] == turbine].drop_duplicates("timestamp").set_index("timestamp").reindex(index)
        wd = one["era5_wd_deg"].to_numpy(dtype=float)
        reanalysis = wd if reanalysis is None else np.where(np.isfinite(reanalysis), reanalysis, wd)
        yaw = one["yaw_deg"].to_numpy(dtype=float)
        direction[str(turbine)] = yaw
        usable[str(turbine)] = np.isfinite(yaw) & np.isfinite(wd)
    return index, direction, usable, reanalysis


def _changepoints(table: pd.DataFrame) -> list[tuple[pd.Timestamp, float]]:
    offsets = table["north_offset"].to_numpy(dtype=float)
    return [(table["timestamp"].iloc[i], float(circ_diff(offsets[i], offsets[i - 1]))) for i in range(1, len(table))]


def _describe(found: list[tuple[pd.Timestamp, float]]) -> str:
    return str([(w.strftime("%Y-%m-%d"), round(s, 1)) for w, s in found])


_RUNS: dict[tuple, dict[str, list[tuple[pd.Timestamp, float]]]] = {}


def run_farm(
    hot: pd.DataFrame, turbines: tuple[str, ...], start: str, end: str
) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    """Changepoints per turbine for one (turbines, window), memoised -- each run costs seconds."""
    key = (turbines, start, end)
    if key not in _RUNS:
        index, direction, usable, reanalysis = _arrays(hot, turbines, start, end)
        tables = north_farm(index, direction_deg=direction, usable=usable, reanalysis_deg=reanalysis)
        _RUNS[key] = {name: _changepoints(table) for name, table in tables.items()}
    return _RUNS[key]


# Two two-year windows rather than one four-year one: the changepoint search costs roughly the
# cube of the record length, so this covers the same events for a quarter of the runtime.
EARLY = ("2017-01-01", "2019-01-01")
LATE = ("2019-01-01", "2021-01-01")

# Every recalibration v0's published table records in each window, and nothing else.
EXPECTED = {
    EARLY: {
        "T01": [("2017-04-23", 21.1), ("2017-05-04", -19.4)],
        "T05": [("2017-05-03", 35.7), ("2018-04-21", -19.7)],
        "T16": [("2017-05-19", 98.7), ("2017-06-18", 9.0), ("2017-08-09", -7.2)],
    },
    LATE: {
        "T11": [("2019-08-19", -4.4)],
        "T12": [("2020-06-18", 170.7)],
        "T19": [("2019-07-12", 98.8), ("2019-12-24", -122.5)],
    },
}
_CASES = [(window, turbine) for window, turbines in EXPECTED.items() for turbine in sorted(turbines)]
_QUIET = [
    (window, turbine) for window, turbines in EXPECTED.items() for turbine in ALL_TURBINES if turbine not in turbines
]


class TestKnownChangepoints:
    """All 21 turbines over two-year windows: v0's changepoints, and no others."""

    @pytest.mark.parametrize(("window", "turbine"), _CASES, ids=lambda v: v if isinstance(v, str) else v[0])
    def test_a_turbines_known_recalibrations_are_found(
        self, hot: pd.DataFrame, window: tuple[str, str], turbine: str
    ) -> None:
        found = run_farm(hot, ALL_TURBINES, *window)[turbine]
        expected = EXPECTED[window][turbine]
        assert len(found) == len(expected), f"{turbine}: {_describe(found)}"
        for (when, step), (expected_when, expected_step) in zip(found, expected, strict=True):
            assert abs(when - pd.Timestamp(expected_when, tz="UTC")) <= pd.Timedelta(days=3), _describe(found)
            assert circ_diff(step, expected_step) == pytest.approx(0.0, abs=3.0), _describe(found)

    @pytest.mark.parametrize(("window", "turbine"), _QUIET, ids=lambda v: v if isinstance(v, str) else v[0])
    def test_every_other_turbine_is_left_alone(self, hot: pd.DataFrame, window: tuple[str, str], turbine: str) -> None:
        found = run_farm(hot, ALL_TURBINES, *window)[turbine]
        assert found == [], f"{turbine}: {_describe(found)}"

    @pytest.mark.parametrize("window", [EARLY, LATE], ids=["early", "late"])
    def test_the_farm_total_matches_the_published_table(self, hot: pd.DataFrame, window: tuple[str, str]) -> None:
        """v0's rate over 21 turbines, not an order more."""
        found = run_farm(hot, ALL_TURBINES, *window)
        assert sum(len(v) for v in found.values()) == sum(len(v) for v in EXPECTED[window].values()), {
            n: _describe(v) for n, v in found.items() if v
        }


class TestOutageArtefacts:
    """Farm-wide self-cancelling excursions must not be reported.

    November 2019 and June 2020 are spells where most of the farm is down and the wind sits in a
    sector it rarely occupies. Every turbine appeared to step by 12-22 degrees and back within a
    week. The cause was the **first** pass: reanalysis carries its own direction-dependent bias,
    so an unusual spell moves every turbine's residual against it together, and correcting for
    that wrote the excursion into the northed directions and hence into the farm consensus the
    second pass trusts.
    """

    OUTAGES = (("2019-11-05", "2019-11-25"), ("2020-06-08", "2020-06-17"))

    def test_no_turbine_steps_during_a_farm_outage(self, hot: pd.DataFrame) -> None:
        during = {
            name: [
                (w, s)
                for w, s in found
                if any(pd.Timestamp(lo, tz="UTC") <= w <= pd.Timestamp(hi, tz="UTC") for lo, hi in self.OUTAGES)
            ]
            for name, found in run_farm(hot, ALL_TURBINES, *LATE).items()
        }
        offenders = {name: _describe(v) for name, v in during.items() if v}
        assert offenders == {}, f"turbines stepped with the outage, not their own calibration: {offenders}"

    def test_the_outage_years_are_quiet(self, hot: pd.DataFrame) -> None:
        """Run 2019-2020 on its own: four changepoints across 21 turbines, all in v0's table."""
        found = run_farm(hot, ALL_TURBINES, *LATE)
        total = sum(len(v) for v in found.values())
        assert total == 4, {n: _describe(v) for n, v in found.items() if v}


class TestEdgeArtefacts:
    """A step near the end of a record is only credible if it is big.

    T13 had an apparent +3.5 deg step on 2018-12-20 that existed only when the record stopped
    twelve days later; extending it by two days removed it. It is not in v0's table.
    """

    @pytest.mark.parametrize("end", ["2019-01-01", "2019-01-03", "2019-02-01"])
    def test_t13_is_clean_wherever_the_record_stops(self, hot: pd.DataFrame, end: str) -> None:
        found = run_farm(hot, ALL_TURBINES, "2017-01-01", end)["T13"]
        assert found == [], _describe(found)


class TestSubsetConsistency:
    """Analysing part of a farm must not invent changepoints the whole farm does not see."""

    WEST = tuple(f"T{n:02d}" for n in range(1, 16))
    EAST = tuple(f"T{n:02d}" for n in range(16, 22))

    @pytest.mark.parametrize("half", ["west", "east"])
    def test_half_the_farm_agrees_with_the_whole(self, hot: pd.DataFrame, half: str) -> None:
        turbines = self.WEST if half == "west" else self.EAST
        tables = run_farm(hot, turbines, *EARLY)
        reference = run_farm(hot, ALL_TURBINES, *EARLY)
        for name in turbines:
            found = tables[name]
            whole = reference[name]
            assert len(found) == len(whole), f"{name}: {half}={_describe(found)} whole={_describe(whole)}"


class TestSingleTurbineAgainstReanalysis:
    """Northing one turbine with no farm to lean on falls back to reanalysis alone."""

    def test_a_lone_turbine_still_finds_its_large_recalibration(self, hot: pd.DataFrame) -> None:
        index, direction, usable, reanalysis = _arrays(hot, ("T16",), "2017-01-01", "2019-01-01")
        table = estimate_north_table(index, direction["T16"], reference_deg=reanalysis, usable=usable["T16"])
        found = _changepoints(table)
        assert len(found) >= 1, _describe(found)
        assert any(abs(w - pd.Timestamp("2017-05-19", tz="UTC")) <= pd.Timedelta(days=3) for w, _ in found), _describe(
            found
        )
