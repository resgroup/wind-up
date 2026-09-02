"""Northing regression tests on real Hill of Towie data.

Synthetic tests pin the algorithm's contract; only real SCADA exercises what it does with site
veer, outages and a reference that is itself derived from the farm. The fixture holds just what
the estimator consumes -- timestamp, raw yaw, the farm-direction reference, and whether that
reference had fallen back to reanalysis -- for six turbines over 2016-2020.

Two groups of test, and the distinction matters:

* **known changepoints** -- recalibrations that v0's published table also records. These must
  keep being found; they are what any change to the estimator must not break.
* **artefacts** -- changepoints that are not real, established by showing they appear and
  disappear with the *window* rather than with the data. A record ending days after an apparent
  step is the clearest case: extend it and the step is gone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wind_up.circular_math import circ_diff
from wind_up.northing import estimate_north_table

FIXTURE = Path(__file__).parents[1] / "test_data" / "hot" / "northing" / "northing_inputs.parquet"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="Hill of Towie northing fixture not available (git-lfs not pulled)"
)


@pytest.fixture(scope="module")
def hot() -> pd.DataFrame:
    """The northing fixture, loaded once for the module."""
    return pd.read_parquet(FIXTURE)


def _changepoints(
    hot: pd.DataFrame,
    turbine: str,
    start: str,
    end: str,
    *,
    exclude_fallback: bool = False,
) -> list[tuple[pd.Timestamp, float]]:
    """Return ``(timestamp, step_deg)`` for each changepoint the estimator finds in a window."""
    rows = hot[
        (hot["turbine"] == turbine)
        & (hot["timestamp"] >= pd.Timestamp(start, tz="UTC"))
        & (hot["timestamp"] < pd.Timestamp(end, tz="UTC"))
    ]
    usable = np.ones(len(rows), dtype=bool)
    if exclude_fallback:
        usable &= ~rows["reference_is_fallback"].to_numpy()
    table = estimate_north_table(
        pd.DatetimeIndex(rows["timestamp"]),
        rows["yaw_deg"].to_numpy(dtype=float),
        reference_deg=rows["farm_reference_deg"].to_numpy(dtype=float),
        usable=usable,
    )
    offsets = table["north_offset"].to_numpy(dtype=float)
    return [(table["timestamp"].iloc[i], float(circ_diff(offsets[i], offsets[i - 1]))) for i in range(1, len(table))]


def _assert_matches(
    found: list[tuple[pd.Timestamp, float]],
    expected: list[tuple[str, float]],
    *,
    days: float = 2.0,
    step_deg: float = 2.0,
) -> None:
    """Assert the found changepoints match ``expected`` in count, date and step size."""
    assert len(found) == len(expected), f"expected {len(expected)} changepoint(s), got {_describe(found)}"
    for (when, step), (expected_when, expected_step) in zip(found, expected, strict=True):
        assert abs(when - pd.Timestamp(expected_when, tz="UTC")) <= pd.Timedelta(days=days), _describe(found)
        assert circ_diff(step, expected_step) == pytest.approx(0.0, abs=step_deg), _describe(found)


def _describe(found: list[tuple[pd.Timestamp, float]]) -> str:
    return str([(w.strftime("%Y-%m-%d"), round(s, 1)) for w, s in found])


class TestKnownChangepoints:
    """Real recalibrations v0's published table also records. These must keep being found."""

    def test_t01_two_steps_in_spring_2017(self, hot: pd.DataFrame) -> None:
        _assert_matches(
            _changepoints(hot, "T01", "2017-01-01", "2019-01-01"),
            [("2017-04-23", 21.0), ("2017-05-04", -19.2)],
        )

    def test_t05_a_large_step_then_a_partial_reversal_a_year_later(self, hot: pd.DataFrame) -> None:
        _assert_matches(
            _changepoints(hot, "T05", "2017-01-01", "2019-01-01"),
            [("2017-05-03", 35.8), ("2018-04-21", -19.7)],
        )

    def test_t16_a_ninety_degree_recalibration_and_two_small_follow_ups(self, hot: pd.DataFrame) -> None:
        """The large step and its near-reversal must both survive: size is what makes them real."""
        _assert_matches(
            _changepoints(hot, "T16", "2017-01-01", "2019-01-01"),
            [("2017-05-19", 98.6), ("2017-06-18", 9.0), ("2017-08-09", -7.2)],
        )

    @pytest.mark.parametrize("turbine", ["T07", "T11"])
    def test_a_stable_turbine_gets_no_changepoints(self, hot: pd.DataFrame, turbine: str) -> None:
        found = _changepoints(hot, turbine, "2017-01-01", "2019-01-01")
        assert found == [], _describe(found)


class TestEdgeArtefacts:
    """A step near the end of a record is only credible if it is big.

    T13 is the case: an apparent +3.5 deg step on 2018-12-20 that exists only when the record
    stops twelve days later. It is not in v0's published table, and extending the record by two
    days removes it -- a step in the data would not care where the record happens to end.
    """

    def test_a_small_step_just_before_the_record_ends_is_not_reported(self, hot: pd.DataFrame) -> None:
        found = _changepoints(hot, "T13", "2017-01-01", "2019-01-01")
        assert found == [], _describe(found)

    def test_the_same_step_with_a_month_of_data_after_it_is_not_reported(self, hot: pd.DataFrame) -> None:
        found = _changepoints(hot, "T13", "2017-01-01", "2019-02-01")
        assert found == [], _describe(found)

    def test_extending_the_record_by_two_days_already_removed_it(self, hot: pd.DataFrame) -> None:
        """The control: this window has always been clean, and must stay clean."""
        found = _changepoints(hot, "T13", "2017-01-01", "2019-01-03")
        assert found == [], _describe(found)

    def test_a_clean_two_year_window_stays_clean(self, hot: pd.DataFrame) -> None:
        found = _changepoints(hot, "T13", "2016-01-01", "2018-01-01")
        assert found == [], _describe(found)


class TestReferenceFallback:
    """Where the farm reference silently became reanalysis, the residual is not comparable.

    v0's ``add_wf_yawdir`` fills a missing farm direction with reanalysis, which sits degrees
    away from the farm consensus, so every turbine appears to step together. Dropping those rows
    is the caller's job -- the estimator only sees ``usable``.
    """

    def test_excluding_fallback_rows_removes_the_august_2020_pair(self, hot: pd.DataFrame) -> None:
        window = ("2019-06-01", "2020-09-01")
        with_fallback = _changepoints(hot, "T13", *window)
        without = _changepoints(hot, "T13", *window, exclude_fallback=True)
        assert len(without) < len(with_fallback), f"{_describe(with_fallback)} -> {_describe(without)}"
        assert not any(w >= pd.Timestamp("2020-08-01", tz="UTC") for w, _ in without), _describe(without)

    @pytest.mark.xfail(
        reason="the farm reference is not one quantity when its composition changes; see the "
        "'reference composition' limitation in the R1 design",
        strict=True,
    )
    def test_the_farm_wide_outage_excursions_should_not_be_reported(self, hot: pd.DataFrame) -> None:
        """Nov 2019 and Jun 2020: nearly every turbine steps together and back within ~8 days.

        Those weeks are farm outages. With few turbines reporting, the farm median is taken over a
        different subset than usual, and since turbines have different veer signatures the
        reference itself shifts -- so every turbine appears to step. Excluding the rows where the
        reference fell back to reanalysis removes some of it but not these two pairs, because the
        fallback never triggers: three turbines still report, just not the usual three.
        """
        found = _changepoints(hot, "T13", "2019-06-01", "2020-09-01", exclude_fallback=True)
        assert found == [], _describe(found)
