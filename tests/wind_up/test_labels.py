"""Tests for the analyst's labels and the direction sector label."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_up import labels
from wind_up.labels import ALL_USES, NORTHING, UPLIFT, USES, Label, direction_sector, upgrade_state_column


def test_a_label_withholds_nothing_by_default() -> None:
    assert Label(name="in sector", column="in_sector").withholds == frozenset()


def test_a_label_takes_one_use_as_a_string_or_several_as_a_collection() -> None:
    assert Label(name="curtailed", column="c", withholds=UPLIFT).withholds == {UPLIFT}  # type: ignore[arg-type]
    assert Label(name="curtailed", column="c", withholds=[UPLIFT, NORTHING]).withholds == {UPLIFT, NORTHING}  # type: ignore[arg-type]


def test_a_label_cannot_withhold_an_unknown_use() -> None:
    with pytest.raises(ValueError, match="waking"):
        Label(name="curtailed", column="c", withholds=frozenset({"waking"}))


def test_a_label_withholding_all_uses_withholds_every_use() -> None:
    assert Label(name="tower shadow", column="c", withholds=ALL_USES).withholds == set(USES)  # type: ignore[arg-type]
    assert Label(name="tower shadow", column="c", withholds=[ALL_USES, UPLIFT]).withholds == set(USES)  # type: ignore[arg-type]


def test_all_uses_covers_a_use_added_later(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(labels, "USES", (*USES, "waking"))
    assert Label(name="tower shadow", column="c", withholds=ALL_USES).withholds == {*USES, "waking"}  # type: ignore[arg-type]


def test_all_uses_does_not_hide_an_unknown_use() -> None:
    with pytest.raises(ValueError, match="waking"):
        Label(name="tower shadow", column="c", withholds=frozenset({ALL_USES, "waking"}))


@pytest.mark.parametrize(("name", "column"), [("", "c"), ("n", " ")])
def test_a_label_needs_a_name_and_a_column(name: str, column: str) -> None:
    with pytest.raises(ValueError, match="needs a name and a column"):
        Label(name=name, column=column)


def test_each_upgrade_has_its_own_state_column() -> None:
    assert upgrade_state_column("AeroUp") != upgrade_state_column("TuneUp")
    assert "AeroUp" in upgrade_state_column("AeroUp")


def directions(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="10min", tz="UTC"))


def test_a_sector_includes_its_first_bound_and_not_its_second() -> None:
    inside = direction_sector(directions([199.9, 200.0, 220.0, 239.9, 240.0]), sector=(200.0, 240.0))
    assert inside.tolist() == [False, True, True, True, False]


def test_a_sector_may_pass_through_north() -> None:
    inside = direction_sector(directions([340.0, 350.0, 0.0, 9.9, 10.0, 180.0]), sector=(350.0, 10.0))
    assert inside.tolist() == [False, True, True, True, False, False]


def test_a_sector_wraps_directions_and_bounds_given_outside_the_circle() -> None:
    inside = direction_sector(directions([-5.0, 365.0, 725.0]), sector=(-10.0, 370.0))
    assert inside.tolist() == [True, True, True]


def test_equal_bounds_are_the_full_circle() -> None:
    assert direction_sector(directions([0.0, 123.0, 359.9]), sector=(90.0, 90.0)).all()


def test_a_direction_that_is_not_finite_is_in_no_sector() -> None:
    values = directions([np.nan, np.inf, 100.0])
    assert direction_sector(values, sector=(0.0, 360.0)).tolist() == [False, False, True]
    assert direction_sector(values, sector=(50.0, 150.0)).tolist() == [False, False, True]
    assert direction_sector(values, sector=(150.0, 120.0)).tolist() == [False, False, True]


def test_the_sector_label_keeps_the_index_and_is_boolean() -> None:
    values = directions([10.0, 20.0])
    inside = direction_sector(values, sector=(0.0, 15.0))
    assert inside.index.equals(values.index)
    assert inside.dtype == bool
