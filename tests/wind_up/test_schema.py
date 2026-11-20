"""Tests for the source-native column vocabulary."""

from __future__ import annotations

import dataclasses

import pytest

from benchmarking.synthetic import ColumnSchema
from wind_up.schema import ROLE_RANGES, ROLE_RANGES_RATED, ROLE_UNITS

_COLUMNS = ColumnSchema(
    turbine="TurbineName",
    active_power="ActivePowerMean",
    wind_speed="WindSpeedMean",
    wind_speed_sd="WindSpeedSD",
    gen_rpm="GenRpmMean",
    availability="Availability",
    nacelle_position="YawAngleMean",
)


class TestNorthed:
    def test_prefixes_the_source_name_of_a_set_role(self) -> None:
        assert _COLUMNS.northed("nacelle_position") == "northed_YawAngleMean"

    def test_raises_for_an_unset_role(self) -> None:
        with pytest.raises(ValueError, match="pitch"):
            _COLUMNS.northed("pitch")

    def test_raises_for_an_unknown_role(self) -> None:
        with pytest.raises(AttributeError):
            _COLUMNS.northed("not_a_role")


class TestUnitsAndRanges:
    def test_every_signal_role_has_a_unit(self) -> None:
        roles = {f.name for f in dataclasses.fields(ColumnSchema)} - {"turbine"}
        assert set(ROLE_UNITS) == roles

    def test_each_range_is_of_a_known_role_and_runs_low_to_high(self) -> None:
        for ranges in (ROLE_RANGES, ROLE_RANGES_RATED):
            for role, (low, high) in ranges.items():
                assert role in ROLE_UNITS
                assert low < high

    def test_a_role_has_at_most_one_range_and_some_roles_are_left_to_other_rules(self) -> None:
        assert not set(ROLE_RANGES) & set(ROLE_RANGES_RATED)
        checked = set(ROLE_RANGES) | set(ROLE_RANGES_RATED)
        assert not checked & {"active_power", "availability", "power_setpoint"}
