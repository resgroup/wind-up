"""Tests for the source-native column vocabulary."""

from __future__ import annotations

import pytest

from benchmarking.synthetic import ColumnSchema

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
