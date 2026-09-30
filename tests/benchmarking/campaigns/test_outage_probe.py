"""Tests for the R4 outage probe's data transforms and its arm declaration.

The probe's actual runs are a driver (they need the Hill of Towie download and the power model);
what is unit-tested here is that each transform removes exactly what it claims -- the distinction
between an absent column and an empty one is the whole point of the matrix -- and that the arms are
declared as intended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns.outage_probe import (
    BASELINE_OUTAGE,
    OUTAGE_REFERENCE,
    PROBE_REFERENCES,
    PROBE_TEST_WTG,
    PROBE_TURBINES,
    drop_columns,
    drop_era5_rows,
    drop_rows,
    drop_turbine,
    null_era5,
    null_signals,
    probe_arms,
)
from benchmarking.campaigns.placebo import PLACEBO_CAMPAIGN_START
from benchmarking.synthetic import HOT_COLUMNS

_POWER = HOT_COLUMNS.active_power
_WS = HOT_COLUMNS.wind_speed
_WINDOW = (pd.Timestamp("2018-03-01", tz="UTC"), pd.Timestamp("2018-04-01", tz="UTC"))


# Spans the whole placebo record, so every arm's real outage window lands inside these frames.
_SPAN = ("2017-01-01", "2019-01-01")


def _scada() -> pd.DataFrame:
    """Long SCADA over the placebo record for the probe turbines, every value column finite."""
    idx = pd.date_range(*_SPAN, freq="D", tz="UTC", name="timestamp")
    values = {
        _POWER: 800.0,
        _WS: 8.0,
        str(HOT_COLUMNS.active_power_min): 700.0,
        str(HOT_COLUMNS.wind_speed_sd): 1.0,
        str(HOT_COLUMNS.gen_rpm): 1500.0,
        str(HOT_COLUMNS.availability): 600.0,
        str(HOT_COLUMNS.nacelle_position): 180.0,
    }
    frames = [pd.DataFrame({HOT_COLUMNS.turbine: name, **values}, index=idx) for name in PROBE_TURBINES]
    return pd.concat(frames)


def _era5() -> pd.DataFrame:
    """Hourly ERA5 carrying every column the probe's arms name, so each one has something to remove."""
    idx = pd.date_range(*_SPAN, freq="6h", tz="UTC", name="timestamp")
    values = {
        "wind_speed_100m": 9.0,
        "wind_direction_100m": 210.0,
        "wind_gusts_10m": 12.0,
        "temperature_2m": 5.0,
    }
    return pd.DataFrame(values, index=idx)


def _in_window(df: pd.DataFrame) -> np.ndarray:
    idx = pd.DatetimeIndex(df.index)
    return np.asarray((idx >= _WINDOW[0]) & (idx < _WINDOW[1]))


class TestNullSignals:
    def test_nulls_only_the_named_turbine_inside_the_window(self) -> None:
        out = null_signals(_scada(), turbines=[OUTAGE_REFERENCE], columns=[_POWER], window=_WINDOW)
        hit = _in_window(out) & (out[HOT_COLUMNS.turbine] == OUTAGE_REFERENCE).to_numpy()
        assert out.loc[hit, _POWER].isna().all()
        assert out.loc[~hit, _POWER].notna().all()

    def test_leaves_the_column_present_and_the_rows_in_place(self) -> None:
        """An outage empties values; it does not remove the column or shorten the frame."""
        scada = _scada()
        out = null_signals(scada, turbines=[OUTAGE_REFERENCE], columns=[_POWER], window=_WINDOW)
        assert _POWER in out.columns
        assert len(out) == len(scada)

    def test_leaves_unnamed_columns_alone(self) -> None:
        out = null_signals(_scada(), turbines=[OUTAGE_REFERENCE], columns=[_POWER], window=_WINDOW)
        assert out[_WS].notna().all()

    def test_a_column_absent_from_the_frame_is_skipped_not_created(self) -> None:
        out = null_signals(_scada(), turbines=[OUTAGE_REFERENCE], columns=["not_a_column"], window=_WINDOW)
        assert "not_a_column" not in out.columns

    def test_does_not_mutate_the_input(self) -> None:
        scada = _scada()
        null_signals(scada, turbines=[OUTAGE_REFERENCE], columns=[_POWER], window=_WINDOW)
        assert scada[_POWER].notna().all()


class TestDropRows:
    def test_removes_only_the_named_turbine_inside_the_window(self) -> None:
        scada = _scada()
        out = drop_rows(scada, turbines=[OUTAGE_REFERENCE], window=_WINDOW)
        remaining = out[out[HOT_COLUMNS.turbine] == OUTAGE_REFERENCE]
        assert not _in_window(remaining).any()
        for other in set(PROBE_TURBINES) - {OUTAGE_REFERENCE}:
            assert (out[HOT_COLUMNS.turbine] == other).sum() == (scada[HOT_COLUMNS.turbine] == other).sum()

    def test_the_window_is_half_open(self) -> None:
        """The end timestamp survives, so two adjacent windows do not overlap."""
        out = drop_rows(_scada(), turbines=[OUTAGE_REFERENCE], window=_WINDOW)
        kept = out[out[HOT_COLUMNS.turbine] == OUTAGE_REFERENCE]
        assert _WINDOW[1] in kept.index
        assert _WINDOW[0] not in kept.index


class TestDropColumnsAndTurbine:
    def test_drop_columns_removes_the_column_entirely(self) -> None:
        out = drop_columns(_scada(), columns=[_POWER])
        assert _POWER not in out.columns
        assert _WS in out.columns

    def test_drop_columns_tolerates_a_column_that_is_not_there(self) -> None:
        out = drop_columns(_scada(), columns=["not_a_column"])
        assert len(out.columns) == len(_scada().columns)

    def test_drop_turbine_removes_every_record_for_it(self) -> None:
        out = drop_turbine(_scada(), turbine=OUTAGE_REFERENCE)
        assert OUTAGE_REFERENCE not in set(out[HOT_COLUMNS.turbine])
        assert set(out[HOT_COLUMNS.turbine]) == set(PROBE_TURBINES) - {OUTAGE_REFERENCE}


class TestEra5Transforms:
    def test_null_era5_empties_every_column_in_the_window_only(self) -> None:
        out = null_era5(_era5(), window=_WINDOW)
        assert out[_in_window(out)].isna().all().all()
        assert out[~_in_window(out)].notna().all().all()

    def test_null_era5_keeps_the_index_intact(self) -> None:
        era5 = _era5()
        assert null_era5(era5, window=_WINDOW).index.equals(era5.index)

    def test_drop_era5_rows_removes_the_timestamps_from_the_index(self) -> None:
        out = drop_era5_rows(_era5(), window=_WINDOW)
        assert not _in_window(out).any()
        assert len(out) < len(_era5())


class TestProbeArms:
    def test_the_clean_control_comes_first_and_is_the_only_one(self) -> None:
        arms = probe_arms()
        assert arms[0].name == "clean"
        assert len([a for a in arms if a.shape == "clean"]) == 1

    def test_arm_names_are_unique(self) -> None:
        names = [a.name for a in probe_arms()]
        assert len(names) == len(set(names))

    def test_every_arm_declares_a_shape_and_a_description(self) -> None:
        for arm in probe_arms():
            assert arm.shape in {"clean", "absent", "empty"}
            assert arm.what

    def test_both_shapes_are_covered(self) -> None:
        """The matrix is only informative if it runs absent and empty columns against each other."""
        shapes = {a.shape for a in probe_arms()}
        assert {"absent", "empty"} <= shapes

    @pytest.mark.parametrize("arm", probe_arms(), ids=lambda a: a.name)
    def test_every_arm_changes_the_data_it_claims_to(self, arm) -> None:  # noqa: ANN001 - the Arm dataclass
        """A cell that silently transforms nothing would read as a pass it never earned."""
        scada, era5 = _scada(), _era5()
        changed = not arm.scada(scada).equals(scada) or not arm.era5(era5).equals(era5)
        assert changed == (arm.shape != "clean")

    def test_the_probe_turbines_are_the_test_turbine_plus_its_references(self) -> None:
        assert (PROBE_TEST_WTG, *PROBE_REFERENCES) == PROBE_TURBINES
        assert OUTAGE_REFERENCE in PROBE_REFERENCES

    def test_the_baseline_window_sits_before_the_changeover(self) -> None:
        assert BASELINE_OUTAGE[0] < BASELINE_OUTAGE[1] <= PLACEBO_CAMPAIGN_START
