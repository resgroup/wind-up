"""Tests for injected data faults: measurement corruptions that leave ground truth alone."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from benchmarking.synthetic import (
    HOT_COLUMNS,
    ConstantCpChange,
    Fault,
    NorthingStep,
    ReferenceCpChange,
    SensorGainDrift,
    SensorGainStep,
    generate_dataset,
)

_COLUMNS = HOT_COLUMNS
_TURBINES = ("T01", "T02", "T03")
_START = pd.Timestamp("2018-01-01", tz="UTC")
_CHANGEOVER = _START + pd.Timedelta(days=30)
_FAULT_AT = _START + pd.Timedelta(days=45)


def _index(days: int = 60) -> pd.DatetimeIndex:
    return pd.date_range(start=_START, periods=days * 24, freq="3600s", tz="UTC")


def _scada(index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = [
        pd.DataFrame(
            {
                _COLUMNS.turbine: turbine,
                _COLUMNS.active_power: rng.uniform(200, 2000, len(index)),
                _COLUMNS.active_power_min: 100.0,
                _COLUMNS.wind_speed: rng.uniform(4, 14, len(index)),
                _COLUMNS.wind_speed_sd: 1.0,
                _COLUMNS.gen_rpm: 1400.0,
                _COLUMNS.availability: 3600.0,
                _COLUMNS.nacelle_position: rng.uniform(0, 360, len(index)),
            },
            index=index,
        )
        for turbine in _TURBINES
    ]
    return pd.concat(frames)


def _generate(faults: list) -> tuple:
    index = _index()
    scada = _scada(index)
    dataset = generate_dataset(
        scada_df=scada,
        test_wtgs=["T01"],
        upgrades=[ConstantCpChange(delta=0.05)],
        mode="prepost",
        upgrade_timing=_CHANGEOVER,
        faults=faults,
        columns=_COLUMNS,
    )
    return dataset, scada


def _direction(frame: pd.DataFrame, turbine: str) -> pd.Series:
    rows = frame[frame[_COLUMNS.turbine] == turbine]
    return rows[_COLUMNS.nacelle_position]


class TestNorthingStep:
    def test_shifts_the_named_turbines_direction_from_the_step_date(self) -> None:
        dataset, scada = _generate([NorthingStep(turbine="T02", at=_FAULT_AT, offset_deg=40.0)])

        before = _direction(dataset.synthetic_df, "T02").loc[:_FAULT_AT].iloc[:-1]
        clean_before = _direction(scada, "T02").loc[:_FAULT_AT].iloc[:-1]
        assert before.to_numpy() == pytest.approx(clean_before.to_numpy())

        after = _direction(dataset.synthetic_df, "T02").loc[_FAULT_AT:]
        clean_after = _direction(scada, "T02").loc[_FAULT_AT:]
        assert after.to_numpy() == pytest.approx((clean_after.to_numpy() + 40.0) % 360.0)

    def test_leaves_other_turbines_alone(self) -> None:
        dataset, scada = _generate([NorthingStep(turbine="T02", at=_FAULT_AT, offset_deg=40.0)])
        for turbine in ("T01", "T03"):
            assert _direction(dataset.synthetic_df, turbine).to_numpy() == pytest.approx(
                _direction(scada, turbine).to_numpy()
            ), turbine

    def test_changes_no_power_so_the_true_uplift_is_untouched(self) -> None:
        """The fault is a measurement corruption; ground truth must not move at all."""
        clean, _ = _generate([])
        faulted, _ = _generate([NorthingStep(turbine="T02", at=_FAULT_AT, offset_deg=40.0)])

        assert faulted.true_uplift().overall == pytest.approx(clean.true_uplift().overall, abs=1e-12)
        assert faulted.synthetic_df[_COLUMNS.active_power].to_numpy() == pytest.approx(
            clean.synthetic_df[_COLUMNS.active_power].to_numpy()
        )

    def test_the_untouched_original_never_carries_the_fault(self) -> None:
        """``original_df`` is the truth reference: a method's corrupted view must not reach it."""
        dataset, scada = _generate([NorthingStep(turbine="T02", at=_FAULT_AT, offset_deg=40.0)])
        assert _direction(dataset.original_df, "T02").to_numpy() == pytest.approx(_direction(scada, "T02").to_numpy())

    def test_wraps_past_360(self) -> None:
        dataset, scada = _generate([NorthingStep(turbine="T02", at=_START, offset_deg=350.0)])
        got = _direction(dataset.synthetic_df, "T02").to_numpy()
        assert got.min() >= 0.0
        assert got.max() < 360.0
        assert got == pytest.approx((_direction(scada, "T02").to_numpy() + 350.0) % 360.0)

    def test_several_faults_compose(self) -> None:
        dataset, scada = _generate(
            [
                NorthingStep(turbine="T02", at=_FAULT_AT, offset_deg=40.0),
                NorthingStep(turbine="T03", at=_FAULT_AT, offset_deg=-25.0),
            ]
        )
        after = _direction(dataset.synthetic_df, "T03").loc[_FAULT_AT:].to_numpy()
        clean = _direction(scada, "T03").loc[_FAULT_AT:].to_numpy()
        assert after == pytest.approx((clean - 25.0) % 360.0)

    def test_is_recorded_in_run_metadata(self) -> None:
        dataset, _ = _generate([NorthingStep(turbine="T02", at=_FAULT_AT, offset_deg=40.0)])
        assert dataset.run_metadata["faults"] == [
            {"kind": "northing_step", "turbine": "T02", "at": str(_FAULT_AT), "offset_deg": 40.0}
        ]

    def test_no_faults_is_the_default_and_records_an_empty_list(self) -> None:
        dataset, scada = _generate([])
        assert dataset.run_metadata["faults"] == []
        assert dataset.synthetic_df[_COLUMNS.nacelle_position].to_numpy() == pytest.approx(
            scada[_COLUMNS.nacelle_position].to_numpy()
        )

    def test_an_unknown_turbine_raises(self) -> None:
        with pytest.raises(ValueError, match="T99"):
            _generate([NorthingStep(turbine="T99", at=_FAULT_AT, offset_deg=40.0)])

    def test_a_missing_direction_column_raises_naming_the_role(self) -> None:
        index = _index(days=10)
        scada = _scada(index).drop(columns=[_COLUMNS.nacelle_position])
        with pytest.raises(ValueError, match=_COLUMNS.nacelle_position):
            generate_dataset(
                scada_df=scada,
                test_wtgs=["T01"],
                upgrades=[],
                mode="prepost",
                upgrade_timing=_CHANGEOVER,
                faults=[NorthingStep(turbine="T02", at=_FAULT_AT, offset_deg=40.0)],
                columns=_COLUMNS,
            )


def _wind_speed(frame: pd.DataFrame, turbine: str) -> pd.Series:
    rows = frame[frame[_COLUMNS.turbine] == turbine]
    return rows[_COLUMNS.wind_speed]


def _wind_speed_sd(frame: pd.DataFrame, turbine: str) -> pd.Series:
    rows = frame[frame[_COLUMNS.turbine] == turbine]
    return rows[_COLUMNS.wind_speed_sd]


class TestSensorGainStep:
    def test_scales_the_named_turbines_wind_speed_from_the_step_date(self) -> None:
        dataset, scada = _generate([SensorGainStep(turbine="T02", at=_FAULT_AT, gain=1.5)])

        before = _wind_speed(dataset.synthetic_df, "T02").loc[:_FAULT_AT].iloc[:-1]
        clean_before = _wind_speed(scada, "T02").loc[:_FAULT_AT].iloc[:-1]
        assert before.to_numpy() == pytest.approx(clean_before.to_numpy())

        after = _wind_speed(dataset.synthetic_df, "T02").loc[_FAULT_AT:]
        clean_after = _wind_speed(scada, "T02").loc[_FAULT_AT:]
        assert after.to_numpy() == pytest.approx(clean_after.to_numpy() * 1.5)

    def test_scales_the_standard_deviation_by_the_same_gain(self) -> None:
        dataset, scada = _generate([SensorGainStep(turbine="T02", at=_FAULT_AT, gain=0.5)])
        after = _wind_speed_sd(dataset.synthetic_df, "T02").loc[_FAULT_AT:]
        clean_after = _wind_speed_sd(scada, "T02").loc[_FAULT_AT:]
        assert after.to_numpy() == pytest.approx(clean_after.to_numpy() * 0.5)

    def test_leaves_turbulence_intensity_invariant(self) -> None:
        """Scaling mean and SD together is a calibration-gain error: only the ws axis moves."""
        dataset, scada = _generate([SensorGainStep(turbine="T02", at=_FAULT_AT, gain=1.5)])
        faulted_ti = _wind_speed_sd(dataset.synthetic_df, "T02") / _wind_speed(dataset.synthetic_df, "T02")
        clean_ti = _wind_speed_sd(scada, "T02") / _wind_speed(scada, "T02")
        assert faulted_ti.to_numpy() == pytest.approx(clean_ti.to_numpy())

    def test_leaves_other_turbines_alone(self) -> None:
        dataset, scada = _generate([SensorGainStep(turbine="T02", at=_FAULT_AT, gain=1.5)])
        for turbine in ("T01", "T03"):
            assert _wind_speed(dataset.synthetic_df, turbine).to_numpy() == pytest.approx(
                _wind_speed(scada, turbine).to_numpy()
            ), turbine

    def test_changes_no_power_so_the_true_uplift_is_untouched(self) -> None:
        clean, _ = _generate([])
        faulted, _ = _generate([SensorGainStep(turbine="T02", at=_FAULT_AT, gain=1.5)])

        assert faulted.true_uplift().overall == pytest.approx(clean.true_uplift().overall, abs=1e-12)
        assert faulted.synthetic_df[_COLUMNS.active_power].to_numpy() == pytest.approx(
            clean.synthetic_df[_COLUMNS.active_power].to_numpy()
        )

    def test_the_untouched_original_never_carries_the_fault(self) -> None:
        dataset, scada = _generate([SensorGainStep(turbine="T02", at=_FAULT_AT, gain=1.5)])
        assert _wind_speed(dataset.original_df, "T02").to_numpy() == pytest.approx(_wind_speed(scada, "T02").to_numpy())

    def test_scales_only_the_named_roles(self) -> None:
        dataset, scada = _generate([SensorGainStep(turbine="T02", at=_FAULT_AT, gain=1.5, roles=("wind_speed",))])
        assert _wind_speed_sd(dataset.synthetic_df, "T02").to_numpy() == pytest.approx(
            _wind_speed_sd(scada, "T02").to_numpy()
        )

    def test_a_role_the_schema_leaves_unset_is_skipped(self) -> None:
        """Sources differ in which channels they carry; an absent role is not a configuration error."""
        columns = dataclasses.replace(_COLUMNS, pitch=None)
        index = _index(days=10)
        scada = _scada(index)
        dataset = generate_dataset(
            scada_df=scada,
            test_wtgs=["T01"],
            upgrades=[],
            mode="prepost",
            upgrade_timing=_CHANGEOVER,
            faults=[SensorGainStep(turbine="T02", at=_FAULT_AT, gain=1.5, roles=("wind_speed", "pitch"))],
            columns=columns,
        )
        after = _wind_speed(dataset.synthetic_df, "T02").loc[_FAULT_AT:]
        assert after.to_numpy() == pytest.approx(_wind_speed(scada, "T02").loc[_FAULT_AT:].to_numpy() * 1.5)

    def test_a_repeated_role_scales_the_channel_once(self) -> None:
        """roles is public and a schema may alias two roles onto one column; gain is not squared."""
        dataset, scada = _generate(
            [SensorGainStep(turbine="T02", at=_FAULT_AT, gain=1.5, roles=("wind_speed", "wind_speed"))]
        )
        after = _wind_speed(dataset.synthetic_df, "T02").loc[_FAULT_AT:]
        clean_after = _wind_speed(scada, "T02").loc[_FAULT_AT:]
        assert after.to_numpy() == pytest.approx(clean_after.to_numpy() * 1.5)

    def test_is_recorded_in_run_metadata(self) -> None:
        dataset, _ = _generate([SensorGainStep(turbine="T02", at=_FAULT_AT, gain=1.5)])
        assert dataset.run_metadata["faults"] == [
            {
                "kind": "sensor_gain_step",
                "turbine": "T02",
                "at": str(_FAULT_AT),
                "gain": 1.5,
                "roles": ["wind_speed", "wind_speed_sd"],
            }
        ]

    def test_an_unknown_turbine_raises(self) -> None:
        with pytest.raises(ValueError, match="T99"):
            _generate([SensorGainStep(turbine="T99", at=_FAULT_AT, gain=1.5)])

    def test_a_missing_column_raises_naming_it(self) -> None:
        index = _index(days=10)
        scada = _scada(index).drop(columns=[_COLUMNS.wind_speed_sd])
        with pytest.raises(ValueError, match=_COLUMNS.wind_speed_sd):
            generate_dataset(
                scada_df=scada,
                test_wtgs=["T01"],
                upgrades=[],
                mode="prepost",
                upgrade_timing=_CHANGEOVER,
                faults=[SensorGainStep(turbine="T02", at=_FAULT_AT, gain=1.5)],
                columns=_COLUMNS,
            )


class TestSensorGainDrift:
    def test_ramps_from_no_gain_at_the_start_to_the_full_gain_at_the_end(self) -> None:
        dataset, scada = _generate([SensorGainDrift(turbine="T02", gain=1.5)])
        faulted = _wind_speed(dataset.synthetic_df, "T02")
        clean = _wind_speed(scada, "T02")

        assert faulted.to_numpy()[0] == pytest.approx(clean.to_numpy()[0])
        assert faulted.to_numpy()[-1] == pytest.approx(clean.to_numpy()[-1] * 1.5)

    def test_reaches_the_halfway_gain_at_the_midpoint(self) -> None:
        dataset, scada = _generate([SensorGainDrift(turbine="T02", gain=1.5)])
        faulted = _wind_speed(dataset.synthetic_df, "T02")
        clean = _wind_speed(scada, "T02")
        midpoint = len(clean) // 2

        expected = clean.to_numpy()[midpoint] * (1.0 + 0.5 * (1.5 - 1.0))
        assert faulted.to_numpy()[midpoint] == pytest.approx(expected, rel=1e-3)

    def test_scales_the_standard_deviation_on_the_same_ramp(self) -> None:
        dataset, scada = _generate([SensorGainDrift(turbine="T02", gain=0.5)])
        faulted_ti = _wind_speed_sd(dataset.synthetic_df, "T02") / _wind_speed(dataset.synthetic_df, "T02")
        clean_ti = _wind_speed_sd(scada, "T02") / _wind_speed(scada, "T02")
        assert faulted_ti.to_numpy() == pytest.approx(clean_ti.to_numpy())

    def test_leaves_other_turbines_alone(self) -> None:
        dataset, scada = _generate([SensorGainDrift(turbine="T02", gain=1.5)])
        for turbine in ("T01", "T03"):
            assert _wind_speed(dataset.synthetic_df, turbine).to_numpy() == pytest.approx(
                _wind_speed(scada, turbine).to_numpy()
            ), turbine

    def test_changes_no_power_so_the_true_uplift_is_untouched(self) -> None:
        clean, _ = _generate([])
        faulted, _ = _generate([SensorGainDrift(turbine="T02", gain=1.5)])

        assert faulted.true_uplift().overall == pytest.approx(clean.true_uplift().overall, abs=1e-12)
        assert faulted.synthetic_df[_COLUMNS.active_power].to_numpy() == pytest.approx(
            clean.synthetic_df[_COLUMNS.active_power].to_numpy()
        )

    def test_is_recorded_in_run_metadata(self) -> None:
        dataset, _ = _generate([SensorGainDrift(turbine="T02", gain=0.5)])
        assert dataset.run_metadata["faults"] == [
            {
                "kind": "sensor_gain_drift",
                "turbine": "T02",
                "gain": 0.5,
                "roles": ["wind_speed", "wind_speed_sd"],
            }
        ]

    def test_an_unknown_turbine_raises(self) -> None:
        with pytest.raises(ValueError, match="T99"):
            _generate([SensorGainDrift(turbine="T99", gain=1.5)])


def _power(frame: pd.DataFrame, turbine: str) -> pd.Series:
    rows = frame[frame[_COLUMNS.turbine] == turbine]
    return rows[_COLUMNS.active_power]


class TestReferenceCpChange:
    def test_raises_the_named_references_power_from_the_change_date(self) -> None:
        dataset, scada = _generate([ReferenceCpChange(turbine="T02", at=_FAULT_AT, delta=0.03)])

        before = _power(dataset.synthetic_df, "T02").loc[:_FAULT_AT].iloc[:-1]
        clean_before = _power(scada, "T02").loc[:_FAULT_AT].iloc[:-1]
        assert before.to_numpy() == pytest.approx(clean_before.to_numpy())

        after = _power(dataset.synthetic_df, "T02").loc[_FAULT_AT:].to_numpy()
        clean_after = _power(scada, "T02").loc[_FAULT_AT:].to_numpy()
        assert (after > clean_after).all()
        # Only the region-2 fraction of each record responds, so the mean gain is under the full 3%.
        assert 1.0 < after.mean() / clean_after.mean() < 1.03

    def test_a_negative_delta_lowers_the_references_power(self) -> None:
        dataset, scada = _generate([ReferenceCpChange(turbine="T02", at=_FAULT_AT, delta=-0.03)])
        after = _power(dataset.synthetic_df, "T02").loc[_FAULT_AT:].to_numpy()
        clean_after = _power(scada, "T02").loc[_FAULT_AT:].to_numpy()
        assert (after < clean_after).all()

    def test_leaves_other_turbines_alone(self) -> None:
        dataset, _ = _generate([ReferenceCpChange(turbine="T02", at=_FAULT_AT, delta=0.03)])
        clean_dataset, _ = _generate([])
        for turbine in ("T01", "T03"):
            assert _power(dataset.synthetic_df, turbine).to_numpy() == pytest.approx(
                _power(clean_dataset.synthetic_df, turbine).to_numpy()
            ), turbine

    def test_leaves_the_test_turbines_true_uplift_untouched(self) -> None:
        """The change is real power on a reference; the test turbine's ground truth must not move."""
        clean, _ = _generate([])
        faulted, _ = _generate([ReferenceCpChange(turbine="T02", at=_FAULT_AT, delta=0.03)])
        assert faulted.true_uplift().overall == pytest.approx(clean.true_uplift().overall, abs=1e-12)

    def test_the_untouched_original_never_carries_the_change(self) -> None:
        dataset, scada = _generate([ReferenceCpChange(turbine="T02", at=_FAULT_AT, delta=0.03)])
        assert _power(dataset.original_df, "T02").to_numpy() == pytest.approx(_power(scada, "T02").to_numpy())

    def test_targeting_a_test_turbine_raises(self) -> None:
        """A power-changing fault on a test turbine would be absorbed into its derived truth."""
        with pytest.raises(ValueError, match="T01"):
            _generate([ReferenceCpChange(turbine="T01", at=_FAULT_AT, delta=0.03)])

    def test_an_unknown_turbine_raises(self) -> None:
        with pytest.raises(ValueError, match="T99"):
            _generate([ReferenceCpChange(turbine="T99", at=_FAULT_AT, delta=0.03)])

    def test_is_recorded_in_run_metadata(self) -> None:
        dataset, _ = _generate([ReferenceCpChange(turbine="T02", at=_FAULT_AT, delta=0.03)])
        assert dataset.run_metadata["faults"] == [
            {"kind": "reference_cp_change", "turbine": "T02", "at": str(_FAULT_AT), "delta": 0.03}
        ]

    def test_declares_that_it_changes_power(self) -> None:
        assert ReferenceCpChange(turbine="T02", at=_FAULT_AT, delta=0.03).changes_power

    @pytest.mark.parametrize(
        "fault",
        [
            NorthingStep(turbine="T02", at=_FAULT_AT, offset_deg=40.0),
            SensorGainStep(turbine="T02", at=_FAULT_AT, gain=1.5),
            SensorGainDrift(turbine="T02", gain=1.5),
        ],
    )
    def test_the_reading_only_faults_declare_that_they_do_not(self, fault: Fault) -> None:
        assert not fault.changes_power
