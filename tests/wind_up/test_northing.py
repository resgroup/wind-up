"""Tests for the northing estimator core."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import yaml

from wind_up.circular_math import circ_diff
from wind_up.northing import (
    DEFAULT_NORTHING,
    NorthingSettings,
    apply_north_table,
    estimate_north_table,
    north_farm,
    veer_normalised,
    write_north_table_yaml,
    yaw_usable,
)

if TYPE_CHECKING:
    from pathlib import Path

TIMEBASE_S = 600
RATED_POWER = 2300.0


def _index(days: float = 400.0, start: str = "2017-01-01") -> pd.DatetimeIndex:
    """A 10-minute UTC index spanning ``days``."""
    periods = round(days * 24 * 3600 / TIMEBASE_S)
    return pd.date_range(start=start, periods=periods, freq=f"{TIMEBASE_S}s", tz="UTC")


def _true_direction(index: pd.DatetimeIndex, *, seed: int = 0) -> np.ndarray:
    """A plausible site wind direction: a slow random walk covering the whole compass."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 2.0, size=len(index))
    return np.cumsum(steps) % 360.0


def _stepped_offset(index: pd.DatetimeIndex, steps: list[tuple[str, float]]) -> np.ndarray:
    """Step-applied offset (deg) over ``index`` from ``(timestamp, offset)`` pairs."""
    out = np.full(len(index), steps[0][1], dtype=float)
    for when, offset in steps[1:]:
        out[index >= pd.Timestamp(when, tz="UTC")] = offset
    return out


def _reported(
    index: pd.DatetimeIndex,
    *,
    steps: list[tuple[str, float]],
    noise_deg: float = 6.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(reported_direction, reference_direction)`` for a turbine miscalibrated by ``steps``.

    The reference is the true site direction; the turbine reports it minus its north offset,
    plus per-record yaw scatter. Recovering ``steps`` from the pair is the estimator's job.
    """
    reference = _true_direction(index, seed=seed)
    rng = np.random.default_rng(seed + 1)
    scatter = rng.normal(0.0, noise_deg, size=len(index))
    reported = (reference + scatter - _stepped_offset(index, steps)) % 360.0
    return reported, reference


def _all_usable(index: pd.DatetimeIndex) -> np.ndarray:
    return np.ones(len(index), dtype=bool)


class TestEstimateNorthTable:
    def test_recovers_a_single_known_step(self) -> None:
        index = _index()
        steps = [("2017-01-01", 12.0), ("2017-08-01", 47.0)]
        reported, reference = _reported(index, steps=steps)

        table = estimate_north_table(index, reported, reference_deg=reference, usable=_all_usable(index))

        assert len(table) == 2
        assert table["timestamp"].iloc[0] == index.min()
        # the changepoint is found within a day of truth
        assert abs(table["timestamp"].iloc[1] - pd.Timestamp("2017-08-01", tz="UTC")) <= pd.Timedelta(days=1)
        assert table["north_offset"].iloc[0] == pytest.approx(12.0, abs=1.0)
        assert table["north_offset"].iloc[1] == pytest.approx(47.0, abs=1.0)

    def test_no_step_returns_a_single_row(self) -> None:
        index = _index()
        reported, reference = _reported(index, steps=[("2017-01-01", 25.0)])

        table = estimate_north_table(index, reported, reference_deg=reference, usable=_all_usable(index))

        assert len(table) == 1
        assert table["north_offset"].iloc[0] == pytest.approx(25.0, abs=1.0)

    def test_recovers_several_steps(self) -> None:
        index = _index(days=700)
        steps = [("2017-01-01", 0.0), ("2017-06-01", 35.0), ("2017-11-15", -20.0), ("2018-05-01", 60.0)]
        reported, reference = _reported(index, steps=steps)

        table = estimate_north_table(index, reported, reference_deg=reference, usable=_all_usable(index))

        assert len(table) == 4
        for row, (when, offset) in zip(table.itertuples(), steps, strict=True):
            assert abs(row.timestamp - pd.Timestamp(when, tz="UTC")) <= pd.Timedelta(days=2)
            assert circ_diff(row.north_offset, offset) == pytest.approx(0.0, abs=1.5)

    def test_handles_wraparound_in_raw_and_corrected_signals(self) -> None:
        # offsets chosen so both the reported signal and the corrected one cross 0/360 often
        index = _index()
        steps = [("2017-01-01", 343.0), ("2017-07-01", 290.0)]
        reported, reference = _reported(index, steps=steps)

        table = estimate_north_table(index, reported, reference_deg=reference, usable=_all_usable(index))
        corrected = apply_north_table(index, reported, north_table=table)

        assert len(table) == 2
        assert circ_diff(corrected, reference).mean() == pytest.approx(0.0, abs=1.0)
        assert corrected.min() >= 0.0
        assert corrected.max() < 360.0

    def test_ignores_unusable_rows(self) -> None:
        index = _index()
        steps = [("2017-01-01", 10.0), ("2017-09-01", 40.0)]
        reported, reference = _reported(index, steps=steps)
        # corrupt half the record, then mark it unusable: the estimate must be unmoved
        usable = _all_usable(index)
        rng = np.random.default_rng(7)
        corrupt = rng.random(len(index)) < 0.5
        reported = np.where(corrupt, rng.uniform(0, 360, len(index)), reported)
        usable &= ~corrupt

        table = estimate_north_table(index, reported, reference_deg=reference, usable=usable)

        assert len(table) == 2
        assert table["north_offset"].iloc[0] == pytest.approx(10.0, abs=1.5)
        assert table["north_offset"].iloc[1] == pytest.approx(40.0, abs=1.5)

    def test_min_segment_prevents_micro_splits(self) -> None:
        index = _index()
        # two steps two days apart -- inside the 7-day min_segment, so they cannot both be kept
        steps = [("2017-01-01", 0.0), ("2017-06-01", 30.0), ("2017-06-03", 60.0)]
        reported, reference = _reported(index, steps=steps)

        table = estimate_north_table(index, reported, reference_deg=reference, usable=_all_usable(index))

        gaps = table["timestamp"].diff().dropna()
        assert (gaps >= pd.Timedelta(days=7)).all()


class TestNoiseFloor:
    """`min_step_deg` is the smallest step the estimator will report."""

    @staticmethod
    def _n_changepoints(step_deg: float, *, settings: NorthingSettings = DEFAULT_NORTHING) -> int:
        index = _index()
        reported, reference = _reported(index, steps=[("2017-01-01", 0.0), ("2017-07-01", step_deg)])
        table = estimate_north_table(
            index, reported, reference_deg=reference, usable=_all_usable(index), settings=settings
        )
        return len(table) - 1

    def test_step_well_above_min_step_is_found(self) -> None:
        assert self._n_changepoints(6.0) == 1

    def test_step_well_below_min_step_is_not_reported(self) -> None:
        assert self._n_changepoints(0.5) == 0

    def test_the_threshold_is_what_decides_not_the_data_volume(self) -> None:
        """The same 2 degree step is invisible by default and found with the threshold lowered."""
        step = 2.0
        assert self._n_changepoints(step) == 0
        assert self._n_changepoints(step, settings=replace(DEFAULT_NORTHING, min_step_deg=1.0)) == 1


class TestDegenerateInput:
    def test_all_unusable_returns_a_zero_offset_row(self) -> None:
        index = _index(days=30)
        reported, reference = _reported(index, steps=[("2017-01-01", 30.0)])

        table = estimate_north_table(index, reported, reference_deg=reference, usable=np.zeros(len(index), dtype=bool))

        assert len(table) == 1
        assert table["north_offset"].iloc[0] == 0.0
        assert table["timestamp"].iloc[0] == index.min()

    def test_all_nan_direction_returns_a_zero_offset_row(self) -> None:
        index = _index(days=30)
        _, reference = _reported(index, steps=[("2017-01-01", 30.0)])

        table = estimate_north_table(
            index,
            np.full(len(index), np.nan),
            reference_deg=reference,
            usable=_all_usable(index),
        )

        assert len(table) == 1
        assert table["north_offset"].iloc[0] == 0.0

    def test_empty_index_raises(self) -> None:
        empty = pd.DatetimeIndex([], tz="UTC")
        with pytest.raises(ValueError, match="empty"):
            estimate_north_table(empty, np.array([]), reference_deg=np.array([]), usable=np.array([], dtype=bool))

    def test_mismatched_lengths_raise(self) -> None:
        index = _index(days=10)
        with pytest.raises(ValueError, match="same length"):
            estimate_north_table(
                index,
                np.zeros(len(index)),
                reference_deg=np.zeros(len(index) - 1),
                usable=_all_usable(index),
            )


class TestApplyNorthTable:
    def test_applies_steps_and_wraps(self) -> None:
        index = _index(days=30)
        table = pd.DataFrame(
            {
                "timestamp": [index.min(), pd.Timestamp("2017-01-15", tz="UTC")],
                "north_offset": [30.0, 350.0],
            }
        )
        direction = np.full(len(index), 20.0)

        corrected = apply_north_table(index, direction, north_table=table)

        before = index < pd.Timestamp("2017-01-15", tz="UTC")
        assert np.allclose(corrected[before], 50.0)
        assert np.allclose(corrected[~before], 10.0)  # (20 + 350) % 360

    def test_one_table_serves_several_fields(self) -> None:
        index = _index(days=30)
        table = pd.DataFrame({"timestamp": [index.min()], "north_offset": [45.0]})
        yaw = np.full(len(index), 100.0)
        measured_wd = np.full(len(index), 110.0)

        assert np.allclose(apply_north_table(index, yaw, north_table=table), 145.0)
        assert np.allclose(apply_north_table(index, measured_wd, north_table=table), 155.0)

    def test_preserves_nan(self) -> None:
        index = _index(days=10)
        table = pd.DataFrame({"timestamp": [index.min()], "north_offset": [45.0]})
        direction = np.full(len(index), 100.0)
        direction[:5] = np.nan

        corrected = apply_north_table(index, direction, north_table=table)

        assert np.isnan(corrected[:5]).all()
        assert np.allclose(corrected[5:], 145.0)

    def test_round_trip_removes_the_injected_offset(self) -> None:
        index = _index()
        steps = [("2017-01-01", 15.0), ("2017-10-01", -25.0)]
        reported, reference = _reported(index, steps=steps, noise_deg=4.0)

        table = estimate_north_table(index, reported, reference_deg=reference, usable=_all_usable(index))
        corrected = apply_north_table(index, reported, north_table=table)

        assert np.abs(circ_diff(corrected, reference)).mean() < np.abs(circ_diff(reported, reference)).mean()
        assert circ_diff(corrected, reference).mean() == pytest.approx(0.0, abs=0.5)


class TestYawUsable:
    def test_requires_power_reference_and_uptime(self) -> None:
        n = 6
        power = np.array([1000.0, 1000.0, 1000.0, 10.0, 1000.0, 1000.0])
        downtime = np.array([0.0, 0.0, 0.0, 0.0, 300.0, 0.0])
        reference = np.array([10.0, 10.0, np.nan, 10.0, 10.0, 10.0])
        power[1] = np.nan

        usable = yaw_usable(
            power=power,
            downtime_s=downtime,
            reference_deg=reference,
            rated_power=RATED_POWER,
            timebase_s=TIMEBASE_S,
        )

        assert usable.tolist() == [True, False, False, False, False, True]
        assert usable.shape == (n,)


class TestNorthFarm:
    @staticmethod
    def _farm(
        index: pd.DatetimeIndex, offsets: dict[str, list[tuple[str, float]]], *, seed: int = 0
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Reported directions for each device plus the shared true site direction."""
        reference = _true_direction(index, seed=seed)
        reported = {}
        for i, (name, steps) in enumerate(offsets.items()):
            rng = np.random.default_rng(100 + i)
            scatter = rng.normal(0.0, 6.0, size=len(index))
            reported[name] = (reference + scatter - _stepped_offset(index, steps)) % 360.0
        return reported, reference

    def test_two_pass_recovers_per_device_steps(self) -> None:
        index = _index()
        offsets = {
            "T01": [("2017-01-01", 0.0)],
            "T02": [("2017-01-01", 8.0)],
            "T03": [("2017-01-01", -5.0), ("2017-08-01", 35.0)],
            "T04": [("2017-01-01", 3.0)],
        }
        reported, reference = self._farm(index, offsets)

        tables = north_farm(
            index,
            direction_deg=reported,
            usable={name: _all_usable(index) for name in reported},
            reanalysis_deg=reference,
        )

        assert set(tables) == set(offsets)
        assert len(tables["T03"]) == 2
        for name, steps in offsets.items():
            corrected = apply_north_table(index, reported[name], north_table=tables[name])
            assert circ_diff(corrected, reference).mean() == pytest.approx(0.0, abs=2.0), name
            assert len(tables[name]) == len(steps), name

    def test_recovers_a_farm_that_is_uniformly_180_degrees_wrong(self) -> None:
        """The reanalysis pass is load-bearing: a common-mode offset is invisible to pass 2 alone.

        Every device agrees with every other, so a farm-relative method sees a perfectly
        consistent farm and reports nothing wrong.
        """
        index = _index()
        offsets = {name: [("2017-01-01", 180.0)] for name in ("T01", "T02", "T03", "T04")}
        reported, reference = self._farm(index, offsets)

        tables = north_farm(
            index,
            direction_deg=reported,
            usable={name: _all_usable(index) for name in reported},
            reanalysis_deg=reference,
        )

        for name in offsets:
            corrected = apply_north_table(index, reported[name], north_table=tables[name])
            assert circ_diff(corrected, reference).mean() == pytest.approx(0.0, abs=2.0), name
            # and the recovered offset really is the 180 that was injected
            assert circ_diff(tables[name]["north_offset"].iloc[0], 180.0) == pytest.approx(0.0, abs=2.0)

    def test_pass_two_beats_reanalysis_alone(self) -> None:
        """The farm reference is less noisy than reanalysis, so two passes beat one."""
        index = _index()
        offsets = {name: [("2017-01-01", 20.0)] for name in ("T01", "T02", "T03", "T04")}
        reported, reference = self._farm(index, offsets)
        # reanalysis is a degraded view of the true direction; the farm's own consensus is better
        rng = np.random.default_rng(11)
        reanalysis = (reference + rng.normal(0.0, 25.0, size=len(index))) % 360.0

        one_pass = estimate_north_table(index, reported["T01"], reference_deg=reanalysis, usable=_all_usable(index))
        two_pass = north_farm(
            index,
            direction_deg=reported,
            usable={name: _all_usable(index) for name in reported},
            reanalysis_deg=reanalysis,
        )["T01"]

        truth = 20.0
        assert abs(circ_diff(two_pass["north_offset"].iloc[0], truth)) <= abs(
            circ_diff(one_pass["north_offset"].iloc[0], truth)
        )

    @pytest.mark.xfail(
        reason="a device is part of the consensus it is northed against; see findings_campaigns.md CF7",
        strict=True,
    )
    def test_pass_two_refines_every_device_on_an_odd_sized_farm(self) -> None:
        """A device may not be part of the consensus it is northed against.

        With an odd device count the per-timestamp median *is* one of the devices, so a device
        that is its own reference scores an exact ``-offset`` residual on those rows. Those rows
        are algebra rather than measurement, and they mass on one value at the centre of the
        distribution, which pins the median to whatever pass 1 already said.
        """
        index = _index()
        names = ("T01", "T02", "T03", "T04", "T05")
        offsets = {name: [("2017-01-01", 20.0)] for name in names}
        reported, reference = self._farm(index, offsets)
        rng = np.random.default_rng(11)
        reanalysis = (reference + rng.normal(0.0, 25.0, size=len(index))) % 360.0
        usable = {name: _all_usable(index) for name in names}

        two_pass = north_farm(index, direction_deg=reported, usable=usable, reanalysis_deg=reanalysis)

        one_pass_errors, two_pass_errors = [], []
        for name in names:
            one = estimate_north_table(index, reported[name], reference_deg=reanalysis, usable=usable[name])
            first, second = one["north_offset"].iloc[0], two_pass[name]["north_offset"].iloc[0]
            assert second != pytest.approx(first, abs=1e-9), f"{name}: pass 2 merely repeated pass 1"
            one_pass_errors.append(abs(circ_diff(first, 20.0)))
            two_pass_errors.append(abs(circ_diff(second, 20.0)))
        # a device pass 1 happened to get right can still move slightly the wrong way; the farm is
        # what has to improve
        assert np.mean(two_pass_errors) < np.mean(one_pass_errors)

    def test_raises_when_too_few_devices_for_a_farm_reference(self) -> None:
        index = _index(days=30)
        offsets = {"T01": [("2017-01-01", 0.0)], "T02": [("2017-01-01", 5.0)]}
        reported, reference = self._farm(index, offsets)

        with pytest.raises(ValueError, match="min_devices_for_farm_reference"):
            north_farm(
                index,
                direction_deg=reported,
                usable={name: _all_usable(index) for name in reported},
                reanalysis_deg=reference,
                min_devices_for_farm_reference=3,
            )


class TestSettings:
    def test_the_default_needs_no_argument(self) -> None:
        index = _index(days=30)
        reported, reference = _reported(index, steps=[("2017-01-01", 10.0)])
        table = estimate_north_table(index, reported, reference_deg=reference, usable=_all_usable(index))
        assert table["north_offset"].iloc[0] == pytest.approx(10.0, abs=2.0)

    def test_a_short_record_still_gets_a_floor_of_changepoints(self) -> None:
        """`ceil(rate * years)` alone would allow one over a month; the floor leaves room for more."""
        index = _index(days=31)
        steps = [("2017-01-01", 0.0), ("2017-01-10", 40.0), ("2017-01-20", 80.0)]
        reported, reference = _reported(index, steps=steps)
        table = estimate_north_table(index, reported, reference_deg=reference, usable=_all_usable(index))
        assert len(table) == 3

    def test_changepoint_budget_scales_with_record_length(self) -> None:
        """`changepoints_per_year` is a rate, so a longer record gets a larger budget."""
        settings = NorthingSettings(changepoints_per_year=1.0, min_step_deg=3.0, refine=False, min_changepoints=0)
        steps = [("2017-01-01", 0.0), ("2017-04-01", 30.0), ("2017-07-01", 60.0), ("2017-10-01", 90.0)]

        short_index = _index(days=200)
        short_reported, short_reference = _reported(short_index, steps=steps[:3])
        short = estimate_north_table(
            short_index,
            short_reported,
            reference_deg=short_reference,
            usable=_all_usable(short_index),
            settings=settings,
        )

        long_index = _index(days=1400)
        long_reported, long_reference = _reported(long_index, steps=steps)
        long = estimate_north_table(
            long_index, long_reported, reference_deg=long_reference, usable=_all_usable(long_index), settings=settings
        )

        assert len(short) - 1 <= 1  # ceil(1 * 0.55 years)
        assert len(long) - 1 == 3  # ceil(1 * 3.8 years) allows all three


class TestVeerNormalisation:
    """Across a site the wind direction differs turbine to turbine, and that difference shifts
    with the bulk direction. A changing direction *mix* must not look like a step."""

    def test_removes_a_direction_dependent_level(self) -> None:
        index = _index()
        reference = _true_direction(index)
        # a turbine reading 8 deg high in the northern half of the compass and 8 low in the south
        veer = np.where((reference < 180.0), 8.0, -8.0)
        residual = veer + np.random.default_rng(3).normal(0.0, 2.0, len(index))

        out = veer_normalised(residual, reference_deg=reference, sector_deg=30.0)

        north = out[reference < 180.0]
        south = out[reference >= 180.0]
        assert np.nanmedian(north) == pytest.approx(0.0, abs=0.5)
        assert np.nanmedian(south) == pytest.approx(0.0, abs=0.5)

    def test_a_uniform_offset_survives_because_it_shifts_every_sector_alike(self) -> None:
        index = _index()
        reference = _true_direction(index)
        residual = np.full(len(index), 20.0)
        shifted = veer_normalised(residual + 5.0, reference_deg=reference, sector_deg=30.0)
        base = veer_normalised(residual, reference_deg=reference, sector_deg=30.0)
        # the constant is absorbed, so what remains is identical -- the step is carried by the
        # segment offsets, which come from the raw residual
        assert np.nanmax(np.abs(shifted - base)) == pytest.approx(0.0, abs=1e-9)

    def test_a_shifting_direction_mix_no_longer_reads_as_a_step(self) -> None:
        """No offset changes; only which directions the wind comes from. Nothing must be found."""
        index = _index(days=400)
        rng = np.random.default_rng(5)
        half = len(index) // 2
        # first half blows from the north, second half from the south
        reference = np.concatenate(
            [rng.normal(0.0, 25.0, half) % 360.0, (rng.normal(180.0, 25.0, len(index) - half)) % 360.0]
        )
        veer = np.where(reference < 180.0, 8.0, -8.0)
        reported = (reference + veer + rng.normal(0.0, 5.0, len(index))) % 360.0

        table = estimate_north_table(index, reported, reference_deg=reference, usable=_all_usable(index))

        assert len(table) == 1, f"veer mistaken for a step: {table}"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("veer_amplitude", [6.0, 8.0])
    def test_veer_strong_enough_to_over_detect_is_still_absorbed(self, veer_amplitude: float, seed: int) -> None:
        """Smoothly direction-dependent veer, no true step: nothing may be found.

        The sector levels are measured on a residual de-stepped by the first detection pass. Veer
        this strong makes that pass split the record, and de-stepping those splits takes the very
        sector levels the signature is meant to capture -- so the splits survive the second pass
        that exists to remove them. Only steps large enough to be real may be de-stepped.
        """
        index = _index(days=700)
        rng = np.random.default_rng(seed)
        reference = np.cumsum(rng.normal(0.0, 2.0, size=len(index))) % 360.0
        scatter = np.random.default_rng(500 + seed).normal(0.0, 6.0, len(index))
        reported = (reference + veer_amplitude * np.cos(np.deg2rad(reference - 40.0)) + scatter) % 360.0

        table = estimate_north_table(index, reported, reference_deg=reference, usable=_all_usable(index))

        assert len(table) == 1, f"veer mistaken for {len(table) - 1} step(s): {table}"


class TestTransientPruning:
    """Site veer wanders away and back; a recalibration does not."""

    @staticmethod
    def _n_changepoints(steps: list[tuple[str, float]]) -> int:
        index = _index(days=700)
        reported, reference = _reported(index, steps=steps)
        table = estimate_north_table(index, reported, reference_deg=reference, usable=_all_usable(index))
        return len(table) - 1

    def test_a_small_self_cancelling_excursion_is_ironed_out(self) -> None:
        # away by 6 deg for two months and back again: the record ends where it started
        assert self._n_changepoints([("2017-01-01", 0.0), ("2017-06-01", 6.0), ("2017-08-01", 0.0)]) == 0

    def test_a_small_persistent_step_is_kept(self) -> None:
        # the same 6 deg, but it stays -- that is a recalibration
        assert self._n_changepoints([("2017-01-01", 0.0), ("2017-06-01", 6.0)]) == 1

    def test_a_large_excursion_is_kept_even_though_it_cancels(self) -> None:
        """A real recalibration is sometimes reversed later; its size is the evidence it happened."""
        assert self._n_changepoints([("2017-01-01", 0.0), ("2017-06-01", 90.0), ("2017-11-01", 0.0)]) == 2

    def test_a_long_oscillation_of_small_steps_is_removed_entirely(self) -> None:
        # six steps of 5 deg that end back where they started -- veer, not six recalibrations
        steps: list[tuple[str, float]] = [("2017-01-01", 0.0)]
        dates = ("2017-04-01", "2017-07-01", "2017-10-01", "2018-01-01", "2018-04-01", "2018-07-01")
        steps.extend((when, 5.0 if i % 2 == 0 else 0.0) for i, when in enumerate(dates))
        assert steps[-1][1] == 0.0, "the oscillation must return to its starting level"
        assert self._n_changepoints(steps) == 0

    def test_an_oscillation_biased_enough_to_shift_the_level_is_not_fully_ironed_out(self) -> None:
        """The pruning is a threshold rule, not an oracle: an oscillation whose halves sit at
        genuinely different levels keeps the changepoints that carry that difference."""
        steps: list[tuple[str, float]] = [("2017-01-01", 0.0)]
        dates = ("2017-04-01", "2017-07-01", "2017-10-01", "2018-01-01", "2018-04-01", "2018-07-01")
        steps.extend((when, 7.0 if i % 2 == 0 else 0.0) for i, when in enumerate(dates))
        kept = self._n_changepoints(steps)
        assert 0 < kept < len(steps) - 1


class TestNearTheRecordEdge:
    """How much data sits either side of a changepoint decides how big a step is credible.

    A step with little record after it is estimated from little data, so a small one is as
    likely to be veer as a recalibration. A large one is not: no amount of veer moves a
    turbine's yaw by tens of degrees, so it must still be found however late it lands.
    """

    @staticmethod
    def _n_changepoints(step_deg: float, *, days_after: float, days: float = 700.0) -> int:
        index = _index(days=days)
        when = (index.max() - pd.Timedelta(days=days_after)).strftime("%Y-%m-%d")
        reported, reference = _reported(index, steps=[("2017-01-01", 0.0), (when, step_deg)])
        table = estimate_north_table(index, reported, reference_deg=reference, usable=_all_usable(index))
        return len(table) - 1

    def test_a_large_jump_ten_days_before_the_end_is_still_found(self) -> None:
        assert self._n_changepoints(60.0, days_after=10.0) == 1

    def test_a_large_jump_ten_days_after_the_start_is_still_found(self) -> None:
        index = _index(days=700)
        when = (index.min() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        reported, reference = _reported(index, steps=[("2017-01-01", 0.0), (when, 60.0)])
        table = estimate_north_table(index, reported, reference_deg=reference, usable=_all_usable(index))
        assert len(table) - 1 == 1

    def test_a_small_step_ten_days_before_the_end_is_not_reported(self) -> None:
        assert self._n_changepoints(4.0, days_after=10.0) == 0

    def test_the_same_small_step_well_inside_the_record_is_reported(self) -> None:
        """The step is identical; only the evidence behind it differs."""
        assert self._n_changepoints(4.0, days_after=300.0) == 1


class TestFarmReferenceComposition:
    """The farm reference must not depend on *which* devices happened to report.

    Turbines sit at different long-run offsets from the farm consensus -- site veer. A plain
    median over whoever is reporting therefore moves when the reporting set changes, so an
    outage, or simply analysing a subset of the farm, looks like every turbine stepping at once.
    Nothing about any turbine's north calibration has changed, so nothing should be found.
    """

    @staticmethod
    def _farm(index: pd.DatetimeIndex, *, veer: dict[str, float]) -> tuple[dict, np.ndarray]:
        """Devices whose veer offset **depends on wind direction**, as real site veer does.

        A fixed per-device offset reproduces nothing: the first pass norths every device to
        reanalysis and removes it, which is why an early attempt at this test passed against the
        very bug it was written for. What survives that pass is the direction-dependent *shape*,
        and that is what moves the reference when the reporting set and the wind direction change
        together.
        """
        reference = _true_direction(index, seed=3)
        reported = {}
        for i, (name, amplitude) in enumerate(veer.items()):
            shape = amplitude * np.cos(np.deg2rad(reference - 60.0 * i))
            scatter = np.random.default_rng(200 + i).normal(0.0, 5.0, len(index))
            reported[name] = (reference + shape + scatter) % 360.0
        return reported, reference

    def test_no_changepoint_when_an_outage_coincides_with_an_unusual_wind_direction(self) -> None:
        """The Hill of Towie failure, in miniature.

        For one week most of the farm is down and the wind sits in a sector it rarely occupies.
        The few devices still reporting have their own veer in that sector, so a median over them
        is not the farm's consensus -- and every device appears to step together and back.
        """
        index = _index(days=700)
        veer = {"T01": 5.0, "T02": 4.0, "T03": 3.0, "T04": 4.5, "T05": 5.5, "T06": 3.5}
        reported, reference = self._farm(index, veer=veer)
        outage = (index >= index.min() + pd.Timedelta(days=350)) & (index < index.min() + pd.Timedelta(days=357))
        still_on = ("T01", "T02", "T03")
        usable = {name: np.asarray(~outage | np.isin(name, still_on), dtype=bool) for name in reported}

        tables = north_farm(
            index, direction_deg=reported, usable=usable, reanalysis_deg=reference, settings=DEFAULT_NORTHING
        )

        # Nothing may be attributed to the outage. A marginal detection elsewhere in the record is
        # ordinary veer sensitivity, not this failure, so the assertion is placed where the bug is.
        window = pd.Timedelta(days=14)
        near = {
            name: [
                c.strftime("%Y-%m-%d")
                for c in pd.DatetimeIndex(table["timestamp"])[1:]
                if index.min() + pd.Timedelta(days=350) - window <= c <= index.min() + pd.Timedelta(days=357) + window
            ]
            for name, table in tables.items()
        }
        offenders = {name: found for name, found in near.items() if found}
        assert offenders == {}, f"the reporting set changed, not the turbines: {offenders}"

    def test_the_reference_gives_the_same_answer_for_a_subset_of_the_farm(self) -> None:
        """Northing three of six devices must agree with northing all six."""
        index = _index(days=700)
        veer = {"T01": 5.0, "T02": 4.0, "T03": 3.0, "T04": 4.5, "T05": 5.5, "T06": 3.5}
        reported, reference = self._farm(index, veer=veer)
        usable = {name: _all_usable(index) for name in reported}
        subset = ("T01", "T02", "T03")

        whole = north_farm(index, direction_deg=reported, usable=usable, reanalysis_deg=reference)
        part = north_farm(
            index,
            direction_deg={k: reported[k] for k in subset},
            usable={k: usable[k] for k in subset},
            reanalysis_deg=reference,
        )

        for name in subset:
            assert len(part[name]) == len(whole[name]), name
            assert part[name]["north_offset"].iloc[0] == pytest.approx(whole[name]["north_offset"].iloc[0], abs=2.0), (
                name
            )


class TestFarmNeedsAnAnchor:
    """Pass 2 alone is blind to a farm that is uniformly wrong, so the anchor must exist."""

    @staticmethod
    def _farm(index: pd.DatetimeIndex) -> tuple[dict[str, np.ndarray], np.ndarray]:
        reference = _true_direction(index)
        offsets = {"A": 0.0, "B": 25.0, "C": -40.0}
        reported = {
            name: (reference + np.random.default_rng(10 + i).normal(0.0, 6.0, len(index)) - off) % 360.0
            for i, (name, off) in enumerate(offsets.items())
        }
        return reported, reference

    def test_an_all_nan_reanalysis_raises(self) -> None:
        index = _index(days=60)
        reported, _ = self._farm(index)
        usable = {name: _all_usable(index) for name in reported}

        with pytest.raises(ValueError, match="cannot anchor the farm"):
            north_farm(
                index,
                direction_deg=reported,
                usable=usable,
                reanalysis_deg=np.full(len(index), np.nan),
                settings=DEFAULT_NORTHING,
            )

    def test_a_reanalysis_that_never_overlaps_usable_rows_raises(self) -> None:
        index = _index(days=60)
        reported, reference = self._farm(index)
        # reanalysis is finite only where no device is usable
        usable = {name: _all_usable(index) for name in reported}
        for mask in usable.values():
            mask[: len(index) // 2] = False
        blinded = reference.copy()
        blinded[len(index) // 2 :] = np.nan

        with pytest.raises(ValueError, match="cannot anchor the farm"):
            north_farm(index, direction_deg=reported, usable=usable, reanalysis_deg=blinded, settings=DEFAULT_NORTHING)

    def test_a_healthy_reanalysis_still_norths(self) -> None:
        index = _index(days=60)
        reported, reference = self._farm(index)
        usable = {name: _all_usable(index) for name in reported}

        tables = north_farm(
            index, direction_deg=reported, usable=usable, reanalysis_deg=reference, settings=DEFAULT_NORTHING
        )

        assert set(tables) == set(reported)
        assert float(tables["B"]["north_offset"].iloc[0]) == pytest.approx(25.0, abs=1.0)


class TestNorthTableYaml:
    """The written table is a prior an analyst can hand edit and feed back."""

    def test_round_trips_through_yaml(self, tmp_path: Path) -> None:
        tables = {
            "T02": pd.DataFrame(
                {
                    "timestamp": pd.DatetimeIndex(["2017-01-01", "2017-06-30 12:20:00"], tz="UTC"),
                    "north_offset": [1.5, -33.25],
                }
            ),
            "T01": pd.DataFrame({"timestamp": pd.DatetimeIndex(["2017-01-01"], tz="UTC"), "north_offset": [-7.125]}),
        }
        path = tmp_path / "northing_corrections.yaml"

        write_north_table_yaml(tables, path=path)
        parsed = yaml.safe_load(path.read_text())

        # the shape north_offsets and v0's northing_corrections_utc both read
        assert [row[0] for row in parsed] == ["T01", "T02", "T02"]
        assert [row[2] for row in parsed] == [-7.125, 1.5, -33.25]
        assert [pd.Timestamp(row[1]).strftime("%Y-%m-%d %H:%M:%S") for row in parsed] == [
            "2017-01-01 00:00:00",
            "2017-01-01 00:00:00",
            "2017-06-30 12:20:00",
        ]
