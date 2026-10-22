"""Tests for the uplift shape of method step 15: measured against predicted power, cast to the other axes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.power_model.shape import (
    bin_codes,
    block_weights,
    freestream_scale,
    reference_wind_speed_drift,
    uplift_shape,
)

RATED = 2000.0
EDGES = [-100.0, 1000.0, 2100.0]  # two power bins
HOURS = 1.0 / 6.0  # ten-minute records
MIN_HOURS = 24.0


def _records(n_per_bin: int, *, low: float, high: float, start: str = "2020-01-01") -> dict[str, object]:
    """Half the records at 500 kW predicted, half at 1500 kW, actual = predicted x (1 + the bin's uplift)."""
    times = pd.date_range(start, periods=2 * n_per_bin, freq="10min", tz="UTC")
    counterfactual = np.where(np.arange(2 * n_per_bin) % 2 == 0, 500.0, 1500.0)
    actual = counterfactual * np.where(counterfactual < 1000.0, 1.0 + low, 1.0 + high)
    return {"times": times, "actual": actual, "counterfactual": counterfactual}


def _shape(records: dict[str, object], *, headline: float, **kwargs: object) -> object:
    times = records["times"]
    defaults: dict[str, object] = {
        "power_edges": EDGES,
        "axes": {},
        "subsets": {},
        "headline": headline,
        "min_hours": MIN_HOURS,
        "hours_per_record": HOURS,
        "period_start": times.min(),  # type: ignore[attr-defined]
        "period_end": times.max(),  # type: ignore[attr-defined]
        "timebase": pd.Timedelta(minutes=10),
        "block_hours": 24.0,
        "n_resamples": 50,
        "seed": 0,
        "subset_power_edges": {},
    }
    return uplift_shape(
        times=times,
        actual=records["actual"],
        counterfactual=records["counterfactual"],
        **{**defaults, **kwargs},
    )


def _row(frame: pd.DataFrame, condition: str, bin_label: str) -> pd.Series:
    return frame[(frame["condition"] == condition) & (frame["condition_bin"] == bin_label)].iloc[0]


class TestBinCodes:
    def test_values_outside_the_edges_or_nan_have_no_bin(self) -> None:
        codes = bin_codes(np.array([-200.0, 0.0, 1500.0, np.nan, 3000.0]), EDGES)
        assert codes.tolist() == [-1, 0, 1, -1, -1]


class TestMeasuredShape:
    def test_each_bin_reads_its_own_uplift_when_the_shape_already_meets_the_headline(self) -> None:
        records = _records(500, low=0.04, high=0.01)
        headline = (500 * 500 * 1.04 + 500 * 1500 * 1.01) / (500 * 500 + 500 * 1500) - 1
        result = _shape(records, headline=headline)
        assert _row(result.frame, "predicted_power", "(-100.0, 1000.0]")["p50_uplift"] == pytest.approx(0.04)
        assert _row(result.frame, "predicted_power", "(1000.0, 2100.0]")["p50_uplift"] == pytest.approx(0.01)

    def test_the_bins_are_shifted_by_one_constant_to_meet_the_headline(self) -> None:
        records = _records(500, low=0.04, high=0.01)
        result = _shape(records, headline=0.0)
        low = _row(result.frame, "predicted_power", "(-100.0, 1000.0]")["p50_uplift"]
        high = _row(result.frame, "predicted_power", "(1000.0, 2100.0]")["p50_uplift"]
        assert low - high == pytest.approx(0.03)
        assert (500 * low + 1500 * high) / 2000 == pytest.approx(0.0, abs=1e-12)

    def test_a_bin_under_the_minimum_hours_is_not_measured_and_carries_the_headline(self) -> None:
        records = _records(100, low=0.04, high=0.01)  # 100 records = 16.7 h per bin
        result = _shape(records, headline=0.02)
        row = _row(result.frame, "predicted_power", "(-100.0, 1000.0]")
        assert not row["measured"]
        assert row["p50_uplift"] == pytest.approx(0.02)
        assert np.isnan(row["sigma_sampling"])

    def test_each_bin_reports_its_hours(self) -> None:
        result = _shape(_records(500, low=0.04, high=0.01), headline=0.0)
        assert _row(result.frame, "predicted_power", "(-100.0, 1000.0]")["hours"] == pytest.approx(500 * HOURS)


class TestCast:
    def test_a_cast_bin_is_the_energy_weighted_mean_of_its_records_power_bin_uplifts(self) -> None:
        records = _records(500, low=0.04, high=0.01)
        headline = (500 * 1.04 + 1500 * 1.01) / 2000 - 1
        # every record labelled 5 on an axis, so the one cast bin holds both power bins
        axis = {"direction": (np.full(1000, 5.0), [0.0, 30.0])}
        result = _shape(records, headline=headline, axes=axis)
        row = _row(result.frame, "direction", "(0.0, 30.0]")
        assert row["kind"] == "cast"
        assert row["p50_uplift"] == pytest.approx(headline)

    def test_casting_onto_the_power_bins_themselves_returns_the_shape(self) -> None:
        records = _records(500, low=0.04, high=0.01)
        axis = {"freestream_ws": (records["counterfactual"], EDGES)}
        result = _shape(records, headline=0.0, axes=axis)
        for label in ("(-100.0, 1000.0]", "(1000.0, 2100.0]"):
            measured = _row(result.frame, "predicted_power", label)["p50_uplift"]
            assert _row(result.frame, "freestream_ws", label)["p50_uplift"] == pytest.approx(measured)

    def test_a_cast_bin_under_the_minimum_hours_is_flagged_thin(self) -> None:
        records = _records(500, low=0.04, high=0.01)
        values = np.full(1000, 5.0)
        values[:10] = 45.0
        result = _shape(records, headline=0.0, axes={"direction": (values, [0.0, 30.0, 60.0])})
        assert _row(result.frame, "direction", "(30.0, 60.0]")["thin"]
        assert not _row(result.frame, "direction", "(0.0, 30.0]")["thin"]

    def test_record_labels_are_returned_per_axis(self) -> None:
        records = _records(500, low=0.04, high=0.01)
        result = _shape(records, headline=0.0, axes={"direction": (np.full(1000, 5.0), [0.0, 30.0])})
        assert result.labels.index.equals(records["times"])
        assert set(result.labels.columns) == {"predicted_power", "direction"}


class TestSampling:
    def test_an_exact_uplift_has_no_sampling_spread(self) -> None:
        records = _records(2000, low=0.02, high=0.02)
        result = _shape(records, headline=0.02, axes={"direction": (np.full(4000, 5.0), [0.0, 30.0])})
        measured = result.frame[result.frame["measured"]]
        assert np.allclose(measured["sigma_sampling"], 0.0, atol=1e-9)

    def test_noise_gives_a_spread(self) -> None:
        records = _records(2000, low=0.02, high=0.02)
        rng = np.random.default_rng(1)
        records["actual"] = records["actual"] * (1 + rng.normal(0, 0.05, 4000))
        result = _shape(records, headline=0.02)
        assert (result.frame.loc[result.frame["measured"], "sigma_sampling"] > 0).all()


class TestBlockWeights:
    def test_each_resample_counts_each_record_a_whole_number_of_times_and_one_on_average(self) -> None:
        times = pd.date_range("2020-01-01", periods=24 * 6 * 30, freq="10min", tz="UTC")
        weights = block_weights(
            times,
            period_start=times.min(),
            period_end=times.max(),
            timebase=pd.Timedelta(minutes=10),
            block_hours=24.0,
            n_resamples=400,
            seed=3,
        )
        assert weights.shape == (400, len(times))
        assert np.array_equal(weights, np.round(weights))
        assert weights.mean() == pytest.approx(1.0, abs=0.05)

    def test_the_same_seed_gives_the_same_weights(self) -> None:
        times = pd.date_range("2020-01-01", periods=1000, freq="10min", tz="UTC")
        kwargs = {"period_start": times.min(), "period_end": times.max(), "timebase": pd.Timedelta(minutes=10)}
        a = block_weights(times, block_hours=24.0, n_resamples=5, seed=7, **kwargs)
        b = block_weights(times, block_hours=24.0, n_resamples=5, seed=7, **kwargs)
        assert np.array_equal(a, b)


class TestSubsets:
    def test_a_subset_with_its_own_shape_is_flagged(self) -> None:
        records = _records(3000, low=0.0, high=0.0)
        side = np.where(np.arange(6000) < 3000, 0.0, 1.0)
        actual = records["actual"].copy()
        actual[(side == 1) & (records["counterfactual"] < 1000)] *= 1.10
        records["actual"] = actual
        result = _shape(
            records,
            headline=0.0,
            subsets={"waked": (side, [-0.5, 0.5, 1.5])},
            subset_power_edges={2: EDGES},
        )
        waked = result.subsets[(result.subsets["axis"] == "waked") & (result.subsets["subset"] == "(0.5, 1.5]")]
        assert waked["flagged"].any()

    def test_a_subset_too_thin_for_any_bin_count_has_no_shape(self) -> None:
        records = _records(500, low=0.0, high=0.0)
        side = np.zeros(1000)
        side[:20] = 1.0
        result = _shape(
            records,
            headline=0.0,
            subsets={"waked": (side, [-0.5, 0.5, 1.5])},
            subset_power_edges={2: EDGES},
        )
        thin = result.subsets[(result.subsets["axis"] == "waked") & (result.subsets["subset"] == "(0.5, 1.5]")]
        assert thin["too_thin"].all()
        assert thin["p50_uplift"].isna().all()

    def test_a_subset_falls_back_to_fewer_bins(self) -> None:
        records = _records(500, low=0.0, high=0.0)
        many = {3: [-100.0, 300.0, 700.0, 2100.0], 2: EDGES}  # with 3 bins (-100, 300] is empty
        result = _shape(
            records, headline=0.0, subsets={"waked": (np.ones(1000), [-0.5, 0.5, 1.5])}, subset_power_edges=many
        )
        rows = result.subsets[(result.subsets["axis"] == "waked") & (result.subsets["subset"] == "(0.5, 1.5]")]
        assert rows["n_bins"].eq(2).all()


class TestFreestreamScale:
    def test_recovers_the_factor_that_meets_the_target_cp(self) -> None:
        diameter, rho, target = 90.0, 1.225, 0.46
        area = np.pi * diameter**2 / 4
        true_ws = np.linspace(5.0, 9.0, 200)
        power = target * 0.5 * rho * area * true_ws**3 / 1000.0
        ref_ws = true_ws / 1.08
        scale = freestream_scale(
            power,
            ref_ws,
            rated_power_kw=power.max() * 1.5,
            rotor_diameter_m=diameter,
            target_cp=target,
            air_density=rho,
        )
        assert scale == pytest.approx(1.08)

    def test_only_records_in_the_power_band_count(self) -> None:
        power = np.array([100.0, 1000.0, 1900.0])
        ref_ws = np.array([1.0, 8.0, 50.0])
        scale = freestream_scale(
            power, ref_ws, rated_power_kw=2000.0, rotor_diameter_m=90.0, target_cp=0.46, air_density=1.225
        )
        only = freestream_scale(
            power[1:2], ref_ws[1:2], rated_power_kw=2000.0, rotor_diameter_m=90.0, target_cp=0.46, air_density=1.225
        )
        assert scale == pytest.approx(only)

    def test_no_record_in_the_band_gives_nan(self) -> None:
        scale = freestream_scale(
            np.array([10.0]),
            np.array([3.0]),
            rated_power_kw=2000.0,
            rotor_diameter_m=90.0,
            target_cp=0.46,
            air_density=1.225,
        )
        assert np.isnan(scale)


class TestReferenceWindSpeedDrift:
    def test_a_reference_whose_anemometer_steps_is_flagged(self) -> None:
        times = pd.date_range("2020-01-01", "2020-12-31", freq="1h", tz="UTC")
        rng = np.random.default_rng(0)
        base = 8 + rng.normal(0, 1, len(times))
        ws = pd.DataFrame({"T1": base, "T2": base * 1.0, "T3": base * 1.0, "T4": base * 1.0}, index=times)
        ws.loc[times >= "2020-07-01", "T2"] *= 1.06
        drift = reference_wind_speed_drift(ws, threshold=0.03)
        assert set(drift.loc[drift["flagged"], "reference"]) == {"T2"}

    def test_steady_anemometers_are_not_flagged(self) -> None:
        times = pd.date_range("2020-01-01", "2020-12-31", freq="1h", tz="UTC")
        base = pd.Series(8.0, index=times)
        ws = pd.DataFrame({"T1": base, "T2": base * 1.02, "T3": base * 0.98})
        assert not reference_wind_speed_drift(ws, threshold=0.03)["flagged"].any()
