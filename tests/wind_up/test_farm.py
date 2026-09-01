"""Tests for the farm-uplift headline and its guards."""

from __future__ import annotations

import math

import pytest

from wind_up.farm import FarmUplift, TurbineUplift, farm_uplift


def _t(
    name: str = "T1",
    *,
    uplift: float = 0.05,
    treated_energy: float = 1000.0,
    n_records: int = 100,
    rated_power_kw: float = 2300.0,
) -> TurbineUplift:
    return TurbineUplift(
        turbine=name,
        uplift=uplift,
        treated_energy=treated_energy,
        n_records=n_records,
        rated_power_kw=rated_power_kw,
    )


def test_equal_uplifts_reproduce_the_pooled_ratio() -> None:
    result = farm_uplift([_t("T1", treated_energy=1000.0), _t("T2", treated_energy=2000.0)])
    assert result.uplift == pytest.approx(0.05)
    assert result.turbines["used"].all()
    assert (result.turbines["guard"] == "").all()


def test_headline_weights_turbines_by_treated_energy() -> None:
    result = farm_uplift([_t("T1", uplift=0.10, treated_energy=110.0), _t("T2", uplift=0.0, treated_energy=900.0)])
    counterfactual = 110.0 / 1.10 + 900.0
    assert result.uplift == pytest.approx((110.0 + 900.0) / counterfactual - 1.0)
    assert result.uplift < 0.02


def test_uplift_spread_reports_the_range_across_used_turbines() -> None:
    result = farm_uplift([_t("T1", uplift=0.02), _t("T2", uplift=0.08), _t("T3", uplift=0.05)])
    assert result.uplift_spread == pytest.approx(0.06)


def test_uplift_spread_is_nan_for_a_single_turbine() -> None:
    assert math.isnan(farm_uplift([_t("T1")]).uplift_spread)


def test_uplift_of_minus_one_is_dropped_not_divided_by_zero() -> None:
    result = farm_uplift([_t("T1", uplift=-1.0), _t("T2", uplift=0.05, treated_energy=2000.0)])
    row = result.turbines.set_index("turbine").loc["T1"]
    assert not row["used"]
    assert row["guard"] == "negative_counterfactual"
    assert result.uplift == pytest.approx(0.05)


def test_uplift_below_minus_one_is_dropped() -> None:
    result = farm_uplift([_t("T1", uplift=-1.5), _t("T2", uplift=0.05, treated_energy=2000.0)])
    assert not result.turbines.set_index("turbine").loc["T1", "used"]
    assert result.uplift == pytest.approx(0.05)


def test_implied_capacity_factor_above_rated_is_capped() -> None:
    result = farm_uplift([_t("T1", uplift=-0.9, treated_energy=100.0, n_records=10, rated_power_kw=50.0)])
    row = result.turbines.set_index("turbine").loc["T1"]
    assert row["used"]
    assert row["guard"] == "capacity_cap"
    assert row["counterfactual_energy"] == pytest.approx(500.0)
    assert result.uplift == pytest.approx(100.0 / 500.0 - 1.0)


def test_negative_treated_energy_is_dropped() -> None:
    result = farm_uplift([_t("T1", treated_energy=-5.0), _t("T2", treated_energy=2000.0)])
    row = result.turbines.set_index("turbine").loc["T1"]
    assert not row["used"]
    assert row["guard"] == "negative_energy"


def test_turbine_with_no_records_is_dropped() -> None:
    result = farm_uplift([_t("T1", n_records=0, treated_energy=0.0), _t("T2")])
    assert result.turbines.set_index("turbine").loc["T1", "guard"] == "no_records"


def test_non_finite_uplift_is_dropped() -> None:
    result = farm_uplift([_t("T1", uplift=float("nan")), _t("T2")])
    assert result.turbines.set_index("turbine").loc["T1", "guard"] == "non_finite_uplift"


def test_headline_is_nan_when_no_turbine_is_usable() -> None:
    result = farm_uplift([_t("T1", uplift=float("nan"))])
    assert math.isnan(result.uplift)


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="at least one turbine"):
        farm_uplift([])


def test_result_is_a_farm_uplift() -> None:
    assert isinstance(farm_uplift([_t("T1")]), FarmUplift)
