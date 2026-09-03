"""The placebo run with the real carried-forward methods on a small fixture.

Two fixtures, because they prove different things. With every turbine sharing one power series
the energy ratio is exactly ``1/n`` in both periods, so any non-zero estimate is a pipeline
fault and the assertion is exact. With independent per-turbine wind noise the methods carry
genuine finite-sample scatter, so that assertion is a tolerance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns import CampaignRunner, carried_forward_methods, per_turbine_table
from benchmarking.campaigns.placebo import placebo_analysis_period, placebo_campaign
from benchmarking.synthetic import HOT_COLUMNS, HOT_RATED_POWER_KW

# End-to-end campaign runs; the fast gate covers the pieces individually.
pytestmark = pytest.mark.slow


if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.campaigns import CampaignResult

# Independent 0.4 m/s per-turbine wind noise through a cubic power curve leaves the energy-ratio
# methods with about a percent of scatter on a campaign of this length.
NOISY_TOLERANCE = 2e-2
MIN_POWER_SD_KW = 100.0
# A small slice of the farm, so the fixtures stay cheap; the production default is all 21 turbines.
TEST_TURBINES = ("T07", "T11")
TEST_PARTICIPANTS = ("T07", "T11", "T01", "T02", "T03")


def placebo_scada(mode: str, *, wind_noise_sd: float, seed: int = 0) -> pd.DataFrame:
    """Hourly SCADA over ``mode``'s placebo period: a shared wind signal plus per-turbine noise.

    :param mode: the placebo mode whose analysis period the frame spans
    :param wind_noise_sd: per-turbine wind-speed noise in m/s; 0 makes every turbine identical
    :param seed: RNG seed for the noise
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(*placebo_analysis_period(mode), freq="1h", tz="UTC", inclusive="left")
    hours = np.arange(len(index), dtype=float)
    # a slow seasonal swing plus a daily cycle, so the pre and post periods differ in wind resource
    shared_ws = 8.0 + 2.5 * np.sin(hours / (24 * 30)) + 1.5 * np.sin(hours / 12)
    frames = []
    for turbine in TEST_PARTICIPANTS:
        noise = rng.normal(0.0, wind_noise_sd, len(index)) if wind_noise_sd else 0.0
        ws = shared_ws + noise
        power = np.clip(HOT_RATED_POWER_KW * (ws / 13.0) ** 3, 0.0, HOT_RATED_POWER_KW)
        frames.append(
            pd.DataFrame(
                {
                    HOT_COLUMNS.turbine: turbine,
                    HOT_COLUMNS.active_power: power,
                    HOT_COLUMNS.active_power_min: power * 0.95,
                    HOT_COLUMNS.wind_speed: ws,
                    HOT_COLUMNS.wind_speed_sd: 0.8,
                    HOT_COLUMNS.gen_rpm: 1400.0,
                    HOT_COLUMNS.availability: 3600.0,
                },
                index=index,
            )
        )
    return pd.concat(frames)


def run_placebo_fixture(mode: str, *, wind_noise_sd: float, out_dir: Path) -> CampaignResult:
    """Declare the placebo for ``mode`` and run the real fast methods over the fixture."""
    declared = placebo_campaign(mode, upgraded=TEST_TURBINES, turbines=TEST_PARTICIPANTS)
    dataset = declared.generate(placebo_scada(mode, wind_noise_sd=wind_noise_sd))
    spec = declared.spec()
    return CampaignRunner(
        spec,
        dataset,
        build_methods=lambda wtg: carried_forward_methods(spec, out_dir=out_dir / wtg, include_power_model=False),
    ).run()


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_identical_turbines_give_exactly_zero(mode: str, tmp_path: Path) -> None:
    result = run_placebo_fixture(mode, wind_noise_sd=0.0, out_dir=tmp_path)
    per_turbine = per_turbine_table(result)
    assert not per_turbine.empty
    assert (per_turbine["estimate"] == 0.0).all(), per_turbine.to_string(index=False)
    assert (result.farm["estimate"] == 0.0).all(), result.farm.to_string(index=False)


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_real_methods_read_about_zero_on_a_noisy_placebo(mode: str, tmp_path: Path) -> None:
    result = run_placebo_fixture(mode, wind_noise_sd=0.4, out_dir=tmp_path)
    per_turbine = per_turbine_table(result)
    assert per_turbine["truth"].abs().max() == 0.0
    assert result.truth_farm_uplift == 0.0
    assert per_turbine["estimate"].abs().max() < NOISY_TOLERANCE, per_turbine.to_string(index=False)
    assert result.farm["estimate"].abs().max() < NOISY_TOLERANCE, result.farm.to_string(index=False)
    assert (result.farm["n_guarded"] == 0).all()


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_pooling_does_not_amplify_the_per_turbine_error(mode: str, tmp_path: Path) -> None:
    result = run_placebo_fixture(mode, wind_noise_sd=0.4, out_dir=tmp_path)
    per_turbine = per_turbine_table(result)
    for method, rows in per_turbine.groupby("method"):
        farm_error = abs(float(result.farm.set_index("method").loc[method, "signed_error"]))
        assert farm_error <= rows["signed_error"].abs().max() + 1e-12


@pytest.mark.parametrize("mode", ["prepost", "toggle"])
def test_every_applicable_method_reported(mode: str, tmp_path: Path) -> None:
    spec = placebo_campaign(mode, upgraded=TEST_TURBINES, turbines=TEST_PARTICIPANTS).spec()
    expected = {m.name for m in carried_forward_methods(spec, out_dir=tmp_path, include_power_model=False)}
    result = run_placebo_fixture(mode, wind_noise_sd=0.0, out_dir=tmp_path)
    assert set(result.farm["method"]) == expected
    assert ("toggle_specialist" in expected) == (mode == "toggle")


def test_the_noisy_fixture_is_not_degenerate() -> None:
    # guards the tolerance test: with constant power a zero uplift would prove nothing
    assert placebo_scada("prepost", wind_noise_sd=0.4)[HOT_COLUMNS.active_power].std() > MIN_POWER_SD_KW
