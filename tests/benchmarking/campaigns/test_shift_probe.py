"""Tests for the shift probe's arm declaration and its shift measures.

The probe's runs are a driver (they need the Hill of Towie download and the power model); what is
unit-tested here is that every arm is the same campaign over a different window, that each shift
measure reads what it claims off the frames and CSVs a run leaves behind, and that the dose-response
fit recovers a slope that is there and reports none where a dose never moves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.campaigns.placebo import PLACEBO_TURBINES, PLACEBO_UPGRADED
from benchmarking.campaigns.shift_probe import (
    ERA5_WIND_SPEED,
    PROBE_BASELINE_MONTHS,
    PROBE_CAMPAIGN_MONTHS,
    analysis_period,
    arm_name,
    arm_summary,
    cem_doses,
    dose_response,
    plot_dose_response,
    probe_campaign,
    probe_changeovers,
    probe_span,
    reference_summary,
    weather_shift,
)

if TYPE_CHECKING:
    from pathlib import Path

TOLERANCE = 1e-9
TEST_TURBINES = ("T07", "T11", "T12")


def test_every_arm_is_a_full_year_of_baseline_and_the_same_length_of_campaign() -> None:
    for changeover in probe_changeovers():
        start, end = analysis_period(changeover)
        assert start == changeover - pd.DateOffset(months=PROBE_BASELINE_MONTHS)
        assert end == changeover + pd.DateOffset(months=PROBE_CAMPAIGN_MONTHS)


def test_arms_differ_only_in_their_window() -> None:
    campaigns = [probe_campaign(c, turbines=TEST_TURBINES, upgraded=TEST_TURBINES[:1]) for c in probe_changeovers()]
    assert len({tuple(c.turbines) for c in campaigns}) == 1
    assert len({tuple(c.upgraded_turbines) for c in campaigns}) == 1
    assert len({c.analysis_period for c in campaigns}) == len(campaigns)


def test_nothing_is_injected_so_the_truth_is_zero() -> None:
    campaign = probe_campaign(probe_changeovers()[0], turbines=TEST_TURBINES, upgraded=TEST_TURBINES[:1])
    assert campaign.upgrades == []
    assert campaign.faults == []


def test_the_default_arm_declares_the_placebo_farm() -> None:
    campaign = probe_campaign(probe_changeovers()[0])
    assert set(campaign.upgraded_turbines) == set(PLACEBO_UPGRADED)
    assert set(campaign.turbines) == set(PLACEBO_TURBINES)


def test_the_span_covers_every_arm() -> None:
    start, end = probe_span()
    for changeover in probe_changeovers():
        arm_start, arm_end = analysis_period(changeover)
        assert start <= arm_start
        assert end >= arm_end


def test_arm_names_are_distinct_and_carry_the_changeover_month() -> None:
    names = [arm_name(c) for c in probe_changeovers()]
    assert len(set(names)) == len(names)
    assert all(f"{c:%Y%m}" in arm_name(c) for c in probe_changeovers())


def _era5(pre_ws: float, post_ws: float, *, changeover: pd.Timestamp) -> pd.DataFrame:
    """Hourly ERA5 over one arm's window, at ``pre_ws`` before the changeover and ``post_ws`` after."""
    start, end = analysis_period(changeover)
    index = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left", name="timestamp")
    speeds = np.where(index < changeover, pre_ws, post_ws)
    return pd.DataFrame({ERA5_WIND_SPEED: speeds}, index=index)


def test_weather_shift_reads_the_ratio_of_campaign_to_baseline() -> None:
    changeover = probe_changeovers()[0]
    shift = weather_shift(_era5(8.0, 10.0, changeover=changeover), changeover=changeover)
    assert shift["pre_era5_ws"] == 8.0
    assert shift["post_era5_ws"] == 10.0
    assert abs(shift["ws_ratio"] - 1.25) < TOLERANCE
    assert abs(shift["energy_ratio"] - 1.25**3) < TOLERANCE


def test_weather_shift_ignores_records_outside_the_arms_window() -> None:
    changeover = probe_changeovers()[0]
    era5 = _era5(8.0, 10.0, changeover=changeover)
    start, end = analysis_period(changeover)
    outside = pd.DataFrame(
        {ERA5_WIND_SPEED: [99.0, 99.0]},
        index=pd.DatetimeIndex([start - pd.Timedelta(hours=1), end], name="timestamp"),
    )
    shift = weather_shift(pd.concat([outside, era5]).sort_index(), changeover=changeover)
    assert abs(shift["ws_ratio"] - 1.25) < TOLERANCE


def _write_run_csvs(turbine_dir: Path) -> None:
    """Write the three diagnostic CSVs a power-model run leaves under a turbine's folder."""
    run_dir = turbine_dir / "power_model" / "power_model_T07_20170301_20180901" / "conditional"
    run_dir.mkdir(parents=True)
    stem = "power_model_T07_20170301_20180901"
    pd.DataFrame([{"retained_fraction_upgraded": 0.74, "retained_fraction_baseline": 0.72}]).to_csv(
        run_dir / f"{stem}_cem_balance_20260916_000000_000000.csv", index=False
    )
    # 200 of the 500 upgraded rows sit in the cell the baseline covers more thinly than they occupy
    pd.DataFrame(
        [
            {"n_baseline": 100, "n_upgraded": 200},
            {"n_baseline": 400, "n_upgraded": 300},
        ]
    ).to_csv(run_dir / f"{stem}_cem_cells_20260916_000000_000000.csv", index=False)
    pd.DataFrame([{"test_wtg": "T07", "implied_shrinkage": 1.0029}]).to_csv(
        run_dir / f"{stem}_conditional_overall_20260916_000000_000000.csv", index=False
    )


def test_cem_doses_read_the_measures_a_run_wrote(tmp_path: Path) -> None:
    _write_run_csvs(tmp_path)
    doses = cem_doses(tmp_path)
    assert abs(doses["retained_fraction_upgraded"] - 0.74) < TOLERANCE
    assert abs(doses["thin_cell_share"] - 0.4) < TOLERANCE
    assert abs(doses["implied_shrinkage"] - 1.0029) < TOLERANCE


def test_cem_doses_are_missing_rather_than_wrong_when_a_run_wrote_none(tmp_path: Path) -> None:
    doses = cem_doses(tmp_path)
    assert set(doses) == {"retained_fraction_upgraded", "thin_cell_share", "implied_shrinkage"}
    assert all(np.isnan(v) for v in doses.values())


def _estimates(slope_pp: float, *, noise: float = 0.0) -> pd.DataFrame:
    """Probe rows whose bias is ``slope_pp`` per unit of wind-speed ratio, with a constant dose too."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for i, changeover in enumerate(probe_changeovers()):
        ws_ratio = 0.8 + 0.08 * i
        rows.extend(
            {
                "arm": arm_name(changeover),
                "changeover": changeover,
                "method": "power_model",
                "test_wtg": wtg,
                "estimate_pp": slope_pp * (ws_ratio - 1) + rng.normal(0, noise),
                "truth": 0.0,
                "ws_ratio": ws_ratio,
                "energy_ratio": ws_ratio**3,
                "retained_fraction_upgraded": 0.74,
                "thin_cell_share": 0.2,
                "implied_shrinkage": 1.0,
            }
            for wtg in TEST_TURBINES
        )
    return pd.DataFrame(rows)


def test_arm_summary_is_one_row_per_arm_and_method() -> None:
    summary = arm_summary(_estimates(2.0))
    assert len(summary) == len(probe_changeovers())
    assert (summary["n"] == len(TEST_TURBINES)).all()
    assert summary["changeover"].is_monotonic_increasing


def test_dose_response_recovers_a_slope_that_is_there() -> None:
    fit = dose_response(_estimates(2.0)).set_index("dose")
    assert abs(fit.loc["ws_ratio", "slope_pp"] - 2.0) < TOLERANCE
    assert abs(fit.loc["ws_ratio", "intercept_pp"]) < TOLERANCE
    assert abs(fit.loc["ws_ratio", "r"] - 1.0) < TOLERANCE


def test_dose_response_reads_flat_when_the_bias_does_not_move() -> None:
    fit = dose_response(_estimates(0.0)).set_index("dose")
    assert abs(fit.loc["ws_ratio", "slope_pp"]) < TOLERANCE


def test_dose_response_leaves_out_a_dose_no_arm_moved() -> None:
    fit = dose_response(_estimates(2.0))
    assert "thin_cell_share" not in set(fit["dose"])
    assert "ws_ratio" in set(fit["dose"])


def test_the_dose_plot_is_written(tmp_path: Path) -> None:
    path = plot_dose_response(_estimates(2.0, noise=0.1), out_dir=tmp_path)
    assert path.exists()
    assert path.parent == tmp_path


def test_the_reference_arm_declares_one_test_turbine_and_the_rest_as_references() -> None:
    campaign = probe_campaign(probe_changeovers()[0], upgraded=["T13"])
    assert campaign.upgraded_turbines == ["T13"]
    assert "T13" not in campaign.candidate_references
    assert len(campaign.candidate_references) == len(PLACEBO_TURBINES) - 1


def test_reference_summary_is_one_row_per_arm() -> None:
    readings = pd.DataFrame(
        {
            "arm": ["a", "a", "b", "b"],
            "ws_ratio": [0.8, 0.8, 1.2, 1.2],
            "turbine": ["T01", "T02", "T01", "T02"],
            "reading_pp": [0.0, 1.0, 2.0, 4.0],
        }
    )
    summary = reference_summary(readings).set_index("arm")
    assert len(summary) == 2
    assert abs(summary.loc["a", "mean_pp"] - 0.5) < TOLERANCE
    assert abs(summary.loc["b", "median_pp"] - 3.0) < TOLERANCE
    assert (summary["n"] == 2).all()
