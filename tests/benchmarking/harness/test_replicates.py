"""Tests for the replicate ensemble (the precision axis)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.harness.replicates import Replicate, StudyConfig, build_replicates, iter_replicates
from benchmarking.synthetic import HOT_COLUMNS, ConstantCpChange, ToggleSchedule
from wind_up_v0.constants import TIMESTAMP_COL

PROFILE = [ConstantCpChange(delta=0.05)]


def _base_scada(turbines: tuple[str, ...] = ("T1", "T3", "T4", "T7", "T99")) -> pd.DataFrame:
    """A small multi-turbine wind farm spanning three years of daily records."""
    index = pd.date_range("2016-01-01", "2018-12-31", freq="1D", tz="UTC")
    frames = [
        pd.DataFrame(
            {
                HOT_COLUMNS.turbine: turbine,
                HOT_COLUMNS.active_power: 1000.0,
                HOT_COLUMNS.wind_speed: 8.0,
                HOT_COLUMNS.wind_speed_sd: 0.8,
                HOT_COLUMNS.gen_rpm: 1400.0,
            },
            index=index,
        )
        for turbine in turbines
    ]
    wf_df = pd.concat(frames)
    wf_df.index.name = TIMESTAMP_COL
    return wf_df


def _study(mode: str = "prepost", n_replicates: int = 5, seed: int = 0) -> StudyConfig:
    return StudyConfig(
        mode=mode,
        turbine_subset=["T1", "T3", "T4", "T7"],
        treatment_start_range=(pd.Timestamp("2017-01-01", tz="UTC"), pd.Timestamp("2017-12-31", tz="UTC")),
        min_pre_months=12,
        campaign_months=[3, 6],
        toggle_period=pd.Timedelta(days=14),
        n_replicates=n_replicates,
        seed=seed,
    )


def _weeks_study(mode: str = "toggle", n_replicates: int = 5, seed: int = 0) -> StudyConfig:
    return StudyConfig(
        mode=mode,
        turbine_subset=["T1", "T3", "T4", "T7"],
        treatment_start_range=(pd.Timestamp("2017-01-01", tz="UTC"), pd.Timestamp("2017-12-31", tz="UTC")),
        min_pre_months=12,
        campaign_weeks=[1, 2, 4, 8],
        toggle_period=pd.Timedelta(days=14),
        n_replicates=n_replicates,
        seed=seed,
    )


def test_max_activity_months_is_the_longest_campaign() -> None:
    assert _study().max_activity_months == 6


class TestCampaignGrid:
    """A study carries exactly one campaign-length grid and describes it generically."""

    def test_months_study_exposes_months(self) -> None:
        study = _study()
        assert study.campaign_lengths == [3, 6]
        assert study.campaign_length_col == "campaign_months"
        assert study.campaign_unit == "months"

    def test_weeks_study_exposes_weeks(self) -> None:
        study = _weeks_study()
        assert study.campaign_lengths == [1, 2, 4, 8]
        assert study.campaign_length_col == "campaign_weeks"
        assert study.campaign_unit == "weeks"

    def test_neither_grid_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            StudyConfig(
                mode="toggle",
                turbine_subset=["T1"],
                treatment_start_range=(pd.Timestamp("2017-01-01", tz="UTC"), pd.Timestamp("2017-12-31", tz="UTC")),
                min_pre_months=12,
                n_replicates=1,
            )

    def test_both_grids_raise(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            StudyConfig(
                mode="toggle",
                turbine_subset=["T1"],
                treatment_start_range=(pd.Timestamp("2017-01-01", tz="UTC"), pd.Timestamp("2017-12-31", tz="UTC")),
                min_pre_months=12,
                campaign_months=[3],
                campaign_weeks=[2],
                n_replicates=1,
            )

    def test_max_activity_months_raises_for_a_weeks_study(self) -> None:
        with pytest.raises(ValueError, match="campaign_unit='weeks'"):
            _ = _weeks_study().max_activity_months


def test_build_replicates_returns_n_replicate_records() -> None:
    reps = build_replicates(_base_scada(), profile=PROFILE, study=_study(n_replicates=5))
    assert len(reps) == 5
    assert all(isinstance(r, Replicate) for r in reps)


def test_data_is_subset_to_turbine_subset() -> None:
    reps = build_replicates(_base_scada(), profile=PROFILE, study=_study())
    present = set(reps[0].dataset.synthetic_df[HOT_COLUMNS.turbine].unique())
    assert present == {"T1", "T3", "T4", "T7"}  # the other ~17 turbines dropped


def test_each_replicate_draws_a_test_turbine_from_the_subset() -> None:
    reps = build_replicates(_base_scada(), profile=PROFILE, study=_study())
    assert all(r.test_wtg in {"T1", "T3", "T4", "T7"} for r in reps)


def test_treatment_start_is_a_pandas_timestamp_supporting_offset_arithmetic() -> None:
    # campaign.py does `treatment_start - pd.DateOffset(...)`, which needs a pd.Timestamp
    reps = build_replicates(_base_scada(), profile=PROFILE, study=_study())
    start = reps[0].treatment_start
    assert isinstance(start, pd.Timestamp)
    assert start.tz is not None  # tz-aware, matching the SCADA index
    _ = start - pd.DateOffset(months=12)  # must not raise


def test_treatment_start_falls_within_the_configured_range() -> None:
    study = _study()
    reps = build_replicates(_base_scada(), profile=PROFILE, study=study)
    lo, hi = study.treatment_start_range
    for r in reps:
        assert lo <= r.treatment_start <= hi


def test_draws_are_deterministic_by_seed() -> None:
    base = _base_scada()
    reps_a = build_replicates(base, profile=PROFILE, study=_study(seed=42))
    reps_b = build_replicates(base, profile=PROFILE, study=_study(seed=42))
    assert [(r.test_wtg, r.treatment_start) for r in reps_a] == [(r.test_wtg, r.treatment_start) for r in reps_b]


def test_different_seed_changes_the_draws() -> None:
    base = _base_scada()
    reps_a = build_replicates(base, profile=PROFILE, study=_study(seed=1))
    reps_b = build_replicates(base, profile=PROFILE, study=_study(seed=2))
    assert [(r.test_wtg, r.treatment_start) for r in reps_a] != [(r.test_wtg, r.treatment_start) for r in reps_b]


def test_prepost_replicate_upgrade_timing_is_the_treatment_start() -> None:
    reps = build_replicates(_base_scada(), profile=PROFILE, study=_study(mode="prepost"))
    r = reps[0]
    assert r.upgrade_timing == r.treatment_start


def test_toggle_replicate_builds_schedule_with_start_and_period() -> None:
    study = _study(mode="toggle")
    reps = build_replicates(_base_scada(), profile=PROFILE, study=study)
    timing = reps[0].upgrade_timing
    assert isinstance(timing, ToggleSchedule)
    assert timing.start == reps[0].treatment_start
    assert timing.period == study.toggle_period


def test_replicate_true_uplift_delegates_to_its_dataset() -> None:
    reps = build_replicates(_base_scada(), profile=PROFILE, study=_study())
    r = reps[0]
    via_replicate = r.true_uplift()
    via_dataset = r.dataset.true_uplift(test_wtg=r.test_wtg)
    assert via_replicate.overall == via_dataset.overall


def test_injected_upgrade_actually_changed_the_test_turbine() -> None:
    reps = build_replicates(_base_scada(), profile=PROFILE, study=_study())
    r = reps[0]
    syn = r.dataset.synthetic_df
    orig = r.dataset.original_df
    test_syn = syn[syn[HOT_COLUMNS.turbine] == r.test_wtg][HOT_COLUMNS.active_power].to_numpy()
    test_orig = orig[orig[HOT_COLUMNS.turbine] == r.test_wtg][HOT_COLUMNS.active_power].to_numpy()
    assert np.any(test_syn != test_orig)  # the profile left a mark


# --- streaming ---------------------------------------------------------------------------------


def test_iter_replicates_yields_the_same_replicates_as_build_replicates() -> None:
    base = _base_scada()
    study = _study(n_replicates=4)
    built = build_replicates(base, profile=PROFILE, study=study)
    streamed = list(iter_replicates(base, profile=PROFILE, study=study))

    assert len(streamed) == len(built)
    for a, b in zip(built, streamed, strict=True):
        assert a.test_wtg == b.test_wtg
        assert a.treatment_start == b.treatment_start
        assert a.replicate_id == b.replicate_id
        pd.testing.assert_frame_equal(a.synthetic_df, b.synthetic_df)


def test_iter_replicates_is_lazy() -> None:
    """The point of streaming: nothing is generated until asked for, so memory stays bounded."""
    base = _base_scada()
    stream = iter_replicates(base, profile=PROFILE, study=_study(n_replicates=100))
    first = next(stream)  # would be unusably slow and huge if all 100 were built up front
    assert first.replicate_id == 0
