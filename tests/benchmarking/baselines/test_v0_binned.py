"""Tests for v0_binned: the v0 run each analysis configures from the facts, and its answer.

v0's own pipeline is stubbed; the slow SMARTEOLE test runs it for real.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

import wind_up
from benchmarking.baselines import v0_binned
from benchmarking.baselines.v0_binned import EXCLUDE_ROW_COL, V0BinnedMethod, v0_records
from tests.wind_up import facts_fixtures as ff
from wind_up.analysis import AnalysisSpec
from wind_up.facts import NonTurbineMeasurement, Reanalysis, Upgrade
from wind_up.labels import Label
from wind_up.method import MethodConfigurationError
from wind_up.operating_state import OperatingStateConfig, StateValidity
from wind_up.toggle import ToggleSchedule
from wind_up_v0.constants import TIMESTAMP_COL, DataColumns
from wind_up_v0.models import PrePost

if TYPE_CHECKING:
    from pathlib import Path

    from wind_up.facts import WindFarmFacts

OPTIONS = {
    "years_offset_for_pre_period": 1,
    "years_for_lt_distribution": 1,
    "years_for_detrend": 1,
    "ws_bin_width": 1.0,
}
TURBINE_TYPE = {"normal_operation_pitch_range": (-10.0, 40.0), "normal_operation_genrpm_range": (500.0, 1600.0)}
TOGGLE_OPTIONS = {"years_for_lt_distribution": 1, "years_for_detrend": 1, "ws_bin_width": 1.0}


def _method(tmp_path: Path, **kwargs: Any) -> V0BinnedMethod:  # noqa: ANN401
    kwargs.setdefault("options", OPTIONS)
    kwargs.setdefault("turbine_type", TURBINE_TYPE)
    return V0BinnedMethod(out_dir=tmp_path, **kwargs)


def _analysis(**fields: Any) -> AnalysisSpec:  # noqa: ANN401
    given = {
        "name": "an analysis",
        "upgrade": "blades",
        "subjects": ["T1"],
        "references": ["T2", "T3"],
        "report": "campaign",
    }
    given.update(fields)
    return AnalysisSpec(**given)


def _toggle_facts(**fields: Any) -> WindFarmFacts:  # noqa: ANN401
    upgrade = Upgrade(name="control", turbines=("T1", "T2"), timing=ff.toggle_frame(), **fields)
    return ff.facts(upgrades=[upgrade])


def _lidar(**columns: str) -> NonTurbineMeasurement:
    records = pd.DataFrame({"ws": 8.0, "wd": 270.0}, index=pd.date_range(ff.START, periods=36, freq=ff.TIMEBASE))
    return NonTurbineMeasurement(name="LiDAR", latitude=57.6, longitude=-3.0, records=records, columns=columns)


class TestOptions:
    def test_a_field_the_facts_set_is_refused(self) -> None:
        with pytest.raises(MethodConfigurationError, match="come from the facts"):
            V0BinnedMethod(options={"exclusion_periods_utc": []})

    def test_a_field_v0_does_not_have_is_refused(self) -> None:
        with pytest.raises(MethodConfigurationError, match="not fields of v0"):
            V0BinnedMethod(options={"no_such_field": 1})

    def test_a_toggle_field_the_facts_set_is_refused(self) -> None:
        with pytest.raises(MethodConfigurationError, match="toggle options"):
            V0BinnedMethod(options={"toggle": {"toggle_change_settling_filter_seconds": 60}})

    def test_a_turbine_type_fact_is_refused(self) -> None:
        with pytest.raises(MethodConfigurationError, match="turbine_type fields"):
            V0BinnedMethod(turbine_type={"rated_power_kw": 2000.0})

    def test_an_unknown_analysis_label_stage_is_refused(self) -> None:
        with pytest.raises(MethodConfigurationError, match="analysis_label_stage"):
            V0BinnedMethod(analysis_label_stage="detrending")  # type: ignore[arg-type]

    def test_a_required_turbine_type_field_missing_is_named(self, tmp_path: Path) -> None:
        method = _method(tmp_path, turbine_type={})
        with pytest.raises(MethodConfigurationError, match="normal_operation_pitch_range"):
            method._config(ff.facts(), _analysis(), use_lt_distribution=False)  # noqa: SLF001

    def test_a_required_option_missing_is_named(self, tmp_path: Path) -> None:
        method = _method(tmp_path, options={"years_offset_for_pre_period": 1})
        with pytest.raises(MethodConfigurationError, match="ws_bin_width"):
            method._config(ff.facts(), _analysis(), use_lt_distribution=False)  # noqa: SLF001

    def test_a_permanent_upgrade_needs_a_pre_period(self, tmp_path: Path) -> None:
        method = _method(tmp_path, options=TOGGLE_OPTIONS)
        with pytest.raises(MethodConfigurationError, match="years_offset_for_pre_period"):
            method._config(ff.facts(), _analysis(), use_lt_distribution=False)  # noqa: SLF001


class TestConfigurationFromTheFacts:
    def test_subjects_and_references(self, tmp_path: Path) -> None:
        cfg = _method(tmp_path)._config(ff.facts(), _analysis(), use_lt_distribution=False)  # noqa: SLF001
        assert [t.name for t in cfg.test_wtgs] == ["T1"]
        assert [t.name for t in cfg.ref_wtgs] == ["T2", "T3"]
        assert cfg.non_wtg_ref_names == []

    def test_the_asset_is_the_turbine_table(self, tmp_path: Path) -> None:
        method = _method(
            tmp_path, turbine_type={**TURBINE_TYPE, "turbine_type": "Type X", "pitch_to_stall": True}, site_name="Farm"
        )
        cfg = method._config(ff.facts(), _analysis(), use_lt_distribution=False)  # noqa: SLF001
        assert cfg.asset.name == "Farm"
        assert [t.name for t in cfg.asset.wtgs] == list(ff.NAMES)
        first = cfg.asset.wtgs[0]
        assert first.turbine_type.rotor_diameter_m == 100.0
        assert first.turbine_type.rated_power_kw == ff.RATED
        assert first.turbine_type.pitch_to_stall
        assert first.latitude == 57.5

    def test_a_measurement_reference_is_a_non_turbine_reference(self, tmp_path: Path) -> None:
        facts = ff.facts(non_turbine_measurements=[_lidar(wind_speed="ws", wind_direction="wd")])
        cfg = _method(tmp_path)._config(facts, _analysis(references=["T2", "LiDAR"]), use_lt_distribution=False)  # noqa: SLF001
        assert [t.name for t in cfg.ref_wtgs] == ["T2"]
        assert cfg.non_wtg_ref_names == ["LiDAR"]
        (mast,) = cfg.asset.masts_and_lidars
        assert (mast.wind_speed_column, mast.wind_direction_column) == ("ws", "wd")

    def test_a_measurement_needs_its_wind_roles(self, tmp_path: Path) -> None:
        facts = ff.facts(non_turbine_measurements=[_lidar(wind_speed="ws")])
        with pytest.raises(MethodConfigurationError, match="wind_direction"):
            _method(tmp_path)._config(facts, _analysis(), use_lt_distribution=False)  # noqa: SLF001

    def test_dates_follow_the_changeover_and_the_period(self, tmp_path: Path) -> None:
        analysis = _analysis(period=(None, ff.at(5)))
        cfg = _method(tmp_path)._config(ff.facts(), analysis, use_lt_distribution=False)  # noqa: SLF001
        assert cfg.upgrade_first_dt_utc_start == ff.at(3)
        assert cfg.analysis_last_dt_utc_start == ff.at(5) - ff.TIMEBASE
        assert cfg.prepost is not None
        assert cfg.prepost.pre_first_dt_utc_start == ff.at(3) - pd.DateOffset(years=1)
        assert cfg.prepost.pre_last_dt_utc_start == ff.at(5) - ff.TIMEBASE - pd.DateOffset(years=1)
        assert cfg.lt_last_dt_utc_start == cfg.prepost.pre_last_dt_utc_start
        assert cfg.detrend_first_dt_utc_start == cfg.lt_last_dt_utc_start - pd.DateOffset(years=1)

    def test_without_a_period_end_the_analysis_runs_to_the_last_record(self, tmp_path: Path) -> None:
        cfg = _method(tmp_path)._config(ff.facts(), _analysis(), use_lt_distribution=False)  # noqa: SLF001
        assert cfg.analysis_last_dt_utc_start == ff.at(35 / 6)

    def test_the_pre_period_ends_a_period_before_the_upgrade_at_latest(self, tmp_path: Path) -> None:
        analysis = _analysis(period=(None, ff.at(3) + pd.DateOffset(years=1, days=1)))
        cfg = _method(tmp_path)._config(ff.facts(), analysis, use_lt_distribution=False)  # noqa: SLF001
        assert cfg.prepost.pre_last_dt_utc_start == ff.at(3) - ff.TIMEBASE

    def test_a_period_starting_after_the_pre_period_is_refused(self, tmp_path: Path) -> None:
        analysis = _analysis(period=(ff.at(1), None))
        with pytest.raises(MethodConfigurationError, match="option prepost"):
            _method(tmp_path)._config(ff.facts(), analysis, use_lt_distribution=False)  # noqa: SLF001

    def test_given_dates_are_kept(self, tmp_path: Path) -> None:
        options = {**OPTIONS, "lt_first_dt_utc_start": ff.at(0), "lt_last_dt_utc_start": ff.at(1)}
        cfg = _method(tmp_path, options=options)._config(ff.facts(), _analysis(), use_lt_distribution=False)  # noqa: SLF001
        assert (cfg.lt_first_dt_utc_start, cfg.lt_last_dt_utc_start) == (ff.at(0), ff.at(1))

    def test_a_given_prepost_is_used(self, tmp_path: Path) -> None:
        prepost = {
            "pre_first_dt_utc_start": ff.at(0),
            "pre_last_dt_utc_start": ff.at(1),
            "post_first_dt_utc_start": ff.at(3),
            "post_last_dt_utc_start": ff.at(5),
        }
        method = _method(tmp_path, options={**OPTIONS, "prepost": prepost})
        cfg = method._config(ff.facts(), _analysis(), use_lt_distribution=False)  # noqa: SLF001
        assert cfg.prepost == PrePost.model_validate(prepost)

    def test_the_report_sets_the_long_term_distribution(self, tmp_path: Path) -> None:
        method = _method(tmp_path)
        assert method._config(ff.facts(), _analysis(report="aep"), use_lt_distribution=True).use_lt_distribution  # noqa: SLF001

    def test_both_reports_run_in_their_own_folders(self, tmp_path: Path) -> None:
        method = _method(tmp_path)
        analysis = _analysis(report=("aep", "campaign"))
        aep = method._config(ff.facts(), analysis, use_lt_distribution=True)  # noqa: SLF001
        campaign = method._config(ff.facts(), analysis, use_lt_distribution=False)  # noqa: SLF001
        assert aep.out_dir != campaign.out_dir

    def test_exclusions_installation_windows_and_northing(self, tmp_path: Path) -> None:
        window = (ff.at(1), ff.at(2))
        facts = ff.facts(
            upgrades=[ff.permanent(installation_windows={"T1": [window]})],
            exclusions=[(None, ff.at(4), ff.at(5)), ("T2", ff.at(0), ff.at(1))],
            north_offsets=[("T1", ff.at(0), 3.0)],
        )
        cfg = _method(tmp_path)._config(facts, _analysis(), use_lt_distribution=False)  # noqa: SLF001
        assert [tuple(x) for x in cfg.exclusion_periods_utc] == [
            ("ALL", ff.at(4), ff.at(5)),
            ("T2", ff.at(0), ff.at(1)),
            ("T1", *window),
        ]
        assert [tuple(x) for x in cfg.northing_corrections_utc] == [("T1", ff.at(0), 3.0)]

    def test_subjects_changing_over_apart_are_refused(self, tmp_path: Path) -> None:
        upgrade = Upgrade(name="blades", turbines=("T1", "T2"), timing={"T1": ff.at(2), "T2": ff.at(3)})
        analysis = _analysis(subjects=["T1", "T2"], references=["T3"])
        with pytest.raises(MethodConfigurationError, match="one changeover"):
            _method(tmp_path)._config(ff.facts(upgrades=[upgrade]), analysis, use_lt_distribution=False)  # noqa: SLF001


class TestToggleConfiguration:
    def test_the_toggle_starts_at_its_first_stated_record_before_settling(self, tmp_path: Path) -> None:
        facts = _toggle_facts(settling_time=pd.Timedelta(minutes=10))
        analysis = _analysis(upgrade="control", references=["T3"])
        method = _method(tmp_path, options={**TOGGLE_OPTIONS, "toggle": {"pairing_filter_method": "none"}})
        cfg = method._config(facts, analysis, use_lt_distribution=False)  # noqa: SLF001
        assert cfg.upgrade_first_dt_utc_start == ff.at(1)
        assert cfg.analysis_first_dt_utc_start == ff.at(1)
        assert cfg.lt_last_dt_utc_start == ff.at(1) - pd.Timedelta(days=7)
        assert cfg.toggle is not None
        assert cfg.toggle.toggle_change_settling_filter_seconds == 600
        assert cfg.toggle.pairing_filter_method == "none"

    def test_the_toggle_is_named_for_the_upgrade(self, tmp_path: Path) -> None:
        analysis = _analysis(upgrade="control", references=["T3"])
        cfg = _method(tmp_path, options=TOGGLE_OPTIONS)._config(_toggle_facts(), analysis, use_lt_distribution=False)  # noqa: SLF001
        assert cfg.toggle is not None
        assert cfg.toggle.name == "control"

    def test_without_years_a_campaign_toggle_takes_its_periods_from_the_analysis(self, tmp_path: Path) -> None:
        options = {"ws_bin_width": 1.0, "toggle": {"detrend_data_selection": "use_toggle_off_data"}}
        analysis = _analysis(upgrade="control", references=["T3"])
        cfg = _method(tmp_path, options=options)._config(_toggle_facts(), analysis, use_lt_distribution=False)  # noqa: SLF001
        span = (cfg.analysis_first_dt_utc_start, cfg.analysis_last_dt_utc_start)
        assert (cfg.lt_first_dt_utc_start, cfg.lt_last_dt_utc_start) == span
        assert (cfg.detrend_first_dt_utc_start, cfg.detrend_last_dt_utc_start) == span

    def test_without_years_an_aep_or_a_toggle_detrended_on_all_records_needs_them(self, tmp_path: Path) -> None:
        analysis = _analysis(upgrade="control", references=["T3"])
        off = {"ws_bin_width": 1.0, "toggle": {"detrend_data_selection": "use_toggle_off_data"}}
        with pytest.raises(MethodConfigurationError, match="years_for_lt_distribution"):
            _method(tmp_path, options=off)._config(_toggle_facts(), analysis, use_lt_distribution=True)  # noqa: SLF001
        with pytest.raises(MethodConfigurationError, match="years_for_detrend"):
            _method(tmp_path, options={"ws_bin_width": 1.0})._config(  # noqa: SLF001
                _toggle_facts(), analysis, use_lt_distribution=False
            )

    def test_detrending_on_off_records_runs_to_the_analysis_end(self, tmp_path: Path) -> None:
        options = {**TOGGLE_OPTIONS, "toggle": {"detrend_data_selection": "use_toggle_off_data"}}
        analysis = _analysis(upgrade="control", references=["T3"])
        cfg = _method(tmp_path, options=options)._config(_toggle_facts(), analysis, use_lt_distribution=False)  # noqa: SLF001
        assert cfg.detrend_last_dt_utc_start == cfg.analysis_last_dt_utc_start

    def test_an_override_moves_the_first_stated_record(self, tmp_path: Path) -> None:
        facts = _toggle_facts(upgrade_state_overrides=[(None, ff.at(0), ff.at(2), "neither")])
        analysis = _analysis(upgrade="control", references=["T3"])
        cfg = _method(tmp_path, options=TOGGLE_OPTIONS)._config(facts, analysis, use_lt_distribution=False)  # noqa: SLF001
        assert cfg.upgrade_first_dt_utc_start == ff.at(2)

    def test_a_toggle_with_no_stated_record_in_the_period_raises(self, tmp_path: Path) -> None:
        analysis = _analysis(upgrade="control", references=["T3"], period=(ff.at(10), None))
        facts = _toggle_facts(upgrade_state_overrides=[(None, ff.at(0), ff.at(12), "neither")])
        with pytest.raises(ValueError, match="no on or off record"):
            _method(tmp_path, options=TOGGLE_OPTIONS)._config(facts, analysis, use_lt_distribution=False)  # noqa: SLF001

    def test_a_schedule_becomes_v0s_toggle_frame(self) -> None:
        schedule = ToggleSchedule(period=pd.Timedelta(minutes=20), start=ff.START)
        upgrade = Upgrade(name="control", turbines=("T1",), timing=schedule)
        index = pd.date_range(ff.START, periods=6, freq=ff.TIMEBASE)
        frame = v0_binned._toggle_frame(upgrade, index=index, turbines=ff.NAMES)  # noqa: SLF001
        assert frame is not None
        assert list(frame.columns[:2]) == ["toggle_on", "toggle_off"]
        assert v0_binned._toggle_frame(ff.permanent(), index=index, turbines=ff.NAMES) is None  # noqa: SLF001

    def test_frames_per_turbine_become_v0s_frame_per_turbine(self, tmp_path: Path) -> None:
        late = ff.toggle_frame(first=12)
        upgrade = Upgrade(name="control", turbines=("T1", "T2"), timing={"T1": ff.toggle_frame(), "T2": late})
        facts = ff.facts(upgrades=[upgrade])
        analysis = _analysis(upgrade="control", subjects=["T2"], references=["T3"])
        cfg = _method(tmp_path, options=TOGGLE_OPTIONS)._config(facts, analysis, use_lt_distribution=False)  # noqa: SLF001
        assert cfg.toggle is not None
        assert cfg.toggle.toggle_file_per_turbine
        assert cfg.toggle.placebo_toggle_wtgs == ["T3"]
        assert cfg.upgrade_first_dt_utc_start == ff.at(2)
        index = pd.date_range(ff.START, periods=36, freq=ff.TIMEBASE)
        frame = v0_binned._toggle_frame(upgrade, index=index, turbines=ff.NAMES)  # noqa: SLF001
        assert frame is not None
        assert list(frame.index.get_level_values(0).unique()) == list(ff.NAMES)
        pd.testing.assert_frame_equal(frame.loc["T2"], late, check_names=False)
        pd.testing.assert_frame_equal(frame.loc["T3"], ff.toggle_frame(), check_names=False)


class TestExcludedRecords:
    @staticmethod
    def _excluded(facts: WindFarmFacts, analysis: AnalysisSpec, *, analysis_labels: bool = False) -> pd.Series:
        upgrade = facts.upgrade(analysis.upgrade)
        mask = v0_binned._excluded_records(  # noqa: SLF001
            facts, analysis, upgrade=upgrade, first=ff.at(3), analysis_labels=analysis_labels
        )
        return pd.Series(mask, index=pd.MultiIndex.from_arrays([facts.records.index, facts.records[ff.COLS.turbine]]))

    @staticmethod
    def _labelled(hours: tuple[float, float], turbine: str | None = None) -> pd.DataFrame:
        frame = ff.records()
        inside = (frame.index >= ff.at(hours[0])) & (frame.index < ff.at(hours[1]))
        if turbine is not None:
            inside &= (frame[ff.COLS.turbine] == turbine).to_numpy()
        return frame.assign(flag=inside)

    def test_nothing_is_excluded_by_default(self) -> None:
        assert not self._excluded(ff.facts(), _analysis()).any()

    def test_a_label_withholding_uplift_excludes_every_turbine_it_holds_on(self) -> None:
        facts = ff.facts(self._labelled((1, 2)), labels=[Label(name="curtailed", column="flag", withholds={"uplift"})])
        excluded = self._excluded(facts, _analysis())
        assert excluded.groupby(level=1).sum().to_dict() == dict.fromkeys(ff.NAMES, 6)

    def test_an_operating_state_override_excludes_its_turbine(self) -> None:
        facts = ff.facts(operating_state_overrides=[("T2", ff.at(1), ff.at(2), "partial downtime")])
        excluded = self._excluded(facts, _analysis())
        assert excluded.groupby(level=1).sum().to_dict() == {"T1": 0, "T2": 6, "T3": 0, "T4": 0}

    def test_a_site_state_not_valid_for_uplift_excludes_its_records(self) -> None:
        frame = ff.records()
        derated = (frame.index >= ff.at(1)) & (frame.index < ff.at(2)) & (frame[ff.COLS.turbine] == "T3").to_numpy()
        frame["state"] = np.where(derated, "derated", None)
        frame.loc[(frame.index >= ff.at(4)) & (frame[ff.COLS.turbine] == "T4").to_numpy(), "state"] = "icing"
        config = OperatingStateConfig(
            site_state_column="state",
            site_states={
                "derated": StateValidity(northing=True, waking="part waking", uplift=False),
                "icing": StateValidity(northing=True, waking="waking", uplift=True),
            },
        )
        excluded = self._excluded(ff.facts(frame, operating_states=config), _analysis())
        assert excluded.groupby(level=1).sum().to_dict() == {"T1": 0, "T2": 0, "T3": 6, "T4": 0}

    def test_an_analysis_label_at_preprocessing_excludes_subject_records_only(self) -> None:
        facts = ff.facts(self._labelled((1, 2)), labels=[Label(name="steered", column="flag")])
        excluded = self._excluded(facts, _analysis(exclude_labels=["steered"]), analysis_labels=True)
        assert excluded.groupby(level=1).sum().to_dict() == {"T1": 6, "T2": 0, "T3": 0, "T4": 0}
        included = self._excluded(facts, _analysis(include_labels=["steered"]), analysis_labels=True)
        assert included.groupby(level=1).sum().to_dict() == {"T1": 30, "T2": 0, "T3": 0, "T4": 0}

    def test_an_analysis_label_at_comparison_excludes_nothing(self) -> None:
        facts = ff.facts(self._labelled((1, 2)), labels=[Label(name="steered", column="flag")])
        assert not self._excluded(facts, _analysis(exclude_labels=["steered"])).any()

    def test_comparison_exclusions_are_each_subjects_left_out_timestamps(self) -> None:
        facts = ff.facts(self._labelled((1, 2)), labels=[Label(name="steered", column="flag")])
        analysis = _analysis(subjects=["T1", "T4"], references=["T2"], exclude_labels=["steered"])
        exclusions = v0_binned._comparison_exclusions(facts, analysis)  # noqa: SLF001
        expected = pd.date_range(ff.at(1), ff.at(2), freq=ff.TIMEBASE, inclusive="left")
        assert sorted(exclusions) == ["T1", "T4"]
        for subject in ("T1", "T4"):
            pd.testing.assert_index_equal(exclusions[subject], expected, check_names=False, check_freq=False)

    def test_a_neither_override_excludes_from_the_first_upgraded_record(self) -> None:
        upgrade = ff.permanent(upgrade_state_overrides=[("T2", ff.at(2), ff.at(4), "neither")])
        excluded = self._excluded(ff.facts(upgrades=[upgrade]), _analysis())
        assert excluded.groupby(level=1).sum().to_dict() == {"T1": 0, "T2": 6, "T3": 0, "T4": 0}

    def test_an_override_to_on_or_off_cannot_reach_v0(self) -> None:
        upgrade = ff.permanent(upgrade_state_overrides=[("T1", ff.at(4), ff.at(5), "off")])
        with pytest.raises(MethodConfigurationError, match="reads one timing"):
            self._excluded(ff.facts(upgrades=[upgrade]), _analysis())


class TestV0Records:
    def test_roles_take_v0s_names_and_downtime_comes_from_availability(self) -> None:
        records = ff.records().assign(**{ff.COLS.availability: 450.0})
        out = v0_records(records, columns=ff.COLS, timebase=ff.TIMEBASE)
        assert {DataColumns.turbine_name, DataColumns.active_power_mean, DataColumns.pitch_angle_mean} <= set(out)
        assert (out[DataColumns.shutdown_duration] == 150.0).all()
        assert out.index.name == TIMESTAMP_COL
        assert records.index.name is None

    def test_the_nacelle_extremes_take_the_yaw_angle_names_v0_filters_on(self) -> None:
        columns = dataclasses.replace(ff.COLS, nacelle_position_min="nac_min", nacelle_position_max="nac_max")
        records = ff.records().assign(nac_min=1.0, nac_max=2.0, ShutdownDuration=0.0)
        out = v0_records(records, columns=columns, timebase=ff.TIMEBASE)
        assert (out[DataColumns.yaw_angle_min] == 1.0).all()
        assert (out[DataColumns.yaw_angle_max] == 2.0).all()

    def test_a_column_already_named_for_v0_is_kept(self) -> None:
        records = ff.records().assign(ActivePowerMean=1.0, ShutdownDuration=0.0)
        out = v0_records(records, columns=ff.COLS, timebase=ff.TIMEBASE)
        assert (out[DataColumns.active_power_mean] == 1.0).all()
        assert ff.COLS.active_power in out.columns

    def test_without_downtime_or_availability_it_raises(self) -> None:
        columns = ff.COLS.__class__(turbine=ff.COLS.turbine, active_power=ff.COLS.active_power)
        with pytest.raises(MethodConfigurationError, match="ShutdownDuration"):
            v0_records(ff.records(), columns=columns, timebase=ff.TIMEBASE)


def _pairs(subjects: list[str], references: list[str], *, uplift: float = 0.02) -> pd.DataFrame:
    rows = [
        {
            "test_wtg": s,
            "ref": r,
            "uplift_frc": uplift,
            "unc_one_sigma_frc": 0.01,
            "pp_valid_hours_pre": 100.0 + i,
            "pp_valid_hours_post": 90.0 + i,
            "mean_power_post": 1000.0,
        }
        for i, (s, r) in enumerate((s, r) for s in subjects for r in references)
    ]
    return pd.DataFrame(rows)


class _Pipeline:
    """v0's pipeline replaced: records what it was given and returns fixed test-reference results."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, fails: bool = False, unmeasured: str = "") -> None:
        self.calls: list[dict] = []
        self.fails = fails
        self.unmeasured = unmeasured
        monkeypatch.setattr(v0_binned.AssessmentInputs, "from_cfg", self.from_cfg)
        monkeypatch.setattr(v0_binned, "run_wind_up_analysis", self.run)

    def from_cfg(self, **kwargs: Any) -> dict:  # noqa: ANN401
        self.calls.append(kwargs)
        return kwargs

    def run(self, inputs: dict) -> pd.DataFrame:
        if self.fails:
            msg = "not enough data"
            raise ValueError(msg)
        cfg = inputs["cfg"]
        pairs = _pairs([t.name for t in cfg.test_wtgs], [t.name for t in cfg.ref_wtgs])
        if self.unmeasured == "nan":
            return pairs.assign(uplift_frc=np.nan, unc_one_sigma_frc=np.nan)
        if self.unmeasured == "missing":
            return pairs.drop(columns=["uplift_frc", "unc_one_sigma_frc"])
        return pairs


class TestAnsweringAnalyses:
    def test_mast_wind_and_operating_defaults_fill_missing_turbine_signals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline = _Pipeline(monkeypatch)
        frame = ff.records().drop(columns=ff.COLS.wind_speed)
        columns = dataclasses.replace(ff.COLS, wind_speed=None)
        times = frame.index.unique()
        mast = NonTurbineMeasurement(
            name="mast",
            latitude=57.5,
            longitude=-3.0,
            records=pd.DataFrame({"speed": np.arange(len(times)), "direction": 220.0}, index=times),
            columns={"wind_speed": "speed", "wind_direction": "direction"},
        )
        facts = ff.facts(frame, columns=columns, non_turbine_measurements=[mast])
        method = _method(tmp_path, turbine_wind_measurement="mast", default_pitch_deg=0, default_gen_rpm=1000)

        method.estimate(facts, [_analysis()])

        converted = pipeline.calls[0]["scada_df"]
        np.testing.assert_array_equal(converted[DataColumns.wind_speed_mean], mast.records["speed"].loc[frame.index])
        assert converted[DataColumns.yaw_angle_mean].eq(220).all()
        assert converted[DataColumns.gen_rpm_mean].eq(1000).all()
        np.testing.assert_array_equal(converted[DataColumns.pitch_angle_mean], frame[ff.COLS.pitch])

    def test_configured_turbine_wind_measurement_must_be_declared(self, tmp_path: Path) -> None:
        method = _method(tmp_path, turbine_wind_measurement="missing mast")
        with pytest.raises(MethodConfigurationError, match="unknown turbine wind measurement 'missing mast'"):
            method.estimate(ff.facts(), [_analysis()])

    def test_one_subject_gives_v0s_combined_uplift_and_sigma(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _Pipeline(monkeypatch)
        (out,) = _method(tmp_path, auto_choose_refs=False).estimate(ff.facts(), [_analysis()])
        assert out.campaign_uplift == pytest.approx(0.02)
        assert np.isfinite(out.campaign_sigma)
        assert np.isnan(out.aep_uplift)
        assert out.per_reference is not None
        assert list(out.per_reference["ref"]) == ["T2", "T3"]
        row = out.per_turbine.iloc[0]
        assert (row["hours_on"], row["hours_off"]) == (90.0, 100.0)
        # the mean over its two pairs of 1000 kW times 90 and 91 h
        assert row["actual_energy"] == pytest.approx(90_500.0)

    def test_a_group_gets_v0s_total_uplift(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _Pipeline(monkeypatch)
        facts = ff.facts(upgrades=[ff.permanent(turbines=("T1", "T2"))])
        analysis = _analysis(subjects=["T1", "T2"], references=["T3", "T4"], report="aep")
        (out,) = _method(tmp_path, auto_choose_refs=False).estimate(facts, [analysis])
        assert out.aep_uplift == pytest.approx(0.02)
        assert np.isfinite(out.aep_sigma)
        assert list(out.per_turbine["turbine"]) == ["T1", "T2"]

    def test_a_group_asked_for_its_net_uplift_gets_it_and_no_sigma(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _Pipeline(monkeypatch)
        facts = ff.facts(upgrades=[ff.permanent(turbines=("T1", "T2"))])
        analysis = _analysis(subjects=["T1", "T2"], references=["T3", "T4"], report="aep", combination="net")
        (out,) = _method(tmp_path, auto_choose_refs=False).estimate(facts, [analysis])
        assert out.aep_uplift == pytest.approx(0.02)
        assert out.aep_uplift == wind_up.combine_subjects(out).aep_uplift
        assert np.isnan(out.aep_sigma)
        assert any(w.severity == "low" and "no sigma for a group's net uplift" in w.message for w in out.warnings)

    @pytest.mark.parametrize("per_turbine_plots", [True, False])
    def test_per_turbine_plots_reach_v0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, per_turbine_plots: bool
    ) -> None:
        pipeline = _Pipeline(monkeypatch)
        method = _method(tmp_path, auto_choose_refs=False, per_turbine_plots=per_turbine_plots)
        method.estimate(ff.facts(), [_analysis()])
        assert pipeline.calls[0]["plot_cfg"].skip_per_turbine_plots is not per_turbine_plots

    def test_both_reports_take_two_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pipeline = _Pipeline(monkeypatch)
        (out,) = _method(tmp_path, auto_choose_refs=False).estimate(ff.facts(), [_analysis(report=("aep", "campaign"))])
        assert [c["cfg"].use_lt_distribution for c in pipeline.calls] == [True, False]
        assert np.isfinite(out.aep_uplift)
        assert np.isfinite(out.campaign_uplift)
        assert set(out.per_reference["report"]) == {"aep", "campaign"}

    def test_a_run_that_fails_leaves_every_subject_unmeasured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _Pipeline(monkeypatch, fails=True)
        (out,) = _method(tmp_path).estimate(ff.facts(), [_analysis()])
        assert np.isnan(out.campaign_uplift)
        assert out.per_reference is None
        assert any(w.severity == "high" and "not enough data" in w.message for w in out.warnings)

    @pytest.mark.parametrize("unmeasured", ["nan", "missing"])
    def test_a_run_v0_cannot_combine_keeps_its_test_reference_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unmeasured: str
    ) -> None:
        _Pipeline(monkeypatch, unmeasured=unmeasured)
        (out,) = _method(tmp_path).estimate(ff.facts(), [_analysis()])
        assert np.isnan(out.campaign_uplift)
        assert out.per_reference is not None
        assert list(out.per_reference["ref"]) == ["T2", "T3"]
        assert np.isnan(out.per_turbine.iloc[0]["campaign_uplift"])
        assert any(w.severity == "high" and "could not combine" in w.message for w in out.warnings)

    def test_the_records_reach_v0_without_label_columns_and_analysis_labels_as_comparison_exclusions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline = _Pipeline(monkeypatch)
        frame = ff.records().assign(flag=lambda f: (f.index >= ff.at(1)) & (f.index < ff.at(2)))
        facts = ff.facts(frame, labels=[Label(name="steered", column="flag")])
        _method(tmp_path).estimate(facts, [_analysis(exclude_labels=["steered"])])
        call = pipeline.calls[0]
        scada = call["scada_df"]
        assert "flag" not in scada.columns
        assert EXCLUDE_ROW_COL not in scada.columns
        assert scada.index.name == TIMESTAMP_COL
        assert {s: len(t) for s, t in call["comparison_exclusions"].items()} == {"T1": 6}

    def test_analysis_labels_at_preprocessing_reach_v0_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline = _Pipeline(monkeypatch)
        frame = ff.records().assign(flag=lambda f: (f.index >= ff.at(1)) & (f.index < ff.at(2)))
        facts = ff.facts(frame, labels=[Label(name="steered", column="flag")])
        method = _method(tmp_path, analysis_label_stage="preprocessing")
        method.estimate(facts, [_analysis(exclude_labels=["steered"])])
        call = pipeline.calls[0]
        assert call["scada_df"][EXCLUDE_ROW_COL].sum() == 6
        assert call["comparison_exclusions"] is None

    def test_the_records_reach_v0_without_the_site_state_column(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline = _Pipeline(monkeypatch)
        frame = ff.records().assign(state=None)
        config = OperatingStateConfig(
            site_state_column="state",
            site_states={"derated": StateValidity(northing=True, waking="part waking", uplift=False)},
        )
        _method(tmp_path).estimate(ff.facts(frame, operating_states=config), [_analysis()])
        assert "state" not in pipeline.calls[0]["scada_df"].columns

    def test_nothing_excluded_adds_no_exclude_row(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pipeline = _Pipeline(monkeypatch)
        _method(tmp_path).estimate(ff.facts(), [_analysis()])
        assert EXCLUDE_ROW_COL not in pipeline.calls[0]["scada_df"].columns

    def test_the_inputs_come_from_the_facts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pipeline = _Pipeline(monkeypatch)
        reanalysis = Reanalysis(name="node", latitude=57.5, longitude=-3.0, records=pd.DataFrame({"x": [1.0]}))
        facts = ff.facts(
            reanalysis=[reanalysis], non_turbine_measurements=[_lidar(wind_speed="ws", wind_direction="wd")]
        )
        _method(tmp_path, to_v0_records=lambda r: r.assign(converted=True)).estimate(facts, [_analysis()])
        call = pipeline.calls[0]
        assert [d.id for d in call["reanalysis_datasets"]] == ["node"]
        assert [d.id for d in call["mast_or_lidar_datasets"]] == ["LiDAR"]
        assert call["scada_df"]["converted"].all()
        assert call["toggle_df"] is None
        metadata = call["metadata_df"]
        assert list(metadata["Name"]) == list(ff.NAMES)
        assert set(metadata["TimeFormat"]) == {"Start"}
        assert set(metadata["TimeSpanMinutes"]) == {10.0}

    def test_a_toggle_hands_v0_the_toggle_frame(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pipeline = _Pipeline(monkeypatch)
        facts = _toggle_facts()
        _method(tmp_path, options=TOGGLE_OPTIONS).estimate(facts, [_analysis(upgrade="control", references=["T3"])])
        pd.testing.assert_frame_equal(pipeline.calls[0]["toggle_df"], ff.toggle_frame())

    def test_the_turbine_table_needs_rotor_diameters(self, tmp_path: Path) -> None:
        facts = ff.facts(turbines=ff.turbine_table().drop(columns="rotor_diameter_m"))
        with pytest.raises(MethodConfigurationError, match="rotor_diameter_m"):
            _method(tmp_path).estimate(facts, [_analysis()])

    def test_without_an_out_dir_a_temp_dir_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pipeline = _Pipeline(monkeypatch)
        V0BinnedMethod(options=OPTIONS, turbine_type=TURBINE_TYPE).estimate(ff.facts(), [_analysis()])
        assert pipeline.calls[0]["cfg"].out_dir.exists()


def test_a_configuration_error_in_a_run_fails_the_call(tmp_path: Path) -> None:
    upgrade = Upgrade(name="blades", turbines=("T1", "T2"), timing={"T1": ff.at(2), "T2": ff.at(3)})
    analysis = _analysis(subjects=["T1", "T2"], references=["T3"])
    with pytest.raises(MethodConfigurationError, match="one changeover"):
        _method(tmp_path).estimate(ff.facts(upgrades=[upgrade]), [analysis])


class TestPlacebo:
    def test_a_placebo_reaches_v0_as_a_placebo_toggle_with_its_own_frame(self, tmp_path: Path) -> None:
        reversed_order = ff.toggle_frame().rename(columns={"toggle_on": "toggle_off", "toggle_off": "toggle_on"})
        analysis = _analysis(upgrade="control", references=["T3", "T4"], placebo_timing={"T3": reversed_order})
        facts = _toggle_facts()
        cfg = _method(tmp_path, options=TOGGLE_OPTIONS)._config(facts, analysis, use_lt_distribution=False)  # noqa: SLF001
        assert cfg.toggle is not None
        assert cfg.toggle.placebo_toggle_wtgs == ["T3", "T4"]
        assert cfg.toggle.toggle_file_per_turbine
        index = pd.date_range(ff.START, periods=36, freq=ff.TIMEBASE)
        frame = v0_binned._toggle_frame(  # noqa: SLF001
            facts.upgrade("control"), index=index, turbines=ff.NAMES, placebo=analysis.placebo_timing
        )
        assert frame is not None
        pd.testing.assert_frame_equal(frame.loc["T3"], reversed_order, check_names=False, check_like=True)
        pd.testing.assert_frame_equal(frame.loc["T1"], ff.toggle_frame(), check_names=False)
        pd.testing.assert_frame_equal(frame.loc["T4"], ff.toggle_frame(), check_names=False)

    def test_a_placebo_row_is_v0s_placebo_test_and_left_out_of_the_uplift(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trdf = pd.DataFrame(
            {
                "test_wtg": ["T1", "T1", "T3"],
                "ref": ["T3", "T4", "T4"],
                "pp_valid_hours_post": [10.0, 11.0, 12.0],
                "pp_valid_hours_pre": [9.0, 9.5, 8.0],
            }
        )
        tdf = pd.DataFrame(
            {"test_wtg": ["T1", "T3"], "is_ref": [False, True], "p50_uplift": [0.03, 0.001], "sigma": [0.01, 0.02]}
        )
        monkeypatch.setattr(V0BinnedMethod, "_run", lambda *_a, **_k: (trdf, tdf))
        monkeypatch.setattr(v0_binned, "_actual_energy", lambda _pairs: 100.0)
        analysis = _analysis(
            upgrade="control",
            subjects=["T1"],
            references=["T3", "T4"],
            report="campaign",
            placebo_timing={"T3": ff.toggle_frame()},
        )
        (out,) = _method(tmp_path, options=TOGGLE_OPTIONS).estimate(_toggle_facts(), [analysis])
        rows = out.per_turbine.set_index("turbine")
        assert rows["placebo"].tolist() == [False, True]
        assert rows.loc["T3", "campaign_uplift"] == 0.001
        assert out.campaign_uplift == 0.03
