"""Offline tests for the V0BinnedMethod adapter.

The full wind_up pipeline (``AssessmentInputs.from_cfg`` / ``run_wind_up_analysis`` /
``combine_results``) is stubbed; these tests cover the seam: the per-campaign config the
adapter builds via ``WindUpConfig.from_yaml``, the P50 extraction, and the prepost-only guard.
A real end-to-end run lives in the ``slow`` integration test.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from benchmarking.baselines import v0_binned
from benchmarking.baselines.hot_context import HotV0Context
from benchmarking.baselines.v0_binned import V0BinnedMethod, _extract_p50, _subset_turbines
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule, treated_mask
from benchmarking.synthetic.sources.hill_of_towie import hot_turbines, long_to_wind_up_format
from wind_up.analysis import AnalysisSpec
from wind_up.facts import Upgrade, WindFarmFacts
from wind_up.labels import Label
from wind_up.method import MethodConfigurationError
from wind_up.subject import SubjectMeasurement, SubjectRecords, measurement_of
from wind_up.toggle import build_toggle_df

# The harness hands v0 source-native SCADA, so the fixtures carry the source-native tag names and
# the full field set the v0 on-ramp (``long_to_wind_up_format``) needs (pitch blades, yaw, the
# time-ready signal) to derive ``PitchAngleMean`` / ``ShutdownDuration``.
_SOURCE_FIELDS = (
    HOT_COLUMNS.active_power,
    HOT_COLUMNS.wind_speed,
    HOT_COLUMNS.wind_speed_sd,
    HOT_COLUMNS.gen_rpm,
    "wtc_ActPower_stddev",
    "wtc_NacelPos_mean",
    "wtc_PitcPosA_mean",
    "wtc_PitcPosB_mean",
    "wtc_PitcPosC_mean",
)


def _long_scada(turbines: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """A source-native long SCADA frame complete enough for the v0 on-ramp to convert."""
    frames = [
        pd.DataFrame(
            {HOT_COLUMNS.turbine: t, **dict.fromkeys(_SOURCE_FIELDS, 1.0), "wtc_ScReToOp_timeon": 600.0},
            index=index,
        )
        for t in turbines
    ]
    return pd.concat(frames)


def _make_scada(turbines: list[str], *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Minimal-span scada (two timestamps) whose meaningful content is the turbines and span."""
    index = pd.DatetimeIndex([start, end], name="TimeStamp_StartFormat")
    return _long_scada(turbines, index)


def _context() -> HotV0Context:
    return HotV0Context(metadata_df=pd.DataFrame({"Name": ["T01"]}), reanalysis_datasets=[])


UPGRADE = pd.Timestamp("2018-01-01", tz="UTC")


def _records(scada: pd.DataFrame, test_wtg: str, timing: pd.Timestamp | ToggleSchedule) -> SubjectRecords:
    return SubjectRecords.from_frame(scada, subject=test_wtg, timing=timing, turbine_col=HOT_COLUMNS.turbine)


def _method_input(turbines: list[str], test_wtg: str) -> SubjectRecords:
    scada = _make_scada(turbines, start=UPGRADE - pd.DateOffset(years=1), end=UPGRADE + pd.DateOffset(months=6))
    return _records(scada, test_wtg, UPGRADE)


def _measure(method: V0BinnedMethod, mi: SubjectRecords) -> SubjectMeasurement:
    return method.estimate_multi(mi, test_wtgs=[mi.subject])[mi.subject]


class TestSubsetTurbines:
    def test_returns_sorted_unique_names(self) -> None:
        scada = _make_scada(["T04", "T01", "T03"], start=UPGRADE, end=UPGRADE + pd.DateOffset(months=1))
        assert _subset_turbines(scada, HOT_COLUMNS.turbine) == ["T01", "T03", "T04"]


class TestBuildConfig:
    def test_sets_test_and_ref_wtgs(self, tmp_path) -> None:  # noqa: ANN001
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path)
        cfg = method._build_config(_method_input(["T01", "T02", "T03", "T04"], "T02"))  # noqa: SLF001
        assert [w.name for w in cfg.test_wtgs] == ["T02"]
        assert sorted(w.name for w in cfg.ref_wtgs) == ["T01", "T03", "T04"]

    def test_sets_prepost_dates_and_knobs(self, tmp_path) -> None:  # noqa: ANN001
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path)
        mi = _method_input(["T01", "T02", "T03", "T04"], "T01")
        cfg = method._build_config(mi)  # noqa: SLF001
        assert cfg.prepost is not None
        assert pd.Timestamp(cfg.prepost.post_first_dt_utc_start) == UPGRADE
        assert pd.Timestamp(cfg.prepost.post_last_dt_utc_start) == pd.Timestamp(mi.records.index.max())
        assert pd.Timestamp(cfg.upgrade_first_dt_utc_start) == UPGRADE
        assert cfg.use_lt_distribution is False
        assert cfg.optimize_northing_corrections is False
        assert len(cfg.northing_corrections_utc) > 0

    def test_filters_asset_to_subset(self, tmp_path) -> None:  # noqa: ANN001
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path)
        cfg = method._build_config(_method_input(["T01", "T02", "T03", "T04"], "T01"))  # noqa: SLF001
        assert sorted(w.name for w in cfg.asset.wtgs) == ["T01", "T02", "T03", "T04"]

    def test_raises_when_no_reference_turbines(self, tmp_path) -> None:  # noqa: ANN001
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path)
        with pytest.raises(ValueError, match="no reference turbines"):
            method._build_config(_method_input(["T01"], "T01"))  # noqa: SLF001

    def test_filter_all_defaults_false(self, tmp_path) -> None:  # noqa: ANN001
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path)
        cfg = method._build_config(_method_input(["T01", "T02", "T03"], "T01"))  # noqa: SLF001
        assert cfg.filter_all_test_wtgs_together is False

    def test_multi_test_wtgs_excludes_all_participants_from_refs_and_sets_flag(self, tmp_path) -> None:  # noqa: ANN001
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path, filter_all_test_wtgs_together=True)
        cfg = method._build_config(  # noqa: SLF001
            _method_input(["T01", "T02", "T04", "T07"], "T01"), test_wtgs=["T01", "T04"]
        )
        assert sorted(w.name for w in cfg.test_wtgs) == ["T01", "T04"]
        assert sorted(w.name for w in cfg.ref_wtgs) == ["T02", "T07"]  # both participants excluded
        assert cfg.filter_all_test_wtgs_together is True

    def test_require_ref_wake_free_defaults_false_and_configurable(self, tmp_path) -> None:  # noqa: ANN001
        mi = _method_input(["T01", "T02", "T03"], "T01")
        assert V0BinnedMethod(_context(), scratch_dir=tmp_path)._build_config(mi).require_ref_wake_free is False  # noqa: SLF001
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path, require_ref_wake_free=True)
        assert method._build_config(mi).require_ref_wake_free is True  # noqa: SLF001


def _dense_scada(turbines: list[str], *, start: pd.Timestamp, end: pd.Timestamp, freq: str = "1D") -> pd.DataFrame:
    """Source-native long scada on a regular grid (enough rows for a toggle split)."""
    index = pd.date_range(start, end, freq=freq, tz="UTC", name="TimeStamp_StartFormat")
    return _long_scada(turbines, index)


class TestBuildToggleDf:
    def test_before_start_both_false_after_exactly_one_true(self) -> None:
        idx = pd.date_range(UPGRADE - pd.Timedelta(minutes=30), periods=9, freq="10min", tz="UTC")
        scada = _dense_scada(["a", "b"], start=idx[0], end=idx[-1], freq="10min")
        schedule = ToggleSchedule(period=pd.Timedelta(minutes=20), start=UPGRADE)
        df = build_toggle_df(scada.index, schedule)

        before = df.index < UPGRADE
        assert not df.loc[before, "toggle_on"].any()
        assert not df.loc[before, "toggle_off"].any()
        after = df.index >= UPGRADE
        assert (df.loc[after, "toggle_on"] ^ df.loc[after, "toggle_off"]).all()
        assert (df["toggle_on"].to_numpy() == treated_mask(df.index, schedule)).all()


class TestBuildConfigToggle:
    def test_sets_toggle_block_not_prepost(self, tmp_path) -> None:  # noqa: ANN001
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path)
        scada = _dense_scada(
            ["T01", "T02", "T03", "T04"], start=UPGRADE - pd.DateOffset(years=1), end=UPGRADE + pd.DateOffset(months=6)
        )
        schedule = ToggleSchedule(period=pd.Timedelta(days=2), start=UPGRADE)
        cfg = method._build_config(_records(scada, "T01", schedule))  # noqa: SLF001
        assert cfg.toggle is not None
        assert cfg.prepost is None
        assert cfg.toggle.detrend_data_selection == "use_toggle_off_data"
        assert cfg.toggle.toggle_change_settling_filter_seconds == 0
        assert cfg.toggle.pairing_filter_method == "none"  # default: no pairing filter
        assert pd.Timestamp(cfg.upgrade_first_dt_utc_start) == UPGRADE

    def test_toggle_pairing_and_settling_configurable(self, tmp_path) -> None:  # noqa: ANN001
        method = V0BinnedMethod(
            _context(),
            scratch_dir=tmp_path,
            pairing_filter_method="any_within_timedelta",
            pairing_filter_timedelta_seconds=3000,
            toggle_change_settling_filter_seconds=300,
        )
        scada = _dense_scada(
            ["T01", "T02", "T03", "T04"], start=UPGRADE - pd.DateOffset(years=1), end=UPGRADE + pd.DateOffset(months=6)
        )
        schedule = ToggleSchedule(period=pd.Timedelta(days=2), start=UPGRADE)
        cfg = method._build_config(_records(scada, "T01", schedule))  # noqa: SLF001
        assert cfg.toggle.pairing_filter_method == "any_within_timedelta"
        assert cfg.toggle.pairing_filter_timedelta_seconds == 3000
        assert cfg.toggle.toggle_change_settling_filter_seconds == 300


class TestEstimateToggle:
    def test_wires_toggle_df_and_returns_p50(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        captured = _stub_pipeline(monkeypatch)
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path)
        scada = _dense_scada(
            ["T01", "T02", "T03"], start=UPGRADE - pd.DateOffset(months=6), end=UPGRADE + pd.DateOffset(months=6)
        )
        schedule = ToggleSchedule(period=pd.Timedelta(days=2), start=UPGRADE)
        out = _measure(method, _records(scada, "T01", schedule))

        assert out.p50_overall == pytest.approx(0.042)
        toggle_df = captured["from_cfg_kwargs"]["toggle_df"]
        assert sorted(toggle_df.columns) == ["toggle_off", "toggle_on"]
        assert toggle_df["toggle_on"].any()


class TestExtractP50:
    def test_picks_non_ref_test_row(self) -> None:
        tdf = pd.DataFrame(
            {
                "test_wtg": ["T01", "T02", "T03"],
                "p50_uplift": [0.05, 0.001, -0.002],
                "is_ref": [False, True, True],
            }
        )
        assert _extract_p50(tdf, "T01") == pytest.approx(0.05)

    def test_raises_when_test_row_absent(self) -> None:
        tdf = pd.DataFrame({"test_wtg": ["T02"], "p50_uplift": [0.0], "is_ref": [True]})
        with pytest.raises(ValueError, match="T01"):
            _extract_p50(tdf, "T01")


def _stub_pipeline(monkeypatch) -> dict:  # noqa: ANN001
    """Patch AssessmentInputs/run/combine to no-ops; return the captured-call dict."""
    captured: dict[str, object] = {}

    class FakeInputs:
        @staticmethod
        def from_cfg(**kwargs: object) -> str:
            captured["from_cfg_kwargs"] = kwargs
            return "inputs-sentinel"

    def fake_run(inputs: object) -> str:
        captured["run_inputs"] = inputs
        return "trdf-sentinel"

    def fake_combine(trdf: object, **kwargs: object) -> pd.DataFrame:
        captured["combine_trdf"] = trdf
        captured["combine_kwargs"] = kwargs
        return pd.DataFrame({"test_wtg": ["T01"], "p50_uplift": [0.042], "is_ref": [False]})

    monkeypatch.setattr(v0_binned, "AssessmentInputs", FakeInputs)
    monkeypatch.setattr(v0_binned, "run_wind_up_analysis", fake_run)
    monkeypatch.setattr(v0_binned, "combine_results", fake_combine)
    return captured


class TestEstimateWiring:
    def test_wires_pipeline_and_returns_p50(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        captured = _stub_pipeline(monkeypatch)

        method = V0BinnedMethod(_context(), scratch_dir=tmp_path)
        out = _measure(method, _method_input(["T01", "T02", "T03", "T04"], "T01"))

        assert isinstance(out, SubjectMeasurement)
        assert out.p50_overall == pytest.approx(0.042)
        assert out.p50_by_condition is None
        assert captured["run_inputs"] == "inputs-sentinel"
        assert captured["combine_trdf"] == "trdf-sentinel"
        assert captured["combine_kwargs"]["auto_choose_refs"] is False
        # the source-native scada is converted to wind-up format on the way into the pipeline,
        # and the metadata/reanalysis from the context are handed through unchanged
        kwargs = captured["from_cfg_kwargs"]
        expected_scada = long_to_wind_up_format(_method_input(["T01", "T02", "T03", "T04"], "T01").records)
        assert kwargs["scada_df"].equals(expected_scada)
        assert kwargs["reanalysis_datasets"] == []

    def test_save_plots_defaults_off(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        captured = _stub_pipeline(monkeypatch)
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path)
        _measure(method, _method_input(["T01", "T02", "T03", "T04"], "T01"))
        assert captured["from_cfg_kwargs"]["plot_cfg"].save_plots is False

    def test_save_plots_on_writes_under_out_dir(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        captured = _stub_pipeline(monkeypatch)
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path, save_plots=True)
        _measure(method, _method_input(["T01", "T02", "T03", "T04"], "T01"))
        plot_cfg = captured["from_cfg_kwargs"]["plot_cfg"]
        assert plot_cfg.save_plots is True
        assert plot_cfg.plots_dir.parent == tmp_path / "v0_T01_20180101_20180701"
        assert plot_cfg.plots_dir.name == "plots"


class TestEstimateMulti:
    def test_single_run_returns_p50_per_test_turbine(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        captured: dict[str, object] = {}
        calls = {"from_cfg": 0}

        class FakeInputs:
            @staticmethod
            def from_cfg(**kwargs: object) -> str:
                calls["from_cfg"] += 1
                captured["from_cfg_kwargs"] = kwargs
                return "inputs-sentinel"

        def fake_combine(trdf: object, **kwargs: object) -> pd.DataFrame:  # noqa: ARG001
            return pd.DataFrame(
                {
                    "test_wtg": ["T01", "T04", "T02"],
                    "p50_uplift": [-0.012, 0.026, 0.0],
                    "is_ref": [False, False, True],
                }
            )

        monkeypatch.setattr(v0_binned, "AssessmentInputs", FakeInputs)
        monkeypatch.setattr(v0_binned, "run_wind_up_analysis", lambda inputs: "trdf")  # noqa: ARG005
        monkeypatch.setattr(v0_binned, "combine_results", fake_combine)

        method = V0BinnedMethod(_context(), scratch_dir=tmp_path, filter_all_test_wtgs_together=True)
        outs = method.estimate_multi(_method_input(["T01", "T02", "T04", "T07"], "T01"), test_wtgs=["T01", "T04"])

        assert set(outs) == {"T01", "T04"}
        assert outs["T01"].p50_overall == pytest.approx(-0.012)
        assert outs["T04"].p50_overall == pytest.approx(0.026)
        assert calls["from_cfg"] == 1  # both turbines analysed in a single wind_up run


class TestReferencesFromTheRecords:
    def test_config_references_come_from_the_campaign(self, tmp_path) -> None:  # noqa: ANN001
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path)
        scada = _dense_scada(
            ["T01", "T02", "T03", "T04"], start=UPGRADE - pd.DateOffset(years=1), end=UPGRADE + pd.DateOffset(months=6)
        )
        campaign = dataclasses.replace(
            _records(scada, "T01", UPGRADE), candidate_references=["T02", "T03"], wake_contributors=["T04"]
        )
        cfg = method._build_config(campaign)  # noqa: SLF001
        assert sorted(w.name for w in cfg.ref_wtgs) == ["T02", "T03"]


class TestAnsweringAnalyses:
    @staticmethod
    def _facts() -> WindFarmFacts:
        scada = _dense_scada(
            ["T01", "T02", "T03", "T04"], start=UPGRADE - pd.DateOffset(months=6), end=UPGRADE + pd.DateOffset(months=6)
        )
        return WindFarmFacts(
            records=scada,
            columns=HOT_COLUMNS,
            timebase=pd.Timedelta(days=1),
            turbines=hot_turbines(["T01", "T02", "T03", "T04"]),
            upgrades=[Upgrade(name="u", turbines=["T01", "T04"], timing=UPGRADE)],
        )

    def test_one_run_answers_every_subject(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        captured = _stub_pipeline(monkeypatch)
        (out,) = V0BinnedMethod(_context(), scratch_dir=tmp_path).estimate(
            self._facts(), [AnalysisSpec(name="a", upgrade="u", subjects=["T01"], report="campaign")]
        )
        assert out.campaign_uplift == pytest.approx(0.042)
        assert captured["combine_kwargs"]["auto_choose_refs"] is False

    def test_a_run_that_fails_leaves_every_subject_unmeasured(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        _stub_pipeline(monkeypatch)
        analysis = AnalysisSpec(name="a", upgrade="u", references=[], report="campaign")
        (out,) = V0BinnedMethod(_context(), scratch_dir=tmp_path).estimate(self._facts(), [analysis])
        assert measurement_of(out, "T01") == measurement_of(out, "T04")
        assert "no reference turbines" in str(measurement_of(out, "T01"))

    def test_labels_are_refused(self, tmp_path) -> None:  # noqa: ANN001
        facts = self._facts()
        facts = dataclasses.replace(
            facts, records=facts.records.assign(flag=False), labels=[Label(name="x", column="flag")]
        )
        analysis = AnalysisSpec(name="a", upgrade="u", include_labels=["x"])
        with pytest.raises(MethodConfigurationError, match="labels"):
            V0BinnedMethod(_context(), scratch_dir=tmp_path).estimate(facts, [analysis])
