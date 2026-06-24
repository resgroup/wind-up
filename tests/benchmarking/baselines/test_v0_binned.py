"""Offline tests for the V0BinnedMethod adapter.

The full wind_up pipeline (``AssessmentInputs.from_cfg`` / ``run_wind_up_analysis`` /
``combine_results``) is stubbed; these tests cover the seam: the per-campaign config the
adapter builds via ``WindUpConfig.from_yaml``, the P50 extraction, and the prepost-only guard.
A real end-to-end run lives in the ``slow`` integration test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from benchmarking.baselines import v0_binned
from benchmarking.baselines.hot_context import HotV0Context
from benchmarking.baselines.v0_binned import V0BinnedMethod, _extract_p50, _subset_turbines
from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.synthetic import ToggleSchedule, treated_mask
from wind_up.constants import DataColumns


def _make_scada(turbines: list[str], *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Minimal scada whose only meaningful content is the turbine names and the index span."""
    index = pd.DatetimeIndex([start, end], name="TimeStamp_StartFormat")
    frames = [
        pd.DataFrame({DataColumns.turbine_name: t, DataColumns.active_power_mean: [1.0, 1.0]}, index=index)
        for t in turbines
    ]
    return pd.concat(frames)


def _context() -> HotV0Context:
    return HotV0Context(metadata_df=pd.DataFrame({"Name": ["T01"]}), reanalysis_datasets=[])


UPGRADE = pd.Timestamp("2018-01-01", tz="UTC")


def _method_input(turbines: list[str], test_wtg: str) -> MethodInput:
    scada = _make_scada(turbines, start=UPGRADE - pd.DateOffset(years=1), end=UPGRADE + pd.DateOffset(months=6))
    return MethodInput(scada_df=scada, test_wtg=test_wtg, upgrade_timing=UPGRADE)


class TestSubsetTurbines:
    def test_returns_sorted_unique_names(self) -> None:
        scada = _make_scada(["T04", "T01", "T03"], start=UPGRADE, end=UPGRADE + pd.DateOffset(months=1))
        assert _subset_turbines(scada) == ["T01", "T03", "T04"]


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
        assert pd.Timestamp(cfg.prepost.post_last_dt_utc_start) == pd.Timestamp(mi.scada_df.index.max())
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


def _dense_scada(turbines: list[str], *, start: pd.Timestamp, end: pd.Timestamp, freq: str = "1D") -> pd.DataFrame:
    """Long-format scada on a regular grid (enough rows for a toggle split)."""
    index = pd.date_range(start, end, freq=freq, tz="UTC", name="TimeStamp_StartFormat")
    frames = [
        pd.DataFrame({DataColumns.turbine_name: t, DataColumns.active_power_mean: 1.0}, index=index) for t in turbines
    ]
    return pd.concat(frames)


class TestBuildToggleDf:
    def test_before_start_both_false_after_exactly_one_true(self) -> None:
        idx = pd.date_range(UPGRADE - pd.Timedelta(minutes=30), periods=9, freq="10min", tz="UTC")
        scada = _dense_scada(["a", "b"], start=idx[0], end=idx[-1], freq="10min")
        schedule = ToggleSchedule(period=pd.Timedelta(minutes=20), start=UPGRADE)
        df = v0_binned._build_toggle_df(scada, schedule)  # noqa: SLF001

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
        cfg = method._build_config(MethodInput(scada_df=scada, test_wtg="T01", upgrade_timing=schedule))  # noqa: SLF001
        assert cfg.toggle is not None
        assert cfg.prepost is None
        assert cfg.toggle.detrend_data_selection == "use_toggle_off_data"
        assert cfg.toggle.toggle_change_settling_filter_seconds == 0
        assert pd.Timestamp(cfg.upgrade_first_dt_utc_start) == UPGRADE


class TestEstimateToggle:
    def test_wires_toggle_df_and_returns_p50(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        captured = _stub_pipeline(monkeypatch)
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path)
        scada = _dense_scada(
            ["T01", "T02", "T03"], start=UPGRADE - pd.DateOffset(months=6), end=UPGRADE + pd.DateOffset(months=6)
        )
        schedule = ToggleSchedule(period=pd.Timedelta(days=2), start=UPGRADE)
        out = method.estimate(MethodInput(scada_df=scada, test_wtg="T01", upgrade_timing=schedule))

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
        out = method.estimate(_method_input(["T01", "T02", "T03", "T04"], "T01"))

        assert isinstance(out, MethodOutput)
        assert out.p50_overall == pytest.approx(0.042)
        assert out.p50_by_condition is None
        assert captured["run_inputs"] == "inputs-sentinel"
        assert captured["combine_trdf"] == "trdf-sentinel"
        assert captured["combine_kwargs"]["auto_choose_refs"] is False
        # the same scada/metadata/reanalysis from the context are handed to the pipeline
        kwargs = captured["from_cfg_kwargs"]
        assert kwargs["scada_df"].equals(_method_input(["T01", "T02", "T03", "T04"], "T01").scada_df)
        assert kwargs["reanalysis_datasets"] == []

    def test_save_plots_defaults_off(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        captured = _stub_pipeline(monkeypatch)
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path)
        method.estimate(_method_input(["T01", "T02", "T03", "T04"], "T01"))
        assert captured["from_cfg_kwargs"]["plot_cfg"].save_plots is False

    def test_save_plots_on_writes_under_out_dir(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        captured = _stub_pipeline(monkeypatch)
        method = V0BinnedMethod(_context(), scratch_dir=tmp_path, save_plots=True)
        method.estimate(_method_input(["T01", "T02", "T03", "T04"], "T01"))
        plot_cfg = captured["from_cfg_kwargs"]["plot_cfg"]
        assert plot_cfg.save_plots is True
        assert plot_cfg.plots_dir.parent == tmp_path / "v0_T01_20180101_20180701"
        assert plot_cfg.plots_dir.name == "plots"
