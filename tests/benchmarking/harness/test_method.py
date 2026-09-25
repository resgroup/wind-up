"""Tests for the thin method seam."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from benchmarking.harness.context import CampaignContext
from benchmarking.harness.method import Method, MethodInput, MethodOutput


def _input() -> MethodInput:
    scada_df = pd.DataFrame({"x": [1.0]}, index=pd.date_range("2020-01-01", periods=1, tz="UTC"))
    return MethodInput(scada_df=scada_df, test_wtg="T1", upgrade_timing=pd.Timestamp("2020-01-01", tz="UTC"))


def test_method_input_carries_data_test_wtg_and_timing() -> None:
    mi = _input()
    assert mi.test_wtg == "T1"
    assert mi.upgrade_timing == pd.Timestamp("2020-01-01", tz="UTC")
    assert list(mi.scada_df.columns) == ["x"]


def test_method_output_defaults_by_condition_to_none() -> None:
    out = MethodOutput(p50_overall=0.03)
    assert out.p50_overall == 0.03
    assert out.p50_by_condition is None


def test_a_conforming_class_satisfies_the_method_protocol() -> None:
    class FixedMethod:
        name = "fixed"

        def estimate(self, mi: MethodInput) -> MethodOutput:  # noqa: ARG002
            return MethodOutput(p50_overall=0.0)

    method = FixedMethod()
    assert isinstance(method, Method)
    assert method.estimate(_input()).p50_overall == 0.0


def test_a_non_conforming_object_is_not_a_method() -> None:
    assert not isinstance(object(), Method)


def _long_scada() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=3, freq="10min", tz="UTC")
    frames = [pd.DataFrame({"TurbineName": t, "x": 1.0}, index=index) for t in ("T1", "T2", "T3")]
    return pd.concat(frames).sort_index()


class TestContext:
    def test_defaults_to_the_frames_own_implicit_contract(self) -> None:
        mi = MethodInput(scada_df=_long_scada(), test_wtg="T1", upgrade_timing=pd.Timestamp("2020-01-01", tz="UTC"))
        assert mi.context.candidate_references == ["T2", "T3"]
        assert mi.context.valid_for_uplift.to_numpy().all()

    def test_a_supplied_context_is_the_source_of_timing_and_turbine_col(self) -> None:
        timing = pd.Timestamp("2021-05-05", tz="UTC")
        context = CampaignContext.from_frame(_long_scada(), test_wtg="T1", timing=timing, turbine_col="TurbineName")
        mi = MethodInput(scada_df=_long_scada(), test_wtg="T1", campaign_context=context)
        assert mi.context is context
        assert mi.upgrade_timing == timing
        assert mi.turbine_col == "TurbineName"

    def test_a_frame_without_a_turbine_column_still_constructs(self) -> None:
        # The default context is built lazily, so a degenerate frame is only a problem if used.
        MethodInput(
            scada_df=pd.DataFrame({"x": [1.0]}, index=pd.date_range("2020-01-01", periods=1, tz="UTC")),
            test_wtg="T1",
            upgrade_timing=pd.Timestamp("2020-01-01", tz="UTC"),
        )

    def test_narrowing_the_frame_keeps_a_supplied_context(self) -> None:
        # restrict_to_campaign narrows the frame with dataclasses.replace; the campaign's declared
        # facts must survive that, so valid_for_uplift still covers what the method then asks for.
        scada = _long_scada()
        context = CampaignContext.from_frame(
            scada, test_wtg="T1", timing=pd.Timestamp("2020-01-01", tz="UTC"), turbine_col="TurbineName"
        )
        mi = MethodInput(scada_df=scada, test_wtg="T1", campaign_context=context)
        narrowed = replace(mi, scada_df=scada.loc[scada.index >= scada.index[-1]])
        assert narrowed.context is context
        assert narrowed.context.valid_over(pd.DatetimeIndex(narrowed.scada_df.index.unique())).to_numpy().all()

    def test_rejects_an_input_with_neither_a_context_nor_a_timing(self) -> None:
        with pytest.raises(ValueError, match="upgrade_timing"):
            MethodInput(scada_df=_long_scada(), test_wtg="T1")

    def test_rejects_a_context_built_for_a_different_turbine(self) -> None:
        # Otherwise a method estimates one turbine while reading another's references and validity.
        context = CampaignContext.from_frame(
            _long_scada(), test_wtg="T2", timing=pd.Timestamp("2020-01-01", tz="UTC"), turbine_col="TurbineName"
        )
        with pytest.raises(ValueError, match=r"'T2'.*'T1'"):
            MethodInput(scada_df=_long_scada(), test_wtg="T1", campaign_context=context)
