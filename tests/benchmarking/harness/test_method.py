"""Tests for the thin method seam."""

from __future__ import annotations

import pandas as pd

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
