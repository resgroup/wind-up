"""Stub methods for exercising the scoring machinery without a real estimator.

The oracle computes the true uplift from its *own* method-input window (windowed synthetic vs
original), so if the harness ever scored truth over different records than it handed the
method, the oracle's error would stop being zero. Biased/Noisy wrap the oracle to drive the
bias/precision metrics; Recording captures inputs for the fairness test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.synthetic import treated_mask
from wind_up.constants import DataColumns

if TYPE_CHECKING:
    import pandas as pd


def oracle_overall_uplift(mi: MethodInput, original_df: pd.DataFrame) -> float:
    """Energy-ratio uplift over the treated test-turbine rows in ``mi``'s window."""
    syn = mi.scada_df
    test_rows = syn[syn[DataColumns.turbine_name] == mi.test_wtg]
    treated = treated_mask(test_rows.index, mi.upgrade_timing)
    treated_rows = test_rows[treated]

    syn_power = treated_rows[DataColumns.active_power_mean].to_numpy(dtype=float)
    orig_test = original_df[original_df[DataColumns.turbine_name] == mi.test_wtg]
    orig_power = orig_test.loc[treated_rows.index, DataColumns.active_power_mean].to_numpy(dtype=float)

    finite = np.isfinite(syn_power) & np.isfinite(orig_power)
    denom = orig_power[finite].sum()
    return syn_power[finite].sum() / denom - 1.0 if denom else float("nan")


class OracleMethod:
    """Returns the true uplift; signed error should be ~0."""

    def __init__(self, original_df: pd.DataFrame, name: str = "oracle") -> None:
        self._original = original_df
        self.name = name

    def estimate(self, mi: MethodInput) -> MethodOutput:
        return MethodOutput(p50_overall=oracle_overall_uplift(mi, self._original))


class BiasedMethod:
    """Returns the true uplift plus a fixed offset; signed error should equal the offset."""

    def __init__(self, original_df: pd.DataFrame, offset: float, name: str = "biased") -> None:
        self._original = original_df
        self._offset = offset
        self.name = name

    def estimate(self, mi: MethodInput) -> MethodOutput:
        return MethodOutput(p50_overall=oracle_overall_uplift(mi, self._original) + self._offset)


class NoisyMethod:
    """Returns the true uplift plus seeded Gaussian noise; error spread should track sigma."""

    def __init__(self, original_df: pd.DataFrame, sigma: float, seed: int = 0, name: str = "noisy") -> None:
        self._original = original_df
        self._sigma = sigma
        self._rng = np.random.default_rng(seed)
        self.name = name

    def estimate(self, mi: MethodInput) -> MethodOutput:
        noise = float(self._rng.normal(0.0, self._sigma))
        return MethodOutput(p50_overall=oracle_overall_uplift(mi, self._original) + noise)


class RecordingMethod:
    """Captures every MethodInput it is handed, for the fairness test."""

    def __init__(self, name: str = "recording") -> None:
        self.name = name
        self.seen: list[MethodInput] = []

    def estimate(self, mi: MethodInput) -> MethodOutput:
        captured = MethodInput(scada_df=mi.scada_df.copy(), test_wtg=mi.test_wtg, upgrade_timing=mi.upgrade_timing)
        self.seen.append(captured)
        return MethodOutput(p50_overall=0.0)
