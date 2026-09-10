"""Stub methods for exercising the scoring machinery without a real estimator.

The oracle computes the true uplift from its *own* method-input window (windowed synthetic vs
original), so if the harness ever scored truth over different records than it handed the
method, the oracle's error would stop being zero. Biased/Noisy wrap the oracle to drive the
bias/precision metrics; Recording captures inputs for the fairness test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarking.harness.conditions import CONDITION_BINS, condition_bins, energy_ratio_by_bin
from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.synthetic import HOT_COLUMNS, treated_mask
from benchmarking.synthetic.sources.hill_of_towie import HOT_RATED_POWER_KW


def oracle_overall_uplift(mi: MethodInput, original_df: pd.DataFrame) -> float:
    """Energy-ratio uplift over the treated test-turbine rows in ``mi``'s window."""
    syn = mi.scada_df
    test_rows = syn[syn[mi.turbine_col] == mi.test_wtg]
    treated = treated_mask(test_rows.index, mi.upgrade_timing)
    treated_rows = test_rows[treated]

    syn_power = treated_rows[HOT_COLUMNS.active_power].to_numpy(dtype=float)
    orig_test = original_df[original_df[mi.turbine_col] == mi.test_wtg]
    orig_power = orig_test.loc[treated_rows.index, HOT_COLUMNS.active_power].to_numpy(dtype=float)

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
        captured = MethodInput(
            scada_df=mi.scada_df.copy(),
            test_wtg=mi.test_wtg,
            upgrade_timing=mi.upgrade_timing,
            turbine_col=mi.turbine_col,
            campaign_context=mi.campaign_context,
        )
        self.seen.append(captured)
        return MethodOutput(p50_overall=0.0)


def conditional_oracle_by_condition(mi: MethodInput, original_df: pd.DataFrame) -> pd.DataFrame:
    """Per-bin oracle uplift over treated rows, binned on the test turbine's measured ws/ti."""
    syn = mi.scada_df
    test_rows = syn[syn[mi.turbine_col] == mi.test_wtg]
    treated = treated_mask(test_rows.index, mi.upgrade_timing)
    treated_rows = test_rows[treated]
    orig_test = original_df[original_df[mi.turbine_col] == mi.test_wtg]
    actual = treated_rows[HOT_COLUMNS.active_power].to_numpy(dtype=float)
    counterfactual = orig_test.loc[treated_rows.index, HOT_COLUMNS.active_power].to_numpy(dtype=float)
    ws = treated_rows[HOT_COLUMNS.wind_speed].to_numpy(dtype=float)
    sd = treated_rows[HOT_COLUMNS.wind_speed_sd].to_numpy(dtype=float)
    ti = np.divide(sd, ws, out=np.full_like(sd, np.nan), where=ws != 0)
    # power bins on the untreated operating point; for the oracle the counterfactual IS the original
    # power, so binning on it matches the truth's original-power binning (near-zero error end-to-end).
    axes = (
        ("ws", ws, CONDITION_BINS["ws"]),
        ("ti", ti, CONDITION_BINS["ti"]),
        ("power", counterfactual, condition_bins("power", rated_power_kw=HOT_RATED_POWER_KW)),
    )
    frames = []
    for name, values, bins in axes:
        table = energy_ratio_by_bin(values, actual, counterfactual, bins=bins)
        table.insert(0, "condition", name)
        frames.append(table[["condition", "condition_bin", "p50_uplift"]])
    return pd.concat(frames, ignore_index=True)


class ConditionalOracleMethod:
    """Oracle that also emits per-bin oracle uplift; conditional signed error should be ~0."""

    def __init__(self, original_df: pd.DataFrame, name: str = "cond_oracle") -> None:
        self._original = original_df
        self.name = name

    def estimate(self, mi: MethodInput) -> MethodOutput:
        return MethodOutput(
            p50_overall=oracle_overall_uplift(mi, self._original),
            p50_by_condition=conditional_oracle_by_condition(mi, self._original),
        )


class UncertainMethod:
    """Reports a fixed sigma and diagnostics, for exercising the seam's uncertainty passthrough.

    Deliberately dumb: the numbers mean nothing, they only have to arrive intact and in the right
    place. ``diagnostic_columns`` lets a test drive the clash guard in ``_merge_diagnostics``.
    """

    def __init__(
        self,
        original_df: pd.DataFrame,
        sigma: float = 0.01,
        name: str = "uncertain",
        diagnostic_columns: dict[str, float] | None = None,
    ) -> None:
        self._original = original_df
        self._sigma = sigma
        self._diagnostics = {"n_blocks": 7.0} if diagnostic_columns is None else diagnostic_columns
        self.name = name

    def estimate(self, mi: MethodInput) -> MethodOutput:
        by_condition = conditional_oracle_by_condition(mi, self._original)
        by_condition["sigma_uplift"] = self._sigma * 2.0
        rows = [{"condition": "overall", "condition_bin": "overall", **self._diagnostics}]
        rows += [
            {"condition": c, "condition_bin": b, **self._diagnostics}
            for c, b in zip(by_condition["condition"], by_condition["condition_bin"], strict=True)
        ]
        return MethodOutput(
            p50_overall=oracle_overall_uplift(mi, self._original),
            p50_by_condition=by_condition,
            sigma_overall=self._sigma,
            uncertainty_diagnostics=pd.DataFrame(rows),
        )
