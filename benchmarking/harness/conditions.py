"""Shared condition bins and the binned energy-ratio reducer.

One source of truth for the wind-speed / TI bin edges, imported by both the method (to bin its
counterfactual ledger) and the harness truth path, so the two bin identically. The reducer is the
same energy ratio as the overall number, taken within each bin: ``Σactual / Σcounterfactual - 1``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

WS_BINS: list[float] = [float(x) for x in np.arange(0.0, 28.0, 2.0)]  # 0,2,…,26
TI_BINS: list[float] = [round(float(x), 2) for x in np.arange(0.0, 0.55, 0.05)]  # 0,0.05,…,0.50
CONDITIONS: tuple[str, ...] = ("ws", "ti", "power")
CONDITION_BINS: dict[str, list[float]] = {"ws": WS_BINS, "ti": TI_BINS}

# ``power`` edges are a fraction of rated (so they scale with the turbine), unlike the fixed ws/ti
# edges, hence they live in ``condition_bins()`` not ``CONDITION_BINS``. The outer edges sit just
# below 0 and just above rated so the six bins center on 0,0.2,…,1.0 of rated and ``pd.cut`` keeps
# untreated-power values that fall slightly negative (cut-in noise) or slightly over rated
# (counterfactual prediction noise) rather than dropping them to NaN.
POWER_FRACTION_EDGES: list[float] = [-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1]


def condition_bins(name: str, *, rated_power_kw: float | None = None) -> list[float]:
    """Bin edges for a condition axis. ws/ti are fixed; power scales with ``rated_power_kw``.

    ws/ti accept ``rated_power_kw`` but ignore it (they are treatment-invariant fixed edges). power
    requires it — its edges are ``POWER_FRACTION_EDGES`` times the rating (in kW). Raises for an
    unknown condition so a mistyped axis fails loudly.
    """
    if name in CONDITION_BINS:
        return CONDITION_BINS[name]
    if name == "power":
        if rated_power_kw is None:
            msg = "rated_power_kw is required for the 'power' condition (edges scale with rating)"
            raise ValueError(msg)
        return [round(f * float(rated_power_kw), 6) for f in POWER_FRACTION_EDGES]
    msg = f"unknown condition {name!r}; expected one of {CONDITIONS}"
    raise ValueError(msg)


def energy_ratio_by_bin(
    condition_values: npt.ArrayLike,
    actual: npt.ArrayLike,
    counterfactual: npt.ArrayLike,
    *,
    bins: list[float],
) -> pd.DataFrame:
    """Energy-ratio uplift within bins of a condition signal (NaN-safe; every bin represented)."""
    cond = np.asarray(condition_values, dtype=float)
    act = np.asarray(actual, dtype=float)
    cf = np.asarray(counterfactual, dtype=float)
    finite = np.isfinite(cond) & np.isfinite(act) & np.isfinite(cf)
    frame = pd.DataFrame(
        {
            "condition_bin": pd.cut(cond[finite], bins=bins).astype(str),
            "actual": act[finite],
            "counterfactual": cf[finite],
        }
    )
    all_bins = pd.cut([], bins=bins).categories.astype(str)
    grouped = frame.groupby("condition_bin", observed=False)
    table = grouped[["actual", "counterfactual"]].sum()
    table["n_records"] = grouped.size()
    table = table.reindex(all_bins)
    table["n_records"] = table["n_records"].fillna(0).astype(int)
    # empty bins get a 0 energy sum (they contribute nothing to a downstream aggregation)
    table[["actual", "counterfactual"]] = table[["actual", "counterfactual"]].fillna(0.0)
    denom = table["counterfactual"].to_numpy()
    table["p50_uplift"] = (
        np.divide(table["actual"].to_numpy(), denom, out=np.full(len(table), np.nan), where=denom != 0) - 1.0
    )
    table = table.rename(columns={"actual": "sum_actual", "counterfactual": "sum_counterfactual"})
    return table.reset_index(names="condition_bin")[
        ["condition_bin", "p50_uplift", "n_records", "sum_actual", "sum_counterfactual"]
    ]
