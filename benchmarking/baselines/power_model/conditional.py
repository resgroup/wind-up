"""Pure conditional-decomposition helpers (imputation + energy-identity re-level).

Extracted from ``method.py`` so they are unit-testable without a fit and keep the method file
focused. The imputation fills the per-bin uplift *shape* for bins the two-direction combine could
not measure (a degenerate non-positive ratio, or too few matched rows once the count floor is on);
the re-level pins those imputed bins and rescales only the measured bins so the whole decomposition
energy-aggregates back to the headline exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def impute_uncovered_bins(
    one_plus_u: np.ndarray,
    *,
    condition: str,
    measured: np.ndarray,
    one_plus_overall: float,
) -> np.ndarray:
    """Fill the per-bin ``1+u`` shape for uncovered bins; measured bins pass through unchanged.

    Bins **must** be in ascending bin order (low ws / low TI first). ``measured`` is the trust mask
    (``True`` = keep the two-direction shape). Uncovered bins:

    - ``ws``: backward-fill from the nearest covered bin above (uplift-vs-ws is Cp-shaped, so a low
      gap looks most like the next covered bin up), then any bins above the last covered one take
      ``1.0`` — 0 uplift, since at rated both baseline and upgraded hit rated power. This 0-at-rated
      prior is wrong for uprating / power-boost upgrades; it is a documented, replaceable default.
    - ``ti``: no ordering physics, so uncovered bins take the overall uplift (``one_plus_overall``).

    Returns an all-finite array provided at least one bin is measured (ws) / ``one_plus_overall`` is
    finite (ti).
    """
    s = pd.Series(np.asarray(one_plus_u, dtype=float))
    s[~np.asarray(measured, dtype=bool)] = np.nan
    s = s.bfill().fillna(1.0) if condition == "ws" else s.fillna(float(one_plus_overall))
    return s.to_numpy()


def relevel_conditional(
    sum_actual_b: np.ndarray,
    one_plus_u_b: np.ndarray,
    *,
    measured: np.ndarray,
    one_plus_overall: float,
) -> np.ndarray:
    """Rescale measured bins by one λ (imputed bins pinned) so the decomposition aggregates to overall.

    The aggregation is the ratio-of-sums ``Σactual / Σ(actual/(1+u))`` and must equal
    ``one_plus_overall``. Imputed bins (``~measured``)
    contribute a fixed counterfactual energy ``C_i = Σ_imp actual/(1+u_imp)``; measured bins scale as
    ``1+u -> λ(1+u)`` so their counterfactual energy is ``S_m/λ`` with ``S_m = Σ_meas actual/(1+u)``.
    Setting ``S_m/λ + C_i = Σactual / one_plus_overall`` gives
    ``λ = S_m / (Σactual/one_plus_overall - C_i)``. If there are no measured bins or the denominator is
    non-positive (imputed energy already exceeds the headline total), λ cannot be solved for a positive
    scale — fall back to reporting ``one_plus_overall`` in every bin.
    """
    a = np.asarray(sum_actual_b, dtype=float)
    u1 = np.asarray(one_plus_u_b, dtype=float)
    is_measured = np.asarray(measured, dtype=bool)
    usable = np.isfinite(u1) & (u1 != 0) & np.isfinite(a)
    m = is_measured & usable
    imp = (~is_measured) & usable
    total_actual = float(a[np.isfinite(a)].sum())
    s_m = float((a[m] / u1[m]).sum()) if m.any() else 0.0
    c_i = float((a[imp] / u1[imp]).sum()) if imp.any() else 0.0
    denom = total_actual / one_plus_overall - c_i
    if not m.any() or denom <= 0:
        return np.full_like(u1, float(one_plus_overall))
    lam = s_m / denom
    out = u1.copy()
    out[m] = lam * u1[m]
    return out
