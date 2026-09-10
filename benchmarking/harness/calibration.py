"""Score an uncertainty: whether a reported sigma matches the error it claims to describe.

The metrics in :mod:`benchmarking.harness.metrics` ask how close an estimate lands to truth; these
ask whether the method was right about how close it would land. The statistic is the standardised
error ``z = signed_error / sigma``, which a calibrated 1-sigma makes a standard normal: ~68.3%
within 1 sigma, ``std(z) == 1``.

Sigma is scored against the deviation from **ground truth**, which includes the method's bias. A
sigma covering only sampling variance will under-cover wherever the method is biased — that is a
finding about the uncertainty, not an unfairness in the metric.

Coverage, ``z_spread`` and ``z_robust`` are reported together because they fail differently, and
disagreement localises the problem (``z_spread >> z_robust`` means the tails, not the typical case).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

# P(|Z| <= 1) for a standard normal: the fraction of estimates a calibrated 1-sigma should cover.
TARGET_COVERAGE_1SIGMA = 0.6827
# median(|x - median(x)|) * this == std(x) for a normal x; scales the MAD onto a sigma.
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class CalibrationSummary:
    """How well a reported sigma described the errors it was reported alongside.

    :param coverage_1sigma: fraction of cases with ``|z| <= 1``; target :data:`TARGET_COVERAGE_1SIGMA`
    :param z_spread: population std of ``z``; target 1.0. Above 1 means sigma was too small.
    :param z_robust: ``1.4826 * MAD(z)``; target 1.0. The outlier-resistant counterpart of ``z_spread``.
    :param mean_sigma: mean reported sigma — the interval width, so inflating sigma to win coverage
        shows up here
    :param rms_error: RMS of the signed errors; the scale ``mean_sigma`` should be near
    :param n: usable cases (finite error, finite and strictly positive sigma)
    :param n_unusable: cases with a finite error but an unusable sigma. Counted, not silently
        dropped: a collapsed sigma is an uncertainty failure.
    """

    coverage_1sigma: float
    z_spread: float
    z_robust: float
    mean_sigma: float
    rms_error: float
    n: int
    n_unusable: int


def calibration_summary(signed_errors: npt.ArrayLike, sigmas: npt.ArrayLike) -> CalibrationSummary:
    """Summarise how well ``sigmas`` describe ``signed_errors``.

    A non-finite error is ignored (nothing to calibrate against). A finite error with an unusable
    sigma is excluded but counted in ``n_unusable``. Empty or fully unusable input gives a NaN
    summary with ``n == 0``.
    """
    errors = np.asarray(signed_errors, dtype=float).ravel()
    sigma = np.asarray(sigmas, dtype=float).ravel()
    if errors.shape != sigma.shape:
        msg = f"signed_errors and sigmas must be the same length; got {errors.shape} and {sigma.shape}"
        raise ValueError(msg)

    scoreable = np.isfinite(errors)
    usable = scoreable & np.isfinite(sigma) & (sigma > 0)
    n_unusable = int((scoreable & ~usable).sum())
    if not usable.any():
        nan = float("nan")
        return CalibrationSummary(
            coverage_1sigma=nan, z_spread=nan, z_robust=nan, mean_sigma=nan, rms_error=nan, n=0, n_unusable=n_unusable
        )

    err = errors[usable]
    sig = sigma[usable]
    z = err / sig
    return CalibrationSummary(
        coverage_1sigma=float(np.mean(np.abs(z) <= 1.0)),
        z_spread=float(z.std(ddof=0)),
        z_robust=float(_MAD_TO_SIGMA * np.median(np.abs(z - np.median(z)))),
        mean_sigma=float(sig.mean()),
        rms_error=float(np.sqrt(np.mean(err**2))),
        n=int(err.size),
        n_unusable=n_unusable,
    )


def summarize_calibration(
    results_df: pd.DataFrame,
    *,
    group_keys: Sequence[str],
    error_col: str = "signed_error",
    sigma_col: str = "sigma",
) -> pd.DataFrame:
    """Reduce tidy results to one calibration row per ``group_keys`` group.

    :param results_df: tidy scoring results carrying ``error_col`` and ``sigma_col``
    :param group_keys: the columns to group by (e.g. ``["block_hours", "campaign_weeks"]``)
    """
    keys = list(group_keys)
    missing = [c for c in [*keys, error_col, sigma_col] if c not in results_df.columns]
    if missing:
        msg = f"results_df is missing column(s) {missing}"
        raise ValueError(msg)

    records = []
    for values, group in results_df.groupby(keys, sort=True, dropna=False):
        summary = calibration_summary(group[error_col].to_numpy(), group[sigma_col].to_numpy())
        key_values = values if isinstance(values, tuple) else (values,)
        records.append(
            {
                **dict(zip(keys, key_values, strict=True)),
                "coverage_1sigma": summary.coverage_1sigma,
                "z_spread": summary.z_spread,
                "z_robust": summary.z_robust,
                "mean_sigma": summary.mean_sigma,
                "rms_error": summary.rms_error,
                "n": summary.n,
                "n_unusable": summary.n_unusable,
            }
        )
    columns = [*keys, "coverage_1sigma", "z_spread", "z_robust", "mean_sigma", "rms_error", "n", "n_unusable"]
    return pd.DataFrame(records, columns=columns)


def coverage_standard_error(n: int, *, coverage: float = TARGET_COVERAGE_1SIGMA) -> float:
    """Binomial standard error of a coverage estimate from ``n`` **independent** cases.

    Supplying ``n`` is the caller's job, and a sweep's row count is not it: profiles sharing campaign
    windows, prefix-nested campaign lengths, and overlapping long campaigns all multiply rows without
    adding independent evidence. Passing a row count here understates the error badly.
    """
    if n <= 0:
        return float("nan")
    return float(np.sqrt(coverage * (1.0 - coverage) / n))
