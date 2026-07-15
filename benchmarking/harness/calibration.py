"""Score an uncertainty: whether a reported sigma matches the error it claims to describe.

The P50 metrics in :mod:`benchmarking.harness.metrics` ask how close an estimate lands to truth.
These ask a different question: **was the method right about how close it would land?** The
statistic is the standardised error

    z = (estimate - truth) / sigma = signed_error / sigma

and a well-calibrated 1-sigma uncertainty makes ``z`` a standard normal: ~68.3% of estimates
within 1 sigma of truth, ``std(z) == 1``.

Note what this scores sigma *against*: the deviation from **ground truth**, which contains the
method's bias as well as its sampling scatter. A sigma covering sampling variance alone (a
bootstrap sees nothing else) will therefore under-cover wherever the method is biased, and that is
a real finding about the uncertainty rather than an unfairness in the metric — a reported
uncertainty is a claim about how much to trust the number, and a biased number is not more
trustworthy for being reliably biased.

Three reads, deliberately, because they fail differently:

- :attr:`~CalibrationSummary.coverage_1sigma` is the headline and the most interpretable, but it
  only asks a yes/no question per case and so wastes the magnitude of each miss.
- :attr:`~CalibrationSummary.z_spread` uses the magnitudes and is much more sensitive, but one wild
  case (a sparse bin whose ratio bootstrap went heavy-tailed) can dominate it.
- :attr:`~CalibrationSummary.z_robust` is the outlier-resistant companion. **z_spread >> z_robust
  says the miscalibration lives in the tails**, not in the typical case, which points at a
  different fix than a uniformly-too-small sigma.

All three are reported together because agreeing on a verdict is informative and disagreeing is
more so.
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
    :param mean_sigma: mean reported sigma — the interval width, so a method cannot win coverage by
        inflating sigma without that showing up here
    :param rms_error: RMS of the signed errors; the scale ``mean_sigma`` should be near
    :param n: usable cases (finite error, finite and strictly positive sigma)
    :param n_unusable: cases dropped for a non-finite or non-positive sigma despite a finite error.
        Counted rather than silently discarded: a collapsed sigma is an uncertainty failure, and
        dropping those cases quietly would flatter exactly the bins most likely to produce them.
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

    Cases with a non-finite error (no estimate, or no truth) are ignored entirely — there is
    nothing to calibrate against. Cases with a finite error but an unusable sigma (NaN, inf, zero
    or negative) are excluded from the statistics but counted in ``n_unusable``. An empty or fully
    unusable input yields a NaN summary with ``n == 0``.
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

    The caller's job is ``n``: a sweep's row count is not its independent-case count. Replicates
    sharing a campaign window across profiles, or campaign lengths that are leading prefixes of one
    another, produce strongly correlated errors — pooling them multiplies rows without adding
    independent evidence, and passing that row count here would understate the error by a lot.
    """
    if n <= 0:
        return float("nan")
    return float(np.sqrt(coverage * (1.0 - coverage) / n))
