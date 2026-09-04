"""Accuracy, precision and combined score over a set of signed errors.

A method's signed error on one campaign is ``estimate - truth``. Aggregated across an
ensemble of replicates these give:

- **accuracy / bias** = mean signed error,
- **precision / spread** = population standard deviation (``ddof=0``),
- **combined score** = RMSE of the signed errors = ``sqrt(mean(error**2))``.

With the population spread the score is exactly ``sqrt(bias**2 + spread**2)``, so it is small
only when *both* accuracy and precision are good.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ErrorSummary:
    """Bias, spread and combined score over ``n`` finite signed errors."""

    bias: float
    spread: float
    score: float
    n: int


def summarize_errors(signed_errors: npt.ArrayLike) -> ErrorSummary:
    """Summarise signed errors into bias, spread (population std) and RMSE score.

    Non-finite errors (e.g. a replicate that produced no estimate) are dropped before
    aggregating. An empty (or all-NaN) input yields a NaN summary with ``n == 0``.
    """
    errors = np.asarray(signed_errors, dtype=float).ravel()
    errors = errors[np.isfinite(errors)]
    n = int(errors.size)
    if n == 0:
        return ErrorSummary(bias=float("nan"), spread=float("nan"), score=float("nan"), n=0)
    bias = float(errors.mean())
    spread = float(errors.std(ddof=0))
    score = float(np.sqrt(np.mean(errors**2)))
    return ErrorSummary(bias=bias, spread=spread, score=score, n=n)
