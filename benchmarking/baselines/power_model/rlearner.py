"""The R-learner estimate of a prepost uplift: one model of both periods, the period partialled out.

The forward estimate trains a model on one period and predicts the other, so any error the model makes on
data it has not seen lands in the uplift. Here both periods are treated alike:

- ``m(X)``, the subject turbine's expected power given the reference features, is fitted on the records of
  **both** periods, and ``e(X)``, the probability that a record with those features belongs to the after
  period, likewise. Both are cross-fitted: the records are cut into contiguous wall-clock blocks, the
  blocks dealt round-robin into folds, and each record is predicted by models that never saw its fold
  (:func:`time_folds`, :func:`cross_fit`).
- With ``T`` 1 after and 0 before, baseline power ``b`` and a relative uplift ``theta``, ``y = b (1 + theta T)``
  and ``m = b (1 + theta e)``, so ``y - m = theta b (T - e)``. The uplift is the regression of the power
  residual on the period residual, weighted to energy: ``theta = sum((T - e)(y - m)) / sum((T - e)^2 b)``
  with ``b = m / (1 + theta e)``, iterated (:func:`overall_theta`). A record whose period is certain from
  its features has ``T - e`` near zero and no say: conditions seen in one period only drop out.
- The uplift is measured per bin of baseline power and the bins are combined by the after period's energy
  (:func:`estimate`), so the result reads as upgraded over baseline energy in the after period whatever
  the two periods' wind distributions.

:func:`estimate` also returns a counterfactual power for every after record, the baseline power scaled
within each power bin so that the bin's forward ratio ``sum(y) / sum(counterfactual) - 1`` is the bin's
uplift. The steps written for a forward counterfactual (the uplift distributions, the row dump, the
diagnostics) then run on it unchanged, and ``sum(y) / sum(counterfactual) - 1`` over the after period is
the headline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm

from benchmarking.baselines.power_model.shape import bin_codes

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

DEFAULT_BLOCK_DAYS = 7.0
DEFAULT_N_FOLDS = 5
# A propensity is kept off 0 and 1 so that no record's weight is exactly zero or its baseline undefined.
_PROPENSITY_CLIP = (0.01, 0.99)
_MAX_ITERATIONS = 50
_TOLERANCE = 1e-9
# A power bin is measured only where its records average at least this share of rated power: below it the
# energy the uplift is relative to is too small for a ratio.
_MIN_BIN_POWER_FRACTION = 0.02
# A bin's uplift outside this range is not a measurement.
_THETA_RANGE = (-0.5, 1.0)
# The after records' counterfactual is the baseline power times a factor per bin; a factor outside this
# range means the bin's energy is too small to scale by, and the baseline power is used as it is.
_SCALE_RANGE = (0.5, 2.0)
_RESAMPLE_CHUNK = 100
_SIGMA_PERCENTILES = (100.0 * norm.cdf(-1.0), 100.0 * norm.cdf(1.0))
_MIN_BLOCKS = 2
_MIN_RESAMPLES = 2
# The sums each power bin carries through a resample: sum(w r), sum(w^2 b), and the after period's
# sum(y) and sum(b).
_N_SUMS = 4


def time_folds(
    times: pd.DatetimeIndex, *, block_days: float = DEFAULT_BLOCK_DAYS, n_folds: int = DEFAULT_N_FOLDS
) -> npt.NDArray[np.int64]:
    """Return each record's fold: contiguous wall-clock blocks of ``block_days``, dealt round-robin.

    Blocks are counted from the first record, so a gap in the records leaves a block empty and does not
    shift the others. Every fold samples the whole span, and a record's neighbours in time share its fold
    except at the block edges.
    """
    if len(times) == 0:
        return np.array([], dtype=np.int64)
    block = np.floor((times - times.min()) / pd.Timedelta(days=block_days)).astype(np.int64)
    return np.asarray(block % n_folds, dtype=np.int64)


def cross_fit(
    folds: npt.ArrayLike,
    *,
    fit_predict: Callable[[npt.NDArray[np.int64], npt.NDArray[np.int64]], npt.ArrayLike],
) -> npt.NDArray[np.float64]:
    """Return every record's out-of-fold prediction.

    :param folds: each record's fold
    :param fit_predict: fits on the records at the first positions and returns predictions for the records
        at the second
    """
    folds = np.asarray(folds)
    out = np.full(len(folds), np.nan)
    for fold in np.unique(folds):
        held = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        if len(train) == 0:
            continue
        out[held] = np.asarray(fit_predict(train, held), dtype=float)
    return out


def clip_propensity(e: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the propensities kept off 0 and 1."""
    return np.clip(np.asarray(e, dtype=float), *_PROPENSITY_CLIP)


def overall_theta(y: npt.ArrayLike, m: npt.ArrayLike, after: npt.ArrayLike, e: npt.ArrayLike) -> float:
    """Return the relative uplift ``sum((T - e)(y - m)) / sum((T - e)^2 b)`` with ``b = m / (1 + theta e)``.

    NaN without overlap between the periods (a zero denominator) or when the iteration leaves the range
    in which ``1 + theta e`` stays positive.
    """
    y, m, e = (np.asarray(v, dtype=float) for v in (y, m, e))
    w = np.asarray(after, dtype=float) - e
    numerator = float(np.sum(w * (y - m)))
    theta = 0.0
    for _ in range(_MAX_ITERATIONS):
        denominator = float(np.sum(w * w * m / (1.0 + theta * e)))
        if not np.isfinite(denominator) or denominator <= 0:
            return float("nan")
        new = numerator / denominator
        if not np.isfinite(new) or new <= -1.0:
            return float("nan")
        if abs(new - theta) < _TOLERANCE:
            return new
        theta = new
    return theta


@dataclass(frozen=True)
class RLearnerEstimate:
    """The R-learner estimate over one subject's used records of both periods.

    ``theta`` is the overall uplift, weighted by the records' overlap. ``bin_theta`` is the uplift per
    power bin, the overall one where the bin is not ``measured``. ``bins`` and ``baseline`` are per record:
    its power bin and its baseline power ``m / (1 + theta e)`` under its bin's uplift. ``counterfactual``
    is per after record, in the records' order, and ``uplift`` is the headline
    ``sum(y) / sum(counterfactual) - 1`` over the after records. ``scaled`` says which bins' counterfactual
    is the baseline power scaled to the bin's uplift; the others' is the baseline power as it is.
    """

    theta: float
    bin_theta: npt.NDArray[np.float64]
    measured: npt.NDArray[np.bool_]
    scaled: npt.NDArray[np.bool_]
    bins: npt.NDArray[np.int64]
    baseline: npt.NDArray[np.float64]
    counterfactual: npt.NDArray[np.float64]
    uplift: float


def estimate(
    y: npt.ArrayLike,
    m: npt.ArrayLike,
    after: npt.ArrayLike,
    e: npt.ArrayLike,
    *,
    power_edges: Sequence[float],
    rated_power_kw: float,
    min_records: float,
) -> RLearnerEstimate:
    """Return the R-learner estimate: the uplift per bin of baseline power, combined by after-period energy.

    A bin is measured when each period holds ``min_records`` of its records, its records average at least
    2 % of rated power and its uplift is finite and plausible; another bin takes the overall uplift.

    :param y: the subject turbine's power per record
    :param m: the cross-fitted expected power per record
    :param after: whether each record is in the after period
    :param e: the cross-fitted probability that each record is in the after period
    :param power_edges: the edges of the baseline-power bins, kW
    :param rated_power_kw: the subject turbine's rated power
    :param min_records: the records of each period a bin needs to be measured
    """
    y, m = np.asarray(y, dtype=float), np.asarray(m, dtype=float)
    e = clip_propensity(e)
    is_after = np.asarray(after, dtype=bool)
    t = is_after.astype(float)
    n_bins = len(power_edges) - 1
    theta = overall_theta(y, m, t, e)
    theta_0 = theta if np.isfinite(theta) else 0.0
    baseline_0 = m / (1.0 + theta_0 * e)
    bins = _bins(baseline_0, power_edges)
    bin_theta = np.full(n_bins, theta_0)
    measured = np.zeros(n_bins, dtype=bool)
    for k in range(n_bins):
        rows = bins == k
        enough = min(np.count_nonzero(rows & is_after), np.count_nonzero(rows & ~is_after)) >= max(min_records, 1)
        if not enough or float(np.mean(baseline_0[rows])) < _MIN_BIN_POWER_FRACTION * rated_power_kw:
            continue
        theta_k = overall_theta(y[rows], m[rows], t[rows], e[rows])
        if np.isfinite(theta_k) and _THETA_RANGE[0] < theta_k < _THETA_RANGE[1]:
            bin_theta[k] = theta_k
            measured[k] = True
    baseline = m / (1.0 + bin_theta[bins] * e)
    y_after = np.bincount(bins[is_after], weights=y[is_after], minlength=n_bins)
    b_after = np.bincount(bins[is_after], weights=baseline[is_after], minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = y_after / ((1.0 + bin_theta) * b_after)
    scaled = np.isfinite(scale) & (scale > _SCALE_RANGE[0]) & (scale < _SCALE_RANGE[1])
    scale = np.where(scaled, scale, 1.0)
    counterfactual = baseline[is_after] * scale[bins[is_after]]
    total = float(counterfactual.sum())
    uplift = float(y[is_after].sum()) / total - 1.0 if np.isfinite(theta) and total > 0 else float("nan")
    return RLearnerEstimate(
        theta=theta,
        bin_theta=bin_theta,
        measured=measured,
        scaled=scaled,
        bins=bins,
        baseline=baseline,
        counterfactual=counterfactual,
        uplift=uplift,
    )


def _bins(baseline: npt.NDArray[np.float64], power_edges: Sequence[float]) -> npt.NDArray[np.int64]:
    """Return each record's power bin; a record outside the edges takes the end bin on its side."""
    codes = bin_codes(baseline, power_edges)
    outside = codes < 0
    if outside.any():
        low = np.nan_to_num(baseline[outside], nan=-np.inf) <= power_edges[0]
        codes[outside] = np.where(low, 0, len(power_edges) - 2)
    return codes


def bootstrap_sigmas(
    times: pd.DatetimeIndex,
    *,
    y: npt.ArrayLike,
    m: npt.ArrayLike,
    after: npt.ArrayLike,
    e: npt.ArrayLike,
    fitted: RLearnerEstimate,
    block_hours: Sequence[float],
    n_resamples: int,
    seed: int,
) -> list[dict[str, float]]:
    """Return the headline's block-bootstrap sigma at each block length, the fitted ``m`` and ``e`` fixed.

    Circular wall-clock blocks are drawn over the whole span of both periods. Each resample re-measures
    every measured bin's uplift and re-forms the headline, with the records' bins and baseline power
    fixed. The records are first summed into atoms as long as the shortest block, so a block starts on
    that grid. One row per block length: ``block_hours``, ``sigma``, ``sigma_robust`` (half the 16th to
    84th percentile range), ``n_blocks`` and ``frac_resamples_finite``.
    """
    y, m = np.asarray(y, dtype=float), np.asarray(m, dtype=float)
    e = clip_propensity(e)
    is_after = np.asarray(after, dtype=bool)
    lengths = sorted({float(h) for h in block_hours})
    nan = float("nan")
    if len(times) == 0 or not lengths or n_resamples < _MIN_RESAMPLES:
        return [
            {"block_hours": h, "sigma": nan, "sigma_robust": nan, "n_blocks": 0, "frac_resamples_finite": nan}
            for h in lengths
        ]
    atom = pd.Timedelta(hours=lengths[0])
    atom_of = np.floor((times - times.min()) / atom).astype(np.int64)
    n_atoms = int(atom_of.max()) + 1
    n_bins = len(fitted.bin_theta)
    w = is_after.astype(float) - e
    per_record = np.column_stack(
        [w * (y - m), w * w * fitted.baseline, np.where(is_after, y, 0.0), np.where(is_after, fitted.baseline, 0.0)]
    )
    sums = np.zeros((n_atoms, n_bins, _N_SUMS))
    np.add.at(sums, (atom_of, fitted.bins), per_record)
    flat = sums.reshape(n_atoms, n_bins * _N_SUMS)
    prefix = np.concatenate([np.zeros((1, flat.shape[1])), np.cumsum(np.concatenate([flat, flat]), axis=0)])
    rng = np.random.default_rng(seed)
    rows = []
    for hours in lengths:
        block = max(1, min(n_atoms, round(hours / lengths[0])))
        n_blocks = math.ceil(n_atoms / block)
        if n_blocks < _MIN_BLOCKS:
            rows.append(
                {
                    "block_hours": hours,
                    "sigma": nan,
                    "sigma_robust": nan,
                    "n_blocks": n_blocks,
                    "frac_resamples_finite": nan,
                }
            )
            continue
        draws = []
        for start in range(0, n_resamples, _RESAMPLE_CHUNK):
            n = min(_RESAMPLE_CHUNK, n_resamples - start)
            starts = rng.integers(0, n_atoms, size=(n, n_blocks))
            totals = (prefix[starts + block] - prefix[starts]).sum(axis=1).reshape(n, n_bins, _N_SUMS)
            draws.append(_headlines(totals, fitted=fitted))
        uplift = np.concatenate(draws)
        finite = uplift[np.isfinite(uplift)]
        enough = finite.size >= _MIN_RESAMPLES
        low, high = np.percentile(finite, _SIGMA_PERCENTILES) if enough else (nan, nan)
        rows.append(
            {
                "block_hours": hours,
                "sigma": float(finite.std(ddof=1)) if enough else nan,
                "sigma_robust": float((high - low) / 2.0),
                "n_blocks": n_blocks,
                "frac_resamples_finite": float(finite.size / uplift.size),
            }
        )
    return rows


def _headlines(totals: npt.NDArray[np.float64], *, fitted: RLearnerEstimate) -> npt.NDArray[np.float64]:
    """Return each resample's headline from its per-bin sums, shape ``(n_resamples, n_bins, 4)``."""
    wr, wwb, y_after, b_after = (totals[:, :, i] for i in range(_N_SUMS))
    with np.errstate(divide="ignore", invalid="ignore"):
        overall = wr.sum(axis=1) / wwb.sum(axis=1)
        theta = np.where(fitted.measured[None, :] & (wwb > 0), wr / wwb, overall[:, None])
        theta = np.where((theta > _THETA_RANGE[0]) & (theta < _THETA_RANGE[1]), theta, np.nan)
        counterfactual = np.where(fitted.scaled[None, :], y_after / (1.0 + theta), b_after)
        # a bin no after record fell in adds nothing, whatever its uplift
        counterfactual = np.where((y_after == 0) & (b_after == 0), 0.0, counterfactual)
        return y_after.sum(axis=1) / counterfactual.sum(axis=1) - 1.0


def propensity_auc(after: npt.ArrayLike, e: npt.ArrayLike, *, folds: npt.ArrayLike | None = None) -> float:
    """Return how well the propensity tells the periods apart: 0.5 for not at all, 1 for perfectly.

    With ``folds`` it is the mean over the folds that hold records of both periods, weighted by their
    records. A fold's propensities come from the other folds, whose share of after records falls as the
    fold's own rises, so propensities are comparable within a fold and not across them. NaN when no fold
    holds both periods.
    """
    is_after = np.asarray(after, dtype=bool)
    e = np.asarray(e, dtype=float)
    groups = np.zeros(is_after.size, dtype=np.int64) if folds is None else np.asarray(folds)
    total = weight = 0.0
    for fold in np.unique(groups):
        rows = groups == fold
        n_after = int(np.count_nonzero(is_after[rows]))
        n_before = int(np.count_nonzero(rows) - n_after)
        if n_after == 0 or n_before == 0:
            continue
        ranks = pd.Series(e[rows]).rank().to_numpy()
        auc = (ranks[is_after[rows]].sum() - n_after * (n_after + 1) / 2) / (n_after * n_before)
        total += auc * (n_after + n_before)
        weight += n_after + n_before
    return float(total / weight) if weight else float("nan")


def propensity_overlap(after: npt.ArrayLike, e: npt.ArrayLike) -> float:
    """Return ``mean((T - e)^2)`` over what it would be were the period unpredictable, ``p (1 - p)``.

    Near 1 when the features do not tell the periods apart; near 0 when they do, and little of the data
    then informs the uplift.
    """
    t = np.asarray(after, dtype=float)
    share = float(t.mean()) if t.size else float("nan")
    if not 0 < share < 1:
        return float("nan")
    return float(np.mean((t - clip_propensity(e)) ** 2) / (share * (1 - share)))
