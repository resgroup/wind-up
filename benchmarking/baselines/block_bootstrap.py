"""Circular block bootstrap for a toggle energy-ratio uplift.

The uncertainty behind ``ToggleSpecialistMethod``, kept separate because it says nothing about
SCADA: it takes paired ``(test, reference)`` sums on a timeline and returns a sigma.

A block is a wall-clock interval carrying its on- and off-rows **together**, so the on/off pairing
that makes the estimate precise survives resampling. Blocks start anywhere on the timebase grid and
wrap past the campaign end. Both ``rho_up`` and ``rho_base`` are recomputed per resample and the
ratio re-formed, rather than linearised.

Block sums come from prefix sums over the records (doubled end-to-end so a wrapped block is still
two lookups), so a resample is a gather-and-subtract rather than a pass over the data.

Rationale for the design and for the block length: findings F28.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import norm, t

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

# Resamples per chunk; keeps the (chunk, n_draw, n_cells, 4) gather to tens of MB.
_RESAMPLE_CHUNK = 250
# Per (cell, segment): the numerator and denominator of each segment's rho.
_N_QUANTITIES = 4
_MIN_RESAMPLES_FOR_SPREAD = 2
# With one block covering the whole campaign, every resample is that campaign: nothing varies.
_MIN_BLOCKS_FOR_SPREAD = 2
# Normal quantiles at -/+1 sigma, so (p84 - p16) / 2 is a sigma for a normal.
_SIGMA_PERCENTILES = (15.865525, 84.134475)
# Floor on the fallback's degrees of freedom, mirroring wind_up's `clip(lower=2)`: with one
# record a side has no scatter of its own, so df=1 is the widest t the convention allows.
_MIN_FALLBACK_DF = 1
# Fewest records a segment needs before its own scatter is worth measuring, for `relative_scatter`.
_MIN_RECORDS_PER_SIDE = 3
# Records per side over which the report ramps linearly from the fallback to the bootstrap (F33):
# pure fallback at/below LO, pure bootstrap at/above HI.
_BLEND_LO_RECORDS = 3
_BLEND_HI_RECORDS = 7


@dataclass(frozen=True)
class CellUncertainty:
    """The uncertainty of one cell's uplift (the headline, or one condition bin).

    :param sigma: the reported 1-sigma: a records-weighted ramp between the two components below (F33).
    :param sigma_bootstrap: std of the resampled uplifts (``ddof=1``); NaN when there were fewer than
        two finite resamples
    :param sigma_fallback: the t-inflated per-record-scatter estimate; finite even for a sparse cell
    :param sigma_robust: ``(p84 - p16) / 2`` of the resampled uplifts
    :param frac_resamples_finite: fraction of resamples with a finite uplift; below 1 means some
        resamples drew no baseline rows for this cell
    """

    sigma: float
    sigma_bootstrap: float
    sigma_fallback: float
    sigma_robust: float
    frac_resamples_finite: float


@dataclass(frozen=True)
class BootstrapResult:
    """Per-cell uncertainties from one circular block bootstrap.

    :param n_blocks: blocks drawn per resample (``ceil(campaign / block)``)
    :param cells: uncertainty per cell name, keyed as the caller keyed ``cell_membership``
    """

    n_blocks: int
    cells: dict[str, CellUncertainty]


def _nan_cells(names: list[str]) -> dict[str, CellUncertainty]:
    """Return an all-NaN uncertainty for every cell (a campaign the bootstrap cannot run on)."""
    nan = float("nan")
    return {
        name: CellUncertainty(
            sigma=nan, sigma_bootstrap=nan, sigma_fallback=nan, sigma_robust=nan, frac_resamples_finite=nan
        )
        for name in names
    }


def relative_scatter(
    test_power: npt.NDArray[np.float64],
    ref_total: npt.NDArray[np.float64],
    *,
    upgraded: npt.NDArray[np.bool_],
    baseline: npt.NDArray[np.bool_],
) -> float:
    """Return the campaign's per-record relative scatter about its own test/reference ratio.

    ``sqrt(sum(y - R*x)^2 / sum(R*x)^2)`` per segment, pooled. A ratio of sums rather than a mean of
    per-record ratios, because dividing by each record's predicted power explodes near cut-in, which
    is exactly where the sparsest bins live.

    Measured over the whole campaign (thousands of records) so it is precise, then applied to a cell
    with that cell's own record count — which is what lets a 1-record cell get a sigma at all.
    """
    residual_sq = 0.0
    predicted_sq = 0.0
    for segment in (upgraded, baseline):
        y, x = test_power[segment], ref_total[segment]
        denom = x.sum()
        if len(y) < _MIN_RECORDS_PER_SIDE or denom == 0:
            continue
        predicted = (y.sum() / denom) * x
        residual_sq += float(((y - predicted) ** 2).sum())
        predicted_sq += float((predicted**2).sum())
    if predicted_sq <= 0:
        return float("nan")
    return math.sqrt(residual_sq / predicted_sq)


def _fallback_sigma(*, n_on: int, n_off: int, s_rel: float) -> float:
    """Return a t-inflated per-record-scatter uncertainty for one cell (F33).

    ``s_rel * sqrt(1/n_on + 1/n_off)`` is the standard ratio-estimator error under multiplicative
    per-record noise, propagated through ``rho_up / rho_base``. The ``scipy.stats.t`` multiplier is
    ``wind_up``'s own convention (``pp_analysis``): ``t.ppf(norm.cdf(1), df)`` is the 1-sigma-equivalent
    quantile, tending to 1.0 as ``df`` grows and widening as data runs out. ``df`` keys off the
    *thinner* side, as ``wind_up`` does, because either side starves the ratio.
    """
    if not math.isfinite(s_rel) or n_on < 1 or n_off < 1:
        return float("nan")
    df = max(min(n_on, n_off) - 1, _MIN_FALLBACK_DF)
    return s_rel * math.sqrt(1.0 / n_on + 1.0 / n_off) * float(t.ppf(norm.cdf(1.0), df))


def bootstrap_ratio_uplift(
    *,
    times: pd.DatetimeIndex,
    test_power: npt.NDArray[np.float64],
    ref_total: npt.NDArray[np.float64],
    upgraded: npt.NDArray[np.bool_],
    baseline: npt.NDArray[np.bool_],
    cell_membership: Mapping[str, npt.NDArray[np.bool_]],
    campaign_start: pd.Timestamp,
    campaign_end: pd.Timestamp,
    timebase: pd.Timedelta,
    block_hours: float,
    n_resamples: int,
    seed: int,
) -> BootstrapResult:
    """Bootstrap the 1-sigma uncertainty of ``rho_up / rho_base - 1`` for every cell.

    All array arguments are parallel over the **used** records only (the rows the point estimate
    summed), in any order; they are sorted here.

    :param times: the used records' timestamps
    :param test_power: the test turbine's power per used record
    :param ref_total: the summed reference power per used record
    :param upgraded: which used records are toggle-on
    :param baseline: which used records are toggle-off
    :param cell_membership: cell name -> which used records belong to it (the headline, or one
        condition bin). Must be fixed by the point estimate so resampling cannot move a record
        between bins.
    :param campaign_start: the campaign's first timestamp; blocks tile forward from here, so gaps in
        the used records are covered rather than closed up
    :param campaign_end: the campaign's last timestamp
    :param timebase: analysis timebase; sets the candidate block-start grid
    :param block_hours: block length (F28). A length at or beyond the campaign leaves one block, which
        nothing can vary, so every cell reports NaN rather than a spurious near-zero sigma.
    :param n_resamples: resamples to draw
    :param seed: RNG seed, so a reported sigma is reproducible
    """
    names = list(cell_membership)
    n_records = len(times)
    campaign_s = (campaign_end - campaign_start) / pd.Timedelta(seconds=1) + timebase.total_seconds()
    if n_records == 0 or campaign_s <= 0 or n_resamples < _MIN_RESAMPLES_FOR_SPREAD:
        return BootstrapResult(n_blocks=0, cells=_nan_cells(names))

    # Elapsed seconds via pandas rather than numpy datetime arithmetic: a tz-aware DatetimeIndex
    # converts to an object array, which numpy cannot subtract.
    elapsed = np.asarray((times - campaign_start) / pd.Timedelta(seconds=1), dtype=float)
    order = np.argsort(elapsed, kind="stable")
    seconds = elapsed[order]
    prefix = _prefix_sums(
        test_power=test_power[order],
        ref_total=ref_total[order],
        upgraded=upgraded[order],
        baseline=baseline[order],
        cell_membership={name: mask[order] for name, mask in cell_membership.items()},
    )

    # Doubled timeline: a wrapped block [s, s+L) is then a plain contiguous range over `doubled`,
    # so it needs no special case and stays two lookups.
    doubled = np.concatenate([seconds, seconds + campaign_s])
    block_s = min(float(block_hours) * 3600.0, campaign_s)
    starts = np.arange(0.0, campaign_s, timebase.total_seconds())
    lo_idx = np.searchsorted(doubled, starts, side="left")
    hi_idx = np.searchsorted(doubled, starts + block_s, side="left")
    n_blocks = math.ceil(campaign_s / block_s)

    rng = np.random.default_rng(seed)
    totals = np.zeros((n_resamples, len(names), _N_QUANTITIES))
    for lo in range(0, n_resamples, _RESAMPLE_CHUNK):
        hi = min(lo + _RESAMPLE_CHUNK, n_resamples)
        drawn = rng.integers(0, len(starts), size=(hi - lo, n_blocks))
        totals[lo:hi] = (prefix[hi_idx[drawn]] - prefix[lo_idx[drawn]]).sum(axis=1)

    # The fallback is computed for every cell regardless, so a caller can re-judge the blend rule
    # offline from a saved sweep instead of re-running one.
    s_rel = relative_scatter(test_power, ref_total, upgraded=upgraded, baseline=baseline)
    counts = {
        name: (int((mask & upgraded).sum()), int((mask & baseline).sum())) for name, mask in cell_membership.items()
    }
    fallback = {name: _fallback_sigma(n_on=on, n_off=off, s_rel=s_rel) for name, (on, off) in counts.items()}

    if n_blocks < _MIN_BLOCKS_FOR_SPREAD:
        # One block spans the whole campaign, so every resample is the same campaign and nothing can
        # vary. Any sigma it returned would be float residue (~1e-15), not a real certainty.
        return BootstrapResult(n_blocks=n_blocks, cells=_fallback_only_cells(names, fallback=fallback))

    uplift = _uplift_from_totals(totals)
    boot_weight = {name: _bootstrap_weight(min(on, off)) for name, (on, off) in counts.items()}
    return BootstrapResult(
        n_blocks=n_blocks,
        cells=_summarise(uplift, names=names, fallback=fallback, boot_weight=boot_weight),
    )


def _bootstrap_weight(n_min: int) -> float:
    """Weight on the bootstrap for a cell with ``n_min`` records on its thinner side (F33)."""
    span = _BLEND_HI_RECORDS - _BLEND_LO_RECORDS
    return float(np.clip((n_min - _BLEND_LO_RECORDS) / span, 0.0, 1.0))


def _fallback_only_cells(names: list[str], *, fallback: dict[str, float]) -> dict[str, CellUncertainty]:
    """Cells for a campaign the bootstrap cannot run on at all: the fallback is all there is."""
    nan = float("nan")
    return {
        name: CellUncertainty(
            sigma=fallback[name],
            sigma_bootstrap=nan,
            sigma_fallback=fallback[name],
            sigma_robust=nan,
            frac_resamples_finite=nan,
        )
        for name in names
    }


def _prefix_sums(
    *,
    test_power: npt.NDArray[np.float64],
    ref_total: npt.NDArray[np.float64],
    upgraded: npt.NDArray[np.bool_],
    baseline: npt.NDArray[np.bool_],
    cell_membership: Mapping[str, npt.NDArray[np.bool_]],
) -> npt.NDArray[np.float64]:
    """Cumulative ``(test, ref)`` sums per ``(cell, segment)`` over the doubled record timeline.

    Returns shape ``(2 * n_records + 1, n_cells, 4)``, where the last axis is
    ``(test_on, ref_on, test_off, ref_off)`` and the leading zero row lets any block's sums be one
    subtraction. Doubling the records mirrors the doubled timeline so a wrapped block reads
    contiguously.
    """
    names = list(cell_membership)
    values = np.zeros((len(test_power), len(names), _N_QUANTITIES))
    for i, name in enumerate(names):
        member = cell_membership[name]
        on = member & upgraded
        off = member & baseline
        values[:, i, 0] = np.where(on, test_power, 0.0)
        values[:, i, 1] = np.where(on, ref_total, 0.0)
        values[:, i, 2] = np.where(off, test_power, 0.0)
        values[:, i, 3] = np.where(off, ref_total, 0.0)
    doubled = np.concatenate([values, values], axis=0)
    return np.concatenate([np.zeros((1, len(names), _N_QUANTITIES)), np.cumsum(doubled, axis=0)], axis=0)


def _uplift_from_totals(totals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Re-form ``rho_up / rho_base - 1`` per (resample, cell) from resampled sums.

    The degeneracy guards mirror the point estimate's, so a resample fails only where the point
    estimate would have failed on the same rows.
    """
    test_on, ref_on, test_off, ref_off = (totals[..., k] for k in range(_N_QUANTITIES))
    nan = np.full(test_on.shape, np.nan)
    rho_up = np.divide(test_on, ref_on, out=nan.copy(), where=ref_on != 0)
    rho_base = np.divide(test_off, ref_off, out=nan.copy(), where=ref_off != 0)
    valid = np.isfinite(rho_base) & (rho_base != 0) & np.isfinite(rho_up)
    return np.divide(rho_up, rho_base, out=nan.copy(), where=valid) - 1.0


def _summarise(
    uplift: npt.NDArray[np.float64],
    *,
    names: list[str],
    fallback: dict[str, float],
    boot_weight: dict[str, float],
) -> dict[str, CellUncertainty]:
    """Reduce each cell's resamples to a bootstrap sigma, and ramp between it and the fallback.

    The reported sigma is ``w*bootstrap + (1-w)*fallback`` where ``w`` is ``boot_weight`` (0 at the
    sparse end, 1 once the cell is well populated), or whichever component is finite when the other is
    not. The bootstrap reports NaN when there are fewer than two finite resamples.
    """
    cells = {}
    for i, name in enumerate(names):
        values = uplift[:, i]
        finite = np.isfinite(values)
        n_finite = int(finite.sum())
        frac = n_finite / len(values)
        kept = values[finite]
        usable = n_finite >= _MIN_RESAMPLES_FOR_SPREAD
        boot = float(kept.std(ddof=1)) if usable else float("nan")
        robust = float(np.subtract(*np.percentile(kept, _SIGMA_PERCENTILES[::-1])) / 2.0) if usable else float("nan")
        cells[name] = CellUncertainty(
            sigma=_blend(boot, fallback[name], boot_weight[name]),
            sigma_bootstrap=boot,
            sigma_fallback=fallback[name],
            sigma_robust=robust,
            frac_resamples_finite=frac,
        )
    return cells


def _blend(bootstrap: float, fallback: float, weight: float) -> float:
    """Linear ramp ``weight*bootstrap + (1-weight)*fallback``, using whichever side is finite."""
    if math.isfinite(bootstrap) and math.isfinite(fallback):
        return weight * bootstrap + (1.0 - weight) * fallback
    if math.isfinite(bootstrap):
        return bootstrap
    return fallback
