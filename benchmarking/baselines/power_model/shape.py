"""The uplift shape of method step 15: measured against predicted power, cast record by record to other axes.

The after period's records are binned by the test turbine's predicted (counterfactual) power. Each bin's
forward uplift ``sum(actual) / sum(predicted) - 1`` is measured where the bin holds enough hours, and the
measured bins are shifted by one constant so that, weighted by predicted energy, they combine to the
headline. A measured bin keeps its own uplift only where its deviation from the overall uplift differs
from the references' mean deviation by more than :data:`GATE_Z` times its sampling sigma and the
references' spread in quadrature; the others read the headline, and every measured bin then moves by
one more constant so the shape still combines to the headline.
Each record carries its power bin's uplift (the headline where its bin is not measured), and any other
axis's bin reports the predicted-energy-weighted mean over its records.

A subset of the records (one direction sector, one solar band, waked or front row) gets its own power
shape on coarser bins, unshifted, beside the whole period's shape on the same bins, so a variable whose
apparent effect is only the power dependence restated is told apart from one with an effect of its own.

Every number is computed from record weights: all ones for the estimate, and block-bootstrap counts for
each resample, so a bin's sampling sigma comes from re-measuring, re-shifting and re-casting the same way,
without the gate. A bin's ``sigma_uplift`` joins its sampling sigma, the headline's level sigma and the
references' spread of deviations in quadrature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PREDICTED_POWER = "predicted_power"
MEASURED = "measured"
CAST = "cast"
# Resamples handled at once, to bound the memory of the weight matrix.
_CHUNK = 250
# A subset bin is flagged when its difference from the whole period's shape exceeds this many sigma.
FLAG_Z = 2.0
# A measured power bin keeps its own uplift when it differs from the references' shape by more than this many sigma.
GATE_Z = 2.0
# The references a power bin needs before their mean and spread of deviations count.
_MIN_REFERENCES = 2
# The power band, as fractions of rated, over which the free-stream scale factor is fitted.
CP_BAND = (0.4, 0.6)
_MIN_RESAMPLES = 2

Axis = tuple[npt.ArrayLike, "Sequence[float]"]


@dataclass(frozen=True)
class ShapeResult:
    """The shape and its casts, the subset shapes, and each record's label per axis.

    ``frame`` has one row per axis and bin: ``condition``, ``condition_bin``, ``kind`` (``measured`` for
    the predicted-power shape, ``cast`` for the others), ``p50_uplift``, ``sigma_sampling``,
    ``sigma_reference`` (the references' spread of deviations), ``sigma_uplift``, ``hours``, ``measured``
    (a power bin with enough hours, or a cast bin with one behind it), ``kept`` (a measured power bin that
    keeps its own uplift, or a cast bin with one behind it) and ``thin`` (fewer hours than the minimum).
    ``subsets`` has one row per subset and power bin. ``labels`` is indexed by the records' times, one
    column per axis.
    """

    frame: pd.DataFrame
    subsets: pd.DataFrame
    labels: pd.DataFrame


def bin_codes(values: npt.ArrayLike, edges: Sequence[float]) -> npt.NDArray[np.int64]:
    """Return each value's bin index under ``(lo, hi]`` edges; -1 for NaN or outside the edges."""
    return np.asarray(pd.cut(np.asarray(values, dtype=float), bins=list(edges)).codes, dtype=np.int64)


def bin_labels(edges: Sequence[float]) -> list[str]:
    """Return the ``(lo, hi]`` label of each bin, as the harness writes them."""
    return [str(c) for c in pd.cut([], bins=list(edges)).categories.astype(str)]


def block_weights(
    times: pd.DatetimeIndex,
    *,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    timebase: pd.Timedelta,
    block_hours: float,
    n_resamples: int,
    seed: int,
) -> npt.NDArray[np.float64] | None:
    """Return how many times each record is drawn in each resample, shape ``(n_resamples, len(times))``.

    Circular blocks of ``block_hours`` tile wall-clock time from ``period_start`` and wrap past
    ``period_end``, as :func:`~benchmarking.baselines.block_bootstrap.bootstrap_energy_ratio` draws them:
    each resample draws ``ceil(period / block)`` block starts with replacement. ``None`` when the period
    holds fewer than two blocks, or there are no records.
    """
    n = len(times)
    period_s = (period_end - period_start) / pd.Timedelta(seconds=1) + timebase.total_seconds()
    if n == 0 or period_s <= 0:
        return None
    block_s = min(float(block_hours) * 3600.0, period_s)
    n_blocks = math.ceil(period_s / block_s)
    if n_blocks < 2:  # noqa: PLR2004 - a spread needs two blocks
        return None
    elapsed = np.asarray((times - period_start) / pd.Timedelta(seconds=1), dtype=float)
    order = np.argsort(elapsed, kind="stable")
    seconds = elapsed[order]
    doubled = np.concatenate([seconds, seconds + period_s])
    starts = np.arange(0.0, period_s, timebase.total_seconds())
    lo_idx = np.searchsorted(doubled, starts, side="left")
    hi_idx = np.searchsorted(doubled, starts + block_s, side="left")
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(starts), size=(n_resamples, n_blocks))
    diff = np.zeros((n_resamples, 2 * n + 1))
    rows = np.repeat(np.arange(n_resamples), n_blocks)
    np.add.at(diff, (rows, lo_idx[drawn].ravel()), 1.0)
    np.add.at(diff, (rows, hi_idx[drawn].ravel()), -1.0)
    cover = np.cumsum(diff, axis=1)[:, : 2 * n]
    sorted_weights = cover[:, :n] + cover[:, n:]
    weights = np.empty_like(sorted_weights)
    weights[:, order] = sorted_weights
    return weights


def freestream_scale(
    power_kw: npt.ArrayLike,
    reference_wind_speed: npt.ArrayLike,
    *,
    rated_power_kw: float,
    rotor_diameter_m: float,
    target_cp: float,
    air_density: float,
) -> float:
    """Return the factor on the references' wind speed at which the turbine's Cp meets ``target_cp``.

    Over the records whose power is within :data:`CP_BAND` of rated,
    ``sum(P) = target_cp * 0.5 * rho * A * sum((k * V)**3)`` is solved for ``k``. NaN with no such record.
    """
    power = np.asarray(power_kw, dtype=float) * 1000.0
    ws = np.asarray(reference_wind_speed, dtype=float)
    lo, hi = CP_BAND
    keep = np.isfinite(power) & np.isfinite(ws) & (power >= lo * rated_power_kw * 1000.0)
    keep &= power <= hi * rated_power_kw * 1000.0
    if not keep.any():
        return float("nan")
    area = math.pi * rotor_diameter_m**2 / 4.0
    cube = float(np.sum(power[keep]) / (target_cp * 0.5 * air_density * area * np.sum(ws[keep] ** 3)))
    return float(np.cbrt(cube))


def reference_wind_speed_drift(wind_speed: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Return each reference's monthly wind speed over the mean of the others, flagged when it moves.

    ``wind_speed`` has one column per reference. A reference is flagged when its monthly ratios span
    more than ``threshold`` of their median. Columns: ``reference``, ``month``, ``ratio``, ``spread``,
    ``flagged``.
    """
    rows = []
    for reference in wind_speed.columns:
        others = wind_speed.drop(columns=reference).mean(axis=1)
        both = pd.DataFrame({"own": wind_speed[reference], "others": others}).dropna()
        if both.empty:
            continue
        monthly = both.groupby(both.index.strftime("%Y-%m")).mean()
        ratio = monthly["own"] / monthly["others"]
        ratio = ratio[np.isfinite(ratio)]
        if ratio.empty:
            continue
        spread = float((ratio.max() - ratio.min()) / ratio.median())
        rows += [
            {
                "reference": reference,
                "month": str(month),
                "ratio": float(r),
                "spread": spread,
                "flagged": spread > threshold,
            }
            for month, r in ratio.items()
        ]
    return pd.DataFrame(rows, columns=["reference", "month", "ratio", "spread", "flagged"])


def power_deviations(
    actual: npt.ArrayLike,
    counterfactual: npt.ArrayLike,
    *,
    power_edges: Sequence[float],
    min_records: float,
) -> npt.NDArray[np.float64]:
    """Return each predicted-power bin's forward uplift less the overall forward uplift; NaN under ``min_records``."""
    a = np.asarray(actual, dtype=float)
    c = np.asarray(counterfactual, dtype=float)
    finite = np.isfinite(a) & np.isfinite(c)
    a, c = a[finite], c[finite]
    n_power = len(power_edges) - 1
    codes = bin_codes(c, power_edges)
    ones = np.ones((1, len(c)))
    counts = np.bincount(codes[codes >= 0], minlength=n_power)
    bins = _ratio(_sums(ones, codes, a, n_power), _sums(ones, codes, c, n_power))[0]
    overall = _ratio(np.array([a.sum()]), np.array([c.sum()]))[0]
    return np.where(counts >= min_records, bins - overall, np.nan)


def reference_spread(reference_deviations: npt.ArrayLike | None, *, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the references' mean and standard deviation of deviation per bin; NaN under two references."""
    nan = np.full(n_bins, np.nan)
    if reference_deviations is None:
        return nan, nan.copy()
    d = np.asarray(reference_deviations, dtype=float).reshape(-1, n_bins)
    count = np.isfinite(d).sum(axis=0)
    mean, spread = nan.copy(), nan.copy()
    for b in np.flatnonzero(count >= _MIN_REFERENCES):
        column = d[:, b][np.isfinite(d[:, b])]
        mean[b], spread[b] = float(column.mean()), float(column.std(ddof=1))
    return mean, spread


def uplift_shape(
    *,
    times: pd.DatetimeIndex,
    actual: npt.ArrayLike,
    counterfactual: npt.ArrayLike,
    power_edges: Sequence[float],
    axes: Mapping[str, Axis],
    subsets: Mapping[str, Axis],
    subset_power_edges: Mapping[int, Sequence[float]],
    headline: float,
    reference_deviations: npt.ArrayLike | None,
    level_sigma: float,
    min_hours: float,
    hours_per_record: float,
    timebase: pd.Timedelta,
    block_hours: float,
    n_resamples: int,
    seed: int,
) -> ShapeResult:
    """Measure the predicted-power shape, cast it to ``axes`` and re-measure it in each subset.

    :param times: the after period's records used by the headline; bootstrap blocks tile from the first
    :param actual: the subject turbine's power per record
    :param counterfactual: its predicted power per record
    :param power_edges: the predicted-power bin edges, in kW
    :param axes: per axis, each record's value and the axis's bin edges
    :param subsets: per subset axis, each record's value and the edges that make its subsets
    :param subset_power_edges: the predicted-power edges a subset's shape may use, keyed by bin count;
        the most bins whose every bin reaches ``min_hours`` are used
    :param headline: the uplift the measured bins are shifted to meet
    :param reference_deviations: each kept reference's :func:`power_deviations` on ``power_edges``, one
        row per reference; ``None`` gates against the sigma alone
    :param level_sigma: the headline's reference and holdout sigma, joined into every bin's sigma
    :param min_hours: the hours a power bin needs to be measured
    :param hours_per_record: the timebase in hours
    :param timebase: the analysis timebase
    :param block_hours: the bootstrap block length
    :param n_resamples: bootstrap resamples
    :param seed: bootstrap seed
    """
    a = np.asarray(actual, dtype=float)
    c = np.asarray(counterfactual, dtype=float)
    period_start, period_end = times.min(), times.max()
    power_code = bin_codes(c, power_edges)
    n_power = len(power_edges) - 1
    min_records = min_hours / hours_per_record
    counts = np.bincount(power_code[power_code >= 0], minlength=n_power)
    measured = counts >= min_records
    axis_codes = {name: (bin_codes(values, edges), list(edges)) for name, (values, edges) in axes.items()}

    def estimate(weights: npt.NDArray[np.float64], kept: npt.NDArray[np.bool_]) -> dict[str, npt.NDArray[np.float64]]:
        predicted = _sums(weights, power_code, c, n_power)
        shifted = _gated(
            _shifted(_sums(weights, power_code, a, n_power), predicted, measured, headline),
            predicted,
            kept=kept,
            headline=headline,
        )
        out = {PREDICTED_POWER: shifted}
        filled = np.where(np.isfinite(shifted), shifted, headline)
        for name, (codes, edges) in axis_codes.items():
            joint = _joint_sums(weights, codes, power_code, c, n_axis=len(edges) - 1, n_power=n_power)
            out[name] = _cast(joint, filled)
        return out

    draws = _draws(
        lambda weights: estimate(weights, measured),
        times=times,
        period_start=period_start,
        period_end=period_end,
        timebase=timebase,
        block_hours=block_hours,
        n_resamples=n_resamples,
        seed=seed,
    )
    ones = np.ones((1, len(c)))
    power_sigma = _spread(draws[PREDICTED_POWER]) if draws is not None else np.full(n_power, np.nan)
    ref_mean, ref_spread = reference_spread(reference_deviations, n_bins=n_power)
    spread = np.nan_to_num(ref_spread)
    deviation = power_deviations(a, c, power_edges=power_edges, min_records=min_records)
    gate_sigma = np.sqrt(np.nan_to_num(power_sigma) ** 2 + spread**2)
    with np.errstate(invalid="ignore"):
        kept = measured & (np.abs(deviation - np.nan_to_num(ref_mean)) > GATE_Z * gate_sigma)
    point = estimate(ones, kept)
    rows = _shape_rows(
        PREDICTED_POWER,
        kind=MEASURED,
        point=point[PREDICTED_POWER][0],
        sigma=power_sigma,
        sigma_reference=spread,
        level_sigma=level_sigma,
        codes=power_code,
        edges=list(power_edges),
        has_measured=measured,
        has_kept=kept,
        min_records=min_records,
        hours_per_record=hours_per_record,
        headline=headline,
    )
    for name, (codes, edges) in axis_codes.items():
        n_axis = len(edges) - 1
        joint = _joint_sums(ones, codes, power_code, c, n_axis=n_axis, n_power=n_power)[0]
        energy = joint.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cast_spread = np.where(energy > 0, joint @ spread / energy, 0.0)
        rows += _shape_rows(
            name,
            kind=CAST,
            point=point[name][0],
            sigma=_spread(draws[name]) if draws is not None else np.full(n_axis, np.nan),
            sigma_reference=cast_spread,
            level_sigma=level_sigma,
            codes=codes,
            edges=edges,
            has_measured=np.array([bool(np.any(measured[power_code[codes == q]])) for q in range(n_axis)]),
            has_kept=np.array([bool(np.any(kept[power_code[codes == q]])) for q in range(n_axis)]),
            min_records=min_records,
            hours_per_record=hours_per_record,
            headline=headline,
        )
    labels = pd.DataFrame(
        {PREDICTED_POWER: c, **{name: np.asarray(values, dtype=float) for name, (values, _) in axes.items()}},
        index=times,
    )
    subset_frame = _subsets(
        subsets,
        actual=a,
        counterfactual=c,
        subset_power_edges=subset_power_edges,
        min_records=min_records,
        hours_per_record=hours_per_record,
        draw=lambda estimate_fn: _draws(
            estimate_fn,
            times=times,
            period_start=period_start,
            period_end=period_end,
            timebase=timebase,
            block_hours=block_hours,
            n_resamples=n_resamples,
            seed=seed,
        ),
    )
    return ShapeResult(frame=pd.DataFrame(rows), subsets=subset_frame, labels=labels)


def _sums(
    weights: npt.NDArray[np.float64], codes: npt.NDArray[np.int64], values: npt.NDArray[np.float64], n_bins: int
) -> npt.NDArray[np.float64]:
    """Return the weighted sum of ``values`` per bin, shape ``(resamples, n_bins)``."""
    out = np.zeros((weights.shape[0], n_bins))
    finite = np.isfinite(values)
    for b in range(n_bins):
        member = (codes == b) & finite
        out[:, b] = weights[:, member] @ values[member]
    return out


def _joint_sums(
    weights: npt.NDArray[np.float64],
    axis_codes: npt.NDArray[np.int64],
    power_codes: npt.NDArray[np.int64],
    values: npt.NDArray[np.float64],
    *,
    n_axis: int,
    n_power: int,
) -> npt.NDArray[np.float64]:
    """Return the weighted sum of ``values`` per (axis bin, power bin), shape ``(resamples, n_axis, n_power)``."""
    out = np.zeros((weights.shape[0], n_axis, n_power))
    finite = np.isfinite(values)
    for q in range(n_axis):
        in_q = (axis_codes == q) & finite
        if not in_q.any():
            continue
        for p in range(n_power):
            member = in_q & (power_codes == p)
            if member.any():
                out[:, q, p] = weights[:, member] @ values[member]
    return out


def _shifted(
    actual_sums: npt.NDArray[np.float64],
    predicted_sums: npt.NDArray[np.float64],
    measured: npt.NDArray[np.bool_],
    headline: float,
) -> npt.NDArray[np.float64]:
    """Return each measured bin's uplift shifted to meet ``headline``; NaN for a bin not measured."""
    with np.errstate(invalid="ignore", divide="ignore"):
        uplift = np.where(predicted_sums > 0, actual_sums / predicted_sums - 1.0, np.nan)
    use = measured[None, :] & np.isfinite(uplift)
    energy = np.where(use, predicted_sums, 0.0)
    total = energy.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = headline - np.where(use, predicted_sums * uplift, 0.0).sum(axis=1) / total
    return np.where(use, uplift + delta[:, None], np.nan)


def _gated(
    shifted: npt.NDArray[np.float64],
    predicted_sums: npt.NDArray[np.float64],
    *,
    kept: npt.NDArray[np.bool_],
    headline: float,
) -> npt.NDArray[np.float64]:
    """Return ``shifted`` with the measured bins outside ``kept`` at the headline, moved together to meet it."""
    measured = np.isfinite(shifted)
    value = np.where(kept[None, :], shifted, headline)
    energy = np.where(measured, predicted_sums, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = headline - np.where(measured, energy * value, 0.0).sum(axis=1) / energy.sum(axis=1)
    return np.where(measured, value + delta[:, None], np.nan)


def _cast(joint: npt.NDArray[np.float64], filled: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return each axis bin's predicted-energy-weighted uplift from its records' power bins."""
    energy = joint.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(energy > 0, (joint * (1.0 + filled[:, None, :])).sum(axis=2) / energy - 1.0, np.nan)


def _draws(
    estimate: object,
    *,
    times: pd.DatetimeIndex,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    timebase: pd.Timedelta,
    block_hours: float,
    n_resamples: int,
    seed: int,
) -> dict[str, npt.NDArray[np.float64]] | None:
    """Run ``estimate`` over block-bootstrap weights in chunks; ``None`` when no spread can be drawn."""
    if n_resamples < _MIN_RESAMPLES:
        return None
    parts: list[dict[str, npt.NDArray[np.float64]]] = []
    for lo in range(0, n_resamples, _CHUNK):
        weights = block_weights(
            times,
            period_start=period_start,
            period_end=period_end,
            timebase=timebase,
            block_hours=block_hours,
            n_resamples=min(_CHUNK, n_resamples - lo),
            seed=seed + lo,
        )
        if weights is None:
            return None
        parts.append(estimate(weights))  # type: ignore[operator]
    return {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0]}


def _spread(draws: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the standard deviation of each column's finite draws; NaN with fewer than two."""
    out = np.full(draws.shape[1], np.nan)
    for j in range(draws.shape[1]):
        finite = draws[:, j][np.isfinite(draws[:, j])]
        if finite.size >= _MIN_RESAMPLES:
            out[j] = float(finite.std(ddof=1))
    return out


def _shape_rows(
    condition: str,
    *,
    kind: str,
    point: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    sigma_reference: npt.NDArray[np.float64],
    level_sigma: float,
    codes: npt.NDArray[np.int64],
    edges: list[float],
    has_measured: npt.NDArray[np.bool_],
    has_kept: npt.NDArray[np.bool_],
    min_records: float,
    hours_per_record: float,
    headline: float,
) -> list[dict[str, object]]:
    """Return one row per bin of one axis."""
    n_bins = len(edges) - 1
    counts = np.bincount(codes[codes >= 0], minlength=n_bins)
    rows = []
    for b, label in enumerate(bin_labels(edges)):
        value = point[b] if np.isfinite(point[b]) else (headline if kind == MEASURED else np.nan)
        sampling = float(sigma[b]) if has_measured[b] else float("nan")
        reference = float(sigma_reference[b]) if has_measured[b] else 0.0
        rows.append(
            {
                "condition": condition,
                "condition_bin": label,
                "kind": kind,
                "p50_uplift": float(value),
                "sigma_sampling": sampling,
                "sigma_reference": reference,
                "sigma_uplift": float(np.sqrt(sampling**2 + level_sigma**2 + reference**2)),
                "hours": float(counts[b] * hours_per_record),
                "measured": bool(has_measured[b]),
                "kept": bool(has_kept[b]),
                "thin": bool(counts[b] < min_records),
            }
        )
    return rows


def _subsets(
    subsets: Mapping[str, Axis],
    *,
    actual: npt.NDArray[np.float64],
    counterfactual: npt.NDArray[np.float64],
    subset_power_edges: Mapping[int, Sequence[float]],
    min_records: float,
    hours_per_record: float,
    draw: object,
) -> pd.DataFrame:
    """Return each subset's own power shape beside the whole period's on the same bins, both unshifted."""
    columns = [
        "axis",
        "subset",
        "n_bins",
        "power_bin",
        "p50_uplift",
        "sigma_sampling",
        "overall_uplift",
        "overall_sigma",
        "z",
        "hours",
        "too_thin",
        "flagged",
    ]
    rows: list[dict[str, object]] = []
    for axis, (values, edges) in subsets.items():
        codes = bin_codes(values, edges)
        for s, label in enumerate(bin_labels(edges)):
            member = codes == s
            choice = _subset_edges(member, counterfactual, subset_power_edges, min_records)
            if choice is None:
                rows.append(
                    {
                        "axis": axis,
                        "subset": label,
                        "hours": float(member.sum() * hours_per_record),
                        "too_thin": True,
                        "flagged": False,
                    }
                )
                continue
            n_bins, power_edges = choice
            rows += _subset_rows(
                axis,
                label,
                member=member,
                actual=actual,
                counterfactual=counterfactual,
                power_edges=power_edges,
                n_bins=n_bins,
                hours_per_record=hours_per_record,
                draw=draw,
            )
    return pd.DataFrame(rows, columns=columns)


def _subset_edges(
    member: npt.NDArray[np.bool_],
    counterfactual: npt.NDArray[np.float64],
    subset_power_edges: Mapping[int, Sequence[float]],
    min_records: float,
) -> tuple[int, list[float]] | None:
    """Return the most bins whose every bin holds ``min_records`` of the subset, or ``None``."""
    for n_bins in sorted(subset_power_edges, reverse=True):
        edges = list(subset_power_edges[n_bins])
        codes = bin_codes(counterfactual[member], edges)
        counts = np.bincount(codes[codes >= 0], minlength=len(edges) - 1)
        if (counts >= min_records).all():
            return n_bins, edges
    return None


def _subset_rows(
    axis: str,
    label: str,
    *,
    member: npt.NDArray[np.bool_],
    actual: npt.NDArray[np.float64],
    counterfactual: npt.NDArray[np.float64],
    power_edges: list[float],
    n_bins: int,
    hours_per_record: float,
    draw: object,
) -> list[dict[str, object]]:
    """Return the subset's unshifted power shape beside the whole period's on the same bins."""
    codes = bin_codes(counterfactual, power_edges)
    sub_codes = np.where(member, codes, -1)

    def estimate(weights: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        sub = _ratio(_sums(weights, sub_codes, actual, n_bins), _sums(weights, sub_codes, counterfactual, n_bins))
        whole = _ratio(_sums(weights, codes, actual, n_bins), _sums(weights, codes, counterfactual, n_bins))
        return {"subset": sub, "overall": whole, "difference": sub - whole}

    point = estimate(np.ones((1, len(actual))))
    draws = draw(estimate)  # type: ignore[operator]
    sigma = {k: _spread(v) if draws is not None else np.full(n_bins, np.nan) for k, v in (draws or point).items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        z = point["difference"][0] / sigma["difference"]
    flagged = bool(np.any(np.abs(z[np.isfinite(z)]) > FLAG_Z))
    hours = np.bincount(sub_codes[sub_codes >= 0], minlength=n_bins) * hours_per_record
    return [
        {
            "axis": axis,
            "subset": label,
            "n_bins": n_bins,
            "power_bin": bin_label,
            "p50_uplift": float(point["subset"][0][b]),
            "sigma_sampling": float(sigma["subset"][b]),
            "overall_uplift": float(point["overall"][0][b]),
            "overall_sigma": float(sigma["overall"][b]),
            "z": float(z[b]),
            "hours": float(hours[b]),
            "too_thin": False,
            "flagged": flagged,
        }
        for b, bin_label in enumerate(bin_labels(power_edges))
    ]


def _ratio(numerator: npt.NDArray[np.float64], denominator: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return ``numerator / denominator - 1``, NaN where the denominator is not positive."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator - 1.0, np.nan)
