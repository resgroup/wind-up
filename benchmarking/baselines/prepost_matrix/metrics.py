"""Reduce the matrix's cell records to the baseline tables, check for leaks, and compare baselines.

Every uplift is a fraction, as in the cell records. The tables:

- ``groups``: per (arm, site, K, L, multiplier), the test turbines' and the farm's ``bias``,
  ``spread`` and ``score``, the reference readings' ``mean``, ``median``, ``sd``, ``worst_abs``
  and ``n``, the combined reference uplift before and after the screen as errors against 0 %, and
  the calibration of every sigma. Where the cells recorded the reverse direction, ``rev_test_*`` and
  ``rev_ref_*`` repeat the test and reference metrics for the full reversal (method step 13 option B
  at lambda 1). Site ``all`` pools every site.
- ``linearity``: per (arm, site, K, L), the mean and sd over campaigns and test turbines of the line
  ``estimate = line_scale * truth + line_bias`` fitted across a campaign's multipliers, and
  ``rev_line_*`` for the full reversal.
- ``real``: per (K, L), the real campaign's reference metrics and T13's own estimate.
- ``bins``: per group, condition and bin, the per-bin errors and the calibration of their sigma.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from benchmarking.baselines.power_model.method import REVERSAL_COLUMNS, is_reading_column
from benchmarking.harness.calibration import calibration_summary
from benchmarking.harness.metrics import summarize_errors

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

POOLED_SITE = "all"
GROUP_KEYS = ["arm", "site", "k", "post_months", "multiplier"]
LINE_KEYS = ["arm", "site", "k", "post_months"]
REAL_KEYS = ["k", "post_months"]
BIN_KEYS = [*GROUP_KEYS, "condition", "condition_bin"]
TABLE_KEYS = {"groups": GROUP_KEYS, "linearity": LINE_KEYS, "real": REAL_KEYS, "bins": BIN_KEYS}
# A value that moves by more than this between baselines is flagged: 0.1 pp on an uplift.
MATERIAL = 0.001
_CELL_KEYS = ["arm", "site", "seed_index", "k", "post_months", "multiplier"]
_MIN_LINE_POINTS = 2
_REFERENCE_METRICS = ["ref_mean", "ref_median", "ref_sd", "ref_worst_abs", "ref_n"]
_COMBINED_REFERENCES = ("reference_uplift_before_screen", "reference_uplift_after_screen", "reference_uplift_spread")


def cell_frames(records: Iterable[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Flatten ok cell records into ``turbines``, ``farm``, ``references`` and ``bins`` frames."""
    turbines: list[dict[str, Any]] = []
    farm: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    bins: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "ok":
            continue
        keys = {k: record[k] for k in _CELL_KEYS}
        turbines.extend({**keys, **row} for row in record["turbines"])
        farm.append({**keys, **record["farm"]})
        references.extend({**keys, **row} for row in record["references"])
        bins.extend({**keys, **row} for row in record.get("bins", []))
    return {
        "turbines": _with_reversal(
            pd.DataFrame(turbines, columns=[*_CELL_KEYS, "turbine", "estimate", "truth", *_reading_keys(turbines)])
        ),
        "farm": pd.DataFrame(
            farm, columns=[*_CELL_KEYS, "estimate", "truth", *(k for k in _COMBINED_REFERENCES if _has(farm, k))]
        ),
        "references": _with_reversal(
            pd.DataFrame(
                references,
                columns=[*_CELL_KEYS, "test_wtg", "turbine", "uplift", "screened", *_reading_keys(references)],
            )
        ),
        "bins": pd.DataFrame(
            bins, columns=[*_CELL_KEYS, "test_wtg", "condition", "condition_bin", "estimate", "truth", "sigma"]
        ),
    }


def _reading_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Return the reading columns the rows carry: ``sigma`` first, then the sweep and the reverse direction."""
    keys = {k for row in rows for k in row if is_reading_column(k)}
    return ["sigma", *sorted(k for k in keys if k != "sigma")] if keys else []


def _has(rows: list[dict[str, Any]], key: str) -> bool:
    return any(key in row for row in rows)


def reversal_uplift(forward: np.ndarray | pd.Series, reverse: np.ndarray | pd.Series) -> np.ndarray:
    """Return the full reversal ``sqrt((1 + forward) / (1 + reverse)) - 1``; NaN where either ratio is not positive."""
    a = 1.0 + np.asarray(forward, dtype=float)
    b = 1.0 + np.asarray(reverse, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(a / b) - 1.0
    return np.where((a > 0) & (b > 0), out, np.nan)


def _with_reversal(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with ``reversal``, the full-reversal uplift, where it carries both directions."""
    if not set(REVERSAL_COLUMNS[:2]) <= set(frame.columns):
        return frame
    return frame.assign(reversal=reversal_uplift(frame["uplift_forward"], frame["uplift_reverse"]))


def calibration_metrics(errors: np.ndarray, sigmas: np.ndarray, *, prefix: str) -> dict[str, float]:
    """Return ``<prefix>_coverage`` (fraction within one sigma), ``<prefix>_z_sd`` and ``<prefix>_sigma_median``."""
    summary = calibration_summary(errors, sigmas)
    finite = np.asarray(sigmas, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    return {
        f"{prefix}_coverage": summary.coverage_1sigma,
        f"{prefix}_z_sd": summary.z_spread,
        f"{prefix}_sigma_median": float(np.median(finite)) if finite.size else np.nan,
    }


def _calibration(turbines: pd.DataFrame, references: pd.DataFrame) -> dict[str, float]:
    """Calibration of every sigma column: test turbines against truth, references against zero."""
    out: dict[str, float] = {}
    for frame, error, prefix in (
        (turbines, (turbines["estimate"] - turbines["truth"]).to_numpy(dtype=float), "test"),
        (references, references["uplift"].to_numpy(dtype=float), "ref"),
    ):
        for column in [c for c in frame.columns if c == "sigma" or str(c).startswith("sigma_block_")]:
            suffix = "" if column == "sigma" else f"_{column.removeprefix('sigma_')}"
            out |= calibration_metrics(error, frame[column].to_numpy(dtype=float), prefix=f"{prefix}{suffix}")
    return out


def baseline_tables(records: Iterable[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Return the ``groups``, ``linearity``, ``real`` and ``bins`` tables of ``records``."""
    frames = cell_frames(records)
    synthetic = {name: frame[frame["arm"] != "real"] for name, frame in frames.items()}
    real = {name: frame[frame["arm"] == "real"] for name, frame in frames.items()}
    return {
        "groups": _groups(synthetic),
        "linearity": _linearity(synthetic["turbines"]),
        "real": _real(real),
        "bins": _bins(synthetic["bins"]),
    }


def _bins(bins: pd.DataFrame) -> pd.DataFrame:
    """Per group and condition bin: the error metrics of the per-bin estimates and the calibration of their sigma."""
    rows = []
    for key, group in _with_pooled(bins).groupby(BIN_KEYS, dropna=False):
        errors = (group["estimate"] - group["truth"]).to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(BIN_KEYS, key, strict=True)),
                **_error_metrics(errors, prefix="bin"),
                **calibration_metrics(errors, group["sigma"].to_numpy(dtype=float), prefix="bin"),
            }
        )
    columns = [
        *BIN_KEYS,
        "bin_bias",
        "bin_spread",
        "bin_score",
        "bin_n",
        "bin_coverage",
        "bin_z_sd",
        "bin_sigma_median",
    ]
    return pd.DataFrame(rows, columns=columns)


def _with_pooled(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([frame, frame.assign(site=POOLED_SITE)], ignore_index=True)


def _error_metrics(errors: np.ndarray, *, prefix: str) -> dict[str, float]:
    summary = summarize_errors(errors)
    return {
        f"{prefix}_bias": summary.bias,
        f"{prefix}_spread": summary.spread,
        f"{prefix}_score": summary.score,
        f"{prefix}_n": summary.n,
    }


def reference_metrics(uplifts: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Return ``ref_mean``, ``ref_median``, ``ref_sd`` (population), ``ref_worst_abs`` and ``ref_n``."""
    values = np.asarray(uplifts, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"ref_mean": np.nan, "ref_median": np.nan, "ref_sd": np.nan, "ref_worst_abs": np.nan, "ref_n": 0}
    return {
        "ref_mean": float(values.mean()),
        "ref_median": float(np.median(values)),
        "ref_sd": float(values.std(ddof=0)),
        "ref_worst_abs": float(np.abs(values).max()),
        "ref_n": int(values.size),
    }


def _groups(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    grouped = {name: _with_pooled(frame).groupby(GROUP_KEYS, dropna=False) for name, frame in frames.items()}
    keys = sorted(set().union(*(g.groups for g in grouped.values())), key=str)
    rows = []
    for key in keys:
        row: dict[str, Any] = dict(zip(GROUP_KEYS, key, strict=True))
        turbines = _group(grouped["turbines"], key)
        farm = _group(grouped["farm"], key)
        references = _group(grouped["references"], key)
        row |= _error_metrics((turbines["estimate"] - turbines["truth"]).to_numpy(dtype=float), prefix="test")
        row |= _error_metrics((farm["estimate"] - farm["truth"]).to_numpy(dtype=float), prefix="farm")
        row |= reference_metrics(references["uplift"].to_numpy(dtype=float))
        for column, prefix in (
            ("reference_uplift_before_screen", "combined_ref_before"),
            ("reference_uplift_after_screen", "combined_ref_after"),
        ):
            if column in farm.columns:
                row |= _error_metrics(farm[column].to_numpy(dtype=float), prefix=prefix)
        if "reversal" in turbines.columns:
            row |= _error_metrics((turbines["reversal"] - turbines["truth"]).to_numpy(dtype=float), prefix="rev_test")
        if "reversal" in references.columns:
            row |= {f"rev_{k}": v for k, v in reference_metrics(references["reversal"].to_numpy(dtype=float)).items()}
        row |= _calibration(turbines, references)
        rows.append(row)
    return pd.DataFrame(rows)


def _group(grouped: Any, key: tuple) -> pd.DataFrame:  # noqa: ANN401 - a pandas GroupBy
    return grouped.get_group(key) if key in grouped.groups else grouped.obj.iloc[:0]


def line_fit(truth: Sequence[float] | np.ndarray, estimate: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Return ``(line_scale, line_bias)`` of ``estimate = line_scale * truth + line_bias``.

    NaN when fewer than two finite points remain or the truths do not vary.
    """
    t, e = np.asarray(truth, dtype=float), np.asarray(estimate, dtype=float)
    finite = np.isfinite(t) & np.isfinite(e)
    t, e = t[finite], e[finite]
    if t.size < _MIN_LINE_POINTS or np.ptp(t) == 0:
        return float("nan"), float("nan")
    scale, bias = np.polyfit(t, e, 1)
    return float(scale), float(bias)


def _linearity(turbines: pd.DataFrame) -> pd.DataFrame:
    estimates = {"line": "estimate", **({"rev_line": "reversal"} if "reversal" in turbines.columns else {})}
    fits = []
    for key, group in turbines.groupby(["arm", "site", "seed_index", "k", "post_months", "turbine"]):
        fit: dict[str, Any] = dict(zip(["arm", "site", "seed_index", "k", "post_months"], key[:5], strict=True))
        for prefix, column in estimates.items():
            fit[f"{prefix}_scale"], fit[f"{prefix}_bias"] = line_fit(group["truth"], group[column])
        fits.append(fit)
    if not fits:
        return pd.DataFrame(columns=[*LINE_KEYS, "line_scale_mean", "line_scale_sd", "line_bias_mean", "line_bias_sd"])
    rows = []
    for key, group in _with_pooled(pd.DataFrame(fits)).groupby(LINE_KEYS):
        row: dict[str, Any] = dict(zip(LINE_KEYS, key, strict=True))
        for prefix in estimates:
            finite = group.dropna(subset=[f"{prefix}_scale", f"{prefix}_bias"])
            row |= {
                f"{prefix}_scale_mean": float(finite[f"{prefix}_scale"].mean()),
                f"{prefix}_scale_sd": float(finite[f"{prefix}_scale"].std(ddof=0)),
                f"{prefix}_bias_mean": float(finite[f"{prefix}_bias"].mean()),
                f"{prefix}_bias_sd": float(finite[f"{prefix}_bias"].std(ddof=0)),
                f"{prefix}_n": len(finite),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _real(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    turbines = frames["turbines"]
    for key, group in frames["references"].groupby(REAL_KEYS):
        t13 = turbines[(turbines["k"] == key[0]) & (turbines["post_months"] == key[1])]
        estimate = t13["estimate"]
        rows.append(
            {
                **dict(zip(REAL_KEYS, key, strict=True)),
                **reference_metrics(group["uplift"].to_numpy(dtype=float)),
                "t13_estimate": float(estimate.iloc[0]) if len(estimate) else float("nan"),
                "t13_sigma": _first(t13, "sigma"),
                "t13_reversal": _first(t13, "reversal"),
                **(
                    {f"rev_{k}": v for k, v in reference_metrics(group["reversal"].to_numpy(dtype=float)).items()}
                    if "reversal" in group.columns
                    else {}
                ),
                **{k: v for k, v in _calibration(t13, group).items() if k.startswith("ref")},
            }
        )
    columns = [*REAL_KEYS, *_REFERENCE_METRICS, "t13_estimate", "t13_sigma"]
    return pd.DataFrame(rows, columns=columns + [c for c in (rows[0] if rows else {}) if c not in columns])


def _first(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].iloc[0]) if len(frame) and column in frame.columns else float("nan")


def leak_check(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Return, per reference reading, how far it moves across one campaign's multipliers.

    One row per (arm, site, seed index, K, L, test turbine, reference) read under more than one
    multiplier, largest ``leak`` first. The upgrade should not move a reference's reading.
    """
    references = cell_frames(records)["references"]
    references = references[references["arm"] != "real"]
    keys = ["arm", "site", "seed_index", "k", "post_months", "test_wtg", "turbine"]
    grouped = references.groupby(keys)["uplift"]
    table = pd.DataFrame({"leak": grouped.max() - grouped.min(), "n_multipliers": grouped.count()}).reset_index()
    table = table[table["n_multipliers"] > 1].astype({"seed_index": "Int64"})
    return table.sort_values("leak", ascending=False, kind="stable").reset_index(drop=True)


def compare_tables(
    candidate: dict[str, pd.DataFrame], baseline: dict[str, pd.DataFrame], *, material: float = MATERIAL
) -> pd.DataFrame:
    """Return one row per table, key and metric: ``baseline``, ``candidate``, ``delta`` and ``moved``.

    A value moves when it changes by more than ``material``, when a count changes at all, or when
    it is present on one side only.
    """
    rows = []
    for name, keys in TABLE_KEYS.items():
        new, old = candidate.get(name, pd.DataFrame(columns=keys)), baseline.get(name, pd.DataFrame(columns=keys))
        metrics = sorted((set(new.columns) | set(old.columns)) - set(keys))
        merged = old.astype(dict.fromkeys(keys, object)).merge(
            new.astype(dict.fromkeys(keys, object)),
            on=keys,
            how="outer",
            suffixes=("__old", "__new"),
        )
        for metric in metrics:
            before = pd.to_numeric(merged.get(f"{metric}__old", pd.Series(np.nan, index=merged.index)))
            after = pd.to_numeric(merged.get(f"{metric}__new", pd.Series(np.nan, index=merged.index)))
            delta = after - before
            one_sided = before.isna() != after.isna()
            threshold = 0 if metric.endswith("_n") else material
            moved = one_sided | (delta.abs() > threshold)
            rows.append(
                merged[keys].assign(
                    table=name, metric=metric, baseline=before, candidate=after, delta=delta, moved=moved
                )
            )
    if not rows:
        return pd.DataFrame(columns=["table", "metric", "baseline", "candidate", "delta", "moved"])
    return pd.concat(rows, ignore_index=True)
