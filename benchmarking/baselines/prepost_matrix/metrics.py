"""Reduce the matrix's cell records to the baseline tables, check for leaks, and compare baselines.

Every uplift is a fraction, as in the cell records. The tables:

- ``groups``: per (arm, site, K, L, multiplier), the test turbines' and the farm's ``bias``,
  ``spread`` and ``score``, and the reference readings' ``mean``, ``median``, ``sd``, ``worst_abs``
  and ``n``. Site ``all`` pools every site.
- ``linearity``: per (arm, site, K, L), the mean and sd over campaigns and test turbines of the line
  ``estimate = line_scale * truth + line_bias`` fitted across a campaign's multipliers.
- ``real``: per (K, L), the real campaign's reference metrics and T13's own estimate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from benchmarking.harness.metrics import summarize_errors

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

POOLED_SITE = "all"
GROUP_KEYS = ["arm", "site", "k", "post_months", "multiplier"]
LINE_KEYS = ["arm", "site", "k", "post_months"]
REAL_KEYS = ["k", "post_months"]
TABLE_KEYS = {"groups": GROUP_KEYS, "linearity": LINE_KEYS, "real": REAL_KEYS}
# A value that moves by more than this between baselines is flagged: 0.1 pp on an uplift.
MATERIAL = 0.001
_CELL_KEYS = ["arm", "site", "seed_index", "k", "post_months", "multiplier"]
_MIN_LINE_POINTS = 2
_REFERENCE_METRICS = ["ref_mean", "ref_median", "ref_sd", "ref_worst_abs", "ref_n"]


def cell_frames(records: Iterable[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Flatten ok cell records into ``turbines``, ``farm`` and ``references`` frames."""
    turbines: list[dict[str, Any]] = []
    farm: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "ok":
            continue
        keys = {k: record[k] for k in _CELL_KEYS}
        turbines.extend({**keys, **row} for row in record["turbines"])
        farm.append({**keys, **record["farm"]})
        references.extend({**keys, **row} for row in record["references"])
    return {
        "turbines": pd.DataFrame(turbines, columns=[*_CELL_KEYS, "turbine", "estimate", "truth"]),
        "farm": pd.DataFrame(farm, columns=[*_CELL_KEYS, "estimate", "truth"]),
        "references": pd.DataFrame(references, columns=[*_CELL_KEYS, "test_wtg", "turbine", "uplift", "screened"]),
    }


def baseline_tables(records: Iterable[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Return the ``groups``, ``linearity`` and ``real`` tables of ``records``."""
    frames = cell_frames(records)
    synthetic = {name: frame[frame["arm"] != "real"] for name, frame in frames.items()}
    real = {name: frame[frame["arm"] == "real"] for name, frame in frames.items()}
    return {
        "groups": _groups(synthetic),
        "linearity": _linearity(synthetic["turbines"]),
        "real": _real(real),
    }


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
    fits = []
    for key, group in turbines.groupby(["arm", "site", "seed_index", "k", "post_months", "turbine"]):
        scale, bias = line_fit(group["truth"], group["estimate"])
        fits.append(
            {
                **dict(zip(["arm", "site", "seed_index", "k", "post_months"], key[:5], strict=True)),
                "line_scale": scale,
                "line_bias": bias,
            }
        )
    if not fits:
        return pd.DataFrame(columns=[*LINE_KEYS, "line_scale_mean", "line_scale_sd", "line_bias_mean", "line_bias_sd"])
    rows = []
    for key, group in _with_pooled(pd.DataFrame(fits)).groupby(LINE_KEYS):
        finite = group.dropna(subset=["line_scale", "line_bias"])
        rows.append(
            {
                **dict(zip(LINE_KEYS, key, strict=True)),
                "line_scale_mean": float(finite["line_scale"].mean()),
                "line_scale_sd": float(finite["line_scale"].std(ddof=0)),
                "line_bias_mean": float(finite["line_bias"].mean()),
                "line_bias_sd": float(finite["line_bias"].std(ddof=0)),
                "line_n": len(finite),
            }
        )
    return pd.DataFrame(rows)


def _real(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    turbines = frames["turbines"]
    for key, group in frames["references"].groupby(REAL_KEYS):
        estimate = turbines[(turbines["k"] == key[0]) & (turbines["post_months"] == key[1])]["estimate"]
        rows.append(
            {
                **dict(zip(REAL_KEYS, key, strict=True)),
                **reference_metrics(group["uplift"].to_numpy(dtype=float)),
                "t13_estimate": float(estimate.iloc[0]) if len(estimate) else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=[*REAL_KEYS, *_REFERENCE_METRICS, "t13_estimate"])


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
