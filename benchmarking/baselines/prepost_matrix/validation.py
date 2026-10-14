"""Judge a prepost matrix study against the validation targets of ``docs/v1/method.md``.

Reads the study's cell records, so it runs on a partial study too. Each target gets a table and a
one-line verdict; :func:`write_report` saves the tables as CSV beside the study and the text as
``validation.md``.

- **i** the combined reference uplift (before and after the screen) and the placebo test turbines
  read 0 % within their standard error;
- **ii** the line of measured against true uplift has slope 1 and intercept 0;
- **iii** the error does not depend on the after-period length, the number of references or the
  season of the changeover;
- **iv** the sigma covers the error at the 68 % it claims, for the headline, the references and the
  bins, at every recorded block length;
- **option 2** the forward headline against the full reversal (lambda 1) and against the computed
  lambda of the reversal spec, here from the recorded fixed-model bootstrap sigmas, which leave out the
  fitting error and so overstate the reverse direction's precision on short after periods;
- **open questions** the error against the share of after-period hours the waking configuration rule
  removes (step 11 ii), and the reference readings against the data screen's measures (step 9).

Every uplift is a fraction; the text prints percentage points.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from benchmarking.baselines.prepost_matrix.metrics import line_fit, reversal_uplift

if TYPE_CHECKING:
    from pathlib import Path

# The share of a normal distribution within one sigma.
TARGET_COVERAGE = 0.6827
# A coverage within this of the target, or within two binomial standard errors, is calibrated.
COVERAGE_TOLERANCE = 0.05
# A mean further than this many standard errors from its target fails.
Z_LIMIT = 2.0
# Below this many values a verdict is not given.
MIN_N = 5
ESTIMATORS = ("forward", "reversal", "lambda")
RARE_CONFIGURATION = "rare waking configuration"
USABLE = "usable"
_SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA"}
_PP = 100.0


def study_records(study_dir: Path) -> list[dict[str, Any]]:
    """Return the ok cell records of ``study_dir``."""
    records = [json.loads(path.read_text()) for path in sorted((study_dir / "cells").glob("*.json"))]
    return [r for r in records if r.get("status") == "ok"]


def season(timestamp: str | pd.Timestamp) -> str:
    """Return the meteorological season (DJF, MAM, JJA, SON) of ``timestamp``."""
    return _SEASONS.get(pd.Timestamp(timestamp).month, "SON")


def lambda_star(
    forward: np.ndarray | pd.Series,
    reverse: np.ndarray | pd.Series,
    *,
    sigma_forward: np.ndarray | pd.Series,
    sigma_reverse: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return the reversal spec's MSE-optimal lambda in [0, 1], the two directions taken as independent.

    In log space ``f = ln(1 + forward)``, ``g = ln(1 + reverse)``, the bias estimate ``(f + g) / 2``
    and its square shrunk by its variance (positive-part James-Stein). Each sigma is carried into
    log space by dividing by its ratio. NaN where an input is not finite.
    """
    f = np.log1p(np.asarray(forward, dtype=float))
    g = np.log1p(np.asarray(reverse, dtype=float))
    var_f = (np.asarray(sigma_forward, dtype=float) / (1.0 + np.asarray(forward, dtype=float))) ** 2
    var_g = (np.asarray(sigma_reverse, dtype=float) / (1.0 + np.asarray(reverse, dtype=float))) ** 2
    beta_sq = np.maximum(((f + g) / 2.0) ** 2 - (var_f + var_g) / 4.0, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lam = (2.0 * beta_sq + var_f) / (2.0 * beta_sq + 0.5 * var_f + 0.5 * var_g)
    return np.clip(lam, 0.0, 1.0)


def lambda_uplift(forward: np.ndarray | pd.Series, reverse: np.ndarray | pd.Series, lam: np.ndarray) -> np.ndarray:
    """Return ``exp(f - lam * (f + g) / 2) - 1``, the uplift at ``lam`` between forward (0) and full reversal (1)."""
    f = np.log1p(np.asarray(forward, dtype=float))
    g = np.log1p(np.asarray(reverse, dtype=float))
    return np.expm1(f - lam * (f + g) / 2.0)


def _keys(record: dict[str, Any]) -> dict[str, Any]:
    return {k: record[k] for k in ("cell_id", "arm", "site", "seed_index", "k", "post_months", "multiplier")}


def turbine_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per measured test turbine: its readings, truth, period, season and hours lost."""
    rows = []
    for record in records:
        for turbine in record["turbines"]:
            wtg = turbine["turbine"]
            plan = record.get("plans", {}).get(wtg, {})
            losses = {r["reason"]: r for r in record.get("record_losses", {}).get(wtg, [])}
            rare = (losses.get(RARE_CONFIGURATION) or {}).get("hours_after")
            usable = (losses.get(USABLE) or {}).get("hours_after")
            rows.append(
                {
                    **_keys(record),
                    **turbine,
                    "truth": np.nan if turbine.get("truth") is None else turbine["truth"],
                    "season": season(plan["works"][1]) if "works" in plan else None,
                    "post_days": plan.get("post_days", np.nan),
                    "rare_share_after": rare / (rare + usable) if rare is not None and usable else np.nan,
                }
            )
    return _with_estimators(pd.DataFrame(rows))


def reference_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per reference reading, with the data screen's measures of that candidate where recorded."""
    rows = []
    for record in records:
        screens = record.get("data_screen", {})
        for reading in record["references"]:
            candidates = {c["candidate"]: c for c in screens.get(reading["test_wtg"], {}).get("candidates", [])}
            screen = candidates.get(reading["turbine"], {})
            rows.append(
                {
                    **_keys(record),
                    **reading,
                    "correlation_margin": _minus(screen.get("correlation"), screen.get("median_correlation")),
                    "hours_fraction": _hours_fraction(screen),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return _with_estimators(frame.rename(columns={"uplift": "estimate"}).assign(truth=0.0))


def _minus(a: float | None, b: float | None) -> float:
    return np.nan if a is None or b is None else a - b


def _hours_fraction(screen: dict[str, Any]) -> float:
    """Return the smaller of a candidate's usable hours over the median, before and after."""
    fractions = [
        screen[f"hours_{side}"] / screen[f"median_hours_{side}"]
        for side in ("before", "after")
        if screen.get(f"hours_{side}") is not None and screen.get(f"median_hours_{side}")
    ]
    return min(fractions) if fractions else np.nan


def _with_estimators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``forward``, ``reversal``, ``lambda`` and ``lambda_star`` where both directions are recorded."""
    if frame.empty:
        return frame
    frame = frame.assign(forward=frame["estimate"])
    if not {"uplift_forward", "uplift_reverse", "sigma_reverse"} <= set(frame.columns):
        return frame.assign(reversal=np.nan, lambda_star=np.nan, **{"lambda": np.nan})
    lam = lambda_star(
        frame["uplift_forward"],
        frame["uplift_reverse"],
        sigma_forward=frame["sigma"],
        sigma_reverse=frame["sigma_reverse"],
    )
    return frame.assign(
        reversal=reversal_uplift(frame["uplift_forward"], frame["uplift_reverse"]),
        lambda_star=lam,
        **{"lambda": lambda_uplift(frame["uplift_forward"], frame["uplift_reverse"], lam)},
    )


def _mean_se(values: pd.Series | np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    mean = float(v.mean()) if n else np.nan
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return {"mean": mean, "se": se, "z": mean / se if se and np.isfinite(se) and se > 0 else np.nan, "n": n}


def _errors(frame: pd.DataFrame, estimator: str) -> pd.Series:
    return frame[estimator] - frame["truth"]


def _site_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [("all", frame), *((str(site), group) for site, group in frame.groupby("site"))]


def references_table(records: list[dict[str, Any]], turbines: pd.DataFrame) -> pd.DataFrame:
    """Target i: the combined reference uplift and the placebo test turbines, each against 0 %."""
    farms = pd.DataFrame([{**_keys(r), **r["farm"]} for r in records])
    columns = [c for c in ("reference_uplift_before_screen", "reference_uplift_after_screen") if c in farms.columns]
    rows = [
        {"site": site, "quantity": column, **_mean_se(group[column])}
        for site, group in _site_groups(farms)
        for column in columns
    ]
    placebo = turbines[(turbines["multiplier"] == 0) & (turbines["arm"] != "real")]
    rows += [
        {"site": site, "quantity": f"placebo_{estimator}", **_mean_se(_errors(group, estimator))}
        for site, group in _site_groups(placebo)
        for estimator in ESTIMATORS
    ]
    return pd.DataFrame(rows)


def linearity_table(turbines: pd.DataFrame) -> pd.DataFrame:
    """Target ii: the slope and intercept of measured against true uplift, per campaign turbine, then pooled."""
    synthetic = turbines[turbines["arm"] != "real"]
    rows = []
    for estimator in ESTIMATORS:
        fits = []
        for key, group in synthetic.groupby(["site", "seed_index", "arm", "k", "post_months", "turbine"]):
            scale, bias = line_fit(group["truth"], group[estimator])
            fits.append({"site": key[0], "scale": scale, "bias": bias})
        frame = pd.DataFrame(fits, columns=["site", "scale", "bias"])
        for site, group in _site_groups(frame):
            slope, intercept = _mean_se(group["scale"] - 1.0), _mean_se(group["bias"])
            rows.append(
                {
                    "estimator": estimator,
                    "site": site,
                    "slope": 1.0 + slope["mean"],
                    "slope_se": slope["se"],
                    "slope_z": slope["z"],
                    "intercept": intercept["mean"],
                    "intercept_se": intercept["se"],
                    "intercept_z": intercept["z"],
                    "slope_sd": float(group["scale"].std(ddof=0)) if len(group) else np.nan,
                    "n": slope["n"],
                }
            )
    return pd.DataFrame(rows)


def dependence_table(turbines: pd.DataFrame) -> pd.DataFrame:
    """Target iii: the synthetic error by after-period length, number of references and season."""
    synthetic = turbines[turbines["arm"] == "main"]
    rows = []
    for axis in ("post_months", "k", "season"):
        for value, group in synthetic.groupby(axis):
            for estimator in ESTIMATORS:
                errors = _errors(group, estimator).to_numpy(dtype=float)
                errors = errors[np.isfinite(errors)]
                rows.append(
                    {
                        "axis": axis,
                        "value": str(value),
                        "estimator": estimator,
                        **_mean_se(errors),
                        "rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _coverage(errors: np.ndarray, sigmas: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(errors) & np.isfinite(sigmas) & (sigmas > 0)
    z = errors[ok] / sigmas[ok]
    return {
        "coverage": float(np.mean(np.abs(z) <= 1.0)) if z.size else np.nan,
        "z_sd": float(np.std(z)) if z.size else np.nan,
        "sigma_median": float(np.median(sigmas[ok])) if z.size else np.nan,
        "n": int(z.size),
    }


def coverage_table(records: list[dict[str, Any]], turbines: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    """Target iv: how often the error falls within one sigma, per quantity and recorded block length."""
    rows = []
    for quantity, frame in (("test", turbines[turbines["arm"] != "real"]), ("reference", references)):
        if frame.empty:
            continue
        for column in [c for c in frame.columns if c == "sigma" or str(c).startswith("sigma_block_")]:
            errors = _errors(frame, "forward").to_numpy(dtype=float)
            rows.append(
                {"quantity": quantity, "block": column, **_coverage(errors, frame[column].to_numpy(dtype=float))}
            )
    bins = pd.DataFrame(
        [{**_keys(r), **b} for r in records if r["arm"] != "real" for b in r.get("bins", [])],
        columns=["condition", "estimate", "truth", "sigma"],
    )
    for condition, group in bins.groupby("condition"):
        errors = group["estimate"].to_numpy(dtype=float) - pd.to_numeric(group["truth"]).to_numpy(dtype=float)
        rows.append(
            {
                "quantity": f"bins_{condition}",
                "block": "sigma",
                **_coverage(errors, group["sigma"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows)


def reversal_table(turbines: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    """Option 2: forward, full reversal and computed lambda, per after-period length, on test and reference error."""
    rows = []
    for quantity, frame in (("test", turbines[turbines["arm"] != "real"]), ("reference", references)):
        if frame.empty:
            continue
        for length, group in frame.groupby("post_months"):
            row: dict[str, Any] = {
                "quantity": quantity,
                "post_months": int(length),
                "lambda_star_median": float(np.nanmedian(group["lambda_star"]))
                if group["lambda_star"].notna().any()
                else np.nan,
            }
            for estimator in ESTIMATORS:
                errors = _errors(group, estimator).to_numpy(dtype=float)
                errors = errors[np.isfinite(errors)]
                row[f"{estimator}_bias"] = float(errors.mean()) if errors.size else np.nan
                row[f"{estimator}_rmse"] = float(np.sqrt(np.mean(errors**2))) if errors.size else np.nan
            row["n"] = len(group)
            rows.append(row)
    return pd.DataFrame(rows)


def configuration_rule_table(turbines: pd.DataFrame) -> pd.DataFrame:
    """Open question (step 11 ii): the forward error by the share of after-period hours the rule removes, in thirds."""
    frame = turbines[(turbines["arm"] != "real") & turbines["rare_share_after"].notna()]
    rows = []
    for interval, group in frame.groupby(_thirds(frame["rare_share_after"])):
        errors = _errors(group, "forward").to_numpy(dtype=float)
        rows.append(
            {
                "rare_share": str(interval),
                "share_median": float(group["rare_share_after"].median()),
                **_mean_se(errors),
                "rmse": float(np.sqrt(np.nanmean(errors**2))),
            }
        )
    return pd.DataFrame(rows, columns=["rare_share", "share_median", "mean", "se", "z", "n", "rmse"])


def screen_table(references: pd.DataFrame) -> pd.DataFrame:
    """Open question (step 9): the synthetic reference readings by the data screen's two measures, in thirds."""
    frame = references[references["arm"] != "real"] if not references.empty else references
    rows = []
    for measure in ("correlation_margin", "hours_fraction"):
        if frame.empty:
            continue
        known = frame[frame[measure].notna()]
        for interval, group in known.groupby(_thirds(known[measure])):
            readings = group["forward"].to_numpy(dtype=float)
            readings = readings[np.isfinite(readings)]
            rows.append(
                {
                    "measure": measure,
                    "range": str(interval),
                    "rms_reading": float(np.sqrt(np.mean(readings**2))) if readings.size else np.nan,
                    "worst_abs": float(np.abs(readings).max()) if readings.size else np.nan,
                    "n": int(readings.size),
                }
            )
    return pd.DataFrame(rows, columns=["measure", "range", "rms_reading", "worst_abs", "n"])


def _thirds(values: pd.Series) -> pd.Series:
    """Label ``values`` by third; by value itself when it takes fewer than three distinct values."""
    if values.nunique() < 3:  # noqa: PLR2004 - three thirds
        return values.astype(str)
    return pd.qcut(values, 3, duplicates="drop").astype(str)


def validation_tables(records: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Return every target's table for ``records``."""
    turbines = turbine_frame(records)
    references = reference_frame(records)
    return {
        "i_references": references_table(records, turbines),
        "ii_linearity": linearity_table(turbines),
        "iii_dependence": dependence_table(turbines),
        "iv_coverage": coverage_table(records, turbines, references),
        "option2_reversal": reversal_table(turbines, references),
        "open_configuration_rule": configuration_rule_table(turbines),
        "open_screen": screen_table(references),
    }


def _verdict_mean(row: pd.Series) -> str:
    if row["n"] < MIN_N or not np.isfinite(row["z"]):
        return "too few"
    return "pass" if abs(row["z"]) <= Z_LIMIT else "FAIL"


def _verdict_coverage(row: pd.Series) -> str:
    """Pass within the tolerance or two binomial standard errors of the target, whichever is wider."""
    if row["n"] < MIN_N or not np.isfinite(row["coverage"]):
        return "too few"
    gap = row["coverage"] - TARGET_COVERAGE
    binomial_se = np.sqrt(TARGET_COVERAGE * (1.0 - TARGET_COVERAGE) / row["n"])
    if abs(gap) <= max(COVERAGE_TOLERANCE, Z_LIMIT * binomial_se):
        return "pass"
    return "FAIL (sigma too small)" if gap < 0 else "FAIL (sigma too large)"


def describe(tables: dict[str, pd.DataFrame], *, n_cells: int) -> str:
    """Return the report as markdown: one verdict line per check, then each table."""
    lines = [f"# Validation of {n_cells} ok cells", ""]
    refs = tables["i_references"]
    lines.append("## i. References and placebos read 0 %")
    lines += [
        f"- {row.quantity}: {row.mean * _PP:+.3f} ± {row.se * _PP:.3f} pp (n {row.n}): "
        f"{_verdict_mean(pd.Series(row._asdict()))}"
        for row in refs[refs["site"] == "all"].itertuples()
    ]
    lines += ["", "## ii. Measured against true uplift: slope 1, intercept 0"]
    for row in tables["ii_linearity"].query("site == 'all'").itertuples():
        slope_ok = _verdict_mean(pd.Series({"z": row.slope_z, "n": row.n}))
        intercept_ok = _verdict_mean(pd.Series({"z": row.intercept_z, "n": row.n}))
        lines.append(
            f"- {row.estimator}: slope {row.slope:.3f} ± {row.slope_se:.3f} ({slope_ok}), intercept "
            f"{row.intercept * _PP:+.3f} ± {row.intercept_se * _PP:.3f} pp ({intercept_ok}), n {row.n}"
        )
    lines += ["", "## iii. No dependence on after-period length, references or season (forward)"]
    dep = tables["iii_dependence"]
    for axis, group in dep[dep["estimator"] == "forward"].groupby("axis", sort=False):
        spread = group["mean"].max() - group["mean"].min()
        worst = group.loc[group["z"].abs().idxmax()] if group["z"].notna().any() else None
        verdict = "too few" if worst is None else ("pass" if abs(worst["z"]) <= Z_LIMIT else "FAIL")
        detail = (
            "" if worst is None else f"; worst {worst['value']} at {worst['mean'] * _PP:+.3f} pp (z {worst['z']:+.1f})"
        )
        lines.append(f"- {axis}: bias ranges over {spread * _PP:.3f} pp{detail}: {verdict}")
    lines += ["", "## iv. Sigma covers 68 % of the errors"]
    for row in tables["iv_coverage"].itertuples():
        lines.append(
            f"- {row.quantity} {row.block}: {row.coverage:.2f} (z sd {row.z_sd:.2f}, n {row.n}): "
            f"{_verdict_coverage(pd.Series(row._asdict()))}"
        )
    lines += ["", "## Option 2: forward, full reversal and computed lambda (RMSE, pp)"]
    for row in tables["option2_reversal"].itertuples():
        lines.append(
            f"- {row.quantity} L{row.post_months}: forward {row.forward_rmse * _PP:.3f}, reversal "
            f"{row.reversal_rmse * _PP:.3f}, lambda {row.lambda_rmse * _PP:.3f} (median lambda "
            f"{row.lambda_star_median:.2f}, n {row.n})"
        )
    lines += ["", "## Tables", ""]
    for name, table in tables.items():
        lines += [
            f"### {name}",
            "",
            table.to_markdown(index=False, floatfmt=".4g") if not table.empty else "(empty)",
            "",
        ]
    return "\n".join(lines)


def write_report(study_dir: Path) -> str:
    """Write ``validation/*.csv`` and ``validation/validation.md`` into ``study_dir`` and return the text."""
    records = study_records(study_dir)
    tables = validation_tables(records)
    out = study_dir / "validation"
    out.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    text = describe(tables, n_cells=len(records))
    (out / "validation.md").write_text(text)
    return text
