"""Judge a prepost matrix study against the validation targets of ``docs/v1/method.md``.

Reads the study's cell records, so it runs on a partial study too. Each target gets a table and a
one-line verdict; :func:`write_report` saves the tables as CSV beside the study and the text as
``validation.md``.

- **i** the combined reference uplift (before and after the screen) and the placebo test turbines
  read 0 % within their standard error;
- **ii** the line of measured against true uplift has slope 1 and intercept 0;
- **iii** the error does not depend on the after-period length, the number of references or the
  season of the changeover; ``season_within`` compares each turbine's error with its mean over the same
  campaign's season shifts, so it needs a matrix whose campaigns run in more than one season;
- **iv** the sigma covers the error at the 68 % it claims, for the headline, the references and the
  bins, at every recorded block length;
- **option 2** the headline and the forward reading against the full reversal (lambda 1) and against the computed
  lambda of the reversal spec, here from the recorded fixed-model bootstrap sigmas, which leave out the
  fitting error and so overstate the reverse direction's precision on short after periods;
- **open questions** the error against the share of after-period hours the waking configuration rule
  removes (step 11 ii), and the reference readings against the data screen's measures (step 9).

The headline is the reported uplift (the reference-corrected one by default) and ``forward`` the forward
reading it was corrected from. Targets i to iv judge the headline. A reference reading is never corrected,
so for a reference the two are the same.

The targets are judged on the method's intended domain only: cells with at least
``DOMAIN_PRE_MONTHS`` before and ``DOMAIN_POST_MONTHS`` after. The rest of the matrix shows how the method
degrades outside it, in the per-period table of every cell (bias, RMSE and coverage per before and after
length) and the tables after it.

Cells of one campaign share its data, so each mean's standard error is clustered by campaign (a
synthetic site's tile, or a real campaign), and a verdict needs at least ``MIN_N`` campaigns.

Every uplift is a fraction; the text prints percentage points.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from benchmarking.baselines.prepost_matrix.metrics import line_fit, reversal_uplift

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# The share of a normal distribution within one sigma.
TARGET_COVERAGE = 0.6827
# A coverage within this of the target, or within two of its clustered standard errors, is calibrated.
COVERAGE_TOLERANCE = 0.05
# A coverage above the target passes as a wide sigma up to this; above it the sigma is too large to be useful.
COVERAGE_CEILING = 0.90
# A mean further than this many standard errors from its target fails.
Z_LIMIT = 2.0
# Below this many campaigns a verdict is not given.
MIN_N = 5
ESTIMATORS = ("headline", "forward", "reversal", "lambda")
# The intended domain the targets are judged on (method.md, Validation).
DOMAIN_PRE_MONTHS = 12
DOMAIN_POST_MONTHS = 6
_DAYS_PER_MONTH = 365.25 / 12
RARE_CONFIGURATION = "rare waking configuration"
USABLE = "usable"
CHANGED_UPWIND_OF_REFERENCE = "changed upwind of reference"
CHANGED_UPWIND_OF_SUBJECT = "changed upwind of subject"
_SHARE_REASONS = {
    "usable": USABLE,
    "changed_upwind_of_reference": CHANGED_UPWIND_OF_REFERENCE,
    "changed_upwind_of_subject": CHANGED_UPWIND_OF_SUBJECT,
}
# Run 4 (b045d99), before the wake reach: the median usable share of before hours by site and K.
RUN4_USABLE_BEFORE = "0.17 to 0.37"
_SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA"}
_PP = 100.0


def study_records(study_dir: Path, *, status: str = "ok") -> list[dict[str, Any]]:
    """Return the cell records of ``study_dir`` with ``status``: ``"ok"``, or ``"unmeasured"`` for none measured."""
    records = [json.loads(path.read_text()) for path in sorted((study_dir / "cells").glob("*.json"))]
    return [r for r in records if r.get("status") == status]


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


def period_months(record: dict[str, Any], *, side: str) -> float:
    """Return a cell's ``side`` (``"pre"`` or ``"post"``) period length in months.

    The cell's own ``pre_months`` or ``post_months``; for a cell that takes all the data there is,
    the shortest of its plans' periods. NaN without either.
    """
    months = record.get(f"{side}_months")
    if months is not None:
        return float(months)
    days = [plan[f"{side}_days"] for plan in record.get("plans", {}).values() if plan.get(f"{side}_days") is not None]
    return min(days) / _DAYS_PER_MONTH if days else np.nan


def in_domain(record: dict[str, Any]) -> bool:
    """Whether a cell has at least ``DOMAIN_PRE_MONTHS`` before and ``DOMAIN_POST_MONTHS`` after."""
    return bool(
        period_months(record, side="pre") >= DOMAIN_PRE_MONTHS
        and period_months(record, side="post") >= DOMAIN_POST_MONTHS
    )


def _keys(record: dict[str, Any]) -> dict[str, Any]:
    keys = {
        k: record[k] for k in ("cell_id", "arm", "site", "seed_index", "k", "pre_months", "post_months", "multiplier")
    }
    return {**keys, "season_shift": record.get("season_shift", 0), "campaign": campaign_key(record)}


def campaign_key(record: dict[str, Any]) -> str:
    """Return the campaign a cell belongs to: ``site_sNN`` for a synthetic tile, the site for a real campaign."""
    if record.get("seed_index") is None:
        return str(record["site"])
    return f"{record['site']}_s{int(record['seed_index']):02d}"


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
    """Add ``headline``, ``forward``, and ``reversal``, ``lambda`` and ``lambda_star`` where recorded.

    ``reversal``, ``lambda`` and ``lambda_star`` need both directions recorded.

    ``forward`` is the recorded forward reading, or the headline where none is recorded.
    """
    if frame.empty:
        return frame
    forward = (
        frame["uplift_forward"].fillna(frame["estimate"]) if "uplift_forward" in frame.columns else frame["estimate"]
    )
    frame = frame.assign(headline=frame["estimate"], forward=forward)
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


def _mean_se(values: pd.Series | np.ndarray, campaigns: pd.Series | np.ndarray) -> dict[str, float]:
    """Return the mean of ``values``, its standard error clustered by ``campaigns``, its z, and both counts."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    v, groups = v[ok], np.asarray(campaigns)[ok]
    n = v.size
    mean = float(v.mean()) if n else np.nan
    se = clustered_se(v, groups) if n else np.nan
    z = mean / se if np.isfinite(se) and se > 0 else np.nan
    return {"mean": mean, "se": se, "z": z, "n": n, "campaigns": len(set(groups.tolist()))}


def clustered_se(values: np.ndarray, campaigns: np.ndarray) -> float:
    """Return the standard error of the mean of ``values`` clustered by ``campaigns`` (CR1); NaN below 2 campaigns."""
    sums = pd.Series(values - values.mean()).groupby(campaigns).sum().to_numpy()
    g = sums.size
    if g < 2:  # noqa: PLR2004 - a spread needs two campaigns
        return np.nan
    return float(np.sqrt(g / (g - 1) * np.sum(sums**2)) / values.size)


def _errors(frame: pd.DataFrame, estimator: str) -> pd.Series:
    return frame[estimator] - frame["truth"]


def _site_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [("all", frame), *((str(site), group) for site, group in frame.groupby("site"))]


def references_table(records: list[dict[str, Any]], turbines: pd.DataFrame) -> pd.DataFrame:
    """Target i: the combined reference uplift and the placebo test turbines, each against 0 %."""
    farms = pd.DataFrame([{**_keys(r), **r["farm"]} for r in records])
    columns = [c for c in ("reference_uplift_before_screen", "reference_uplift_after_screen") if c in farms.columns]
    rows = [
        {"site": site, "quantity": column, **_mean_se(group[column], group["campaign"])}
        for site, group in _site_groups(farms)
        for column in columns
    ]
    placebo = turbines[(turbines["multiplier"] == 0) & (turbines["arm"] != "real")]
    rows += [
        {"site": site, "quantity": f"placebo_{estimator}", **_mean_se(_errors(group, estimator), group["campaign"])}
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
        for key, group in synthetic.groupby(["site", "campaign", "arm", "k", "post_months", "turbine"]):
            scale, bias = line_fit(group["truth"], group[estimator])
            fits.append({"site": key[0], "campaign": key[1], "scale": scale, "bias": bias})
        frame = pd.DataFrame(fits, columns=["site", "campaign", "scale", "bias"])
        for site, group in _site_groups(frame):
            slope = _mean_se(group["scale"] - 1.0, group["campaign"])
            intercept = _mean_se(group["bias"], group["campaign"])
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
                    "campaigns": slope["campaigns"],
                }
            )
    return pd.DataFrame(rows)


def dependence_table(turbines: pd.DataFrame) -> pd.DataFrame:
    """Target iii: the synthetic error by before-period length, after-period length, number of references and season."""
    synthetic = turbines[turbines["arm"] == "main"]
    rows = []
    for axis in ("pre_months", "post_months", "k", "season"):
        for value, group in synthetic.groupby(axis):
            for estimator in ESTIMATORS:
                errors = _errors(group, estimator).to_numpy(dtype=float)
                finite = errors[np.isfinite(errors)]
                rows.append(
                    {
                        "axis": axis,
                        "value": str(value),
                        "estimator": estimator,
                        **_mean_se(errors, group["campaign"]),
                        "rmse": float(np.sqrt(np.mean(finite**2))) if finite.size else np.nan,
                    }
                )
    rows += _season_within_rows(synthetic)
    return pd.DataFrame(rows)


def _season_within_rows(synthetic: pd.DataFrame) -> list[dict[str, Any]]:
    """Return the error less its mean over the same campaign turbine's seasons, by season, where it has several."""
    if "season" not in synthetic.columns or synthetic.empty:
        return []
    keys = ["campaign", "arm", "multiplier", "k", "pre_months", "post_months", "turbine"]
    crossed = synthetic[synthetic.groupby(keys, dropna=False)["season"].transform("nunique") > 1]
    rows = []
    for estimator in ESTIMATORS:
        error = _errors(crossed, estimator)
        within = error - error.groupby([crossed[k] for k in keys], dropna=False).transform("mean")
        for value, group in within.groupby(crossed["season"]):
            finite = group.to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            rows.append(
                {
                    "axis": "season_within",
                    "value": str(value),
                    "estimator": estimator,
                    **_mean_se(group, crossed.loc[group.index, "campaign"]),
                    "rmse": float(np.sqrt(np.mean(finite**2))) if finite.size else np.nan,
                }
            )
    return rows


def _coverage(errors: np.ndarray, sigmas: np.ndarray, campaigns: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(errors) & np.isfinite(sigmas) & (sigmas > 0)
    z = errors[ok] / sigmas[ok]
    within = (np.abs(z) <= 1.0).astype(float)
    groups = np.asarray(campaigns)[ok]
    return {
        "coverage": float(np.mean(within)) if z.size else np.nan,
        "coverage_se": clustered_se(within, groups) if z.size else np.nan,
        "z_sd": float(np.std(z)) if z.size else np.nan,
        "sigma_median": float(np.median(sigmas[ok])) if z.size else np.nan,
        "n": int(z.size),
        "campaigns": len(set(groups.tolist())),
    }


def component_sigmas(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return the headline sigma's recorded components alone and in quadrature pairs, by name.

    The components are ``sampling``, ``reference`` (the correction's standard error) and ``holdout``;
    a combination is left out when a component it needs is not recorded.
    """
    columns = {"sampling": "sigma_sampling", "reference": "sigma_reference", "holdout": "sigma_holdout"}
    parts = {name: frame[c].to_numpy(dtype=float) for name, c in columns.items() if c in frame.columns}
    out: dict[str, np.ndarray] = {}
    if "sampling" in parts:
        out["sampling"] = parts["sampling"]
        for other in ("reference", "holdout"):
            if other in parts:
                out[f"sampling+{other}"] = np.hypot(parts["sampling"], parts[other])
    return out


def coverage_table(records: list[dict[str, Any]], turbines: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    """Target iv: how often the error falls within one sigma, per quantity and recorded block length."""
    rows = []
    for quantity, frame in (("test", turbines[turbines["arm"] != "real"]), ("reference", references)):
        if frame.empty:
            continue
        for column in [c for c in frame.columns if c == "sigma" or str(c).startswith("sigma_block_")]:
            errors = _errors(frame, "headline").to_numpy(dtype=float)
            sigmas = frame[column].to_numpy(dtype=float)
            rows.append(
                {"quantity": quantity, "block": column, **_coverage(errors, sigmas, frame["campaign"].to_numpy())}
            )
        errors = _errors(frame, "headline").to_numpy(dtype=float)
        rows += [
            {"quantity": quantity, "block": name, **_coverage(errors, sigmas, frame["campaign"].to_numpy())}
            for name, sigmas in component_sigmas(frame).items()
        ]
    bins = pd.DataFrame(
        [{**_keys(r), **b} for r in records if r["arm"] != "real" for b in r.get("bins", [])],
        columns=["campaign", "condition", "estimate", "truth", "sigma"],
    )
    for condition, group in bins.groupby("condition"):
        errors = group["estimate"].to_numpy(dtype=float) - pd.to_numeric(group["truth"]).to_numpy(dtype=float)
        rows.append(
            {
                "quantity": f"bins_{condition}",
                "block": "sigma",
                **_coverage(errors, group["sigma"].to_numpy(dtype=float), group["campaign"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def shape_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Step 15: each axis and bin's error against truth, the flat headline's, and the sigma's coverage.

    The flat headline is the test turbine's headline uplift reported in every bin. Rows are the
    synthetic cells' bins with a truth; the bias's standard error is clustered by campaign.
    """
    rows = []
    for record in records:
        if record["arm"] == "real":
            continue
        headline = {t["turbine"]: t["estimate"] for t in record["turbines"]}
        rows += [
            {
                "campaign": campaign_key(record),
                "condition": b["condition"],
                "condition_bin": b["condition_bin"],
                "error": b["estimate"] - b["truth"],
                "flat_error": headline[b["test_wtg"]] - b["truth"],
                "sigma": b["sigma"],
            }
            for b in record.get("bins", [])
            if b.get("truth") is not None and np.isfinite(b["truth"]) and b["test_wtg"] in headline
        ]
    columns = ["condition", "condition_bin", "bias", "se", "z", "rmse", "rmse_flat", "coverage", "n", "campaigns"]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    out = []
    for (condition, bin_label), group in frame.groupby(["condition", "condition_bin"], sort=False):
        errors = group["error"].to_numpy(dtype=float)
        bias = _mean_se(errors, group["campaign"])
        out.append(
            {
                "condition": condition,
                "condition_bin": bin_label,
                "bias": bias["mean"],
                "se": bias["se"],
                "z": bias["z"],
                "rmse": float(np.sqrt(np.nanmean(errors**2))),
                "rmse_flat": float(np.sqrt(np.nanmean(group["flat_error"].to_numpy(dtype=float) ** 2))),
                "coverage": _coverage(
                    errors, pd.to_numeric(group["sigma"]).to_numpy(dtype=float), group["campaign"].to_numpy()
                )["coverage"],
                "n": bias["n"],
                "campaigns": bias["campaigns"],
            }
        )
    return pd.DataFrame(out, columns=columns)


def period_coverage_table(turbines: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    """Every before and after length: the headline error's bias, its RMSE and the headline sigma's coverage.

    ``coverage_<component>`` is the coverage of each of :func:`component_sigmas` the test turbines record.
    """
    rows = []
    for quantity, frame in (("test", turbines[turbines["arm"] != "real"]), ("reference", references)):
        if frame.empty:
            continue
        for (pre, post), group in frame.groupby(["pre_months", "post_months"]):
            errors = _errors(group, "headline").to_numpy(dtype=float)
            finite = errors[np.isfinite(errors)]
            bias = _mean_se(errors, group["campaign"])
            rows.append(
                {
                    "quantity": quantity,
                    "pre_months": int(pre),
                    "post_months": int(post),
                    "bias": bias["mean"],
                    "bias_se": bias["se"],
                    "rmse": float(np.sqrt(np.mean(finite**2))) if finite.size else np.nan,
                    **_coverage(errors, group["sigma"].to_numpy(dtype=float), group["campaign"].to_numpy()),
                    **{
                        f"coverage_{name}": _coverage(errors, sigmas, group["campaign"].to_numpy())["coverage"]
                        for name, sigmas in component_sigmas(group).items()
                    },
                }
            )
    return pd.DataFrame(rows)


def unmeasured_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """How many test turbines each (before length, after length) left unmeasured, and how many for too few hours."""
    rows = []
    for record in records:
        if record["arm"] == "real":
            continue
        unplanned = record.get("unplanned", {})
        rows.append(
            {
                "pre_months": record["pre_months"],
                "post_months": record["post_months"],
                "measured": len(record["turbines"]),
                "unmeasured": len(unplanned),
                "too_few_hours": sum("usable hours" in str(reason) for reason in unplanned.values()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["pre_months", "post_months", "measured", "unmeasured", "too_few_hours", "unmeasured_share"]
        )
    table = pd.DataFrame(rows).groupby(["pre_months", "post_months"], as_index=False).sum()
    table["unmeasured_share"] = table["unmeasured"] / (table["measured"] + table["unmeasured"])
    return table


def reversal_table(turbines: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    """Option 2: the headline, forward, full reversal and computed lambda per after length, test and reference."""
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
    """Open question (step 11 ii): the headline error by the after-period share the rule removes, in thirds."""
    frame = turbines[(turbines["arm"] != "real") & turbines["rare_share_after"].notna()]
    rows = []
    for interval, group in frame.groupby(_thirds(frame["rare_share_after"])):
        errors = _errors(group, "headline").to_numpy(dtype=float)
        rows.append(
            {
                "rare_share": str(interval),
                "share_median": float(group["rare_share_after"].median()),
                **_mean_se(errors, group["campaign"]),
                "rmse": float(np.sqrt(np.nanmean(errors**2))),
            }
        )
    return pd.DataFrame(rows, columns=["rare_share", "share_median", "mean", "se", "z", "n", "campaigns", "rmse"])


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


def usable_share_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Return the median shares of each test turbine's before and after hours that are usable or lost upwind.

    One row per synthetic arm, site and (P, L), and one per arm and site over all of them. The losses
    are to a changed turbine upwind of a reference and of the subject.
    """
    rows = []
    for record in records:
        if record["arm"] == "real":
            continue
        for rows_of_turbine in record.get("record_losses", {}).values():
            hours = {r["reason"]: r for r in rows_of_turbine}
            row: dict[str, Any] = {k: record[k] for k in ("arm", "site", "pre_months", "post_months")}
            for side in ("before", "after"):
                total = sum(float(r.get(f"hours_{side}") or 0.0) for r in rows_of_turbine)
                for name, reason in _SHARE_REASONS.items():
                    lost = float((hours.get(reason) or {}).get(f"hours_{side}") or 0.0)
                    row[f"{name}_{side}"] = lost / total if total else np.nan
            rows.append(row)
    shares = [f"{name}_{side}" for side in ("before", "after") for name in _SHARE_REASONS]
    columns = ["arm", "site", "pre_months", "post_months", "turbines", *shares]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    by_period = frame.groupby(["arm", "site", "pre_months", "post_months"], as_index=False)
    overall = frame.groupby(["arm", "site"], as_index=False)
    table = pd.concat(
        [
            by_period[shares].median().merge(by_period.size().rename(columns={"size": "turbines"})),
            overall[shares]
            .median()
            .merge(overall.size().rename(columns={"size": "turbines"}))
            .assign(pre_months="all", post_months="all"),
        ],
        ignore_index=True,
    )
    return table[columns]


def validation_tables(
    records: list[dict[str, Any]], *, unmeasured: Sequence[dict[str, Any]] = ()
) -> dict[str, pd.DataFrame]:
    """Return every target's table for ``records``: targets i to iv on the domain's cells, the rest on all.

    ``unmeasured`` are the cells that measured no test turbine; only the unmeasured table counts them.
    """
    domain = [r for r in records if in_domain(r)]
    turbines = turbine_frame(records)
    references = reference_frame(records)
    domain_turbines = turbine_frame(domain)
    domain_references = reference_frame(domain)
    return {
        "i_references": references_table(domain, domain_turbines),
        "ii_linearity": linearity_table(domain_turbines),
        "iii_dependence": dependence_table(domain_turbines),
        "iv_coverage": coverage_table(domain, domain_turbines, domain_references),
        "iv_coverage_by_period": period_coverage_table(turbines, references),
        "v_shape": shape_table(domain),
        "unmeasured": unmeasured_table([*records, *unmeasured]),
        "usable_share": usable_share_table(records),
        "option2_reversal": reversal_table(turbines, references),
        "open_configuration_rule": configuration_rule_table(turbines),
        "open_screen": screen_table(references),
    }


def _verdict_mean(row: pd.Series) -> str:
    if row["campaigns"] < MIN_N or not np.isfinite(row["z"]):
        return "too few"
    return "pass" if abs(row["z"]) <= Z_LIMIT else "FAIL"


def verdict_coverage(row: pd.Series) -> str:
    """Judge a coverage: a sigma too small fails, one a little too large passes as wide.

    Within the tolerance or two clustered standard errors of the target, whichever is wider, passes;
    below that fails; above it passes as wide up to ``COVERAGE_CEILING``, and fails beyond.
    """
    if row["campaigns"] < MIN_N or not np.isfinite(row["coverage"]) or not np.isfinite(row["coverage_se"]):
        return "too few"
    gap = row["coverage"] - TARGET_COVERAGE
    if abs(gap) <= max(COVERAGE_TOLERANCE, Z_LIMIT * row["coverage_se"]):
        return "pass"
    if gap < 0:
        return "FAIL (sigma too small)"
    return "pass (sigma wide)" if row["coverage"] <= COVERAGE_CEILING else "FAIL (sigma too large)"


def describe(tables: dict[str, pd.DataFrame], *, n_cells: int, n_domain: int, n_unmeasured: int = 0) -> str:
    """Return the report as markdown: one verdict line per check, then each table."""
    unmeasured = f" ({n_unmeasured} more measured no test turbine)" if n_unmeasured else ""
    lines = [
        f"# Validation of {n_cells} ok cells{unmeasured}",
        "",
        f"Targets i to iv are judged on the {n_domain} cells with at least {DOMAIN_PRE_MONTHS} months before and "
        f"{DOMAIN_POST_MONTHS} months after; the per-period section covers every cell.",
        "",
    ]
    refs = tables["i_references"]
    lines.append("## i. References and placebos read 0 %")
    lines.append(
        "- the reference readings are forward readings; a reference-corrected headline removes their mean, so the "
        "placebo headline is its check"
    )
    lines += [
        f"- {row.quantity}: {row.mean * _PP:+.3f} ± {row.se * _PP:.3f} pp (n {row.n}, {row.campaigns} campaigns): "
        f"{_verdict_mean(pd.Series(row._asdict()))}"
        for row in refs[refs["site"] == "all"].itertuples()
    ]
    lines += ["", "## ii. Measured against true uplift: slope 1, intercept 0"]
    for row in tables["ii_linearity"].query("site == 'all'").itertuples():
        slope_ok = _verdict_mean(pd.Series({"z": row.slope_z, "campaigns": row.campaigns}))
        intercept_ok = _verdict_mean(pd.Series({"z": row.intercept_z, "campaigns": row.campaigns}))
        lines.append(
            f"- {row.estimator}: slope {row.slope:.3f} ± {row.slope_se:.3f} ({slope_ok}), intercept "
            f"{row.intercept * _PP:+.3f} ± {row.intercept_se * _PP:.3f} pp ({intercept_ok}), n {row.n}, "
            f"{row.campaigns} campaigns"
        )
    lines += ["", "## iii. No dependence on before-period length, after-period length, references or season (headline)"]
    dep = tables["iii_dependence"]
    for axis, group in dep[dep["estimator"] == "headline"].groupby("axis", sort=False):
        spread = group["mean"].max() - group["mean"].min()
        worst = group.loc[group["z"].abs().idxmax()] if group["z"].notna().any() else None
        if worst is None or worst["campaigns"] < MIN_N:
            verdict = "too few"
        else:
            verdict = "pass" if abs(worst["z"]) <= Z_LIMIT else "FAIL"
        detail = (
            "" if worst is None else f"; worst {worst['value']} at {worst['mean'] * _PP:+.3f} pp (z {worst['z']:+.1f})"
        )
        lines.append(f"- {axis}: bias ranges over {spread * _PP:.3f} pp{detail}: {verdict}")
    lines += ["", "## iv. Sigma covers 68 % of the errors"]
    for row in tables["iv_coverage"].itertuples():
        lines.append(
            f"- {row.quantity} {row.block}: {row.coverage:.2f} ± {row.coverage_se:.2f} (z sd {row.z_sd:.2f}, "
            f"n {row.n}, {row.campaigns} campaigns): "
            f"{verdict_coverage(pd.Series(row._asdict()))}"
        )
    lines += ["", "## v. Uplift distributions (step 15): per axis, against the flat headline in every bin"]
    shape = tables["v_shape"]
    for condition, group in shape.groupby("condition", sort=False):
        worst = group.loc[group["z"].abs().idxmax()] if group["z"].notna().any() else None
        low = group.loc[group["coverage"].idxmin()] if group["coverage"].notna().any() else None
        better = float(np.sqrt(np.nanmean(group["rmse"] ** 2))) < float(np.sqrt(np.nanmean(group["rmse_flat"] ** 2)))
        detail = (
            ""
            if worst is None
            else f"; worst bin {worst['condition_bin']} {worst['bias'] * _PP:+.3f} pp (z {worst['z']:+.1f})"
        )
        cover = "" if low is None else f"; lowest coverage {low['coverage']:.2f} in {low['condition_bin']}"
        lines.append(
            f"- {condition}: RMSE {float(np.sqrt(np.nanmean(group['rmse'] ** 2))) * _PP:.3f} pp against "
            f"{float(np.sqrt(np.nanmean(group['rmse_flat'] ** 2))) * _PP:.3f} pp flat"
            f" ({'beats flat' if better else 'does NOT beat flat'}){detail}{cover}"
        )
    lines += [
        "",
        "## Every period, in and out of the domain: test bias ± se, RMSE (pp) and coverage per before length P and "
        "after length L",
    ]
    by_period = tables["iv_coverage_by_period"]
    for pre, group in by_period[by_period["quantity"] == "test"].groupby("pre_months"):
        cells = ", ".join(
            f"L{int(r.post_months)} {r.bias * _PP:+.2f} ± {r.bias_se * _PP:.2f}, {r.rmse * _PP:.2f}, {r.coverage:.2f}"
            for r in group.itertuples()
        )
        lines.append(f"- P{int(pre)}: {cells}")
    unmeasured = tables["unmeasured"]
    if not unmeasured.empty:
        total = int(unmeasured["unmeasured"].sum())
        hours = int(unmeasured["too_few_hours"].sum())
        worst = unmeasured.loc[unmeasured["unmeasured_share"].idxmax()]
        lines.append(
            f"- unmeasured test turbines: {total} ({hours} for too few usable hours); worst "
            f"P{int(worst['pre_months'])} L{int(worst['post_months'])} at {worst['unmeasured_share']:.0%}"
        )
    usable = tables["usable_share"]
    if not usable.empty:
        lines += ["", f"## Usable hours (C10); run 4 median usable share of before hours {RUN4_USABLE_BEFORE}"]
        for row in usable[usable["pre_months"] == "all"].itertuples():
            lines.append(
                f"- usable share of before hours: {row.arm} {row.site} {row.usable_before:.2f}, after "
                f"{row.usable_after:.2f}; lost before to changed upwind of reference "
                f"{row.changed_upwind_of_reference_before:.2f}, of subject {row.changed_upwind_of_subject_before:.2f} "
                f"({row.turbines} test turbines)"
            )
    lines += ["", "## Option 2: the headline, forward, full reversal and computed lambda (RMSE, pp)"]
    for row in tables["option2_reversal"].itertuples():
        lines.append(
            f"- {row.quantity} L{row.post_months}: headline {row.headline_rmse * _PP:.3f}, forward "
            f"{row.forward_rmse * _PP:.3f}, reversal "
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
    unmeasured = study_records(study_dir, status="unmeasured")
    tables = validation_tables(records, unmeasured=unmeasured)
    out = study_dir / "validation"
    out.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    text = describe(
        tables, n_cells=len(records), n_domain=sum(in_domain(r) for r in records), n_unmeasured=len(unmeasured)
    )
    (out / "validation.md").write_text(text)
    return text
