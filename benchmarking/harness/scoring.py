"""The scoring orchestrator: run methods over an ensemble and a campaign-length grid.

For one study it builds the replicate ensemble, materialises the ``(replicate, campaign)``
instances **once**, and scores every method against that identical list. Two fairness
properties follow structurally:

1. **Across methods** — instances are built before any method runs, so every method sees the
   exact same ``MethodInput``s; only the estimate differs.
2. **Across campaign lengths** — all lengths of one replicate share its
   ``(turbine, baseline_start, treatment_start)`` and differ only in ``activity_end``, so
   shorter windows are leading prefixes of longer ones (a property of ``campaign_windows``).

For each instance the method's estimate and the ground truth are computed over the **same
records**, so the signed error is exact at every campaign length. Output is one tidy
long-format DataFrame, the input to the leaderboard and plots.

:func:`score_one` is the per-``(method, instance)`` unit :func:`score_study` is built from, public so
a caller that cannot afford the materialised ensemble (a large replicate ensemble would exhaust
memory) can drive its own replicate-outer loop over :func:`iter_replicates` without reimplementing
the truth alignment.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.harness.campaign import campaign_windows, treated_activity_mask, window_row_mask
from benchmarking.harness.conditions import condition_bins
from benchmarking.harness.method import MethodInput
from benchmarking.harness.replicates import build_replicates
from benchmarking.synthetic import HOT_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import numpy.typing as npt

    from benchmarking.harness.campaign import CampaignWindow
    from benchmarking.harness.method import Method
    from benchmarking.harness.replicates import Replicate, StudyConfig
    from benchmarking.synthetic import ColumnSchema

# The columns a method's ``uncertainty_diagnostics`` keys on, and therefore does not contribute.
_DIAGNOSTIC_KEYS = ("condition", "condition_bin")
# Columns the harness itself writes on every result row. A method's diagnostics may not use these
# names — see :func:`_merge_diagnostics`. The campaign-length column is study-dependent
# (``campaign_months`` / ``campaign_weeks``), so both are reserved regardless of the study's unit.
_RESERVED_COLUMNS = frozenset(
    {
        "method",
        "profile",
        "replicate",
        "test_wtg",
        "campaign_months",
        "campaign_weeks",
        "treatment_start",
        "baseline_start",
        "activity_end",
        "condition",
        "condition_bin",
        "estimate",
        "truth",
        "signed_error",
        "sigma",
        "wall_time_s",
    }
)


def score_study(
    base_scada: pd.DataFrame,
    *,
    profile: list,
    methods: list[Method],
    study: StudyConfig,
    profile_name: str = "profile",
    columns: ColumnSchema = HOT_COLUMNS,
    on_method_complete: Callable[[str, pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    """Score ``methods`` on ``study`` over ``profile`` injected into ``base_scada``.

    Returns a tidy long-format frame: one row per method x replicate x campaign length, with
    the P50 ``estimate``, the ground-truth ``truth`` and their ``signed_error``. Each row also
    carries the window it was tested over — ``treatment_start`` (the upgrade start),
    ``baseline_start`` and ``activity_end`` — so a result is self-describing.

    :param columns: the source-native column schema ``base_scada`` is keyed by
    :param on_method_complete: optional hook called as each method finishes its full instance
        sweep, with ``(method_name, that_method's_rows)`` (the same rows it contributes to the
        returned frame). Lets a caller act on a method's results early — e.g. plot them — instead
        of waiting for every method, useful when a slow method runs last. It never changes the
        returned frame; order methods fastest-first to get the earliest feedback.
    """
    replicates = build_replicates(base_scada, profile=profile, study=study, columns=columns)
    data_start = base_scada.index.min()
    data_end = base_scada.index.max()
    instances = _materialise_instances(replicates, study, data_start=data_start, data_end=data_end)
    # Truth depends only on ``(replicate, window)``, so compute it once here rather than once
    # per method (it would otherwise carry an avoidable ``len(methods)`` multiplier).
    truth_masks = [truth_mask(r, w) for r, w in instances]
    truths = [r.true_uplift(mask=m).overall for (r, _), m in zip(instances, truth_masks, strict=True)]

    rows = []
    for method in methods:
        method_rows: list[dict[str, object]] = []
        for (replicate, window), truth, mask in zip(instances, truths, truth_masks, strict=True):
            method_rows.extend(
                score_one(method, replicate=replicate, window=window, truth=truth, mask=mask, profile_name=profile_name)
            )
        if on_method_complete is not None:
            on_method_complete(method.name, pd.DataFrame(method_rows))
        rows.extend(method_rows)
    return pd.DataFrame(rows)


def score_one(
    method: Method,
    *,
    replicate: Replicate,
    window: CampaignWindow,
    truth: float,
    mask: npt.NDArray[np.bool_],
    profile_name: str = "profile",
) -> list[dict[str, object]]:
    """Run one method over one ``(replicate, window)`` instance and return its tidy result rows.

    The overall row plus one per condition bin the method reports, each with the estimate, ``truth``,
    their signed error and the method's ``sigma`` (NaN when it reports no uncertainty).

    ``truth`` and ``mask`` are passed in because they depend only on ``(replicate, window)``;
    deriving them here would multiply that cost by ``len(methods)``. Build them with
    :func:`truth_mask` and ``replicate.true_uplift(mask=...)``.
    """
    method_input = _method_input(replicate, window)
    start = time.perf_counter()
    output = method.estimate(method_input)
    wall_time_s = time.perf_counter() - start
    base_fields: dict[str, object] = {
        "method": method.name,
        "profile": profile_name,
        "replicate": replicate.replicate_id,
        "test_wtg": replicate.test_wtg,
        window.length_col: window.length,
        "treatment_start": window.treatment_start,
        "baseline_start": window.baseline_start,
        "activity_end": window.activity_end,
    }
    rows: list[dict[str, object]] = [
        {
            **base_fields,
            "condition": "overall",
            "condition_bin": "overall",
            "estimate": output.p50_overall,
            "truth": truth,
            "signed_error": output.p50_overall - truth,
            "sigma": _optional_float(output.sigma_overall),
            "wall_time_s": wall_time_s,
        }
    ]
    if output.p50_by_condition is not None:
        rows.extend(_conditional_rows(output.p50_by_condition, replicate, window, mask, base_fields))
    _merge_diagnostics(rows, output.uncertainty_diagnostics)
    return rows


def _optional_float(value: float | None) -> float:
    """Return ``value`` as a float, with ``None`` (no uncertainty reported) becoming NaN."""
    return float("nan") if value is None else float(value)


def _merge_diagnostics(rows: list[dict[str, object]], diagnostics: pd.DataFrame | None) -> None:
    """Merge a method's uncertainty diagnostics onto ``rows`` in place, keyed by condition and bin.

    Attached without interpretation. Rows with no matching diagnostics row are left alone. A column
    colliding with one the harness owns raises rather than silently overwriting the estimate or truth
    the row exists to report.
    """
    if diagnostics is None:
        return
    missing = [c for c in _DIAGNOSTIC_KEYS if c not in diagnostics.columns]
    if missing:
        msg = f"uncertainty_diagnostics must be keyed by {list(_DIAGNOSTIC_KEYS)}; missing {missing}"
        raise ValueError(msg)
    extra_cols = [c for c in diagnostics.columns if c not in _DIAGNOSTIC_KEYS]
    clashes = sorted(set(extra_cols) & _RESERVED_COLUMNS)
    if clashes:
        msg = (
            f"uncertainty_diagnostics columns {clashes} clash with columns the harness owns "
            f"({sorted(_RESERVED_COLUMNS)}); rename them in the method"
        )
        raise ValueError(msg)
    keys = [(str(row["condition"]), str(row["condition_bin"])) for _, row in diagnostics.iterrows()]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        msg = (
            f"uncertainty_diagnostics has duplicate {list(_DIAGNOSTIC_KEYS)} keys {duplicates}; "
            f"the frame must hold one row per key"
        )
        raise ValueError(msg)
    by_key = {
        key: {col: row[col] for col in extra_cols} for key, (_, row) in zip(keys, diagnostics.iterrows(), strict=True)
    }
    for row in rows:
        row.update(by_key.get((str(row["condition"]), str(row["condition_bin"])), {}))


def _materialise_instances(
    replicates: list[Replicate],
    study: StudyConfig,
    *,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
) -> list[tuple[Replicate, CampaignWindow]]:
    """Build the fixed ``(replicate, campaign window)`` list shared by every method."""
    instances: list[tuple[Replicate, CampaignWindow]] = []
    for replicate in replicates:
        windows = campaign_windows(
            replicate.treatment_start,
            min_pre_months=study.min_pre_months,
            campaign_months=study.campaign_months,
            campaign_weeks=study.campaign_weeks,
            data_start=data_start,
            data_end=data_end,
        )
        instances.extend((replicate, window) for window in windows)
    return instances


def _method_input(replicate: Replicate, window: CampaignWindow) -> MethodInput:
    """Return the method-facing rows: all subset turbines within ``[baseline_start, activity_end)``."""
    synthetic = replicate.synthetic_df
    row_mask = window_row_mask(synthetic.index, window)
    return MethodInput(
        scada_df=synthetic.loc[row_mask],
        test_wtg=replicate.test_wtg,
        upgrade_timing=replicate.upgrade_timing,
        turbine_col=replicate.dataset.columns.turbine,
    )


def truth_mask(replicate: Replicate, window: CampaignWindow) -> npt.NDArray[np.bool_]:
    """Return the treated-activity mask over the test turbine's rows for this window.

    The mask ``replicate.true_uplift(mask=...)`` and :func:`score_one` must agree on, so a caller
    driving its own loop scores against the same records :func:`score_study` would.
    """
    synthetic = replicate.synthetic_df
    test_index = synthetic.loc[synthetic[replicate.dataset.columns.turbine] == replicate.test_wtg].index
    return treated_activity_mask(test_index, replicate.upgrade_timing, window=window)


def _conditional_rows(
    by_condition: pd.DataFrame,
    replicate: Replicate,
    window: CampaignWindow,  # noqa: ARG001 (kept for symmetry / future use)
    mask: npt.NDArray[np.bool_],
    base_fields: dict[str, object],
) -> list[dict[str, object]]:
    """Join each method per-bin estimate to per-bin truth for every condition present.

    A ``sigma_uplift`` column on ``by_condition`` is carried through as the row's ``sigma``; a
    method that reports per-bin P50s without per-bin uncertainty simply omits it and reads NaN.
    """
    rows: list[dict[str, object]] = []
    # power edges scale with the source's baseline rating; ws/ti ignore it (fixed edges).
    rated_power_kw = float(replicate.dataset.run_metadata["rated_power_kw"])
    has_sigma = "sigma_uplift" in by_condition.columns
    for condition, est in by_condition.groupby("condition"):
        bins = condition_bins(str(condition), rated_power_kw=rated_power_kw)
        truth_df = replicate.true_uplift(mask=mask, by=condition, bins=bins).by_condition
        if truth_df is None:  # always set when by= is passed; guard for type narrowing
            continue  # pragma: no cover
        truth_series = truth_df.assign(condition_bin=truth_df["condition_bin"].astype(str)).set_index("condition_bin")[
            "true_uplift"
        ]
        for _, r in est.iterrows():
            bin_label = str(r["condition_bin"])
            t = float(truth_series.get(bin_label, float("nan")))
            e = float(r["p50_uplift"])
            rows.append(
                {
                    **base_fields,
                    "condition": condition,
                    "condition_bin": bin_label,
                    "estimate": e,
                    "truth": t,
                    "signed_error": e - t,
                    "sigma": float(r["sigma_uplift"]) if has_sigma else float("nan"),
                    "wall_time_s": float("nan"),
                }
            )
    return rows
