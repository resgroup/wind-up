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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.harness.campaign import campaign_windows, treated_activity_mask, window_row_mask
from benchmarking.harness.method import MethodInput
from benchmarking.harness.replicates import build_replicates
from benchmarking.synthetic import HOT_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Callable

    from benchmarking.harness.campaign import CampaignWindow
    from benchmarking.harness.method import Method
    from benchmarking.harness.replicates import Replicate, StudyConfig
    from benchmarking.synthetic import ColumnSchema


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
    truths = [_truth_overall(replicate, window) for replicate, window in instances]

    rows = []
    for method in methods:
        method_rows = []
        for (replicate, window), truth in zip(instances, truths, strict=True):
            method_input = _method_input(replicate, window)
            output = method.estimate(method_input)
            method_rows.append(
                {
                    "method": method.name,
                    "profile": profile_name,
                    "replicate": replicate.replicate_id,
                    "test_wtg": replicate.test_wtg,
                    "campaign_months": window.months,
                    "treatment_start": window.treatment_start,
                    "baseline_start": window.baseline_start,
                    "activity_end": window.activity_end,
                    "condition": "overall",
                    "estimate": output.p50_overall,
                    "truth": truth,
                    "signed_error": output.p50_overall - truth,
                }
            )
        if on_method_complete is not None:
            on_method_complete(method.name, pd.DataFrame(method_rows))
        rows.extend(method_rows)
    return pd.DataFrame(rows)


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


def _truth_overall(replicate: Replicate, window: CampaignWindow) -> float:
    """Ground-truth uplift over the test turbine's treated rows within the activity window."""
    synthetic = replicate.synthetic_df
    test_index = synthetic.loc[synthetic[replicate.dataset.columns.turbine] == replicate.test_wtg].index
    mask = treated_activity_mask(test_index, replicate.upgrade_timing, window=window)
    return replicate.true_uplift(mask=mask).overall
