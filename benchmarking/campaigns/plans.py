"""Resolve an analysis plan per upgraded turbine from a campaign's public facts and its SCADA."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.harness.wind_rose import era5_rose, rose_label
from benchmarking.synthetic import treated_mask
from wind_up.analysis_period import DEFAULT_PLAN_SETTINGS, AnalysisPlan, PlanSettings, plan_analysis

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from benchmarking.campaigns.declaration import CampaignSpec, Window
    from benchmarking.harness.record_selection import WorksWindow
    from benchmarking.synthetic import ColumnSchema, ToggleSchedule

logger = logging.getLogger(__name__)


def data_extents(scada_df: pd.DataFrame, *, turbine_col: str, power_col: str) -> dict[str, Window]:
    """Each turbine's first finite power and the end of its last, one timebase after it."""
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    timebase = pd.Timedelta(np.median(np.diff(index.to_numpy()))) if len(index) > 1 else pd.Timedelta(minutes=10)
    finite = scada_df[np.isfinite(scada_df[power_col].to_numpy(dtype=float))]
    grouped = pd.Series(finite.index, index=finite[turbine_col].astype(str).to_numpy()).groupby(level=0)
    return {
        str(t): (first, last + timebase)
        for t, first, last in zip(grouped.min().index, grouped.min(), grouped.max(), strict=True)
    }


@dataclass(frozen=True)
class CampaignPlans:
    """Each upgraded turbine's plan, and every turbine the selector could not plan with its reason.

    :param plans: upgraded turbine to its analysis plan
    :param unplanned: upgraded turbine to the selector's reason for giving up on it
    :param works: the works windows the plans were chosen from, the upgraded turbines' included
    :param extents: each turbine's data extent the plans were chosen from
    :param warnings: per upgraded turbine, the turbines with an indicator of an undeclared change
        inside its periods (itself included), each with the warning
    """

    plans: dict[str, AnalysisPlan] = field(default_factory=dict)
    unplanned: dict[str, str] = field(default_factory=dict)
    works: dict[str, list[Window]] = field(default_factory=dict)
    extents: dict[str, Window] = field(default_factory=dict)
    warnings: dict[str, dict[str, str]] = field(default_factory=dict)


def with_wake_ranking(
    settings: PlanSettings, *, spec: CampaignSpec, era5_df: pd.DataFrame, source: str
) -> PlanSettings:
    """Return ``settings`` ranking references with the campaign's wake reach and the rose of ``era5_df``.

    :param settings: the period selector's settings
    :param spec: the campaign, for its reach
    :param era5_df: the reanalysis the rose is built from, over the campaign's data
    :param source: what the reanalysis is, for the reports
    """
    return replace(
        settings,
        wake_reach_d=spec.wake_reach_d,
        rose=era5_rose(era5_df),
        rose_source=rose_label(era5_df, source=source),
    )


def campaign_works(spec: CampaignSpec) -> dict[str, list[Window]]:
    """Every turbine's works windows, each upgraded turbine given one.

    A shared changeover is a window of no length at the changeover; a toggle campaign's is at the
    start of toggling.
    """
    works = {t: list(w) for t, w in spec.works.items()}
    if spec.mode == "toggle":
        start = spec.treatment_start
        works.update({t: [(start, start)] for t in spec.upgraded_turbines})
    else:
        works.update({t: [spec.works_window(t)] for t in spec.upgraded_turbines})
    return works


def changed_works(spec: CampaignSpec) -> dict[str, list[WorksWindow]]:
    """Every turbine's works windows as the changed-upwind rule sees them, each with its reach.

    Every declared window and each upgrade.
    """
    works = {t: list(w) for t, w in spec.works.items()}
    for turbine in spec.upgraded_turbines:
        works[turbine] = sorted({*works.get(turbine, []), *campaign_works(spec)[turbine]})
    return {t: [(w0, w1, spec.reach_of(t, (w0, w1))) for w0, w1 in windows] for t, windows in works.items()}


def other_works(spec: CampaignSpec, turbine: str) -> list[Window]:
    """Return an upgraded turbine's declared works windows other than its upgrade.

    A staggered campaign's only works window is its upgrade. Under a shared changeover the upgrade
    is any window that contains the changeover, and every window of a toggle campaign is another change.
    """
    windows = list(spec.works.get(turbine, []))
    if spec.mode == "toggle":
        return windows
    if spec.upgrade_timing is None:
        return []
    changeover = pd.Timestamp(spec.upgrade_timing)  # type: ignore[arg-type]
    return [(w0, w1) for w0, w1 in windows if not w0 <= changeover <= w1]


def extent_around_works(extent: Window, *, upgrade: Window, others: Sequence[Window]) -> Window:
    """Return ``extent`` cut to between the nearest of ``others`` before ``upgrade`` and after it."""
    first, last = extent
    for w0, w1 in others:
        if w1 <= upgrade[0]:
            first = max(first, w1)
        elif w0 >= upgrade[1]:
            last = min(last, w0)
    return first, last


def plans_for(
    spec: CampaignSpec,
    scada_df: pd.DataFrame,
    *,
    columns: ColumnSchema,
    settings: PlanSettings = DEFAULT_PLAN_SETTINGS,
    possible_changes: Mapping[str, Sequence[pd.Timestamp]] | None = None,
) -> CampaignPlans:
    """Return each upgraded turbine's plan, and the turbines no span could be found for.

    A turbine the selector cannot plan is reported with its reason rather than stopping the
    campaign, so the others are still analysed.

    Declared references are the candidates; otherwise every turbine not excluded is, the other
    upgraded turbines included. Under a declared span, declared references are forced in even when
    their works overlap it.

    :param spec: the campaign's public facts
    :param scada_df: long-format SCADA; data extents are read from its finite power
    :param columns: the schema ``scada_df`` is keyed by
    :param settings: how spans and references are chosen
    :param possible_changes: each turbine's indicators of an undeclared change; one inside a plan's
        periods is reported as a warning against that turbine
    """
    extents = data_extents(scada_df, turbine_col=spec.turbine_col, power_col=columns.active_power)
    works = campaign_works(spec)
    excluded = set(spec.excluded_turbines)
    if spec.references_declared:
        candidates = [r for r in spec.candidate_references if r not in excluded]
    else:
        candidates = [
            r for r in dict.fromkeys([*spec.candidate_references, *spec.upgraded_turbines]) if r not in excluded
        ]

    plans: dict[str, AnalysisPlan] = {}
    unplanned: dict[str, str] = {}
    warnings: dict[str, dict[str, str]] = {}
    for turbine in sorted(spec.upgraded_turbines):
        span = spec.period_for(turbine)
        others = other_works(spec, turbine)
        # The test turbine's own extent is cut so that neither period holds another of its changes.
        own_extents = dict(extents)
        if others and turbine in extents:
            own_extents[turbine] = extent_around_works(extents[turbine], upgrade=works[turbine][0], others=others)
        try:
            plan = plan_analysis(
                spec.layout,
                turbine=turbine,
                works=works,
                exclusions=spec.exclusions,
                extents=own_extents,
                candidates=candidates,
                span=span,
                allow_works_overlap=candidates if spec.references_declared and span is not None else (),
                settings=settings,
            )
        except ValueError as exc:
            logger.warning("%s is not analysed: %s", turbine, exc)
            unplanned[turbine] = str(exc)
            continue
        for reference in plan.forced:
            logger.warning("%s: declared reference %s is used although its works overlap the span", turbine, reference)
        logger.info(
            "%s: analysis span %s..%s (%s), before %.0f days, after %.0f days, power references %s, pool rule %s",
            turbine,
            plan.start,
            plan.end,
            "declared" if plan.declared else "chosen",
            plan.pre / pd.Timedelta(days=1),
            plan.post / pd.Timedelta(days=1),
            list(plan.power_references),
            "met" if plan.pool_rule_met else f"NOT met: {plan.pool_rule_reason}",
        )
        plans[turbine] = plan
        warnings[turbine] = plan_warnings(plan, possible_changes=possible_changes or {})
        inside = [(w0, w1) for w0, w1 in others if w0 < plan.end and w1 > plan.start]
        if inside:
            days = ", ".join(f"{w0:%Y-%m-%d}..{w1:%Y-%m-%d}" for w0, w1 in inside)
            warnings[turbine][turbine] = f"declared works {days} inside the declared span"
        for other, warning in warnings[turbine].items():
            logger.warning("%s: %s: %s", turbine, other, warning)
    return CampaignPlans(plans=plans, unplanned=unplanned, works=works, extents=extents, warnings=warnings)


def plan_warnings(plan: AnalysisPlan, *, possible_changes: Mapping[str, Sequence[pd.Timestamp]]) -> dict[str, str]:
    """Return the warning against each turbine of ``plan`` with a possible undeclared change inside its periods.

    The test turbine is checked over its before and after periods, each power reference and reserve
    over the whole span.
    """
    found: dict[str, str] = {}
    windows = {plan.turbine: [plan.before, plan.after]}
    windows.update({r: [(plan.start, plan.end)] for r in (*plan.power_references, *plan.reserves)})
    for turbine, periods in windows.items():
        inside = sorted(t for t in possible_changes.get(turbine, ()) if any(s <= t < e for s, e in periods if s < e))
        if inside:
            days = ", ".join(t.strftime("%Y-%m-%d") for t in inside)
            found[turbine] = f"possible undeclared change on {days}"
    return found


def treated_activity(
    index: pd.DatetimeIndex, *, timing: pd.Timestamp | ToggleSchedule, window: Window
) -> npt.NDArray[np.bool_]:
    """Return the test turbine's treated rows within ``window``, its after period."""
    start, end = window
    in_activity = np.asarray((index >= start) & (index < end))
    return treated_mask(index, timing) & in_activity
