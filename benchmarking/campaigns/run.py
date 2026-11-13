"""The truth-free campaign core: estimate a declared campaign from SCADA alone.

Nothing here reads ground truth, so this is the path a real campaign runs on. The benchmark
layers truth, scoring and truth-vs-estimate plots on top; see
:mod:`benchmarking.campaigns.runner`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.baselines.power_model.method import headline_detail, reference_overall_uplift
from benchmarking.campaigns.context import context_for_plan
from benchmarking.campaigns.data_screen import (
    DEFAULT_DATA_SCREEN,
    DROPPED,
    DataScreen,
    DataScreenSettings,
    reference_data_table,
    reference_hours_table,
    screen_plan,
)
from benchmarking.campaigns.plans import CampaignPlans, plans_for, treated_activity
from benchmarking.harness import MethodConfigurationError, MethodInput
from benchmarking.harness.northing import DEFAULT_NORTHING_ROLES, north_scada
from wind_up import TurbineUplift, farm_uplift
from wind_up.analysis_period import DEFAULT_PLAN_SETTINGS
from wind_up.analysis_warnings import AnalysisWarning, ranked
from wind_up.northing import DEFAULT_NORTHING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import numpy.typing as npt

    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.harness import Method, MethodOutput
    from benchmarking.synthetic import ColumnSchema
    from wind_up import FarmUplift
    from wind_up.analysis_period import AnalysisPlan, PlanSettings
    from wind_up.northing import NorthingSettings

logger = logging.getLogger(__name__)

# Each table's columns, named here so a campaign with nothing to report still returns a frame
# consumers can index.
_PER_TURBINE_COLUMNS = ("method", "test_wtg", "estimate", "sigma")
_FARM_COLUMNS = (
    "method",
    "estimate",
    "uplift_spread",
    "n_guarded",
    "reference_uplift_before_screen",
    "reference_uplift_after_screen",
    "reference_uplift_spread",
)
_REFERENCE_COLUMNS = (
    "method",
    "test_wtg",
    "turbine",
    "uplift",
    "sigma",
    "actual_energy",
    "n_records",
    "screened",
    "unjudged",
)
_CONDITIONAL_COLUMNS = ("method", "test_wtg", "condition", "condition_bin", "p50_uplift")


@dataclass
class CampaignReport:
    """Everything a truth-free campaign run produced.

    :param spec: the campaign that was run
    :param scada_df: the frame the methods were given -- every turbine, clipped to the analysis
        period and north-calibrated by the shared northing step
    :param per_turbine: one row per method and upgraded turbine -- ``method``, ``test_wtg``,
        ``estimate`` and ``sigma`` (NaN for a method that reports no uncertainty), and the power
        model's reference correction and sigma components where it reports them (``reference_correction``,
        the kept references' mean uncorrected reading, ``sigma_sampling``, ``sigma_reference``,
        ``sigma_holdout`` and the counts behind them)
    :param farm: one row per method -- ``estimate``, ``uplift_spread``, ``n_guarded`` and the
        combined reference uplift before and after the screen (``reference_uplift_before_screen``,
        ``reference_uplift_after_screen``) with the spread of the references after it
        (``reference_uplift_spread``)
    :param farm_uplifts: each method's :class:`~wind_up.FarmUplift`, including per-turbine detail
    :param reference_stability: each method's per-reference self-uplift, the campaign's own
        analysis turned on the turbines it used as references. A healthy campaign reads near 0%.
    :param conditional: each method's per-condition estimates, empty when none reports any
    :param outputs: each ``(method, turbine)``'s raw :class:`~benchmarking.harness.MethodOutput`
    :param wall_time_s: how long each ``(method, turbine)`` estimate took
    :param timeline: each upgraded turbine's analysis plan, or the reason none could be chosen,
        with the works and data extents they were chosen from and the warnings against them
    :param data_screens: each analysed turbine's data screen, empty when the screen is off
    :param warnings: every analysis warning the run raised, ranked by severity
    """

    spec: CampaignSpec
    scada_df: pd.DataFrame
    per_turbine: pd.DataFrame
    farm: pd.DataFrame
    farm_uplifts: dict[str, FarmUplift]
    reference_stability: pd.DataFrame
    conditional: pd.DataFrame
    outputs: dict[tuple[str, str], MethodOutput]
    wall_time_s: dict[tuple[str, str], float]
    timeline: CampaignPlans = field(default_factory=CampaignPlans)
    data_screens: dict[str, DataScreen] = field(default_factory=dict)
    warnings: tuple[AnalysisWarning, ...] = ()

    @property
    def plans(self) -> dict[str, AnalysisPlan]:
        """Each analysed turbine's plan."""
        return self.timeline.plans

    @property
    def reference_data(self) -> pd.DataFrame:
        """Every analysed turbine's candidates with their usable hours, correlation and verdict."""
        return reference_data_table(self.data_screens)

    @property
    def reference_hours(self) -> pd.DataFrame:
        """Every analysed turbine's joint usable hours against the number of references."""
        return reference_hours_table(self.data_screens)

    @property
    def unplanned(self) -> dict[str, str]:
        """Each upgraded turbine no plan could be chosen for or no measurement made of, with the reason."""
        return self.timeline.unplanned


def visible_mask(spec: CampaignSpec, frame: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Rows of ``frame`` inside the declared period that are usable or held back.

    Held-back rows are a turbine's own exclusions: kept in the frame, marked invalid for uplift.
    """
    usable, held = _row_masks(spec, frame)
    return usable | held


def _row_masks(spec: CampaignSpec, frame: pd.DataFrame) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Rows inside the declared period that are usable, and those held back."""
    bounds = spec.period_bounds()
    in_period = (
        np.ones(len(frame), dtype=bool)
        if bounds is None
        else np.asarray((frame.index >= bounds[0]) & (frame.index < bounds[1]))
    )
    usable = in_period.copy()
    held = np.zeros(len(frame), dtype=bool)
    turbines = frame[spec.turbine_col].to_numpy()
    for turbine in pd.unique(turbines):
        is_turbine = turbines == turbine
        rows = pd.DatetimeIndex(frame.index[is_turbine])
        usable[is_turbine] &= spec.usable_mask(str(turbine), rows)
        held[is_turbine] = in_period[is_turbine] & spec.held_back_mask(str(turbine), rows)
    return usable, held


def visible_scada(
    spec: CampaignSpec,
    frame: pd.DataFrame,
    *,
    columns: ColumnSchema,
    era5_wd: pd.Series | None = None,
    roles: Sequence[str] = DEFAULT_NORTHING_ROLES,
    settings: NorthingSettings = DEFAULT_NORTHING,
    out_dir: Path | None = None,
    northing_plots: bool = True,
) -> pd.DataFrame:
    """Return ``frame`` cut to what a method may see, north-calibrated.

    The shared northing step runs here, farm-wide and once, so every method downstream inherits
    the north-calibrated direction rather than each hand-rolling one. Discovery learns from the
    usable rows only; held-back rows are corrected too.

    :param era5_wd: reanalysis wind direction, the anchor the shared step discovers against.
        Required when ``spec.north_offsets`` is ``None``; a declared table needs none.
    :param out_dir: where the shared step writes its table and plots when it discovers corrections
    :param northing_plots: write the plots as well as the table
    """
    usable, held = _row_masks(spec, frame)
    visible = frame[usable | held]
    return north_scada(
        frame[usable],
        columns=columns,
        north_offsets=spec.north_offsets,
        rated_power_kw=spec.rated_power_kw,
        layout=spec.layout,
        era5_wd=era5_wd,
        roles=roles,
        settings=settings,
        out_dir=out_dir,
        plots=northing_plots,
        apply_to=visible,
    )


class NoTurbineMeasuredError(ValueError):
    """No upgraded turbine of a campaign could be measured; ``unmeasured`` gives each one's reason."""

    def __init__(self, unmeasured: Mapping[str, str]) -> None:
        """Keep each unmeasured turbine's reason and name them all in the message."""
        self.unmeasured = dict(unmeasured)
        super().__init__(
            "no upgraded turbine could be measured: " + "; ".join(f"{t}: {r}" for t, r in unmeasured.items())
        )


def _measure(method: Method, method_input: MethodInput) -> MethodOutput | str:
    """Return the method's output, or why it could not measure the turbine.

    A ``ValueError`` from the estimate is the turbine's own (too few usable hours, no reference) and
    comes back as the reason; a :class:`MethodConfigurationError` is the method's configuration,
    wrong for every turbine, and propagates to fail the campaign.
    """
    try:
        return method.estimate(method_input)
    except MethodConfigurationError:
        raise
    except ValueError as e:
        logger.warning("%s: not measured by %s: %s", method_input.test_wtg, method.name, e)
        return f"not measured by {method.name}: {e}"


def estimate_campaign(
    spec: CampaignSpec,
    scada_df: pd.DataFrame,
    *,
    build_methods: Callable[[str], list[Method]],
    columns: ColumnSchema,
    era5_wd: pd.Series | None = None,
    northing_roles: Sequence[str] = DEFAULT_NORTHING_ROLES,
    northing_settings: NorthingSettings = DEFAULT_NORTHING,
    northing_out_dir: Path | None = None,
    northing_plots: bool = True,
    plan_settings: PlanSettings = DEFAULT_PLAN_SETTINGS,
    possible_changes: Mapping[str, Sequence[pd.Timestamp]] | None = None,
    data_screen: DataScreenSettings | None = DEFAULT_DATA_SCREEN,
) -> CampaignReport:
    """Estimate every applicable method on every upgraded turbine and aggregate to one headline.

    Each upgraded turbine runs over its own span against its own power references, chosen from
    the campaign timeline or used as declared.
    A turbine a method cannot measure (its estimate raises ``ValueError``) is reported in
    ``unplanned`` with the reason, and the campaign continues without it.

    :param spec: the public campaign facts; methods see nothing else
    :param scada_df: long-format source-native SCADA covering the campaign
    :param build_methods: given an upgraded turbine's name, the methods to run for it
    :param columns: the source-native schema ``scada_df`` is keyed by
    :param era5_wd: reanalysis wind direction for the shared northing step
    :param northing_roles: the direction roles the shared step corrects
    :param northing_settings: how the shared step's changepoint search is bounded
    :param northing_out_dir: where the shared step writes its table and plots when it discovers corrections
    :param northing_plots: write the shared step's plots as well as its table
    :param plan_settings: how each turbine's span and power references are chosen
    :param possible_changes: each turbine's indicators of an undeclared change, reported as warnings
        against the plans they fall inside
    :param data_screen: how each plan's power references and reserves are screened on their data;
        ``None`` uses the plans as chosen
    """
    resolved = plans_for(spec, scada_df, columns=columns, settings=plan_settings, possible_changes=possible_changes)
    plans, unplanned = resolved.plans, resolved.unplanned
    if not plans:
        reasons = "; ".join(unplanned[t] for t in sorted(unplanned))
        msg = f"no upgraded turbine could be planned: {reasons}"
        raise ValueError(msg)
    visible = visible_scada(
        spec,
        scada_df,
        columns=columns,
        era5_wd=era5_wd,
        roles=northing_roles,
        settings=northing_settings,
        out_dir=northing_out_dir,
        northing_plots=northing_plots,
    )

    outputs: dict[tuple[str, str], MethodOutput] = {}
    wall_time_s: dict[tuple[str, str], float] = {}
    estimates: dict[str, list[TurbineUplift]] = {}
    per_turbine_rows: list[dict[str, object]] = []
    reference_frames: list[pd.DataFrame] = []
    conditional_frames: list[pd.DataFrame] = []
    screens: dict[str, DataScreen] = {}
    unmeasured: dict[str, str] = {}

    # Sorted rather than in declaration order, so a glance at the log says how far through the
    # campaign a run is, whatever order the turbines were declared in.
    for wtg, plan in sorted(plans.items()):
        timing = spec.timing_for(wtg)
        frame = visible[(visible.index >= plan.start) & (visible.index < plan.end)]
        if data_screen is not None:
            screens[wtg] = screen_plan(plan, frame, spec=spec, columns=columns, settings=data_screen)
        context = context_for_plan(spec, screens[wtg].plan if wtg in screens else plan, scada_df=frame)
        rows = frame[frame[spec.turbine_col] == wtg]
        row_index = pd.DatetimeIndex(rows.index)
        treated = treated_activity(row_index, timing=timing, window=plan.after)
        treated &= spec.usable_mask(wtg, row_index)
        energy, n_records = _actual_energy(rows, columns=columns, mask=treated)

        measured: list[tuple[str, MethodOutput, float]] = []
        for method in build_methods(wtg):
            method_input = MethodInput(
                scada_df=frame,
                test_wtg=wtg,
                turbine_col=spec.turbine_col,
                campaign_context=context,
            )
            start = time.perf_counter()
            output = _measure(method, method_input)
            if isinstance(output, str):
                unmeasured[wtg] = output
                break
            measured.append((method.name, output, time.perf_counter() - start))
        if wtg in unmeasured:
            continue
        for name, output, seconds in measured:
            wall_time_s[name, wtg] = seconds
            outputs[name, wtg] = output
            per_turbine_rows.append(
                {
                    "method": name,
                    "test_wtg": wtg,
                    "estimate": output.p50_overall,
                    "sigma": float("nan") if output.sigma_overall is None else float(output.sigma_overall),
                    **headline_detail(output),
                }
            )
            estimates.setdefault(name, []).append(
                TurbineUplift(
                    turbine=wtg,
                    uplift=output.p50_overall,
                    actual_energy=energy,
                    n_records=n_records,
                    rated_power_kw=spec.rated_power_kw,
                )
            )
            if output.reference_uplifts is not None:
                reference_frames.append(output.reference_uplifts.assign(method=name, test_wtg=wtg))
            if output.p50_by_condition is not None:
                conditional_frames.append(output.p50_by_condition.assign(method=name, test_wtg=wtg))

    if unmeasured:
        if len(unmeasured) == len(plans):
            raise NoTurbineMeasuredError(unmeasured)
        resolved = replace(
            resolved,
            plans={t: p for t, p in plans.items() if t not in unmeasured},
            unplanned={**unplanned, **unmeasured},
        )
    farm_uplifts = {name: farm_uplift(rows) for name, rows in estimates.items()}
    stability = _stack(reference_frames, lead=_REFERENCE_COLUMNS)
    headlines = reference_headlines(stability, rated_power_kw=spec.rated_power_kw)
    return CampaignReport(
        spec=spec,
        scada_df=visible,
        per_turbine=pd.DataFrame(per_turbine_rows, columns=list(_PER_TURBINE_COLUMNS)),
        farm=pd.DataFrame(
            [_farm_row(name, result, references=headlines.get(name)) for name, result in farm_uplifts.items()],
            columns=list(_FARM_COLUMNS),
        ),
        farm_uplifts=farm_uplifts,
        reference_stability=stability,
        conditional=_stack(conditional_frames, lead=_CONDITIONAL_COLUMNS),
        outputs=outputs,
        wall_time_s=wall_time_s,
        timeline=resolved,
        data_screens=screens,
        warnings=analysis_warnings(
            outputs=outputs, screens=screens, timeline=resolved, possible_changes=possible_changes or {}
        ),
    )


def analysis_warnings(
    *,
    outputs: Mapping[tuple[str, str], MethodOutput],
    screens: Mapping[str, DataScreen],
    timeline: CampaignPlans,
    possible_changes: Mapping[str, Sequence[pd.Timestamp]],
) -> tuple[AnalysisWarning, ...]:
    """Return every analysis warning of a run, ranked by severity.

    The methods' own warnings are kept as raised. A reference the reference screen ruled out or the data
    screen dropped is a medium warning of step 9 against its test turbine; an indicator of an undeclared
    change inside a plan's periods a medium warning of step 8; a candidate yaw-alignment change a low
    warning of step 4 against that turbine.
    """
    found: list[AnalysisWarning] = []
    several = len({name for name, _ in outputs}) > 1
    for (name, wtg), output in outputs.items():
        found.extend(output.warnings)
        references = output.reference_uplifts
        if references is None or "screened" not in references.columns:
            continue
        by = f" by {name}" if several else ""
        found.extend(
            AnalysisWarning("medium", 9, wtg, f"the reference screen ruled out {reference}{by}")
            for reference in references.loc[references["screened"].astype(bool), "turbine"]
        )
    for wtg, screen in screens.items():
        dropped = screen.table[screen.table["verdict"] == DROPPED]
        found.extend(
            AnalysisWarning("medium", 9, wtg, f"the data screen dropped {candidate}: {reason}")
            for candidate, reason in zip(dropped["candidate"], dropped["reason"], strict=True)
        )
    for wtg, others in timeline.warnings.items():
        found.extend(AnalysisWarning("medium", 8, wtg, f"{other}: {message}") for other, message in others.items())
    for turbine, times in possible_changes.items():
        found.extend(
            AnalysisWarning("low", 4, turbine, f"a candidate yaw alignment change on {pd.Timestamp(t):%Y-%m-%d}")
            for t in times
        )
    return tuple(ranked(found))


def _farm_row(method: str, result: FarmUplift, *, references: tuple[float, float, float] | None) -> dict[str, object]:
    """One method's farm headline row; ``references`` is from :func:`reference_headlines`."""
    nan = float("nan")
    before, after, spread = references if references is not None else (nan, nan, nan)
    return {
        "method": method,
        "estimate": result.uplift,
        "uplift_spread": result.uplift_spread,
        "n_guarded": int((result.turbines["guard"] != "").sum()),
        "reference_uplift_before_screen": before,
        "reference_uplift_after_screen": after,
        "reference_uplift_spread": spread,
    }


def reference_headlines(stability: pd.DataFrame, *, rated_power_kw: float) -> dict[str, tuple[float, float, float]]:
    """Return each method's combined reference uplift before and after the screen, and the spread after it.

    Each reference counts once per method, as the median of its readings across the test turbines;
    it is screened or unjudged when any test turbine's screen says so. The spread is the largest
    minus the smallest uplift of the references the screen kept, NaN below two (method step 16).
    """
    if stability.empty:
        return {}
    flags = {c: (c, "any") for c in ("screened", "unjudged") if c in stability.columns}
    once = stability.groupby(["method", "turbine"], as_index=False).agg(
        uplift=("uplift", "median"),
        actual_energy=("actual_energy", "median"),
        n_records=("n_records", "median"),
        **flags,
    )
    return {
        str(method): (
            reference_overall_uplift(rows, rated_power_kw=rated_power_kw, after_screen=False),
            reference_overall_uplift(rows, rated_power_kw=rated_power_kw, after_screen=True),
            _spread(rows),
        )
        for method, rows in once.groupby("method")
    }


def _spread(once: pd.DataFrame) -> float:
    """Return the largest minus the smallest finite uplift of the references the screen kept; NaN below two."""
    kept = once[np.isfinite(once["uplift"].to_numpy(dtype=float))]
    for column in ("screened", "unjudged"):
        if column in kept.columns:
            kept = kept[~kept[column].astype(bool)]
    values = kept["uplift"].to_numpy(dtype=float)
    return float(values.max() - values.min()) if len(values) > 1 else float("nan")


def _stack(frames: list[pd.DataFrame], *, lead: Sequence[str]) -> pd.DataFrame:
    """Concatenate per-method frames with ``lead`` first, or return an empty frame with ``lead``.

    Columns a method adds beyond ``lead`` are kept, after them.
    """
    if not frames:
        return pd.DataFrame(columns=list(lead))
    stacked = pd.concat(frames, ignore_index=True)
    ordered = [c for c in lead if c in stacked.columns]
    return stacked[[*ordered, *[c for c in stacked.columns if c not in ordered]]]


def _actual_energy(rows: pd.DataFrame, *, columns: ColumnSchema, mask: npt.NDArray[np.bool_]) -> tuple[float, int]:
    """Return the energy one turbine actually produced over its upgraded records.

    Finite records only, with the record count that sum covers.
    """
    power = rows[columns.active_power].to_numpy(dtype=float)
    selected = mask & np.isfinite(power)
    return float(power[selected].sum()), int(selected.sum())
