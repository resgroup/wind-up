"""The truth-free campaign core: estimate a declared campaign from SCADA alone.

Nothing here reads ground truth, so this is the path a real campaign runs on. The benchmark
layers truth, scoring and truth-vs-estimate plots on top; see
:mod:`benchmarking.campaigns.runner`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.campaigns.context import context_for
from benchmarking.harness import MethodInput
from benchmarking.harness.northing import DEFAULT_NORTHING_ROLES, north_scada
from benchmarking.synthetic import treated_mask
from wind_up import TurbineUplift, farm_uplift
from wind_up.northing import DEFAULT_NORTHING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import numpy.typing as npt

    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.harness import Method, MethodOutput
    from benchmarking.synthetic import ColumnSchema, ToggleSchedule
    from wind_up import FarmUplift
    from wind_up.northing import NorthingSettings

# Each table's columns, named here so a campaign with nothing to report still returns a frame
# consumers can index.
_PER_TURBINE_COLUMNS = ("method", "test_wtg", "estimate")
_FARM_COLUMNS = ("method", "estimate", "uplift_spread", "n_guarded")
_REFERENCE_COLUMNS = ("method", "test_wtg", "turbine", "uplift", "actual_energy", "n_records", "screened")
_CONDITIONAL_COLUMNS = ("method", "test_wtg", "condition", "condition_bin", "p50_uplift")


@dataclass
class CampaignReport:
    """Everything a truth-free campaign run produced.

    :param spec: the campaign that was run
    :param scada_df: the frame the methods were given -- clipped to the analysis period, the
        excluded turbines dropped, and north-calibrated by the shared northing step
    :param per_turbine: one row per method and upgraded turbine -- ``method``, ``test_wtg``,
        ``estimate``
    :param farm: one row per method -- ``estimate``, ``uplift_spread``, ``n_guarded``
    :param farm_uplifts: each method's :class:`~wind_up.FarmUplift`, including per-turbine detail
    :param reference_stability: each method's per-reference self-uplift, the campaign's own
        analysis turned on the turbines it used as references. A healthy campaign reads near 0%.
    :param conditional: each method's per-condition estimates, empty when none reports any
    :param outputs: each ``(method, turbine)``'s raw :class:`~benchmarking.harness.MethodOutput`
    :param wall_time_s: how long each ``(method, turbine)`` estimate took
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


def visible_mask(spec: CampaignSpec, frame: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Rows of ``frame`` inside the analysis period whose turbine may be used."""
    start, end = spec.analysis_period
    keep = np.asarray((frame.index >= start) & (frame.index < end))
    turbines = frame[spec.turbine_col].to_numpy()
    for turbine in pd.unique(turbines):
        is_turbine = turbines == turbine
        rows = pd.DatetimeIndex(frame.index[is_turbine])
        keep[is_turbine] &= spec.usable_mask(str(turbine), rows)
    return keep


def visible_scada(
    spec: CampaignSpec,
    frame: pd.DataFrame,
    *,
    columns: ColumnSchema,
    era5_wd: pd.Series | None = None,
    roles: Sequence[str] = DEFAULT_NORTHING_ROLES,
    settings: NorthingSettings = DEFAULT_NORTHING,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Return ``frame`` cut to what a method may see, north-calibrated.

    The shared northing step runs here, farm-wide and once, so every method downstream inherits
    the north-calibrated direction rather than each hand-rolling one.

    :param era5_wd: reanalysis wind direction, the anchor the shared step discovers against.
        Required when ``spec.north_offsets`` is ``None``; a declared table needs none.
    :param out_dir: where the shared step writes its plots when it discovers corrections
    """
    return north_scada(
        frame[visible_mask(spec, frame)],
        columns=columns,
        north_offsets=spec.north_offsets,
        rated_power_kw=spec.rated_power_kw,
        era5_wd=era5_wd,
        roles=roles,
        settings=settings,
        out_dir=out_dir,
    )


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
) -> CampaignReport:
    """Estimate every applicable method on every upgraded turbine and aggregate to one headline.

    :param spec: the public campaign facts; methods see nothing else
    :param scada_df: long-format source-native SCADA covering the campaign
    :param build_methods: given an upgraded turbine's name, the methods to run for it
    :param columns: the source-native schema ``scada_df`` is keyed by
    :param era5_wd: reanalysis wind direction for the shared northing step
    :param northing_roles: the direction roles the shared step corrects
    :param northing_settings: how the shared step's changepoint search is bounded
    :param northing_out_dir: where the shared step writes its plots when it discovers corrections
    """
    visible = visible_scada(
        spec,
        scada_df,
        columns=columns,
        era5_wd=era5_wd,
        roles=northing_roles,
        settings=northing_settings,
        out_dir=northing_out_dir,
    )

    outputs: dict[tuple[str, str], MethodOutput] = {}
    wall_time_s: dict[tuple[str, str], float] = {}
    estimates: dict[str, list[TurbineUplift]] = {}
    per_turbine_rows: list[dict[str, object]] = []
    reference_frames: list[pd.DataFrame] = []
    conditional_frames: list[pd.DataFrame] = []

    for wtg in spec.upgraded_turbines:
        timing = spec.timing_for(wtg)
        rows = visible[visible[spec.turbine_col] == wtg]
        treated = _treated_activity(pd.DatetimeIndex(rows.index), timing=timing, spec=spec)
        energy, n_records = _actual_energy(rows, columns=columns, mask=treated)
        context = context_for(spec, turbine=wtg, scada_df=visible)

        for method in build_methods(wtg):
            method_input = MethodInput(
                scada_df=visible,
                test_wtg=wtg,
                turbine_col=spec.turbine_col,
                campaign_context=context,
            )
            start = time.perf_counter()
            output = method.estimate(method_input)
            wall_time_s[method.name, wtg] = time.perf_counter() - start
            outputs[method.name, wtg] = output
            per_turbine_rows.append({"method": method.name, "test_wtg": wtg, "estimate": output.p50_overall})
            estimates.setdefault(method.name, []).append(
                TurbineUplift(
                    turbine=wtg,
                    uplift=output.p50_overall,
                    actual_energy=energy,
                    n_records=n_records,
                    rated_power_kw=spec.rated_power_kw,
                )
            )
            if output.reference_uplifts is not None:
                reference_frames.append(output.reference_uplifts.assign(method=method.name, test_wtg=wtg))
            if output.p50_by_condition is not None:
                conditional_frames.append(output.p50_by_condition.assign(method=method.name, test_wtg=wtg))

    farm_uplifts = {name: farm_uplift(rows) for name, rows in estimates.items()}
    return CampaignReport(
        spec=spec,
        scada_df=visible,
        per_turbine=pd.DataFrame(per_turbine_rows, columns=list(_PER_TURBINE_COLUMNS)),
        farm=pd.DataFrame(
            [_farm_row(name, result) for name, result in farm_uplifts.items()], columns=list(_FARM_COLUMNS)
        ),
        farm_uplifts=farm_uplifts,
        reference_stability=_stack(reference_frames, lead=_REFERENCE_COLUMNS),
        conditional=_stack(conditional_frames, lead=_CONDITIONAL_COLUMNS),
        outputs=outputs,
        wall_time_s=wall_time_s,
    )


def _farm_row(method: str, result: FarmUplift) -> dict[str, object]:
    """One method's farm headline row."""
    return {
        "method": method,
        "estimate": result.uplift,
        "uplift_spread": result.uplift_spread,
        "n_guarded": int((result.turbines["guard"] != "").sum()),
    }


def _stack(frames: list[pd.DataFrame], *, lead: Sequence[str]) -> pd.DataFrame:
    """Concatenate per-method frames with ``lead`` first, or return an empty frame with ``lead``.

    Columns a method adds beyond ``lead`` are kept, after them.
    """
    if not frames:
        return pd.DataFrame(columns=list(lead))
    stacked = pd.concat(frames, ignore_index=True)
    ordered = [c for c in lead if c in stacked.columns]
    return stacked[[*ordered, *[c for c in stacked.columns if c not in ordered]]]


def _treated_activity(
    index: pd.DatetimeIndex, *, timing: pd.Timestamp | ToggleSchedule, spec: CampaignSpec
) -> npt.NDArray[np.bool_]:
    """Return the test turbine's treated rows within the campaign's activity period."""
    _, end = spec.analysis_period
    in_activity = np.asarray((index >= spec.treatment_start) & (index < end))
    return treated_mask(index, timing) & in_activity


def _actual_energy(rows: pd.DataFrame, *, columns: ColumnSchema, mask: npt.NDArray[np.bool_]) -> tuple[float, int]:
    """Return the energy one turbine actually produced over its upgraded records.

    Finite records only, with the record count that sum covers.
    """
    power = rows[columns.active_power].to_numpy(dtype=float)
    selected = mask & np.isfinite(power)
    return float(power[selected].sum()), int(selected.sum())
