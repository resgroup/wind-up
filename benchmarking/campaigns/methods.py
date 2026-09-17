"""Which methods a campaign runs, and how each is configured."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.baselines.naive_ratio import NaiveRatioMethod
from benchmarking.baselines.power_model import CURATED_ERA5_EXCLUDE, TUNED_MODEL_PARAMS, PowerModelMethod
from benchmarking.baselines.toggle_specialist import ToggleSpecialistMethod
from benchmarking.diagnostics.context import ERA5_UNLOCATED
from benchmarking.synthetic import HOT_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.harness import Method


def carried_forward_methods(
    spec: CampaignSpec,
    *,
    out_dir: Path,
    era5_hourly_df: pd.DataFrame | None = None,
    include_power_model: bool = True,
    reference_stat_cols: tuple[str, ...] = (),
    screen_cache: dict | None = None,
    era5_label: str = ERA5_UNLOCATED,
    reference_screen: bool = True,
    report_reference_uplifts: bool = True,
) -> list[Method]:
    """Build the methods applicable to ``spec``, each writing into its own subfolder of ``out_dir``.

    ``toggle_specialist`` accepts only toggle campaigns and is left out of a prepost one. The power
    model reports per-condition estimates only when ``era5_hourly_df`` is given, since its
    conditional step matches on ERA5 weather columns.

    :param spec: the campaign being run
    :param out_dir: the turbine's output folder; each method gets a subfolder named after it
    :param era5_hourly_df: reanalysis for the power model; omit to run it without ERA5 features
    :param include_power_model: build the power model. It is the method under test, so this is
        off only to avoid the ``ml`` dependency group -- a campaign without it is not testing
        v1 wind-up
    :param screen_cache: one dict shared by every turbine's method in a campaign, so the reference
        screen runs once for the campaign rather than once per test turbine. ``None`` screens per
        estimate.
    :param era5_label: how reanalysis is named in output; :func:`era5_source_label` of where it
        was fetched from.
    :param reference_stat_cols: extra per-reference channels the power model carries as features.
        Empty by default: reference anemometry is deliberately not a feature, and the R2 fixture
        passes it in only to measure what that exclusion is worth.
    :param reference_screen: run the power model's reference screen. Off holds the reference pool
        identical across the arms of a study whose campaigns differ in length or coverage.
    :param report_reference_uplifts: report each candidate reference's own uplift. It costs a model
        fit per reference per test turbine and never moves a headline, so a study that scores only
        its test turbines turns it off.
    """
    methods: list[Method] = [NaiveRatioMethod(columns=HOT_COLUMNS, out_dir=out_dir / "naive_ratio", save_plots=True)]
    if spec.mode == "toggle":
        methods.append(
            ToggleSpecialistMethod(
                columns=HOT_COLUMNS,
                out_dir=out_dir / "toggle_specialist",
                save_plots=True,
                conditions=("power",),
                rated_power_kw=spec.rated_power_kw,
            )
        )
    if include_power_model:
        methods.append(
            PowerModelMethod(
                columns=HOT_COLUMNS,
                baseline_rated_power_kw=spec.rated_power_kw,
                era5_hourly_df=era5_hourly_df,
                conditions=PowerModelMethod.conditions if era5_hourly_df is not None else (),
                availability_feature=False,
                era5_exclude=CURATED_ERA5_EXCLUDE,
                reference_stat_cols=reference_stat_cols,
                model_params=dict(TUNED_MODEL_PARAMS),
                out_dir=out_dir / "power_model",
                save_plots=True,
                screen_cache=screen_cache,
                era5_label=era5_label,
                reference_screen=reference_screen,
                report_reference_uplifts=report_reference_uplifts,
            )
        )
    return methods
