"""Derive the facts and analyses methods see from a campaign declaration.

The one place a :class:`~benchmarking.campaigns.declaration.CampaignSpec` is turned into the
:class:`~wind_up.facts.WindFarmFacts` and :class:`~wind_up.analysis.AnalysisSpec` methods see, and
so the one place to audit that no ground truth reaches a method.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.campaigns.declaration import layout_coords
from wind_up.analysis import CAMPAIGN, AnalysisSpec
from wind_up.facts import Reanalysis, Upgrade, WindFarmFacts
from wind_up.layout import LATITUDE_COL, LONGITUDE_COL, NAME_COL, ROTOR_DIAMETER_COL, centroid
from wind_up.operating_state import OperatingStateConfig
from wind_up.toggle import ToggleSchedule

if TYPE_CHECKING:
    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.synthetic import ColumnSchema
    from wind_up.analysis_period import AnalysisPlan

logger = logging.getLogger(__name__)

# The name of the upgrade a campaign declares, and of the reanalysis it carries.
UPGRADE = "upgrade"
REANALYSIS = "ERA5"


def campaign_facts(
    spec: CampaignSpec,
    records: pd.DataFrame,
    *,
    columns: ColumnSchema,
    era5_hourly_df: pd.DataFrame | None = None,
    operating_states: OperatingStateConfig | None = None,
) -> WindFarmFacts:
    """Return the facts of a campaign over ``records``.

    The upgraded turbines receive one upgrade per wake reach among them; every other works window is
    an upgrade of its own turbine, installed over that window. Exclusions and label overrides are
    declared as given; ``records`` already carries the labels they lead to. Records of a turbine the
    layout does not hold are left out, with a warning.

    :param records: long-format records of every turbine in the layout, labelled and northed
    :param columns: the column roles of ``records``
    :param era5_hourly_df: reanalysis at the farm's centroid, hourly or aligned to the records
    :param operating_states: the site's operating states, those ``records`` were labelled with; the
        generic states alone when ``None``
    """
    frame = spec.layout.frame
    turbines = pd.DataFrame(
        {
            NAME_COL: frame[NAME_COL].astype(str).to_list(),
            LATITUDE_COL: frame[LATITUDE_COL].astype(float).to_list(),
            LONGITUDE_COL: frame[LONGITUDE_COL].astype(float).to_list(),
            "rated_power_kw": spec.rated_power_kw,
            ROTOR_DIAMETER_COL: frame[ROTOR_DIAMETER_COL].astype(float).to_list(),
        }
    )
    listed = records[spec.turbine_col].astype(str).isin(set(turbines[NAME_COL]))
    if not listed.all():
        unlisted = sorted(set(records.loc[~listed.to_numpy(), spec.turbine_col].astype(str)))
        logger.warning("the layout does not hold %s, so their records are left out", unlisted)
        records = records[listed.to_numpy()]
    reanalysis: tuple[Reanalysis, ...] = ()
    if era5_hourly_df is not None:
        lat, lon = centroid(layout_coords(spec.layout))
        reanalysis = (Reanalysis(name=REANALYSIS, latitude=lat, longitude=lon, records=era5_hourly_df),)
    index = pd.DatetimeIndex(records.index.unique()).sort_values()
    timebase = pd.Timedelta(index.to_series().diff().median()) if len(index) > 1 else pd.Timedelta(minutes=10)
    return WindFarmFacts(
        records=records,
        columns=columns,
        timebase=timebase,
        turbines=turbines,
        upgrades=_upgrades(spec),
        operating_states=operating_states if operating_states is not None else OperatingStateConfig(),
        exclusions=list(spec.exclusions),
        operating_state_overrides=list(spec.label_overrides),
        north_offsets=list(spec.north_offsets or []),
        reanalysis=reanalysis,
    )


def upgrade_of(spec: CampaignSpec, turbine: str) -> str:
    """Return the name of the upgrade ``turbine`` received in :func:`campaign_facts`."""
    reaches = _reaches(spec)
    return UPGRADE if len(set(reaches.values())) == 1 else f"{UPGRADE} ({reaches[turbine]:g} D)"


def plan_analysis(spec: CampaignSpec, plan: AnalysisPlan) -> AnalysisSpec:
    """Return the analysis of ``plan.turbine`` over its planned span.

    Its references are the plan's power references, then its reserves, so a method taking the first
    ``len(plan.power_references)`` as its pool keeps the rest as reserves.
    """
    return AnalysisSpec(
        name=plan.turbine,
        upgrade=upgrade_of(spec, plan.turbine),
        subjects=[plan.turbine],
        references=[*plan.power_references, *plan.reserves],
        period=(plan.start, plan.end),
        report=CAMPAIGN,
    )


def _reaches(spec: CampaignSpec) -> dict[str, float]:
    """Return each upgraded turbine's wake reach, in rotor diameters."""
    if isinstance(spec.upgrade_timing, ToggleSchedule):
        start = spec.treatment_start
        return {t: spec.reach_of(t, (start, start)) for t in spec.upgraded_turbines}
    return {t: spec.reach_of(t, spec.works_window(t)) for t in spec.upgraded_turbines}


def _upgrades(spec: CampaignSpec) -> list[Upgrade]:
    """Return the campaign's upgrades: the upgraded turbines', and one per other works window."""
    reaches = _reaches(spec)
    groups: dict[float, list[str]] = {}
    for turbine in spec.upgraded_turbines:
        groups.setdefault(reaches[turbine], []).append(turbine)
    upgrades = []
    timing = spec.upgrade_timing
    for reach, turbines in groups.items():
        name = upgrade_of(spec, turbines[0])
        if isinstance(timing, ToggleSchedule):
            upgrades.append(
                Upgrade(
                    name=name,
                    turbines=turbines,
                    timing=timing,
                    toggle_cycle=timing.period,
                    toggle_datum=timing.start,
                    wake_reach_d=reach,
                )
            )
        elif timing is None:
            upgrades.append(
                Upgrade(
                    name=name,
                    turbines=turbines,
                    installation_windows={t: [spec.works[t][0]] for t in turbines},
                    wake_reach_d=reach,
                )
            )
        else:
            upgrades.append(Upgrade(name=name, turbines=turbines, timing=pd.Timestamp(timing), wake_reach_d=reach))
    upgraded = set(spec.upgraded_turbines)
    for turbine, windows in sorted(spec.works.items()):
        for i, window in enumerate(windows):
            if timing is None and turbine in upgraded:
                continue
            upgrades.append(
                Upgrade(
                    name=f"works {turbine} {i}",
                    turbines=[turbine],
                    installation_windows={turbine: [window]},
                    wake_reach_d=spec.reach_of(turbine, window),
                )
            )
    return upgrades
