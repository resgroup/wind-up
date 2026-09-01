"""What a campaign is: the private declaration and the public spec derived from it.

``SyntheticCampaign`` holds the injected upgrades and so is ground truth; ``CampaignSpec``
carries only the facts an analyst would know and is what methods are given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule, generate_dataset

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

    from benchmarking.synthetic import ColumnSchema, SyntheticDataset


@dataclass(frozen=True)
class CampaignSpec:
    """The public facts of a campaign -- everything a method may see, and nothing else.

    Read per-turbine facts through :meth:`timing_for` and :meth:`usable_mask` rather than the
    flat fields, and the mode through :attr:`mode` rather than the type of ``upgrade_timing``.

    :param upgraded_turbines: the turbines whose uplift is being estimated
    :param upgrade_timing: the changeover timestamp (prepost) or the ``ToggleSchedule`` (toggle)
    :param candidate_references: turbines a method may use as references
    :param excluded_turbines: turbines whose data must not be used at all
    :param coords: turbine name to ``(latitude, longitude)`` in degrees
    :param north_offsets: step-applied northing corrections, ``(turbine, from, offset_deg)``
    :param rated_power_kw: the turbines' rated power
    :param analysis_period: ``(start, end)`` of the whole record, end exclusive
    :param turbine_col: the turbine-identifier column of the SCADA frame
    """

    upgraded_turbines: list[str]
    upgrade_timing: pd.Timestamp | ToggleSchedule
    candidate_references: list[str]
    excluded_turbines: list[str]
    coords: dict[str, tuple[float, float]]
    north_offsets: list[tuple[str, pd.Timestamp, float]]
    rated_power_kw: float
    analysis_period: tuple[pd.Timestamp, pd.Timestamp]
    turbine_col: str = HOT_COLUMNS.turbine

    @property
    def mode(self) -> Literal["prepost", "toggle"]:
        """``"toggle"`` for a scheduled campaign, ``"prepost"`` for a single changeover."""
        return "toggle" if isinstance(self.upgrade_timing, ToggleSchedule) else "prepost"

    @property
    def treatment_start(self) -> pd.Timestamp:
        """When treatment begins: the changeover, or when toggling starts."""
        if isinstance(self.upgrade_timing, ToggleSchedule):
            return self.upgrade_timing.start if self.upgrade_timing.start is not None else self.analysis_period[0]
        return self.upgrade_timing

    def timing_for(self, turbine: str) -> pd.Timestamp | ToggleSchedule:
        """Return the upgrade timing of one upgraded turbine."""
        if turbine not in self.upgraded_turbines:
            msg = f"{turbine!r} is not an upgraded turbine of this campaign"
            raise KeyError(msg)
        return self.upgrade_timing

    def usable_mask(self, turbine: str, index: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
        """Boolean mask over ``index`` of the records ``turbine``'s data may be used over."""
        return np.full(len(index), turbine not in self.excluded_turbines, dtype=bool)

    def change_label(self) -> str:
        """How report and plot titles refer to what is being assessed."""
        return "the change"


@dataclass
class SyntheticCampaign:
    """A declared campaign: its turbines and roles, its timing, and the upgrades to inject.

    Private to the benchmark -- it holds the injected upgrades, which are the ground truth.

    :param upgraded_turbines: turbines to upgrade and estimate
    :param upgrade_timing: changeover timestamp (prepost) or ``ToggleSchedule`` (toggle)
    :param candidate_references: turbines offered to methods as references
    :param upgrades: the upgrade callables to inject; empty for a placebo
    :param coords: turbine name to ``(latitude, longitude)`` in degrees
    :param north_offsets: step-applied northing corrections, ``(turbine, from, offset_deg)``
    :param rated_power_kw: the turbines' rated power
    :param analysis_period: ``(start, end)`` of the whole record, end exclusive
    :param excluded_turbines: turbines whose data must not be used
    :param columns: the source-native column schema the SCADA is keyed by
    :param seed: recorded in the generated dataset's run metadata
    """

    upgraded_turbines: list[str]
    upgrade_timing: pd.Timestamp | ToggleSchedule
    candidate_references: list[str]
    upgrades: list
    coords: dict[str, tuple[float, float]]
    north_offsets: list[tuple[str, pd.Timestamp, float]]
    rated_power_kw: float
    analysis_period: tuple[pd.Timestamp, pd.Timestamp]
    excluded_turbines: list[str] = field(default_factory=list)
    columns: ColumnSchema = HOT_COLUMNS
    seed: int = 0

    @property
    def turbines(self) -> list[str]:
        """Every declared turbine, upgraded first, in declaration order and without duplicates."""
        seen: dict[str, None] = {}
        for wtg in [*self.upgraded_turbines, *self.candidate_references]:
            seen.setdefault(wtg, None)
        return list(seen)

    def spec(self) -> CampaignSpec:
        """Derive the public spec: the same campaign with the injected upgrades dropped."""
        return CampaignSpec(
            upgraded_turbines=list(self.upgraded_turbines),
            upgrade_timing=self.upgrade_timing,
            candidate_references=list(self.candidate_references),
            excluded_turbines=list(self.excluded_turbines),
            coords=dict(self.coords),
            north_offsets=list(self.north_offsets),
            rated_power_kw=self.rated_power_kw,
            analysis_period=self.analysis_period,
            turbine_col=self.columns.turbine,
        )

    def generate(self, scada_df: pd.DataFrame) -> SyntheticDataset:
        """Inject the declared upgrades into ``scada_df`` over the analysis period."""
        start, end = self.analysis_period
        in_period = (scada_df.index >= start) & (scada_df.index < end)
        declared = scada_df[self.columns.turbine].isin(self.turbines).to_numpy()
        return generate_dataset(
            scada_df=scada_df[in_period & declared],
            test_wtgs=list(self.upgraded_turbines),
            upgrades=list(self.upgrades),
            mode="toggle" if isinstance(self.upgrade_timing, ToggleSchedule) else "prepost",
            upgrade_timing=self.upgrade_timing,
            rated_power_kw=self.rated_power_kw,
            columns=self.columns,
            seed=self.seed,
        )
