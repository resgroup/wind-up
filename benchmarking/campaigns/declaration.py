"""What a campaign is: the private declaration and the public spec derived from it.

``SyntheticCampaign`` holds the injected upgrades and so is ground truth; ``CampaignSpec``
carries only the facts an analyst would know and is what methods are given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule, generate_dataset
from wind_up.layout import LATITUDE_COL, LONGITUDE_COL, NAME_COL, ROTOR_DIAMETER_COL, Layout

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import numpy.typing as npt

    from benchmarking.synthetic import ColumnSchema, SyntheticDataset


Window = tuple[pd.Timestamp, pd.Timestamp]
# (turbine, start, end), end exclusive; turbine None is farm-wide.
Exclusion = tuple[str | None, pd.Timestamp, pd.Timestamp]


def in_windows(index: pd.DatetimeIndex, windows: Iterable[Window]) -> npt.NDArray[np.bool_]:
    """Boolean mask over ``index`` of the records inside any of ``windows``, ends exclusive."""
    inside = np.zeros(len(index), dtype=bool)
    for start, end in windows:
        inside |= np.asarray((index >= start) & (index < end))
    return inside


def layout_from_coords(coords: Mapping[str, tuple[float, float]], *, rotor_diameter_m: float) -> Layout:
    """Return the :class:`~wind_up.layout.Layout` of ``coords`` with every rotor ``rotor_diameter_m`` across."""
    return Layout.from_frame(
        pd.DataFrame(
            {
                NAME_COL: list(coords),
                LATITUDE_COL: [lat for lat, _ in coords.values()],
                LONGITUDE_COL: [lon for _, lon in coords.values()],
                ROTOR_DIAMETER_COL: rotor_diameter_m,
            }
        )
    )


def layout_coords(layout: Layout) -> dict[str, tuple[float, float]]:
    """Turbine name to ``(latitude, longitude)`` for every named turbine in ``layout``."""
    frame = layout.frame
    return {
        str(name): (float(lat), float(lon))
        for name, lat, lon in zip(frame[NAME_COL], frame[LATITUDE_COL], frame[LONGITUDE_COL], strict=True)
        if name is not None
    }


@dataclass(frozen=True)
class CampaignSpec:
    """The public facts of a campaign -- everything a method may see, and nothing else.

    Read per-turbine facts through :meth:`timing_for` and :meth:`usable_mask` rather than the
    flat fields, and the mode through :attr:`mode` rather than the type of ``upgrade_timing``.

    :param upgraded_turbines: the turbines whose uplift is being estimated
    :param upgrade_timing: the shared changeover timestamp (prepost), the ``ToggleSchedule``
        (toggle), or ``None`` when each upgraded turbine's changeover is the end of its works window
    :param candidate_references: turbines a method may use as references
    :param excluded_turbines: turbines never tested and never offered as a reference. Their data
        still enters every estimate for their wake, as every turbine's does.
    :param layout: the farm layout, rotor diameters included; :attr:`coords` reads its positions
    :param north_offsets: step-applied northing corrections, ``(turbine, from, offset_deg)``.
        ``None`` (the default) means the analyst supplied none and the shared northing step
        discovers them from the data -- the usual case. A list, **including an empty one**,
        is applied exactly as given and nothing is discovered.
    :param rated_power_kw: the turbines' rated power
    :param analysis_period: ``(start, end)`` of the whole record, end exclusive; a span per
        upgraded turbine; or ``None`` for wind-up to choose each upgraded turbine's span
    :param turbine_col: the turbine-identifier column of the SCADA frame
    :param works: works windows per turbine, any turbine, analysed or not
    :param exclusions: ``(turbine, start, end)`` periods whose data is not used; turbine ``None``
        is farm-wide
    :param references_declared: ``candidate_references`` was declared rather than defaulted, so a
        declared span may force them in as power references
    """

    upgraded_turbines: list[str]
    upgrade_timing: pd.Timestamp | ToggleSchedule | None
    candidate_references: list[str]
    excluded_turbines: list[str]
    layout: Layout
    north_offsets: list[tuple[str, pd.Timestamp, float]] | None
    rated_power_kw: float
    analysis_period: Window | dict[str, Window] | None
    turbine_col: str = HOT_COLUMNS.turbine
    works: dict[str, list[Window]] = field(default_factory=dict)
    exclusions: list[Exclusion] = field(default_factory=list)
    references_declared: bool = False

    def __post_init__(self) -> None:
        """Raise when the timing does not give every upgraded turbine one changeover."""
        if isinstance(self.upgrade_timing, ToggleSchedule) and not isinstance(self.analysis_period, tuple):
            msg = "a toggle campaign needs one declared analysis_period (start, end)"
            raise ValueError(msg)  # noqa: TRY004 - a missing declaration, not a wrong argument type
        if self.upgrade_timing is None:
            wrong = sorted(t for t in self.upgraded_turbines if len(self.works.get(t, [])) != 1)
            if wrong:
                msg = (
                    f"{wrong} need exactly one works window each: with no shared changeover, an upgraded "
                    f"turbine's changeover is the end of its works window"
                )
                raise ValueError(msg)

    @property
    def coords(self) -> dict[str, tuple[float, float]]:
        """Turbine name to ``(latitude, longitude)`` in degrees, from :attr:`layout`."""
        return layout_coords(self.layout)

    @property
    def mode(self) -> Literal["prepost", "toggle"]:
        """``"toggle"`` for a scheduled campaign, ``"prepost"`` for a single changeover."""
        return "toggle" if isinstance(self.upgrade_timing, ToggleSchedule) else "prepost"

    @property
    def treatment_start(self) -> pd.Timestamp:
        """When treatment begins: the earliest changeover, or when toggling starts."""
        if isinstance(self.upgrade_timing, ToggleSchedule):
            if self.upgrade_timing.start is not None:
                return self.upgrade_timing.start
            start, _ = self.analysis_period  # type: ignore[misc]  # a toggle campaign declares one period
            return start
        return min(self.works_window(t)[1] for t in self.upgraded_turbines)

    @property
    def uses_plans(self) -> bool:
        """Whether each upgraded turbine's span and references are planned rather than declared flat."""
        return self.upgrade_timing is None or not isinstance(self.analysis_period, tuple)

    def timing_for(self, turbine: str) -> pd.Timestamp | ToggleSchedule:
        """Return the upgrade timing of one upgraded turbine."""
        if turbine not in self.upgraded_turbines:
            msg = f"{turbine!r} is not an upgraded turbine of this campaign"
            raise KeyError(msg)
        if self.upgrade_timing is None:
            return self.works[turbine][0][1]
        return self.upgrade_timing

    def changeovers(self) -> dict[str, list[pd.Timestamp]]:
        """Return each turbine's changeover dates: the ends of its works windows and an upgraded turbine's timing.

        Empty for a toggle campaign.
        """
        if self.mode == "toggle":
            return {}
        dates = {turbine: [end for _, end in windows] for turbine, windows in self.works.items()}
        for turbine in self.upgraded_turbines:
            dates.setdefault(turbine, []).append(self.works_window(turbine)[1])
        return {turbine: sorted(set(found)) for turbine, found in dates.items()}

    def works_window(self, turbine: str) -> Window:
        """Return an upgraded turbine's works window; a shared changeover is a window of no length."""
        timing = self.timing_for(turbine)
        if isinstance(timing, ToggleSchedule):
            msg = "a toggle campaign has no works windows"
            raise TypeError(msg)
        return self.works[turbine][0] if self.upgrade_timing is None else (timing, timing)

    def period_for(self, turbine: str) -> Window | None:
        """Return the declared span for ``turbine``, or None when wind-up chooses it."""
        if isinstance(self.analysis_period, dict):
            return self.analysis_period.get(turbine)
        return self.analysis_period

    def period_bounds(self) -> Window | None:
        """Return the outer bounds of every declared span, or None when none is declared."""
        if isinstance(self.analysis_period, dict):
            if not self.analysis_period:
                return None
            spans = list(self.analysis_period.values())
            return min(s for s, _ in spans), max(e for _, e in spans)
        return self.analysis_period

    def usable_mask(self, turbine: str, index: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
        """Boolean mask over ``index`` of the records ``turbine``'s data may be used over.

        Masks the turbine's own works windows, its own exclusions and farm-wide exclusions.
        """
        own = [(start, end) for who, start, end in self.exclusions if who in (None, turbine)]
        return ~in_windows(index, [*self.works.get(turbine, []), *own])

    def held_back_mask(self, turbine: str, index: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
        """Return the records unusable only because of ``turbine``'s own exclusions.

        They stay in the frame, marked invalid, so a method may read the turbine's operating state
        over them.
        """
        own = in_windows(index, [(start, end) for who, start, end in self.exclusions if who == turbine])
        farm_wide = [(start, end) for who, start, end in self.exclusions if who is None]
        return own & ~in_windows(index, [*self.works.get(turbine, []), *farm_wide])

    def change_label(self) -> str:
        """How report and plot titles refer to what is being assessed."""
        return "the change"


@dataclass
class SyntheticCampaign:
    """A declared campaign: its turbines and roles, its timing, and the upgrades to inject.

    Private to the benchmark -- it holds the injected upgrades, which are the ground truth.

    :param upgraded_turbines: turbines to upgrade and estimate
    :param upgrade_timing: changeover timestamp (prepost), ``ToggleSchedule`` (toggle), or ``None``
        when each upgraded turbine is changed at the end of its works window
    :param candidate_references: turbines offered to methods as references
    :param upgrades: the upgrade callables to inject; empty for a placebo
    :param faults: measurement corruptions to inject after the upgrades (an R-series fault such
        as :class:`~benchmarking.synthetic.faults.NorthingStep`). Private ground truth like
        ``upgrades``: ``CampaignSpec`` never carries them, so a method must cope undeclared.
    :param layout: the farm layout, rotor diameters included
    :param north_offsets: step-applied northing corrections, ``(turbine, from, offset_deg)``;
        ``None`` leaves them to be discovered (see :class:`CampaignSpec`)
    :param rated_power_kw: the turbines' rated power
    :param analysis_period: ``(start, end)`` of the whole record, end exclusive
    :param excluded_turbines: turbines never tested and never offered as a reference; their data
        still carries their wake
    :param columns: the source-native column schema the SCADA is keyed by
    :param seed: recorded in the generated dataset's run metadata
    :param works: works windows per turbine (see :class:`CampaignSpec`)
    :param exclusions: excluded periods (see :class:`CampaignSpec`)
    :param other_upgraded: turbines the upgrades are injected into but that are not analysed, each
        changed at the end of its works window, or at the shared changeover
    """

    upgraded_turbines: list[str]
    upgrade_timing: pd.Timestamp | ToggleSchedule | None
    candidate_references: list[str]
    upgrades: list
    layout: Layout
    north_offsets: list[tuple[str, pd.Timestamp, float]] | None
    rated_power_kw: float
    analysis_period: Window | dict[str, Window] | None
    faults: list = field(default_factory=list)
    excluded_turbines: list[str] = field(default_factory=list)
    columns: ColumnSchema = HOT_COLUMNS
    seed: int = 0
    works: dict[str, list[Window]] = field(default_factory=dict)
    exclusions: list[Exclusion] = field(default_factory=list)
    other_upgraded: list[str] = field(default_factory=list)

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
            layout=self.layout,
            north_offsets=None if self.north_offsets is None else list(self.north_offsets),
            rated_power_kw=self.rated_power_kw,
            analysis_period=self.analysis_period,
            turbine_col=self.columns.turbine,
            works={t: list(w) for t, w in self.works.items()},
            exclusions=list(self.exclusions),
        )

    def generate(self, scada_df: pd.DataFrame) -> SyntheticDataset:
        """Inject the declared upgrades into ``scada_df`` over the analysis period.

        Every turbine ``scada_df`` carries is kept, declared or not, since each can wake another.
        Without a declared period the whole frame is kept.
        """
        bounds = self.spec().period_bounds()
        if bounds is not None:
            start, end = bounds
            scada_df = scada_df[(scada_df.index >= start) & (scada_df.index < end)]
        return generate_dataset(
            scada_df=scada_df,
            test_wtgs=[*self.upgraded_turbines, *self.other_upgraded],
            upgrades=list(self.upgrades),
            mode="toggle" if isinstance(self.upgrade_timing, ToggleSchedule) else "prepost",
            upgrade_timing=self._injection_timing(),
            faults=list(self.faults),
            rated_power_kw=self.rated_power_kw,
            columns=self.columns,
            seed=self.seed,
        )

    def _injection_timing(self) -> pd.Timestamp | ToggleSchedule | dict[str, pd.Timestamp]:
        """Return the shared timing, or each upgraded turbine's works end."""
        if self.upgrade_timing is not None:
            return self.upgrade_timing
        return {t: self.works[t][0][1] for t in [*self.upgraded_turbines, *self.other_upgraded]}
