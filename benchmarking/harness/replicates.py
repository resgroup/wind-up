"""The replicate ensemble — the precision axis of the harness.

Precision needs an ensemble: a single dataset gives one estimate, hence one error. Holding
the upgrade *profile* fixed, an ensemble is built by varying the base ingredients — the test
turbine and the treatment-start date (the changeover for prepost; when toggling begins for
toggle). The spread of a method's error across these replicates is its precision.

The data is first subset to ``turbine_subset`` (one test turbine is drawn per replicate; the
rest are its references) so each run stays light — important when scoring many replicates.
Draws are a pure deterministic function of ``(StudyConfig, seed)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from benchmarking.harness.campaign import resolve_campaign_grid
from benchmarking.synthetic import HOT_COLUMNS, ToggleSchedule, generate_dataset

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt
    import pandas as pd

    from benchmarking.harness.campaign import CampaignUnit
    from benchmarking.synthetic import ColumnSchema, SyntheticDataset, UpliftResult


@dataclass(frozen=True)
class StudyConfig:
    """One study: a profile evaluated over an ensemble and a campaign-length grid.

    The campaign-length grid is given in **either** months or weeks — exactly one of
    ``campaign_months`` / ``campaign_weeks`` must be set. Months is the original grid (used by the
    overnight studies and their frozen baselines); weeks suits short campaigns, where a month is too
    coarse a step. Read the grid generically via :attr:`campaign_lengths` / :attr:`campaign_unit` /
    :attr:`campaign_length_col` rather than touching the two fields directly.

    :param mode: ``"prepost"`` or ``"toggle"``
    :param turbine_subset: the only turbines kept in the data; per replicate one is drawn as
        the test turbine and the rest are its references
    :param treatment_start_range: ``(earliest, latest)`` treatment-start timestamp to draw from
    :param min_pre_months: fixed baseline length before treatment start, in months (always months,
        independent of the campaign grid's unit)
    :param n_replicates: number of ``(turbine, treatment_start)`` instances to draw
    :param campaign_months: the campaign-length sweep grid, in months (exclusive with ``campaign_weeks``)
    :param campaign_weeks: the campaign-length sweep grid, in weeks (exclusive with ``campaign_months``)
    :param toggle_period: toggle on/off cycle length (toggle mode only)
    :param seed: RNG seed for the draws
    """

    mode: Literal["prepost", "toggle"]
    turbine_subset: list[str]
    treatment_start_range: tuple[pd.Timestamp, pd.Timestamp]
    min_pre_months: int
    n_replicates: int
    campaign_months: list[int] | None = None
    campaign_weeks: list[int] | None = None
    toggle_period: pd.Timedelta | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate that exactly one campaign-length grid is set."""
        resolve_campaign_grid(campaign_months=self.campaign_months, campaign_weeks=self.campaign_weeks)

    @property
    def campaign_lengths(self) -> list[int]:
        """The campaign-length grid values, whichever unit they are in."""
        return resolve_campaign_grid(campaign_months=self.campaign_months, campaign_weeks=self.campaign_weeks)[0]

    @property
    def campaign_unit(self) -> CampaignUnit:
        """The campaign grid's unit: ``"months"`` or ``"weeks"``."""
        return resolve_campaign_grid(campaign_months=self.campaign_months, campaign_weeks=self.campaign_weeks)[1]

    @property
    def campaign_length_col(self) -> str:
        """The result column campaign lengths are reported under (``campaign_months``/``_weeks``)."""
        return f"campaign_{self.campaign_unit}"

    @property
    def max_activity_months(self) -> int:
        """The longest campaign the study scores, in months. Raises for a weeks study.

        Raises rather than converting so a weeks study reaching a months-only consumer fails loudly
        instead of silently reporting a week count as a month count.
        """
        if self.campaign_unit != "months":
            msg = (
                "max_activity_months is only defined for a months grid, but this study has "
                f"campaign_unit={self.campaign_unit!r}; read campaign_lengths instead"
            )
            raise ValueError(msg)
        return max(self.campaign_lengths)


@dataclass
class Replicate:
    """One generated dataset paired with the ingredients that produced it."""

    dataset: SyntheticDataset
    test_wtg: str
    treatment_start: pd.Timestamp
    upgrade_timing: pd.Timestamp | ToggleSchedule
    replicate_id: int = field(default=0)

    @property
    def synthetic_df(self) -> pd.DataFrame:
        """The method-facing synthetic SCADA (all subset turbines)."""
        return self.dataset.synthetic_df

    def true_uplift(self, **kwargs: object) -> UpliftResult:
        """Ground-truth uplift for this replicate's test turbine (delegates to the dataset)."""
        kwargs.setdefault("test_wtg", self.test_wtg)
        return self.dataset.true_uplift(**kwargs)  # type: ignore[arg-type]


def iter_replicates(
    base_scada: pd.DataFrame,
    *,
    profile: list,
    study: StudyConfig,
    columns: ColumnSchema = HOT_COLUMNS,
) -> Iterator[Replicate]:
    """Yield ``study.n_replicates`` replicates of ``profile`` one at a time.

    The streaming counterpart of :func:`build_replicates`, yielding the same replicates in the same
    order. A replicate carries a ``synthetic_df`` *and* the ``original_df`` its ground truth needs
    (~0.5 GB for a multi-year, few-turbine dataset), so a large ensemble must iterate here and let
    each be freed rather than materialising them all.

    :param columns: the source-native column schema ``base_scada`` is keyed by
    """
    subset = base_scada[base_scada[columns.turbine].isin(study.turbine_subset)]
    candidates = _candidate_starts(subset.index, study.treatment_start_range)

    rng = np.random.default_rng(study.seed)
    turbines = np.asarray(study.turbine_subset)
    for replicate_id in range(study.n_replicates):
        test_wtg = str(rng.choice(turbines))
        treatment_start = candidates[int(rng.integers(len(candidates)))]
        upgrade_timing = _upgrade_timing(study, treatment_start)
        dataset = generate_dataset(
            scada_df=subset,
            test_wtgs=[test_wtg],
            upgrades=profile,
            mode=study.mode,
            upgrade_timing=upgrade_timing,
            columns=columns,
            seed=study.seed,
        )
        yield Replicate(
            dataset=dataset,
            test_wtg=test_wtg,
            treatment_start=treatment_start,
            upgrade_timing=upgrade_timing,
            replicate_id=replicate_id,
        )


def build_replicates(
    base_scada: pd.DataFrame,
    *,
    profile: list,
    study: StudyConfig,
    columns: ColumnSchema = HOT_COLUMNS,
) -> list[Replicate]:
    """Draw ``study.n_replicates`` replicates of ``profile`` from ``base_scada``.

    Subsets the data to ``turbine_subset``, then draws ``(test turbine, treatment_start)`` pairs
    deterministically from ``seed`` and injects the profile via the synthetic generator.

    Materialises every replicate at once; see :func:`iter_replicates` to stream them instead when
    the ensemble is large enough for that to matter.

    :param columns: the source-native column schema ``base_scada`` is keyed by
    """
    return list(iter_replicates(base_scada, profile=profile, study=study, columns=columns))


def _candidate_starts(
    index: pd.DatetimeIndex, treatment_start_range: tuple[pd.Timestamp, pd.Timestamp]
) -> npt.NDArray[np.datetime64]:
    """Return unique on-grid timestamps within the draw range (so draws land on real records)."""
    lo, hi = treatment_start_range
    unique = index.unique()
    candidates = unique[(unique >= lo) & (unique <= hi)]
    if len(candidates) == 0:
        msg = f"no records in treatment_start_range {treatment_start_range}"
        raise ValueError(msg)
    return candidates.sort_values().to_numpy()


def _upgrade_timing(study: StudyConfig, treatment_start: pd.Timestamp) -> pd.Timestamp | ToggleSchedule:
    if study.mode == "toggle":
        if study.toggle_period is None:
            msg = "toggle_period is required for mode='toggle'"
            raise ValueError(msg)
        return ToggleSchedule(period=study.toggle_period, start=treatment_start)
    return treatment_start
