"""What is asked of the facts, and what a method answers.

An :class:`AnalysisSpec` names an upgrade, the turbines to measure it on and the records to use.
A method returns one :class:`AnalysisOutput` per analysis. :func:`combine` gives the energy-weighted
uplift of several outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from wind_up.analysis_warnings import AnalysisWarning

if TYPE_CHECKING:
    from collections.abc import Sequence

AEP = "aep"
CAMPAIGN = "campaign"
REPORTS = (AEP, CAMPAIGN)

# The step of the method that combines measurements, for the warnings register.
COMBINE_STEP = 16

PER_TURBINE_COLUMNS = (
    "turbine",
    "aep_uplift",
    "aep_sigma",
    "campaign_uplift",
    "campaign_sigma",
    "actual_energy",
    "hours_on",
    "hours_off",
)


@dataclass(frozen=True)
class AnalysisSpec:
    """One question asked of the facts: the uplift of an upgrade on a group of turbines.

    :param name: the analysis's name, for output folders and the warnings register
    :param upgrade: the name of the upgrade measured
    :param subjects: the turbines measured, together; every turbine that received the upgrade when None
    :param references: the candidate references; every turbine unchanged over the period when None
    :param period: ``(start, end)``, either side None for open. Left open, a method uses every record
        it can. Given, no record outside ``[start, end)`` is compared; earlier records stay available
        as history.
    :param include_labels: only a subject's records holding all of these labels are compared
    :param exclude_labels: no subject's record holding any of these labels is compared
    :param report: the uplifts wanted, from :data:`REPORTS`
    """

    name: str
    upgrade: str
    subjects: Sequence[str] | None = None
    references: Sequence[str] | None = None
    period: tuple[pd.Timestamp | None, pd.Timestamp | None] = (None, None)
    include_labels: Sequence[str] = ()
    exclude_labels: Sequence[str] = ()
    report: Sequence[str] = REPORTS

    def __post_init__(self) -> None:
        """Freeze the collections and reject an analysis that contradicts itself."""
        for name in ("subjects", "references"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else tuple(str(t) for t in value))
        for name in ("include_labels", "exclude_labels"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "report", (self.report,) if isinstance(self.report, str) else tuple(self.report))
        if not self.name.strip() or not self.upgrade.strip():
            msg = f"an analysis needs a name and an upgrade, got name {self.name!r} and upgrade {self.upgrade!r}"
            raise ValueError(msg)
        if self.subjects is not None and (not self.subjects or len(set(self.subjects)) != len(self.subjects)):
            msg = f"analysis {self.name!r} needs at least one subject, each named once; got {list(self.subjects)}"
            raise ValueError(msg)
        if not self.report or set(self.report) - set(REPORTS) or len(set(self.report)) != len(self.report):
            msg = f"analysis {self.name!r} reports {list(self.report)}; report is one or both of {list(REPORTS)}"
            raise ValueError(msg)
        start, end = self.period
        if start is not None and end is not None and end <= start:
            msg = f"analysis {self.name!r} has a period that ends at or before its start: {self.period}"
            raise ValueError(msg)
        both = sorted(set(self.include_labels) & set(self.exclude_labels))
        if both:
            msg = f"analysis {self.name!r} both includes and excludes the labels {both}"
            raise ValueError(msg)


@dataclass
class AnalysisOutput:
    """A method's answer to one analysis.

    A figure is NaN when it was not asked for or the method cannot give it; ``warnings`` says which.

    :param analysis: the analysis answered
    :param aep_uplift: the subjects' uplift in annual energy, together, as a fraction
    :param aep_sigma: its 1-sigma
    :param campaign_uplift: the subjects' uplift over the records compared, as a fraction
    :param campaign_sigma: its 1-sigma
    :param per_turbine: one row per subject with :data:`PER_TURBINE_COLUMNS`; a subject that could
        not be measured is a row of NaN
    :param per_reference: one row per subject and reference, from a method that works pairwise
    :param by_condition: per-condition estimates: ``condition``, ``condition_bin``, ``p50_uplift``
        and optionally ``sigma_uplift``
    :param labeled_rows: every subject's records with the method's selection: ``used``, ``segment``
        and a bin column per condition, beside the records' own columns
    :param reference_uplifts: what the analysis says each candidate reference did: ``turbine``,
        ``uplift``, ``actual_energy``, ``n_records``, ``screened``
    :param screen_passes: one row per pass of a method's reference screen
    :param selection_accounting: records kept after each of a method's selection stages
    :param uncertainty_diagnostics: how the uncertainty was reached, keyed by ``condition`` and
        ``condition_bin``
    :param warnings: the analysis warnings the estimate raised
    """

    analysis: AnalysisSpec
    aep_uplift: float = float("nan")
    aep_sigma: float = float("nan")
    campaign_uplift: float = float("nan")
    campaign_sigma: float = float("nan")
    per_turbine: pd.DataFrame | None = None
    per_reference: pd.DataFrame | None = None
    by_condition: pd.DataFrame | None = None
    labeled_rows: pd.DataFrame | None = None
    reference_uplifts: pd.DataFrame | None = None
    screen_passes: pd.DataFrame | None = None
    selection_accounting: pd.DataFrame | None = None
    uncertainty_diagnostics: pd.DataFrame | None = None
    warnings: tuple[AnalysisWarning, ...] = ()

    @property
    def actual_energy(self) -> float:
        """Return the energy the subjects produced with the upgrade on; NaN without a per-turbine table."""
        if self.per_turbine is None or "actual_energy" not in self.per_turbine.columns:
            return float("nan")
        return float(self.per_turbine["actual_energy"].sum(min_count=1))


@dataclass(frozen=True)
class CombinedUplift:
    """The energy-weighted uplift of several outputs.

    Each sigma is given twice: with the outputs' errors independent, and with them fully correlated.

    :param aep_uplift: the combined AEP uplift; NaN unless every output has one
    :param aep_sigma_independent: its 1-sigma with independent errors
    :param aep_sigma_correlated: its 1-sigma with fully correlated errors
    :param campaign_uplift: the combined campaign uplift; NaN unless every output has one
    :param campaign_sigma_independent: its 1-sigma with independent errors
    :param campaign_sigma_correlated: its 1-sigma with fully correlated errors
    :param outputs: one row per output: ``analysis``, ``actual_energy`` and its four figures
    :param warnings: the outputs' own warnings, then those of the combination
    """

    aep_uplift: float
    aep_sigma_independent: float
    aep_sigma_correlated: float
    campaign_uplift: float
    campaign_sigma_independent: float
    campaign_sigma_correlated: float
    outputs: pd.DataFrame
    warnings: tuple[AnalysisWarning, ...]


def combine(outputs: Sequence[AnalysisOutput]) -> CombinedUplift:
    """Return the energy-weighted uplift of ``outputs`` on each basis they all report.

    Each output's energy without the upgrade is its actual energy over one plus its uplift; the
    combined uplift is the total actual energy over the total without the upgrade, less one. An
    output needs a per-turbine table with ``actual_energy``.
    """
    if not outputs:
        msg = "combine needs at least one output"
        raise ValueError(msg)
    table = pd.DataFrame(
        {
            "analysis": [o.analysis.name for o in outputs],
            "actual_energy": [o.actual_energy for o in outputs],
            "aep_uplift": [o.aep_uplift for o in outputs],
            "aep_sigma": [o.aep_sigma for o in outputs],
            "campaign_uplift": [o.campaign_uplift for o in outputs],
            "campaign_sigma": [o.campaign_sigma for o in outputs],
        }
    )
    energy = table["actual_energy"].to_numpy(dtype=float)
    warnings = [w for o in outputs for w in o.warnings]
    figures: dict[str, float] = {}
    for basis in REPORTS:
        uplift, independent, correlated = _weighted(
            energy=energy,
            uplift=table[f"{basis}_uplift"].to_numpy(dtype=float),
            sigma=table[f"{basis}_sigma"].to_numpy(dtype=float),
        )
        figures |= {
            f"{basis}_uplift": uplift,
            f"{basis}_sigma_independent": independent,
            f"{basis}_sigma_correlated": correlated,
        }
        if not math.isfinite(uplift):
            message = f"no combined {basis} uplift: not every output has one with a positive energy behind it"
            warnings.append(AnalysisWarning(severity="medium", step=COMBINE_STEP, turbine="all", message=message))
    message = (
        "the combined sigma is a range: the outputs' errors are taken as independent at one end and as fully "
        "correlated at the other"
    )
    warnings.append(AnalysisWarning(severity="low", step=COMBINE_STEP, turbine="all", message=message))
    return CombinedUplift(outputs=table, warnings=tuple(warnings), **figures)


def _weighted(*, energy: np.ndarray, uplift: np.ndarray, sigma: np.ndarray) -> tuple[float, float, float]:
    """Return the energy-weighted uplift and its sigma with independent and with fully correlated errors."""
    nan = float("nan")
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(uplift) & np.isfinite(energy) & (energy > 0) & (uplift > -1.0)
    if not usable.all():
        return nan, nan, nan
    without = energy / (1.0 + uplift)
    combined = float(energy.sum() / without.sum() - 1.0)
    if not np.isfinite(sigma).all():
        return combined, nan, nan
    weighted = sigma * without / without.sum()
    return combined, float(np.sqrt(np.square(weighted).sum())), float(weighted.sum())
