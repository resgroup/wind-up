"""One subject's measurement inside an analysis: the records it reads and what it may know about them.

A method answers an analysis by measuring each subject in turn. :func:`subject_records` gathers
what one subject's measurement sees from the facts; :class:`SubjectMeasurement` is what it returns,
and :func:`group_output` combines a group's measurements into the analysis's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from wind_up.analysis import AEP, CAMPAIGN, PER_TURBINE_COLUMNS, AnalysisOutput
from wind_up.analysis_warnings import AnalysisWarning
from wind_up.farm import TurbineUplift, farm_uplift
from wind_up.layout import LATITUDE_COL, LONGITUDE_COL, NAME_COL
from wind_up.method import MethodConfigurationError
from wind_up.operating_state import STATE_COL, STATE_COLUMNS, VALID_NORTHING_COL, VALID_UPLIFT_COL
from wind_up.record_selection import ChangedInterval
from wind_up.toggle import ToggleSchedule, is_toggle

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from wind_up.analysis import AnalysisSpec
    from wind_up.facts import WindFarmFacts
    from wind_up.record_selection import ChangedTimes

# The method step a measurement and its combination belong to, for the warnings register.
CANDIDATES_STEP = 8
MEASURE_STEP = 10
COMBINE_STEP = 16

# How the warning of a subject that could not be measured begins.
NOT_MEASURED = "not measured: "

# The column naming the subject in a stacked frame whose ``turbine`` column names a reference.
SUBJECT_COL = "subject"

BOOLEAN_STATE_COLUMNS = (VALID_NORTHING_COL, VALID_UPLIFT_COL)

# The segments of a measurement's labeled rows.
BASELINE = "baseline"
UPGRADED = "upgraded"


@dataclass(frozen=True)
class SubjectRecords:
    """The records one subject's measurement reads, and what it may know about them.

    :param records: long-format records of every turbine over the measurement's span, keyed by the
        facts' column names
    :param subject: the turbine measured
    :param timing: the subject's changeover timestamp, or its toggle schedule or frame (boolean
        ``toggle_on``/``toggle_off`` on a ``DatetimeIndex``). Read :attr:`mode` rather than switching on the type.
    :param turbine_col: the turbine-identifier column of ``records``
    :param candidate_references: the turbines the measurement may use as references
    :param wake_contributors: every other turbine in the records, read for its wake alone
    :param valid_for_uplift: timestamps x turbines: may each turbine's record at each timestamp
        contribute to the uplift. Its index covers the records' timestamps; narrow it with :meth:`valid_over`.
    :param coords: turbine name to ``(latitude, longitude)``
    :param reserve_references: wake contributors that may replace a candidate reference the method rules out,
        first choice first
    :param changed_turbines: each changed turbine and the times it counts as changed, the subject among them
    :param works: the subject's installation window ``[start, end)``; no measurement uses a record inside it
    """

    records: pd.DataFrame
    subject: str
    timing: pd.Timestamp | ToggleSchedule | pd.DataFrame
    turbine_col: str
    candidate_references: list[str]
    wake_contributors: list[str]
    valid_for_uplift: pd.DataFrame
    coords: dict[str, tuple[float, float]] | None = None
    reserve_references: list[str] = field(default_factory=list)
    changed_turbines: ChangedTimes = field(default_factory=dict)
    works: tuple[pd.Timestamp, pd.Timestamp] | None = None

    def __post_init__(self) -> None:
        """Raise when a turbine holds two roles, or a reserve is not a wake contributor."""
        clash = sorted({self.subject, *self.candidate_references} & set(self.wake_contributors))
        if clash:
            msg = (
                f"wake contributors {clash} are also the subject {self.subject!r} or a candidate reference "
                f"{sorted(self.candidate_references)}; a turbine holds one role"
            )
            raise ValueError(msg)
        stray = sorted(set(self.reserve_references) - set(self.wake_contributors))
        if stray:
            msg = f"reserve references {stray} are not wake contributors; a reserve waits among them"
            raise ValueError(msg)

    @property
    def mode(self) -> Literal["prepost", "toggle"]:
        """``"toggle"`` for a toggle test, ``"prepost"`` for a single changeover."""
        return "toggle" if is_toggle(self.timing) else "prepost"

    def valid_over(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return :attr:`valid_for_uplift` narrowed to ``index``. Raises on a timestamp not covered."""
        missing = pd.DatetimeIndex(index).difference(self.valid_for_uplift.index)
        if len(missing) > 0:
            msg = f"valid_for_uplift does not cover {len(missing)} of the requested timestamps, first {missing[0]}"
            raise ValueError(msg)
        return self.valid_for_uplift.loc[index]

    def references_among(self, columns: Iterable[str]) -> list[str]:
        """Return the entries of ``columns`` that are candidate references, in ``columns``' order."""
        offered = set(self.candidate_references)
        return [str(c) for c in columns if str(c) in offered]

    def mask_invalid(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a timestamps x turbines ``frame`` with cells not valid for uplift set to NaN.

        Columns of turbines without validity are left untouched.
        """
        valid = self.valid_over(pd.DatetimeIndex(frame.index))
        shared = [c for c in frame.columns if c in valid.columns]
        if not shared:
            return frame
        return frame.mask(~valid[shared].reindex(columns=frame.columns, fill_value=True))

    def select(self, records: pd.DataFrame | None = None, *, also: Iterable[str] = ()) -> pd.DataFrame:
        """Return the rows the measurement may use: its turbines' records that are valid for uplift.

        Keeps the subject, its candidate references and its wake contributors.

        :param records: long-format records to select from; :attr:`records` when ``None``
        :param also: turbines to keep besides those
        """
        frame = self.records if records is None else records
        valid = self.valid_over(pd.DatetimeIndex(frame.index.unique()))
        turbines = frame[self.turbine_col].to_numpy()
        keep = np.zeros(len(frame), dtype=bool)
        for wtg in dict.fromkeys([self.subject, *self.candidate_references, *self.wake_contributors, *also]):
            is_turbine = turbines == wtg
            if not is_turbine.any():
                continue
            if wtg not in valid.columns:
                msg = f"{wtg!r} has no validity here (covers {list(valid.columns)}), so its records cannot be used"
                raise ValueError(msg)
            rows = pd.DatetimeIndex(frame.index[is_turbine])
            keep[is_turbine] = valid[wtg].reindex(rows).to_numpy()
        return frame[keep]

    @classmethod
    def from_frame(
        cls,
        records: pd.DataFrame,
        *,
        subject: str,
        timing: pd.Timestamp | ToggleSchedule | pd.DataFrame,
        turbine_col: str,
    ) -> SubjectRecords:
        """Return the records of ``subject`` with every other turbine a reference and every record valid."""
        references = sorted({str(t) for t in records[turbine_col].unique()} - {subject})
        index = pd.DatetimeIndex(records.index.unique()).sort_values()
        return cls(
            records=records,
            subject=subject,
            timing=timing,
            turbine_col=turbine_col,
            candidate_references=references,
            wake_contributors=[],
            valid_for_uplift=pd.DataFrame(data=True, index=index, columns=sorted({subject, *references})),
            changed_turbines={subject: (ChangedInterval(None, None),)},
        )


@dataclass
class SubjectMeasurement:
    """One subject's measured uplift over the records it compared.

    :param p50_overall: the uplift over the compared records, a fraction
    :param p50_by_condition: per condition bin (``condition``, ``condition_bin``, ``p50_uplift`` and
        optionally ``sigma_uplift``); ``None`` when only the overall figure is measured
    :param sigma_overall: the 1-sigma of ``p50_overall``
    :param uncertainty_diagnostics: what the uncertainty was reached from, keyed by ``condition`` and
        ``condition_bin``; ``("overall", "overall")`` is the headline row
    :param labeled_rows: the subject's records with ``used``, ``segment`` (``baseline``, ``upgraded`` or
        ``excluded``) and a ``<condition>_bin`` column per condition
    :param reference_uplifts: each candidate reference's own uplift (``turbine``, ``uplift``,
        ``actual_energy``, ``n_records``, ``screened``)
    :param screen_passes: each pass of the method's reference screen
    :param selection_accounting: records kept after each selection stage, per ``stage`` and ``segment``
    :param warnings: the analysis warnings the measurement raised
    """

    p50_overall: float
    p50_by_condition: pd.DataFrame | None = None
    sigma_overall: float | None = None
    uncertainty_diagnostics: pd.DataFrame | None = None
    labeled_rows: pd.DataFrame | None = None
    reference_uplifts: pd.DataFrame | None = None
    screen_passes: pd.DataFrame | None = None
    selection_accounting: pd.DataFrame | None = None
    warnings: tuple[AnalysisWarning, ...] = ()


def subject_records(
    facts: WindFarmFacts,
    analysis: AnalysisSpec,
    *,
    subject: str,
    span: tuple[pd.Timestamp, pd.Timestamp | None],
    timing: pd.Timestamp | ToggleSchedule | pd.DataFrame | None = None,
    valid_for_uplift: pd.DataFrame | None = None,
    n_references: int | None = None,
    record_labels: pd.DataFrame | None = None,
) -> SubjectRecords:
    """Return what measuring ``subject`` for ``analysis`` over ``span`` reads from ``facts``.

    The candidate references are the analysis's, in its order, less the subject and any turbine
    without records in the span; with ``n_references`` the first that many named are the
    candidates and the rest wait as reserves. Every other turbine with records is a wake
    contributor.

    :param span: ``[start, end)`` of the records read; an end of ``None`` reads to the last record
    :param timing: the subject's timing; by default its toggle schedule or frame, or the end of
        its changeover
    :param valid_for_uplift: timestamps x turbines validity covering the span; by default each turbine's
        records are valid outside its own installation windows
    :param n_references: how many of the analysis's references are candidates
    :param record_labels: the frame :meth:`~wind_up.facts.WindFarmFacts.record_labels` returned; each
        record then carries its operating state and that state's uses, replacing any columns of those names
    """
    start, end = span
    records = facts.records[np.asarray(facts.records.index >= start)]
    if end is not None:
        records = records[np.asarray(records.index < end)]
    turbine_col = facts.columns.turbine
    if record_labels is not None:
        records = with_states(records, record_labels=record_labels, turbine_col=turbine_col)
    index = pd.DatetimeIndex(records.index.unique()).sort_values()
    held = {str(t) for t in records[turbine_col].unique()}
    present = [t for t in facts.turbine_names if t in held]
    named = [r for r in facts.references_of(analysis) if r != subject]
    chosen = named if n_references is None else named[:n_references]
    references = [r for r in chosen if r in held]
    reserves = [r for r in named[len(chosen) :] if r in held]
    upgrade = facts.upgrade(analysis.upgrade)
    if timing is None:
        timing = upgrade.timing if upgrade.mode == "toggle" else upgrade.changeover(subject)[1]  # type: ignore[assignment]
    if valid_for_uplift is None:
        valid_for_uplift = pd.DataFrame(
            {t: ~_inside(index, _installation_windows(facts, t)) for t in sorted(present)}, index=index, dtype=bool
        )
    coords = {
        str(n): (float(lat), float(lon))
        for n, lat, lon in zip(
            facts.turbines[NAME_COL], facts.turbines[LATITUDE_COL], facts.turbines[LONGITUDE_COL], strict=True
        )
    }
    span_end = index.max() + facts.timebase if end is None and len(index) else end
    return SubjectRecords(
        records=records,
        subject=subject,
        timing=timing,  # type: ignore[arg-type]
        turbine_col=turbine_col,
        candidate_references=references,
        wake_contributors=sorted(set(present) - {subject} - set(references)),
        valid_for_uplift=valid_for_uplift,
        coords=coords,
        reserve_references=reserves,
        changed_turbines=facts.changed_turbines(analysis, subject=subject, span=(start, span_end)),
        works=None if upgrade.mode == "toggle" else upgrade.changeover(subject),
    )


def with_states(records: pd.DataFrame, *, record_labels: pd.DataFrame, turbine_col: str) -> pd.DataFrame:
    """Return long-format ``records`` with each record's operating state and its uses from ``record_labels``.

    Columns of those names already in ``records`` are replaced. Raises when a record has no label.
    """
    keys = pd.MultiIndex.from_arrays([pd.DatetimeIndex(records.index), records[turbine_col].astype(str)])
    labelled = record_labels.set_index(record_labels[turbine_col].astype(str), append=True)[list(STATE_COLUMNS)]
    found = labelled.reindex(keys)
    if found[STATE_COL].isna().any():
        msg = f"{int(found[STATE_COL].isna().sum())} records have no label; the labels must cover every record"
        raise ValueError(msg)
    out = records.copy()
    for column in STATE_COLUMNS:
        out[column] = found[column].to_numpy(dtype=bool if column in BOOLEAN_STATE_COLUMNS else object)
    return out


def _installation_windows(facts: WindFarmFacts, turbine: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return ``turbine``'s installation windows under every upgrade."""
    return [w for upgrade in facts.upgrades for w in upgrade.installation_windows.get(turbine, ())]


def _inside(index: pd.DatetimeIndex, windows: Iterable[tuple[pd.Timestamp, pd.Timestamp]]) -> np.ndarray:
    """Return a mask over ``index`` of the timestamps inside any of ``windows``, ends exclusive."""
    inside = np.zeros(len(index), dtype=bool)
    for start, end in windows:
        inside |= np.asarray((index >= start) & (index < end))
    return inside


def measured_row(subject: str, *, measurement: SubjectMeasurement, power_col: str, timebase: pd.Timedelta) -> dict:
    """Return a measured subject's ``per_turbine`` row, its energy and hours read from its labeled rows.

    The energy and hours are NaN when the labeled rows do not say which records were used on each side.
    """
    rows = measurement.labeled_rows
    row: dict[str, object] = {
        "turbine": subject,
        "aep_uplift": np.nan,
        "aep_sigma": np.nan,
        "campaign_uplift": float(measurement.p50_overall),
        "campaign_sigma": np.nan if measurement.sigma_overall is None else float(measurement.sigma_overall),
        "actual_energy": np.nan,
        "hours_on": np.nan,
        "hours_off": np.nan,
    }
    if rows is None or not {"used", "segment", power_col} <= set(rows.columns):
        return row
    used = rows["used"].fillna(value=False).astype(bool)
    hours = timebase / pd.Timedelta(hours=1)
    upgraded = used & (rows["segment"] == UPGRADED)
    row["actual_energy"] = float(rows.loc[upgraded, power_col].sum())
    row["hours_on"] = float(upgraded.sum() * hours)
    row["hours_off"] = float((used & (rows["segment"] == BASELINE)).sum() * hours)
    return row


def group_output(
    analysis: AnalysisSpec,
    *,
    facts: WindFarmFacts,
    measurements: Mapping[str, SubjectMeasurement | str],
    name: str,
    warnings: Sequence[AnalysisWarning] = (),
) -> AnalysisOutput:
    """Return the analysis's output from each subject's measurement, or why it could not be measured.

    One measured subject gives the analysis its uplift and sigma. Several give the uplift of their
    energy on the upgrade, leaving out any whose energy is unknown, and no sigma. A subject not
    measured is a row of NaN and a high warning beginning :data:`NOT_MEASURED`. No AEP uplift is
    given; asking for one is a medium warning.
    """
    found: list[AnalysisWarning] = list(warnings)
    rows: list[dict[str, object]] = []
    weights: list[TurbineUplift] = []
    measured: dict[str, SubjectMeasurement] = {}
    for subject, measurement in measurements.items():
        if isinstance(measurement, str):
            message = f"{NOT_MEASURED}{measurement}"
            found.append(AnalysisWarning(severity="high", step=MEASURE_STEP, turbine=subject, message=message))
            rows.append({"turbine": subject, **{c: np.nan for c in PER_TURBINE_COLUMNS if c != "turbine"}})
            continue
        measured[subject] = measurement
        row = measured_row(
            subject, measurement=measurement, power_col=facts.columns.active_power, timebase=facts.timebase
        )
        rows.append(row)
        energy, hours_on = float(row["actual_energy"]), float(row["hours_on"])  # type: ignore[arg-type]
        if np.isfinite(energy) and np.isfinite(hours_on):
            weights.append(
                TurbineUplift(
                    turbine=subject,
                    uplift=float(row["campaign_uplift"]),  # type: ignore[arg-type]
                    actual_energy=energy,
                    n_records=round(hours_on * pd.Timedelta(hours=1) / facts.timebase),
                    rated_power_kw=facts.rated_power_kw(subject),
                )
            )
    per_turbine = pd.DataFrame(rows, columns=list(PER_TURBINE_COLUMNS))
    uplift, sigma = np.nan, np.nan
    if len(measured) == 1:
        (only,) = measured.values()
        uplift = float(only.p50_overall)
        sigma = np.nan if only.sigma_overall is None else float(only.sigma_overall)
    elif len(measured) > 1:
        uplift = farm_uplift(weights).uplift if weights else np.nan
        message = f"{name} gives no sigma for a group of subjects; each subject's sigma is in per_turbine"
        found.append(AnalysisWarning(severity="low", step=COMBINE_STEP, turbine="all", message=message))
    else:
        message = f"{name} measured none of the subjects {list(per_turbine['turbine'])}"
        found.append(AnalysisWarning(severity="high", step=MEASURE_STEP, turbine="all", message=message))
    if AEP in analysis.report:
        message = f"{name} gives no AEP uplift: it has no long-term step"
        found.append(AnalysisWarning(severity="medium", step=COMBINE_STEP, turbine="all", message=message))
    if CAMPAIGN not in analysis.report:
        uplift, sigma = np.nan, np.nan
    return AnalysisOutput(
        analysis=analysis,
        campaign_uplift=uplift,
        campaign_sigma=sigma,
        per_turbine=per_turbine,
        by_condition=_stacked({k: m.p50_by_condition for k, m in measured.items()}),
        labeled_rows=_stacked({k: m.labeled_rows for k, m in measured.items()}),
        reference_uplifts=_stacked({k: m.reference_uplifts for k, m in measured.items()}, key=SUBJECT_COL),
        screen_passes=_stacked({k: m.screen_passes for k, m in measured.items()}, key=SUBJECT_COL),
        selection_accounting=_stacked({k: m.selection_accounting for k, m in measured.items()}),
        uncertainty_diagnostics=_stacked({k: m.uncertainty_diagnostics for k, m in measured.items()}),
        warnings=(*found, *(w for m in measured.values() for w in m.warnings)),
    )


def _stacked(frames: Mapping[str, pd.DataFrame | None], *, key: str = "turbine") -> pd.DataFrame | None:
    """Return the subjects' frames one under another with a leading ``key`` column naming the subject.

    None when there are none.
    """
    present = {k: v for k, v in frames.items() if v is not None}
    if not present:
        return None
    stacked = pd.concat([frame.assign(**{key: subject}) for subject, frame in present.items()])
    return stacked[[key, *[c for c in stacked.columns if c != key]]]


def unused_measurement_warnings(
    facts: WindFarmFacts, analysis: AnalysisSpec, *, name: str
) -> tuple[AnalysisWarning, ...]:
    """Return a medium warning when ``analysis`` names non-turbine references, which ``name`` cannot use."""
    unused = facts.measurement_references(analysis)
    if not unused:
        return ()
    message = f"{name} compares turbines' power, so it leaves out the non-turbine reference(s) {unused}"
    return (AnalysisWarning(severity="medium", step=CANDIDATES_STEP, turbine="all", message=message),)


def answer_analyses(
    facts: WindFarmFacts,
    analyses: Sequence[AnalysisSpec],
    *,
    name: str,
    measure: Callable[[SubjectRecords], SubjectMeasurement],
    n_references: int | None = None,
) -> list[AnalysisOutput]:
    """Answer each analysis by measuring each subject in turn over the analysis's period, in order.

    Each record a subject's measurement reads carries its operating state and that state's uses
    from the facts' record labels, computed once per call.

    A ``ValueError`` from ``measure`` leaves that subject unmeasured, with the error as the reason;
    a :class:`~wind_up.method.MethodConfigurationError` fails the call. Raises when an analysis
    names labels, which these methods do not read.

    :param name: the method's name, for the warnings
    :param measure: measures one subject from its records
    :param n_references: how many of each analysis's references are candidates; the rest are reserves
    """
    for analysis in analyses:
        facts.check(analysis)
        if analysis.include_labels or analysis.exclude_labels:
            msg = f"{name} does not read labels; analysis {analysis.name!r} names some"
            raise MethodConfigurationError(msg)
    labels = facts.record_labels()
    outputs = []
    for analysis in analyses:
        start, end = analysis.period
        span = (pd.DatetimeIndex(facts.records.index).min() if start is None else start, end)
        measured: dict[str, SubjectMeasurement | str] = {}
        for subject in facts.subjects_of(analysis):
            records = subject_records(
                facts, analysis, subject=subject, span=span, n_references=n_references, record_labels=labels
            )
            try:
                measured[subject] = measure(records)
            except MethodConfigurationError:
                raise
            except ValueError as error:
                measured[subject] = str(error)
        warnings = (*facts.warnings(), *unused_measurement_warnings(facts, analysis, name=name))
        outputs.append(group_output(analysis, facts=facts, measurements=measured, name=name, warnings=warnings))
    return outputs


def measurement_of(output: AnalysisOutput, subject: str) -> SubjectMeasurement | str:
    """Return ``subject``'s own measurement from an analysis's output, or why it was not measured.

    The frames are the subject's rows without the column naming it; the warnings are the analysis's.
    """
    rows = output.per_turbine
    if rows is None or subject not in set(rows["turbine"]):
        msg = f"{subject!r} is not a subject of analysis {output.analysis.name!r}"
        raise KeyError(msg)
    reasons = [
        w.message.removeprefix(NOT_MEASURED)
        for w in output.warnings
        if w.turbine == subject and w.step == MEASURE_STEP and w.message.startswith(NOT_MEASURED)
    ]
    if reasons:
        return reasons[0]
    row = rows[rows["turbine"] == subject].iloc[0]

    def mine(frame: pd.DataFrame | None, key: str = "turbine") -> pd.DataFrame | None:
        if frame is None:
            return None
        return frame[frame[key] == subject].drop(columns=[key])

    sigma = float(row["campaign_sigma"])
    return SubjectMeasurement(
        p50_overall=float(row["campaign_uplift"]),
        p50_by_condition=mine(output.by_condition),
        sigma_overall=sigma if np.isfinite(sigma) else None,
        uncertainty_diagnostics=mine(output.uncertainty_diagnostics),
        labeled_rows=mine(output.labeled_rows),
        reference_uplifts=mine(output.reference_uplifts, SUBJECT_COL),
        screen_passes=mine(output.screen_passes, SUBJECT_COL),
        selection_accounting=mine(output.selection_accounting),
        warnings=output.warnings,
    )
