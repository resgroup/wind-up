"""v0's binned power curve method behind the v1 interface.

``V0BinnedMethod`` answers each analysis with one v0 run (``run_wind_up_analysis``) over the
analysis's subjects against its references, then combines each subject's test-reference results
with v0's ``combine_results``. The run is configured from three places:

* the facts: the turbines and their positions, the non-turbine measurements, the reanalysis
  nodes, the north offsets, the exclusions and installation windows, the upgrade's timing and
  settling time, and the timebase;
* the analysis: its subjects and references, its period and which uplift it reports (an AEP
  uplift turns on v0's long-term distribution, a campaign uplift turns it off; both is two runs);
* the method's own ``options``, any field of v0's ``WindUpConfig`` the facts and the analysis do
  not set, and ``turbine_type``, the fields of v0's ``TurbineType`` beyond rotor diameter and
  rated power.

v0 keeps its own filters and waking rule and reads the records as measured. A record the facts
withhold from the uplift by a label or an operating state override reaches v0 flagged in its
``exclude_row`` column, which v0 treats as it treats an excluded period. A subject's record an
analysis's labels leave out reaches v0 as a comparison exclusion: v0 preprocesses it and counts it
in the long-term distribution, and leaves it out of the subject's test-reference comparisons.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from wind_up.analysis import (
    AEP,
    CAMPAIGN,
    NET,
    PER_TURBINE_COLUMNS,
    AnalysisOutput,
    combine_subjects,
    subjects_only,
)
from wind_up.analysis_warnings import AnalysisWarning
from wind_up.labels import NEITHER, OFF, ON, UPLIFT
from wind_up.layout import LATITUDE_COL, LONGITUDE_COL, NAME_COL, ROTOR_DIAMETER_COL
from wind_up.method import MethodConfigurationError
from wind_up.subject import NOT_MEASURED, PLACEBO_NOT_MEASURED
from wind_up.toggle import ToggleSchedule, build_toggle_df
from wind_up_v0.combine_results import _calculate_total_uplift_of_turbine_group, combine_results
from wind_up_v0.constants import TIMESTAMP_COL, DataColumns
from wind_up_v0.interface import AssessmentInputs
from wind_up_v0.main_analysis import run_wind_up_analysis
from wind_up_v0.models import Asset, MastOrLidar, PlotConfig, PrePost, Turbine, TurbineType, WindUpConfig
from wind_up_v0.reanalysis_data import MastOrLiDARDataset, ReanalysisDataset

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from wind_up.analysis import AnalysisSpec
    from wind_up.facts import Upgrade, WindFarmFacts
    from wind_up.schema import ColumnSchema

# v0's configuration fields the facts and the analysis set.
FACT_FIELDS = frozenset(
    {
        "assessment_name",
        "timebase_s",
        "out_dir",
        "test_wtgs",
        "ref_wtgs",
        "non_wtg_ref_names",
        "upgrade_first_dt_utc_start",
        "analysis_last_dt_utc_start",
        "asset",
        "exclusion_periods_utc",
        "northing_corrections_utc",
        "use_lt_distribution",
    }
)
# v0's toggle fields an option may set; the rest come from the facts.
TOGGLE_OPTIONS = frozenset(
    {"name", "detrend_data_selection", "pairing_filter_method", "pairing_filter_timedelta_seconds"}
)
# v0's turbine type fields the turbine table sets.
TURBINE_TYPE_FACTS = frozenset({"rotor_diameter_m", "rated_power_kw"})

# The column v0 reads to drop a record as it drops an excluded period.
EXCLUDE_ROW_COL = "exclude_row"

# A measurement's column roles v0 reads.
WIND_SPEED_ROLE = "wind_speed"
WIND_DIRECTION_ROLE = "wind_direction"

# v0's column for each role of the facts' column schema.
V0_COLUMN_OF_ROLE = {
    "active_power": DataColumns.active_power_mean,
    "reactive_power": DataColumns.reactive_power_mean,
    "wind_speed": DataColumns.wind_speed_mean,
    "wind_speed_sd": DataColumns.wind_speed_sd,
    "nacelle_position": DataColumns.yaw_angle_mean,
    "nacelle_position_min": DataColumns.yaw_angle_min,
    "nacelle_position_max": DataColumns.yaw_angle_max,
    "pitch": DataColumns.pitch_angle_mean,
    "gen_rpm": DataColumns.gen_rpm_mean,
    "ambient_temp": DataColumns.ambient_temp,
}

# The step of the method each warning belongs to, for the warnings register.
MEASURE_STEP = 10
COMBINE_STEP = 16

LT_WEEK_BEFORE_TOGGLE = pd.Timedelta(days=7)

# v0's fields that have no default and are not facts.
REQUIRED_OPTIONS = ("years_for_lt_distribution", "years_for_detrend", "ws_bin_width")


def v0_records(records: pd.DataFrame, *, columns: ColumnSchema, timebase: pd.Timedelta) -> pd.DataFrame:
    """Return ``records`` with v0's column names, the turbine column ``TurbineName`` and a ``ShutdownDuration``.

    Each role's column takes v0's name unless a column of that name is already there. Without a
    ``ShutdownDuration`` column it is the period less the availability counter, a missing counter
    reading as fully available; a ``ShutdownDuration`` column already there is kept as given.
    Raises without either.
    """
    renames = {columns.turbine: DataColumns.turbine_name}
    for role, v0_name in V0_COLUMN_OF_ROLE.items():
        column = getattr(columns, role)
        if column is not None and column != v0_name and v0_name not in records.columns:
            renames[column] = v0_name
    out = records.rename(columns=renames)
    if DataColumns.shutdown_duration not in out.columns:
        if columns.availability is None:
            msg = "v0 reads a ShutdownDuration column; the records have none and no availability role to derive it"
            raise MethodConfigurationError(msg)
        period = timebase.total_seconds()
        out[DataColumns.shutdown_duration] = period - out[columns.availability].fillna(period)
    return out.rename_axis(index=TIMESTAMP_COL)


@dataclass
class V0BinnedMethod:
    """v0's binned power curve method, one v0 run per analysis.

    Requires the turbine table's ``rotor_diameter_m``, the records' turbine and power roles and,
    when the records carry no ``ShutdownDuration`` column, the availability role, from which
    :func:`v0_records` derives it; give the records a ``ShutdownDuration`` column to keep a downtime
    definition of your own. Reads every
    non-turbine measurement the analysis names as a reference through its ``wind_speed`` and
    ``wind_direction`` column roles, and every reanalysis node by its name. Gives the uplift the
    analysis reports with v0's sigma. A group of subjects gets v0's total uplift of the group, or, when
    the analysis asks for the ``net`` combination, the net uplift of :func:`wind_up.combine_subjects`
    and no sigma; that function gives the net uplift's sigma as a range.

    :param name: method name shown in the outputs
    :param options: fields of v0's ``WindUpConfig`` the facts and the analysis do not set, by name;
        ``toggle`` holds a mapping of v0's toggle fields among ``name``, ``detrend_data_selection``,
        ``pairing_filter_method`` and ``pairing_filter_timedelta_seconds``. Dates v0 works out from
        its ``years_*`` fields when not given, as its YAML loader does.
    :param turbine_type: fields of v0's ``TurbineType`` beyond rotor diameter and rated power, given to every turbine
    :param site_name: the wind farm's name in v0's outputs
    :param auto_choose_refs: let ``combine_results`` keep only the references that agree best
    :param to_v0_records: converts the facts' records to v0's columns; :func:`v0_records` when ``None``
    :param out_dir: where each run's folder is written, named for the analysis; a temp dir when ``None``
    :param save_plots: also write v0's plots
    :param per_turbine_plots: with ``save_plots``, also write v0's preprocessing plots of each turbine
        beside those of each turbine type
    :param cache_dir: v0's cache of prepared inputs; none when ``None``
    :param turbine_wind_measurement: a declared mast or LiDAR to supply missing turbine wind speed
        and nacelle direction columns, matched by timestamp
    :param default_pitch_deg: blade pitch to use if the turbine records have no pitch column
    :param default_gen_rpm: generator speed to use if the turbine records have no generator speed column
    :param analysis_label_stage: where v0 leaves out a subject's records the analysis's labels leave out:
        ``comparison`` only from its test-reference comparisons, ``preprocessing`` from everything, as
        an excluded period
    """

    name: str = "v0_binned"
    options: Mapping[str, Any] = field(default_factory=dict)
    turbine_type: Mapping[str, Any] = field(default_factory=dict)
    site_name: str = "wind farm"
    auto_choose_refs: bool = True
    to_v0_records: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    out_dir: Path | None = None
    save_plots: bool = False
    per_turbine_plots: bool = True
    cache_dir: Path | None = None
    turbine_wind_measurement: str | None = None
    default_pitch_deg: float | None = None
    default_gen_rpm: float | None = None
    analysis_label_stage: Literal["comparison", "preprocessing"] = "comparison"

    def __post_init__(self) -> None:
        """Reject options the facts or the analysis set, fields v0 does not have, and an unknown label stage."""
        if self.analysis_label_stage not in ("comparison", "preprocessing"):
            msg = f"{self.name}: analysis_label_stage must be 'comparison' or 'preprocessing'"
            raise MethodConfigurationError(msg)
        set_by_facts = sorted(set(self.options) & FACT_FIELDS)
        if set_by_facts:
            msg = f"{self.name}: options {set_by_facts} come from the facts and the analysis"
            raise MethodConfigurationError(msg)
        unknown = sorted(set(self.options) - set(WindUpConfig.model_fields))
        if unknown:
            msg = f"{self.name}: options {unknown} are not fields of v0's WindUpConfig"
            raise MethodConfigurationError(msg)
        toggle = sorted(set(self.options.get("toggle", {})) - TOGGLE_OPTIONS)
        if toggle:
            msg = f"{self.name}: toggle options {toggle} are not among {sorted(TOGGLE_OPTIONS)}"
            raise MethodConfigurationError(msg)
        turbine_type = sorted(set(self.turbine_type) - (set(TurbineType.model_fields) - TURBINE_TYPE_FACTS))
        if turbine_type:
            msg = f"{self.name}: turbine_type fields {turbine_type} are facts or not fields of v0's TurbineType"
            raise MethodConfigurationError(msg)

    def estimate(self, facts: WindFarmFacts, analyses: Sequence[AnalysisSpec]) -> list[AnalysisOutput]:
        """Answer each analysis with v0's run over its subjects, against its references.

        A run that fails for want of data leaves every subject of the analysis unmeasured.
        """
        for analysis in analyses:
            facts.check(analysis)
        if ROTOR_DIAMETER_COL not in facts.turbines.columns:
            msg = f"{self.name} needs the turbine table's {ROTOR_DIAMETER_COL!r}"
            raise MethodConfigurationError(msg)
        facts.columns.require_roles(["turbine", "active_power"])
        return [self._answer(facts, analysis) for analysis in analyses]

    def _answer(self, facts: WindFarmFacts, analysis: AnalysisSpec) -> AnalysisOutput:
        """Run v0 once per uplift the analysis reports, and give its results.

        A placebo turbine's row is v0's result for it as a placebo test; one v0 did not test, which
        it does only with more than one reference, is a medium warning.
        """
        subjects = facts.subjects_of(analysis)
        placebo = list(analysis.placebo_timing)
        warnings: list[AnalysisWarning] = list(facts.warnings())
        runs: dict[str, tuple[pd.DataFrame, pd.DataFrame | str] | str] = {}
        for report in (r for r in (AEP, CAMPAIGN) if r in analysis.report):
            runs[report] = self._run_or_reason(facts, analysis, use_lt_distribution=report == AEP)
        empty = {c: np.nan for c in PER_TURBINE_COLUMNS if c not in ("turbine", "placebo")}
        rows = {t: {"turbine": t, **empty, "placebo": t in placebo} for t in [*subjects, *placebo]}
        output: dict[str, Any] = {"analysis": analysis}
        per_reference: list[pd.DataFrame] = []
        combined: dict[str, pd.DataFrame] = {}
        for report, run in runs.items():
            if isinstance(run, str):
                for subject in subjects:
                    message = f"{NOT_MEASURED}{self.name} could not run for the {report} uplift: {run}"
                    warnings.append(
                        AnalysisWarning(severity="high", step=MEASURE_STEP, turbine=subject, message=message)
                    )
                continue
            trdf, tdf = run
            per_reference.append(trdf.assign(report=report))
            if isinstance(tdf, str):
                for subject in subjects:
                    message = f"{NOT_MEASURED}v0 could not combine the {report} uplift's test-reference results: {tdf}"
                    warnings.append(
                        AnalysisWarning(severity="high", step=MEASURE_STEP, turbine=subject, message=message)
                    )
                continue
            measured = tdf[tdf["test_wtg"].isin(subjects) & ~tdf["is_ref"]].set_index("test_wtg")
            tested = tdf[tdf["test_wtg"].isin(placebo) & tdf["is_ref"]].set_index("test_wtg")
            for turbine in [*subjects, *placebo]:
                results = tested if turbine in placebo else measured
                if turbine not in results.index:
                    if turbine in placebo:
                        message = f"{PLACEBO_NOT_MEASURED}v0 tested no placebo for the {report} uplift"
                        warnings.append(
                            AnalysisWarning(severity="medium", step=MEASURE_STEP, turbine=turbine, message=message)
                        )
                    continue
                rows[turbine][f"{report}_uplift"] = float(results.loc[turbine, "p50_uplift"])
                rows[turbine][f"{report}_sigma"] = float(results.loc[turbine, "sigma"])
                pairs = trdf[trdf["test_wtg"] == turbine]
                rows[turbine]["hours_on"] = float(pairs["pp_valid_hours_post"].min())
                rows[turbine]["hours_off"] = float(pairs["pp_valid_hours_pre"].min())
                rows[turbine]["actual_energy"] = _actual_energy(pairs)
            combined[report] = measured
        per_turbine = pd.DataFrame(list(rows.values()), columns=list(PER_TURBINE_COLUMNS))
        output |= self._headline(analysis, per_turbine=per_turbine, combined=combined, warnings=warnings)
        output["per_turbine"] = per_turbine
        output["per_reference"] = pd.concat(per_reference, ignore_index=True) if per_reference else None
        return AnalysisOutput(**output, warnings=tuple(warnings))

    def _headline(
        self,
        analysis: AnalysisSpec,
        *,
        per_turbine: pd.DataFrame,
        combined: Mapping[str, pd.DataFrame],
        warnings: list[AnalysisWarning],
    ) -> dict[str, float]:
        """Return the analysis's uplift and sigma for each report v0 combined, from its subjects' results.

        One subject gives its own. A group gives v0's total uplift of the group, or for a ``net``
        combination the net uplift and no sigma, with a low warning added to ``warnings``.
        """
        if len(subjects_only(per_turbine)) == 1 or (analysis.combination != NET and combined):
            return {
                f"{report}_{figure}": value
                for report, measured in combined.items()
                for figure, value in zip(("uplift", "sigma"), _group_total(measured), strict=True)
            }
        if not combined:
            return {}
        net = combine_subjects(AnalysisOutput(analysis=analysis, per_turbine=per_turbine))
        message = (
            f"{self.name} gives no sigma for a group's net uplift; each subject's sigma is in per_turbine, and "
            "wind_up.combine_subjects gives the group's as a range"
        )
        warnings.append(AnalysisWarning(severity="low", step=COMBINE_STEP, turbine="all", message=message))
        return {f"{report}_uplift": getattr(net, f"{report}_uplift") for report in combined}

    def _run_or_reason(
        self, facts: WindFarmFacts, analysis: AnalysisSpec, *, use_lt_distribution: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame | str] | str:
        """Return :meth:`_run`'s results, or why v0 could not run for want of data."""
        try:
            return self._run(facts, analysis, use_lt_distribution=use_lt_distribution)
        except MethodConfigurationError:
            raise
        except (ValueError, RuntimeError) as error:
            return str(error)

    def _run(
        self, facts: WindFarmFacts, analysis: AnalysisSpec, *, use_lt_distribution: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame | str]:
        """Return v0's per test-reference results of one run and their combination per turbine.

        The combination is the reason instead when v0 cannot combine the results.
        """
        cfg = self._config(facts, analysis, use_lt_distribution=use_lt_distribution)
        upgrade = facts.upgrade(analysis.upgrade)
        records = self._records(facts, analysis, upgrade=upgrade, first=cfg.upgrade_first_dt_utc_start)
        plot_cfg = PlotConfig(
            show_plots=False,
            save_plots=self.save_plots,
            skip_per_turbine_plots=not self.per_turbine_plots,
            plots_dir=cfg.out_dir / "plots",
        )
        inputs = AssessmentInputs.from_cfg(
            cfg=cfg,
            plot_cfg=plot_cfg,
            scada_df=records,
            metadata_df=_metadata(facts),
            toggle_df=_toggle_frame(
                upgrade,
                index=pd.DatetimeIndex(records.index.unique()).sort_values(),
                turbines=facts.turbine_names,
                placebo=analysis.placebo_timing,
                borrowed_from=facts.subjects_of(analysis)[0],
            ),
            reanalysis_datasets=[ReanalysisDataset(id=r.name, data=r.records) for r in facts.reanalysis],
            cache_dir=self.cache_dir,
            mast_or_lidar_datasets=[
                MastOrLiDARDataset(id=m.name, data=m.records) for m in facts.non_turbine_measurements
            ],
            comparison_exclusions=(
                _comparison_exclusions(facts, analysis) if self.analysis_label_stage == "comparison" else None
            ),
        )
        trdf = run_wind_up_analysis(inputs)
        try:
            tdf = combine_results(
                trdf,
                auto_choose_refs=self.auto_choose_refs,
                plot_config=plot_cfg if self.save_plots else None,
            )
        except (ValueError, KeyError) as error:
            return trdf, f"{type(error).__name__}: {error}"
        return trdf, tdf

    def _config(self, facts: WindFarmFacts, analysis: AnalysisSpec, *, use_lt_distribution: bool) -> WindUpConfig:
        """Return v0's configuration for one run of ``analysis``."""
        upgrade = facts.upgrade(analysis.upgrade)
        subjects = facts.subjects_of(analysis)
        turbines = _turbines(facts, turbine_type=self.turbine_type)
        by_name = {t.name: t for t in turbines}
        references = [r for r in facts.turbine_references(analysis) if r not in subjects]
        index = pd.DatetimeIndex(facts.records.index.unique()).sort_values()
        timebase = facts.timebase
        start, end = analysis.period
        last = (end - timebase) if end is not None else index.max()
        first = _upgrade_first(upgrade, subjects=subjects, index=index, start=start)
        out_root = Path(self.out_dir) if self.out_dir is not None else Path(tempfile.mkdtemp(prefix="v0_"))
        suffix = "" if len(analysis.report) == 1 else ("_aep" if use_lt_distribution else "_campaign")
        cfg: dict[str, Any] = {
            **{k: v for k, v in self.options.items() if k != "toggle"},
            "assessment_name": analysis.name,
            "timebase_s": int(timebase.total_seconds()),
            "out_dir": out_root / f"{analysis.name}{suffix}",
            "test_wtgs": [by_name[s] for s in subjects],
            "ref_wtgs": [by_name[r] for r in references],
            "non_wtg_ref_names": facts.measurement_references(analysis),
            "upgrade_first_dt_utc_start": first,
            "analysis_last_dt_utc_start": last,
            "use_lt_distribution": use_lt_distribution,
            "asset": Asset(name=self.site_name, wtgs=turbines, masts_and_lidars=_masts_and_lidars(facts)),
            "exclusion_periods_utc": _exclusion_periods(facts),
            "northing_corrections_utc": [
                (t, pd.Timestamp(when), float(offset)) for t, when, offset in facts.north_offsets
            ],
        }
        if upgrade.mode == "toggle":
            cfg["toggle"] = {
                "name": upgrade.name,
                **self.options.get("toggle", {}),
                "toggle_file_per_turbine": _per_turbine_toggle(upgrade, analysis=analysis),
                "placebo_toggle_wtgs": _v0_placebo(upgrade, analysis=analysis, references=references),
                "toggle_filename": "not_used",
                "toggle_change_settling_filter_seconds": int(upgrade.settling_time.total_seconds()),
            }
        _derive_dates(cfg, period_start=start, use_lt_distribution=use_lt_distribution)
        cfg["out_dir"].mkdir(parents=True, exist_ok=True)
        return WindUpConfig.model_validate(cfg)

    def _records(
        self, facts: WindFarmFacts, analysis: AnalysisSpec, *, upgrade: Upgrade, first: pd.Timestamp
    ) -> pd.DataFrame:
        """Return the facts' records in v0's columns, with ``exclude_row`` flagging what v0 must leave out.

        The analyst's label columns and the site state column are left out: v0 reads them through
        ``exclude_row`` alone.
        """
        exclude = _excluded_records(
            facts,
            analysis,
            upgrade=upgrade,
            first=first,
            analysis_labels=self.analysis_label_stage == "preprocessing",
        )
        dropped = [label.column for label in facts.labels]
        if facts.operating_states.site_state_column is not None:
            dropped.append(facts.operating_states.site_state_column)
        records = facts.records.drop(columns=list(dict.fromkeys(dropped)))
        converted = (
            self.to_v0_records(records)
            if self.to_v0_records is not None
            else v0_records(records, columns=facts.columns, timebase=facts.timebase)
        )
        converted = self._with_turbine_signal_defaults(converted, facts)
        if exclude.any():
            converted = converted.assign(**{EXCLUDE_ROW_COL: exclude.astype(int)})
        return converted.rename_axis(index=TIMESTAMP_COL)

    def _with_turbine_signal_defaults(self, records: pd.DataFrame, facts: WindFarmFacts) -> pd.DataFrame:
        """Fill missing columns with the requested proxy wind signals and operating defaults."""
        defaults = {
            DataColumns.pitch_angle_mean: self.default_pitch_deg,
            DataColumns.gen_rpm_mean: self.default_gen_rpm,
        }
        additions: dict[str, Any] = {
            column: default for column, default in defaults.items() if default is not None and column not in records
        }
        if self.turbine_wind_measurement is not None:
            measurement = next(
                (m for m in facts.non_turbine_measurements if m.name == self.turbine_wind_measurement), None
            )
            if measurement is None:
                msg = f"{self.name}: unknown turbine wind measurement {self.turbine_wind_measurement!r}"
                raise MethodConfigurationError(msg)
            wind = measurement.records.reindex(records.index)
            for role, column in (
                (WIND_SPEED_ROLE, DataColumns.wind_speed_mean),
                (WIND_DIRECTION_ROLE, DataColumns.yaw_angle_mean),
            ):
                if column not in records:
                    additions[column] = wind[measurement.columns[role]].to_numpy()
        return records.assign(**additions) if additions else records


def _group_total(measured: pd.DataFrame) -> tuple[float, float]:
    """Return v0's uplift and sigma of the measured subjects: one subject's own, or v0's total uplift of the group."""
    if measured.empty:
        return float("nan"), float("nan")
    if len(measured) == 1:
        return float(measured["p50_uplift"].iloc[0]), float(measured["sigma"].iloc[0])
    total = _calculate_total_uplift_of_turbine_group(measured.reset_index())
    return float(total["p50_uplift"]), float(total["sigma"])


def _actual_energy(pairs: pd.DataFrame) -> float:
    """Return a subject's energy with the upgrade on, in kWh: its pairs' mean power times valid hours, averaged."""
    if "mean_power_post" not in pairs.columns:
        return float("nan")
    return float((pairs["mean_power_post"] * pairs["pp_valid_hours_post"]).mean())


def _turbines(facts: WindFarmFacts, *, turbine_type: Mapping[str, Any]) -> list[Turbine]:
    """Return v0's turbines from the turbine table, in its order."""
    table = facts.turbines
    type_fields = {"turbine_type": "turbine", **turbine_type}
    absent = sorted(
        name
        for name, info in TurbineType.model_fields.items()
        if info.is_required() and name not in type_fields and name not in TURBINE_TYPE_FACTS
    )
    if absent:
        msg = f"v0 needs the turbine_type field(s) {absent}"
        raise MethodConfigurationError(msg)
    return [
        Turbine(
            name=str(row[NAME_COL]),
            turbine_type=TurbineType(
                rotor_diameter_m=float(row[ROTOR_DIAMETER_COL]),
                rated_power_kw=float(row["rated_power_kw"]),
                **type_fields,
            ),
            latitude=float(row[LATITUDE_COL]),
            longitude=float(row[LONGITUDE_COL]),
        )
        for _, row in table.iterrows()
    ]


def _masts_and_lidars(facts: WindFarmFacts) -> list[MastOrLidar]:
    """Return v0's masts and LiDARs from the non-turbine measurements. Raises when one lacks a role v0 reads."""
    out = []
    for m in facts.non_turbine_measurements:
        absent = [role for role in (WIND_SPEED_ROLE, WIND_DIRECTION_ROLE) if role not in m.columns]
        if absent:
            msg = f"v0 reads the {absent} column role(s) of measurement {m.name!r}, which it does not declare"
            raise MethodConfigurationError(msg)
        out.append(
            MastOrLidar(
                name=m.name,
                latitude=m.latitude,
                longitude=m.longitude,
                wind_speed_column=m.columns[WIND_SPEED_ROLE],
                wind_direction_column=m.columns[WIND_DIRECTION_ROLE],
            )
        )
    return out


def _metadata(facts: WindFarmFacts) -> pd.DataFrame:
    """Return v0's metadata: each turbine's position, timestamps in UTC at the start of each period."""
    table = facts.turbines
    return pd.DataFrame(
        {
            "Name": table[NAME_COL].astype(str).to_numpy(),
            "Latitude": table[LATITUDE_COL].to_numpy(dtype=float),
            "Longitude": table[LONGITUDE_COL].to_numpy(dtype=float),
            "TimeZone": "UTC",
            "TimeSpanMinutes": facts.timebase.total_seconds() / 60,
            "TimeFormat": "Start",
        }
    )


def _exclusion_periods(facts: WindFarmFacts) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Return v0's exclusion periods: the facts' exclusions, then every installation window."""
    periods = [("ALL" if who is None else who, pd.Timestamp(a), pd.Timestamp(b)) for who, a, b in facts.exclusions]
    for upgrade in facts.upgrades:
        for turbine, windows in upgrade.installation_windows.items():
            periods.extend((turbine, a, b) for a, b in windows if b > a)
    return periods


def _upgrade_first(
    upgrade: Upgrade, *, subjects: Sequence[str], index: pd.DatetimeIndex, start: pd.Timestamp | None
) -> pd.Timestamp:
    """Return v0's first upgraded timestamp: the end of a permanent changeover, or a toggle's first stated record.

    A toggle's first record in the on or off state within the period and each subject's own
    signal span, over the subjects. Raises when the subjects of a permanent upgrade change over at different times.
    """
    if upgrade.mode != "toggle":
        ends = {upgrade.changeover(s)[1] for s in subjects}
        if len(ends) > 1:
            msg = f"v0 runs one changeover; subjects {list(subjects)} change over at {sorted(ends)}"
            raise MethodConfigurationError(msg)
        return next(iter(ends))
    firsts = []
    for subject in subjects:
        signal_start = upgrade.signal_span(turbine=subject)[0]
        earliest = max(t for t in (start, signal_start, index.min()) if t is not None)
        within = index[index >= earliest]
        stated = within[np.isin(upgrade.states(within, turbine=subject), [ON, OFF])]
        if len(stated):
            firsts.append(stated.min())
    if not firsts:
        msg = f"upgrade {upgrade.name!r} states no on or off record for {list(subjects)} within the period"
        raise ValueError(msg)
    return min(firsts)


def _per_turbine_toggle(upgrade: Upgrade, *, analysis: AnalysisSpec) -> bool:
    """Return whether v0 reads a toggle frame per turbine: for frames per turbine or a placebo."""
    return upgrade.mode == "toggle" and (upgrade.per_turbine or bool(analysis.placebo_timing))


def _v0_placebo(upgrade: Upgrade, *, analysis: AnalysisSpec, references: Sequence[str]) -> list[str]:
    """Return v0's placebo turbines: with a toggle frame per turbine, every reference the upgrade does not toggle.

    v0 keeps all of a placebo turbine's records as a reference, as it does every reference's with
    one shared toggle frame.
    """
    if not _per_turbine_toggle(upgrade, analysis=analysis):
        return []
    return [r for r in references if r not in upgrade.turbines]


def _toggle_frame(
    upgrade: Upgrade,
    *,
    index: pd.DatetimeIndex,
    turbines: Sequence[str],
    placebo: Mapping[str, ToggleSchedule | pd.DataFrame] | None = None,
    borrowed_from: str | None = None,
) -> pd.DataFrame | None:
    """Return the toggle frame v0 reads: the upgrade's own frame, or one built from its schedule.

    With a frame per turbine or a placebo, v0's frame is indexed by turbine and timestamp over
    ``turbines``: a turbine the upgrade toggles has its own timing, a placebo turbine its placebo
    timing, and every other turbine the timing of ``borrowed_from``, a turbine the upgrade toggles,
    as v0 tests a reference against a shared toggle.
    """
    if upgrade.mode != "toggle":
        return None
    if not upgrade.per_turbine and not placebo:
        return _as_frame(upgrade.timing, index=index)  # type: ignore[arg-type]
    own = {t: upgrade.toggle_timing(t) for t in upgrade.turbines} | dict(placebo or {})
    borrowed = upgrade.toggle_timing(borrowed_from or upgrade.turbines[0])
    frames = {t: _as_frame(own.get(t, borrowed), index=index) for t in turbines}
    return pd.concat(frames, names=[DataColumns.turbine_name, TIMESTAMP_COL])


def _as_frame(timing: ToggleSchedule | pd.DataFrame, *, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return a toggle frame, or the frame a schedule gives over ``index``."""
    return build_toggle_df(index, timing) if isinstance(timing, ToggleSchedule) else timing


def _left_out_by_analysis(facts: WindFarmFacts, analysis: AnalysisSpec) -> np.ndarray:
    """Return a mask over the facts' records of the subjects' records the analysis's labels leave out."""
    records = facts.records
    turbines = records[facts.columns.turbine].astype(str).to_numpy()
    labels = {label.name: label.column for label in facts.labels}
    kept = np.ones(len(records), dtype=bool)
    for name in analysis.include_labels:
        kept &= records[labels[name]].to_numpy(dtype=bool)
    for name in analysis.exclude_labels:
        kept &= ~records[labels[name]].to_numpy(dtype=bool)
    return np.isin(turbines, facts.subjects_of(analysis)) & ~kept


def _comparison_exclusions(facts: WindFarmFacts, analysis: AnalysisSpec) -> dict[str, pd.DatetimeIndex]:
    """Return, per subject, the timestamps of its records the analysis's labels leave out."""
    records = facts.records
    left_out = _left_out_by_analysis(facts, analysis)
    turbines = records[facts.columns.turbine].astype(str).to_numpy()
    index = pd.DatetimeIndex(records.index)
    return {s: index[left_out & (turbines == s)] for s in facts.subjects_of(analysis)}


def _excluded_records(
    facts: WindFarmFacts, analysis: AnalysisSpec, *, upgrade: Upgrade, first: pd.Timestamp, analysis_labels: bool
) -> np.ndarray:
    """Return a mask over the facts' records of those v0 must leave out.

    A record a label withholds from the uplift; one whose site state is not valid for uplift; one
    inside an operating state override; with ``analysis_labels``, a subject's record the analysis's
    labels leave out; and, from the first upgraded timestamp, a record of a turbine an override puts
    in neither upgrade state.
    """
    records = facts.records
    index = pd.DatetimeIndex(records.index)
    turbines = records[facts.columns.turbine].astype(str).to_numpy()
    exclude = np.zeros(len(records), dtype=bool)
    for label in facts.labels:
        if UPLIFT in label.withholds:
            exclude |= records[label.column].to_numpy(dtype=bool)
    config = facts.operating_states
    if config.site_state_column is not None:
        not_uplift = [name for name, validity in config.validity().items() if not validity.uplift]
        exclude |= records[config.site_state_column].isin(not_uplift).to_numpy()
    for who, a, b, _ in facts.operating_state_overrides:
        exclude |= _in_window(index, a, b) & ((turbines == who) if who is not None else True)
    if analysis_labels:
        exclude |= _left_out_by_analysis(facts, analysis)
    for override in facts.upgrades:
        for who, a, b, state in override.upgrade_state_overrides:
            if state == NEITHER and b > first:
                exclude |= _in_window(index, max(a, first), b) & ((turbines == who) if who is not None else True)
            elif state != NEITHER and override is upgrade:
                msg = f"v0 cannot take the override of upgrade {upgrade.name!r} to {state!r}; it reads one timing"
                raise MethodConfigurationError(msg)
    return exclude


def _in_window(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Return a mask over ``index`` of ``[start, end)``."""
    return np.asarray((index >= start) & (index < end))


def _derive_dates(cfg: dict[str, Any], *, period_start: pd.Timestamp | None, use_lt_distribution: bool) -> None:
    """Fill v0's dates not given, as v0's YAML loader works them out from its ``years_*`` fields.

    A toggle's analysis starts at its first upgraded timestamp. A permanent upgrade's pre period
    ends ``years_offset_for_pre_period`` years before the analysis does, at latest a period before
    the upgrade, and starts that many years before the upgrade; an analysis period starting later
    raises. The long-term period ends a week before a toggle starts, or with the pre
    period, and lasts ``years_for_lt_distribution`` years; the detrend period ends with it, or with
    the analysis for a toggle detrended on its off records, and lasts ``years_for_detrend`` years.

    A toggle without ``years_for_lt_distribution`` that gives no AEP uplift takes the analysis period as
    its long-term period, and one detrended on its off records without ``years_for_detrend`` takes it as
    its detrend period.
    """
    toggle = cfg.get("toggle")
    lt_is_analysis = toggle is not None and not use_lt_distribution and "years_for_lt_distribution" not in cfg
    detrend_is_analysis = (
        toggle is not None
        and toggle.get("detrend_data_selection") == "use_toggle_off_data"
        and "years_for_detrend" not in cfg
    )
    if lt_is_analysis:
        cfg["years_for_lt_distribution"] = 0
    if detrend_is_analysis:
        cfg["years_for_detrend"] = 0
    absent = [k for k in REQUIRED_OPTIONS if k not in cfg]
    if absent:
        msg = f"v0 needs the option(s) {absent}"
        raise MethodConfigurationError(msg)
    upgrade_first = cfg["upgrade_first_dt_utc_start"]
    analysis_last = cfg["analysis_last_dt_utc_start"]
    if toggle is not None:
        cfg.setdefault("analysis_first_dt_utc_start", upgrade_first)
        lt_last = upgrade_first - LT_WEEK_BEFORE_TOGGLE
    else:
        if "prepost" not in cfg:
            years = cfg.get("years_offset_for_pre_period")
            if years is None:
                msg = "v0 needs the option years_offset_for_pre_period, or prepost, for a permanent upgrade"
                raise MethodConfigurationError(msg)
            pre_last = analysis_last - pd.DateOffset(years=years)
            pre_last = min(pre_last, upgrade_first - pd.Timedelta(seconds=cfg["timebase_s"]))
            pre_first = upgrade_first - pd.DateOffset(years=years)
            if period_start is not None and period_start > pre_first:
                msg = (
                    f"the analysis starts at {period_start}, after v0's pre period starts at {pre_first}; "
                    "give the pre period as the option prepost"
                )
                raise MethodConfigurationError(msg)
            cfg["prepost"] = PrePost(
                pre_first_dt_utc_start=pre_first,
                pre_last_dt_utc_start=pre_last,
                post_first_dt_utc_start=upgrade_first,
                post_last_dt_utc_start=analysis_last,
            )
        prepost = cfg["prepost"] if isinstance(cfg["prepost"], PrePost) else PrePost.model_validate(cfg["prepost"])
        cfg.setdefault("analysis_first_dt_utc_start", prepost.pre_first_dt_utc_start)
        lt_last = prepost.pre_last_dt_utc_start
    analysis_span = {"first_dt_utc_start": cfg["analysis_first_dt_utc_start"], "last_dt_utc_start": analysis_last}
    for period, is_analysis in (("lt", lt_is_analysis), ("detrend", detrend_is_analysis)):
        if is_analysis:
            for end, when in analysis_span.items():
                cfg.setdefault(f"{period}_{end}", when)
    cfg.setdefault("lt_last_dt_utc_start", lt_last)
    cfg.setdefault(
        "lt_first_dt_utc_start", cfg["lt_last_dt_utc_start"] - pd.DateOffset(years=cfg["years_for_lt_distribution"])
    )
    detrend_last = cfg["lt_last_dt_utc_start"]
    if toggle is not None and toggle.get("detrend_data_selection") == "use_toggle_off_data":
        detrend_last = analysis_last
    cfg.setdefault("detrend_last_dt_utc_start", detrend_last)
    cfg.setdefault(
        "detrend_first_dt_utc_start", cfg["detrend_last_dt_utc_start"] - pd.DateOffset(years=cfg["years_for_detrend"])
    )
