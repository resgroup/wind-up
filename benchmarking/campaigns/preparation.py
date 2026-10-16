"""Part A of the method, data preparation (steps 1 to 6), done once per dataset.

:func:`prepare_data` labels the operating states, aligns the reanalysis, norths the whole farm over the
whole record, finds the possible yaw-alignment changes, adds the waking columns and, when asked, writes
the outputs of each step. The composed run and the benchmark studies share it, so a method is judged on
the same prepared data wherever it runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import pandas as pd

from benchmarking.baselines.power_model.features import era5_feature_frame
from benchmarking.campaigns.declaration import layout_coords
from benchmarking.campaigns.loader import centroid
from benchmarking.diagnostics import stages
from benchmarking.diagnostics.context import era5_source_label, infer_timebase
from benchmarking.diagnostics.features import write_feature_outputs
from benchmarking.diagnostics.input_data import write_input_data_plots
from benchmarking.diagnostics.operating_states import (
    NORTHING_VIEWS,
    WAKING_VIEWS,
    write_operating_state_plots,
    write_validity_plots,
)
from benchmarking.diagnostics.reanalysis import write_reanalysis_outputs
from benchmarking.diagnostics.waking_scenarios import write_waking_outputs
from benchmarking.diagnostics.yaw_changes import possible_yaw_change_report, write_possible_yaw_changes
from benchmarking.harness.northing import (
    DEFAULT_NORTHING_ROLES,
    apply_north_tables,
    north_offsets_of,
    north_tables,
    northing_rows,
)
from benchmarking.harness.operating_state import VALID_NORTHING_COL, OperatingStateConfig, label_operating_states
from benchmarking.harness.reanalysis import ERA5_WD_RAW, prepare_reanalysis
from benchmarking.harness.waking import WAKING_COLUMNS, add_waking_scenarios, waking_directions

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.harness.reanalysis import ReanalysisResult
    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedData:
    """What Part A produced.

    :param scada_df: every record provided, with the operating-state columns of step 2 and the waking
        columns of step 5
    :param reanalysis: the aligned reanalysis of step 3 and its checks
    :param north_tables: each turbine's north table of step 4, over the whole record
    :param possible_changes: the possible yaw-alignment changes of step 4
    :param timebase: the records' period
    """

    scada_df: pd.DataFrame
    reanalysis: ReanalysisResult
    north_tables: dict[str, pd.DataFrame]
    possible_changes: pd.DataFrame
    timebase: pd.Timedelta

    @property
    def aligned_era5(self) -> pd.DataFrame:
        """The reanalysis on the SCADA index."""
        return self.reanalysis.aligned

    def with_north_offsets(self, spec: CampaignSpec) -> CampaignSpec:
        """Return ``spec`` declaring the north tables found here, so the measurement applies them as given."""
        return replace(spec, north_offsets=north_offsets_of(self.north_tables))

    def possible_changes_by_turbine(self) -> dict[str, list[pd.Timestamp]]:
        """Each turbine's possible yaw-alignment change dates, for the warnings of step 8."""
        return {str(t): list(rows["timestamp"]) for t, rows in self.possible_changes.groupby("turbine", sort=True)}


def prepare_data(
    scada_df: pd.DataFrame,
    *,
    spec: CampaignSpec,
    columns: ColumnSchema,
    era5_hourly_df: pd.DataFrame,
    operating_state: OperatingStateConfig | None = None,
    cell_selection: Literal["land", "sea"] = "land",
    out_dir: Path | None = None,
    plots: bool = True,
) -> PreparedData:
    """Run Part A on ``scada_df`` and return what it produced.

    :param scada_df: long-format SCADA of every turbine over every record provided, indexed by
        period-start UTC timestamps
    :param spec: the campaign, for its layout, rating, test turbines and changeovers
    :param columns: the schema ``scada_df`` is keyed by
    :param era5_hourly_df: hourly reanalysis covering the record, or one already aligned to it
    :param operating_state: the site's operating-state labels; the generic states alone when ``None``
    :param cell_selection: the reanalysis grid cell asked for, named in the reanalysis summary
    :param out_dir: where each step's outputs go, in a folder per step; nothing is written when ``None``
    :param plots: write the plots as well as the tables
    """
    index = pd.DatetimeIndex(scada_df.index.unique()).sort_values()
    timebase = infer_timebase(index)
    config = operating_state if operating_state is not None else OperatingStateConfig()
    scada_df = label_operating_states(
        scada_df, columns=columns, config=config, timebase=timebase, rated_power_kw=spec.rated_power_kw
    )
    changeovers = spec.changeovers()
    draw = out_dir is not None and plots
    if out_dir is not None and draw:
        write_input_data_plots(
            scada_df,
            columns=columns,
            out_dir=out_dir / stages.CHANGES,
            changeovers=changeovers,
            rated_power_kw=spec.rated_power_kw,
        )
        write_operating_state_plots(
            scada_df,
            columns=columns,
            timebase=timebase,
            out_dir=out_dir / stages.OPERATING_STATES,
            changeovers=changeovers,
        )
        used_for_northing = scada_df.assign(
            **{VALID_NORTHING_COL: northing_rows(scada_df, columns=columns, rated_power_kw=spec.rated_power_kw)}
        )
        for frame, views, stage in (
            (used_for_northing, NORTHING_VIEWS, stages.NORTHING),
            (scada_df, WAKING_VIEWS, stages.WAKING),
        ):
            write_validity_plots(frame, views=views, columns=columns, timebase=timebase, out_dir=out_dir / stage)

    unchanged = [
        w for w in layout_coords(spec.layout) if w not in set(spec.upgraded_turbines) | set(spec.excluded_turbines)
    ]
    reanalysis = prepare_reanalysis(
        era5_hourly_df, scada_df=scada_df, columns=columns, unchanged=unchanged, timebase=timebase
    )
    if out_dir is not None and draw:
        write_reanalysis_outputs(
            reanalysis,
            era5_hourly_df=era5_hourly_df,
            cell_selection=cell_selection,
            out_dir=out_dir / stages.REANALYSIS,
        )

    tables = north_tables(
        scada_df,
        columns=columns,
        north_offsets=spec.north_offsets,
        rated_power_kw=spec.rated_power_kw,
        layout=spec.layout,
        era5_wd=reanalysis.aligned[ERA5_WD_RAW],
        out_dir=out_dir / stages.NORTHING if out_dir is not None else None,
        plots=plots,
    )
    yaw = {
        "tables": tables,
        "columns": columns,
        "rated_power_kw": spec.rated_power_kw,
        "timebase": timebase,
        "changeovers": changeovers,
        "analysis_period": spec.period_bounds(),
    }
    if out_dir is not None:
        possible_changes = write_possible_yaw_changes(scada_df, out_dir=out_dir / stages.NORTHING, plots=plots, **yaw)
    else:
        possible_changes, _ = possible_yaw_change_report(scada_df, **yaw)

    scada_df = _add_waking(
        scada_df,
        spec=spec,
        tables=tables,
        columns=columns,
        timebase=timebase,
        out_dir=out_dir / stages.WAKING if draw and out_dir is not None else None,
    )
    if out_dir is not None and draw:
        write_feature_outputs(
            scada_df,
            era5_features=era5_feature_frame(reanalysis.aligned),
            era5_label=era5_source_label(*centroid(layout_coords(spec.layout))),
            turbine_col=columns.turbine,
            changeovers=changeovers,
            out_dir=out_dir / stages.FEATURES,
        )
    return PreparedData(
        scada_df=scada_df,
        reanalysis=reanalysis,
        north_tables=tables,
        possible_changes=possible_changes,
        timebase=timebase,
    )


def _add_waking(
    scada_df: pd.DataFrame,
    *,
    spec: CampaignSpec,
    tables: dict[str, pd.DataFrame],
    columns: ColumnSchema,
    timebase: pd.Timedelta,
    out_dir: Path | None,
    roles: Sequence[str] = DEFAULT_NORTHING_ROLES,
) -> pd.DataFrame:
    """Return ``scada_df`` with the step 5 waking columns, found from the northed direction.

    Returns ``scada_df`` unchanged when it has no nacelle position. Writes the step's outputs under
    ``out_dir`` when given.
    """
    if columns.nacelle_position is None or columns.nacelle_position not in scada_df.columns:
        logger.warning("no waking scenarios: the SCADA has no nacelle position")
        return scada_df
    northed = apply_north_tables(scada_df, tables=tables, columns=columns, roles=roles)
    directions = waking_directions(northed, columns=columns, layout=spec.layout, test_turbines=spec.upgraded_turbines)
    waking = add_waking_scenarios(
        northed,
        columns=columns,
        layout=spec.layout,
        test_turbines=spec.upgraded_turbines,
        directions=directions,
    )
    if out_dir is not None:
        write_waking_outputs(
            waking, directions=directions, layout=spec.layout, columns=columns, timebase=timebase, out_dir=out_dir
        )
    return scada_df.assign(**{col: waking[col].to_numpy() for col in WAKING_COLUMNS})
