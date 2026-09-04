"""The shared northing step: north-calibrate every turbine's direction, upstream of every method.

Runs on both paths that feed methods -- the campaign runner and the study replicates -- so every
method inherits the correction rather than each hand-rolling one. The step writes
``columns.northed(role)`` alongside the untouched original, so plots and diagnostics of the raw
signal keep meaning what they say; whether it has run is written in the frame as the presence of
that column, with no separate flag to disagree with it.

``north_offsets`` decides which of two things happens:

* ``None`` -- discover the corrections from the data;
* a list (possibly empty) -- apply exactly those, discovering nothing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_up.northing import (
    DEFAULT_NORTHING,
    NorthingSettings,
    apply_north_table,
    north_farm,
    write_north_table_yaml,
    yaw_usable,
)
from wind_up.northing_plots import plot_northing, plot_northing_farm

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)

# The direction roles corrected by default. One table per turbine is derived from its nacelle
# position and may be applied to further direction channels of the same turbine.
DEFAULT_NORTHING_ROLES: tuple[str, ...] = ("nacelle_position",)

# The plots show the residual against reanalysis: it is the anchor available here, whereas the
# farm consensus pass 2 uses is internal to north_farm.
_PLOT_REFERENCE_NAME = "reanalysis"

# The discovered table, written in the format ``north_offsets`` and v0's
# ``northing_corrections_utc`` both read, so it can be hand edited and supplied back as a prior.
NORTH_TABLE_YAML = "northing_corrections.yaml"

# Open-Meteo's hub-height wind direction, the reanalysis anchor discovery is measured against.
ERA5_WD_COL = "wind_direction_100m"


def era5_direction(era5_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Return the hourly ERA5 wind direction carried onto ``index``, held within each hour."""
    hourly = era5_df[ERA5_WD_COL]
    return hourly.reindex(hourly.index.union(index)).ffill(limit=6).reindex(index)


def _north_table_from_offsets(
    offsets: Sequence[tuple[str, pd.Timestamp, float]], *, turbine: str, start: pd.Timestamp
) -> pd.DataFrame:
    """Return one turbine's declared north table, or a zero-offset table when none is declared.

    :raises ValueError: if the turbine's first declared offset begins after ``start``, which would
        leave the earliest rows to be corrected by a later offset
    """
    rows = sorted(((ts, off) for (t, ts, off) in offsets if t == turbine), key=lambda e: e[0])
    if not rows:
        return pd.DataFrame({"timestamp": pd.DatetimeIndex([start]), "north_offset": [0.0]})
    if rows[0][0] > start:
        msg = (
            f"the declared north offsets for {turbine!r} begin at {rows[0][0]}, after the data starts at "
            f"{start}; rows before the first offset would be corrected by a later one. Declare an offset "
            f"covering the start of the data."
        )
        raise ValueError(msg)
    return pd.DataFrame(
        {"timestamp": pd.DatetimeIndex([ts for ts, _ in rows]), "north_offset": [off for _, off in rows]}
    )


def _usable_masks(
    scada_df: pd.DataFrame,
    *,
    columns: ColumnSchema,
    turbines: Sequence[str],
    index: pd.DatetimeIndex,
    reference_deg: np.ndarray,
    rated_power_kw: float,
    timebase_s: float,
) -> dict[str, np.ndarray]:
    """Per-turbine rows usable for northing, positional on the shared ``index``."""
    masks = {}
    for turbine in turbines:
        rows = scada_df[scada_df[columns.turbine] == turbine]
        frame = rows[~rows.index.duplicated()].reindex(index)
        power = frame[columns.active_power].to_numpy(dtype=float)
        # the schema's availability is a "ready to operate" counter, so downtime is what is left
        available = frame[columns.availability].to_numpy(dtype=float)
        masks[turbine] = yaw_usable(
            power=power,
            downtime_s=timebase_s - np.nan_to_num(available, nan=0.0),
            reference_deg=reference_deg,
            rated_power=rated_power_kw,
            timebase_s=timebase_s,
        )
    return masks


def _directions(
    scada_df: pd.DataFrame, *, columns: ColumnSchema, turbines: Sequence[str], index: pd.DatetimeIndex, col: str
) -> dict[str, np.ndarray]:
    """Per-turbine direction signal, positional on the shared ``index``."""
    out = {}
    for turbine in turbines:
        rows = scada_df[scada_df[columns.turbine] == turbine]
        frame = rows[~rows.index.duplicated()].reindex(index)
        out[turbine] = frame[col].to_numpy(dtype=float)
    return out


def north_scada(
    scada_df: pd.DataFrame,
    *,
    columns: ColumnSchema,
    north_offsets: Sequence[tuple[str, pd.Timestamp, float]] | None,
    rated_power_kw: float,
    era5_wd: pd.Series | None = None,
    roles: Sequence[str] = DEFAULT_NORTHING_ROLES,
    settings: NorthingSettings = DEFAULT_NORTHING,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Return ``scada_df`` with a north-calibrated companion column for each direction role.

    One north table per turbine is derived from its ``nacelle_position`` and applied to every
    requested role, so a turbine's channels stay mutually consistent. The originals are untouched.

    :param scada_df: long-format SCADA, timestamps indexed, turbines in ``columns.turbine``
    :param columns: the source-native schema naming the turbine and direction role(s)
    :param north_offsets: ``None`` to discover the corrections, or the exact table to apply
    :param rated_power_kw: turbine rating, for deciding which rows are usable for northing
    :param era5_wd: reanalysis wind direction (deg) covering the frame, the absolute anchor for
        discovery. Required when ``north_offsets`` is ``None``.
    :param roles: the direction roles to write a ``northed_`` companion for
    :param settings: how the changepoint search is bounded, when discovering
    :param out_dir: when given and corrections are discovered, the discovered table
        (:data:`NORTH_TABLE_YAML`, hand-editable and usable as a prior), the farm overview and one
        plot per device are written here, so the correction can be judged rather than trusted
    :return: a copy of ``scada_df`` with ``columns.northed(role)`` added for each role
    """
    columns.require_roles(roles)
    scada_df = scada_df.copy()
    index = pd.DatetimeIndex(scada_df.index.unique()).sort_values()
    turbines = sorted(str(t) for t in scada_df[columns.turbine].unique())
    # A source that does not ship a direction channel has nothing to north; that is a property of
    # the data, not an error. ``require_roles`` has already checked the schema names the roles.
    present = [role for role in roles if getattr(columns, role) in scada_df.columns]
    skipped = [role for role in roles if role not in present]
    if skipped:
        logger.info("no northing for role(s) %s: their columns are not in scada_df", skipped)
    if not turbines or not present:
        return scada_df
    roles = present

    if north_offsets is not None:
        tables = {wtg: _north_table_from_offsets(north_offsets, turbine=wtg, start=index.min()) for wtg in turbines}
        logger.info("applying %d declared northing correction(s); discovering none", len(north_offsets))
    else:
        if era5_wd is None:
            msg = (
                "north_scada needs era5_wd to discover northing corrections: reanalysis is the "
                "absolute anchor, without which a farm that is uniformly wrong looks self-consistent. "
                "Supply era5_wd, or declare north_offsets to apply a known table instead."
            )
            raise ValueError(msg)
        reference = era5_wd.reindex(index).to_numpy(dtype=float)
        timebase_s = _timebase_seconds(index)
        source = columns.nacelle_position
        if source is None or source not in scada_df.columns:
            msg = (
                f"northing discovery needs the nacelle_position column {source!r}, which is not in scada_df; "
                f"the north table for every role is derived from it. Columns present: {sorted(scada_df.columns)}"
            )
            raise ValueError(msg)
        directions = _directions(scada_df, columns=columns, turbines=turbines, index=index, col=source)
        usable = _usable_masks(
            scada_df,
            columns=columns,
            turbines=turbines,
            index=index,
            reference_deg=reference,
            rated_power_kw=rated_power_kw,
            timebase_s=timebase_s,
        )
        tables = north_farm(index, direction_deg=directions, usable=usable, reanalysis_deg=reference, settings=settings)
        found = sum(len(t) - 1 for t in tables.values())
        logger.info("discovered %d northing changepoint(s) across %d turbines", found, len(turbines))
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            write_north_table_yaml(tables, path=out_dir / NORTH_TABLE_YAML)
            _write_northing_plots(
                index, directions=directions, usable=usable, reference=reference, tables=tables, out_dir=out_dir
            )

    turbine_of = scada_df[columns.turbine].to_numpy()
    row_index = pd.DatetimeIndex(scada_df.index)
    for role in roles:
        source_col = getattr(columns, role)
        target = columns.northed(role)
        values = scada_df[source_col].to_numpy(dtype=float).copy()
        for wtg, table in tables.items():
            rows = turbine_of == wtg
            if not rows.any():
                continue
            values[rows] = apply_north_table(row_index[rows], values[rows], north_table=table)
        scada_df[target] = values
    return scada_df


def _write_northing_plots(
    index: pd.DatetimeIndex,
    *,
    directions: dict[str, np.ndarray],
    usable: dict[str, np.ndarray],
    reference: np.ndarray,
    tables: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Write the farm overview and one plot per device, then close the figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figure = plot_northing_farm(
        index,
        direction_deg=directions,
        reference_deg=reference,
        usable=usable,
        north_tables=tables,
        reference_name=_PLOT_REFERENCE_NAME,
        out_dir=out_dir,
    )
    plt.close(figure)
    for device in sorted(directions):
        figure = plot_northing(
            index,
            directions[device],
            reference_deg=reference,
            usable=usable[device],
            north_table=tables[device],
            device=device,
            reference_name=_PLOT_REFERENCE_NAME,
            out_dir=out_dir,
        )
        plt.close(figure)
    logger.info("wrote northing plots for %d device(s) to %s", len(directions), out_dir)


def _timebase_seconds(index: pd.DatetimeIndex) -> float:
    """Return the frame's record length in seconds, from the most common gap between timestamps."""
    if len(index) < 2:  # noqa: PLR2004 - two timestamps are needed for a gap
        return 600.0
    gaps = pd.Series(index).diff().dropna()
    return float(gaps.mode().iloc[0].total_seconds()) if len(gaps) else 600.0
