"""Method step 3: add reanalysis data.

* :func:`normalise_timestamps` -- check the SCADA index against its declared time zone and convention,
  and move it to period-start UTC.
* :func:`interpolate_era5` -- align hourly ERA5 to the SCADA periods: instantaneous fields interpolated
  to each period's centre (directions through their wind vector), hour-ending fields taken from the
  hour the period falls in.
* :func:`check_shift` -- the time shift that best correlates ERA5 with the site wind speed, checked but
  never applied.
* :func:`prepare_reanalysis` -- the step: align, report coverage, and check the shift.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from benchmarking.harness.operating_state import VALID_NORTHING_COL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarking.synthetic import ColumnSchema

logger = logging.getLogger(__name__)

# Neutral aliases of the hub-height wind speed and direction, added beside the Open-Meteo columns.
ERA5_WS = "era5_ws"
ERA5_WD = "era5_wd"
ERA5_WS_RAW = "wind_speed_100m"
ERA5_WD_RAW = "wind_direction_100m"

# Open-Meteo ERA5 fields valid at their timestamp.
INSTANTANEOUS_FIELDS = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
)
# Open-Meteo ERA5 fields that are a total, mean or maximum over the hour ending at their timestamp, or
# categorical.
HOUR_ENDING_FIELDS = (
    "precipitation",
    "rain",
    "snowfall",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_gusts_10m",
    "weather_code",
)
_SPEED_PREFIX = "wind_speed_"
_DIRECTION_PREFIX = "wind_direction_"

_HOUR = pd.Timedelta(hours=1)
SHIFT_SEARCH = pd.Timedelta(hours=24)
SHIFT_WARN = pd.Timedelta(minutes=30)
SHIFT_RAISE = pd.Timedelta(minutes=60)
_MIN_OVERLAP = 3


@dataclass(frozen=True)
class TimestampConvention:
    """How the SCADA timestamps are declared.

    :param convention: whether a timestamp labels the ``start`` or the ``end`` of its period
    :param time_zone: the time zone the SCADA index is in
    """

    convention: Literal["start", "end"] = "start"
    time_zone: str = "UTC"

    def __post_init__(self) -> None:
        """Reject a convention other than ``start`` or ``end``."""
        if self.convention not in ("start", "end"):
            msg = f"timestamp convention must be 'start' or 'end', not {self.convention!r}"
            raise ValueError(msg)


def normalise_timestamps(
    scada_df: pd.DataFrame, *, timestamps: TimestampConvention, timebase: pd.Timedelta
) -> pd.DataFrame:
    """Return ``scada_df`` indexed by period-start UTC.

    :raises ValueError: if the index is timezone-naive or not in the declared time zone
    """
    index = scada_df.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        msg = (
            "the SCADA index must be timezone-aware, in the declared time zone "
            f"({timestamps.time_zone}); it is timezone-naive"
        )
        raise ValueError(msg)
    if _zone_name(str(index.tz)) != _zone_name(timestamps.time_zone):
        msg = f"the SCADA index is in {index.tz}, but the declared time zone is {timestamps.time_zone}"
        raise ValueError(msg)
    out = scada_df.copy()
    utc = index.tz_convert("UTC")
    out.index = utc - timebase if timestamps.convention == "end" else utc
    return out


def _zone_name(zone: str) -> str:
    return "UTC" if zone.upper() in {"UTC", "Z", "ETC/UTC"} else zone


def interpolate_era5(era5_hourly_df: pd.DataFrame, *, index: pd.DatetimeIndex, timebase: pd.Timedelta) -> pd.DataFrame:
    """Return ERA5 on the SCADA periods ``[t, t + timebase)`` of ``index``, which is period-start UTC.

    Instantaneous fields are interpolated linearly to each period's centre, wind directions through
    the wind vector at the same height. Hour-ending fields take the value of the hour ending after
    the period's centre. Outside the ERA5 record the values are NaN. Every column keeps its Open-Meteo
    name, and :data:`ERA5_WS` / :data:`ERA5_WD` are added when the hub-height columns are present.
    A frame already on ``timebase`` (shorter than an hour) is taken as aligned and only reindexed.

    :raises ValueError: if a column is not a known ERA5 field, a direction has no speed at its height,
        or the record has a gap (a missing hour, or a NaN inside it)
    """
    if timebase < _HOUR and len(era5_hourly_df) > 1 and _step(era5_hourly_df.index) == timebase:
        return _with_aliases(era5_hourly_df.drop(columns=[ERA5_WS, ERA5_WD], errors="ignore").reindex(index))
    era5 = _validated(era5_hourly_df)
    hours = era5.index.asi8.astype(float)
    centres = (index + timebase / 2).asi8.astype(float)
    inside = (centres >= hours[0]) & (centres <= hours[-1])
    hour_ending = (index + timebase / 2).ceil("1h")
    out = pd.DataFrame(index=index)
    for col in era5.columns:
        values = era5[col].to_numpy(dtype=float)
        if col in HOUR_ENDING_FIELDS:
            out[col] = era5[col].reindex(hour_ending).to_numpy(dtype=float)
        elif col.startswith(_DIRECTION_PREFIX):
            speed = era5[_SPEED_PREFIX + col.removeprefix(_DIRECTION_PREFIX)].to_numpy(dtype=float)
            out[col] = _interpolate_direction(values, speed=speed, hours=hours, centres=centres, inside=inside)
        else:
            out[col] = np.where(inside, np.interp(centres, hours, values), np.nan)
    return _with_aliases(out)


def _step(index: pd.Index) -> pd.Timedelta:
    return pd.Timedelta(pd.Series(index).diff().median())


def _with_aliases(aligned: pd.DataFrame) -> pd.DataFrame:
    """Return ``aligned`` with :data:`ERA5_WS` / :data:`ERA5_WD` added where the hub-height columns exist."""
    out = aligned.copy()
    if ERA5_WS_RAW in out.columns:
        out[ERA5_WS] = out[ERA5_WS_RAW]
    if ERA5_WD_RAW in out.columns:
        out[ERA5_WD] = out[ERA5_WD_RAW]
    return out


def _validated(era5_hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Return the ERA5 record with its all-NaN leading and trailing hours trimmed, or raise."""
    index = era5_hourly_df.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or _zone_name(str(index.tz)) != "UTC":
        msg = "the reanalysis index must be timezone-aware UTC"
        raise ValueError(msg)
    unknown = [
        c
        for c in era5_hourly_df.columns
        if c not in INSTANTANEOUS_FIELDS
        and c not in HOUR_ENDING_FIELDS
        and not c.startswith((_SPEED_PREFIX, _DIRECTION_PREFIX))
    ]
    if unknown:
        msg = f"reanalysis columns {unknown} are not known ERA5 fields, so cannot be aligned"
        raise ValueError(msg)
    for col in era5_hourly_df.columns:
        if col.startswith(_DIRECTION_PREFIX):
            speed = _SPEED_PREFIX + col.removeprefix(_DIRECTION_PREFIX)
            if speed not in era5_hourly_df.columns:
                msg = f"reanalysis column {col} needs {speed} to interpolate through the wind vector"
                raise ValueError(msg)
    populated = era5_hourly_df.notna().any(axis=1).to_numpy()
    if not populated.any():
        msg = "the reanalysis record holds no values"
        raise ValueError(msg)
    first, last = np.flatnonzero(populated)[[0, -1]]
    era5 = era5_hourly_df.iloc[first : last + 1].sort_index()
    steps = era5.index.to_series().diff().dropna()
    if (steps != _HOUR).any():
        bad = steps[steps != _HOUR]
        msg = f"the reanalysis record has a gap: the hour after {bad.index[0] - bad.iloc[0]} is missing or repeated"
        raise ValueError(msg)
    holes = era5.columns[era5.isna().any()].tolist()
    if holes:
        msg = f"the reanalysis record has a gap: NaN inside the record in {holes}"
        raise ValueError(msg)
    return era5


def _interpolate_direction(
    direction: np.ndarray, *, speed: np.ndarray, hours: np.ndarray, centres: np.ndarray, inside: np.ndarray
) -> np.ndarray:
    """Interpolate a direction in degrees through its wind vector, held at the earlier hour where the vector is zero."""
    rad = np.deg2rad(direction)
    u = np.interp(centres, hours, speed * np.sin(rad))
    v = np.interp(centres, hours, speed * np.cos(rad))
    out = np.mod(np.rad2deg(np.arctan2(u, v)), 360.0)
    calm = (u == 0) & (v == 0)
    if calm.any():
        earlier = np.clip(np.searchsorted(hours, centres[calm], side="right") - 1, 0, len(hours) - 1)
        out[calm] = direction[earlier]
    return np.where(inside, out, np.nan)


def uncovered_spans(aligned: pd.DataFrame, *, index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the ``(first, last)`` SCADA timestamps of each run of periods ERA5 does not cover."""
    covered = aligned.reindex(index).notna().any(axis=1).to_numpy()
    spans = []
    start = None
    for i, ok in enumerate(covered):
        if not ok and start is None:
            start = i
        elif ok and start is not None:
            spans.append((index[start], index[i - 1]))
            start = None
    if start is not None:
        spans.append((index[start], index[-1]))
    return spans


@dataclass(frozen=True)
class ShiftCheck:
    """The time-shift check of ERA5 against the site wind speed.

    :param best_shift: the shift of ERA5 that best correlates with the site; positive means the site lags ERA5
    :param best_corr: the correlation at ``best_shift``
    :param zero_corr: the correlation with no shift
    :param sweep: the correlation at every shift searched (columns ``shift``, ``corr``)
    """

    best_shift: pd.Timedelta
    best_corr: float
    zero_corr: float
    sweep: pd.DataFrame


def check_shift(
    *, era5_ws: pd.Series, site_ws: pd.Series, timebase: pd.Timedelta, search: pd.Timedelta = SHIFT_SEARCH
) -> ShiftCheck:
    """Search the shift of ``era5_ws`` that best correlates with ``site_ws``, both on one index.

    Logs a warning beyond :data:`SHIFT_WARN`.

    :raises ValueError: at or beyond :data:`SHIFT_RAISE`, which points at a mis-declared time zone or convention
    """
    a = era5_ws.to_numpy(dtype=float)
    b = site_ws.reindex(era5_ws.index).to_numpy(dtype=float)
    max_rows = min(round(search / timebase), len(a) - _MIN_OVERLAP)
    min_overlap = max(_MIN_OVERLAP, len(a) // 2)
    shifts = np.arange(-max_rows, max_rows + 1)
    corrs = np.array([_shift_corr(a, b, shift=int(s), min_overlap=min_overlap) for s in shifts])
    sweep = pd.DataFrame({"shift": shifts * timebase, "corr": corrs})
    zero_corr = float(corrs[shifts == 0][0]) if (shifts == 0).any() else math.nan
    if np.isnan(corrs).all():
        logger.warning("The reanalysis time-shift check could not run: too few records with a site wind speed")
        return ShiftCheck(best_shift=pd.Timedelta(0), best_corr=math.nan, zero_corr=zero_corr, sweep=sweep)
    best = int(np.nanargmax(corrs))
    result = ShiftCheck(
        best_shift=shifts[best] * timebase, best_corr=float(corrs[best]), zero_corr=zero_corr, sweep=sweep
    )
    size = abs(result.best_shift)
    if size >= SHIFT_RAISE:
        msg = (
            f"the reanalysis correlates best with the site wind speed at a shift of {minutes(result.best_shift)}, "
            f"at or beyond {minutes(SHIFT_RAISE)}. Check the declared SCADA time zone and timestamp convention."
        )
        raise ValueError(msg)
    if size > SHIFT_WARN:
        logger.warning(
            "The reanalysis correlates best with the site wind speed at a shift of %s, beyond %s. "
            "Check the declared SCADA time zone and timestamp convention.",
            minutes(result.best_shift),
            minutes(SHIFT_WARN),
        )
    return result


def minutes(shift: pd.Timedelta) -> str:
    """Return ``shift`` as signed whole minutes, eg ``-20 min``."""
    return f"{round(shift / pd.Timedelta(minutes=1)):+d} min"


def _shift_corr(a: np.ndarray, b: np.ndarray, *, shift: int, min_overlap: int) -> float:
    """Pearson correlation of ``a`` shifted forward by ``shift`` rows against ``b``, over finite pairs."""
    if shift >= 0:
        x, y = a[: len(a) - shift], b[shift:]
    else:
        x, y = a[-shift:], b[: len(b) + shift]
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < min_overlap:
        return math.nan
    x, y = x[finite], y[finite]
    if x.std() == 0 or y.std() == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


@dataclass(frozen=True)
class ReanalysisResult:
    """The output of step 3.

    :param aligned: ERA5 on the SCADA index (see :func:`interpolate_era5`)
    :param check: the time-shift check
    :param uncovered: the runs of SCADA periods ERA5 does not cover
    :param site_ws: the site wind speed the check was made against
    """

    aligned: pd.DataFrame
    check: ShiftCheck
    uncovered: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)
    site_ws: pd.Series | None = None

    @property
    def best_shift(self) -> pd.Timedelta:
        """The best shift of the time-shift check."""
        return self.check.best_shift

    @property
    def best_corr(self) -> float:
        """The correlation at the best shift."""
        return self.check.best_corr


def prepare_reanalysis(
    era5_hourly_df: pd.DataFrame,
    *,
    scada_df: pd.DataFrame,
    columns: ColumnSchema,
    unchanged: Sequence[str],
    timebase: pd.Timedelta,
) -> ReanalysisResult:
    """Align ERA5 to the SCADA index, report what it does not cover, and check the time shift.

    The site wind speed is the mean nacelle wind speed of the ``unchanged`` turbines, over the records
    valid for northing when step 2's labels are present.

    :raises ValueError: if the ERA5 record has a gap, or the best shift is at or beyond :data:`SHIFT_RAISE`
    """
    index = pd.DatetimeIndex(scada_df.index.unique()).sort_values()
    aligned = interpolate_era5(era5_hourly_df, index=index, timebase=timebase)
    uncovered = uncovered_spans(aligned, index=index)
    if uncovered:
        logger.warning(
            "The reanalysis does not cover the whole SCADA record: %d span(s) uncovered, %s",
            len(uncovered),
            ", ".join(f"{a}..{b}" for a, b in uncovered),
        )
    site_ws = site_wind_speed(scada_df, columns=columns, unchanged=unchanged).reindex(index)
    check = check_shift(era5_ws=aligned[ERA5_WS_RAW], site_ws=site_ws, timebase=timebase)
    logger.info(
        "Reanalysis aligned to the SCADA periods; best shift %s (corr %.3f, %.3f unshifted)",
        minutes(check.best_shift),
        check.best_corr,
        check.zero_corr,
    )
    return ReanalysisResult(aligned=aligned, check=check, uncovered=uncovered, site_ws=site_ws)


def site_wind_speed(scada_df: pd.DataFrame, *, columns: ColumnSchema, unchanged: Sequence[str]) -> pd.Series:
    """Return the mean nacelle wind speed of the ``unchanged`` turbines per timestamp."""
    rows = scada_df[scada_df[columns.turbine].isin(list(unchanged))]
    if VALID_NORTHING_COL in rows.columns:
        rows = rows[rows[VALID_NORTHING_COL].astype(bool)]
    ws = rows[columns.wind_speed].astype(float)
    return ws.groupby(level=0).mean()
