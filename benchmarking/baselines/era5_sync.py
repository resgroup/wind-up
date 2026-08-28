"""Align hourly ERA5 reanalysis to the 10-min SCADA grid (shared across benchmarking methods).

ERA5 arrives hourly; SCADA is 10-min. Two steps, adapted from the logic in
``wind_up.reanalysis_data`` (``_reanalysis_upsample`` / ``_find_best_shift_and_corr``,
which are private there) but kept local so the methods stay v0-independent:

1. :func:`upsample_era5_to_timebase` resamples ERA5 onto the analysis timebase and
   forward-fills within each hour. **Every raw column is passed through under its original
   Open-Meteo name** (no renaming); for back-compat with the R-learner and the shared
   diagnostics, neutral ``era5_ws`` / ``era5_wd`` *aliases* of wind speed / direction are
   added alongside the raw columns.
2. :func:`find_best_lag` sweeps the integer row-shift that maximises the correlation
   between ERA5 wind speed and a reference (wind-farm) wind speed, recovering the lag
   between the reanalysis and the site.

:func:`sync_era5` combines both and returns the aligned ERA5 (all columns), plus the chosen
lag and the correlation-vs-lag sweep for diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

_MIN_OVERLAP = 3

# Neutral, source-agnostic aliases (no wind_up / v0 vocabulary) added alongside the raw columns
# so the R-learner's ``era5_features`` and the shared diagnostics keep a stable ws/wd handle.
ERA5_WS = "era5_ws"
ERA5_WD = "era5_wd"

# Open-Meteo raw column names ERA5 ships with (see ``wind_up.era5``).
_RAW_WS = "wind_speed_100m"
_RAW_WD = "wind_direction_100m"

_DEFAULT_MAX_LAG = pd.Timedelta(hours=24)


@dataclass
class Era5SyncResult:
    """ERA5 aligned to a target index, with the recovered lag and the sweep.

    :param aligned: frame indexed by the target index with **all** raw ERA5 columns (original
        Open-Meteo names) plus the :data:`ERA5_WS` / :data:`ERA5_WD` aliases (lag applied)
    :param best_lag_rows: the integer row shift applied to ERA5 (positive = ERA5 shifted
        forward to align with a lagging site signal)
    :param best_corr: the wind-speed correlation at ``best_lag_rows``
    :param sweep: the correlation-vs-lag table (columns ``shift_rows``, ``corr``)
    """

    aligned: pd.DataFrame
    best_lag_rows: int
    best_corr: float
    sweep: pd.DataFrame


def upsample_era5_to_timebase(era5_hourly_df: pd.DataFrame, *, timebase: pd.Timedelta) -> pd.DataFrame:
    """Resample hourly ERA5 onto ``timebase`` and forward-fill within each source step.

    Mirrors ``wind_up.reanalysis_data._reanalysis_upsample``: resample with ``last``, extend
    the index so the final source step's trailing slots exist, then forward-fill up to one
    source step. **All** raw columns are kept under their original names; ``era5_ws`` /
    ``era5_wd`` aliases of wind speed / direction are added for back-compat.
    """
    source_step = pd.Timedelta(pd.Series(era5_hourly_df.index).diff().median())
    upsample_factor = round(source_step / timebase)
    resampled = era5_hourly_df.resample(timebase, label="left").last()
    if upsample_factor > 1:
        tail = pd.DataFrame(
            index=pd.date_range(
                start=resampled.index[-1] + timebase,
                periods=upsample_factor - 1,
                freq=timebase,
            )
        )
        resampled = pd.concat([resampled, tail])
        resampled = resampled.ffill(limit=upsample_factor - 1)
    missing = {_RAW_WS, _RAW_WD} - set(era5_hourly_df.columns)
    if missing:
        msg = f"ERA5 frame is missing expected columns {sorted(missing)}; have {list(era5_hourly_df.columns)}"
        raise ValueError(msg)
    out = resampled.copy()
    out[ERA5_WS] = resampled[_RAW_WS]
    out[ERA5_WD] = resampled[_RAW_WD]
    out.index.name = era5_hourly_df.index.name
    return out


def find_best_lag(
    *,
    reference_ws: pd.Series,
    era5_ws: pd.Series,
    timebase: pd.Timedelta,
    max_lag: pd.Timedelta = _DEFAULT_MAX_LAG,
) -> tuple[int, float, pd.DataFrame]:
    """Find the integer row shift of ERA5 wind speed that best correlates with ``reference_ws``.

    Both series must share the analysis-grid index. Sweeps shifts in ``±max_lag`` (in steps of
    ~10 min of rows, like wind_up) and returns the shift maximising ``corr(era5_ws.shift(s),
    reference_ws)`` — so a positive shift advances ERA5 to meet a site signal that lags it.
    """
    rows_per_hour = pd.Timedelta(hours=1) / timebase
    # cap the sweep so the most extreme shift still overlaps the data (avoids empty slices)
    max_rows = min(round(max_lag / timebase), len(era5_ws) - _MIN_OVERLAP)
    step = max(1, math.ceil(rows_per_hour / 6))
    shifts = list(range(-max_rows, max_rows + 1, step))
    # a shift is only meaningful if a substantial chunk of data still overlaps; otherwise a
    # handful of coincidentally-collinear points can score a spurious corr of 1.0.
    min_overlap = max(_MIN_OVERLAP, len(era5_ws) // 2)
    corrs = [_shift_corr(era5_ws=era5_ws, reference_ws=reference_ws, shift=s, min_overlap=min_overlap) for s in shifts]
    sweep = pd.DataFrame({"shift_rows": shifts, "corr": corrs})
    if sweep["corr"].isna().all():
        return 0, float("nan"), sweep
    best = sweep.loc[sweep["corr"].idxmax()]
    return int(best["shift_rows"]), float(best["corr"]), sweep


def _shift_corr(*, era5_ws: pd.Series, reference_ws: pd.Series, shift: int, min_overlap: int) -> float:
    """Pearson correlation of ``era5_ws.shift(shift)`` and ``reference_ws`` over finite pairs.

    Computed on the overlapping non-NaN pairs only, returning NaN below ``min_overlap`` points,
    so extreme shifts neither raise numpy warnings (tests treat warnings as errors) nor score a
    spurious perfect correlation off a handful of points.
    """
    shifted = era5_ws.shift(shift)
    pair = pd.concat([shifted, reference_ws], axis=1).dropna()
    if len(pair) < min_overlap:
        return float("nan")
    a = pair.iloc[:, 0].to_numpy()
    b = pair.iloc[:, 1].to_numpy()
    if a.std() == 0 or b.std() == 0:  # zero variance -> correlation undefined (and numpy warns)
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def sync_era5(
    era5_hourly_df: pd.DataFrame,
    *,
    target_index: pd.DatetimeIndex,
    reference_ws: pd.Series,
    timebase: pd.Timedelta | None = None,
    max_lag: pd.Timedelta = _DEFAULT_MAX_LAG,
) -> Era5SyncResult:
    """Upsample ERA5, find its lag vs ``reference_ws``, and align it to ``target_index``.

    :param era5_hourly_df: raw hourly ERA5 (Open-Meteo column names)
    :param target_index: the analysis grid to align ERA5 onto (the SCADA timestamps)
    :param reference_ws: a site wind speed on ``target_index`` (e.g. reference-turbine mean)
    :param timebase: analysis timebase; inferred from ``target_index`` spacing when ``None``
    """
    if timebase is None:
        timebase = pd.Timedelta(pd.Series(target_index).diff().median())
    upsampled = upsample_era5_to_timebase(era5_hourly_df, timebase=timebase).reindex(target_index)
    best_lag, best_corr, sweep = find_best_lag(
        reference_ws=reference_ws.reindex(target_index),
        era5_ws=upsampled[ERA5_WS],
        timebase=timebase,
        max_lag=max_lag,
    )
    aligned = upsampled.shift(best_lag)
    return Era5SyncResult(aligned=aligned, best_lag_rows=best_lag, best_corr=best_corr, sweep=sweep)
