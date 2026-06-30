"""ERA5 alignment for the R-learner.

The sync now lives in :mod:`benchmarking.baselines.era5_sync` because more than one method uses
it; this module re-exports it for backwards compatibility.
"""

from __future__ import annotations

from benchmarking.baselines.era5_sync import (
    ERA5_WD,
    ERA5_WS,
    Era5SyncResult,
    find_best_lag,
    sync_era5,
    upsample_era5_to_timebase,
)

__all__ = [
    "ERA5_WD",
    "ERA5_WS",
    "Era5SyncResult",
    "find_best_lag",
    "sync_era5",
    "upsample_era5_to_timebase",
]
