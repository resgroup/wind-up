"""The shared inputs the cross-method diagnostics draw from.

A method adapts its internals to a :class:`DiagnosticContext` (the test turbine, the long SCADA
slice, per-timestamp treatment/used masks, the timebase, optional aligned ERA5) and the shared
plotting functions take it from there. This keeps the plot code method-agnostic: it knows
nothing about R-learner folds or the naive ratio, only the common picture of "which turbine,
which rows, used or not, baseline or upgraded".

The few computed views the plots need (the unique index, the test turbine's rows aligned to it,
a wide per-turbine pivot, a reference-mean signal) live here so the plotting modules stay lean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from benchmarking.synthetic import ColumnSchema

# Column names the optional ``era5_df`` is expected to carry (the R-learner's ERA5 sync output).
ERA5_WS_COL = "era5_ws"
ERA5_WD_COL = "era5_wd"

_MIN_POINTS_FOR_TIMEBASE = 2


def infer_timebase(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Infer the analysis timebase as the median spacing of the sorted unique timestamps."""
    unique = pd.DatetimeIndex(pd.unique(index)).sort_values()
    if len(unique) < _MIN_POINTS_FOR_TIMEBASE:
        return pd.Timedelta(minutes=10)
    return pd.Timedelta(np.median(np.diff(unique.to_numpy())))


@dataclass
class DiagnosticContext:
    """Method-agnostic inputs for the shared per-run diagnostics.

    :param run_dir: the method's per-run output folder (plots land in ``run_dir/"plots"``)
    :param test_wtg: the test turbine name
    :param turbine_col: the long frame's turbine-identifier column
    :param columns: the source-native column schema (diagnostic roles may be ``None``)
    :param scada_df: the ``MethodInput`` SCADA slice (long format, all subset turbines)
    :param treated_ts: treatment flag (0/1) per unique timestamp, in sorted-index order
    :param used_ts: "used by the method" flag per unique timestamp (the test turbine's kept rows)
    :param timebase: the analysis timebase
    :param mode: ``"prepost"`` or ``"toggle"``
    :param era5_df: optional ERA5 aligned to the unique index (columns :data:`ERA5_WS_COL` /
        :data:`ERA5_WD_COL`)
    :param excluded_ts: optional per-timestamp ``ColumnSchema.exclude_row`` mask (``None`` for a
        method with no exclusion concept). Separate from ``used_ts``, which also folds in downtime.
    """

    run_dir: Path
    test_wtg: str
    turbine_col: str
    columns: ColumnSchema
    scada_df: pd.DataFrame
    treated_ts: npt.NDArray[np.bool_]
    used_ts: npt.NDArray[np.bool_]
    timebase: pd.Timedelta
    mode: str
    era5_df: pd.DataFrame | None = None
    excluded_ts: npt.NDArray[np.bool_] | None = None

    @property
    def index(self) -> pd.DatetimeIndex:
        """The unique, sorted analysis timestamps (one entry per timebase slot)."""
        return pd.DatetimeIndex(pd.unique(self.scada_df.index)).sort_values()

    @property
    def plots_dir(self) -> Path:
        """Root folder the diagnostic plots are written under (stage subfolders live here)."""
        return self.run_dir / "plots"

    def stage_dir(self, stage: str) -> Path:
        """Return (and create) the plots subfolder for an analysis ``stage`` (see :mod:`stages`)."""
        path = self.plots_dir / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def baseline_ts(self) -> npt.NDArray[np.bool_]:
        """Per-timestamp mask of baseline (un-upgraded) slots."""
        return ~np.asarray(self.treated_ts, dtype=bool)

    @property
    def upgraded_ts(self) -> npt.NDArray[np.bool_]:
        """Per-timestamp mask of upgraded slots."""
        return np.asarray(self.treated_ts, dtype=bool)

    def excluded_mask(self) -> npt.NDArray[np.bool_] | None:
        """Caller-flagged exclusions as a bool mask; ``None`` when unset or empty (nothing to draw)."""
        if self.excluded_ts is None:
            return None
        mask = np.asarray(self.excluded_ts, dtype=bool)
        return mask if mask.any() else None

    def references(self) -> list[str]:
        """Sorted reference turbine names (every turbine present except the test turbine)."""
        return sorted(t for t in self.scada_df[self.turbine_col].unique() if t != self.test_wtg)

    def has_column(self, col: str | None) -> bool:
        """Return True if ``col`` is named (not ``None``) and present in the SCADA frame."""
        return col is not None and col in self.scada_df.columns

    def turbine_series(self, turbine: str, col: str | None) -> pd.Series:
        """Return a turbine's ``col`` aligned to the unique index (NaN where missing/absent)."""
        if col is None or col not in self.scada_df.columns:
            return pd.Series(np.nan, index=self.index)
        rows = self.scada_df[self.scada_df[self.turbine_col] == turbine]
        series = pd.Series(rows[col].to_numpy(dtype=float), index=pd.DatetimeIndex(rows.index))
        return series[~series.index.duplicated()].reindex(self.index)

    def test_series(self, col: str | None) -> pd.Series:
        """Return the test turbine's ``col`` aligned to the unique index."""
        return self.turbine_series(self.test_wtg, col)

    def reference_mean(self, col: str | None) -> pd.Series:
        """Mean of ``col`` across reference turbines, aligned to the unique index."""
        refs = self.references()
        if not refs:
            return pd.Series(np.nan, index=self.index)
        frame = pd.concat([self.turbine_series(r, col) for r in refs], axis=1)
        return frame.mean(axis=1)

    def wide(self, col: str) -> pd.DataFrame:
        """Timestamp x turbine pivot of ``col`` (NaN where missing), on the unique index."""
        tmp = self.scada_df[[self.turbine_col, col]].copy()
        tmp["_ts"] = self.scada_df.index
        return tmp.pivot_table(index="_ts", columns=self.turbine_col, values=col, aggfunc="first").reindex(self.index)
