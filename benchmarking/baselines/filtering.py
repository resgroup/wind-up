"""Shared test-turbine normal-operation filtering for the benchmarking methods.

The outcome is the test turbine's power, so abnormal operation unrelated to the upgrade —
downtime, curtailment, frozen/stuck sensors — would otherwise be attributed to the upgrade (a
downward bias, worst when it clusters in the upgrade period). Every method must select the
normally-operating test-turbine rows the same way, so this filter lives in one shared place
(the benchmarking methods all use it; it has no ``wind_up`` dependency).

Three checks:

* **finite power** — rows with NaN active power are downtime / missing energy, always dropped.
* **downtime / availability** — drop rows where an availability counter shows the turbine was not
  ready to operate for the full period. This is **required** by the methods (a missing availability
  column is a configuration error, not a silent no-op).
* **stuck data** — drop rows where every measured schema signal is unchanged from the previous
  record (a frozen data stream), exempting calms; the rule is
  :func:`~benchmarking.harness.operating_state.stuck_records`.

The central rule is **filter on cause, not effect**: selection uses operational signals and finite
power, never "power lower than expected" — that would drop genuine low-uplift records and bias the
estimate. This is row selection, not a feature rule, so using the test turbine's own operational
signals here does not violate the upgrade-invariant feature rule. References are deliberately not
filtered (the naive ratio keeps complete-case refs; the power model learns their operating modes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmarking.harness.operating_state import stuck_records

if TYPE_CHECKING:
    import pandas as pd

    from benchmarking.synthetic import ColumnSchema


@dataclass
class NormalOperationFilter:
    """Selects normally-operating test-turbine timestamps (cause, not effect).

    :param active_power_col: the test turbine's active-power column (rows with NaN power are
        always dropped — that is downtime/missing energy)
    :param columns: the schema whose measured signals the stuck filter reads; required when
        ``apply_stuck_filter`` is set
    :param availability_col: an operational "ready to operate" counter (e.g. seconds in the
        period); ``None`` disables the downtime filter
    :param full_period_seconds: the counter value that means fully available; defaults to the
        timebase length in seconds when ``None``
    :param apply_stuck_filter: drop frozen/stuck rows (all signals unchanged vs the previous row)
    """

    active_power_col: str
    columns: ColumnSchema | None = None
    availability_col: str | None = None
    full_period_seconds: float | None = None
    apply_stuck_filter: bool = True

    def keep_mask(self, test_rows: pd.DataFrame, *, timebase: pd.Timedelta) -> pd.Series:
        """Boolean Series (True = keep) of normally-operating test-turbine rows, index-aligned."""
        rows = test_rows.sort_index()
        keep = rows[self.active_power_col].notna()
        if self.apply_stuck_filter:
            keep &= ~self._stuck(rows)
        if self.availability_col is not None:
            keep &= self._available(rows, timebase=timebase)
        return keep.astype(bool)

    def _stuck(self, rows: pd.DataFrame) -> pd.Series:
        """Return True where every measured signal is unchanged from the previous row (not a calm)."""
        if self.columns is None:
            msg = "the stuck filter needs columns, the schema whose signals it reads"
            raise ValueError(msg)
        return stuck_records(rows, columns=self.columns)

    def _available(self, rows: pd.DataFrame, *, timebase: pd.Timedelta) -> pd.Series:
        """Return True where the availability counter shows a full period (NaN -> not available)."""
        full = self.full_period_seconds if self.full_period_seconds is not None else timebase.total_seconds()
        counter = rows[self.availability_col]
        return (counter >= full) & counter.notna()
