"""Shared test-turbine normal-operation filtering for the benchmarking methods.

The outcome is the test turbine's power, so abnormal operation unrelated to the upgrade —
downtime, curtailment, frozen/stuck sensors — would otherwise be attributed to the upgrade (a
downward bias, worst when it clusters in the upgrade period). Every method must select the
normally-operating test-turbine rows the same way, so this filter lives in one shared place
(the R-learner and the naive ratio both use it; it has no ``wind_up`` dependency).

Three checks:

* **finite power** — rows with NaN active power are downtime / missing energy, always dropped.
* **downtime / availability** — drop rows where an availability counter shows the turbine was not
  ready to operate for the full period. This is **required** by the methods (a missing availability
  column is a configuration error, not a silent no-op).
* **stuck data** — drop rows where every signal is unchanged from the previous record (a frozen
  data stream), exempting genuine very-low-wind calms.

The central rule is **filter on cause, not effect**: selection uses operational signals and finite
power, never "power lower than expected" — that would drop genuine low-uplift records and bias the
estimate. This is row selection, not a feature rule, so using the test turbine's own operational
signals here does not violate the upgrade-invariant feature rule. References are deliberately not
filtered (the R-learner learns their operating modes; the naive ratio keeps complete-case refs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Below this wind speed a flat/constant signal is a genuine calm, not a stuck sensor.
_VERY_LOW_WIND = 1.5


@dataclass
class NormalOperationFilter:
    """Selects normally-operating test-turbine timestamps (cause, not effect).

    :param active_power_col: the test turbine's active-power column (rows with NaN power are
        always dropped — that is downtime/missing energy)
    :param wind_speed_col: the test turbine's wind-speed column, used only to exempt very-low-wind
        calms from the stuck filter (``None`` disables that exemption)
    :param availability_col: an operational "ready to operate" counter (e.g. seconds in the
        period); ``None`` disables the downtime filter
    :param full_period_seconds: the counter value that means fully available; defaults to the
        timebase length in seconds when ``None``
    :param apply_stuck_filter: drop frozen/stuck rows (all signals unchanged vs the previous row)
    """

    active_power_col: str
    wind_speed_col: str | None = None
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
        """Return True where every numeric signal is unchanged from the previous row (not low wind)."""
        numeric = rows.select_dtypes(include="number")
        diffs = numeric.ffill().fillna(0).diff()
        frozen = (diffs == 0).all(axis=1)
        frozen.iloc[0] = False  # the first row has no predecessor to repeat
        if self.wind_speed_col is not None:
            calm = rows[self.wind_speed_col] < _VERY_LOW_WIND
            frozen &= ~calm
        return frozen

    def _available(self, rows: pd.DataFrame, *, timebase: pd.Timedelta) -> pd.Series:
        """Return True where the availability counter shows a full period (NaN -> not available)."""
        full = self.full_period_seconds if self.full_period_seconds is not None else timebase.total_seconds()
        counter = rows[self.availability_col]
        return (counter >= full) & counter.notna()
