"""A deliberately naive energy-ratio uplift method behind the harness ``Method`` seam.

``NaiveRatioMethod`` is an honest, independent baseline that shares no code with v0 and has
no wind_up dependency. For a set of rows it forms the test-to-reference ratio

    rho(rows) = sum(test power) / sum(reference total power)

over *used* timestamps (complete-case: the test turbine and every reference finite), and
estimates uplift as the ratio-of-ratios ``rho(treated) / rho(baseline) - 1``. Its only error
source on the synthetic data is genuine pre/post covariate shift -- it applies no
conditioning by design -- so it is the "what if you don't condition at all" leaderboard floor.

It uses **only** the active-power-mean column (test and references); it never reads wind speed,
direction, rpm or any other SCADA tag, which keeps it honest under design-note section 3 (the
test turbine's own wind speed is post-treatment and is never touched).

Each run writes a per-run folder ``naive_<test>_<upgradestart>_<lastdate>/`` (v0-style naming)
under ``out_dir`` (a temp dir by default), holding a per-segment data-stats CSV, a headline
results CSV, and -- when ``save_plots`` -- three diagnostic plots (a test-vs-reference scatter,
a per-segment daily-ratio timeseries, and a per-segment used-data-coverage timeseries). The rich stats let a human
confirm the right data was received and interpreted: the headline uplift is re-derivable from
the stats CSV as ``rho = used_test_mwh / used_ref_total_mwh`` per segment.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.synthetic import ToggleSchedule, treated_mask
from wind_up.constants import DataColumns

if TYPE_CHECKING:
    import numpy.typing as npt

_SEGMENTS = ("all", "baseline", "upgraded")
_MIN_POINTS_FOR_TIMEBASE = 2


def _infer_timebase(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Infer the analysis timebase as the median spacing of the sorted unique timestamps."""
    unique = pd.DatetimeIndex(pd.unique(index)).sort_values()
    if len(unique) < _MIN_POINTS_FOR_TIMEBASE:
        return pd.Timedelta(minutes=10)
    return pd.Timedelta(np.median(np.diff(unique.to_numpy())))


def _wide_power(scada_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long SCADA to a timestamp x turbine table of active power (NaN where missing)."""
    tmp = scada_df[[DataColumns.turbine_name, DataColumns.active_power_mean]].copy()
    tmp["_ts"] = scada_df.index
    return tmp.pivot_table(
        index="_ts",
        columns=DataColumns.turbine_name,
        values=DataColumns.active_power_mean,
        aggfunc="first",
    )


def _upgrade_start(upgrade_timing: pd.Timestamp | ToggleSchedule, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the upgrade-start timestamp (changeover for prepost; toggle origin for toggle)."""
    if isinstance(upgrade_timing, ToggleSchedule):
        return upgrade_timing.start if upgrade_timing.start is not None else index.min()
    return pd.Timestamp(upgrade_timing)


@dataclass
class NaiveRatioMethod:
    """Pluggable naive energy-ratio baseline (prepost and toggle).

    :param name: method name shown in the leaderboard
    :param out_dir: where per-run folders are written; a temp dir when ``None``
    :param save_plots: also write the three diagnostic plots under ``<run>/plots``
    :param timebase: analysis timebase; inferred from the data when ``None``
    """

    name: str = "naive_ratio"
    out_dir: Path | None = None
    save_plots: bool = False
    timebase: pd.Timedelta | None = None

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Estimate the test turbine's P50 uplift for one campaign and write diagnostics."""
        wide = _wide_power(mi.scada_df)
        test = mi.test_wtg
        refs = [c for c in wide.columns if c != test]
        if not refs:
            msg = (
                f"no reference turbines available for test_wtg {test!r}: scada_df contains only "
                f"{list(wide.columns)}. The naive ratio method needs at least one reference turbine."
            )
            raise ValueError(msg)

        timebase = self.timebase if self.timebase is not None else _infer_timebase(mi.scada_df.index)
        ts_treated = np.asarray(treated_mask(wide.index, mi.upgrade_timing))
        test_pw = wide[test].to_numpy(dtype=float)
        ref_total = wide[refs].sum(axis=1).to_numpy(dtype=float)
        used = wide[[test, *refs]].notna().all(axis=1).to_numpy()

        rho_base = _rho(test_pw, ref_total, used & ~ts_treated)
        rho_up = _rho(test_pw, ref_total, used & ts_treated)
        recoverable = np.isfinite(rho_base) and rho_base != 0 and np.isfinite(rho_up)
        uplift = rho_up / rho_base - 1.0 if recoverable else np.nan

        stats = _segment_stats(mi, wide=wide, used=used, ts_treated=ts_treated, refs=refs, timebase=timebase)
        self._write_outputs(
            mi, wide=wide, stats=stats, rho_base=rho_base, rho_up=rho_up, uplift=uplift, n_refs=len(refs)
        )
        return MethodOutput(p50_overall=float(uplift))

    def _write_outputs(
        self,
        mi: MethodInput,
        *,
        wide: pd.DataFrame,
        stats: pd.DataFrame,
        rho_base: float,
        rho_up: float,
        uplift: float,
        n_refs: int,
    ) -> None:
        """Write the data-stats CSV, the headline results CSV and (optionally) the plots."""
        upgrade_start = _upgrade_start(mi.upgrade_timing, wide.index)
        last_dt = wide.index.max()
        run_name = f"naive_{mi.test_wtg}_{upgrade_start:%Y%m%d}_{last_dt:%Y%m%d}"
        out_root = Path(self.out_dir) if self.out_dir is not None else Path(tempfile.mkdtemp(prefix="naive_"))
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S_%f")

        stats.to_csv(run_dir / f"{run_name}_data_stats_{ts}.csv", index=False)

        mode = "toggle" if isinstance(mi.upgrade_timing, ToggleSchedule) else "prepost"
        used_base = int(stats.loc[stats["segment"] == "baseline", "n_used_timestamps"].iloc[0])
        used_up = int(stats.loc[stats["segment"] == "upgraded", "n_used_timestamps"].iloc[0])
        results = pd.DataFrame(
            [
                {
                    "test_wtg": mi.test_wtg,
                    "mode": mode,
                    "n_turbines": wide.shape[1],
                    "n_refs": n_refs,
                    "ratio_baseline": rho_base,
                    "ratio_upgraded": rho_up,
                    "uplift_frc": uplift,
                    "n_used_timestamps_baseline": used_base,
                    "n_used_timestamps_upgraded": used_up,
                    "time_calculated": pd.Timestamp.utcnow(),
                }
            ]
        )
        results.to_csv(run_dir / f"{run_name}_results_{ts}.csv", index=False)

        if self.save_plots:
            _save_plots(run_dir / "plots", wide=wide, mi=mi, test=mi.test_wtg)


def _rho(test_pw: npt.NDArray[np.float64], ref_total: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_]) -> float:
    """Test-to-reference ratio over ``mask``: sum(test) / sum(ref_total). NaN if degenerate."""
    if not mask.any():
        return float("nan")
    denom = ref_total[mask].sum()
    if denom == 0:
        return float("nan")
    return float(test_pw[mask].sum() / denom)


def _segment_stats(
    mi: MethodInput,
    *,
    wide: pd.DataFrame,
    used: npt.NDArray[np.bool_],
    ts_treated: npt.NDArray[np.bool_],
    refs: list[str],
    timebase: pd.Timedelta,
) -> pd.DataFrame:
    """Build the per-segment (all/baseline/upgraded) diagnostics table."""
    test = mi.test_wtg
    test_pw = wide[test].to_numpy(dtype=float)
    ref_total = wide[refs].sum(axis=1).to_numpy(dtype=float)
    n_turbines = wide.shape[1]
    timebase_hours = timebase / pd.Timedelta(hours=1)

    row_treated = np.asarray(treated_mask(mi.scada_df.index, mi.upgrade_timing))
    row_power = mi.scada_df[DataColumns.active_power_mean].to_numpy(dtype=float)

    ts_masks = {"all": np.ones(len(wide), dtype=bool), "baseline": ~ts_treated, "upgraded": ts_treated}
    row_masks = {"all": np.ones(len(mi.scada_df), dtype=bool), "baseline": ~row_treated, "upgraded": row_treated}

    rows = []
    for segment in _SEGMENTS:
        ts_mask = ts_masks[segment]
        row_mask = row_masks[segment]
        seg_ts = wide.index[ts_mask]
        seg_used = used & ts_mask
        n_used = int(seg_used.sum())

        if len(seg_ts):
            first, last = seg_ts.min(), seg_ts.max()
            expected_ts = round((last - first) / timebase) + 1
        else:
            first = last = pd.NaT
            expected_ts = 0
        expected_rows = n_turbines * expected_ts

        n_rows = int(row_mask.sum())
        n_power_finite = int(np.isfinite(row_power[row_mask]).sum())

        used_test = test_pw[seg_used]
        used_ref = ref_total[seg_used]
        rows.append(
            {
                "segment": segment,
                "first_timestamp": first,
                "last_timestamp": last,
                "n_turbines": n_turbines,
                "expected_timestamps": expected_ts,
                "n_rows": n_rows,
                "expected_rows": expected_rows,
                "rows_data_coverage": n_rows / expected_rows if expected_rows else np.nan,
                "n_power_finite_rows": n_power_finite,
                "power_finite_coverage": n_power_finite / expected_rows if expected_rows else np.nan,
                "n_used_timestamps": n_used,
                "used_data_coverage": n_used / expected_ts if expected_ts else np.nan,
                "used_test_mean_power_kw": float(used_test.mean()) if n_used else np.nan,
                "used_test_mwh": float(used_test.sum()) * timebase_hours / 1000.0 if n_used else np.nan,
                "used_ref_total_mean_power_kw": float(used_ref.mean()) if n_used else np.nan,
                "used_ref_total_mwh": float(used_ref.sum()) * timebase_hours / 1000.0 if n_used else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _daily_segment_ratio(
    index: pd.DatetimeIndex,
    test_pw: npt.NDArray[np.float64],
    ref_total: npt.NDArray[np.float64],
    seg_mask: npt.NDArray[np.bool_],
) -> pd.Series:
    """Daily sum-based test/reference ratio (Sum test / Sum ref) over ``seg_mask`` rows; NaN on empty days.

    This matches the method's own ``rho`` definition (a ratio of sums, not a mean of per-timestamp
    ratios), so the daily series fluctuates around the scalar ``rho`` the estimate uses instead of
    blowing up on low-wind timestamps.
    """
    test = pd.Series(np.where(seg_mask, test_pw, np.nan), index=index)
    ref = pd.Series(np.where(seg_mask, ref_total, np.nan), index=index)
    return test.resample("1D").sum(min_count=1) / ref.resample("1D").sum(min_count=1)


def _expected_per_day(index: pd.DatetimeIndex, timebase: pd.Timedelta) -> pd.Series:
    """Daily count of timestamps the analysis timebase grid expects between the data's first and last."""
    grid = pd.date_range(index.min(), index.max(), freq=timebase)
    return pd.Series(1.0, index=grid).resample("1D").sum()


def _daily_segment_coverage(
    index: pd.DatetimeIndex,
    used: npt.NDArray[np.bool_],
    seg_mask: npt.NDArray[np.bool_],
    expected_per_day: pd.Series,
) -> pd.Series:
    """Daily used-data coverage in [0, 1], as a fraction of the day's expected timestamps.

    Numerator: complete-case timestamps (test and every reference finite) assigned to this segment
    each day. Denominator: the day's expected timestamp count on the analysis timebase grid, which
    is shared across segments. So the two segments' coverages sum to the day's overall complete-case
    coverage, and under toggle each segment is capped near the duty cycle (~50%) of slots it can ever
    occupy. NaN on days the grid does not reach.
    """
    used_seg = pd.Series((used & seg_mask).astype(float), index=index)
    daily_used = used_seg.resample("1D").sum()
    return daily_used / expected_per_day.reindex(daily_used.index)


def _save_plots(plots_dir: Path, *, wide: pd.DataFrame, mi: MethodInput, test: str) -> None:
    """Write the scatter, ratio-timeseries and used-coverage-timeseries diagnostic plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    refs = [c for c in wide.columns if c != test]
    ts_treated = np.asarray(treated_mask(wide.index, mi.upgrade_timing))
    test_pw = wide[test].to_numpy(dtype=float)
    ref_total = wide[refs].sum(axis=1).to_numpy(dtype=float)
    used = wide[[test, *refs]].notna().all(axis=1).to_numpy()
    upgrade_start = _upgrade_start(mi.upgrade_timing, wide.index)
    segments = (("baseline", used & ~ts_treated, "C0"), ("upgraded", used & ts_treated, "C1"))

    # 1) scatter of test vs reference-total power, baseline/upgraded coloured, with rho slopes.
    fig, ax = plt.subplots(figsize=(7, 7))
    for label, seg, color in segments:
        ax.scatter(ref_total[seg], test_pw[seg], s=8, alpha=0.4, color=color, label=label)
        rho = _rho(test_pw, ref_total, seg)
        if np.isfinite(rho) and seg.any():
            x_max = float(np.nanmax(ref_total[seg]))
            ax.plot([0, x_max], [0, rho * x_max], color=color, linewidth=1.5)
    ax.set_xlabel("reference total power [kW]")
    ax.set_ylabel(f"{test} power [kW]")
    ax.set_title(f"{test}: test vs reference-total power")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{test}_scatter.png", dpi=150)
    plt.close(fig)

    # 2) daily sum-based test/ref ratio, one series per segment, with each segment's scalar rho overlaid.
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, seg, color in segments:
        daily = _daily_segment_ratio(wide.index, test_pw, ref_total, seg)
        ax.plot(daily.index.to_numpy(), daily.to_numpy(), marker=".", linewidth=0.8, color=color, label=label)
        rho = _rho(test_pw, ref_total, seg)
        span = wide.index[seg]
        if np.isfinite(rho) and len(span):
            ax.hlines(rho, span.min(), span.max(), color=color, linestyle="--", linewidth=1.5)
    ax.axvline(upgrade_start, color="k", linestyle="--", label="upgrade start")
    ax.set_xlabel("date")
    ax.set_ylabel("test / reference-total ratio")
    ax.set_title(f"{test}: daily test/reference ratio (dashed = rho used by estimate)")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{test}_ratio_timeseries.png", dpi=150)
    plt.close(fig)

    # 3) daily used-data coverage as a fraction of the day's expected timestamps, one series per
    # segment, so each segment is seen to receive its share (under toggle, ~50% each post-upgrade).
    expected_per_day = _expected_per_day(wide.index, _infer_timebase(wide.index))
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, _seg, color in segments:
        seg_mask = ~ts_treated if label == "baseline" else ts_treated
        daily = _daily_segment_coverage(wide.index, used, seg_mask, expected_per_day)
        ax.plot(daily.index.to_numpy(), daily.to_numpy(), marker=".", linewidth=0.8, color=color, label=label)
    ax.axvline(upgrade_start, color="k", linestyle="--", label="upgrade start")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("date")
    ax.set_ylabel("used-data coverage")
    ax.set_title(f"{test}: daily used-data coverage (complete-case, % of expected timestamps)")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"{test}_coverage_timeseries.png", dpi=150)
    plt.close(fig)
