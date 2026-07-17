"""A toggle-only energy-ratio uplift method behind the harness ``Method`` seam.

``ToggleSpecialistMethod`` is a specialist for **toggle campaigns**: campaigns whose on/off
comparison is drawn entirely from the interleaved campaign blocks. It therefore accepts only
toggle inputs and raises on a prepost changeover.

**Used timestamps** require every turbine (test and references) to be available (an availability
counter at a full period) and have finite power — a down turbine on either side of the ratio would
otherwise bias it. The availability column is therefore **required**. Only the active-power column
enters the ``rho`` *computation*; the availability column is used solely for row selection (cause,
not effect), so the estimate still
never conditions on the test turbine's post-treatment wind speed (design-note §3). It speaks the
data source's own column names and has no wind_up dependency.

Every uplift — the headline and each power bin — comes with a non-optional 1-sigma uncertainty from
a circular block bootstrap (:mod:`benchmarking.baselines.block_bootstrap`). It is computed after the
uplift, from the uplift's own frozen row selection and bin assignment, and only when the uplift is
finite, so it cannot change any uplift result. The bootstrap sees sampling variability only, so
sigma under-covers where the method is biased (F29).

Each run writes a per-run folder ``toggle_specialist_<test>_<upgradestart>_<lastdate>/``
(v0-style naming) under ``out_dir`` (a temp dir by default), holding a per-segment data-stats CSV,
a headline results CSV, and -- when ``save_plots`` -- three diagnostic plots (a test-vs-reference
scatter, a per-segment daily-ratio timeseries, and a per-segment used-data-coverage timeseries).
The rich stats let a human confirm the right data was received and interpreted: the headline uplift
is re-derivable from the stats CSV as ``rho = used_test_mwh / used_ref_total_mwh`` per segment.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from benchmarking.baselines.block_bootstrap import BootstrapResult, bootstrap_ratio_uplift
from benchmarking.baselines.filtering import NormalOperationFilter
from benchmarking.diagnostics import DiagnosticContext, stages, write_common_diagnostics, write_run_config
from benchmarking.harness.conditions import condition_bins, energy_ratio_by_bin, validate_conditions
from benchmarking.harness.method import MethodInput, MethodOutput
from benchmarking.harness.toggle import ToggleRowSets, is_toggle, resolve_toggle, toggle_upgrade_start
from benchmarking.synthetic import ToggleSchedule

if TYPE_CHECKING:
    import numpy.typing as npt

    from benchmarking.synthetic import ColumnSchema

_SEGMENTS = ("all", "baseline", "upgraded")
_MIN_POINTS_FOR_TIMEBASE = 2
# The cell name (and the ``(condition, condition_bin)`` key) of the headline uplift.
_OVERALL = "overall"

# ``MethodOutput.labeled_rows`` segment labels. "excluded" = claimed by neither side (e.g.
# pre-campaign), which is distinct from a row in a segment that failed the filters (``used`` False).
_BASELINE = "baseline"
_UPGRADED = "upgraded"
_EXCLUDED = "excluded"
# Circular-block length for the uncertainty bootstrap, in hours. Must hold several on/off toggle
# cycles, so raise it for a campaign with a slow toggle period.
DEFAULT_BLOCK_HOURS = 6.0
# ``power`` is the only axis this method can offer: it is derived from the references, so the
# treatment cannot move a row between bins. Binning by the test turbine's ws/TI would condition on
# post-treatment signals, which this method exists not to do (see the module docstring).
_SUPPORTED_CONDITIONS: tuple[str, ...] = ("power",)


def _infer_timebase(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Infer the analysis timebase as the median spacing of the sorted unique timestamps."""
    unique = pd.DatetimeIndex(pd.unique(index)).sort_values()
    if len(unique) < _MIN_POINTS_FOR_TIMEBASE:
        return pd.Timedelta(minutes=10)
    return pd.Timedelta(np.median(np.diff(unique.to_numpy())))


def _wide_column(scada_df: pd.DataFrame, *, turbine_col: str, value_col: str) -> pd.DataFrame:
    """Pivot long SCADA to a timestamp x turbine table of ``value_col`` (NaN where missing)."""
    tmp = scada_df[[turbine_col, value_col]].copy()
    tmp["_ts"] = scada_df.index
    return tmp.pivot_table(
        index="_ts",
        columns=turbine_col,
        values=value_col,
        aggfunc="first",
    )


def restrict_to_campaign(mi: MethodInput) -> MethodInput:
    """Drop pre-campaign rows so the on/off comparison shares a distribution.

    The harness window can also carry a pre-campaign baseline, whose distribution differs from the
    campaign and reintroduces the covariate shift toggling exists to avoid. Restrict the input to
    records at/after the toggle start, leaving only the interleaved on/off blocks. A no-op when the
    schedule has no explicit start (e.g. an already-campaign-only ``toggle_df``).
    """
    timing = mi.upgrade_timing
    if not (isinstance(timing, ToggleSchedule) and timing.start is not None):
        return mi
    return replace(mi, scada_df=mi.scada_df.loc[mi.scada_df.index >= timing.start])


@dataclass
class ToggleSpecialistMethod:
    """Pluggable toggle-only energy-ratio baseline.

    Accepts only toggle campaigns; ``estimate`` raises on a prepost changeover. Always fits on the
    interleaved campaign on/off blocks, so on and off share a wind distribution.

    :param columns: **required** source-native column schema. Reads the ``active_power`` role (the
        only signal in the ``rho`` computation) and the ``availability`` role (the required downtime
        filter, applied to the test turbine and every reference); other roles feed diagnostics only.
    :param name: method name shown in the leaderboard
    :param out_dir: where per-run folders are written; a temp dir when ``None``
    :param save_plots: also write the diagnostic plots under ``<run>/plots``
    :param timebase: analysis timebase; inferred from the data when ``None``
    :param conditions: condition axes to report a per-bin uplift over. Only ``"power"`` is supported
        (see :meth:`_conditional_frame`); defaults to reporting none.
    :param rated_power_kw: the test turbine's rated power, **required** when ``"power"`` is in
        ``conditions``, since the power bin edges scale with the rating.
    :param block_hours: circular-block length for the uncertainty bootstrap. Must hold several on/off
        toggle cycles and stay a small fraction of the campaign; **raise it for a slow toggle
        period**, for which :data:`DEFAULT_BLOCK_HOURS` may span only a cycle or two.
    :param n_resamples: bootstrap resamples; block sums are precomputed, so this can be generous.
    :param bootstrap_seed: RNG seed for the bootstrap, so a reported sigma is reproducible.
    """

    columns: ColumnSchema
    name: str = "toggle_specialist"
    out_dir: Path | None = None
    save_plots: bool = False
    timebase: pd.Timedelta | None = None
    conditions: tuple[str, ...] = ()
    rated_power_kw: float | None = None
    block_hours: float = DEFAULT_BLOCK_HOURS
    n_resamples: int = 1000
    bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        """Validate ``columns`` names every role this method reads, and the requested ``conditions``."""
        self.columns.require_roles(("active_power", "availability"))
        validate_conditions(self.conditions, supported=_SUPPORTED_CONDITIONS, method_name=self.name)
        if "power" in self.conditions and self.rated_power_kw is None:
            msg = (
                f"{self.name}: rated_power_kw is required when 'power' is in conditions — the power bin "
                f"edges are fractions of the turbine's rating."
            )
            raise ValueError(msg)

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Estimate the test turbine's P50 uplift for one toggle campaign and write diagnostics."""
        if not is_toggle(mi.upgrade_timing):
            msg = (
                f"ToggleSpecialistMethod only supports toggle campaigns, but upgrade_timing is a "
                f"{type(mi.upgrade_timing).__name__} (a prepost changeover). Pass a toggle schedule "
                f"or toggle_df; this method has no prepost baseline to compare against."
            )
            raise ValueError(msg)

        mi = restrict_to_campaign(mi)
        wide = _wide_column(mi.scada_df, turbine_col=mi.turbine_col, value_col=self.columns.active_power)
        test = mi.test_wtg
        refs = [c for c in wide.columns if c != test]
        if not refs:
            msg = (
                f"no reference turbines available for test_wtg {test!r}: scada_df contains only "
                f"{list(wide.columns)}. The toggle specialist method needs at least one reference turbine."
            )
            raise ValueError(msg)

        if self.columns.availability not in mi.scada_df.columns:
            msg = (
                f"the availability column {self.columns.availability!r} (columns.availability) is not in "
                f"scada_df; the downtime filter is required for the toggle specialist method and cannot be skipped."
            )
            raise ValueError(msg)

        timebase = self.timebase if self.timebase is not None else _infer_timebase(mi.scada_df.index)
        rows = resolve_toggle(mi.upgrade_timing, wide.index)
        baseline = rows.campaign_baseline
        test_pw = wide[test].to_numpy(dtype=float)
        ref_total = wide[refs].sum(axis=1).to_numpy(dtype=float)
        used = self._used_mask(mi, wide=wide, test=test, refs=refs, timebase=timebase).to_numpy()

        rho_base = _rho(test_pw, ref_total, used & baseline)
        rho_up = _rho(test_pw, ref_total, used & rows.upgraded)
        recoverable = np.isfinite(rho_base) and rho_base != 0 and np.isfinite(rho_up)
        uplift = rho_up / rho_base - 1.0 if recoverable else np.nan
        rho_label = _rho_label(rho_base, rho_up)

        per_bin = (
            self._conditional_frame(
                test_pw=test_pw,
                ref_total=ref_total,
                rho_label=rho_label,
                baseline=used & baseline,
                upgraded=used & rows.upgraded,
            )
            if "power" in self.conditions
            else None
        )

        # Uncertainty runs strictly after the uplift, off the same frozen row selection and bin
        # assignment, and only when there is a finite uplift to qualify.
        membership = self._cell_membership(rho_label=rho_label, ref_total=ref_total, used=used)
        boot = (
            self._bootstrap(
                index=wide.index,
                test_pw=test_pw,
                ref_total=ref_total,
                used=used,
                upgraded=rows.upgraded,
                baseline=baseline,
                membership=membership,
                timebase=timebase,
            )
            if np.isfinite(uplift)
            else None
        )
        if per_bin is not None:
            per_bin["sigma_uplift"] = [_cell_sigma(boot, str(b)) for b in per_bin["condition_bin"]]
        diagnostics = _uncertainty_diagnostics(
            boot,
            membership=membership,
            upgraded=used & rows.upgraded,
            baseline=used & baseline,
            used=used,
        )

        stats = _segment_stats(
            mi,
            wide=wide,
            used=used,
            toggle_rows=rows,
            refs=refs,
            timebase=timebase,
            active_power_col=self.columns.active_power,
        )
        sigma_overall = _cell_sigma(boot, _OVERALL)
        self._write_outputs(
            mi,
            wide=wide,
            stats=stats,
            used=used,
            rho_base=rho_base,
            rho_up=rho_up,
            uplift=uplift,
            sigma_overall=sigma_overall,
            n_refs=len(refs),
            timebase=timebase,
            per_bin=per_bin,
            diagnostics=diagnostics,
        )
        return MethodOutput(
            p50_overall=float(uplift),
            p50_by_condition=per_bin,
            sigma_overall=sigma_overall,
            uncertainty_diagnostics=diagnostics,
            labeled_rows=self._labeled_rows(
                mi, wide=wide, test=test, used=used, rows=rows, rho_label=rho_label, ref_total=ref_total
            ),
        )

    def _labeled_rows(
        self,
        mi: MethodInput,
        *,
        wide: pd.DataFrame,
        test: str,
        used: npt.NDArray[np.bool_],
        rows: ToggleRowSets,
        rho_label: float,
        ref_total: npt.NDArray[np.float64],
    ) -> pd.DataFrame:
        """Return the test turbine's own records, tagged with the labels this estimate was built from.

        Labels are reindexed from the arrays the uplift and bootstrap used, not recomputed, so an
        aggregation of this frame lands on the same rows and bins the estimate did.
        """
        labeled = mi.scada_df[mi.scada_df[mi.turbine_col] == test].copy()

        def _on_test_rows(values: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
            return pd.Series(values, index=wide.index).reindex(labeled.index).to_numpy()

        labeled["used"] = _on_test_rows(used)
        labeled["segment"] = _on_test_rows(
            np.where(rows.upgraded, _UPGRADED, np.where(rows.campaign_baseline, _BASELINE, _EXCLUDED))
        )

        # The bin label is the same reference-derived baseline power the uplift binned on, so a row
        # cannot sit in one bin here and another there. Outside the outer edges pd.cut gives NaN,
        # which is carried through as "this row belongs to no bin" rather than clipped to an edge.
        if "power" in self.conditions and np.isfinite(rho_label):
            assert self.rated_power_kw is not None  # noqa: S101 - guaranteed by __post_init__
            bins = condition_bins("power", rated_power_kw=self.rated_power_kw)
            labeled["power_bin"] = _on_test_rows(np.asarray(pd.cut(rho_label * ref_total, bins=bins)))
        return labeled

    def _cell_membership(
        self,
        *,
        rho_label: float,
        ref_total: npt.NDArray[np.float64],
        used: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.bool_]]:
        """Which **used** records belong to each bootstrap cell: the headline, plus each power bin.

        Reuses :meth:`_conditional_frame`'s own label and edges, so a record's cell is fixed by the
        uplift computation and cannot move under resampling.
        """
        used_idx = np.flatnonzero(used)
        membership: dict[str, npt.NDArray[np.bool_]] = {_OVERALL: np.ones(len(used_idx), dtype=bool)}
        if "power" not in self.conditions or not np.isfinite(rho_label):
            return membership
        assert self.rated_power_kw is not None  # noqa: S101 - guaranteed by __post_init__
        bins = condition_bins("power", rated_power_kw=self.rated_power_kw)
        assigned = pd.cut(rho_label * ref_total[used_idx], bins=bins)
        for category in assigned.categories:
            membership[str(category)] = np.asarray(assigned == category)
        return membership

    def _bootstrap(
        self,
        *,
        index: pd.DatetimeIndex,
        test_pw: npt.NDArray[np.float64],
        ref_total: npt.NDArray[np.float64],
        used: npt.NDArray[np.bool_],
        upgraded: npt.NDArray[np.bool_],
        baseline: npt.NDArray[np.bool_],
        membership: dict[str, npt.NDArray[np.bool_]],
        timebase: pd.Timedelta,
    ) -> BootstrapResult:
        """Run the circular block bootstrap over the used records of the campaign.

        The campaign span is taken from the on/off rows rather than from ``index``, so blocks tile
        the campaign itself even when the caller's window carries pre-campaign rows the estimate
        never used.
        """
        used_idx = np.flatnonzero(used)
        campaign = upgraded | baseline
        return bootstrap_ratio_uplift(
            times=index[used_idx],
            test_power=test_pw[used_idx],
            ref_total=ref_total[used_idx],
            upgraded=upgraded[used_idx],
            baseline=baseline[used_idx],
            cell_membership=membership,
            campaign_start=index[campaign].min(),
            campaign_end=index[campaign].max(),
            timebase=timebase,
            block_hours=self.block_hours,
            n_resamples=self.n_resamples,
            seed=self.bootstrap_seed,
        )

    def _conditional_frame(
        self,
        *,
        test_pw: npt.NDArray[np.float64],
        ref_total: npt.NDArray[np.float64],
        rho_label: float,
        baseline: npt.NDArray[np.bool_],
        upgraded: npt.NDArray[np.bool_],
    ) -> pd.DataFrame:
        """Per-power-bin uplift: ``rho_up(b) / rho_base(b) - 1``, on bins of the mean operating point.

        Two decisions carry this, and both are needed:

        **The bin label is** ``rho_label * ref_total`` (see :func:`_rho_label`): reference-derived and
        state-neutral, so neither the upgrade nor which state is called baseline can move a row
        between bins, and it is on the test turbine's own kW scale.

        **The denominator is the per-bin** ``rho_base(b)``, not the global one: the test-to-reference
        ratio varies with power, and a global denominator would read that structure as uplift. The
        price is that the per-bin numbers no longer aggregate exactly to ``p50_overall``, which is
        deliberate and un-relevelled; ``sum_actual`` / ``sum_counterfactual`` expose the gap.

        Sparse bins report NaN with ``n_records = 0`` rather than being imputed.
        """
        assert self.rated_power_kw is not None  # noqa: S101 - guaranteed by __post_init__
        bins = condition_bins("power", rated_power_kw=self.rated_power_kw)
        label = rho_label * ref_total
        counterfactual = _per_bin_counterfactual(
            label=label, test_pw=test_pw, ref_total=ref_total, baseline=baseline, bins=bins
        )
        frame = energy_ratio_by_bin(label[upgraded], test_pw[upgraded], counterfactual[upgraded], bins=bins)
        frame.insert(0, "condition", "power")
        return frame

    def _used_mask(
        self, mi: MethodInput, *, wide: pd.DataFrame, test: str, refs: list[str], timebase: pd.Timedelta
    ) -> pd.Series:
        """Complete-case timestamps that also pass downtime filtering on the test turbine and every reference.

        Returns a bool Series on ``wide.index``. Every turbine (test and references) must be
        available (counter >= a full period) and have finite power — a down turbine on either side
        of the ratio is therefore excluded. The test turbine additionally goes through the shared
        :class:`NormalOperationFilter` (the same downtime + finite-power logic the R-learner uses;
        the stuck filter is left off here as the ratio sums raw power rather than fitting a model).
        """
        turbines = [test, *refs]
        complete = wide[turbines].notna().all(axis=1)

        full = timebase.total_seconds()
        avail = _wide_column(mi.scada_df, turbine_col=mi.turbine_col, value_col=self.columns.availability).reindex(
            index=wide.index, columns=turbines
        )
        all_available = (avail >= full).all(axis=1)

        test_rows = mi.scada_df[mi.scada_df[mi.turbine_col] == test]
        test_keep = (
            NormalOperationFilter(
                active_power_col=self.columns.active_power,
                availability_col=self.columns.availability,
                apply_stuck_filter=False,
            )
            .keep_mask(test_rows, timebase=timebase)
            .reindex(wide.index, fill_value=False)
        )
        return complete & all_available & test_keep

    def _write_outputs(
        self,
        mi: MethodInput,
        *,
        wide: pd.DataFrame,
        stats: pd.DataFrame,
        used: np.ndarray,
        rho_base: float,
        rho_up: float,
        uplift: float,
        sigma_overall: float,
        n_refs: int,
        timebase: pd.Timedelta,
        per_bin: pd.DataFrame | None = None,
        diagnostics: pd.DataFrame | None = None,
    ) -> None:
        """Write the data-stats CSV, the headline results CSV, the per-bin CSV and (optionally) the plots."""
        upgrade_start = toggle_upgrade_start(mi.upgrade_timing, wide.index)
        last_dt = wide.index.max()
        run_name = f"toggle_specialist_{mi.test_wtg}_{upgrade_start:%Y%m%d}_{last_dt:%Y%m%d}"
        out_root = (
            Path(self.out_dir) if self.out_dir is not None else Path(tempfile.mkdtemp(prefix="toggle_specialist_"))
        )
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S_%f")

        stats.to_csv(run_dir / f"{run_name}_data_stats_{ts}.csv", index=False)

        used_base = int(stats.loc[stats["segment"] == "baseline", "n_used_timestamps"].iloc[0])
        used_up = int(stats.loc[stats["segment"] == "upgraded", "n_used_timestamps"].iloc[0])
        results = pd.DataFrame(
            [
                {
                    "test_wtg": mi.test_wtg,
                    "mode": "toggle",
                    "n_turbines": wide.shape[1],
                    "n_refs": n_refs,
                    "ratio_baseline": rho_base,
                    "ratio_upgraded": rho_up,
                    "uplift_frc": uplift,
                    "uplift_sigma_frc": sigma_overall,
                    "block_hours": self.block_hours,
                    "n_resamples": self.n_resamples,
                    "n_used_timestamps_baseline": used_base,
                    "n_used_timestamps_upgraded": used_up,
                    "time_calculated": pd.Timestamp.utcnow(),
                }
            ]
        )
        results.to_csv(run_dir / f"{run_name}_results_{ts}.csv", index=False)

        if per_bin is not None:
            per_bin.to_csv(run_dir / f"{run_name}_by_power_bin_{ts}.csv", index=False)
        if diagnostics is not None:
            diagnostics.to_csv(run_dir / f"{run_name}_uncertainty_{ts}.csv", index=False)

        if self.save_plots:
            _save_plots(
                run_dir / "plots",
                wide=wide,
                mi=mi,
                test=mi.test_wtg,
                used=used,
                timebase=timebase,
                active_power_col=self.columns.active_power,
            )
            if per_bin is not None:
                _save_per_bin_plot(
                    run_dir / "plots" / stages.CONDITIONAL_UPLIFT / f"{mi.test_wtg}_per_bin_uplift.png",
                    per_bin=per_bin,
                    test=mi.test_wtg,
                    active_power_col=self.columns.active_power,
                )
            self._write_shared_diagnostics(mi, run_dir=run_dir, wide=wide, timebase=timebase)

    def _write_shared_diagnostics(
        self, mi: MethodInput, *, run_dir: Path, wide: pd.DataFrame, timebase: pd.Timedelta
    ) -> None:
        """Emit the shared cross-method diagnostics (coverage/curves/histograms) and the run config."""
        # ``wide`` (a pivot) drops all-NaN timestamps, so align the masks to the full unique index
        # the DiagnosticContext uses (timestamps absent from ``wide`` are simply not used).
        index = pd.DatetimeIndex(pd.unique(mi.scada_df.index)).sort_values()
        test, refs = mi.test_wtg, [c for c in wide.columns if c != mi.test_wtg]
        used_series = self._used_mask(mi, wide=wide, test=test, refs=refs, timebase=timebase)
        used = used_series.reindex(index, fill_value=False).to_numpy()
        treated = resolve_toggle(mi.upgrade_timing, index).upgraded.astype(bool)
        ctx = DiagnosticContext(
            run_dir=run_dir,
            test_wtg=mi.test_wtg,
            turbine_col=mi.turbine_col,
            columns=self.columns,
            scada_df=mi.scada_df,
            treated_ts=treated,
            used_ts=used,
            timebase=timebase,
            mode="toggle",
            era5_df=None,
        )
        write_common_diagnostics(ctx)
        params = {
            "active_power_col": self.columns.active_power,
            "availability_col": self.columns.availability,
        }
        write_run_config(ctx, method_name=self.name, method_params=params)


def _per_bin_counterfactual(
    *,
    label: npt.NDArray[np.float64],
    test_pw: npt.NDArray[np.float64],
    ref_total: npt.NDArray[np.float64],
    baseline: npt.NDArray[np.bool_],
    bins: list[float],
) -> npt.NDArray[np.float64]:
    """Each row's counterfactual test power: its own bin's baseline ratio times its reference total.

    ``rho_base(b)`` is measured over the baseline rows of bin ``b``; every row (of either segment) then
    takes the ``rho_base`` of the bin its ``label`` falls in. Rows in a bin with no baseline rows get
    NaN, which is what makes an uncovered bin report NaN rather than an imputed value.
    """
    assigned = pd.cut(label, bins=bins)
    rho_by_bin = {
        category: _rho(test_pw, ref_total, baseline & np.asarray(assigned == category))
        for category in assigned.categories
    }
    rho_row = np.asarray(pd.Series(assigned).map(rho_by_bin).astype(float))
    return rho_row * ref_total


def _cell_sigma(boot: BootstrapResult | None, cell: str) -> float:
    """Return one cell's 1-sigma, or NaN when the bootstrap did not run or never saw that cell."""
    if boot is None or cell not in boot.cells:
        return float("nan")
    return boot.cells[cell].sigma


def _uncertainty_diagnostics(
    boot: BootstrapResult | None,
    *,
    membership: dict[str, npt.NDArray[np.bool_]],
    upgraded: npt.NDArray[np.bool_],
    baseline: npt.NDArray[np.bool_],
    used: npt.NDArray[np.bool_],
) -> pd.DataFrame:
    """Per-cell account of how the uncertainty was reached, keyed by ``(condition, condition_bin)``.

    Carried through the harness seam uninterpreted, so an uncertainty model can be developed against
    a saved sweep rather than by re-running one. Emitted even when the bootstrap did not run: the
    counts are what explain why. Both counts are reported because a cell fails when either side of
    its ratio runs out, and a single total would hide which.
    """
    used_idx = np.flatnonzero(used)
    up_used = upgraded[used_idx]
    base_used = baseline[used_idx]
    nan = float("nan")
    rows = []
    for cell, member in membership.items():
        cell_boot = boot.cells[cell] if boot is not None and cell in boot.cells else None
        rows.append(
            {
                "condition": _OVERALL if cell == _OVERALL else "power",
                "condition_bin": cell,
                "n_upgraded_records": int((member & up_used).sum()),
                "n_baseline_records": int((member & base_used).sum()),
                "n_blocks": boot.n_blocks if boot is not None else 0,
                # Both components, not just the reported max: a blend rule can then be re-judged from
                # a saved sweep rather than by re-running one.
                "sigma_bootstrap": cell_boot.sigma_bootstrap if cell_boot is not None else nan,
                "sigma_fallback": cell_boot.sigma_fallback if cell_boot is not None else nan,
                "sigma_robust": cell_boot.sigma_robust if cell_boot is not None else nan,
                "frac_resamples_finite": cell_boot.frac_resamples_finite if cell_boot is not None else nan,
            }
        )
    return pd.DataFrame(rows)


def _rho(test_pw: npt.NDArray[np.float64], ref_total: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_]) -> float:
    """Test-to-reference ratio over ``mask``: sum(test) / sum(ref_total). NaN if degenerate."""
    if not mask.any():
        return float("nan")
    denom = ref_total[mask].sum()
    if denom == 0:
        return float("nan")
    return float(test_pw[mask].sum() / denom)


def _rho_label(rho_base: float, rho_up: float) -> float:
    """Return the test-to-reference ratio used to *label* bins: the mean of the two states.

    State-neutral by construction, so relabelling which state is the baseline cannot move a row
    between bins. Still a campaign-level scalar, so the upgrade cannot move a row either.
    """
    return 0.5 * (rho_base + rho_up)


def _segment_stats(
    mi: MethodInput,
    *,
    wide: pd.DataFrame,
    used: npt.NDArray[np.bool_],
    toggle_rows: ToggleRowSets,
    refs: list[str],
    timebase: pd.Timedelta,
    active_power_col: str,
) -> pd.DataFrame:
    """Build the per-segment (all/baseline/upgraded) diagnostics table."""
    test = mi.test_wtg
    test_pw = wide[test].to_numpy(dtype=float)
    ref_total = wide[refs].sum(axis=1).to_numpy(dtype=float)
    n_turbines = wide.shape[1]
    timebase_hours = timebase / pd.Timedelta(hours=1)

    row_rows = resolve_toggle(mi.upgrade_timing, mi.scada_df.index)
    row_power = mi.scada_df[active_power_col].to_numpy(dtype=float)

    ts_baseline = toggle_rows.campaign_baseline
    row_baseline = row_rows.campaign_baseline
    ts_masks = {"all": np.ones(len(wide), dtype=bool), "baseline": ts_baseline, "upgraded": toggle_rows.upgraded}
    row_masks = {"all": np.ones(len(mi.scada_df), dtype=bool), "baseline": row_baseline, "upgraded": row_rows.upgraded}

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


def _save_plots(
    plots_dir: Path,
    *,
    wide: pd.DataFrame,
    mi: MethodInput,
    test: str,
    used: np.ndarray,
    timebase: pd.Timedelta,
    active_power_col: str,
) -> None:
    """Write the scatter, ratio-timeseries and used-coverage-timeseries diagnostic plots (by stage).

    ``used`` is the method's real downtime-filtered mask (test + every reference passing the
    availability/finite filter), so the scatter shows only the rows the estimate actually uses.
    The baseline is the strict campaign off-blocks the estimate used, so the plots never disagree
    with the headline.
    """
    refs = [c for c in wide.columns if c != test]
    toggle_rows = resolve_toggle(mi.upgrade_timing, wide.index)
    baseline_mask = toggle_rows.campaign_baseline
    test_pw = wide[test].to_numpy(dtype=float)
    ref_total = wide[refs].sum(axis=1).to_numpy(dtype=float)
    upgrade_start = toggle_upgrade_start(mi.upgrade_timing, wide.index)
    segments = (
        ("baseline", used & baseline_mask, "C0"),
        ("upgraded", used & toggle_rows.upgraded, "C1"),
    )

    # 1) scatter of test vs reference-total power, baseline/upgraded coloured, with rho slopes.
    fig, ax = plt.subplots(figsize=(7, 7))
    for label, seg, color in segments:
        ax.scatter(ref_total[seg], test_pw[seg], s=8, alpha=0.4, color=color, label=label)
        rho = _rho(test_pw, ref_total, seg)
        if np.isfinite(rho) and seg.any():
            x_max = float(np.nanmax(ref_total[seg]))
            ax.plot([0, x_max], [0, rho * x_max], color=color, linewidth=1.5)
    ax.set_xlabel(f"sum of reference {active_power_col} [kW]")
    ax.set_ylabel(f"{active_power_col} @ {test} [kW]")
    ax.set_title(f"{test}: test vs reference-total power")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, plots_dir / stages.UPLIFT_INPUTS / f"{test}_scatter.png")

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
    _save(fig, plots_dir / stages.UPLIFT_RESULTS / f"{test}_ratio_timeseries.png")

    # 3) daily used-data coverage as a fraction of the day's expected timestamps, one series per
    # segment, so each segment is seen to receive its share (under toggle, ~50% each post-upgrade).
    expected_per_day = _expected_per_day(wide.index, timebase)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, _seg, color in segments:
        seg_mask = baseline_mask if label == "baseline" else toggle_rows.upgraded
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
    _save(fig, plots_dir / stages.FILTER / f"{test}_coverage_timeseries.png")


def _save_per_bin_plot(path: Path, *, per_bin: pd.DataFrame, test: str, active_power_col: str) -> None:
    """Plot the per-power-bin uplift with each bin's used-record count underneath.

    The record count is the point of the second panel: a per-bin uplift is only as trustworthy as the
    data behind it, and the sparse bins are exactly where a reader must not over-read the top panel.
    Empty bins are gaps, never plotted as zero.
    """
    populated = per_bin["n_records"].to_numpy() > 0
    x = np.arange(len(per_bin))
    uplift = np.where(populated, per_bin["p50_uplift"].to_numpy() * 100.0, np.nan)

    fig, (ax_uplift, ax_n) = plt.subplots(2, 1, sharex=True, figsize=(9, 7), height_ratios=[2, 1])
    ax_uplift.plot(x, uplift, marker="o", color="C1")
    ax_uplift.axhline(0.0, color="k", linewidth=0.8)
    ax_uplift.set_ylabel("uplift [pp]")
    ax_uplift.set_title(f"{test}: uplift by {active_power_col} bin")
    ax_uplift.grid(visible=True, alpha=0.3)

    ax_n.bar(x, per_bin["n_records"].to_numpy(), color="C0", alpha=0.7)
    ax_n.set_ylabel("used records")
    ax_n.set_xlabel(f"{active_power_col} bin [kW] (predicted baseline)")
    ax_n.set_xticks(x)
    ax_n.set_xticklabels(per_bin["condition_bin"].astype(str), rotation=20, ha="right", fontsize=8)
    ax_n.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> None:
    """Write a figure to ``path`` (creating its stage subfolder) and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
