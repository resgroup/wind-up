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

An optional **pairing filter** (``pairing_max_gap``) then drops any used row with no used row of the
other segment nearby, so a filter that removes one segment's rows in some conditions also removes the
other segment's rows from those conditions. Rows kept per segment after each selection stage are
reported as ``MethodOutput.selection_accounting`` and written to a selection CSV. A caller may also
flag rows through ``columns.exclude_block``: such a row removes its whole reference block (one
toggle cycle), so both states lose the same conditions — a symmetric alternative to ``exclude_row``
for a selection that would otherwise thin one state's rows in particular conditions.

**Two legs, blended.** The test power is compared against two references, each estimated on
its own row selection, and the two estimates are blended by minimum-variance weights, headline and
per bin. The *reference leg* (``sum``) compares against the summed power of a **reference subset**:
the ``sum`` estimate runs on every non-empty subset of the available references and keeps the
subset whose noise is smallest (ties to the larger subset), so a reference that tracks badly, or is
screened out for long stretches, loses to the subsets without it; every subset's result is written
to a CSV. The *block leg* (``block_mean``) needs **no references at all**: the campaign is tiled
into fixed wall-clock blocks of ``reference_block`` (one toggle cycle) and every used row
of a block is compared against the block's mean test power, so the on/off ratio cancels the block's
wind level exactly; a block whose used rows lack either state is dropped (the ``block`` selection
stage). The two legs' errors come from different places (reference mismatch versus within-cycle
drift), which is what the blend exploits; their bootstraps share their block draws, so the blend's
sigma includes the legs' correlation (:func:`benchmarking.baselines.block_bootstrap.combine_estimates`).
With no reference turbine at all the estimate is the block leg alone.

**Decisions on permuted labels.** The two data-driven decisions, the subset choice and the blend
weights, are judged on a bootstrap of the same rows with the on/off labels shuffled within each
reference block: the same noise, blind to how it happened to split between the states. Ranking on
the realised sigma would favour the alternative whose realised uplift came out low (the on/off
contrast noise is skewed), biasing the estimate down and under-reporting sigma. The reported uplift
and sigma are always the actual ones; only the decisions read the permuted bootstrap.

Every uplift — the headline and each power bin — comes with a non-optional 1-sigma uncertainty from
a circular block bootstrap (:mod:`benchmarking.baselines.block_bootstrap`). It is computed after the
uplift, from the uplift's own frozen row selection and bin assignment, and only when the uplift is
finite, so it cannot change any uplift result. The bootstrap sees sampling variability only, so
sigma under-covers where the method is biased.

Each run writes a per-run folder ``toggle_specialist_<test>_<upgradestart>_<lastdate>/``
(v0-style naming) under ``out_dir`` (a temp dir by default), holding a per-segment data-stats CSV,
a headline results CSV, the per-leg selection and reference-subset CSVs, and -- when
``save_plots`` -- three diagnostic plots (a test-vs-reference scatter, a per-segment daily-ratio
timeseries, and a per-segment used-data-coverage timeseries).
The rich stats let a human confirm the right data was received and interpreted: the headline uplift
is re-derivable from the stats CSV as ``rho = used_test_mwh / used_ref_total_mwh`` per segment.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from benchmarking.baselines.block_bootstrap import (
    BootstrapResult,
    CombinedEstimate,
    bootstrap_ratio_uplift,
    combine_estimates,
)
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
# The two legs the estimate blends (see the module docstring), in blend order: the first is the one
# the row-selection accounting and the reference-side diagnostics report.
_REFERENCE_LEG = "sum"
_BLOCK_LEG = "block_mean"
_COMBINED = "combined"
# The subset choice runs 2**n - 1 estimates; warn, rather than refuse, past this many references.
_MAX_REFS_FOR_SUBSETS = 6

logger = logging.getLogger(__name__)


class _Selection(NamedTuple):
    """The cumulative used-row mask after each selection stage."""

    filtered: pd.Series
    not_excluded: pd.Series
    paired: pd.Series
    blocked: pd.Series


@dataclass(frozen=True)
class _Estimate:
    """One leg's (or the blend's) complete estimate: what the outputs, the labels and the blend read."""

    mode: str
    refs: list[str]
    selection: _Selection
    used: npt.NDArray[np.bool_]
    ref_total: npt.NDArray[np.float64]
    label: npt.NDArray[np.float64]
    rho_base: float
    rho_up: float
    uplift: float
    sigma_overall: float
    per_bin: pd.DataFrame | None
    boot: BootstrapResult | None
    accounting: pd.DataFrame
    diagnostics: pd.DataFrame
    # The label-permuted bootstrap the decisions read; None for the blend (which decides nothing further).
    decision_boot: BootstrapResult | None = None

    @property
    def decision_sigma(self) -> float:
        """The headline sigma the decisions are judged on: that of the label-permuted bootstrap."""
        return _cell_sigma(self.decision_boot, _OVERALL)


def permute_labels_within_blocks(
    *,
    upgraded: npt.NDArray[np.bool_],
    baseline: npt.NDArray[np.bool_],
    used: npt.NDArray[np.bool_],
    ids: npt.NDArray[np.int64] | None,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Shuffle the on/off labels among the used campaign rows of each block; unused rows keep theirs.

    Every block keeps its on and off counts, so the permuted campaign has the same structure as the
    real one and differs only in which rows are called on. With ``ids`` None the labels are shuffled
    over the whole campaign.
    """
    up_p = upgraded.copy()
    base_p = baseline.copy()
    campaign_used = np.flatnonzero(used & (upgraded | baseline))
    if ids is None:
        groups = [campaign_used]
    else:
        groups = [campaign_used[ids[campaign_used] == b] for b in np.unique(ids[campaign_used])]
    for rows in groups:
        if len(rows) < 2:  # noqa: PLR2004 - nothing to shuffle
            continue
        order = rng.permutation(rows)
        up_p[rows] = upgraded[order]
        base_p[rows] = baseline[order]
    return up_p, base_p


class PairedRows(NamedTuple):
    """The rows of each segment that have a row of the other segment within the pairing gap."""

    baseline: npt.NDArray[np.bool_]
    upgraded: npt.NDArray[np.bool_]


def pair_within(
    index: pd.DatetimeIndex,
    *,
    baseline: npt.NDArray[np.bool_],
    upgraded: npt.NDArray[np.bool_],
    max_gap: pd.Timedelta,
) -> PairedRows:
    """Keep each flagged row only if the other segment has a flagged row within ``max_gap``, inclusive."""
    gap = gap_to_other_segment(index, baseline=baseline, upgraded=upgraded)
    within = gap <= max_gap.total_seconds()
    return PairedRows(baseline=baseline & within, upgraded=upgraded & within)


def gap_to_other_segment(
    index: pd.DatetimeIndex, *, baseline: npt.NDArray[np.bool_], upgraded: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Seconds from each flagged row to the nearest flagged row of the other segment; NaN if unflagged."""
    times = index.as_unit("ns").asi8
    gap = np.full(len(index), np.nan)
    gap[baseline] = _nearest_gap_s(times[baseline], others=times[upgraded])
    gap[upgraded] = _nearest_gap_s(times[upgraded], others=times[baseline])
    return gap


def _nearest_gap_s(times: npt.NDArray[np.int64], *, others: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    if not len(others):
        return np.full(len(times), np.inf)
    others = np.sort(others)
    position = np.searchsorted(others, times)
    after = others[np.minimum(position, len(others) - 1)]
    before = others[np.maximum(position - 1, 0)]
    return np.minimum(np.abs(after - times), np.abs(times - before)) / 1e9


def _infer_timebase(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Infer the analysis timebase as the median spacing of the sorted unique timestamps."""
    unique = pd.DatetimeIndex(pd.unique(index)).sort_values()
    if len(unique) < _MIN_POINTS_FOR_TIMEBASE:
        return pd.Timedelta(minutes=10)
    return pd.Timedelta(np.median(np.diff(unique.to_numpy())))


def block_ids(index: pd.DatetimeIndex, *, start: pd.Timestamp, block: pd.Timedelta) -> npt.NDArray[np.int64]:
    """Index of the fixed wall-clock block of length ``block``, tiled forward from ``start``, holding each row."""
    return np.floor(np.asarray((index - start) / block, dtype=float)).astype(np.int64)


def blocks_with_both_states(
    ids: npt.NDArray[np.int64], *, baseline: npt.NDArray[np.bool_], upgraded: npt.NDArray[np.bool_]
) -> npt.NDArray[np.bool_]:
    """Per row: whether its block holds at least one ``baseline`` and one ``upgraded`` row."""
    both = set(np.unique(ids[baseline])) & set(np.unique(ids[upgraded]))
    return np.isin(ids, list(both))


def _block_sum(ids: npt.NDArray[np.int64], values: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_]) -> pd.Series:
    """Sum of ``values`` over the ``mask`` rows of each block, indexed by block id."""
    return pd.Series(values[mask]).groupby(ids[mask]).sum()


def _per_row(ids: npt.NDArray[np.int64], per_block: pd.Series) -> npt.NDArray[np.float64]:
    """Broadcast a per-block value back onto rows; NaN for a row whose block has no value."""
    return per_block.reindex(ids).to_numpy(dtype=float)


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
    :param pairing_max_gap: when set, a used row of one segment is kept only if the other segment has
        a used row within this gap, inclusive (a gap of exactly ``pairing_max_gap`` is kept). Must be
        a positive whole multiple of the timebase; ``None`` disables pairing.
    :param segment_imbalance_warning: warn when the two segments' kept fractions at any selection
        stage differ by more than this.
    :param reference_block: **required** block length for the block leg, the label permutation and
        ``columns.exclude_block``: one toggle cycle (the campaign's own on + off duration), a
        positive whole multiple of the timebase. It is the caller's to state, not inferred, and it
        is independent of ``pairing_max_gap``.
    """

    columns: ColumnSchema
    reference_block: pd.Timedelta
    name: str = "toggle_specialist"
    out_dir: Path | None = None
    save_plots: bool = False
    timebase: pd.Timedelta | None = None
    conditions: tuple[str, ...] = ()
    rated_power_kw: float | None = None
    block_hours: float = DEFAULT_BLOCK_HOURS
    n_resamples: int = 1000
    bootstrap_seed: int = 0
    pairing_max_gap: pd.Timedelta | None = None
    segment_imbalance_warning: float = 0.1

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
        if self.columns.availability not in mi.scada_df.columns:
            msg = (
                f"the availability column {self.columns.availability!r} (columns.availability) is not in "
                f"scada_df; the downtime filter is required for the toggle specialist method and cannot be skipped."
            )
            raise ValueError(msg)

        mi = restrict_to_campaign(mi)
        wide_all = _wide_column(mi.scada_df, turbine_col=mi.turbine_col, value_col=self.columns.active_power)
        test = mi.test_wtg
        timebase = self.timebase if self.timebase is not None else _infer_timebase(mi.scada_df.index)
        self._check_pairing_gap(timebase)
        rows = resolve_toggle(mi.upgrade_timing, wide_all.index)
        block = self._reference_block(timebase)

        components: dict[str, _Estimate] = {}
        subsets: pd.DataFrame | None = None
        n_refs_available = len(mi.context.references_among(wide_all.columns))
        if n_refs_available:
            components[_REFERENCE_LEG], subsets = self._choose_references(
                mi, wide_all=wide_all, test=test, timebase=timebase, block=block, rows=rows
            )
        else:
            # Nothing for the reference leg to work with: the estimate is the block leg alone, which
            # needs no references, rather than a failed estimate.
            logger.warning(
                "%s: no reference turbines available for test_wtg %r; the estimate is the %r leg alone",
                self.name,
                test,
                _BLOCK_LEG,
            )
        components[_BLOCK_LEG] = self._component(
            mi, mode=_BLOCK_LEG, wide_all=wide_all, test=test, timebase=timebase, block=block, rows=rows
        )
        est = (
            _combine(components[_REFERENCE_LEG], components[_BLOCK_LEG], upgraded=rows.upgraded, bins=self._bins())
            if _REFERENCE_LEG in components
            else components[_BLOCK_LEG]
        )

        ref_set = set(est.refs)
        wide = wide_all[[c for c in wide_all.columns if c == test or c in ref_set]]
        stats = _segment_stats(
            mi,
            wide=wide,
            used=est.used,
            toggle_rows=rows,
            ref_total=est.ref_total,
            timebase=timebase,
            active_power_col=self.columns.active_power,
        )
        self._write_outputs(
            mi,
            wide=wide,
            stats=stats,
            est=est,
            components=components,
            timebase=timebase,
            rows=rows,
            subsets=subsets,
            n_refs_available=n_refs_available,
        )
        return MethodOutput(
            p50_overall=float(est.uplift),
            p50_by_condition=est.per_bin,
            sigma_overall=est.sigma_overall,
            uncertainty_diagnostics=est.diagnostics,
            labeled_rows=self._labeled_rows(mi, wide=wide, test=test, used=est.used, rows=rows, label=est.label),
            selection_accounting=est.accounting,
        )

    def _component(
        self,
        mi: MethodInput,
        *,
        mode: str,
        wide_all: pd.DataFrame,
        test: str,
        timebase: pd.Timedelta,
        block: pd.Timedelta | None,
        rows: ToggleRowSets,
        refs: list[str] | None = None,
        with_actual: bool = True,
    ) -> _Estimate:
        """Run one leg end to end: row selection, reference, uplift, bins and the two bootstraps.

        ``refs`` names the references the ``sum`` leg uses (the block leg compares the test turbine
        against its own block, so references play no part, not even in the row filters: a reference
        outage must not cost it rows). ``with_actual=False`` skips the actual bootstrap (a subset
        that is only being ranked on its permuted sigma); the label-permuted bootstrap always runs.
        """
        refs = [] if mode == _BLOCK_LEG else list(refs or [])
        if mode == _REFERENCE_LEG and not refs:
            msg = f"{self.name}: the {mode!r} leg needs at least one reference turbine"
            raise ValueError(msg)
        # Narrow to the mode's turbines and blank the cells the campaign says may not contribute, so
        # the estimate and every diagnostic see one consistent selection.
        ref_set = set(refs)
        wide = mi.context.mask_invalid(wide_all[[c for c in wide_all.columns if c == test or c in ref_set]])

        baseline = rows.campaign_baseline
        test_pw = wide[test].to_numpy(dtype=float)
        campaign = baseline | rows.upgraded
        ids = (
            block_ids(wide.index, start=wide.index[campaign].min(), block=block)
            if block is not None and campaign.any()
            else None
        )
        selection = self._selection(
            mi, wide=wide, test=test, refs=refs, timebase=timebase, rows=rows, ids=ids, both_states=mode == _BLOCK_LEG
        )
        used = selection.blocked.to_numpy()
        accounting = self._selection_accounting(selection, rows=rows)
        ref_total = self._reference_total(mode=mode, wide=wide, refs=refs, test_pw=test_pw, used=used, ids=ids)

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

        def _run_bootstrap(up: npt.NDArray[np.bool_], base: npt.NDArray[np.bool_]) -> BootstrapResult:
            return self._bootstrap(
                index=wide.index,
                test_pw=test_pw,
                ref_total=ref_total,
                used=used,
                upgraded=up,
                baseline=base,
                membership=membership,
                timebase=timebase,
            )

        boot = _run_bootstrap(rows.upgraded, baseline) if with_actual and np.isfinite(uplift) else None
        decision_boot = None
        if np.isfinite(uplift):
            # Same rows, span, blocks, seed and cells as the actual bootstrap; only the labels differ.
            up_p, base_p = permute_labels_within_blocks(
                upgraded=rows.upgraded,
                baseline=baseline,
                used=used,
                ids=ids,
                rng=np.random.default_rng(self.bootstrap_seed),
            )
            decision_boot = _run_bootstrap(up_p, base_p)
        if per_bin is not None:
            per_bin["sigma_uplift"] = [_cell_sigma(boot, str(b)) for b in per_bin["condition_bin"]]
        diagnostics = _uncertainty_diagnostics(
            boot,
            membership=membership,
            upgraded=used & rows.upgraded,
            baseline=used & baseline,
            used=used,
        )
        diagnostics.insert(0, "component", mode)
        return _Estimate(
            mode=mode,
            refs=refs,
            selection=selection,
            used=used,
            ref_total=ref_total,
            label=rho_label * ref_total,
            rho_base=rho_base,
            rho_up=rho_up,
            uplift=uplift,
            sigma_overall=_cell_sigma(boot, _OVERALL),
            per_bin=per_bin,
            boot=boot,
            accounting=accounting,
            diagnostics=diagnostics,
            decision_boot=decision_boot,
        )

    def _choose_references(
        self,
        mi: MethodInput,
        *,
        wide_all: pd.DataFrame,
        test: str,
        timebase: pd.Timedelta,
        block: pd.Timedelta | None,
        rows: ToggleRowSets,
    ) -> tuple[_Estimate, pd.DataFrame | None]:
        """Run the ``sum`` leg on every non-empty reference subset; return the smallest-sigma one.

        The sigma ranked is the ``decision_sigma`` of the label-permuted bootstrap; the subsets skip
        their actual bootstrap and the winner is re-run in full. Ties go to the larger subset, then
        to the earlier in enumeration (the context's reference order). No finite sigma anywhere ->
        the full set. The second value lists every subset's result with the winner marked.
        """
        refs_all = mi.context.references_among(wide_all.columns)
        if len(refs_all) > _MAX_REFS_FOR_SUBSETS:
            logger.warning(
                "%s: the reference-subset choice over %d references runs %d estimates",
                self.name,
                len(refs_all),
                2 ** len(refs_all) - 1,
            )
        subsets = [list(c) for k in range(1, len(refs_all) + 1) for c in combinations(refs_all, k)]
        estimates = [
            self._component(
                mi,
                mode=_REFERENCE_LEG,
                wide_all=wide_all,
                test=test,
                timebase=timebase,
                block=block,
                rows=rows,
                refs=refs,
                with_actual=False,
            )
            for refs in subsets
        ]

        def _rank(i: int) -> tuple[float, int, int]:
            sigma = estimates[i].decision_sigma
            return (sigma if np.isfinite(sigma) else np.inf, -len(subsets[i]), i)

        best = min(range(len(subsets)), key=_rank)
        estimates[best] = self._component(
            mi,
            mode=_REFERENCE_LEG,
            wide_all=wide_all,
            test=test,
            timebase=timebase,
            block=block,
            rows=rows,
            refs=subsets[best],
        )
        table = pd.DataFrame(
            {
                "refs": [";".join(refs) for refs in subsets],
                "n_refs": [len(refs) for refs in subsets],
                "n_used_timestamps": [int(est.used.sum()) for est in estimates],
                "uplift_frc": [est.uplift for est in estimates],
                "uplift_sigma_frc": [est.sigma_overall for est in estimates],
                "decision_sigma_frc": [est.decision_sigma for est in estimates],
                "chosen": [i == best for i in range(len(subsets))],
            }
        )
        return estimates[best], table

    def _labeled_rows(
        self,
        mi: MethodInput,
        *,
        wide: pd.DataFrame,
        test: str,
        used: npt.NDArray[np.bool_],
        rows: ToggleRowSets,
        label: npt.NDArray[np.float64],
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
        if "power" in self.conditions and np.isfinite(label).any():
            assert self.rated_power_kw is not None  # noqa: S101 - guaranteed by __post_init__
            bins = condition_bins("power", rated_power_kw=self.rated_power_kw)
            labeled["power_bin"] = _on_test_rows(np.asarray(pd.cut(label, bins=bins)))
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

    def _bins(self) -> list[float] | None:
        """Return the power bin edges when the method reports per bin; None otherwise."""
        if "power" not in self.conditions:
            return None
        assert self.rated_power_kw is not None  # noqa: S101 - guaranteed by __post_init__
        return condition_bins("power", rated_power_kw=self.rated_power_kw)

    def _check_pairing_gap(self, timebase: pd.Timedelta) -> None:
        gap = self.pairing_max_gap
        if gap is None:
            return
        if gap <= pd.Timedelta(0) or gap % timebase != pd.Timedelta(0):
            msg = (
                f"{self.name}: pairing_max_gap {gap} must be a positive whole multiple of the timebase "
                f"{timebase}; a shorter gap can never reach a row of the other segment."
            )
            raise ValueError(msg)

    def _reference_block(self, timebase: pd.Timedelta) -> pd.Timedelta:
        """Return ``reference_block`` after checking it tiles the timebase grid."""
        block = self.reference_block
        ratio = block / timebase
        if block <= pd.Timedelta(0) or ratio != round(ratio):
            msg = (
                f"{self.name}: reference_block {block} must be a positive whole multiple of the timebase "
                f"{timebase}; a block that does not tile the timebase grid cannot hold whole records."
            )
            raise ValueError(msg)
        return block

    def _reference_total(
        self,
        *,
        mode: str,
        wide: pd.DataFrame,
        refs: list[str],
        test_pw: npt.NDArray[np.float64],
        used: npt.NDArray[np.bool_],
        ids: npt.NDArray[np.int64] | None,
    ) -> npt.NDArray[np.float64]:
        """Return the per-row reference power for ``mode``; NaN on rows no surviving block covers.

        The block leg gives every used row of a block the block's mean test power, so within a block
        the on/off ratio cancels the block's wind level whatever the on/off count mix (the block mean
        carries the uplift at ``n_on / n`` strength in both denominators, a second-order effect).
        """
        if mode == _REFERENCE_LEG:
            return wide[refs].sum(axis=1).to_numpy(dtype=float)
        if ids is None or not used.any():
            return np.full(len(wide), np.nan)
        block_test = _block_sum(ids, test_pw, used)
        n_used = pd.Series(used.astype(float)).groupby(ids).sum()
        return _per_row(ids, block_test / n_used.reindex(block_test.index))

    def _selection(
        self,
        mi: MethodInput,
        *,
        wide: pd.DataFrame,
        test: str,
        refs: list[str],
        timebase: pd.Timedelta,
        rows: ToggleRowSets,
        ids: npt.NDArray[np.int64] | None = None,
        both_states: bool = True,
    ) -> _Selection:
        """Return the used-row mask after each selection stage, each a bool Series on ``wide.index``.

        ``ids`` are the reference-block ids (``None`` when there are no campaign rows to tile). The
        ``block`` stage drops every row of a block that carries a ``columns.exclude_block`` flag on
        any of the turbines (both legs) or, when ``both_states`` (the block leg), lacks either state.
        """
        turbines = [test, *refs]
        filtered = self._filtered_mask(mi, wide=wide, test=test, refs=refs, timebase=timebase)
        not_excluded = filtered & ~self._excluded(mi, turbines=turbines, index=wide.index)
        in_segment = rows.campaign_baseline | rows.upgraded
        kept = not_excluded.to_numpy()
        if self.pairing_max_gap is not None:
            paired = pair_within(
                wide.index,
                baseline=kept & rows.campaign_baseline,
                upgraded=kept & rows.upgraded,
                max_gap=self.pairing_max_gap,
            )
            kept = kept & (~in_segment | paired.baseline | paired.upgraded)
        paired_series = pd.Series(kept, index=wide.index)
        block_flag = self._excluded(mi, turbines=turbines, index=wide.index, role="exclude_block").to_numpy()
        if block_flag.any():
            if ids is None:
                msg = (
                    f"{self.name}: columns.exclude_block {self.columns.exclude_block!r} carries flags but there is "
                    f"no campaign row to tile into blocks."
                )
                raise ValueError(msg)
            kept = kept & ~np.isin(ids, np.unique(ids[block_flag]))
        if ids is not None and both_states:
            both = blocks_with_both_states(ids, baseline=kept & rows.campaign_baseline, upgraded=kept & rows.upgraded)
            kept = kept & (~in_segment | both)
        return _Selection(
            filtered=filtered,
            not_excluded=not_excluded,
            paired=paired_series,
            blocked=pd.Series(kept, index=wide.index),
        )

    def _selection_accounting(self, selection: _Selection, *, rows: ToggleRowSets) -> pd.DataFrame:
        """Rows kept per segment after each stage, warning when a stage treats the segments unevenly.

        ``kept_fraction`` is relative to the segment's rows; ``stage_kept_fraction`` to the previous
        stage, so it isolates which stage discriminates between segments.
        """
        stages_masks = (
            ("segment", None),
            ("filters", selection.filtered),
            ("exclude_row", selection.not_excluded),
            ("pairing", selection.paired),
            ("block", selection.blocked),
        )
        records = []
        for segment, in_segment in ((_BASELINE, rows.campaign_baseline), (_UPGRADED, rows.upgraded)):
            n_segment = int(in_segment.sum())
            previous = n_segment
            for stage, mask in stages_masks:
                n_kept = n_segment if mask is None else int((mask.to_numpy() & in_segment).sum())
                records.append(
                    {
                        "stage": stage,
                        "segment": segment,
                        "n_kept": n_kept,
                        "kept_fraction": n_kept / n_segment if n_segment else np.nan,
                        "stage_kept_fraction": n_kept / previous if previous else np.nan,
                    }
                )
                previous = n_kept
        accounting = pd.DataFrame(records)

        by_stage = accounting.pivot(  # noqa: PD010 - one row per (stage, segment), nothing to aggregate
            index="stage", columns="segment", values="stage_kept_fraction"
        )
        imbalance = (by_stage[_BASELINE] - by_stage[_UPGRADED]).abs()
        for stage, gap in imbalance[imbalance > self.segment_imbalance_warning].items():
            logger.warning(
                "%s: stage %r kept %.1f%% of baseline vs %.1f%% of upgraded rows (difference %.1f%% > %.1f%%)",
                self.name,
                stage,
                100 * by_stage.loc[stage, _BASELINE],
                100 * by_stage.loc[stage, _UPGRADED],
                100 * gap,
                100 * self.segment_imbalance_warning,
            )
        return accounting

    def _filtered_mask(
        self, mi: MethodInput, *, wide: pd.DataFrame, test: str, refs: list[str], timebase: pd.Timedelta
    ) -> pd.Series:
        """Timestamps at which the test turbine and every reference pass the same normal-operation filter.

        Returns a bool Series on ``wide.index``. One :class:`NormalOperationFilter` (the downtime +
        finite-power logic the power model uses; the stuck filter is left off here as the ratio sums
        raw power rather than fitting a model) is applied to each turbine's own rows, so a turbine
        that is down or has no power on either side of the ratio excludes the timestamp. The same
        filter on every turbine means any future change to it (e.g. the stuck filter) applies to
        the references exactly as to the test turbine.
        """
        normal = NormalOperationFilter(
            active_power_col=self.columns.active_power,
            availability_col=self.columns.availability,
            apply_stuck_filter=False,
        )
        keep = pd.Series(data=True, index=wide.index, dtype=bool)
        for turbine in (test, *refs):
            rows = mi.scada_df[mi.scada_df[mi.turbine_col] == turbine]
            keep &= normal.keep_mask(rows, timebase=timebase).reindex(wide.index, fill_value=False)
            # ``wide`` is the masked pivot: a cell the campaign context blanked must not count either.
            keep &= wide[turbine].notna()
        return keep

    def _excluded(
        self, mi: MethodInput, *, turbines: list[str], index: pd.DatetimeIndex, role: str = "exclude_row"
    ) -> pd.Series:
        """Boolean mask on *index*: timestamps at which any of ``turbines`` carries a caller-set ``role`` flag.

        Reads the ``columns.<role>`` column (``exclude_row`` or ``exclude_block``) of each listed
        turbine's rows (the test turbine and the references the mode uses), so a flagged reference
        row counts the way a reference outage does. Absent column or unset role -> nothing flagged.
        Reindex fills missing timestamps with ``False`` so an expanded index never becomes an
        exclusion. NaN raises rather than coercing: ``astype(bool)`` reads a missing flag as ``True``
        and drops the row, the opposite of the safe default.
        """
        col: str | None = getattr(self.columns, role)
        excluded = pd.Series(data=False, index=index, dtype=bool)
        if not col or col not in mi.scada_df.columns:
            return excluded
        for turbine in turbines:
            flags = mi.scada_df.loc[mi.scada_df[mi.turbine_col] == turbine, col]
            if flags.isna().any():
                msg = (
                    f"{role} column {col!r} has {int(flags.isna().sum())} NaN value(s) for turbine "
                    f"{turbine!r}; it must be boolean with no missing values (fill unknown rows with False explicitly)"
                )
                raise ValueError(msg)
            excluded |= flags.astype(bool).reindex(index, fill_value=False)
        return excluded

    def _write_outputs(
        self,
        mi: MethodInput,
        *,
        wide: pd.DataFrame,
        stats: pd.DataFrame,
        est: _Estimate,
        components: dict[str, _Estimate],
        timebase: pd.Timedelta,
        rows: ToggleRowSets,
        subsets: pd.DataFrame | None = None,
        n_refs_available: int | None = None,
    ) -> None:
        """Write the data-stats, results, per-bin, uncertainty, selection and reference-subsets CSVs and the plots."""
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
                    "n_refs": len(est.refs),
                    "n_refs_available": len(est.refs) if n_refs_available is None else n_refs_available,
                    "refs_used": ";".join(est.refs),
                    "ratio_baseline": est.rho_base,
                    "ratio_upgraded": est.rho_up,
                    "uplift_frc": est.uplift,
                    "uplift_sigma_frc": est.sigma_overall,
                    **{f"uplift_frc_{mode}": component.uplift for mode, component in components.items()},
                    **{f"uplift_sigma_frc_{mode}": component.sigma_overall for mode, component in components.items()},
                    **{f"decision_sigma_frc_{mode}": c.decision_sigma for mode, c in components.items()},
                    "block_hours": self.block_hours,
                    "n_resamples": self.n_resamples,
                    "n_used_timestamps_baseline": used_base,
                    "n_used_timestamps_upgraded": used_up,
                    "time_calculated": pd.Timestamp.utcnow(),
                }
            ]
        )
        results.to_csv(run_dir / f"{run_name}_results_{ts}.csv", index=False)

        if est.per_bin is not None:
            est.per_bin.to_csv(run_dir / f"{run_name}_by_power_bin_{ts}.csv", index=False)
        est.diagnostics.to_csv(run_dir / f"{run_name}_uncertainty_{ts}.csv", index=False)
        if subsets is not None:
            subsets.to_csv(run_dir / f"{run_name}_reference_subsets_{ts}.csv", index=False)
        # Every leg's row selection, so the blend's two selections can be told apart.
        pd.concat(
            [component.accounting.assign(component=mode) for mode, component in components.items()], ignore_index=True
        ).to_csv(run_dir / f"{run_name}_selection_{ts}.csv", index=False)

        if self.save_plots:
            _save_plots(
                run_dir / "plots",
                wide=wide,
                mi=mi,
                test=mi.test_wtg,
                used=est.used,
                ref_total=est.ref_total,
                reference_mode=est.mode,
                timebase=timebase,
                active_power_col=self.columns.active_power,
            )
            if est.per_bin is not None:
                _save_per_bin_plot(
                    run_dir / "plots" / stages.CONDITIONAL_UPLIFT / f"{mi.test_wtg}_per_bin_uplift.png",
                    per_bin=est.per_bin,
                    test=mi.test_wtg,
                    active_power_col=self.columns.active_power,
                )
            _save_pairing_gap_plot(
                run_dir / "plots" / stages.FILTER / f"{mi.test_wtg}_pairing_gap.png",
                index=wide.index,
                selection=est.selection,
                rows=rows,
                timebase=timebase,
                max_gap=self.pairing_max_gap,
                test=mi.test_wtg,
            )
            self._write_shared_diagnostics(
                mi, run_dir=run_dir, wide=wide, used=est.used, timebase=timebase, turbines=[mi.test_wtg, *est.refs]
            )

    def _write_shared_diagnostics(
        self,
        mi: MethodInput,
        *,
        run_dir: Path,
        wide: pd.DataFrame,
        used: np.ndarray,
        timebase: pd.Timedelta,
        turbines: list[str],
    ) -> None:
        """Emit the shared cross-method diagnostics (coverage/curves/histograms) and the run config."""
        # ``wide`` (a pivot) drops all-NaN timestamps, so align the masks to the full unique index
        # the DiagnosticContext uses (timestamps absent from ``wide`` are simply not used).
        index = pd.DatetimeIndex(pd.unique(mi.scada_df.index)).sort_values()
        used = pd.Series(used, index=wide.index).reindex(index, fill_value=False).to_numpy()
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
            # the exclusion alone, so the plots show what the flag removed that downtime did not
            excluded_ts=self._excluded(mi, turbines=turbines, index=index).to_numpy(),
        )
        write_common_diagnostics(ctx)
        params = {
            "active_power_col": self.columns.active_power,
            "availability_col": self.columns.availability,
            "pairing_max_gap": None if self.pairing_max_gap is None else str(self.pairing_max_gap),
            "reference_block": str(self.reference_block),
            "exclude_block_col": self.columns.exclude_block,
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


def _combine(a: _Estimate, b: _Estimate, *, upgraded: npt.NDArray[np.bool_], bins: list[float] | None) -> _Estimate:
    """Blend the two legs' estimates by minimum-variance weights, headline and per bin.

    The weights (and the correlation they need) come from the two legs' label-permuted bootstraps,
    cell by cell; the blend's sigma comes from the actual ones.

    The blend's used rows are the union of the components' (a row contributed to at least one
    estimate); its bin label is ``a``'s where ``a`` used the row and ``b``'s otherwise, so the two
    components' bins, which share edges, are merged by name, and each bin's ``n_records`` counts
    the used ``upgraded`` rows that merged label puts in it (what ``labeled_rows`` reproduces).
    Row-selection accounting, ``rho`` and the reference-side diagnostics are ``a``'s (the reference
    leg); ``b``'s selection is written to the selection CSV. The diagnostics carry both legs' rows and
    a ``combined`` row per cell with the blend's sigma, the weight on ``a`` and the correlation it used.
    """
    if a.boot is not None and b.boot is not None and a.boot.n_blocks != b.boot.n_blocks:
        # Both bootstraps come from the same method instance (same span, timebase, block length,
        # resample count and seed), so their draws pair up; a block-count mismatch means they do not.
        msg = f"component bootstraps drew different block counts ({a.boot.n_blocks} vs {b.boot.n_blocks})"
        raise ValueError(msg)

    def _resamples(boot: BootstrapResult | None, cell: str) -> npt.NDArray[np.float64]:
        return boot.resamples.get(cell, np.array([])) if boot is not None else np.array([])

    def _cell(cell: str, est_a: float, sig_a: float, est_b: float, sig_b: float) -> CombinedEstimate:
        weighting = (
            _cell_sigma(a.decision_boot, cell),
            _cell_sigma(b.decision_boot, cell),
            _resamples(a.decision_boot, cell),
            _resamples(b.decision_boot, cell),
        )
        return combine_estimates(
            (est_a, sig_a, _resamples(a.boot, cell)), (est_b, sig_b, _resamples(b.boot, cell)), weighting=weighting
        )

    overall = _cell(_OVERALL, a.uplift, a.sigma_overall, b.uplift, b.sigma_overall)
    blended_cells = {_OVERALL: overall}

    used = a.used | b.used
    label = np.where(a.used, a.label, b.label)
    per_bin = None
    if a.per_bin is not None and b.per_bin is not None and bins is not None:
        fa = a.per_bin.set_index(a.per_bin["condition_bin"].astype(str))
        fb = b.per_bin.set_index(b.per_bin["condition_bin"].astype(str))
        upgraded_bin = pd.Series(pd.cut(label[used & upgraded], bins=bins).astype(str))
        records = []
        for cell in fa.index:
            ra, rb = fa.loc[cell], fb.loc[cell]
            blend = _cell(cell, ra["p50_uplift"], ra["sigma_uplift"], rb["p50_uplift"], rb["sigma_uplift"])
            blended_cells[cell] = blend
            take = ra if np.isfinite(ra["p50_uplift"]) else rb
            records.append(
                {
                    "condition": ra["condition"],
                    "condition_bin": ra["condition_bin"],
                    "p50_uplift": blend.estimate,
                    "n_records": int((upgraded_bin == cell).sum()),
                    "sum_actual": take["sum_actual"],
                    "sum_counterfactual": take["sum_counterfactual"],
                    "sigma_uplift": blend.sigma,
                }
            )
        per_bin = pd.DataFrame(records)

    combined_rows = pd.DataFrame(
        [
            {
                "component": _COMBINED,
                "condition": _OVERALL if cell == _OVERALL else "power",
                "condition_bin": cell,
                "sigma": blend.sigma,
                f"weight_{a.mode}": blend.weight_a,
                "correlation": blend.correlation,
                f"decision_sigma_{a.mode}": _cell_sigma(a.decision_boot, cell),
                f"decision_sigma_{b.mode}": _cell_sigma(b.decision_boot, cell),
            }
            for cell, blend in blended_cells.items()
        ]
    )
    diagnostics = pd.concat([a.diagnostics, b.diagnostics, combined_rows], ignore_index=True)

    return _Estimate(
        mode=_COMBINED,
        refs=a.refs,
        selection=a.selection,
        used=used,
        ref_total=a.ref_total,
        label=label,
        rho_base=a.rho_base,
        rho_up=a.rho_up,
        uplift=overall.estimate,
        sigma_overall=overall.sigma,
        per_bin=per_bin,
        boot=None,
        accounting=a.accounting,
        diagnostics=diagnostics,
    )


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
                "sigma": cell_boot.sigma if cell_boot is not None else nan,
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
    ref_total: npt.NDArray[np.float64],
    timebase: pd.Timedelta,
    active_power_col: str,
) -> pd.DataFrame:
    """Build the per-segment (all/baseline/upgraded) diagnostics table."""
    test = mi.test_wtg
    test_pw = wide[test].to_numpy(dtype=float)
    n_turbines = wide.shape[1]
    timebase_hours = timebase / pd.Timedelta(hours=1)

    row_rows = resolve_toggle(mi.upgrade_timing, mi.scada_df.index)
    row_power = mi.scada_df[active_power_col].to_numpy(dtype=float)
    # Only the turbines the estimate used (the test and its chosen references) count as rows.
    in_wide = mi.scada_df[mi.turbine_col].isin(wide.columns).to_numpy()

    ts_baseline = toggle_rows.campaign_baseline
    row_baseline = row_rows.campaign_baseline & in_wide
    ts_masks = {"all": np.ones(len(wide), dtype=bool), "baseline": ts_baseline, "upgraded": toggle_rows.upgraded}
    row_masks = {"all": in_wide, "baseline": row_baseline, "upgraded": row_rows.upgraded & in_wide}

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
    ref_total: npt.NDArray[np.float64],
    reference_mode: str,
    timebase: pd.Timedelta,
    active_power_col: str,
) -> None:
    """Write the scatter, ratio-timeseries and used-coverage-timeseries diagnostic plots (by stage).

    ``used`` is the method's real downtime-filtered mask (test + every reference passing the
    availability/finite filter), so the scatter shows only the rows the estimate actually uses.
    The baseline is the strict campaign off-blocks the estimate used, so the plots never disagree
    with the headline.
    """
    toggle_rows = resolve_toggle(mi.upgrade_timing, wide.index)
    baseline_mask = toggle_rows.campaign_baseline
    test_pw = wide[test].to_numpy(dtype=float)
    ref_label = f"{reference_mode} reference {active_power_col}"
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
    ax.set_xlabel(f"{ref_label} [kW]")
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


def _save_pairing_gap_plot(
    path: Path,
    *,
    index: pd.DatetimeIndex,
    selection: _Selection,
    rows: ToggleRowSets,
    timebase: pd.Timedelta,
    max_gap: pd.Timedelta | None,
    test: str,
) -> None:
    """Histogram each used row's gap to the nearest used row of the other segment, before and after pairing.

    Gaps beyond the plotted range are piled into the last bin.
    """
    step_min = timebase.total_seconds() / 60.0
    cap_steps = max(24, 3 * round(max_gap / timebase)) if max_gap is not None else 24
    edges = (np.arange(cap_steps + 2) - 0.5) * step_min
    hours_per_row = timebase / pd.Timedelta(hours=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    text = []
    drawn = [("before pairing" if max_gap is not None else "used rows", selection.not_excluded, "C0")]
    if max_gap is not None:
        drawn.append(("after pairing", selection.paired, "C1"))
    for label, mask, color in drawn:
        kept = mask.to_numpy()
        gap_min = (
            gap_to_other_segment(index, baseline=kept & rows.campaign_baseline, upgraded=kept & rows.upgraded) / 60.0
        )
        gap_min = gap_min[~np.isnan(gap_min)]
        ax.hist(
            np.minimum(gap_min, cap_steps * step_min),
            bins=edges,
            histtype="stepfilled",
            alpha=0.45,
            color=color,
            label=label,
        )
        n_base = int((kept & rows.campaign_baseline).sum())
        n_up = int((kept & rows.upgraded).sum())
        text.append(f"{label}: {n_base * hours_per_row:.1f} h baseline, {n_up * hours_per_row:.1f} h upgraded")
    if max_gap is not None:
        ax.axvline(
            max_gap / pd.Timedelta(minutes=1),
            color="k",
            linestyle=":",
            label=f"pairing_max_gap {max_gap / pd.Timedelta(minutes=1):g} min",
        )
    else:
        text.append("no pairing filter configured")
    ax.text(0.98, 0.95, "\n".join(text), transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.set_xlabel(f"gap to nearest used row of the other segment [min] (last bin: >= {cap_steps * step_min:g})")
    ax.set_ylabel("used rows")
    ax.set_yscale("log")
    ax.set_title(f"{test}: gap from each used row to the opposite toggle state")
    ax.grid(visible=True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.83))
    fig.tight_layout()
    _save(fig, path)


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
