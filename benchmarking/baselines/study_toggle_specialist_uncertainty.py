"""Measure how well ``toggle_specialist``'s reported 1-sigma matches its actual error.

The P50 studies ask how close the estimate lands. This one asks whether the method was **right
about how close it would land** — the quality test of an uncertainty is coverage against ground
truth: ~68.3% of estimates should sit within 1 sigma of truth. Note that this scores sigma against
the *total* deviation from truth, bias included; a block bootstrap sees only sampling variance, so
where the method is biased the sigma will under-cover, and that is a finding rather than an
unfairness (see :mod:`benchmarking.harness.calibration`).

Three design points, each with evidence behind it in F28-F30:

**Replicates, not cells, are the evidence.** The three profiles reuse the same
``(turbine, treatment_start)`` draws, so their errors correlate 0.977-0.995; campaign lengths are
prefix-nested; long campaigns overlap each other. Coverage SE is therefore quoted on the independent
draw count, never on the row count — see :func:`~benchmarking.harness.coverage_standard_error`.

**Block length is swept as a method variant.** Uncertainty never changes uplift and an estimate is
cheap, so one method per block length (:data:`BLOCK_HOURS_GRID`) gets the sweep from the existing
multi-method seam. There is no sigma-vs-L plateau to read here; coverage against truth decides.

**Memory forces a streaming loop.** A replicate carries a ``synthetic_df`` and an ``original_df``
(~0.5 GB), so ``score_study``'s materialised ensemble would be OOM-killed at this replicate count.
This drives a replicate-outer loop over :func:`~benchmarking.harness.iter_replicates`, reusing
:func:`~benchmarking.harness.score_one` so the truth alignment is shared rather than reimplemented.

``cases.csv`` is the point of the run: every scored cell with its estimate, truth, sigma and record
counts, so a further uncertainty component can be fitted offline without re-running the sweep.

Run from the repo root::

    uv run python -m benchmarking.baselines.study_toggle_specialist_uncertainty

Use ``--replicates 8 --profiles cp_0pct`` for a fast smoke run.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.baselines.example_prepost_study import (
    DEFAULT_END_DT_EXCL,
    DEFAULT_START_DT,
    DEFAULT_TURBINE_SUBSET,
    DEFAULT_WTG_NUMBERS,
)
from benchmarking.baselines.example_toggle_study import DEFAULT_TOGGLE_PERIOD
from benchmarking.baselines.study_toggle_methods_compare import TOGGLE_PROFILES
from benchmarking.baselines.toggle_specialist import DEFAULT_BLOCK_HOURS, ToggleSpecialistMethod
from benchmarking.harness import (
    TARGET_COVERAGE_1SIGMA,
    StudyConfig,
    campaign_windows,
    coverage_standard_error,
    iter_replicates,
    score_one,
    summarize_calibration,
    truth_mask,
)
from benchmarking.synthetic import HOT_COLUMNS, HOT_RATED_POWER_KW
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

logger = logging.getLogger(__name__)

CAMPAIGN_WEEKS = [1, 2, 4, 8, 26, 52]
N_REPLICATES = 64
SEED = 0
# toggle_specialist drops pre-campaign rows (`restrict_to_campaign`), so a pre-campaign baseline buys
# it nothing and `min_pre_months` only costs start-range span. Widening the range to the whole dataset
# doubles the non-overlapping positions available to the long campaigns, which is where independent
# evidence is scarcest: a 52-week campaign is 364d, so a 730d range holds only ~2 of them.
TREATMENT_START_RANGE = (DEFAULT_START_DT, pd.Timestamp("2020-01-01", tz="UTC"))
MIN_PRE_MONTHS_TOGGLE = 0
# Brackets the default on both sides, because coverage degrades in both directions (F28) and a grid
# that only reached upwards would hide half of that. The bottom end (1h ~ 1.5 cycles of a 40-minute
# toggle) over-covers; the top end starves the bootstrap of distinct blocks and biases sigma low.
# 96h is dropped: at 1 week it is ~2 blocks and its verdict (coverage 0.438) is already recorded.
BLOCK_HOURS_GRID = [1.0, 2.0, 3.0, 6.0, 12.0, 24.0, 48.0]

_LENGTH_COL = "campaign_weeks"
_DEFAULT_OUTPUT_DIR = Path.home() / "temp" / "wind-up-benchmarking" / "toggle_specialist_uncertainty"
_PP = 100.0  # fraction -> percentage points
# Record-count buckets for the per-bin coverage-vs-count read, roughly by decade because the
# hypothesis is about order of magnitude ("sparse bins fail"), not about a particular count.
_COUNT_BIN_EDGES = (0, 30, 100, 300, 1000, 3000, 10000)


def uncertainty_study(n_replicates: int) -> StudyConfig:
    """Return the study every run scores: a toggle grid from one week to a year."""
    return StudyConfig(
        mode="toggle",
        turbine_subset=DEFAULT_TURBINE_SUBSET,
        treatment_start_range=TREATMENT_START_RANGE,
        min_pre_months=MIN_PRE_MONTHS_TOGGLE,
        campaign_weeks=CAMPAIGN_WEEKS,
        toggle_period=DEFAULT_TOGGLE_PERIOD,
        n_replicates=n_replicates,
        seed=SEED,
    )


def independent_draws(campaign_weeks: int, *, n_replicates: int, n_turbines: int = len(DEFAULT_TURBINE_SUBSET)) -> int:
    """Roughly how many *independent* cases a campaign length really has.

    Replicates are only independent while their windows do not overlap. A 52-week campaign drawn from
    a ~4-year start range has ~4 non-overlapping positions, so 64 replicates carry ~4 x n_turbines
    draws, not 64 — quoting coverage SE on 64 there would understate it ~2x. Capped at
    ``n_replicates``, since drawing more positions than replicates buys nothing.
    """
    lo, hi = TREATMENT_START_RANGE
    positions = max((hi - lo) / pd.Timedelta(weeks=campaign_weeks), 1.0)
    return int(min(n_replicates, round(positions * n_turbines)))


def build_methods(block_hours_grid: list[float], out_dir: Path | None = None) -> list[ToggleSpecialistMethod]:
    """One ``toggle_specialist`` per block length, identical in every other respect.

    They therefore produce identical uplifts and differ only in sigma, which is what makes the
    block-length sweep a method comparison the existing harness already knows how to run.

    ``out_dir`` is left ``None`` by default: the sweep would otherwise write thousands of per-run
    diagnostic folders nothing reads, and its actual output is ``cases.csv``.
    """
    return [
        ToggleSpecialistMethod(
            columns=HOT_COLUMNS,
            name=f"toggle_specialist_bl{block_hours:g}",
            conditions=("power",),
            rated_power_kw=HOT_RATED_POWER_KW,
            block_hours=block_hours,
            out_dir=out_dir,
        )
        for block_hours in block_hours_grid
    ]


def _select_profiles(requested: list[str] | None) -> dict[str, list]:
    """Return the profiles to score: all when ``requested`` is ``None``, else the named subset."""
    if requested is None:
        return TOGGLE_PROFILES
    unknown = [name for name in requested if name not in TOGGLE_PROFILES]
    if unknown:
        msg = f"unknown profile(s) {unknown}; available: {sorted(TOGGLE_PROFILES)}"
        raise ValueError(msg)
    return {name: TOGGLE_PROFILES[name] for name in requested}


def run_sweep(
    scada_df: pd.DataFrame,
    *,
    study: StudyConfig,
    methods: list[ToggleSpecialistMethod],
    profiles: dict[str, list],
) -> pd.DataFrame:
    """Score every method over every profile, streaming replicates to bound memory.

    Mirrors :func:`~benchmarking.harness.score_study` but loops replicate-outer, so exactly one
    replicate is alive at a time. Every method still sees the identical ``MethodInput`` for an
    instance (it is built once per instance and shared), so the cross-method fairness that matters
    for the block-length comparison is preserved.
    """
    data_start, data_end = scada_df.index.min(), scada_df.index.max()
    rows: list[dict[str, object]] = []
    for profile_name, profile in profiles.items():
        for replicate in iter_replicates(scada_df, profile=profile, study=study):
            windows = campaign_windows(
                replicate.treatment_start,
                min_pre_months=study.min_pre_months,
                campaign_months=study.campaign_months,
                campaign_weeks=study.campaign_weeks,
                data_start=data_start,
                data_end=data_end,
            )
            for window in windows:
                mask = truth_mask(replicate, window)
                truth = replicate.true_uplift(mask=mask).overall
                for method in methods:
                    rows.extend(
                        score_one(
                            method,
                            replicate=replicate,
                            window=window,
                            truth=truth,
                            mask=mask,
                            profile_name=profile_name,
                        )
                    )
            logger.info("scored %s replicate %d (%d rows so far)", profile_name, replicate.replicate_id, len(rows))
    frame = pd.DataFrame(rows)
    return frame.assign(block_hours=frame["method"].map(_block_hours_of))


def _block_hours_of(method_name: str) -> float:
    """Recover a variant's block length from its name (``toggle_specialist_bl48`` -> 48.0)."""
    return float(method_name.rsplit("_bl", 1)[1])


def calibration_tables(cases: pd.DataFrame, *, n_replicates: int) -> dict[str, pd.DataFrame]:
    """Reduce the scored cases to the calibration reads worth looking at.

    Headline and per-bin are split because they fail for different reasons. The by-length table also
    carries its own ``n_independent`` / ``coverage_se``, since long campaigns overlap and are far
    weaker evidence than their row count suggests (:func:`independent_draws`).
    """
    headline = cases[cases["condition"] == "overall"]
    per_bin = cases[cases["condition"] != "overall"]
    by_length = summarize_calibration(headline, group_keys=["block_hours", _LENGTH_COL])
    # Per-length SE, because the independent-draw count collapses as campaigns lengthen and overlap:
    # a flat SE would make the 52-week reads look far firmer than they are.
    draws = by_length[_LENGTH_COL].map(lambda w: independent_draws(int(w), n_replicates=n_replicates))
    by_length = by_length.assign(n_independent=draws, coverage_se=draws.map(coverage_standard_error))
    return {
        "headline_by_block": summarize_calibration(headline, group_keys=["block_hours"]),
        "headline_by_block_and_length": by_length,
        "headline_by_block_and_profile": summarize_calibration(headline, group_keys=["block_hours", "profile"]),
        "per_bin_by_block": summarize_calibration(per_bin, group_keys=["block_hours"]),
        "per_bin_by_block_and_bin": summarize_calibration(per_bin, group_keys=["block_hours", "condition_bin"]),
        "per_bin_by_block_and_length": summarize_calibration(per_bin, group_keys=["block_hours", _LENGTH_COL]),
    }


def plot_results(cases: pd.DataFrame, tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Write the four plots that carry the findings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    focus = _focus_block_hours(cases)
    _plot_coverage_by_length(tables["headline_by_block_and_length"], out_dir / "coverage_by_campaign_length.png")
    _plot_sigma_plateau(cases, out_dir / "sigma_vs_block_length.png")
    _plot_error_vs_sigma(cases, out_dir / "error_vs_sigma.png", block_hours=focus)
    _plot_coverage_vs_count(cases, out_dir / "coverage_vs_record_count.png", block_hours=focus)


def _focus_block_hours(cases: pd.DataFrame) -> float:
    """Return the block length the per-case plots show: the method default, else the nearest swept.

    ``--block-hours`` is a free grid, so the default need not be in it. Picking the nearest length
    actually present keeps those plots meaningful for any grid, rather than silently emptying them.
    """
    present = np.sort(cases["block_hours"].unique())
    if DEFAULT_BLOCK_HOURS in present:
        return DEFAULT_BLOCK_HOURS
    return float(present[np.argmin(np.abs(present - DEFAULT_BLOCK_HOURS))])


def _plot_coverage_by_length(table: pd.DataFrame, path: Path) -> None:
    """Headline coverage against campaign length, one line per block length."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for block_hours, group in table.groupby("block_hours"):
        ax.plot(group[_LENGTH_COL], group["coverage_1sigma"], marker="o", label=f"{block_hours:g}h")
    ax.axhline(TARGET_COVERAGE_1SIGMA, color="k", linestyle="--", label="target 0.683")
    ax.set_xlabel("campaign length [weeks]")
    ax.set_ylabel("coverage at 1 sigma")
    ax.set_title("Headline coverage vs campaign length (below the line = sigma too small)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(visible=True, alpha=0.3)
    ax.legend(title="block length")
    _save(fig, path)


def _plot_sigma_plateau(cases: pd.DataFrame, path: Path) -> None:
    """Mean headline sigma against block length, one line per campaign length.

    There is no plateau to read here (F28): the curve is flat to falling. The measured RMS error is
    drawn alongside as the level sigma should reach, which makes the long-block collapse legible.
    """
    headline = cases[cases["condition"] == "overall"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for length, group in headline.groupby(_LENGTH_COL):
        by_block = group.groupby("block_hours")
        line = ax.plot(
            by_block["sigma"].mean().index,
            by_block["sigma"].mean().to_numpy() * _PP,
            marker="o",
            label=f"{length}w sigma",
        )
        rms = float(np.sqrt(np.mean(group["signed_error"].dropna().to_numpy() ** 2))) * _PP
        ax.axhline(rms, color=line[0].get_color(), linestyle=":", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("block length [h] (log scale)")
    ax.set_ylabel("mean sigma [pp]")
    ax.set_title("Sigma vs block length; dotted = that campaign's actual RMS error (the level to reach)")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    _save(fig, path)


def _plot_error_vs_sigma(cases: pd.DataFrame, path: Path, *, block_hours: float) -> None:
    """Absolute error against reported sigma for the headline, at one block length."""
    focus = cases[(cases["condition"] == "overall") & (cases["block_hours"] == block_hours)]
    if focus.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for length, group in focus.groupby(_LENGTH_COL):
        ax.scatter(group["sigma"] * _PP, group["signed_error"].abs() * _PP, s=14, alpha=0.6, label=f"{length}w")
    lim = float(np.nanmax([focus["sigma"].max(), focus["signed_error"].abs().max()])) * _PP * 1.05
    ax.plot([0, lim], [0, lim], color="k", linestyle="--", linewidth=1, label="|error| = sigma")
    ax.set_xlabel("reported sigma [pp]")
    ax.set_ylabel("|signed error| [pp]")
    ax.set_title(f"Headline |error| vs reported sigma ({block_hours:g}h blocks); ~68% should fall below the line")
    ax.grid(visible=True, alpha=0.3)
    ax.legend(title="campaign")
    _save(fig, path)


def _plot_coverage_vs_count(cases: pd.DataFrame, path: Path, *, block_hours: float) -> None:
    """Per-bin coverage against the bin's record count, at one block length.

    The hypothesis this plot exists to test: the bootstrap holds up where a bin is well populated
    and fails where it is not, which would make record count the covariate of a further term.
    """
    per_bin = cases[(cases["condition"] != "overall") & (cases["block_hours"] == block_hours)].copy()
    usable = per_bin[np.isfinite(per_bin["sigma"]) & (per_bin["sigma"] > 0) & np.isfinite(per_bin["signed_error"])]
    if usable.empty:
        return
    # Decade-ish edges, clipped to the counts actually present: a short or single-profile run does
    # not reach the upper decades, and a fixed edge above the data makes pd.cut non-monotonic.
    largest = int(usable["n_upgraded_records"].max())
    edges = [e for e in _COUNT_BIN_EDGES if e < largest] + [largest + 1]
    usable = usable.assign(
        count_bin=pd.cut(usable["n_upgraded_records"], bins=edges),
        covered=(usable["signed_error"] / usable["sigma"]).abs() <= 1.0,
    )
    grouped = usable.groupby("count_bin", observed=True)
    coverage = grouped["covered"].mean()
    counts = grouped.size()

    fig, (ax, ax_n) = plt.subplots(2, 1, sharex=True, figsize=(9, 7), height_ratios=[2, 1])
    x = np.arange(len(coverage))
    ax.plot(x, coverage.to_numpy(), marker="o", color="C1")
    ax.axhline(TARGET_COVERAGE_1SIGMA, color="k", linestyle="--", label="target 0.683")
    ax.set_ylabel("coverage at 1 sigma")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Per-bin coverage vs bin record count ({block_hours:g}h blocks)")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    ax_n.bar(x, counts.to_numpy(), color="C0", alpha=0.7)
    ax_n.set_ylabel("cases")
    ax_n.set_xlabel("upgraded records in bin")
    ax_n.set_xticks(x)
    ax_n.set_xticklabels([str(c) for c in coverage.index], rotation=20, ha="right", fontsize=8)
    ax_n.grid(visible=True, alpha=0.3)
    _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> None:
    """Write a figure to ``path`` (creating its folder) and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _log_tables(tables: dict[str, pd.DataFrame], *, n_replicates: int) -> None:
    """Log every calibration table, with the coverage target and per-length standard errors."""
    per_length = ", ".join(
        f"{w}w ~{independent_draws(w, n_replicates=n_replicates)} draws "
        f"(SE {coverage_standard_error(independent_draws(w, n_replicates=n_replicates)):.3f})"
        for w in CAMPAIGN_WEEKS
    )
    logger.info(
        "Target coverage %.3f. Row counts below are cells, NOT independent samples: profiles share "
        "campaign windows, campaign lengths are prefix-nested, and long campaigns overlap each other. "
        "Independent draws per campaign length: %s.",
        TARGET_COVERAGE_1SIGMA,
        per_length,
    )
    for name, table in tables.items():
        logger.info("%s:\n%s", name, table.round(4).to_string(index=False))


def main() -> None:
    """Score the block-length variants over the grid, then report and plot the calibration."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR, help="where outputs are written")
    parser.add_argument(
        "--profiles", nargs="+", choices=sorted(TOGGLE_PROFILES), default=None, help="restrict to a profile subset"
    )
    parser.add_argument("--replicates", type=int, default=N_REPLICATES, help="replicate ensemble size")
    parser.add_argument(
        "--block-hours",
        nargs="+",
        type=float,
        default=BLOCK_HOURS_GRID,
        help="block lengths to sweep, one method variant each",
    )
    parser.add_argument(
        "--save-run-dirs",
        action="store_true",
        help="also write each estimate's per-run diagnostic folder (thousands of them; off by default)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    scada_df, _ = load_hot_scada(
        start_dt=DEFAULT_START_DT,
        end_dt_excl=DEFAULT_END_DT_EXCL,
        wtg_numbers=DEFAULT_WTG_NUMBERS,
        wtg_names=DEFAULT_TURBINE_SUBSET,
    )
    study = uncertainty_study(args.replicates)
    methods = build_methods(args.block_hours, out_dir=output_dir / "runs" if args.save_run_dirs else None)
    profiles = _select_profiles(args.profiles)
    logger.info(
        "Sweeping %d replicates x %d profiles x %d campaign lengths x %d block lengths = %d estimates",
        args.replicates,
        len(profiles),
        len(CAMPAIGN_WEEKS),
        len(methods),
        args.replicates * len(profiles) * len(CAMPAIGN_WEEKS) * len(methods),
    )

    cases = run_sweep(scada_df, study=study, methods=methods, profiles=profiles)
    cases_path = output_dir / "cases.csv"
    cases.to_csv(cases_path, index=False)
    logger.info("Wrote %d scored cells to %s", len(cases), cases_path)

    tables = calibration_tables(cases, n_replicates=args.replicates)
    for name, table in tables.items():
        table.to_csv(output_dir / f"calibration_{name}.csv", index=False)
    _log_tables(tables, n_replicates=args.replicates)
    plot_results(cases, tables, output_dir / "plots")
    logger.info("All done. Outputs under %s", output_dir)


if __name__ == "__main__":
    main()
