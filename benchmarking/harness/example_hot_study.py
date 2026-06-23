"""Driver: run the P50 evaluation harness end-to-end on real Hill of Towie data.

A runnable, inspectable companion to ``tests/.../test_hot_end_to_end.py`` (which saves to a
pytest ``tmp_path`` and then throws it away). This wires the open Hill of Towie SCADA through
the full harness — load -> inject a constant-Cp upgrade -> replicate ensemble -> campaign
sweep -> score -> leaderboard -> plot — and saves the leaderboard CSV, the tidy per-replicate
results and the campaign-length curve PNG to a directory you can open.

The "methods" scored here are illustrative stand-ins (no real estimator ships in this phase):

- ``oracle`` returns the injected truth, so its error is ~0 at every campaign length;
- ``biased`` adds a fixed offset, so its error sits at that offset;
- ``realistic`` adds noise that shrinks with campaign length, so its spread band narrows as
  the campaign grows — the precision-vs-data-volume effect the campaign sweep exists to show.

Run it::

    uv run python -m benchmarking.harness.example_hot_study

First run downloads and caches the Hill of Towie v2 year zips from Zenodo (a few GB; a
12-month baseline plus a 12-month campaign spans 2016..2018). Override the window, output and
cache directories via the ``main`` arguments or the ``WIND_UP_BENCHMARKING_*`` env vars.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarking.harness import (
    Method,
    MethodInput,
    MethodOutput,
    StudyConfig,
    leaderboard,
    plot_campaign_curves,
    score_study,
)
from benchmarking.synthetic import ConstantCpChange, treated_mask
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada
from wind_up.constants import DataColumns

logger = logging.getLogger(__name__)

# A stable, no-upgrade Hill of Towie window wide enough for a 12-month baseline plus a
# 12-month campaign with the changeover at the 2017 new year. All of 2016-2020 was previously
# confirmed stable for these turbines (well before the real T13 AeroUp, installed Sep 2021).
DEFAULT_START_DT = pd.Timestamp("2016-01-01", tz="UTC")
DEFAULT_END_DT_EXCL = pd.Timestamp("2018-02-01", tz="UTC")
# The stable south-west turbines used by the end-to-end test (T01, T03, T04, T07).
DEFAULT_WTG_NUMBERS = [1, 3, 4, 7]
DEFAULT_TURBINE_SUBSET = ["T01", "T03", "T04", "T07"]


def default_output_root() -> Path:
    """Return the directory the harness example writes its outputs under.

    Overridable via the ``WIND_UP_BENCHMARKING_OUTPUT_DIR`` environment variable; defaults to
    ``~/temp/wind-up-benchmarking/harness``.
    """
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "harness"


def _oracle_overall_uplift(mi: MethodInput, original_df: pd.DataFrame) -> float:
    """Energy-ratio uplift over the treated test-turbine rows in ``mi``'s window.

    Compares the (upgraded) synthetic power against the original no-upgrade power over the
    exact same treated records, so it recovers the injected truth. The illustrative methods
    below wrap this to drive bias and spread.
    """
    syn = mi.scada_df
    test_rows = syn[syn[DataColumns.turbine_name] == mi.test_wtg]
    treated = treated_mask(test_rows.index, mi.upgrade_timing)
    treated_rows = test_rows[treated]

    syn_power = treated_rows[DataColumns.active_power_mean].to_numpy(dtype=float)
    orig_test = original_df[original_df[DataColumns.turbine_name] == mi.test_wtg]
    orig_power = orig_test.loc[treated_rows.index, DataColumns.active_power_mean].to_numpy(dtype=float)

    finite = np.isfinite(syn_power) & np.isfinite(orig_power)
    denom = orig_power[finite].sum()
    return syn_power[finite].sum() / denom - 1.0 if denom else float("nan")


class OracleMethod:
    """Returns the injected truth; its signed error is ~0 at every campaign length."""

    def __init__(self, original_df: pd.DataFrame, *, name: str = "oracle") -> None:
        """Store the no-upgrade baseline used to recover the injected truth."""
        self._original = original_df
        self.name = name

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Return the injected truth as the P50 estimate."""
        return MethodOutput(p50_overall=_oracle_overall_uplift(mi, self._original))


class BiasedMethod:
    """Returns the truth plus a fixed offset; its error sits at that offset."""

    def __init__(self, original_df: pd.DataFrame, *, offset: float, name: str = "biased") -> None:
        """Store the baseline and the fixed offset added to every estimate."""
        self._original = original_df
        self._offset = offset
        self.name = name

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Return the injected truth plus the fixed offset."""
        return MethodOutput(p50_overall=_oracle_overall_uplift(mi, self._original) + self._offset)


class ShrinkingNoiseMethod:
    """Returns the truth plus noise whose size shrinks with campaign length.

    A crude stand-in for a real estimator: more campaign data -> a tighter estimate. The noise
    sigma scales as ``base_sigma / sqrt(campaign_months)``, and the draw is seeded
    deterministically from ``(test turbine, treatment start, campaign length)`` so each
    replicate gets its own value while the whole study stays reproducible. Across the ensemble
    this makes the plotted spread band narrow as the campaign grows.
    """

    def __init__(self, original_df: pd.DataFrame, *, base_sigma: float = 0.03, name: str = "realistic") -> None:
        """Store the baseline and the base noise level scaled by campaign length."""
        self._original = original_df
        self._base_sigma = base_sigma
        self.name = name

    def estimate(self, mi: MethodInput) -> MethodOutput:
        """Return the truth plus a deterministic, campaign-length-shrinking noise draw."""
        truth = _oracle_overall_uplift(mi, self._original)
        treatment_start = mi.upgrade_timing  # prepost: a Timestamp
        post_days = max((mi.scada_df.index.max() - treatment_start).days, 1)
        months = post_days / 30.4
        sigma = self._base_sigma / np.sqrt(months)
        wtg_term = sum(ord(c) for c in mi.test_wtg) * 1000 + round(months)
        seed = (int(pd.Timestamp(treatment_start).value) % (2**32)) ^ wtg_term
        rng = np.random.default_rng(seed)
        return MethodOutput(p50_overall=truth + float(rng.normal(0.0, sigma)))


def run_example_study(
    scada_df: pd.DataFrame,
    *,
    out_root: str | Path | None = None,
    turbine_subset: list[str] | None = None,
    treatment_start_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    min_pre_months: int = 12,
    campaign_months: list[int] | None = None,
    n_replicates: int = 6,
    delta: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Score the illustrative methods on a constant-Cp study and save inspectable outputs.

    :param scada_df: wind-up-format real SCADA (all subset turbines), the no-upgrade baseline
    :param out_root: output directory; defaults to :func:`default_output_root`
    :param turbine_subset: turbines kept in the study (one drawn as test per replicate)
    :param treatment_start_range: ``(earliest, latest)`` changeover to draw from
    :param min_pre_months: fixed baseline length before the changeover, in months
    :param campaign_months: the campaign-length sweep grid, in months
    :param n_replicates: ensemble size (drives the spread estimate)
    :param delta: the injected constant-Cp uplift fraction
    :param seed: top-level seed for reproducibility
    :return: the leaderboard summary frame
    """
    out_dir = Path(out_root) if out_root is not None else default_output_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    turbine_subset = turbine_subset if turbine_subset is not None else DEFAULT_TURBINE_SUBSET
    if treatment_start_range is None:
        treatment_start_range = (pd.Timestamp("2017-01-01", tz="UTC"), pd.Timestamp("2017-01-15", tz="UTC"))
    campaign_months = campaign_months if campaign_months is not None else [3, 6, 9, 12]

    study = StudyConfig(
        mode="prepost",
        turbine_subset=turbine_subset,
        treatment_start_range=treatment_start_range,
        min_pre_months=min_pre_months,
        campaign_months=campaign_months,
        n_replicates=n_replicates,
        seed=seed,
    )
    methods: list[Method] = [
        OracleMethod(scada_df),
        BiasedMethod(scada_df, offset=0.02),
        ShrinkingNoiseMethod(scada_df, base_sigma=0.03),
    ]

    profile_name = f"constant_cp_{delta:.0%}".replace("%", "pct")
    logger.info(
        "Scoring %d methods over %d replicates x campaigns %s (profile %s, true uplift %+.1f%%)",
        len(methods),
        n_replicates,
        campaign_months,
        profile_name,
        delta * 100,
    )
    results = score_study(
        scada_df, profile=[ConstantCpChange(delta=delta)], methods=methods, study=study, profile_name=profile_name
    )
    summary = leaderboard(results)

    results_path = out_dir / "results_tidy.csv"
    summary_path = out_dir / "leaderboard.csv"
    plot_path = out_dir / "campaign_curves.png"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_campaign_curves(
        summary,
        save_path=plot_path,
        title=f"Hill of Towie {profile_name} (Cp delta {delta:+.1%}, {n_replicates} replicates)",
    )

    logger.info("Saved leaderboard -> %s", summary_path)
    logger.info("Saved tidy results -> %s", results_path)
    logger.info("Saved campaign-length curve -> %s", plot_path)
    return summary


def main(
    *,
    out_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    start_dt: pd.Timestamp = DEFAULT_START_DT,
    end_dt_excl: pd.Timestamp = DEFAULT_END_DT_EXCL,
    wtg_numbers: list[int] | None = None,
) -> pd.DataFrame:
    """Run the harness end-to-end on real Hill of Towie data and save inspectable outputs.

    Downloads (and caches) the open Hill of Towie SCADA for a stable, no-upgrade window, then
    scores the illustrative methods over a campaign-length sweep and writes the leaderboard,
    the tidy results and the campaign-length curve under ``out_root``.

    :param out_root: output directory; defaults to :func:`default_output_root`
    :param data_dir: Hill of Towie data/cache dir; defaults to the package default
    :param start_dt: inclusive UTC window start
    :param end_dt_excl: exclusive UTC window end
    :param wtg_numbers: turbine numbers to load; defaults to the stable south-west cluster
    :return: the leaderboard summary frame
    """
    wtg_numbers = wtg_numbers if wtg_numbers is not None else DEFAULT_WTG_NUMBERS
    logger.info("Loading Hill of Towie SCADA %s..%s for turbines %s", start_dt, end_dt_excl, wtg_numbers)
    scada_df, _metadata_df = load_hot_scada(
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        wtg_numbers=wtg_numbers,
        data_dir=Path(data_dir) if data_dir is not None else None,
    )
    summary = run_example_study(scada_df, out_root=out_root)
    logger.info("Leaderboard:\n%s", summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
