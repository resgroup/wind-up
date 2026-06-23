"""Driver: score the v0 binned baseline across all synthetic profiles on real Hill of Towie data.

Wires the open Hill of Towie SCADA through the full Issue 3 stack: load -> build the shared v0
context (metadata + ERA5) -> for each synthetic upgrade profile inject it, build the replicate
ensemble, run the **real** wind_up pre/post analysis per campaign via :class:`V0BinnedMethod`,
score against the injected truth, and save a per-profile leaderboard CSV, the tidy
per-replicate results, and a campaign-length curve PNG. An ``oracle`` anchor is scored
alongside so its ~0 error confirms the harness is wired correctly.

This is the baseline every new method must beat (Issue 3 "Done when"). It is heavy: each
``(replicate, campaign)`` is a full wind_up run, so the default sweep is a few dozen runs per
profile. Reduce ``n_replicates`` / ``campaign_months`` for a quicker look.

Run it::

    uv run python -m benchmarking.baselines.example_v0_study

The first run downloads and caches the Hill of Towie v2 year zips from Zenodo and the ERA5
reanalysis from Open-Meteo (needs the ``era5`` optional dependency group). Override the window,
output and cache directories via the ``main`` arguments or the ``WIND_UP_BENCHMARKING_*`` /
``WIND_UP_CACHE_DIR`` env vars.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.v0_binned import V0BinnedMethod
from benchmarking.harness import Method, StudyConfig, leaderboard, plot_campaign_curves, score_study
from benchmarking.harness.example_hot_study import OracleMethod
from benchmarking.synthetic.make_example_datasets import example_profiles
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

logger = logging.getLogger(__name__)

# The full stable, no-upgrade Hill of Towie window usable for this exercise: all of 2016-2020,
# well before the real T13 AeroUp (Sep 2021). Treatment start is drawn per replicate from
# 2018-01-01..2020-01-01. With min_pre_months=24 the earliest upgrade (2018-01-01) puts the
# baseline start at 2016-01-01 (the data start), giving v0 its full one-year detrend window even
# for the shortest campaign (whose detrend reaches ~21 months before the upgrade). Late upgrades
# near 2020 simply lose their longest campaign lengths (campaign_windows drops infeasible ones).
DEFAULT_START_DT = pd.Timestamp("2016-01-01", tz="UTC")
DEFAULT_END_DT_EXCL = pd.Timestamp("2021-01-01", tz="UTC")
DEFAULT_WTG_NUMBERS = [1, 3, 4, 7]
DEFAULT_TURBINE_SUBSET = [f"T{x:02d}" for x in DEFAULT_WTG_NUMBERS]
DEFAULT_TREATMENT_START_RANGE = (pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2020-01-01", tz="UTC"))
MIN_PRE_MONTHS = 24


def default_output_root() -> Path:
    """Return the directory the v0 study writes its outputs under.

    Overridable via ``WIND_UP_BENCHMARKING_OUTPUT_DIR``; defaults to
    ``~/temp/wind-up-benchmarking/baselines``.
    """
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "baselines"


def run_v0_study(
    base_scada: pd.DataFrame,
    *,
    profiles: dict[str, list],
    study: StudyConfig,
    out_root: str | Path | None = None,
    include_oracle: bool = True,
) -> pd.DataFrame:
    """Score the v0 baseline (and an oracle anchor) over ``profiles`` and save per-profile outputs.

    :param base_scada: wind-up-format real SCADA (all subset turbines), the no-upgrade baseline
    :param profiles: mapping of profile name -> list of upgrade callables to inject
    :param study: the replicate/campaign sweep configuration
    :param out_root: output directory; defaults to :func:`default_output_root`
    :param include_oracle: also score an oracle that returns the injected truth (sanity anchor)
    :return: the concatenated tidy per-replicate results across all profiles
    """
    out_dir = Path(out_root) if out_root is not None else default_output_root()
    out_dir.mkdir(parents=True, exist_ok=True)

    context = build_hot_v0_context()
    scratch_dir = out_dir / "windup_runs"

    all_results = []
    for profile_name, profile in profiles.items():
        methods: list[Method] = [V0BinnedMethod(context, scratch_dir=scratch_dir)]
        if include_oracle:
            methods.append(OracleMethod(base_scada))
        logger.info("Scoring profile %s with methods %s", profile_name, [m.name for m in methods])
        results = score_study(base_scada, profile=profile, methods=methods, study=study, profile_name=profile_name)
        summary = leaderboard(results)

        results.to_csv(out_dir / f"results_{profile_name}.csv", index=False)
        summary.to_csv(out_dir / f"leaderboard_{profile_name}.csv", index=False)
        plot_campaign_curves(summary, save_path=out_dir / f"campaign_curves_{profile_name}.png", title=profile_name)
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    combined_summary = leaderboard(combined)
    combined_summary.to_csv(out_dir / "leaderboard_all_profiles.csv", index=False)
    logger.info("Leaderboard (all profiles):\n%s", combined_summary.to_string(index=False))
    return combined


def main(
    *,
    out_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    start_dt: pd.Timestamp = DEFAULT_START_DT,
    end_dt_excl: pd.Timestamp = DEFAULT_END_DT_EXCL,
    wtg_numbers: list[int] | None = None,
    n_replicates: int = 4,
    campaign_months: list[int] | None = None,
) -> pd.DataFrame:
    """Run the v0 baseline study end-to-end on real Hill of Towie data and save outputs.

    :param out_root: output directory; defaults to :func:`default_output_root`
    :param data_dir: Hill of Towie data/cache dir; defaults to the source package default
    :param start_dt: inclusive UTC window start
    :param end_dt_excl: exclusive UTC window end
    :param wtg_numbers: turbine numbers to load; defaults to the stable south-west cluster
    :param n_replicates: ensemble size per profile (each replicate x campaign is a full v0 run)
    :param campaign_months: the campaign-length sweep grid, in months
    :return: the combined tidy results across all profiles
    """
    wtg_numbers = wtg_numbers if wtg_numbers is not None else DEFAULT_WTG_NUMBERS
    campaign_months = campaign_months if campaign_months is not None else [3, 6, 9, 12]
    logger.info("Loading Hill of Towie SCADA %s..%s for turbines %s", start_dt, end_dt_excl, wtg_numbers)
    scada_df, _metadata_df = load_hot_scada(
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        wtg_numbers=wtg_numbers,
        data_dir=Path(data_dir) if data_dir is not None else None,
    )
    study = StudyConfig(
        mode="prepost",
        turbine_subset=DEFAULT_TURBINE_SUBSET,
        treatment_start_range=DEFAULT_TREATMENT_START_RANGE,
        min_pre_months=MIN_PRE_MONTHS,
        campaign_months=campaign_months,
        n_replicates=n_replicates,
        seed=0,
    )
    return run_v0_study(scada_df, profiles=example_profiles(), study=study, out_root=out_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
