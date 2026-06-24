"""Driver: score v0 and the naive ratio method on a TOGGLE campaign on real Hill of Towie data.

Mirrors :mod:`benchmarking.baselines.example_prepost_study` but runs in toggle mode: a fast
20-min-on / 20-min-off schedule (``toggle_period = 40min``) and a single 3% constant-Cp upgrade.
``V0BinnedMethod`` runs wind_up's native toggle assessment; ``NaiveRatioMethod`` splits on/off
via the shared toggle mask. An oracle anchor is scored too, so its ~0 error confirms the
toggle harness path is wired correctly.

Toggle is where the naive ratio method should shine: interleaved on/off blocks share a wind
distribution, so it carries little covariate-shift bias -- a useful contrast with its prepost
behaviour in the v0 study.

Run it::

    uv run python -m benchmarking.baselines.example_toggle_study

The first run downloads and caches the Hill of Towie v2 year zips from Zenodo and the ERA5
reanalysis from Open-Meteo (needs the ``era5`` optional dependency group).
"""

from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path

import pandas as pd

from benchmarking.baselines.example_prepost_study import (
    DEFAULT_END_DT_EXCL,
    DEFAULT_START_DT,
    DEFAULT_TREATMENT_START_RANGE,
    DEFAULT_TURBINE_SUBSET,
    DEFAULT_WTG_NUMBERS,
    MIN_PRE_MONTHS,
    save_per_method_curve,
)
from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.naive_ratio import NaiveRatioMethod
from benchmarking.baselines.v0_binned import V0BinnedMethod
from benchmarking.harness import Method, StudyConfig, leaderboard, plot_campaign_curves, score_study
from benchmarking.harness.example_hot_study import OracleMethod
from benchmarking.synthetic import ConstantCpChange
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

logger = logging.getLogger(__name__)

# 20 minutes on, 20 minutes off -> a 40-minute on/off cycle.
DEFAULT_TOGGLE_PERIOD = pd.Timedelta(minutes=40)
# One upgrade for the toggle study: a flat +3% Cp change in region 2.
TOGGLE_PROFILES: dict[str, list] = {"cp_plus_3pct": [ConstantCpChange(delta=0.03)]}


def default_output_root() -> Path:
    """Return the directory the toggle study writes its outputs under.

    Overridable via ``WIND_UP_BENCHMARKING_OUTPUT_DIR``; defaults to
    ``~/temp/wind-up-benchmarking/toggle``.
    """
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "toggle"


def run_toggle_study(
    base_scada: pd.DataFrame,
    *,
    profiles: dict[str, list],
    study: StudyConfig,
    out_root: str | Path | None = None,
    include_oracle: bool = True,
) -> pd.DataFrame:
    """Score v0 and the naive ratio method on a toggle study over ``profiles`` and save outputs.

    :param base_scada: wind-up-format real SCADA (all subset turbines), the no-upgrade baseline
    :param profiles: mapping of profile name -> list of upgrade callables to inject
    :param study: the replicate/campaign sweep configuration (``mode="toggle"``)
    :param out_root: output directory; defaults to :func:`default_output_root`
    :param include_oracle: also score an oracle that returns the injected truth (sanity anchor)
    :return: the concatenated tidy per-replicate results across all profiles
    """
    out_dir = Path(out_root) if out_root is not None else default_output_root()
    out_dir.mkdir(parents=True, exist_ok=True)

    context = build_hot_v0_context(wtg_names=DEFAULT_TURBINE_SUBSET)
    scratch_dir = out_dir / "windup_runs"

    all_results = []
    for profile_name, profile in profiles.items():
        # Fastest first (oracle is instant, naive has no wind_up pipeline, v0 is a full wind_up run
        # per campaign), so the per-method curves appear early and a bad method is caught fast.
        methods: list[Method] = []
        if include_oracle:
            methods.append(OracleMethod(base_scada))
        methods.append(NaiveRatioMethod(out_dir=out_dir / "naive_runs"))
        methods.append(V0BinnedMethod(context, scratch_dir=scratch_dir))
        logger.info("Scoring toggle profile %s with methods %s", profile_name, [m.name for m in methods])
        results = score_study(
            base_scada,
            profile=profile,
            methods=methods,
            study=study,
            profile_name=profile_name,
            on_method_complete=partial(save_per_method_curve, out_dir, profile_name),
        )
        summary = leaderboard(results)

        results.to_csv(out_dir / f"results_{profile_name}.csv", index=False)
        summary.to_csv(out_dir / f"leaderboard_{profile_name}.csv", index=False)
        plot_campaign_curves(summary, save_path=out_dir / f"campaign_curves_{profile_name}.png", title=profile_name)
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    combined_summary = leaderboard(combined)
    combined_summary.to_csv(out_dir / "leaderboard_all_profiles.csv", index=False)
    logger.info("Toggle leaderboard (all profiles):\n%s", combined_summary.to_string(index=False))
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
    toggle_period: pd.Timedelta = DEFAULT_TOGGLE_PERIOD,
) -> pd.DataFrame:
    """Run the toggle study end-to-end on real Hill of Towie data and save outputs.

    :param out_root: output directory; defaults to :func:`default_output_root`
    :param data_dir: Hill of Towie data/cache dir; defaults to the source package default
    :param start_dt: inclusive UTC window start
    :param end_dt_excl: exclusive UTC window end
    :param wtg_numbers: turbine numbers to load; defaults to the stable south-west cluster
    :param n_replicates: ensemble size per profile
    :param campaign_months: the campaign-length (toggling-duration) sweep grid, in months
    :param toggle_period: the on/off cycle length (20 min on + 20 min off = 40 min by default)
    :return: the combined tidy results across all profiles
    """
    wtg_numbers = wtg_numbers if wtg_numbers is not None else DEFAULT_WTG_NUMBERS
    campaign_months = campaign_months if campaign_months is not None else [3, 6, 9, 12]
    logger.info("Loading Hill of Towie SCADA %s..%s for turbines %s", start_dt, end_dt_excl, wtg_numbers)
    scada_df, _metadata_df = load_hot_scada(
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        wtg_numbers=wtg_numbers,
        wtg_names=DEFAULT_TURBINE_SUBSET,
        data_dir=Path(data_dir) if data_dir is not None else None,
    )
    study = StudyConfig(
        mode="toggle",
        turbine_subset=DEFAULT_TURBINE_SUBSET,
        treatment_start_range=DEFAULT_TREATMENT_START_RANGE,
        min_pre_months=MIN_PRE_MONTHS,
        campaign_months=campaign_months,
        toggle_period=toggle_period,
        n_replicates=n_replicates,
        seed=0,
    )
    return run_toggle_study(scada_df, profiles=TOGGLE_PROFILES, study=study, out_root=out_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
