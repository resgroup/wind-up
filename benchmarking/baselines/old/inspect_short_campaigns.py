"""Short-campaign (1-2 month) exploration: check whether the Issue 9-13 accepted choices hold there.

The committed benchmark sweeps 3-12 month campaigns; this driver scores 1- and 2-month campaigns
(outside the benchmark grid, so no reference-run merge — oracle + naive anchor the numbers instead
of v0, which is slow) and A/Bs the regime-dependent choices at those lengths.

Which choices can even flip at short campaigns:

* **prepost** — a shorter campaign shrinks only the *prediction* window; the training set (the
  full pre-changeover baseline) is unchanged, so the fit-side choices (capacity, features) cannot
  flip. Only the time-decay weights act on the training side, so prepost trials just those.
* **toggle** — the campaign length scales the training data itself, so capacity
  (``min_child_samples``) and the decay weights are genuinely in play.

Run from the repo root (defaults: both modes, all variants)::

    uv run python -m benchmarking.baselines.old.inspect_short_campaigns

Outputs one ``results_<mode>_<variant>_<profile>.csv`` per run plus a combined
``short_campaign_summary.csv`` / log table of power_model bias/spread/score per
``(mode, variant, campaign_months)`` under ``--output-dir``
(default ``~/temp/wind-up-benchmarking/short_campaigns``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from benchmarking.baselines.example_prepost_study import (
    DEFAULT_END_DT_EXCL,
    DEFAULT_START_DT,
    DEFAULT_TREATMENT_START_RANGE,
    DEFAULT_TURBINE_SUBSET,
    DEFAULT_WTG_NUMBERS,
    MIN_PRE_MONTHS,
)
from benchmarking.baselines.example_toggle_study import DEFAULT_TOGGLE_PERIOD
from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.naive_ratio import NaiveRatioMethod
from benchmarking.baselines.overnight_profiles import overnight_profiles
from benchmarking.baselines.study_power_model_compare import _make_power_model
from benchmarking.harness import Method, StudyConfig, leaderboard, score_study
from benchmarking.harness.example_hot_study import OracleMethod
from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

logger = logging.getLogger(__name__)

CAMPAIGN_MONTHS = [1, 2]
N_REPLICATES = 4
SEED = 0
PROFILES = ("cp_0pct", "cp_plus_3pct")  # placebo (bias/spread) + a plain recovery check
_DEFAULT_OUTPUT_DIR = Path.home() / "temp" / "wind-up-benchmarking" / "short_campaigns"

# Variant name -> (modes it is meaningful for, PowerModelMethod overrides). The "default" anchor
# also scores oracle + naive for context. See the module docstring for why prepost trials only
# the decay weights.
VARIANTS: dict[str, tuple[tuple[str, ...], dict[str, Any]]] = {
    "default": (("prepost", "toggle"), {}),
    "hl90": (("prepost", "toggle"), {"adaptive_time_decay": False, "time_decay_half_life_days": 90}),
    "hl365": (("prepost", "toggle"), {"adaptive_time_decay": False, "time_decay_half_life_days": 365}),
    "mcs200": (("toggle",), {"model_params": {"min_child_samples": 200}}),
}


def _study(mode: str) -> StudyConfig:
    return StudyConfig(
        mode=cast("Literal['prepost', 'toggle']", mode),
        turbine_subset=DEFAULT_TURBINE_SUBSET,
        treatment_start_range=DEFAULT_TREATMENT_START_RANGE,
        min_pre_months=MIN_PRE_MONTHS,
        campaign_months=CAMPAIGN_MONTHS,
        toggle_period=DEFAULT_TOGGLE_PERIOD if mode == "toggle" else None,
        n_replicates=N_REPLICATES,
        seed=SEED,
    )


def run_variant(
    mode: str,
    variant: str,
    out_dir: Path,
    *,
    scada_df: pd.DataFrame,
    era5_hourly_df: pd.DataFrame,
    profiles: dict[str, list],
) -> pd.DataFrame:
    """Score one (mode, variant) over the short-campaign grid; anchor methods only on ``default``."""
    overrides = VARIANTS[variant][1]
    frames = []
    for profile_name, profile in profiles.items():
        methods: list[Method] = []
        if variant == "default":
            methods.append(OracleMethod(scada_df))
            methods.append(
                NaiveRatioMethod(
                    columns=HOT_COLUMNS,
                    out_dir=out_dir / "naive_runs",
                )
            )
        methods.append(
            _make_power_model(out_dir / variant / profile_name, era5_hourly_df=era5_hourly_df, overrides=overrides)
        )
        logger.info("Scoring %s / %s / %s", mode, variant, profile_name)
        results = score_study(scada_df, profile=profile, methods=methods, study=_study(mode), profile_name=profile_name)
        results.to_csv(out_dir / f"results_{mode}_{variant}_{profile_name}.csv", index=False)
        frames.append(results.assign(variant=variant, mode=mode))
    return pd.concat(frames, ignore_index=True)


def summarise(all_results: pd.DataFrame, out_dir: Path) -> None:
    """Write/log power_model bias/spread/score per (mode, variant, profile, campaign) + the anchors."""
    rows = []
    for (mode, variant), chunk in all_results.groupby(["mode", "variant"]):
        lb = leaderboard(chunk)
        lb = lb.assign(mode=mode, variant=variant)
        rows.append(lb)
    summary = pd.concat(rows, ignore_index=True)
    cols = ["mode", "variant", "method", "profile", "campaign_months", "bias", "spread", "score"]
    summary = summary[cols].sort_values(["mode", "profile", "campaign_months", "method", "variant"])
    summary.to_csv(out_dir / "short_campaign_summary.csv", index=False)
    show = summary.copy()
    for col in ("bias", "spread", "score"):
        show[col] = (100 * show[col]).round(3)
    logger.info("Short-campaign summary [pp]:\n%s", show.to_string(index=False))


def main() -> None:
    """Run the short-campaign exploration for the requested modes/variants."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", choices=["prepost", "toggle"], default=["prepost", "toggle"])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=None)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    out_dir = args.output_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    scada_df, _ = load_hot_scada(
        start_dt=DEFAULT_START_DT,
        end_dt_excl=DEFAULT_END_DT_EXCL,
        wtg_numbers=DEFAULT_WTG_NUMBERS,
        wtg_names=DEFAULT_TURBINE_SUBSET,
    )
    context = build_hot_v0_context(wtg_names=DEFAULT_TURBINE_SUBSET)
    era5 = context.reanalysis_datasets[0].data
    profiles = {name: overnight_profiles()[name] for name in PROFILES}

    all_results = []
    for mode in args.modes:
        for variant, (variant_modes, _) in VARIANTS.items():
            if args.variants is not None and variant not in args.variants:
                continue
            if mode not in variant_modes:
                continue
            all_results.append(
                run_variant(mode, variant, out_dir, scada_df=scada_df, era5_hourly_df=era5, profiles=profiles)
            )
    summarise(pd.concat(all_results, ignore_index=True), out_dir)


if __name__ == "__main__":
    main()
