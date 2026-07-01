"""Ablate reactive-power / pitch reference features on the F1 hard prepost case.

A focused follow-up to :mod:`benchmarking.baselines.inspect_prepost_hard_case` (findings F1/F2).
It pins the **same** placebo case (``cp_0pct`` on ``T07`` at 6 months, true uplift 0%) and the
**identical** ``MethodInput``, then runs the R-learner three times while removing reference
features from the SCADA frame before feature-building:

1. ``full`` — all reference features (the F1 regime);
2. ``no reactive power`` — drop ``wtc_ReactPwr_mean`` from every reference turbine;
3. ``no reactive power, no pitch`` — also drop the three blade-pitch sensors.

Motivation (the F2 hypothesis): reactive power and pitch are the top propensity features, but the
reactive-power diagnostics show its *control* changes over calendar time, so it (and pitch) may be
acting as proxies for "is this the upgraded season?" — i.e. driving the F1 overlap/positivity
failure rather than carrying physics. Dropping them isolates how much of the prepost bias they own.

Only the feature set differs between arms; the outcome (test active power), availability, wind speed
and ERA5 features are untouched, and the cross-fitting seed is fixed, so the comparison is
apples-to-apples and deterministic.

Run it::

    uv run python -m benchmarking.baselines.inspect_prepost_feature_ablation
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.baselines.example_prepost_study import (
    DEFAULT_END_DT_EXCL,
    DEFAULT_START_DT,
    DEFAULT_TURBINE_SUBSET,
    DEFAULT_WTG_NUMBERS,
)
from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.inspect_prepost_hard_case import (
    DEFAULT_CAMPAIGN_MONTHS,
    DEFAULT_PROFILE,
    DEFAULT_TEST_WTG,
    _overnight_study,
    _pin_case,
)
from benchmarking.baselines.rlearner import RLearnerMethod
from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

if TYPE_CHECKING:
    from benchmarking.harness import MethodInput

logger = logging.getLogger(__name__)

REACTIVE_TAG = "wtc_ReactPwr_mean"
PITCH_TAGS = ("wtc_PitcPosA_mean", "wtc_PitcPosB_mean", "wtc_PitcPosC_mean")

ARMS: dict[str, tuple[str, ...]] = {
    "full (all features)": (),
    "no reactive power": (REACTIVE_TAG,),
    "no reactive power, no pitch": (REACTIVE_TAG, *PITCH_TAGS),
}


def _drop_cols(mi: MethodInput, cols: tuple[str, ...]) -> MethodInput:
    """Return a copy of ``mi`` with the given source-native tags removed from every turbine."""
    present = [c for c in cols if c in mi.scada_df.columns]
    return replace(mi, scada_df=mi.scada_df.drop(columns=present))


def _make_method(era5_df: pd.DataFrame) -> RLearnerMethod:
    """Build an R-learner configured exactly as the inspection driver, plots off (estimate only)."""
    return RLearnerMethod(
        active_power_col=HOT_COLUMNS.active_power,
        wind_speed_col=HOT_COLUMNS.wind_speed,
        availability_col=HOT_COLUMNS.availability,
        era5_hourly_df=era5_df,
        out_dir=None,
        save_plots=False,
    )


def inspect_prepost_feature_ablation() -> pd.DataFrame:
    """Run the three feature-ablation arms on the pinned F1 case; return a tidy comparison frame."""
    study = _overnight_study()
    logger.info("Loading Hill of Towie SCADA %s..%s", DEFAULT_START_DT, DEFAULT_END_DT_EXCL)
    scada_df, _ = load_hot_scada(
        start_dt=DEFAULT_START_DT,
        end_dt_excl=DEFAULT_END_DT_EXCL,
        wtg_numbers=DEFAULT_WTG_NUMBERS,
        wtg_names=DEFAULT_TURBINE_SUBSET,
    )
    _rep, mi, truth, _window = _pin_case(
        scada_df,
        study=study,
        profile_name=DEFAULT_PROFILE,
        test_wtg=DEFAULT_TEST_WTG,
        campaign_months=DEFAULT_CAMPAIGN_MONTHS,
    )

    era5_df = build_hot_v0_context(wtg_names=DEFAULT_TURBINE_SUBSET).reanalysis_datasets[0].data

    rows = []
    for label, cols in ARMS.items():
        estimate = _make_method(era5_df).estimate(_drop_cols(mi, cols)).p50_overall
        logger.info(
            "%-32s estimate %+.3f%%  truth %+.3f%%  error %+.3f%%  (dropped %s)",
            label,
            100 * estimate,
            100 * truth,
            100 * (estimate - truth),
            list(cols) or "nothing",
        )
        rows.append({"arm": label, "estimate": estimate, "truth": truth, "signed_error": estimate - truth})

    summary = pd.DataFrame(rows)
    logger.info("\n%s", summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    inspect_prepost_feature_ablation()
