"""Longer (overnight) TOGGLE study: oracle + naive + R-learner + v0 over seven upgrade profiles.

Run on a server from the repo root::

    uv run python -m benchmarking.baselines.study_overnight_toggle

Same seven profiles and outputs as the prepost study (see
:mod:`benchmarking.baselines.study_overnight_prepost`), but with a 20-min-on / 20-min-off
toggle. Outputs go under ``WIND_UP_BENCHMARKING_OUTPUT_DIR/toggle`` (default
``~/temp/wind-up-benchmarking/toggle``). The naive and R-learner methods fit the interleaved
on/off campaign window only (``toggle_campaign_only`` default), so on and off share a wind
distribution.

Tune runtime vs precision with the two constants below.
"""

from __future__ import annotations

import logging

from benchmarking.baselines.example_prepost_study import (
    DEFAULT_END_DT_EXCL,
    DEFAULT_START_DT,
    DEFAULT_TREATMENT_START_RANGE,
    DEFAULT_TURBINE_SUBSET,
    DEFAULT_WTG_NUMBERS,
    MIN_PRE_MONTHS,
)
from benchmarking.baselines.example_toggle_study import DEFAULT_TOGGLE_PERIOD, run_toggle_study
from benchmarking.baselines.overnight_profiles import overnight_profiles
from benchmarking.harness import StudyConfig
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

logger = logging.getLogger(__name__)

# --- tune these two for the runtime budget -------------------------------------------------
N_REPLICATES = 4  # (turbine, treatment-start) draws per profile; the precision axis
CAMPAIGN_MONTHS = [3, 6, 9, 12]  # toggling-duration sweep grid
# -------------------------------------------------------------------------------------------


def main() -> None:
    """Load real Hill of Towie SCADA and run the overnight toggle study (incl. v0)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    scada_df, _ = load_hot_scada(
        start_dt=DEFAULT_START_DT,
        end_dt_excl=DEFAULT_END_DT_EXCL,
        wtg_numbers=DEFAULT_WTG_NUMBERS,
        wtg_names=DEFAULT_TURBINE_SUBSET,
    )
    study = StudyConfig(
        mode="toggle",
        turbine_subset=DEFAULT_TURBINE_SUBSET,
        treatment_start_range=DEFAULT_TREATMENT_START_RANGE,
        min_pre_months=MIN_PRE_MONTHS,
        campaign_months=CAMPAIGN_MONTHS,
        toggle_period=DEFAULT_TOGGLE_PERIOD,
        n_replicates=N_REPLICATES,
        seed=0,
    )
    run_toggle_study(scada_df, profiles=overnight_profiles(), study=study, include_v0=True)


if __name__ == "__main__":
    main()
