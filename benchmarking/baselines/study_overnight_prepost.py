"""Longer (overnight) PREPOST study: oracle + naive + R-learner + v0 over seven upgrade profiles.

Run on a server from the repo root::

    uv run python -m benchmarking.baselines.study_overnight_prepost

Outputs (per-profile leaderboards with uplift bias/spread/score and wall-time, the tidy
per-replicate results, per-method campaign-length curves, and each method's per-run
diagnostics) are written under ``WIND_UP_BENCHMARKING_OUTPUT_DIR`` (default
``~/temp/wind-up-benchmarking/prepost``). The first run downloads + caches the Hill of Towie
SCADA (Zenodo) and ERA5 (Open-Meteo, needs the ``era5`` group); install the ``ml`` group too
for the R-learner (lightgbm).

Tune runtime vs precision with the two constants below. v0 (a full wind_up run per campaign)
and the R-learner are the cost; oracle and naive are ~free.
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
    default_output_root,
    run_prepost_study,
)
from benchmarking.baselines.overnight_common import start_overnight_run
from benchmarking.baselines.overnight_profiles import overnight_profiles
from benchmarking.harness import StudyConfig
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada

logger = logging.getLogger(__name__)

# --- tune these two for the runtime budget -------------------------------------------------
N_REPLICATES = 4  # (turbine, treatment-start) draws per profile; the precision axis
CAMPAIGN_MONTHS = [3, 6, 12]  # campaign-length sweep grid
# -------------------------------------------------------------------------------------------


def main() -> None:
    """Load real Hill of Towie SCADA and run the overnight prepost study (incl. v0)."""
    study = StudyConfig(
        mode="prepost",
        turbine_subset=DEFAULT_TURBINE_SUBSET,
        treatment_start_range=DEFAULT_TREATMENT_START_RANGE,
        min_pre_months=MIN_PRE_MONTHS,
        campaign_months=CAMPAIGN_MONTHS,
        n_replicates=N_REPLICATES,
        seed=0,
    )
    out_dir = start_overnight_run("prepost", study, output_root=default_output_root().parent)
    scada_df, _ = load_hot_scada(
        start_dt=DEFAULT_START_DT,
        end_dt_excl=DEFAULT_END_DT_EXCL,
        wtg_numbers=DEFAULT_WTG_NUMBERS,
        wtg_names=DEFAULT_TURBINE_SUBSET,
    )
    run_prepost_study(scada_df, profiles=overnight_profiles(), study=study, out_root=out_dir, include_v0=True)


if __name__ == "__main__":
    main()
