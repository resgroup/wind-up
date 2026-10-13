"""Draw the ``docs/analysis-period-and-reference-selection.md`` figures on Hill of Towie: the period timeline.

Every turbine's works come from the Zenodo AeroUp install-dates table, the data extents from the
SCADA over ``WINDOW``, and the one farm-wide exclusion from the real T13 declaration
(:data:`~benchmarking.campaigns.real.HOT_AEROUP_T13`). Each turbine with works is then planned as
the test turbine, with every other turbine a candidate, under the default settings:

* ``analysis_timeline_T13.png`` -- the 2021 trial turbine, whose span the selector chooses;
* ``analysis_timeline_<turbine>.png`` -- one turbine of the 2022 rollout, which it cannot plan.

Needs the cached Hill of Towie open data (see :mod:`benchmarking.synthetic.sources.hill_of_towie`).

    python -m benchmarking.baselines.study_analysis_period_doc_plot
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchmarking.campaigns.loader import read_works
from benchmarking.campaigns.plans import data_extents
from benchmarking.campaigns.real import HOT_AEROUP_T13
from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.sources.hill_of_towie import (
    ensure_hot_data_files,
    get_data_dir,
    load_hot_metadata,
    load_hot_scada,
)
from wind_up.analysis_period import DEFAULT_PLAN_SETTINGS, AnalysisPlan, plan_analysis
from wind_up.analysis_period_plots import plot_analysis_timeline
from wind_up.layout import Layout

logger = logging.getLogger("analysis_period_doc_plot")

WINDOW = ("2019-01-01", "2024-01-01")
WORKS_FILENAME = "Hill_of_Towie_AeroUp_install_dates.csv"
TRIAL_TURBINE = "T13"
ROLLOUT_EXAMPLE = "T14"
DOCS_IMAGES_DIR = Path(__file__).resolve().parents[2] / "docs" / "images" / "analysis-period"


def hot_layout() -> Layout:
    """Return the Hill of Towie layout with its 82 m rotors."""
    meta = load_hot_metadata()
    return Layout.from_frame(
        pd.DataFrame(
            {
                "name": meta["Name"],
                "latitude": meta["Latitude"],
                "longitude": meta["Longitude"],
                "rotor_diameter_m": 82.0,
                "wind_farm": "Hill of Towie",
            }
        )
    )


def hot_exclusions() -> list[tuple[str | None, pd.Timestamp, pd.Timestamp]]:
    """Return the real T13 declaration's exclusions, farm-wide ones as ``None``."""
    return [
        (None if row["turbine"] == "ALL" else str(row["turbine"]), pd.Timestamp(row["start"]), pd.Timestamp(row["end"]))
        for row in HOT_AEROUP_T13["exclusions"]
    ]


def main(*, out_dir: Path = DOCS_IMAGES_DIR) -> dict[str, AnalysisPlan | str]:
    """Plan every turbine with works as the test turbine, draw the two doc figures, and return each plan or reason."""
    layout = hot_layout()
    turbines = list(layout.frame["name"])
    data_dir = get_data_dir()
    ensure_hot_data_files([WORKS_FILENAME], data_dir=data_dir)
    works = read_works(data_dir / WORKS_FILENAME)
    scada, _ = load_hot_scada(
        start_dt=pd.Timestamp(WINDOW[0], tz="UTC"),
        end_dt_excl=pd.Timestamp(WINDOW[1], tz="UTC"),
        wtg_names=turbines,
        data_dir=data_dir,
    )
    extents = data_extents(scada, turbine_col=HOT_COLUMNS.turbine, power_col=HOT_COLUMNS.active_power)
    exclusions = hot_exclusions()

    outcomes: dict[str, AnalysisPlan | str] = {}
    for turbine in sorted(works):
        try:
            plan = plan_analysis(
                layout,
                turbine=turbine,
                works=works,
                exclusions=exclusions,
                extents=extents,
                candidates=[t for t in turbines if t != turbine],
                settings=DEFAULT_PLAN_SETTINGS,
            )
        except ValueError as exc:
            outcomes[turbine] = str(exc)
            logger.info("%s: not analysed: %s", turbine, exc)
            continue
        outcomes[turbine] = plan
        logger.info(
            "%s: span %s..%s, pre %.0f d, post %.0f d, references %s, pool rule %s",
            turbine,
            plan.start.date(),
            plan.end.date(),
            plan.pre / pd.Timedelta(days=1),
            plan.post / pd.Timedelta(days=1),
            list(plan.power_references),
            "met" if plan.pool_rule_met else f"not met ({plan.pool_rule_reason})",
        )

    for turbine in (TRIAL_TURBINE, ROLLOUT_EXAMPLE):
        outcome = outcomes[turbine]
        fig = plot_analysis_timeline(
            turbine,
            layout=layout,
            works=works,
            exclusions=exclusions,
            extents=extents,
            plan=outcome if isinstance(outcome, AnalysisPlan) else None,
            reason=outcome if isinstance(outcome, str) else "",
            out_dir=out_dir,
        )
        plt.close(fig)
    logger.info("Wrote the figures to %s", out_dir)
    return outcomes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
