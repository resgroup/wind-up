"""Real campaigns declared from open data, written as campaign folders ``wind-up`` can run.

Write the Hill of Towie AeroUp T13 campaign, or another of :data:`REAL_CAMPAIGNS`::

    python -m benchmarking.campaigns.real OUT_DIR [--campaign hot_tuneup]
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

from benchmarking.campaigns.loader import EXPECTED_UPLIFT_PREFIX, operating_state_config
from benchmarking.synthetic.sources.hill_of_towie import (
    HOT_ACTIVE_POWER_MAX_COL,
    HOT_AMBIENT_TEMP_COL,
    HOT_BM_START,
    HOT_COLUMNS,
    HOT_COORDINATES,
    HOT_CURTAILMENT_REACH,
    HOT_HIGH_WIND_DERATE_MS,
    HOT_ICING_MAX_TEMP_C,
    HOT_ICING_MIN_BIN_RECORDS,
    HOT_ICING_MIN_EXPECTED_KW,
    HOT_ICING_MIN_RUN_RECORDS,
    HOT_ICING_POWER_FRACTION,
    HOT_ICING_REFERENCE_MIN_TEMP_C,
    HOT_NOISE_SETPOINTS_KW,
    HOT_POWER_SETPOINT_COL,
    HOT_RATED_POWER_KW,
    HOT_ROTOR_DIAMETER_M,
    HOT_STARTUP_SETPOINT_KW,
    HOT_STOP_SETPOINT_KW,
    HOT_TURBINE_COL,
    TIMEBASE_S,
    ensure_hot_data_files,
    get_data_dir,
    load_hot_scada,
)
from wind_up.operating_state import PARTIAL_DOWNTIME, label_operating_states

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

HOT_AEROUP_WORKS = "Hill_of_Towie_AeroUp_install_dates.csv"
AEROUP_EXPECTED_UPLIFT = 0.05
TUNEUP_EXPECTED_UPLIFT = 0.01
# The TuneUp was deployed on the upgraded turbines during this working day; it ends at the changeover.
TUNEUP_DEPLOYMENT = ("2024-03-14T07:00:00Z", "2024-03-14T18:00:00Z")

# The works table a real campaign writes.
WORKS_COLUMNS = ("Turbine", "Works", "First date of works", "Last date of works", EXPECTED_UPLIFT_PREFIX)

# Hill of Towie's site operating states, written into the campaign's SCADA as HOT_STATE_COL.
HOT_STATE_COL = "state"
NOISE_MODE = "noise mode"
BM_CURTAILMENT = "BM curtailment"
HIGH_WIND_DERATE = "high wind derate"
ICING = "icing"
HOT_OPERATING_STATE: dict[str, Any] = {
    "site_state_column": HOT_STATE_COL,
    "parked_pitch_above_deg": 45,
    "site_states": {
        NOISE_MODE: {"northing": "valid", "waking": "waking", "uplift": "not valid"},
        BM_CURTAILMENT: {"northing": "valid", "waking": "part waking", "uplift": "not valid"},
        HIGH_WIND_DERATE: {"northing": "valid", "waking": "part waking", "uplift": "not valid"},
        ICING: {"northing": "not valid", "waking": "part waking", "uplift": "not valid"},
    },
}

# T13's AeroUp retrofit, with every other turbine's AeroUp works in the works table. The farm's
# curtailment from September 2022 is labelled record by record (BM curtailment), so it is not declared.
HOT_AEROUP_T13: dict[str, Any] = {
    "name": "hot_aeroup_t13",
    "data": {
        "scada": "data/scada.parquet",
        "schema": "hill_of_towie",
        "turbines": "data/turbines.csv",
        "installation_windows": "data/works.csv",
    },
    "turbines": {"upgraded": ["T13"], "rated_power_kw": 2300.0},
    "timing": {"mode": "prepost"},
    "upgrade": {"expected_uplift": AEROUP_EXPECTED_UPLIFT},
    "northing": {"discover": True},
    "operating_state": HOT_OPERATING_STATE,
}

# The 2024 pitch TuneUp on nine turbines, sharing one changeover at the end of its deployment. Every
# turbine's AeroUp works are in the works table at the AeroUp's expected uplift, and each upgraded
# turbine's deployment window at the TuneUp's.
HOT_TUNEUP: dict[str, Any] = {
    "name": "hot_tuneup",
    "data": dict(HOT_AEROUP_T13["data"]),
    "turbines": {
        "upgraded": ["T02", "T03", "T06", "T08", "T09", "T13", "T16", "T20", "T21"],
        "rated_power_kw": 2300.0,
    },
    "timing": {"mode": "prepost", "changeover": "2024-03-14T18:00:00Z"},
    "upgrade": {"expected_uplift": TUNEUP_EXPECTED_UPLIFT},
    "northing": {"discover": True},
    "operating_state": HOT_OPERATING_STATE,
}


TUNEUP_TURBINES = HOT_TUNEUP["turbines"]["upgraded"]

# The same TuneUp declared as two upgrades: the AeroUp on every turbine over its own works, then the
# TuneUp on the nine turbines. The turbines with only the AeroUp serve as the TuneUp's references.
HOT_TUNEUP_UPGRADES: dict[str, Any] = {
    "name": "hot_tuneup_upgrades",
    "data": {k: v for k, v in HOT_AEROUP_T13["data"].items() if k != "installation_windows"},
    "turbines": {"rated_power_kw": 2300.0},
    "upgrades": [
        {"name": "AeroUp", "installation_windows": "data/works.csv", "expected_uplift": AEROUP_EXPECTED_UPLIFT},
        {
            "name": "TuneUp",
            "turbines": TUNEUP_TURBINES,
            "timing": {"mode": "prepost", "changeover": TUNEUP_DEPLOYMENT[1]},
            "installation_windows": [
                {"turbine": t, "start": TUNEUP_DEPLOYMENT[0], "end": TUNEUP_DEPLOYMENT[1]} for t in TUNEUP_TURBINES
            ],
            "expected_uplift": TUNEUP_EXPECTED_UPLIFT,
        },
    ],
    "analyses": [{"name": "TuneUp", "upgrade": "TuneUp"}],
    "northing": {"discover": True},
    "operating_state": HOT_OPERATING_STATE,
}


@dataclass(frozen=True)
class RealCampaign:
    """A real Hill of Towie campaign: its declaration and the span of SCADA written with it.

    :param declaration: the contents of its ``campaign.yaml``
    :param start: the first SCADA record written
    :param end: the end of the SCADA written, exclusive
    :param works_expected_uplift: the expected uplift written on every AeroUp row of the works table, or
        None for rows that take the campaign's upgrade
    :param deployment: ``(start, end)`` of a works window written for every upgraded turbine, at the
        campaign's expected uplift; None for none
    """

    declaration: dict[str, Any]
    start: pd.Timestamp
    end: pd.Timestamp
    works_expected_uplift: float | None = None
    deployment: tuple[str, str] | None = None

    @property
    def name(self) -> str:
        """The campaign's declared name."""
        return str(self.declaration["name"])


HOT_AEROUP_T13_CAMPAIGN = RealCampaign(
    HOT_AEROUP_T13, start=pd.Timestamp("2019-01-01", tz="UTC"), end=pd.Timestamp("2023-01-01", tz="UTC")
)
HOT_TUNEUP_CAMPAIGN = RealCampaign(
    HOT_TUNEUP,
    start=pd.Timestamp("2022-01-01", tz="UTC"),
    end=pd.Timestamp("2024-09-01", tz="UTC"),
    works_expected_uplift=AEROUP_EXPECTED_UPLIFT,
    deployment=TUNEUP_DEPLOYMENT,
)
HOT_TUNEUP_UPGRADES_CAMPAIGN = RealCampaign(
    HOT_TUNEUP_UPGRADES, start=HOT_TUNEUP_CAMPAIGN.start, end=HOT_TUNEUP_CAMPAIGN.end
)
REAL_CAMPAIGNS: dict[str, RealCampaign] = {
    c.name: c for c in (HOT_AEROUP_T13_CAMPAIGN, HOT_TUNEUP_CAMPAIGN, HOT_TUNEUP_UPGRADES_CAMPAIGN)
}


def label_hot_operating_states(scada_long: pd.DataFrame) -> pd.DataFrame:
    """Return Hill of Towie SCADA with its site labels and the operating-state columns of :data:`HOT_OPERATING_STATE`.

    :param scada_long: long-format Hill of Towie SCADA, indexed by timestamp
    """
    scada_long = scada_long.assign(**{HOT_STATE_COL: label_hot_site_states(scada_long)})
    return label_operating_states(
        scada_long,
        columns=HOT_COLUMNS,
        config=operating_state_config(HOT_OPERATING_STATE),
        timebase=pd.Timedelta(seconds=TIMEBASE_S),
        rated_power_kw=HOT_RATED_POWER_KW,
    )


def label_hot_site_states(scada_long: pd.DataFrame) -> pd.Series:
    """Return each Hill of Towie record's site operating-state label, None where it has none.

    Reads the power setpoint at the end of each record; the setpoint at its start is the turbine's
    previous record's end, NaN after a gap. A record takes, from either end and in this order:

    1. a setpoint of 0: BM curtailment from the Balancing Mechanism start, partial downtime before it;
    2. no label: the start-up setpoint;
    3. high wind derate: wind speed at or above ``HOT_HIGH_WIND_DERATE_MS`` and a setpoint below rated
       that is not one of the turbine's noise setpoints;
    4. BM curtailment: from the Balancing Mechanism start, below that wind speed, a setpoint below
       rated that is not one of the turbine's noise setpoints and that the record's power max reaches
       ``HOT_CURTAILMENT_REACH`` of, or a missing power max;
    5. noise mode: one of the turbine's noise setpoints.

    A record without a setpoint label is icing when it is cold and produces under a fraction of the
    turbine's warm-weather median power at its wind speed, for a run of consecutive records (the
    ``HOT_ICING_*`` constants).

    :param scada_long: long-format Hill of Towie SCADA, indexed by timestamp
    :return: the labels on ``scada_long``'s index, in its order
    """
    turbine = scada_long[HOT_TURBINE_COL].astype(str).to_numpy()
    stamps = pd.DatetimeIndex(scada_long.index)
    order = np.lexsort((stamps.asi8, pd.factorize(turbine)[0]))
    wtg = turbine[order]
    when = stamps[order]
    end = scada_long[HOT_POWER_SETPOINT_COL].to_numpy(dtype=float)[order]
    start = np.concatenate([[np.nan], end[:-1]])
    follows = np.concatenate(
        [[False], (wtg[1:] == wtg[:-1]) & (np.diff(when.as_unit("ns").asi8) == TIMEBASE_S * 10**9)]
    )
    start[~follows] = np.nan
    wind_speed = scada_long[HOT_COLUMNS.wind_speed].to_numpy(dtype=float)[order]
    high_wind = wind_speed >= HOT_HIGH_WIND_DERATE_MS
    power_max = scada_long[HOT_ACTIVE_POWER_MAX_COL].to_numpy(dtype=float)[order]

    def noise(setpoint: np.ndarray) -> np.ndarray:
        hit = np.zeros(len(setpoint), dtype=bool)
        for name, steps in HOT_NOISE_SETPOINTS_KW.items():
            hit |= (wtg == name) & np.isin(setpoint, list(steps))
        return hit

    def reduced(setpoint: np.ndarray) -> np.ndarray:
        return (setpoint < HOT_RATED_POWER_KW) & ~noise(setpoint)

    def derated(setpoint: np.ndarray) -> np.ndarray:
        return high_wind & reduced(setpoint)

    def curtailed(setpoint: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            reached = ~(power_max < HOT_CURTAILMENT_REACH * setpoint)
        return (when >= HOT_BM_START) & ~high_wind & reduced(setpoint) & reached

    def either(rule: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        return rule(end) | rule(start)

    # Lowest precedence first, so a higher rule overwrites.
    labels = np.full(len(end), None, dtype=object)
    labels[either(noise)] = NOISE_MODE
    labels[either(curtailed)] = BM_CURTAILMENT
    labels[either(derated)] = HIGH_WIND_DERATE
    labels[either(lambda s: s == HOT_STARTUP_SETPOINT_KW)] = None
    stopped = either(lambda s: s == HOT_STOP_SETPOINT_KW)
    labels[stopped] = np.where(when[stopped] >= HOT_BM_START, BM_CURTAILMENT, PARTIAL_DOWNTIME)
    labels[_iced(scada_long, order=order, follows=follows, unlabelled=pd.isna(labels))] = ICING
    result = np.empty(len(end), dtype=object)
    result[order] = labels
    return pd.Series(result, index=scada_long.index, name=HOT_STATE_COL, dtype=object)


def _iced(scada_long: pd.DataFrame, *, order: np.ndarray, follows: np.ndarray, unlabelled: np.ndarray) -> np.ndarray:
    """Return which of the ``order``-sorted records are icing, among the ``unlabelled`` ones.

    ``follows`` marks the records that directly follow the previous one of the same turbine.
    """
    rows = pd.DataFrame(
        {
            "turbine": scada_long[HOT_TURBINE_COL].astype(str).to_numpy()[order],
            "bin": np.round(scada_long[HOT_COLUMNS.wind_speed].to_numpy(dtype=float)[order] * 2) / 2,
            "power": scada_long[HOT_COLUMNS.active_power].to_numpy(dtype=float)[order],
        }
    )
    ambient = scada_long[HOT_AMBIENT_TEMP_COL].to_numpy(dtype=float)[order]
    warm = unlabelled & (ambient > HOT_ICING_REFERENCE_MIN_TEMP_C)
    curve = rows[warm].groupby(["turbine", "bin"])["power"].agg(["median", "count"])
    curve = curve.loc[curve["count"] >= HOT_ICING_MIN_BIN_RECORDS, "median"].rename("expected")
    expected = rows.join(curve, on=["turbine", "bin"])["expected"].to_numpy()
    with np.errstate(invalid="ignore"):
        low = (
            unlabelled
            & (ambient <= HOT_ICING_MAX_TEMP_C)
            & (expected >= HOT_ICING_MIN_EXPECTED_KW)
            & (rows["power"].to_numpy() < HOT_ICING_POWER_FRACTION * expected)
        )
    continues = follows & np.concatenate([[False], low[:-1]])
    run = np.cumsum(~continues)
    run_length = pd.Series(low).groupby(run).transform("sum").to_numpy()
    return low & (run_length >= HOT_ICING_MIN_RUN_RECORDS)


def works_table(aeroup: pd.DataFrame, *, campaign: RealCampaign) -> pd.DataFrame:
    """Return ``campaign``'s works table: the published AeroUp rows, then its deployment rows.

    An AeroUp row carries ``works_expected_uplift`` when the campaign gives one and is blank otherwise;
    a deployment row carries the campaign's expected uplift.

    :param aeroup: the published AeroUp install dates: the turbine, then its first and last date
    :param campaign: the campaign
    """
    turbine, first, last = aeroup.columns[:3]
    uplift = campaign.works_expected_uplift
    rows = [
        {
            "Turbine": t,
            "Works": "AeroUp",
            "First date of works": f,
            "Last date of works": la,
            EXPECTED_UPLIFT_PREFIX: uplift if uplift is not None else None,
        }
        for t, f, la in zip(aeroup[turbine], aeroup[first], aeroup[last], strict=True)
    ]
    if campaign.deployment is not None:
        start, end = campaign.deployment
        rows += [
            {
                "Turbine": t,
                "Works": "deployment",
                "First date of works": start,
                "Last date of works": end,
                EXPECTED_UPLIFT_PREFIX: campaign.declaration["upgrade"]["expected_uplift"],
            }
            for t in campaign.declaration["turbines"]["upgraded"]
        ]
    return pd.DataFrame(rows, columns=list(WORKS_COLUMNS))


def write_real_campaign(
    out_dir: Path,
    *,
    campaign: RealCampaign,
    data_dir: Path | None = None,
) -> Path:
    """Write ``campaign``'s folder under ``out_dir`` and return its declaration.

    Writes ``campaign.yaml`` and, under ``data/``, every turbine's SCADA over the campaign's span with
    its site operating states, the turbines file and the works table (:func:`works_table`), which a
    declaration that lists its upgrades names as its AeroUp's installation windows.

    :param out_dir: the campaign folder
    :param campaign: which campaign
    :param data_dir: the Hill of Towie download cache; defaults to its usual place
    """
    data_dir = data_dir if data_dir is not None else get_data_dir()
    data = out_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    scada, _ = load_hot_scada(start_dt=campaign.start, end_dt_excl=campaign.end, data_dir=data_dir)
    scada[HOT_STATE_COL] = label_hot_site_states(scada)
    scada.to_parquet(data / "scada.parquet")
    pd.DataFrame(
        {
            "name": list(HOT_COORDINATES),
            "latitude": [lat for lat, _ in HOT_COORDINATES.values()],
            "longitude": [lon for _, lon in HOT_COORDINATES.values()],
            "rotor_diameter_m": HOT_ROTOR_DIAMETER_M,
        }
    ).to_csv(data / "turbines.csv", index=False)
    ensure_hot_data_files([HOT_AEROUP_WORKS], data_dir=data_dir)
    works_table(pd.read_csv(data_dir / HOT_AEROUP_WORKS), campaign=campaign).to_csv(data / "works.csv", index=False)
    path = out_dir / "campaign.yaml"
    path.write_text(yaml.safe_dump(campaign.declaration, sort_keys=False))
    logger.info("Wrote the Hill of Towie campaign %s to %s", campaign.name, out_dir)
    return path


def main(argv: list[str] | None = None) -> None:
    """Write a real campaign's folder."""
    parser = argparse.ArgumentParser(prog="python -m benchmarking.campaigns.real", description=__doc__)
    parser.add_argument("out_dir", type=Path, help="the campaign folder to write")
    parser.add_argument(
        "--campaign", choices=list(REAL_CAMPAIGNS), default=HOT_AEROUP_T13_CAMPAIGN.name, help="which campaign"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    write_real_campaign(args.out_dir, campaign=REAL_CAMPAIGNS[args.campaign])


if __name__ == "__main__":
    main()
