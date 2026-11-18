"""Characterization pin for the methods a campaign carries, held across the public interface rework.

A small seeded campaign with turbine-specific gains and noise, so each method's estimate differs
from the truth and from the other methods'. The numbers pin what ``naive_ratio`` and
``toggle_energy_ratio`` read through the campaign runner: a change to which rows reach a method, to
its toggle grid or to its validity fails here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.naive_ratio import NaiveRatioMethod
from benchmarking.campaigns import CampaignRunner
from benchmarking.synthetic import HOT_COLUMNS, ConstantCpChange, ToggleSchedule
from wind_up.methods.toggle_energy_ratio import ToggleEnergyRatioMethod

from .test_declaration import CHANGEOVER, campaign, scada

TOGGLE_PERIOD = pd.Timedelta(hours=4)
GAINS = {"T1": 1.00, "T2": 0.96, "T3": 1.03, "T4": 0.99, "T5": 1.01}
NOISE = 0.05
RELATIVE = 1e-9

# Score rows by method, test turbine and condition bin, as (estimate, sigma, truth).
TOGGLE_SCORES = {
    ("naive_ratio", "T1", "overall"): (0.04921706978502827, np.nan, 0.04982801008530457),
    ("toggle_energy_ratio", "T1", "overall"): (0.04849299002119843, 0.00275832148806285, 0.04982801008530457),
    ("toggle_energy_ratio", "T1", "(-230.0, 230.0]"): (np.nan, np.nan, np.nan),
    ("toggle_energy_ratio", "T1", "(230.0, 690.0]"): (0.04935553381355885, 0.00433584320232391, 0.04999808797651539),
    ("toggle_energy_ratio", "T1", "(690.0, 1150.0]"): (0.05578643193501798, 0.00422358532295519, 0.04995323974849786),
    ("toggle_energy_ratio", "T1", "(1150.0, 1610.0]"): (0.03127940481027568, 0.00438461330155152, 0.04960576602528399),
    ("toggle_energy_ratio", "T1", "(1610.0, 2070.0]"): (np.nan, np.nan, np.nan),
    ("toggle_energy_ratio", "T1", "(2070.0, 2530.0]"): (np.nan, np.nan, np.nan),
    ("naive_ratio", "T2", "overall"): (0.05199500950320801, np.nan, 0.04988706850036406),
    ("toggle_energy_ratio", "T2", "overall"): (0.05115716766756796, 0.00284212013832506, 0.04988706850036406),
    ("toggle_energy_ratio", "T2", "(-230.0, 230.0]"): (np.nan, np.nan, np.nan),
    ("toggle_energy_ratio", "T2", "(230.0, 690.0]"): (0.05345915359024603, 0.00423762067232212, 0.04999817112188132),
    ("toggle_energy_ratio", "T2", "(690.0, 1150.0]"): (0.05406893462616733, 0.00419966921304816, 0.04994850261273820),
    ("toggle_energy_ratio", "T2", "(1150.0, 1610.0]"): (0.03198184585262524, 0.00442495435673531, 0.04970545232061285),
    ("toggle_energy_ratio", "T2", "(1610.0, 2070.0]"): (np.nan, np.nan, np.nan),
    ("toggle_energy_ratio", "T2", "(2070.0, 2530.0]"): (np.nan, np.nan, np.nan),
}
# Farm rows by method, as (estimate, truth).
TOGGLE_FARM = {
    "naive_ratio": (0.05057683065243146, 0.04985695650892663),
    "toggle_energy_ratio": (0.04979713685137943, 0.04985695650892663),
}
PREPOST_SCORES = {
    ("naive_ratio", "T1", "overall"): (0.05254115315391483, np.nan, 0.04982686527297875),
    ("naive_ratio", "T2", "overall"): (0.05076889079752034, np.nan, 0.04988568437814567),
}
PREPOST_FARM = {"naive_ratio": (0.05167231300495323, 0.04985567536423208)}


def noisy_scada() -> pd.DataFrame:
    """The fixture frame with a gain per turbine and seeded noise on every record's power."""
    frame = scada()
    rng = np.random.default_rng(0)
    power = frame[HOT_COLUMNS.active_power].to_numpy() * frame[HOT_COLUMNS.turbine].map(GAINS).to_numpy()
    frame[HOT_COLUMNS.active_power] = power * (1.0 + NOISE * rng.standard_normal(len(frame)))
    return frame


def run(timing: object) -> tuple[dict, dict]:
    """Run the campaign and return its score rows and farm rows keyed as the pinned tables are."""
    declared = campaign(upgrades=[ConstantCpChange(delta=0.05)], upgrade_timing=timing)
    dataset = declared.generate(noisy_scada())
    spec = declared.spec()

    def build(_wtg: str) -> list:
        methods: list = [NaiveRatioMethod(columns=HOT_COLUMNS)]
        if spec.mode == "toggle":
            methods.append(
                ToggleEnergyRatioMethod(
                    columns=HOT_COLUMNS,
                    toggle_period=TOGGLE_PERIOD,
                    toggle_datum=CHANGEOVER,
                    conditions=("power",),
                    rated_power_kw=spec.rated_power_kw,
                )
            )
        return methods

    result = CampaignRunner(spec, dataset, build_methods=build).run()
    scores = {
        (str(row.method), str(row.test_wtg), str(row.condition_bin)): (
            float(row.estimate),
            float(row.sigma),
            float(row.truth),
        )
        for row in result.scores.itertuples()
    }
    farm = {str(row.method): (float(row.estimate), float(row.truth)) for row in result.farm.itertuples()}
    return scores, farm


def assert_pinned(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key, values in expected.items():
        assert actual[key] == pytest.approx(values, rel=RELATIVE, nan_ok=True), key


def test_a_toggle_campaign_reads_the_pinned_numbers() -> None:
    scores, farm = run(ToggleSchedule(period=TOGGLE_PERIOD, start=CHANGEOVER))
    assert_pinned(scores, TOGGLE_SCORES)
    assert_pinned(farm, TOGGLE_FARM)


def test_a_prepost_campaign_reads_the_pinned_numbers() -> None:
    scores, farm = run(CHANGEOVER)
    assert_pinned(scores, PREPOST_SCORES)
    assert_pinned(farm, PREPOST_FARM)
