"""Tests for the real campaigns declared from open data."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import yaml

from benchmarking.campaigns.loader import EXPECTED_UPLIFT_PREFIX, load_declaration
from benchmarking.campaigns.plans import plans_for
from benchmarking.campaigns.real import (
    BM_CURTAILMENT,
    HIGH_WIND_DERATE,
    HOT_AEROUP_T13,
    HOT_AEROUP_T13_CAMPAIGN,
    HOT_STATE_COL,
    HOT_TUNEUP,
    HOT_TUNEUP_CAMPAIGN,
    HOT_TUNEUP_UPGRADES,
    HOT_TUNEUP_UPGRADES_CAMPAIGN,
    ICING,
    NOISE_MODE,
    REAL_CAMPAIGNS,
    WORKS_COLUMNS,
    label_hot_site_states,
    works_table,
    write_real_campaign,
)
from benchmarking.synthetic.sources.hill_of_towie import (
    HOT_ACTIVE_POWER_MAX_COL,
    HOT_AMBIENT_TEMP_COL,
    HOT_COLUMNS,
    HOT_POWER_SETPOINT_COL,
    HOT_RATED_POWER_KW,
    HOT_TURBINE_COL,
)
from wind_up.analysis_period import MONTH
from wind_up.operating_state import PARTIAL_DOWNTIME

if TYPE_CHECKING:
    from pathlib import Path

WORKS = """Turbine,First date of AeroUp works,Last date of AeroUp works
T13,2021-09-23,2021-09-29
T14,2022-08-30,2022-10-03
"""


def test_the_t13_declaration_loads(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "scada.parquet").write_bytes(b"")
    (data / "works.csv").write_text(WORKS)
    (data / "turbines.csv").write_text(
        "name,latitude,longitude,rotor_diameter_m\nT13,57.50,-3.08,82\nT14,57.51,-3.08,82\n"
    )
    (tmp_path / "campaign.yaml").write_text(yaml.safe_dump(HOT_AEROUP_T13))
    spec = load_declaration(tmp_path / "campaign.yaml").spec
    assert spec.timing_for("T13") == pd.Timestamp("2021-09-30", tz="UTC")
    assert spec.wake_reach_d == pytest.approx(12.0)
    assert spec.analysis_period is None
    # the curtailment from September 2022 is labelled record by record, so nothing is declared over it
    assert spec.exclusions == []
    assert spec.operating_state_overrides == []


def test_the_tuneup_declaration_loads(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "scada.parquet").write_bytes(b"")
    aeroup = pd.read_csv(io.StringIO(WORKS + "T02,2022-07-18,2022-07-28\n"))
    works_table(aeroup, campaign=HOT_TUNEUP_CAMPAIGN).to_csv(data / "works.csv", index=False)
    names = sorted({*HOT_TUNEUP["turbines"]["upgraded"], "T14"})
    (data / "turbines.csv").write_text(
        "name,latitude,longitude,rotor_diameter_m\n"
        + "".join(f"{name},{57.5 + i / 100},-3.08,82\n" for i, name in enumerate(names))
    )
    (tmp_path / "campaign.yaml").write_text(yaml.safe_dump(HOT_TUNEUP))
    spec = load_declaration(tmp_path / "campaign.yaml").spec
    changeover = pd.Timestamp("2024-03-14 18:00", tz="UTC")
    assert len(spec.upgraded_turbines) == 9
    assert {spec.timing_for(t) for t in spec.upgraded_turbines} == {changeover}
    assert spec.candidate_references == ["T14"]
    assert spec.changeovers()["T13"] == [pd.Timestamp("2021-09-30", tz="UTC"), changeover]
    assert spec.wake_reach_d == pytest.approx(4.0)
    # an AeroUp row reaches as far as the AeroUp's expected uplift does
    assert spec.reach_of("T02", (pd.Timestamp("2022-07-18", tz="UTC"), pd.Timestamp("2022-07-29", tz="UTC"))) == 12.0
    assert spec.exclusions == []
    deployment = (pd.Timestamp("2024-03-14 07:00", tz="UTC"), changeover)
    for turbine in spec.upgraded_turbines:
        assert deployment in spec.works[turbine]
        assert spec.reach_of(turbine, deployment) == pytest.approx(spec.wake_reach_d)
    assert "T14" not in spec.works or deployment not in spec.works["T14"]


def test_the_tuneup_declared_as_two_upgrades_is_the_same_campaign(tmp_path: Path) -> None:
    aeroup = pd.read_csv(io.StringIO(WORKS + "T02,2022-07-18,2022-07-28\n"))
    names = sorted({*HOT_TUNEUP["turbines"]["upgraded"], "T14"})
    specs = {}
    for campaign in (HOT_TUNEUP_CAMPAIGN, HOT_TUNEUP_UPGRADES_CAMPAIGN):
        folder = tmp_path / campaign.name
        data = folder / "data"
        data.mkdir(parents=True)
        (data / "scada.parquet").write_bytes(b"")
        works_table(aeroup, campaign=campaign).to_csv(data / "works.csv", index=False)
        (data / "turbines.csv").write_text(
            "name,latitude,longitude,rotor_diameter_m\n"
            + "".join(f"{name},{57.5 + i / 100},-3.08,82\n" for i, name in enumerate(names))
        )
        (folder / "campaign.yaml").write_text(yaml.safe_dump(campaign.declaration))
        specs[campaign.name] = load_declaration(folder / "campaign.yaml").spec
    short, listed = specs["hot_tuneup"], specs["hot_tuneup_upgrades"]
    for name in ("upgraded_turbines", "upgrade_timing", "candidate_references", "works", "wake_reach_d"):
        assert getattr(listed, name) == getattr(short, name), name
    for turbine, windows in short.works.items():
        for window in windows:
            assert listed.reach_of(turbine, window) == pytest.approx(short.reach_of(turbine, window))
    assert listed.changeovers() == short.changeovers()
    assert [u["name"] for u in HOT_TUNEUP_UPGRADES["upgrades"]] == ["AeroUp", "TuneUp"]


def test_the_works_table_holds_the_aeroup_rows_then_the_deployment() -> None:
    aeroup = pd.read_csv(io.StringIO(WORKS))
    table = works_table(aeroup, campaign=HOT_TUNEUP_CAMPAIGN)
    assert list(table.columns) == list(WORKS_COLUMNS)
    assert table["Works"].tolist() == ["AeroUp"] * 2 + ["deployment"] * 9
    assert table.loc[table["Works"] == "AeroUp", EXPECTED_UPLIFT_PREFIX].tolist() == [0.05, 0.05]
    deployment = table[table["Works"] == "deployment"]
    assert set(deployment["Turbine"]) == set(HOT_TUNEUP["turbines"]["upgraded"])
    assert set(deployment["First date of works"]) == {"2024-03-14T07:00:00Z"}
    assert set(deployment["Last date of works"]) == {"2024-03-14T18:00:00Z"}
    assert set(deployment[EXPECTED_UPLIFT_PREFIX]) == {0.01}


def test_the_t13_works_table_is_the_aeroup_rows_at_the_campaigns_reach() -> None:
    table = works_table(pd.read_csv(io.StringIO(WORKS)), campaign=HOT_AEROUP_T13_CAMPAIGN)
    assert table["Works"].tolist() == ["AeroUp", "AeroUp"]
    assert table[EXPECTED_UPLIFT_PREFIX].isna().all()


def test_the_real_campaigns_end_their_scada_where_declared() -> None:
    assert set(REAL_CAMPAIGNS) == {"hot_aeroup_t13", "hot_tuneup", "hot_tuneup_upgrades"}
    assert HOT_TUNEUP_CAMPAIGN.end == pd.Timestamp("2024-09-01", tz="UTC")
    for campaign in REAL_CAMPAIGNS.values():
        assert campaign.start < campaign.end


@pytest.mark.slow
def test_the_t13_folder_plans_sensibly(tmp_path: Path) -> None:
    declaration = load_declaration(write_real_campaign(tmp_path, campaign=HOT_AEROUP_T13_CAMPAIGN))
    scada = pd.read_parquet(declaration.scada_path)
    plan = plans_for(declaration.spec, scada, columns=declaration.columns).plans["T13"]
    assert plan.pool_rule_met
    assert plan.pre >= 3 * MONTH
    assert plan.post >= 3 * MONTH
    for reference in plan.power_references:
        windows = declaration.spec.works.get(reference, [])
        assert all(not (w0 < plan.end and w1 > plan.start) for w0, w1 in windows)
    nearest = plan.table()["turbine"].iloc[0]
    assert nearest in plan.power_references


def setpoints(
    turbine: str,
    values: list[float],
    *,
    start: str = "2019-01-01",
    wind_speed: float | list[float] = 8.0,
    power: float | list[float] = 1000.0,
    power_max: float | list[float] | None = None,
    ambient: float | list[float] = 10.0,
) -> pd.DataFrame:
    """One turbine's consecutive ten-minute records with the given end setpoints; power max defaults to rated."""
    index = pd.date_range(start, periods=len(values), freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            HOT_TURBINE_COL: turbine,
            HOT_POWER_SETPOINT_COL: values,
            HOT_COLUMNS.wind_speed: wind_speed,
            HOT_COLUMNS.active_power: power,
            HOT_ACTIVE_POWER_MAX_COL: HOT_RATED_POWER_KW if power_max is None else power_max,
            HOT_AMBIENT_TEMP_COL: ambient,
        },
        index=index,
    )


def hot_labels(scada: pd.DataFrame) -> list[str | None]:
    return label_hot_site_states(scada).tolist()


def test_a_zero_setpoint_at_either_end_is_partial_downtime_before_the_bm_start() -> None:
    # The record after a 0 starts at 0.
    assert hot_labels(setpoints("T13", [2300.0, 0.0, 2300.0, 2300.0], start="2018-01-01")) == [
        None,
        PARTIAL_DOWNTIME,
        PARTIAL_DOWNTIME,
        None,
    ]


def test_a_zero_setpoint_is_curtailment_from_the_bm_start_even_in_high_wind() -> None:
    scada = setpoints("T13", [2300.0, 0.0, 2300.0, 2300.0], wind_speed=21.0)
    assert hot_labels(scada) == [None, BM_CURTAILMENT, BM_CURTAILMENT, None]


def test_the_startup_setpoint_has_no_label() -> None:
    # A start-up limit at one end outranks a curtailment at the other.
    assert hot_labels(setpoints("T13", [2300.0, 100.0, 1500.0, 1500.0])) == [None, None, None, BM_CURTAILMENT]


def test_a_reduced_setpoint_is_curtailment_only_from_the_bm_start() -> None:
    # The third record starts at the reduced setpoint and is the first at or after the BM start.
    scada = setpoints("T13", [1500.0, 1500.0, 2300.0, 2300.0], start="2018-11-06T23:40:00")
    assert hot_labels(scada) == [None, None, BM_CURTAILMENT, None]
    # Rated, and a NaN setpoint, are not reduced.
    assert hot_labels(setpoints("T13", [2300.0, np.nan, 2300.0])) == [None, None, None]


def test_a_reduced_setpoint_the_power_never_reaches_is_not_curtailment() -> None:
    # Power max reaches 90 % of the setpoint at either end only in the second and third records.
    scada = setpoints("T13", [2300.0, 1000.0, 1000.0, 1000.0], power=600.0, power_max=[700.0, 950.0, 900.0, 899.0])
    assert hot_labels(scada) == [None, BM_CURTAILMENT, BM_CURTAILMENT, None]
    # The record after a release still reaches the setpoint it started at.
    scada = setpoints("T13", [1000.0, 2300.0, 2300.0], power=900.0, power_max=[950.0, 1500.0, 1500.0])
    assert hot_labels(scada) == [BM_CURTAILMENT, BM_CURTAILMENT, None]


def test_a_noise_setpoint_is_noise_mode_on_its_own_turbine_only() -> None:
    scada = pd.concat([setpoints("T16", [2300.0, 1993.0]), setpoints("T13", [2300.0, 1993.0])])
    assert hot_labels(scada) == [None, NOISE_MODE, None, BM_CURTAILMENT]


def test_a_reduced_setpoint_in_high_wind_is_a_derate_at_any_date() -> None:
    scada = setpoints("T13", [2300.0, 2140.0, 2300.0, 2300.0], start="2017-01-01", wind_speed=[19.0, 21.0, 21.0, 21.0])
    assert hot_labels(scada) == [None, HIGH_WIND_DERATE, HIGH_WIND_DERATE, None]
    # Below the threshold it is curtailment.
    scada = setpoints("T13", [2300.0, 2140.0, 2300.0], wind_speed=[19.0, 19.9, 20.0])
    assert hot_labels(scada) == [None, BM_CURTAILMENT, HIGH_WIND_DERATE]


def test_a_noise_setpoint_in_high_wind_stays_noise_mode() -> None:
    assert hot_labels(setpoints("T16", [2300.0, 1993.0], wind_speed=21.0)) == [None, NOISE_MODE]


def warm_curve(*, records: int = 36, wind_speed: float = 8.0, power: float = 1000.0) -> pd.DataFrame:
    """T13's warm records at one wind speed, a day before the cold records."""
    return setpoints("T13", [2300.0] * records, start="2018-12-31", wind_speed=wind_speed, power=power)


def cold(power: list[float], *, setpoint: list[float] | None = None, wind_speed: float = 8.0) -> pd.DataFrame:
    """T13's consecutive cold records at one wind speed."""
    values = setpoint if setpoint is not None else [2300.0] * len(power)
    return setpoints("T13", values, start="2019-01-02", wind_speed=wind_speed, power=power, ambient=0.0)


def test_three_cold_records_far_below_the_warm_power_curve_are_icing() -> None:
    labels = hot_labels(pd.concat([warm_curve(), cold([250.0, 250.0, 250.0, 900.0, 250.0, 250.0])]))
    # The run of three is icing; the run of two after the 900 kW record is not.
    assert labels[36:] == [ICING, ICING, ICING, None, None, None]


def test_a_reduced_setpoint_breaks_an_icing_run() -> None:
    labels = hot_labels(pd.concat([warm_curve(), cold([250.0] * 4, setpoint=[2300.0, 2300.0, 1500.0, 2300.0])]))
    assert labels[36:] == [None, None, BM_CURTAILMENT, BM_CURTAILMENT]


def test_a_warm_record_breaks_an_icing_run() -> None:
    scada = pd.concat([warm_curve(), cold([250.0] * 5)])
    scada.iloc[38, scada.columns.get_loc(HOT_AMBIENT_TEMP_COL)] = 5.0
    assert hot_labels(scada)[36:] == [None] * 5


def test_icing_needs_a_well_populated_warm_power_curve_of_meaningful_power() -> None:
    # 35 warm records are too few for the curve.
    assert hot_labels(pd.concat([warm_curve(records=35), cold([250.0] * 3)]))[35:] == [None] * 3
    # At 4 m/s the warm curve is under the minimum power.
    scada = pd.concat([warm_curve(wind_speed=4.0, power=200.0), cold([50.0] * 3, wind_speed=4.0)])
    assert hot_labels(scada)[36:] == [None] * 3


def test_the_start_setpoint_is_nan_after_a_gap() -> None:
    scada = setpoints("T13", [0.0, 2300.0, 2300.0])
    scada.index = scada.index[:1].append(scada.index[1:] + pd.Timedelta(hours=1))
    assert hot_labels(scada) == [BM_CURTAILMENT, None, None]


def test_the_t13_declaration_carries_the_site_states(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "scada.parquet").write_bytes(b"")
    (data / "works.csv").write_text(WORKS)
    (data / "turbines.csv").write_text(
        "name,latitude,longitude,rotor_diameter_m\nT13,57.50,-3.08,82\nT14,57.51,-3.08,82\n"
    )
    (tmp_path / "campaign.yaml").write_text(yaml.safe_dump(HOT_AEROUP_T13))
    config = load_declaration(tmp_path / "campaign.yaml").operating_state
    assert config.site_state_column == HOT_STATE_COL
    assert config.parked_pitch_above_deg == 45
    assert set(config.site_states) == {NOISE_MODE, BM_CURTAILMENT, HIGH_WIND_DERATE, ICING}
    assert not config.site_states[NOISE_MODE].uplift
    assert config.site_states[BM_CURTAILMENT].waking == "part waking"
    for label in (HIGH_WIND_DERATE, ICING):
        assert not config.site_states[label].uplift
        assert config.site_states[label].waking == "part waking"
