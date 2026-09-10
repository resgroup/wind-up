"""Tests for the campaign declaration and the public spec derived from it."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns import CampaignSpec, SyntheticCampaign
from benchmarking.synthetic import HOT_COLUMNS, ConstantCpChange, ToggleSchedule

PERIOD = (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-07-01", tz="UTC"))
CHANGEOVER = pd.Timestamp("2020-04-01", tz="UTC")


def campaign(*, upgrades: list | None = None, upgrade_timing: object = CHANGEOVER) -> SyntheticCampaign:
    """A five-turbine campaign: T1/T2 upgraded, T3/T4 references, T5 excluded."""
    return SyntheticCampaign(
        upgraded_turbines=["T1", "T2"],
        upgrade_timing=upgrade_timing,
        candidate_references=["T3", "T4", "T5"],
        excluded_turbines=["T5"],
        upgrades=[] if upgrades is None else upgrades,
        coords={f"T{i}": (57.5 + i * 0.01, -3.25) for i in range(1, 6)},
        north_offsets=[("T1", pd.Timestamp("2020-01-01", tz="UTC"), 1.5)],
        rated_power_kw=2300.0,
        analysis_period=PERIOD,
    )


def scada(turbines: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5")) -> pd.DataFrame:
    """A tiny hourly frame over the campaign period, flat power, fully available."""
    index = pd.date_range(PERIOD[0], PERIOD[1], freq="1h", tz="UTC", inclusive="left")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    HOT_COLUMNS.turbine: wtg,
                    HOT_COLUMNS.active_power: 900.0,
                    HOT_COLUMNS.active_power_min: 850.0,
                    HOT_COLUMNS.wind_speed: 8.0,
                    HOT_COLUMNS.wind_speed_sd: 0.8,
                    HOT_COLUMNS.gen_rpm: 1400.0,
                    HOT_COLUMNS.availability: 3600.0,
                },
                index=index,
            )
            for wtg in turbines
        ]
    )


def test_spec_exposes_no_upgrade_physics() -> None:
    spec = campaign(upgrades=[ConstantCpChange(delta=0.05)]).spec()
    assert "upgrades" not in {f.name for f in dataclasses.fields(spec)}
    assert "0.05" not in repr(spec)


def test_spec_carries_the_public_facts() -> None:
    spec = campaign().spec()
    assert spec.upgraded_turbines == ["T1", "T2"]
    assert spec.candidate_references == ["T3", "T4", "T5"]
    assert spec.excluded_turbines == ["T5"]
    assert spec.rated_power_kw == 2300.0
    assert spec.analysis_period == PERIOD
    assert spec.turbine_col == HOT_COLUMNS.turbine


def test_mode_is_prepost_for_a_changeover_timestamp() -> None:
    assert campaign().spec().mode == "prepost"


def test_mode_is_toggle_for_a_schedule() -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=4), start=CHANGEOVER)
    assert campaign(upgrade_timing=schedule).spec().mode == "toggle"


def test_timing_for_returns_the_same_timing_for_every_upgraded_turbine() -> None:
    spec = campaign().spec()
    assert spec.timing_for("T1") == CHANGEOVER
    assert spec.timing_for("T2") == CHANGEOVER


def test_timing_for_rejects_a_turbine_that_is_not_upgraded() -> None:
    with pytest.raises(KeyError, match="T3"):
        campaign().spec().timing_for("T3")


def test_usable_mask_keeps_every_record_of_a_participating_turbine() -> None:
    spec = campaign().spec()
    index = pd.date_range(PERIOD[0], periods=5, freq="1h", tz="UTC")
    assert spec.usable_mask("T3", index).all()
    assert spec.usable_mask("T1", index).all()


def test_usable_mask_drops_every_record_of_an_excluded_turbine() -> None:
    spec = campaign().spec()
    index = pd.date_range(PERIOD[0], periods=5, freq="1h", tz="UTC")
    assert not spec.usable_mask("T5", index).any()


def test_usable_mask_is_a_boolean_array_matching_the_index() -> None:
    spec = campaign().spec()
    index = pd.date_range(PERIOD[0], periods=7, freq="1h", tz="UTC")
    assert spec.usable_mask("T3", index).shape == (7,)
    assert spec.usable_mask("T3", index).dtype == np.bool_


def test_change_label_is_neutral() -> None:
    assert campaign().spec().change_label() == "the change"


def test_treatment_start_is_the_changeover_for_prepost() -> None:
    assert campaign().spec().treatment_start == CHANGEOVER


def test_treatment_start_is_the_schedule_start_for_toggle() -> None:
    schedule = ToggleSchedule(period=pd.Timedelta(hours=4), start=CHANGEOVER)
    assert campaign(upgrade_timing=schedule).spec().treatment_start == CHANGEOVER


def test_generate_returns_an_unchanged_dataset_when_there_are_no_upgrades() -> None:
    dataset = campaign().generate(scada())
    pd.testing.assert_frame_equal(dataset.synthetic_df, dataset.original_df)


def test_generate_injects_the_declared_upgrade() -> None:
    dataset = campaign(upgrades=[ConstantCpChange(delta=0.05)]).generate(scada())
    assert not dataset.synthetic_df[HOT_COLUMNS.active_power].equals(dataset.original_df[HOT_COLUMNS.active_power])


def test_generate_restricts_the_data_to_the_analysis_period() -> None:
    wide = scada()
    earlier = wide.copy()
    earlier.index = earlier.index - pd.Timedelta(days=90)
    dataset = campaign().generate(pd.concat([earlier, wide]))
    assert dataset.synthetic_df.index.min() >= PERIOD[0]
    assert dataset.synthetic_df.index.max() < PERIOD[1]


def test_generate_drops_turbines_the_campaign_does_not_declare() -> None:
    dataset = campaign().generate(scada(turbines=("T1", "T2", "T3", "T4", "T5", "T99")))
    assert "T99" not in set(dataset.synthetic_df[HOT_COLUMNS.turbine])


def test_turbines_lists_every_declared_turbine() -> None:
    assert campaign().turbines == ["T1", "T2", "T3", "T4", "T5"]


def test_spec_is_a_campaign_spec() -> None:
    assert isinstance(campaign().spec(), CampaignSpec)
