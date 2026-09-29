"""Tests for the campaign declaration and the public spec derived from it."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from benchmarking.campaigns import CampaignSpec, SyntheticCampaign, layout_from_coords
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
        layout=layout_from_coords({f"T{i}": (57.5 + i * 0.01, -3.25) for i in range(1, 6)}, rotor_diameter_m=82.0),
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


def test_the_spec_carries_the_declared_layout_and_its_coordinates() -> None:
    declared = campaign()
    spec = declared.spec()
    assert spec.layout is declared.layout
    assert list(spec.layout.frame["rotor_diameter_m"]) == [82.0] * 5
    assert spec.coords == {f"T{i}": (57.5 + i * 0.01, -3.25) for i in range(1, 6)}


def test_layout_from_coords_needs_a_rotor_diameter() -> None:
    with pytest.raises(TypeError):
        layout_from_coords({"T1": (57.5, -3.25)})  # type: ignore[call-arg]


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


def test_usable_mask_keeps_every_record_of_an_excluded_turbine() -> None:
    # Excluded is a role, never a reference or tested; its data still carries its wake.
    spec = campaign().spec()
    index = pd.date_range(PERIOD[0], periods=5, freq="1h", tz="UTC")
    assert spec.usable_mask("T5", index).all()


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


def test_generate_keeps_a_turbine_the_campaign_does_not_declare() -> None:
    # Every turbine with data is a potential wake contributor.
    dataset = campaign().generate(scada(turbines=("T1", "T2", "T3", "T4", "T5", "T99")))
    assert "T99" in set(dataset.synthetic_df[HOT_COLUMNS.turbine])


def test_turbines_lists_every_declared_turbine() -> None:
    assert campaign().turbines == ["T1", "T2", "T3", "T4", "T5"]


def test_spec_is_a_campaign_spec() -> None:
    assert isinstance(campaign().spec(), CampaignSpec)


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def staggered(**overrides: object) -> SyntheticCampaign:
    """T1 worked 2020-03-01..05, T2 worked 2020-05-01..03, T3 worked later; no shared changeover."""
    fields: dict[str, object] = {
        "upgraded_turbines": ["T1", "T2"],
        "upgrade_timing": None,
        "candidate_references": ["T3", "T4"],
        "upgrades": [],
        "layout": layout_from_coords({f"T{i}": (57.5 + i * 0.01, -3.25) for i in range(1, 6)}, rotor_diameter_m=82.0),
        "north_offsets": None,
        "rated_power_kw": 2300.0,
        "analysis_period": None,
        "works": {
            "T1": [(utc("2020-03-01"), utc("2020-03-06"))],
            "T2": [(utc("2020-05-01"), utc("2020-05-04"))],
            "T3": [(utc("2020-06-10"), utc("2020-06-12"))],
        },
        "exclusions": [
            ("T4", utc("2020-02-01"), utc("2020-02-03")),
            (None, utc("2020-04-10"), utc("2020-04-11")),
        ],
    }
    fields.update(overrides)
    return SyntheticCampaign(**fields)  # type: ignore[arg-type]


def hours(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(utc(start), utc(end), freq="1h", inclusive="left")


class TestTheTimeline:
    def test_each_turbines_changeover_is_its_works_end(self) -> None:
        spec = staggered().spec()
        assert spec.timing_for("T1") == utc("2020-03-06")
        assert spec.timing_for("T2") == utc("2020-05-04")

    def test_treatment_starts_at_the_first_works_end(self) -> None:
        assert staggered().spec().treatment_start == utc("2020-03-06")

    def test_the_mode_is_prepost_without_a_shared_changeover(self) -> None:
        assert staggered().spec().mode == "prepost"

    def test_the_works_window_of_a_shared_changeover_has_no_length(self) -> None:
        assert campaign().spec().works_window("T1") == (CHANGEOVER, CHANGEOVER)

    def test_the_works_window_of_a_staggered_turbine_is_its_own(self) -> None:
        assert staggered().spec().works_window("T2") == (utc("2020-05-01"), utc("2020-05-04"))

    def test_an_analysed_turbine_needs_exactly_one_works_window(self) -> None:
        works = {"T1": [], "T2": [(utc("2020-05-01"), utc("2020-05-04"))]}
        with pytest.raises(ValueError, match="T1"):
            staggered(works=works).spec()

    def test_a_toggle_campaign_needs_one_declared_period(self) -> None:
        schedule = ToggleSchedule(period=pd.Timedelta(days=14), start=CHANGEOVER)
        with pytest.raises(ValueError, match="analysis_period"):
            staggered(upgrade_timing=schedule, works={}).spec()


class TestThePeriod:
    def test_a_tuple_period_serves_every_turbine(self) -> None:
        spec = campaign().spec()
        assert spec.period_for("T1") == PERIOD
        assert spec.period_bounds() == PERIOD
        assert not spec.uses_plans

    def test_no_period_means_plans_are_chosen(self) -> None:
        spec = staggered().spec()
        assert spec.period_for("T1") is None
        assert spec.period_bounds() is None
        assert spec.uses_plans

    def test_a_shared_changeover_without_a_period_uses_plans(self) -> None:
        assert dataclasses.replace(campaign().spec(), analysis_period=None).uses_plans

    def test_a_per_turbine_period_is_read_per_turbine(self) -> None:
        per = {"T1": PERIOD, "T2": (utc("2020-02-01"), utc("2020-09-01"))}
        spec = staggered(analysis_period=per).spec()
        assert spec.period_for("T2") == per["T2"]
        assert spec.period_bounds() == (PERIOD[0], utc("2020-09-01"))
        assert spec.uses_plans


class TestTheMasks:
    def test_a_turbines_own_works_are_masked(self) -> None:
        index = hours("2020-03-05", "2020-03-07")
        mask = staggered().spec().usable_mask("T1", index)
        assert not mask[index < utc("2020-03-06")].any()
        assert mask[index >= utc("2020-03-06")].all()

    def test_a_turbines_own_exclusion_is_masked_but_held_back(self) -> None:
        index = hours("2020-02-01", "2020-02-04")
        spec = staggered().spec()
        inside = index < utc("2020-02-03")
        assert not spec.usable_mask("T4", index)[inside].any()
        assert spec.held_back_mask("T4", index)[inside].all()
        assert not spec.held_back_mask("T4", index)[~inside].any()

    def test_another_turbines_exclusion_is_not_its_own(self) -> None:
        index = hours("2020-02-01", "2020-02-04")
        assert staggered().spec().usable_mask("T3", index).all()

    def test_a_farm_wide_exclusion_masks_everyone_and_holds_nothing_back(self) -> None:
        index = hours("2020-04-10", "2020-04-11")
        spec = staggered().spec()
        for wtg in ("T1", "T3", "T5"):
            assert not spec.usable_mask(wtg, index).any()
            assert not spec.held_back_mask(wtg, index).any()

    def test_the_end_of_a_window_is_exclusive(self) -> None:
        index = pd.DatetimeIndex([utc("2020-02-03")])
        assert staggered().spec().usable_mask("T4", index).all()

    def test_a_flat_campaign_masks_nothing(self) -> None:
        index = hours("2020-01-01", "2020-07-01")
        spec = campaign().spec()
        assert spec.usable_mask("T1", index).all()
        assert not spec.held_back_mask("T1", index).any()


def test_generate_injects_each_turbine_from_its_works_end() -> None:
    declared = staggered(upgrades=[ConstantCpChange(delta=0.05)], analysis_period=PERIOD)
    dataset = declared.generate(scada())
    for wtg in ("T1", "T2"):
        start = declared.works[wtg][0][1]
        syn = dataset.synthetic_df[dataset.synthetic_df[HOT_COLUMNS.turbine] == wtg]
        orig = dataset.original_df[dataset.original_df[HOT_COLUMNS.turbine] == wtg]
        changed = syn[HOT_COLUMNS.active_power].to_numpy() != orig[HOT_COLUMNS.active_power].to_numpy()
        assert not changed[syn.index < start].any()
        assert changed[syn.index >= start].all()


def test_generate_keeps_the_whole_frame_without_a_period() -> None:
    frame = scada()
    assert len(staggered().generate(frame).synthetic_df) == len(frame)


def test_the_spec_carries_the_timeline() -> None:
    declared = staggered()
    spec = declared.spec()
    assert spec.works == declared.works
    assert spec.exclusions == declared.exclusions
