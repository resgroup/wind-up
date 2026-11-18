"""Tests for the toggle energy ratio method answering analyses of wind farm facts."""

from __future__ import annotations

import dataclasses
import math

import pandas as pd
import pytest

from wind_up.analysis import CAMPAIGN, PER_TURBINE_COLUMNS, AnalysisSpec
from wind_up.campaign_context import CampaignContext
from wind_up.facts import Upgrade, WindFarmFacts
from wind_up.labels import Label
from wind_up.method import MethodConfigurationError, MethodInput, MethodOutput
from wind_up.methods.toggle_energy_ratio import ToggleEnergyRatioMethod

from .test_toggle_energy_ratio import _COLUMNS, _CYCLE, _DATUM, _RATED_KW, _TURBINE_COL, _noisy_toggle_case

_POWER = _COLUMNS.active_power


def _turbines(names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": names,
            "latitude": [57.5 + 0.005 * i for i in range(len(names))],
            "longitude": -3.0,
            "rated_power_kw": _RATED_KW,
        }
    )


def _facts(scada: pd.DataFrame, *, subjects: tuple[str, ...] = ("T1",), **fields: object) -> WindFarmFacts:
    _, schedule = _noisy_toggle_case()
    names = list(pd.unique(scada[_TURBINE_COL]))
    return WindFarmFacts(
        records=scada,
        columns=_COLUMNS,
        timebase=pd.Timedelta(minutes=10),
        turbines=_turbines(names),
        upgrades=[Upgrade(name="control", turbines=subjects, timing=schedule)],
        **fields,  # type: ignore[arg-type]
    )


def _old(
    scada: pd.DataFrame, *, test: str = "T1", references: list[str] | None = None, **context: object
) -> MethodOutput:
    _, schedule = _noisy_toggle_case()
    method = ToggleEnergyRatioMethod(
        columns=_COLUMNS, toggle_period=_CYCLE, toggle_datum=_DATUM, conditions=("power",), rated_power_kw=_RATED_KW
    )
    if references is None and not context:
        return method.estimate(
            MethodInput(scada_df=scada, test_wtg=test, upgrade_timing=schedule, turbine_col=_TURBINE_COL)
        )  # type: ignore[return-value]
    base = CampaignContext.from_frame(scada, test_wtg=test, timing=schedule, turbine_col=_TURBINE_COL)
    refs = base.candidate_references if references is None else references
    valid = context.get("valid_for_uplift", base.valid_for_uplift)
    built = CampaignContext(
        test_wtg=test,
        timing=schedule,
        turbine_col=_TURBINE_COL,
        candidate_references=refs,
        wake_contributors=[t for t in base.valid_for_uplift.columns if t != test and t not in refs],
        valid_for_uplift=valid,  # type: ignore[arg-type]
    )
    return method.estimate(MethodInput(scada_df=scada, test_wtg=test, campaign_context=built))  # type: ignore[return-value]


def _new(facts: WindFarmFacts, *analyses: AnalysisSpec) -> list:
    return ToggleEnergyRatioMethod(conditions=("power",)).estimate(facts, analyses)  # type: ignore[return-value]


def _second_subject(scada: pd.DataFrame, *, scale: float = 0.9) -> pd.DataFrame:
    """The fixture with a second upgraded turbine T2, T1's records scaled."""
    t2 = scada[scada[_TURBINE_COL] == "T1"].assign(**{_TURBINE_COL: "T2"})
    t2[_POWER] = t2[_POWER] * scale
    return pd.concat([scada, t2]).sort_index(kind="stable")


def test_one_subject_reads_what_the_method_input_reads() -> None:
    scada, _ = _noisy_toggle_case()
    old = _old(scada)
    (out,) = _new(_facts(scada), AnalysisSpec(name="control", upgrade="control"))
    assert out.campaign_uplift == old.p50_overall
    assert out.campaign_sigma == old.sigma_overall
    by_condition = out.by_condition.drop(columns="turbine").reset_index(drop=True)
    pd.testing.assert_frame_equal(by_condition, old.p50_by_condition.reset_index(drop=True))
    row = out.per_turbine.iloc[0]
    assert list(out.per_turbine.columns) == list(PER_TURBINE_COLUMNS)
    assert row["turbine"] == "T1"
    assert row["campaign_uplift"] == old.p50_overall
    used = old.labeled_rows[old.labeled_rows["used"].astype(bool)]
    assert row["actual_energy"] == pytest.approx(float(used.loc[used["segment"] == "upgraded", _POWER].sum()))
    assert row["hours_on"] == pytest.approx((used["segment"] == "upgraded").sum() / 6)
    assert row["hours_off"] == pytest.approx((used["segment"] == "baseline").sum() / 6)
    assert math.isnan(out.aep_uplift)


def test_the_named_references_are_the_ones_used() -> None:
    scada, _ = _noisy_toggle_case()
    old = _old(scada, references=["R2"])
    (out,) = _new(_facts(scada), AnalysisSpec(name="control", upgrade="control", references=["R2"]))
    assert out.campaign_uplift == old.p50_overall
    assert out.campaign_sigma == old.sigma_overall


def test_a_declared_exclusion_leaves_out_the_turbines_records() -> None:
    scada, _ = _noisy_toggle_case()
    start, end = pd.Timestamp("2020-01-03", tz="UTC"), pd.Timestamp("2020-01-06", tz="UTC")
    index = pd.DatetimeIndex(pd.unique(scada.index))
    valid = pd.DataFrame(data=True, index=index, columns=["R1", "R2", "T1"])
    valid.loc[(index >= start) & (index < end), "T1"] = False
    old = _old(scada, valid_for_uplift=valid)
    (out,) = _new(_facts(scada, exclusions=[("T1", start, end)]), AnalysisSpec(name="control", upgrade="control"))
    assert out.campaign_uplift == old.p50_overall
    assert out.campaign_sigma == old.sigma_overall
    assert out.campaign_uplift != _old(scada).p50_overall


def test_a_period_restricts_the_records_compared() -> None:
    scada, _ = _noisy_toggle_case()
    start = pd.Timestamp("2020-01-05", tz="UTC")
    old = _old(scada[scada.index >= start])
    (out,) = _new(_facts(scada), AnalysisSpec(name="later", upgrade="control", period=(start, None)))
    assert out.campaign_uplift == pytest.approx(old.p50_overall, rel=1e-12)
    end = pd.Timestamp("2020-01-10", tz="UTC")
    (cut,) = _new(_facts(scada), AnalysisSpec(name="cut", upgrade="control", period=(start, end)))
    old_cut = _old(scada[(scada.index >= start) & (scada.index < end)])
    assert cut.campaign_uplift == pytest.approx(old_cut.p50_overall, rel=1e-12)


def test_a_group_is_weighted_by_the_energy_with_the_upgrade_on_and_has_no_group_sigma() -> None:
    scada = _second_subject(_noisy_toggle_case()[0])
    facts = _facts(scada, subjects=("T1", "T2"))
    (out,) = _new(facts, AnalysisSpec(name="both", upgrade="control", references=["R1", "R2"]))
    rows = out.per_turbine.set_index("turbine")
    assert list(rows.index) == ["T1", "T2"]
    assert rows.loc["T1", "campaign_uplift"] == _old(scada, references=["R1", "R2"]).p50_overall
    energy = rows["actual_energy"]
    without = energy / (1 + rows["campaign_uplift"])
    assert out.campaign_uplift == pytest.approx(energy.sum() / without.sum() - 1)
    assert math.isnan(out.campaign_sigma)
    assert any(w.severity == "low" and "no sigma for a group" in w.message for w in out.warnings)
    assert set(out.labeled_rows["turbine"]) == {"T1", "T2"}
    assert set(out.selection_accounting["turbine"]) == {"T1", "T2"}
    assert out.uncertainty_diagnostics.columns[0] == "turbine"


def test_each_analysis_is_answered_in_order() -> None:
    scada = _second_subject(_noisy_toggle_case()[0])
    facts = _facts(scada, subjects=("T1", "T2"))
    analyses = [AnalysisSpec(name=n, upgrade="control", subjects=[n], references=["R1", "R2"]) for n in ("T2", "T1")]
    outs = _new(facts, *analyses)
    assert [o.analysis.name for o in outs] == ["T2", "T1"]
    assert [o.per_turbine["turbine"].tolist() for o in outs] == [["T2"], ["T1"]]


def test_no_aep_uplift_is_given_and_a_warning_says_so_when_asked_for() -> None:
    scada, _ = _noisy_toggle_case()
    facts = _facts(scada)
    (both,) = _new(facts, AnalysisSpec(name="x", upgrade="control"))
    assert any(w.severity == "medium" and "no AEP uplift" in w.message for w in both.warnings)
    (campaign,) = _new(facts, AnalysisSpec(name="x", upgrade="control", report=CAMPAIGN))
    assert not any("no AEP uplift" in w.message for w in campaign.warnings)
    assert math.isfinite(campaign.campaign_uplift)
    (aep,) = _new(facts, AnalysisSpec(name="x", upgrade="control", report="aep"))
    assert math.isnan(aep.campaign_uplift)
    assert math.isnan(aep.campaign_sigma)


def test_a_subject_that_cannot_be_measured_is_a_row_of_nan_and_a_warning() -> None:
    scada, _ = _noisy_toggle_case()
    r1 = scada[scada[_TURBINE_COL] == "R1"]
    extra = pd.concat([r1.assign(**{_TURBINE_COL: f"R{i}"}) for i in range(3, 10)])
    facts = _facts(pd.concat([scada, extra]).sort_index(kind="stable"))
    (out,) = _new(facts, AnalysisSpec(name="x", upgrade="control"))
    assert math.isnan(out.campaign_uplift)
    assert out.per_turbine["turbine"].tolist() == ["T1"]
    assert out.per_turbine.drop(columns="turbine").isna().all(axis=None)
    assert out.by_condition is None
    high = [w for w in out.warnings if w.severity == "high"]
    assert any("subset estimates" in w.message for w in high)
    assert any("measured none of the subjects" in w.message for w in high)


def test_a_configuration_error_fails_the_analysis() -> None:
    scada, _ = _noisy_toggle_case()
    method = ToggleEnergyRatioMethod(pairing_max_gap=pd.Timedelta(minutes=15))
    with pytest.raises(MethodConfigurationError, match="pairing_max_gap"):
        method.estimate(_facts(scada), [AnalysisSpec(name="x", upgrade="control")])


def test_a_permanent_upgrade_is_refused() -> None:
    scada, _ = _noisy_toggle_case()
    facts = _facts(scada)
    permanent = Upgrade(name="blades", turbines=["T1"], timing=pd.Timestamp("2020-01-07", tz="UTC"))
    facts = WindFarmFacts(
        records=facts.records,
        columns=facts.columns,
        timebase=facts.timebase,
        turbines=facts.turbines,
        upgrades=[permanent],
    )
    with pytest.raises(MethodConfigurationError, match="toggle tests only"):
        _new(facts, AnalysisSpec(name="x", upgrade="blades"))


def test_the_facts_need_the_power_and_availability_roles() -> None:
    scada, _ = _noisy_toggle_case()
    facts = _facts(scada)
    without = WindFarmFacts(
        records=facts.records,
        columns=facts.columns.__class__(turbine=_TURBINE_COL, active_power=_POWER),
        timebase=facts.timebase,
        turbines=facts.turbines,
        upgrades=facts.upgrades,
    )
    with pytest.raises(ValueError, match="availability"):
        _new(without, AnalysisSpec(name="x", upgrade="control"))


def test_records_left_out_by_a_label_are_filtered_like_excluded_rows() -> None:
    scada, schedule = _noisy_toggle_case()
    left_out = (scada.index >= pd.Timestamp("2020-01-03", tz="UTC")) & (
        scada.index < pd.Timestamp("2020-01-05", tz="UTC")
    )
    flagged = scada.assign(flag=left_out & (scada[_TURBINE_COL] == "T1").to_numpy())
    old = ToggleEnergyRatioMethod(
        columns=dataclasses.replace(_COLUMNS, exclude_row="flag"),
        toggle_period=_CYCLE,
        toggle_datum=_DATUM,
        conditions=("power",),
        rated_power_kw=_RATED_KW,
    ).estimate(MethodInput(scada_df=flagged, test_wtg="T1", upgrade_timing=schedule, turbine_col=_TURBINE_COL))
    facts = _facts(scada.assign(flag=left_out), labels=[Label(name="flag", column="flag")])
    (out,) = _new(facts, AnalysisSpec(name="x", upgrade="control", exclude_labels=["flag"]))
    assert out.campaign_uplift == old.p50_overall
    assert out.campaign_sigma == old.sigma_overall
    (kept,) = _new(facts, AnalysisSpec(name="x", upgrade="control", include_labels=["flag"]))
    assert kept.per_turbine["hours_on"].iloc[0] < out.per_turbine["hours_on"].iloc[0]
