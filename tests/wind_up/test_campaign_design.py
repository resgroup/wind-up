import itertools

import numpy as np
import pandas as pd
import pytest

from tests.wind_up.layouts import grid_layout, line_layout, offset, scatter_layout
from wind_up.campaign_design import check_design, design_campaign

# An east-west line with every gap different, so no two distances tie. 100 m rotors, so the gaps
# are 3, 4, 5, 6, 7 and 8.5 diameters. T6 has only two turbines within 20 diameters.
IRREGULAR_LINE = line_layout([0, 300, 700, 1200, 1800, 2500, 3350])


def _problems_about(report_problems: tuple[str, ...], name: str) -> list[str]:
    return [p for p in report_problems if p.startswith((f"{name} ", f"{name}'s "))]


# --- the checker -----------------------------------------------------------------------------


def test_pool_rule_allows_one_test_turbine_among_the_four_nearest() -> None:
    report = check_design(IRREGULAR_LINE, test_turbines=["T0", "T3"])
    assert report.compliant, report.problems


def test_pool_of_three_is_the_strict_rule() -> None:
    report = check_design(IRREGULAR_LINE, test_turbines=["T0", "T3"], reference_pool_size=3)
    assert not report.compliant
    assert _problems_about(report.problems, "T0") == [
        "T0 has only 2 non-test turbines among its 3 nearest (T3 is a test turbine)"
    ]
    assert _problems_about(report.problems, "T3") == []


def test_nearest_neighbour_must_not_be_a_test_turbine() -> None:
    report = check_design(IRREGULAR_LINE, test_turbines=["T0", "T1"])
    assert not report.compliant
    assert "T0's nearest neighbour T1 is a test turbine" in report.problems
    assert "T1's nearest neighbour T0 is a test turbine" in report.problems


def test_references_must_be_within_the_distance_limit() -> None:
    report = check_design(IRREGULAR_LINE, test_turbines=["T6"])
    assert report.problems == ("T6 has only 2 reference-eligible turbines within 20 rotor diameters",)
    assert check_design(IRREGULAR_LINE, test_turbines=["T6"], max_reference_distance_d=25).compliant


def test_references_may_be_shared() -> None:
    report = check_design(IRREGULAR_LINE, test_turbines=["T0", "T3"])
    table = report.table.set_index("test_turbine")
    assert [table.loc["T0", f"reference_{k}"] for k in (1, 2, 3)] == ["T1", "T2", "T4"]
    assert [table.loc["T3", f"reference_{k}"] for k in (1, 2, 3)] == ["T2", "T4", "T1"]


def test_compliance_table_gives_distances_ranks_and_front_row() -> None:
    table = check_design(IRREGULAR_LINE, test_turbines=["T0", "T3"]).table.set_index("test_turbine")
    assert table.loc["T0", "reference_1_distance_m"] == pytest.approx(300, abs=1e-6)
    assert table.loc["T0", "reference_1_distance_d"] == pytest.approx(3, abs=1e-8)
    assert table.loc["T0", "reference_1_rank"] == 1
    # T3 is T0's third nearest but a test turbine, so T4 (fourth nearest) is the third reference.
    assert table.loc["T0", "reference_3_rank"] == 4
    assert bool(table.loc["T0", "front_row"])
    assert bool(table.loc["T0", "reference_1_front_row"])
    assert bool(table.loc["T0", "compliant"])


def test_failing_turbine_lists_what_references_it_has() -> None:
    table = check_design(IRREGULAR_LINE, test_turbines=["T6"]).table.set_index("test_turbine")
    assert [table.loc["T6", f"reference_{k}"] for k in (1, 2)] == ["T5", "T4"]
    assert pd.isna(table.loc["T6", "reference_3"])
    assert not bool(table.loc["T6", "compliant"])


def test_front_row_share_blocks_overshoot() -> None:
    # 5x5 grid: 16 of 25 are front row, so 2 test turbines get a fair share of 1.28 -> exactly 1.
    report = check_design(grid_layout(rows=5, cols=5, spacing_m=300), test_turbines=["R0C0", "R4C4"])
    assert report.problems == (
        "2 front-row test turbines against a fair share of 1.28 (16 of 25 available turbines are front row); "
        "a compliant count is 1",
    )


def test_front_row_share_blocks_undershoot() -> None:
    report = check_design(grid_layout(rows=5, cols=5, spacing_m=300), test_turbines=["R1C1", "R3C3"])
    assert not report.compliant
    assert report.problems[0].startswith("0 front-row test turbines against a fair share of 1.28")


def test_front_row_share_is_met_by_the_nearest_whole_number() -> None:
    assert check_design(grid_layout(rows=5, cols=5, spacing_m=300), test_turbines=["R0C0", "R2C2"]).compliant


def test_front_row_share_tie_accepts_either_count() -> None:
    # 4x4 grid: 12 of 16 front row, so 2 test turbines get a fair share of exactly 1.5.
    layout = grid_layout(rows=4, cols=4, spacing_m=300)
    assert check_design(layout, test_turbines=["R0C0", "R0C3"]).compliant
    assert check_design(layout, test_turbines=["R0C0", "R2C2"]).compliant
    assert not check_design(layout, test_turbines=["R1C1", "R2C2"]).compliant


OUTER_RING = [f"R{r}C{c}" for r in range(5) for c in range(5) if r in (0, 4) or c in (0, 4)]


def test_excluded_turbines_leave_the_front_row_share() -> None:
    layout = grid_layout(rows=5, cols=5, spacing_m=300)
    assert not check_design(layout, test_turbines=["R2C2"]).compliant
    # Excluding 12 front-row turbines leaves 4 of 13: a fair share of 0.31, so none is compliant.
    assert check_design(layout, test_turbines=["R2C2"], excluded=OUTER_RING[:12]).compliant


def test_reference_only_turbines_stay_in_the_front_row_share() -> None:
    layout = grid_layout(rows=5, cols=5, spacing_m=300)
    assert not check_design(layout, test_turbines=["R2C2"], reference_only=OUTER_RING[:12]).compliant


def test_excluded_and_reference_only_turbines_cannot_be_tested() -> None:
    report = check_design(IRREGULAR_LINE, test_turbines=["T0", "T3"], excluded=["T3"], reference_only=["T0"])
    assert "T0 is reference-only, so it cannot be a test turbine" in report.problems
    assert "T3 is excluded, so it cannot be a test turbine" in report.problems


def test_excluded_turbines_are_never_references() -> None:
    table = check_design(IRREGULAR_LINE, test_turbines=["T0"], excluded=["T1"]).table.set_index("test_turbine")
    assert [table.loc["T0", f"reference_{k}"] for k in (1, 2, 3)] == ["T2", "T3", "T4"]


def test_reference_only_turbines_can_be_references() -> None:
    table = check_design(IRREGULAR_LINE, test_turbines=["T0"], reference_only=["T1"]).table.set_index("test_turbine")
    assert table.loc["T0", "reference_1"] == "T1"


def test_turbines_of_another_farm_are_neither_test_turbines_nor_references() -> None:
    far_farm = scatter_layout([(0, 5000), (300, 5000), (700, 5000), (1200, 5000)]).assign(
        name=["N0", "N1", "N2", "N3"], wind_farm="Neighbour"
    )
    layout = pd.concat([IRREGULAR_LINE.assign(wind_farm="Home"), far_farm], ignore_index=True)
    report = check_design(layout, wind_farm="Home", test_turbines=["T0", "N0"])
    assert "N0 is not a turbine of Home, so it cannot be a test turbine" in report.problems


def test_summary_counts() -> None:
    summary = check_design(
        grid_layout(rows=5, cols=5, spacing_m=300), test_turbines=["R0C0", "R2C2"], excluded=["R4C4"]
    ).summary
    assert summary["farm_turbines"] == 25
    assert summary["available_turbines"] == 24
    assert summary["test_turbines"] == 2
    assert summary["available_front_row"] == 15
    assert summary["front_row_test_turbines"] == 1
    assert summary["fair_front_row_share"] == pytest.approx(2 * 15 / 24)


def test_checker_raises_on_unknown_names() -> None:
    with pytest.raises(ValueError, match="T99"):
        check_design(IRREGULAR_LINE, test_turbines=["T99"])


# --- the selector ----------------------------------------------------------------------------


def test_design_passes_its_own_check() -> None:
    design = design_campaign(grid_layout(rows=5, cols=5, spacing_m=300))
    assert design.compliance.compliant
    assert design.compliance == check_design(
        grid_layout(rows=5, cols=5, spacing_m=300), test_turbines=design.test_turbines
    )


def test_a_higher_priority_neighbour_takes_the_next_as_its_reference() -> None:
    design = design_campaign(IRREGULAR_LINE, test_priority=["T2", "T1"], n_test=2)
    outcomes = {o.name: o for o in design.candidates}
    assert outcomes["T2"].outcome == "test"
    assert outcomes["T1"].outcome == "skipped"
    assert outcomes["T1"].reason == "reference of T2 (its nearest neighbour)"
    assert design.references["T2"][0] == "T1"


def test_turbines_without_enough_references_are_skipped_with_the_reason() -> None:
    design = design_campaign(IRREGULAR_LINE, test_priority=["T6"])
    t6 = next(o for o in design.candidates if o.name == "T6")
    assert t6.outcome == "skipped"
    assert t6.reason == "only 2 reference-eligible turbines within 20 rotor diameters"


def test_count_is_maximised_before_priority() -> None:
    layout = grid_layout(rows=5, cols=5, spacing_m=300)
    unconstrained = design_campaign(layout)
    # R2C2 is in no design of the maximum size, so it gives way; R0C3 and R4C1 are, so they stay.
    prioritised = design_campaign(layout, test_priority=["R2C2", "R0C3", "R4C1"])
    assert len(prioritised.test_turbines) == len(unconstrained.test_turbines) == unconstrained.max_test_turbines
    assert "R2C2" not in prioritised.test_turbines
    assert prioritised.test_turbines[:2] == ("R0C3", "R4C1")


def test_turbines_after_the_last_commit_are_not_needed() -> None:
    design = design_campaign(grid_layout(rows=5, cols=5, spacing_m=300), n_test=1, test_priority=["R0C0"])
    assert design.test_turbines == ("R0C0",)
    assert {o.outcome for o in design.candidates[1:]} == {"not needed"}
    assert all(o.reason is None for o in design.candidates[1:])


def test_unlisted_turbines_follow_in_a_seeded_order() -> None:
    layout = grid_layout(rows=5, cols=5, spacing_m=300)
    first = design_campaign(layout, test_priority=["R0C0"], seed=3)
    again = design_campaign(layout, test_priority=["R0C0"], seed=3)
    assert first.test_turbines == again.test_turbines
    assert first.candidates[0].name == "R0C0"
    assert first.candidates[0].from_test_priority
    assert not any(o.from_test_priority for o in first.candidates[1:])
    orders = {tuple(o.name for o in design_campaign(layout, seed=s).candidates) for s in range(5)}
    assert len(orders) == 5


def test_requesting_more_than_the_site_supports_says_the_maximum() -> None:
    layout = grid_layout(rows=5, cols=5, spacing_m=300)
    most = design_campaign(layout).max_test_turbines
    with pytest.raises(ValueError, match=f"at most {most}"):
        design_campaign(layout, n_test=most + 1)


def test_maximum_is_reported_when_a_count_is_requested() -> None:
    layout = grid_layout(rows=5, cols=5, spacing_m=300)
    design = design_campaign(layout, n_test=2)
    assert len(design.test_turbines) == 2
    assert design.max_test_turbines == design_campaign(layout).max_test_turbines


def test_a_farm_too_small_for_any_test_turbine_says_so() -> None:
    with pytest.raises(ValueError, match="needs at least 4 available turbines"):
        design_campaign(line_layout([0, 300, 700]))


def test_a_farm_too_sparse_for_any_test_turbine_says_so() -> None:
    with pytest.raises(ValueError, match="no candidate has 3 reference-eligible turbines within 20 rotor diameters"):
        design_campaign(line_layout([0, 1500, 3000, 4500]))


def test_four_turbines_support_one_test_turbine() -> None:
    design = design_campaign(scatter_layout([(0, 0), (300, 0), (0, 400), (350, 380)]))
    assert len(design.test_turbines) == 1


def test_reference_only_turbines_are_never_tested() -> None:
    design = design_campaign(grid_layout(rows=5, cols=5, spacing_m=300), reference_only=OUTER_RING[:8])
    assert design.test_turbines
    assert not set(design.test_turbines) & set(OUTER_RING[:8])


def test_share_rule_can_leave_no_compliant_design() -> None:
    # Every front-row turbine reference-only: no test set can carry its front-row share.
    with pytest.raises(ValueError, match="front-row share"):
        design_campaign(grid_layout(rows=5, cols=5, spacing_m=300), reference_only=OUTER_RING)


def test_roles_offer_every_available_non_test_turbine_as_a_reference() -> None:
    design = design_campaign(IRREGULAR_LINE, excluded=["T5"], n_test=1, test_priority=["T0"])
    assert design.roles() == {
        "upgraded": ["T0"],
        "references": ["T1", "T2", "T3", "T4", "T6"],
        "excluded": ["T5"],
    }


def test_rotor_diameter_fills_are_reported() -> None:
    layout = IRREGULAR_LINE.assign(rotor_diameter_m=[100.0, np.nan, 100, 100, 100, 100, 100])
    assert design_campaign(layout).filled_rotor_diameters == ("T1",)


# --- input validation -------------------------------------------------------------------------


def test_farm_turbines_must_be_named() -> None:
    layout = IRREGULAR_LINE.assign(name=["T0", None, "T2", "T3", "T4", "T5", "T6"])
    with pytest.raises(ValueError, match="name"):
        design_campaign(layout)


def test_unnamed_turbines_of_other_farms_are_fine() -> None:
    lat, lon = offset(east_m=0, north_m=6000)
    stranger = pd.DataFrame([{"latitude": lat, "longitude": lon, "wind_farm": None}])
    layout = pd.concat([IRREGULAR_LINE.assign(wind_farm="Home"), stranger], ignore_index=True)
    assert design_campaign(layout, wind_farm="Home").compliance.compliant


def test_a_name_may_hold_one_role() -> None:
    with pytest.raises(ValueError, match="T1"):
        design_campaign(IRREGULAR_LINE, test_priority=["T1"], excluded=["T1"])


def test_role_names_must_be_farm_turbines() -> None:
    with pytest.raises(ValueError, match="T99"):
        design_campaign(IRREGULAR_LINE, reference_only=["T99"])


def test_wind_farm_defaults_to_the_only_one() -> None:
    assert design_campaign(IRREGULAR_LINE.assign(wind_farm="Home")).wind_farm == "Home"


def test_wind_farm_must_be_chosen_when_there_are_several() -> None:
    layout = IRREGULAR_LINE.assign(wind_farm=["A", "A", "A", "A", "B", "B", "B"])
    with pytest.raises(ValueError, match="wind_farm"):
        design_campaign(layout)


def test_wind_farm_must_exist() -> None:
    with pytest.raises(ValueError, match="Nowhere"):
        design_campaign(IRREGULAR_LINE.assign(wind_farm="Home"), wind_farm="Nowhere")


# --- brute force ------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_design_matches_brute_force(seed: int) -> None:
    """The solver's maximum is the true maximum; the walk returns the first compliant set in priority order."""
    rng = np.random.default_rng(seed)
    layout = scatter_layout([tuple(p) for p in rng.uniform(0, 1500, size=(8, 2))])
    names = list(layout["name"])
    priority = [str(n) for n in rng.permutation(names)]
    rank = {name: i for i, name in enumerate(priority)}

    compliant = [
        combo
        for size in range(1, len(names) + 1)
        for combo in itertools.combinations(names, size)
        if check_design(layout, test_turbines=list(combo)).compliant
    ]
    most = max(len(c) for c in compliant)
    first = min((c for c in compliant if len(c) == most), key=lambda c: sorted(rank[n] for n in c))

    design = design_campaign(layout, test_priority=priority)
    assert design.max_test_turbines == most
    assert set(design.test_turbines) == set(first)
