"""Tests for judging a prepost matrix study against method.md's validation targets."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.prepost_matrix.validation import (
    TARGET_COVERAGE,
    clustered_se,
    component_sigmas,
    describe,
    in_domain,
    lambda_star,
    lambda_uplift,
    period_months,
    season,
    turbine_frame,
    usable_share_table,
    validation_tables,
    write_report,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_lambda_star_trusts_a_precise_reverse_against_a_real_bias() -> None:
    lam = lambda_star(np.array([0.06]), np.array([-0.02]), sigma_forward=np.array([0.001]), sigma_reverse=[0.001])
    assert lam[0] == pytest.approx(1.0, abs=0.01)


def test_lambda_star_falls_to_zero_for_a_useless_reverse() -> None:
    lam = lambda_star(np.array([0.06]), np.array([-0.02]), sigma_forward=np.array([0.005]), sigma_reverse=[10.0])
    assert lam[0] < 0.01


def test_lambda_star_is_nan_without_a_sigma() -> None:
    assert np.isnan(lambda_star(np.array([0.06]), np.array([-0.02]), sigma_forward=[np.nan], sigma_reverse=[0.01])[0])


def test_lambda_uplift_runs_from_forward_to_full_reversal() -> None:
    forward, reverse = np.array([0.05]), np.array([-0.03])
    assert lambda_uplift(forward, reverse, np.array([0.0]))[0] == pytest.approx(0.05)
    full = np.sqrt(1.05 / 0.97) - 1
    assert lambda_uplift(forward, reverse, np.array([1.0]))[0] == pytest.approx(full)


def test_season_is_meteorological() -> None:
    assert [season(f"2020-{m:02d}-15") for m in (1, 4, 7, 10, 12)] == ["DJF", "MAM", "JJA", "SON", "DJF"]


def test_clustered_se_ignores_repeats_within_a_campaign() -> None:
    values = np.array([0.01, -0.02, 0.03, 0.0])
    campaigns = np.array(["a", "b", "c", "d"])
    once = clustered_se(values, campaigns)
    repeated = clustered_se(np.repeat(values, 5), np.repeat(campaigns, 5))
    assert repeated == pytest.approx(once)
    assert once == pytest.approx(np.sqrt(4 / 3) * values.std(ddof=0) / 2)


def test_clustered_se_needs_two_campaigns() -> None:
    assert np.isnan(clustered_se(np.array([0.01, 0.02]), np.array(["a", "a"])))


def _records(n_seeds: int = 6) -> list[dict[str, Any]]:
    """Campaigns whose forward headline reads truth + 0.2 pp, with the full reversal exact."""
    rng = np.random.default_rng(0)
    records = []
    for seed in range(n_seeds):
        for m in (1, 0, -1):
            truth = 0.04 * m
            forward = truth + 0.002 + rng.normal(0, 0.001)
            reverse = (1 + forward) / (1 + truth) ** 2 - 1
            records.append(
                {
                    "cell_id": f"kel_s{seed:02d}_m{m:+d}",
                    "status": "ok",
                    "arm": "main",
                    "site": "kel",
                    "seed_index": seed,
                    "k": 4,
                    "pre_months": 12,
                    "post_months": 6,
                    "multiplier": m,
                    "turbines": [
                        {
                            "turbine": "T01",
                            "estimate": forward,
                            "truth": truth,
                            "sigma": 0.003,
                            "sigma_block_6h": 0.0001,
                            "uplift_forward": forward,
                            "uplift_reverse": reverse,
                            "sigma_reverse": 0.003,
                        }
                    ],
                    "farm": {
                        "estimate": forward,
                        "truth": truth,
                        "reference_uplift_before_screen": rng.normal(0, 0.001),
                        "reference_uplift_after_screen": 0.01,
                    },
                    "references": [
                        {
                            "test_wtg": "T01",
                            "turbine": "T02",
                            "uplift": 0.001,
                            "screened": False,
                            "unjudged": False,
                            "sigma": 0.002,
                        }
                    ],
                    "bins": [
                        {
                            "test_wtg": "T01",
                            "condition": "power",
                            "condition_bin": "(0, 1]",
                            "estimate": truth,
                            "truth": truth,
                            "sigma": 0.01,
                        }
                    ],
                    "plans": {"T01": {"works": ["2020-07-01", "2020-07-02"], "post_days": 180.0}},
                    "record_losses": {
                        "T01": [
                            {"reason": "usable", "hours_after": 900.0},
                            {"reason": "rare waking configuration", "hours_after": 100.0 * (seed + 1)},
                        ]
                    },
                    "data_screen": {
                        "T01": {
                            "candidates": [
                                {
                                    "candidate": "T02",
                                    "correlation": 0.90 + 0.01 * seed,
                                    "median_correlation": 0.96,
                                    "hours_before": 500.0,
                                    "median_hours_before": 1000.0,
                                    "hours_after": 300.0,
                                    "median_hours_after": 300.0,
                                }
                            ],
                            "hours": [],
                        }
                    },
                }
            )
    return records


def test_each_test_turbine_carries_its_season_period_and_rule_share() -> None:
    frame = turbine_frame(_records(n_seeds=1))
    assert set(frame["season"]) == {"JJA"}
    assert frame["rare_share_after"].iloc[0] == pytest.approx(100.0 / 1000.0)
    assert frame["reversal"].to_numpy() == pytest.approx(frame["truth"].to_numpy())


def test_the_tables_judge_each_target() -> None:
    tables = validation_tables(_records())
    refs = tables["i_references"].set_index(["site", "quantity"])
    assert refs.loc[("all", "reference_uplift_after_screen"), "mean"] == pytest.approx(0.01)
    assert refs.loc[("all", "placebo_forward"), "mean"] == pytest.approx(0.002, abs=0.001)
    assert refs.loc[("all", "placebo_reversal"), "mean"] == pytest.approx(0.0, abs=1e-9)
    line = tables["ii_linearity"].set_index(["estimator", "site"])
    assert line.loc[("reversal", "all"), "slope"] == pytest.approx(1.0)
    assert line.loc[("reversal", "all"), "intercept"] == pytest.approx(0.0, abs=1e-9)
    coverage = tables["iv_coverage"].set_index(["quantity", "block"])
    assert coverage.loc[("test", "sigma_block_6h"), "coverage"] == 0.0
    assert coverage.loc[("bins_power", "sigma"), "coverage"] == 1.0
    assert coverage.loc[("reference", "sigma"), "coverage"] == 1.0
    assert set(tables["iii_dependence"]["axis"]) == {"pre_months", "post_months", "k", "season"}
    assert len(tables["open_configuration_rule"]) == 3
    assert set(tables["open_screen"]["measure"]) == {"correlation_margin", "hours_fraction"}


def test_the_headline_is_judged_and_forward_is_the_recorded_forward_reading() -> None:
    records = _records()
    for record in records:
        turbine = record["turbines"][0]
        turbine["estimate"] = turbine["truth"]  # a corrected headline that removes the 0.2 pp
    tables = validation_tables(records)
    refs = tables["i_references"].set_index(["site", "quantity"])
    assert refs.loc[("all", "placebo_headline"), "mean"] == pytest.approx(0.0, abs=1e-12)
    assert refs.loc[("all", "placebo_forward"), "mean"] == pytest.approx(0.002, abs=0.001)
    by_period = tables["iv_coverage_by_period"].set_index("quantity")
    assert by_period.loc["test", "rmse"] == pytest.approx(0.0, abs=1e-12)
    coverage = tables["iv_coverage"].set_index(["quantity", "block"])
    assert coverage.loc[("test", "sigma"), "coverage"] == 1.0


def test_without_a_recorded_forward_reading_forward_is_the_headline() -> None:
    records = _records(n_seeds=1)
    for record in records:
        for key in ("uplift_forward", "uplift_reverse", "sigma_reverse"):
            del record["turbines"][0][key]
    frame = turbine_frame(records)
    assert frame["forward"].to_numpy() == pytest.approx(frame["headline"].to_numpy())


def test_the_sigma_components_are_judged_alone_and_in_pairs() -> None:
    records = _records()
    for record in records:
        record["turbines"][0].update({"sigma_sampling": 0.0001, "sigma_reference": 0.0001, "sigma_holdout": 0.01})
    tables = validation_tables(records)
    coverage = tables["iv_coverage"].set_index(["quantity", "block"])
    assert coverage.loc[("test", "sampling"), "coverage"] == 0.0
    assert coverage.loc[("test", "sampling+reference"), "coverage"] == 0.0
    assert coverage.loc[("test", "sampling+holdout"), "coverage"] == 1.0
    by_period = tables["iv_coverage_by_period"].set_index("quantity")
    assert by_period.loc["test", "coverage_sampling+holdout"] == 1.0


def test_component_sigmas_leave_out_what_is_not_recorded() -> None:
    frame = pd.DataFrame({"sigma_sampling": [0.03], "sigma_holdout": [0.04]})
    out = component_sigmas(frame)
    assert set(out) == {"sampling", "sampling+holdout"}
    assert out["sampling+holdout"][0] == pytest.approx(0.05)


def test_the_domain_needs_twelve_months_before_and_six_after() -> None:
    record = _records(n_seeds=1)[0]
    assert in_domain(record)
    assert not in_domain({**record, "pre_months": 6})
    assert not in_domain({**record, "post_months": 3})


def test_a_cell_taking_all_the_data_is_measured_by_its_shortest_plan() -> None:
    record = {**_records(n_seeds=1)[0], "pre_months": None, "post_months": None}
    record["plans"] = {"T01": {"pre_days": 400.0, "post_days": 170.0}, "T02": {"pre_days": 380.0, "post_days": 200.0}}
    assert period_months(record, side="pre") == pytest.approx(380.0 / (365.25 / 12))
    assert period_months(record, side="post") == pytest.approx(170.0 / (365.25 / 12))
    assert not in_domain(record)


def test_the_targets_leave_out_the_cells_outside_the_domain_and_the_periods_keep_them() -> None:
    inside = _records()
    outside = [
        {**r, "cell_id": f"{r['cell_id']}_short", "post_months": 1, "turbines": [{**r["turbines"][0], "estimate": 0.5}]}
        for r in _records()
    ]
    tables = validation_tables(inside + outside)
    refs = tables["i_references"].set_index(["site", "quantity"])
    assert refs.loc[("all", "placebo_headline"), "mean"] == pytest.approx(0.002, abs=0.001)
    by_period = tables["iv_coverage_by_period"].set_index(["quantity", "post_months"])
    assert by_period.loc[("test", 1), "bias"] > 0.4
    assert by_period.loc[("test", 6), "bias"] == pytest.approx(0.002, abs=0.001)


def test_the_text_gives_a_verdict_per_check() -> None:
    text = describe(validation_tables(_records()), n_cells=18, n_domain=18)
    assert "reference_uplift_after_screen: +1.000" in text
    assert "FAIL" in text  # the after-screen reference reads +1 % exactly, with no spread
    assert "test sigma_block_6h: 0.00" in text
    assert f"{TARGET_COVERAGE:.0%}"[:2] in text


def test_the_report_is_written_beside_the_study(tmp_path: Path) -> None:
    (tmp_path / "cells").mkdir()
    for record in _records():
        (tmp_path / "cells" / f"{record['cell_id']}.json").write_text(json.dumps(record))
    (tmp_path / "cells" / "failed.json").write_text(json.dumps({"status": "failed"}))
    text = write_report(tmp_path)
    assert text.startswith("# Validation of 18 ok cells")
    assert (tmp_path / "validation" / "validation.md").read_text() == text
    assert (tmp_path / "validation" / "ii_linearity.csv").exists()


def test_the_usable_share_table_gives_median_shares_per_site_and_period() -> None:
    records = _records(n_seeds=2)
    for record in records:
        record["record_losses"] = {
            "T01": [
                {"reason": "usable", "hours_before": 300.0, "hours_after": 600.0},
                {"reason": "changed upwind of reference", "hours_before": 500.0, "hours_after": 300.0},
                {"reason": "changed upwind of subject", "hours_before": 200.0, "hours_after": 100.0},
            ]
        }
    table = usable_share_table(records).set_index(["arm", "site", "pre_months", "post_months"])
    row = table.loc[("main", "kel", 12, 6)]
    assert row["usable_before"] == pytest.approx(0.3)
    assert row["usable_after"] == pytest.approx(0.6)
    assert row["changed_upwind_of_reference_before"] == pytest.approx(0.5)
    assert row["changed_upwind_of_subject_before"] == pytest.approx(0.2)
    assert row["turbines"] == len(records)
    assert table.loc[("main", "kel", "all", "all"), "usable_before"] == pytest.approx(0.3)
    text = describe(validation_tables(records), n_cells=len(records), n_domain=len(records))
    assert "usable share of before hours: main kel 0.30" in text
