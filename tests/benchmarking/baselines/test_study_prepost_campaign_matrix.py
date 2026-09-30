"""Tests for the prepost campaign matrix: cells, the resumable study, merging and comparing."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines import study_prepost_campaign_matrix as driver
from benchmarking.baselines.prepost_matrix.cells import (
    SIZES,
    Cell,
    MatrixSettings,
    campaign_seed,
    cell_cost_s,
    draw_exclusions,
    matrix_cells,
)
from benchmarking.baselines.prepost_matrix.execute import plan_settings
from benchmarking.baselines.prepost_matrix.metrics import (
    baseline_tables,
    compare_tables,
    leak_check,
    line_fit,
    reference_metrics,
)
from benchmarking.baselines.prepost_matrix.plan import makespan_s, measured_costs, plan_study
from benchmarking.baselines.prepost_matrix.study import (
    CANDIDATE,
    accept_candidate,
    baseline_path_for,
    cell_dir,
    detail_dir,
    list_studies,
    merge_study,
    open_study,
    read_cell,
    record_path,
    run_cell,
    run_study,
    study_dir_for,
    study_settings,
    study_size,
)
from wind_up.analysis_period import PlanSettings

if TYPE_CHECKING:
    from pathlib import Path

SMALL = MatrixSettings(seeds={"hot": 2}, multipliers=(1, 0, -1), ks=(4,), post_months=(1, 6), real=False)
SIZE = "test"


# --- cells -------------------------------------------------------------------------------------


def test_cell_ids_are_readable() -> None:
    assert Cell("main", "pen", 7, 1, 4, 6).cell_id == "pen_s07_m+1_K4_L6"
    assert Cell("main", "pen", 7, -1, 4, 6).cell_id == "pen_s07_m-1_K4_L6"
    assert Cell("excl_nan", "kel", 0, 0, 4, 12).cell_id == "kel_s00_m+0_K4_L12_xnan"
    assert Cell("real", "hot_t13", None, None, 4, 6).cell_id == "hot_t13_K4_L6"


def test_cell_ids_are_unique() -> None:
    cells = matrix_cells(MatrixSettings())
    assert len({c.cell_id for c in cells}) == len(cells)


def test_the_default_matrix_size() -> None:
    settings = MatrixSettings()
    n_campaigns = sum(settings.seeds.values())
    per_campaign = len(settings.multipliers) * len(settings.post_months)
    expected = n_campaigns * per_campaign * (len(settings.ks) + 2) + len(settings.ks) * len(settings.post_months)
    assert len(matrix_cells(settings)) == expected


def test_a_cell_does_not_depend_on_the_other_cells() -> None:
    small = matrix_cells(SMALL)
    large = matrix_cells(MatrixSettings(seeds={"hot": 4, "pen": 1}, ks=(3, 4), post_months=(1, 3, 6)))
    assert {c.cell_id for c in small} <= {c.cell_id for c in large}
    assert campaign_seed(0, site="hot", seed_index=1) == campaign_seed(0, site="hot", seed_index=1)
    assert campaign_seed(0, site="hot", seed_index=1) != campaign_seed(0, site="pen", seed_index=1)
    assert campaign_seed(0, site="hot", seed_index=1) != campaign_seed(1, site="hot", seed_index=1)


def test_cells_run_most_expensive_first() -> None:
    cells = matrix_cells(MatrixSettings(seeds={"hot": 1, "kel": 1}, post_months=(1, 12)))
    costs = [cell_cost_s(c) for c in cells]
    assert costs == sorted(costs, reverse=True)
    assert (cells[0].site, cells[0].k) == ("hot", 6)
    assert cells[-1].site == "kel"


def test_the_small_and_big_sizes() -> None:
    small, big = matrix_cells(SIZES["small"].settings), matrix_cells(SIZES["big"].settings)
    assert (len(small), len(big)) == (38, 1368)
    assert "hot" not in {c.site for c in small}
    assert {c.arm for c in small} == {"main", "excl_booleans", "excl_nan", "real"}
    assert (SIZES["small"].workers, SIZES["big"].workers) == (3, 16)
    assert {c.cell_id for c in small} <= {c.cell_id for c in matrix_cells(MatrixSettings())}


# --- planning ----------------------------------------------------------------------------------


def test_the_makespan_of_a_greedy_longest_first_schedule() -> None:
    # Two workers: 5 | 4, then each 3 goes to the least-loaded: 4+3, 5+3, 7+3.
    assert makespan_s([5, 4, 3, 3, 3], workers=2) == 10
    assert makespan_s([5, 4, 3, 3, 3], workers=1) == 18
    assert makespan_s([], workers=4) == 0


KEL_PAIR = MatrixSettings(seeds={"kel": 2}, multipliers=(1,), ks=(4,), post_months=(1,), exclusion_arms=(), real=False)


def test_plan_projects_cpu_hours_wall_time_and_memory() -> None:
    plan = plan_study(KEL_PAIR, workers=2)
    assert plan.n_cells == {"kel": 2}
    assert plan.cpu_hours == pytest.approx(2 * cell_cost_s(Cell("main", "kel", 0, 1, 4, 1)) / 3600)
    assert plan.wall_hours == pytest.approx(plan.cpu_hours / 2)
    assert plan.peak_memory_gb > 0


def test_the_plan_command_prints_a_sizes_projection(capsys: pytest.CaptureFixture[str]) -> None:
    driver.main(["plan", "--size", "small"])
    out = capsys.readouterr().out
    assert "38 cells" in out
    assert "on 3 worker(s)" in out
    driver.main(["plan", "--size", "big", "--workers", "8"])
    assert "1368 cells" in capsys.readouterr().out


def test_run_and_download_need_a_size() -> None:
    with pytest.raises(SystemExit):
        driver.parse_args(["run"])
    with pytest.raises(SystemExit):
        driver.parse_args(["download"])
    args = driver.parse_args(["run", "--size", "big", "--retry-failed"])
    assert (args.size, args.retry_failed, args.workers) == ("big", True, None)


def test_plan_can_use_a_studys_measured_costs(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "site": ["kel", "kel", "kel"],
            "k": [4, 4, 4],
            "status": ["ok", "ok", "failed"],
            "wall_time_s": [100.0, 200.0, 9999.0],
            "peak_rss_mb": [1000.0, 2048.0, 9999.0],
        }
    ).to_csv(tmp_path / "cells.csv", index=False)
    costs, rss = measured_costs(tmp_path)
    assert costs[("kel", 4)] == pytest.approx(150.0)
    assert rss["kel"] == pytest.approx(2048.0)
    plan = plan_study(KEL_PAIR, workers=1, measured=tmp_path)
    assert plan.cpu_hours == pytest.approx(300 / 3600)
    assert plan.peak_memory_gb == pytest.approx(2.0)


def test_half_the_campaigns_roll_out_to_the_whole_farm() -> None:
    assert [Cell("main", "hot", i, 1, 4, 6).full_rollout for i in range(4)] == [True, False, True, False]


def test_settings_round_trip_through_json() -> None:
    settings = MatrixSettings(seeds={"hot": 3}, ks=(4,), exclusion_arms=("excl_nan",))
    assert MatrixSettings.from_json(json.loads(json.dumps(settings.to_json()))) == settings


def test_exclusions_are_0_to_2_whole_day_periods_of_3_to_14_days_in_the_span() -> None:
    start, end = pd.Timestamp("2016-01-01", tz="UTC"), pd.Timestamp("2021-01-01", tz="UTC")
    turbines = [f"T{i:02d}" for i in range(40)]
    exclusions = draw_exclusions(turbines, start=start, end=end, seed=5)
    counts = pd.Series([t for t, _, _ in exclusions]).value_counts()
    assert counts.max() <= 2
    assert len(counts) < len(turbines)  # some turbines draw none
    for _, first, last in exclusions:
        assert start <= first < last <= end
        assert pd.Timedelta(days=3) <= last - first <= pd.Timedelta(days=14)
        assert first == first.normalize()
    assert draw_exclusions(turbines, start=start, end=end, seed=5) == exclusions


# --- running cells -----------------------------------------------------------------------------


def fake_execute(cell: Cell, *, detail_dir: Path, cell_dir: Path, settings: MatrixSettings) -> dict[str, Any]:  # noqa: ARG001
    """Record the call; fail when the detail directory holds a ``fail_<cell id>`` file."""
    with (detail_dir / "calls.txt").open("a") as calls:
        calls.write(cell.cell_id + "\n")
    (cell_dir / "work.txt").write_text("scratch")
    if (detail_dir / f"fail_{cell.cell_id}").exists():
        msg = "this campaign is broken"
        raise RuntimeError(msg)
    truth = 0.04 * (cell.multiplier or 0)
    return {
        "turbines": [{"turbine": "T01", "estimate": truth + 0.001, "truth": truth}],
        "farm": {"estimate": truth + 0.001, "truth": truth},
        "references": [{"test_wtg": "T01", "turbine": "T02", "uplift": 0.002, "screened": False}],
    }


def no_prefetch(detail_dir: Path, cells: list[Cell]) -> None:
    """Fetch nothing."""


def calls(study_dir: Path) -> list[str]:
    path = detail_dir(study_dir) / "calls.txt"
    return path.read_text().split() if path.exists() else []


def test_a_raising_cell_is_recorded_as_failed_with_its_traceback(tmp_path: Path) -> None:
    study, cell = tmp_path / "study", matrix_cells(SMALL)[0]
    detail_dir(study).mkdir()
    (detail_dir(study) / f"fail_{cell.cell_id}").touch()
    record = run_cell(study, cell, settings=SMALL, execute=fake_execute)
    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert "this campaign is broken" in record["traceback"]
    assert read_cell(study, cell) == json.loads(json.dumps(record))
    assert "this campaign is broken" in (cell_dir(study, cell) / "cell.log").read_text()


def test_an_ok_cell_records_its_wall_time_and_peak_memory(tmp_path: Path) -> None:
    cell = matrix_cells(SMALL)[0]
    record = run_cell(tmp_path / "study", cell, settings=SMALL, execute=fake_execute)
    assert record["status"] == "ok"
    assert record["cell_id"] == cell.cell_id
    assert record["wall_time_s"] >= 0
    assert record["peak_rss_mb"] > 0


def test_a_cell_writes_its_record_to_the_study_and_its_work_to_the_detail_directory(tmp_path: Path) -> None:
    study = tmp_path / "prepost_matrix__abc1234__small"
    cell = matrix_cells(SMALL)[0]
    run_cell(study, cell, settings=SMALL, execute=fake_execute)
    assert record_path(study, cell) == study / "cells" / f"{cell.cell_id}.json"
    assert record_path(study, cell).exists()
    assert cell_dir(study, cell) == tmp_path / "prepost_matrix__abc1234__small__detail" / "cells" / cell.cell_id
    assert (cell_dir(study, cell) / "cell.log").exists()
    assert (cell_dir(study, cell) / "work.txt").exists()
    assert [p.name for p in (study / "cells").iterdir()] == [f"{cell.cell_id}.json"]


def test_a_study_is_named_by_commit_and_size_beside_its_detail_directory(tmp_path: Path) -> None:
    study = study_dir_for(tmp_path, commit="abc1234", dirty=True, size="small")
    assert study.name == "prepost_matrix__abc1234-dirty__small"
    assert detail_dir(study).name == "prepost_matrix__abc1234-dirty__small__detail"
    assert detail_dir(study).parent == tmp_path


def _run(settings: MatrixSettings, tmp_path: Path, **kwargs: Any) -> Path:  # noqa: ANN401
    return run_study(
        settings,
        size=SIZE,
        root=tmp_path,
        workers=kwargs.pop("workers", 1),
        execute=fake_execute,
        prefetch_sources=no_prefetch,
        baseline_path=tmp_path / "none.json",
        **kwargs,
    )


def test_run_skips_ok_and_failed_cells_and_reruns_interrupted_ones(tmp_path: Path) -> None:
    cells = matrix_cells(SMALL)
    failing, interrupted = cells[0], cells[1]
    probe = _run(SMALL, tmp_path, limit=0)
    (detail_dir(probe) / f"fail_{failing.cell_id}").touch()

    study = _run(SMALL, tmp_path)
    assert sorted(calls(study)) == sorted(c.cell_id for c in cells)
    assert read_cell(study, failing)["status"] == "failed"  # type: ignore[index]

    (detail_dir(study) / f"fail_{failing.cell_id}").unlink()
    record_path(study, interrupted).unlink()
    (detail_dir(study) / "calls.txt").unlink()
    _run(SMALL, tmp_path)
    assert calls(study) == [interrupted.cell_id]
    assert read_cell(study, failing)["status"] == "failed"  # type: ignore[index]


def test_run_retries_failed_cells_when_asked(tmp_path: Path) -> None:
    failing = matrix_cells(SMALL)[0]
    probe = _run(SMALL, tmp_path, limit=0)
    (detail_dir(probe) / f"fail_{failing.cell_id}").touch()
    study = _run(SMALL, tmp_path)

    (detail_dir(study) / f"fail_{failing.cell_id}").unlink()
    (detail_dir(study) / "calls.txt").unlink()
    _run(SMALL, tmp_path, retry_failed=True)
    assert calls(study) == [failing.cell_id]
    assert json.loads((study / CANDIDATE).read_text())["complete"]


def test_run_on_worker_processes(tmp_path: Path) -> None:
    settings = MatrixSettings(seeds={"hot": 1}, multipliers=(1,), ks=(4,), post_months=(1,), real=False)
    study = _run(settings, tmp_path, workers=2)
    statuses = pd.read_csv(study / "cells.csv")["status"]
    assert list(statuses) == ["ok"] * 3


def test_a_study_directory_refuses_another_matrix(tmp_path: Path) -> None:
    open_study(tmp_path, SMALL, size=SIZE, commit="abc1234", dirty=False)
    open_study(tmp_path, SMALL, size=SIZE, commit="abc1234", dirty=False)
    assert study_settings(tmp_path) == SMALL
    assert study_size(tmp_path) == SIZE
    with pytest.raises(ValueError, match="another matrix"):
        open_study(tmp_path, MatrixSettings(), size=SIZE, commit="abc1234", dirty=False)


def test_each_size_has_its_own_committed_baseline() -> None:
    assert baseline_path_for("small").name == "study_prepost_campaign_matrix_baseline_small.json"
    assert baseline_path_for("big").name == "study_prepost_campaign_matrix_baseline_big.json"
    assert baseline_path_for("small").parent == baseline_path_for("big").parent


def test_list_skips_detail_directories_and_reports_size_and_disk_use(tmp_path: Path) -> None:
    _run(SMALL, tmp_path)
    table = list_studies(tmp_path)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["size"] == SIZE
    assert row["n_ok"] == len(matrix_cells(SMALL))
    assert row["download_mb"] > 0
    assert row["detail_mb"] > 0


# --- accepting ---------------------------------------------------------------------------------


def _write_candidate(study: Path, **fields: object) -> None:
    doc = {"n_failed": 0, "complete": True, "n_ok": 3, "n_cells": 3, "git_commit": "abc1234", **fields}
    study.mkdir(parents=True, exist_ok=True)
    (study / CANDIDATE).write_text(json.dumps(doc))


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"complete": False, "n_ok": 2}, "only 2 of 3"),
        ({"n_failed": 1, "complete": False}, "failed"),
        ({"git_commit": "abc1234-dirty"}, "dirty"),
    ],
)
def test_accepting_refuses_an_incomplete_failed_or_dirty_study(tmp_path: Path, fields: dict, reason: str) -> None:
    _write_candidate(tmp_path / "study", **fields)
    baseline = tmp_path / "baseline.json"
    with pytest.raises(ValueError, match=reason):
        accept_candidate(tmp_path / "study", baseline_path=baseline)
    assert not baseline.exists()


def test_accepting_a_clean_complete_study_copies_the_candidate(tmp_path: Path) -> None:
    _write_candidate(tmp_path / "study")
    accept_candidate(tmp_path / "study", baseline_path=tmp_path / "baseline.json")
    assert (tmp_path / "baseline.json").read_text() == (tmp_path / "study" / CANDIDATE).read_text()


# --- merging -----------------------------------------------------------------------------------


def _record(site: str, seed: int, multiplier: int, **fields: object) -> dict[str, Any]:
    return {
        "arm": "main",
        "site": site,
        "seed_index": seed,
        "k": 4,
        "post_months": 6,
        "multiplier": multiplier,
        "status": "ok",
        **fields,
    }


def _line_records() -> list[dict[str, Any]]:
    """Two campaigns whose estimates lie exactly on ``0.9 * truth + 0.002``."""
    records = []
    for site, seed, size in (("hot", 0, 0.04), ("pen", 1, 0.02)):
        for m in (1, 0, -1):
            truth = size * m
            estimate = 0.9 * truth + 0.002
            records.append(
                _record(
                    site,
                    seed,
                    m,
                    turbines=[{"turbine": "T01", "estimate": estimate, "truth": truth}],
                    farm={"estimate": estimate, "truth": truth},
                    references=[
                        {"test_wtg": "T01", "turbine": "T02", "uplift": 0.001 * (seed + 1), "screened": False},
                        {"test_wtg": "T01", "turbine": "T03", "uplift": -0.004, "screened": True},
                    ],
                )
            )
    return records


def test_line_fit_recovers_an_exact_line() -> None:
    scale, bias = line_fit([0.04, 0.0, -0.04], [0.038, 0.002, -0.034])
    assert scale == pytest.approx(0.9)
    assert bias == pytest.approx(0.002)
    assert np.isnan(line_fit([0.0, 0.0], [0.1, 0.2])[0])
    assert np.isnan(line_fit([0.04, np.nan], [0.1, 0.2])[0])


def test_merge_computes_linearity_on_a_constructed_exact_line() -> None:
    linearity = baseline_tables(_line_records())["linearity"].set_index("site")
    for site in ("hot", "pen", "all"):
        assert linearity.loc[site, "line_scale_mean"] == pytest.approx(0.9)
        assert linearity.loc[site, "line_bias_mean"] == pytest.approx(0.002)
        assert linearity.loc[site, "line_scale_sd"] == pytest.approx(0.0, abs=1e-12)
    assert linearity.loc["all", "line_n"] == 2


def test_merge_computes_test_farm_and_reference_metrics() -> None:
    groups = baseline_tables(_line_records())["groups"]
    row = groups[(groups["site"] == "all") & (groups["multiplier"] == 1)].iloc[0]
    errors = np.array([0.9 * 0.04 + 0.002 - 0.04, 0.9 * 0.02 + 0.002 - 0.02])
    assert row["test_bias"] == pytest.approx(errors.mean())
    assert row["test_spread"] == pytest.approx(errors.std())
    assert row["test_score"] == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert row["test_n"] == 2
    assert row["farm_bias"] == pytest.approx(errors.mean())
    readings = [0.001, -0.004, 0.002, -0.004]
    assert row["ref_mean"] == pytest.approx(np.mean(readings))
    assert row["ref_median"] == pytest.approx(np.median(readings))
    assert row["ref_sd"] == pytest.approx(np.std(readings))
    assert row["ref_worst_abs"] == pytest.approx(0.004)
    assert row["ref_n"] == 4
    assert set(groups["site"]) == {"hot", "pen", "all"}


def test_reference_metrics_ignore_non_finite_readings() -> None:
    assert reference_metrics([0.01, np.nan, -0.03])["ref_n"] == 2
    assert reference_metrics([])["ref_n"] == 0


def test_real_rows_are_kept_separate() -> None:
    real = {
        "arm": "real",
        "site": "hot_t13",
        "seed_index": None,
        "k": 4,
        "post_months": 6,
        "multiplier": None,
        "status": "ok",
        "turbines": [{"turbine": "T13", "estimate": 0.027, "truth": None}],
        "farm": {"estimate": 0.027, "truth": None},
        "references": [{"test_wtg": "T13", "turbine": "T14", "uplift": 0.005, "screened": False}],
    }
    tables = baseline_tables([*_line_records(), real])
    assert "hot_t13" not in set(tables["groups"]["site"])
    row = tables["real"].iloc[0]
    assert row["t13_estimate"] == pytest.approx(0.027)
    assert row["ref_mean"] == pytest.approx(0.005)


def test_failed_cells_are_left_out_of_the_metrics() -> None:
    records = [*_line_records(), _record("hot", 5, 1, status="failed")]
    assert baseline_tables(records)["groups"].equals(baseline_tables(_line_records())["groups"])


def test_the_leak_check_measures_how_far_a_reference_moves_across_multipliers() -> None:
    records = _line_records()
    records[0]["references"][0]["uplift"] = 0.011  # hot seed 0, multiplier +1
    leaks = leak_check(records)
    top = leaks.iloc[0]
    assert (top["site"], top["turbine"]) == ("hot", "T02")
    assert top["leak"] == pytest.approx(0.010)
    assert leaks["leak"].iloc[1:].abs().max() == pytest.approx(0.0)


def test_merge_needs_only_the_downloaded_study_directory(tmp_path: Path) -> None:
    study = _run(SMALL, tmp_path)
    shutil.rmtree(detail_dir(study))
    (study / CANDIDATE).unlink()
    doc = merge_study(study)
    assert doc["complete"]
    assert doc["n_ok"] == len(matrix_cells(SMALL))


def test_merge_writes_the_candidate_and_marks_a_partial_study_incomplete(tmp_path: Path) -> None:
    study = tmp_path / "study"
    open_study(study, SMALL, size=SIZE, commit="abc1234", dirty=False)
    cells = matrix_cells(SMALL)
    for cell in cells[:3]:
        run_cell(study, cell, settings=SMALL, execute=fake_execute)
    doc = merge_study(study)
    assert (doc["n_ok"], doc["n_cells"], doc["complete"]) == (3, len(cells), False)
    assert doc["git_commit"] == "abc1234"
    assert "host" not in doc
    assert json.loads((study / CANDIDATE).read_text())["groups"]
    table = pd.read_csv(study / "cells.csv")
    assert (table["status"] == "pending").sum() == len(cells) - 3


# --- comparing ---------------------------------------------------------------------------------


def test_compare_flags_a_moved_cell() -> None:
    before = baseline_tables(_line_records())
    moved = _line_records()
    moved[0]["turbines"][0]["estimate"] += 0.01  # hot seed 0, multiplier +1
    after = baseline_tables(moved)
    comparison = compare_tables(after, before)
    flagged = comparison[comparison["moved"]]
    assert not flagged.empty
    groups = flagged[flagged["table"] == "groups"]
    assert set(groups["site"]) == {"hot", "all"}
    assert set(groups["multiplier"]) == {1}
    assert "test_bias" in set(groups["metric"])
    assert not compare_tables(before, before)["moved"].any()


def test_compare_flags_a_value_present_on_one_side_only() -> None:
    before = baseline_tables(_line_records())
    after = baseline_tables([r for r in _line_records() if r["site"] == "hot"])
    flagged = compare_tables(after, before)
    assert flagged[flagged["moved"]]["site"].isin(["pen", "all"]).all()
    assert (flagged[flagged["moved"]]["site"] == "pen").any()


# --- end to end --------------------------------------------------------------------------------


@pytest.mark.slow
def test_one_small_campaign_runs_end_to_end(tmp_path: Path) -> None:
    settings = MatrixSettings(
        seeds={"hot": 1}, multipliers=(1,), ks=(3,), post_months=(1,), real=False, exclusion_arms=()
    )
    study = run_study(settings, size=SIZE, root=tmp_path, workers=1, baseline_path=tmp_path / "none.json")
    (cell,) = matrix_cells(settings)
    record = read_cell(study, cell)
    assert record is not None
    assert record["status"] == "ok", record.get("traceback")
    assert record["turbines"]
    for turbine in record["turbines"]:
        assert np.isfinite(turbine["estimate"])
        assert turbine["truth"] > 0
    assert set(record["plans"]) == {t["turbine"] for t in record["turbines"]}
    assert record["references"]
    assert json.loads((study / CANDIDATE).read_text())["complete"]


def test_the_period_selector_takes_the_cells_k_and_the_matrix_shortest_side() -> None:
    chosen = plan_settings(Cell("main", "hot", 0, 1, 6, 1), settings=MatrixSettings(min_side_days=10))
    assert chosen.k == 6
    assert chosen.min_side == pd.Timedelta(days=10)
    assert chosen.pre_cap == PlanSettings().pre_cap
