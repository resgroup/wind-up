"""Tests for the prepost campaign matrix: cells, the resumable study, merging and comparing."""

from __future__ import annotations

import json
import logging
import shutil
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines import study_prepost_campaign_matrix as driver
from benchmarking.baselines.prepost_matrix.cells import (
    REAL_SITES,
    SIZES,
    Cell,
    MatrixSettings,
    campaign_seed,
    cell_cost_s,
    draw_exclusions,
    matrix_cells,
    real_pre_months,
)
from benchmarking.baselines.prepost_matrix.execute import (
    _matrix_method,
    describe_cell,
    diagnostics,
    measurement_details,
    plan_settings,
    site_operating_state,
    with_reanalysis_wind_speed,
)
from benchmarking.baselines.prepost_matrix.metrics import (
    baseline_tables,
    compare_tables,
    leak_check,
    line_fit,
    reference_metrics,
    reversal_uplift,
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
    plot_cell,
    read_cell,
    record_path,
    run_cell,
    run_study,
    study_dir_for,
    study_settings,
    study_size,
)
from benchmarking.campaigns.composed import DATA_PREPARATION_DIRNAME
from benchmarking.campaigns.loader import operating_state_config
from benchmarking.campaigns.real import REAL_CAMPAIGNS
from benchmarking.diagnostics import stages
from benchmarking.harness.northing import NORTH_TABLE_YAML
from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.sources.greenbyte import GREENBYTE_OPERATING_STATE, GREENBYTE_STATE_COL
from wind_up.analysis_period import PlanSettings

if TYPE_CHECKING:
    from pathlib import Path

SMALL = MatrixSettings(seeds={"hot": 2}, multipliers=(1, 0, -1), ks=(4,), post_months=(1, 6), real=())
SIZE = "test"


# --- cells -------------------------------------------------------------------------------------


def test_cell_ids_are_readable() -> None:
    assert Cell("main", "pen", 7, 1, 4, 12, 6).cell_id == "pen_s07_m+1_K4_P12_L6"
    assert Cell("main", "pen", 7, -1, 4, 6, 6).cell_id == "pen_s07_m-1_K4_P6_L6"
    assert Cell("excl", "kel", 0, 0, 4, 24, 12).cell_id == "kel_s00_m+0_K4_P24_L12_excl"
    assert Cell("real", "hot_t13", None, None, 4, 24, 6).cell_id == "hot_t13_K4_P24_L6"
    assert Cell("real", "hot_tuneup", None, None, 4, None, None).cell_id == "hot_tuneup_K4_Pmax_Lmax"


def test_cell_ids_are_unique() -> None:
    cells = matrix_cells(MatrixSettings())
    assert len({c.cell_id for c in cells}) == len(cells)


def test_the_default_matrix_size() -> None:
    settings = MatrixSettings()
    n_campaigns = sum(settings.seeds.values())
    per_campaign = len(settings.multipliers) * len(settings.pre_months) * len(settings.post_months)
    t13 = len([6, 12, 24]) * len(settings.post_months)
    tuneup = len([6, 12, None]) * len([1, 2, 3, None])
    real = len(settings.ks) * (t13 + tuneup)
    arms_at_one_k = len(settings.exclusion_arms) + len(settings.small_arms)
    assert len(matrix_cells(settings)) == n_campaigns * per_campaign * (len(settings.ks) + arms_at_one_k) + real


def test_tuneup_runs_the_post_lengths_its_data_holds_and_lmax() -> None:
    cells = matrix_cells(MatrixSettings(seeds={}, ks=(4,), real=("hot_tuneup",)))
    assert sorted(c.cell_id for c in cells) == [
        "hot_tuneup_K4_P12_L1",
        "hot_tuneup_K4_P12_L2",
        "hot_tuneup_K4_P12_L3",
        "hot_tuneup_K4_P12_Lmax",
        "hot_tuneup_K4_P6_L1",
        "hot_tuneup_K4_P6_L2",
        "hot_tuneup_K4_P6_L3",
        "hot_tuneup_K4_P6_Lmax",
        "hot_tuneup_K4_Pmax_L1",
        "hot_tuneup_K4_Pmax_L2",
        "hot_tuneup_K4_Pmax_L3",
        "hot_tuneup_K4_Pmax_Lmax",
    ]


def test_real_campaigns_run_the_pre_lengths_their_data_holds_then_pmax() -> None:
    settings = MatrixSettings()
    assert real_pre_months("hot_t13", settings) == [6, 12, 24]
    assert real_pre_months("hot_tuneup", settings) == [6, 12, None]
    assert real_pre_months("hot_t13", MatrixSettings(pre_months=(12,))) == [12, None]


def test_each_real_site_holds_its_longest_pre_length() -> None:
    for key, site in REAL_SITES.items():
        campaign = REAL_CAMPAIGNS[site.campaign]
        timing = campaign.declaration["timing"].get("changeover")
        if timing is None:
            continue
        changeover = pd.Timestamp(timing)
        blocked = [
            pd.Timestamp(e["end"])
            for e in campaign.declaration["exclusions"]
            if e["turbine"] == "ALL" and pd.Timestamp(e["end"]) < changeover
        ]
        usable_from = max([campaign.start, *blocked])
        assert usable_from + pd.DateOffset(months=site.longest_pre_months) <= changeover, key
        assert usable_from + pd.DateOffset(months=site.longest_pre_months + 1) > changeover, key


def test_each_real_site_holds_its_longest_post_length() -> None:
    for site in REAL_SITES.values():
        campaign = REAL_CAMPAIGNS[site.campaign]
        timing = campaign.declaration["timing"].get("changeover")
        if timing is None:
            continue
        changeover = pd.Timestamp(timing)
        assert changeover + pd.DateOffset(months=site.longest_post_months) <= campaign.end
        assert changeover + pd.DateOffset(months=site.longest_post_months + 1) > campaign.end


def test_an_older_real_flag_reads_as_the_t13_campaign() -> None:
    assert MatrixSettings.from_json({**MatrixSettings().to_json(), "real": True}).real == ("hot_t13",)
    assert MatrixSettings.from_json({**MatrixSettings().to_json(), "real": False}).real == ()


def test_a_cell_does_not_depend_on_the_other_cells() -> None:
    small = matrix_cells(SMALL)
    large = matrix_cells(MatrixSettings(seeds={"hot": 4, "pen": 1}, ks=(3, 4), post_months=(1, 3, 6)))
    assert {c.cell_id for c in small} <= {c.cell_id for c in large}
    assert campaign_seed(0, site="hot", seed_index=1) == campaign_seed(0, site="hot", seed_index=1)
    assert campaign_seed(0, site="hot", seed_index=1) != campaign_seed(0, site="pen", seed_index=1)
    assert campaign_seed(0, site="hot", seed_index=1) != campaign_seed(1, site="hot", seed_index=1)


def test_cells_run_most_expensive_first() -> None:
    cells = matrix_cells(MatrixSettings(seeds={"hot": 1, "kel": 1}, post_months=(1, 12), real=("hot_t13",)))
    costs = [cell_cost_s(c) for c in cells]
    assert costs == sorted(costs, reverse=True)
    assert (cells[0].site, cells[0].k) == ("hot", 6)
    assert cells[-1].site == "kel"


def test_the_small_and_big_sizes() -> None:
    small, big = matrix_cells(SIZES["small"].settings), matrix_cells(SIZES["big"].settings)
    assert (len(small), len(big)) == (40, 4140)
    assert "hot" not in {c.site for c in small}
    assert {c.arm for c in small} == {"main", "excl", "small", "real"}
    assert (SIZES["small"].workers, SIZES["big"].workers) == (3, 16)
    assert {c.cell_id for c in small} <= {c.cell_id for c in matrix_cells(MatrixSettings())}


# --- planning ----------------------------------------------------------------------------------


def test_the_makespan_of_a_greedy_longest_first_schedule() -> None:
    # Two workers: 5 | 4, then each 3 goes to the least-loaded: 4+3, 5+3, 7+3.
    assert makespan_s([5, 4, 3, 3, 3], workers=2) == 10
    assert makespan_s([5, 4, 3, 3, 3], workers=1) == 18
    assert makespan_s([], workers=4) == 0


KEL_PAIR = MatrixSettings(
    seeds={"kel": 2},
    multipliers=(1,),
    ks=(4,),
    pre_months=(12,),
    post_months=(1,),
    exclusion_arms=(),
    small_arms=(),
    real=(),
)


def test_plan_projects_cpu_hours_wall_time_and_memory() -> None:
    plan = plan_study(KEL_PAIR, workers=2)
    assert plan.n_cells == {"kel": 2}
    assert plan.cpu_hours == pytest.approx(2 * cell_cost_s(Cell("main", "kel", 0, 1, 4, 12, 1)) / 3600)
    assert plan.wall_hours == pytest.approx(plan.cpu_hours / 2)
    assert plan.peak_memory_gb > 0


def test_the_plan_command_prints_a_sizes_projection(capsys: pytest.CaptureFixture[str]) -> None:
    driver.main(["plan", "--size", "small"])
    out = capsys.readouterr().out
    assert "40 cells" in out
    assert "on 3 worker(s)" in out
    driver.main(["plan", "--size", "big", "--workers", "8"])
    assert "4140 cells" in capsys.readouterr().out


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


def test_the_small_change_arm_runs_at_the_exclusion_k_with_its_own_ids() -> None:
    cells = matrix_cells(
        MatrixSettings(seeds={"kel": 1}, multipliers=(1,), pre_months=(12,), post_months=(6,), real=())
    )
    small = [c for c in cells if c.arm == "small"]
    assert [c.cell_id for c in small] == ["kel_s00_m+1_K4_P12_L6_1pc"]
    assert small[0].is_small
    assert not any(c.is_small for c in cells if c.arm != "small")


def test_half_the_campaigns_roll_out_to_the_whole_farm() -> None:
    assert [Cell("main", "hot", i, 1, 4, 12, 6).full_rollout for i in range(4)] == [True, False, True, False]


def test_settings_round_trip_through_json() -> None:
    settings = MatrixSettings(seeds={"hot": 3}, ks=(4,), exclusion_arms=("excl",))
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


def test_a_failed_cell_still_records_its_campaign_draw(tmp_path: Path) -> None:
    study, cell = tmp_path / "study", matrix_cells(SMALL)[0]
    detail_dir(study).mkdir()
    (detail_dir(study) / f"fail_{cell.cell_id}").touch()
    record = run_cell(study, cell, settings=SMALL, execute=fake_execute)
    assert record["status"] == "failed"
    assert record["draw"] == describe_cell(cell, settings=SMALL)["draw"]
    assert record["draw"]["campaign_seed"] == campaign_seed(SMALL.master_seed, site="hot", seed_index=0)
    assert record["draw"]["trial"]


def test_the_real_campaign_describes_no_draw_before_it_runs() -> None:
    assert describe_cell(Cell("real", "hot_t13", None, None, 4, 12, 3), settings=SMALL) == {}


def test_the_cell_record_names_each_unplanned_turbine(tmp_path: Path) -> None:
    references = pd.DataFrame(columns=["method", "test_wtg", "turbine", "uplift", "n_records", "screened", "unjudged"])
    record = diagnostics(references, plans={}, unplanned={"T02": "T02: no span"}, cell_dir=tmp_path)
    assert record["unplanned"] == {"T02": "T02: no span"}


def test_the_cell_record_reads_the_north_table_of_part_a(tmp_path: Path) -> None:
    table_dir = tmp_path / DATA_PREPARATION_DIRNAME / stages.NORTHING
    table_dir.mkdir(parents=True)
    (table_dir / NORTH_TABLE_YAML).write_text(
        "- [T01, '2017-01-01T00:00:00+00:00', 3.0]\n- [T02, '2017-01-01T00:00:00+00:00', -1.5]\n"
    )
    references = pd.DataFrame(columns=["method", "test_wtg", "turbine", "uplift", "n_records", "screened", "unjudged"])
    record = diagnostics(references, plans={}, unplanned={}, cell_dir=tmp_path)
    assert record["northing"] == [["T01", "2017-01-01T00:00:00+00:00", 3.0], ["T02", "2017-01-01T00:00:00+00:00", -1.5]]
    assert record["n_northing_changepoints"] == 0


@pytest.mark.parametrize("site", ["kel", "pen"])
def test_a_greenbyte_site_reads_the_labels_its_loader_wrote(site: str) -> None:
    frame = pd.DataFrame({"a": [1.0]})
    labelled, config = site_operating_state(site, frame)
    assert labelled is frame
    assert config == operating_state_config(GREENBYTE_OPERATING_STATE)
    assert config.label_column == GREENBYTE_STATE_COL


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


def test_a_failed_cells_reason_reaches_the_run_log(tmp_path: Path) -> None:
    failing = matrix_cells(SMALL)[0]
    probe = _run(SMALL, tmp_path, limit=0)
    (detail_dir(probe) / f"fail_{failing.cell_id}").touch()
    study = _run(SMALL, tmp_path)
    assert f"{failing.cell_id} failed" in (study / "run.log").read_text()
    assert "RuntimeError: this campaign is broken" in (study / "run.log").read_text()


def test_seed_indices_are_written_as_integers(tmp_path: Path) -> None:
    study = _run(SMALL, tmp_path)
    cells = pd.read_csv(study / "cells.csv", dtype=str)
    assert set(cells["seed_index"]) == {"0", "1"}


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
    settings = MatrixSettings(seeds={"hot": 1}, multipliers=(1,), ks=(4,), pre_months=(12,), post_months=(1,), real=())
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
        "pre_months": 12,
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


def _real_record(site: str, post_months: int | None, turbines: list[str], **farm: float) -> dict[str, Any]:
    return {
        "arm": "real",
        "site": site,
        "seed_index": None,
        "k": 4,
        "pre_months": 24,
        "post_months": post_months,
        "multiplier": None,
        "status": "ok",
        "turbines": [{"turbine": t, "estimate": 0.027, "truth": None, "sigma": 0.002} for t in turbines],
        "farm": {"estimate": 0.011, "truth": None, **farm},
        "references": [
            {"test_wtg": turbines[0], "turbine": "T14", "uplift": 0.005, "screened": False},
            {"test_wtg": turbines[0], "turbine": "T17", "uplift": 0.04, "screened": True},
        ],
        "record_losses": {
            t: [
                {"reason": "changed upwind of subject", "hours_before": hours, "hours_after": 10.0},
                {"reason": "changed upwind of reference", "hours_before": 100.0, "hours_after": 10.0},
                {"reason": "usable", "hours_before": 800.0 - hours, "hours_after": 80.0},
            ]
            for t, hours in zip(turbines, [100.0, 300.0], strict=False)
        },
        "data_screen": {turbines[0]: {"candidates": [{"candidate": "T15", "verdict": "dropped"}], "hours": []}},
    }


def test_real_rows_are_kept_separate() -> None:
    real = _real_record("hot_t13", 6, ["T13"])
    tables = baseline_tables([*_line_records(), real])
    assert "hot_t13" not in set(tables["groups"]["site"])
    row = tables["real"].iloc[0]
    assert row["test_estimate"] == pytest.approx(0.011)
    assert row["test_sigma"] == pytest.approx(0.002)
    assert row["ref_mean"] == pytest.approx(0.0225)


def test_the_real_table_has_a_row_per_campaign_and_keeps_lmax() -> None:
    tuneup = _real_record(
        "hot_tuneup", None, ["T02", "T03"], reference_uplift_before_screen=0.004, reference_uplift_after_screen=0.001
    )
    real = baseline_tables([_real_record("hot_t13", 6, ["T13"]), tuneup])["real"]
    assert list(zip(real["site"], real["post_months"], strict=True)) == [("hot_t13", 6), ("hot_tuneup", pd.NA)]
    row = real.iloc[1]
    assert np.isnan(row["test_sigma"])
    assert row["test_n"] == 2
    assert (row["combined_ref_before"], row["combined_ref_after"]) == (0.004, 0.001)
    # Changed upwind is (100 + 100) / 900 of T02's before hours and (300 + 100) / 900 of T03's.
    assert row["changed_upwind_share_median"] == pytest.approx(300 / 900)
    assert row["changed_upwind_share_max"] == pytest.approx(400 / 900)
    assert (row["data_screen_dropped"], row["screen_ejections"]) == (1, 1)


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
    assert str(leaks["seed_index"].dtype) == "Int64"


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
        seeds={"hot": 1}, multipliers=(1,), ks=(3,), pre_months=(12,), post_months=(1,), real=(), exclusion_arms=()
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
    chosen = plan_settings(Cell("main", "hot", 0, 1, 6, 12, 1), settings=MatrixSettings(min_side_days=10))
    assert chosen.k == 6
    assert chosen.min_side == pd.Timedelta(days=10)
    assert chosen.pre_cap == PlanSettings().pre_cap


# --- method overrides --------------------------------------------------------------------------


def test_method_overrides_reach_the_matrix_method(tmp_path: Path) -> None:
    spec = SimpleNamespace(rated_power_kw=2300.0)
    common = {"columns": HOT_COLUMNS, "out_dir": tmp_path, "era5": None, "screen_cache": {}, "era5_label": "x"}
    plain = _matrix_method(spec, **common)  # type: ignore[arg-type]
    assert plain.row_dump_dir is None  # type: ignore[attr-defined]
    dumped = _matrix_method(
        spec,  # type: ignore[arg-type]
        method_overrides={"row_dump_dir": tmp_path / "dump"},
        **common,
    )
    assert dumped.row_dump_dir == tmp_path / "dump"  # type: ignore[attr-defined]
    # the matrix's own overrides survive
    assert dumped.model_params["n_jobs"] == 1  # type: ignore[attr-defined]
    assert dumped.record_reversal is True  # type: ignore[attr-defined]


def _sigma_records() -> list[dict[str, Any]]:
    """Two campaigns whose test error and reference readings sit at known multiples of their sigma."""
    records = []
    for site, seed, error, ref in (("hot", 0, 0.005, 0.015), ("pen", 1, 0.02, 0.005)):
        records.append(
            _record(
                site,
                seed,
                1,
                turbines=[
                    {"turbine": "T01", "estimate": 0.04 + error, "truth": 0.04, "sigma": 0.01, "sigma_block_6h": 0.03}
                ],
                farm={"estimate": 0.04 + error, "truth": 0.04},
                references=[
                    {"test_wtg": "T01", "turbine": "T02", "uplift": ref, "screened": False, "sigma": 0.01},
                    {"test_wtg": "T01", "turbine": "T03", "uplift": 0.0, "screened": False, "sigma": 0.02},
                ],
                bins=[
                    {
                        "test_wtg": "T01",
                        "condition": "power",
                        "condition_bin": "(0, 1]",
                        "estimate": 0.05 + error,
                        "truth": 0.05,
                        "sigma": 0.01,
                    },
                    {
                        "test_wtg": "T01",
                        "condition": "power",
                        "condition_bin": "(1, 2]",
                        "estimate": 0.03,
                        "truth": 0.03,
                        "sigma": float("nan"),
                    },
                ],
            )
        )
    return records


def test_merge_computes_the_coverage_of_every_sigma() -> None:
    groups = baseline_tables(_sigma_records())["groups"]
    row = groups[(groups["site"] == "all") & (groups["multiplier"] == 1)].iloc[0]
    assert row["test_coverage"] == pytest.approx(0.5)  # 0.005 within 0.01, 0.02 not
    assert row["test_z_sd"] == pytest.approx(np.std([0.5, 2.0]))
    assert row["test_sigma_median"] == pytest.approx(0.01)
    assert row["test_block_6h_coverage"] == pytest.approx(1.0)
    assert row["ref_coverage"] == pytest.approx(0.75)  # 0.015/0.01 is the only reading past one sigma
    assert row["ref_sigma_median"] == pytest.approx(0.015)


def test_records_without_a_sigma_report_no_calibration() -> None:
    groups = baseline_tables(_line_records())["groups"]
    assert "test_coverage" not in groups.columns


def test_merge_computes_the_per_bin_error_and_coverage() -> None:
    bins = baseline_tables(_sigma_records())["bins"]
    pooled = bins[bins["site"] == "all"].set_index("condition_bin")
    assert pooled.loc["(0, 1]", "bin_bias"] == pytest.approx(np.mean([0.005, 0.02]))
    assert pooled.loc["(0, 1]", "bin_coverage"] == pytest.approx(0.5)
    assert pooled.loc["(0, 1]", "bin_n"] == 2
    assert pooled.loc["(1, 2]", "bin_bias"] == pytest.approx(0.0)
    assert np.isnan(pooled.loc["(1, 2]", "bin_coverage"])


def _reversal_records() -> list[dict[str, Any]]:
    """Two campaigns whose directions are recorded: the full reversal reads truth, the forward 1 pp high."""
    records = []
    for site, seed in (("hot", 0), ("pen", 1)):
        for m in (1, 0, -1):
            truth = 0.04 * m
            forward = truth + 0.01
            reverse = (1 + forward) / (1 + truth) ** 2 - 1  # makes the full reversal exactly truth
            records.append(
                _record(
                    site,
                    seed,
                    m,
                    turbines=[
                        {
                            "turbine": "T01",
                            "estimate": forward,
                            "truth": truth,
                            "uplift_forward": forward,
                            "uplift_reverse": reverse,
                            "sigma_reverse": 0.01,
                        }
                    ],
                    farm={
                        "estimate": forward,
                        "truth": truth,
                        "reference_uplift_before_screen": 0.004,
                        "reference_uplift_after_screen": 0.002,
                        "reference_uplift_spread": 0.01,
                    },
                    references=[
                        {
                            "test_wtg": "T01",
                            "turbine": "T02",
                            "uplift": 0.01,
                            "screened": False,
                            "uplift_forward": 0.01,
                            "uplift_reverse": 0.01,
                            "sigma_reverse": 0.01,
                        }
                    ],
                )
            )
    return records


def test_merge_scores_the_full_reversal_beside_the_forward_headline() -> None:
    tables = baseline_tables(_reversal_records())
    row = tables["groups"].set_index(["site", "multiplier"]).loc[("all", 1)]
    assert row["test_bias"] == pytest.approx(0.01)
    assert row["rev_test_bias"] == pytest.approx(0.0, abs=1e-12)
    assert row["rev_ref_mean"] == pytest.approx(0.0, abs=1e-12)  # equal directions cancel
    linearity = tables["linearity"].set_index("site")
    assert linearity.loc["all", "line_bias_mean"] == pytest.approx(0.01)
    assert linearity.loc["all", "rev_line_scale_mean"] == pytest.approx(1.0)
    assert linearity.loc["all", "rev_line_bias_mean"] == pytest.approx(0.0, abs=1e-12)


def test_merge_scores_the_combined_reference_uplift_against_zero() -> None:
    row = baseline_tables(_reversal_records())["groups"].set_index(["site", "multiplier"]).loc[("all", 0)]
    assert row["combined_ref_before_bias"] == pytest.approx(0.004)
    assert row["combined_ref_after_bias"] == pytest.approx(0.002)
    assert row["combined_ref_after_n"] == 2


def test_records_without_the_reverse_direction_report_no_reversal_metrics() -> None:
    tables = baseline_tables(_line_records())
    assert not any(str(c).startswith("rev_") for c in tables["groups"].columns)
    assert not any(str(c).startswith("rev_") for c in tables["linearity"].columns)


def test_reversal_uplift_is_nan_where_a_ratio_is_not_positive() -> None:
    assert reversal_uplift(np.array([0.21, -1.5]), np.array([0.0, 0.0])) == pytest.approx([0.1, np.nan], nan_ok=True)


def test_records_without_bins_give_an_empty_bins_table() -> None:
    assert baseline_tables(_line_records())["bins"].empty


def test_the_truth_rows_gain_the_aligned_reanalysis_wind_speed() -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="10min", tz="UTC")
    original = pd.DataFrame({"TurbineName": ["T01"] * 4 + ["T02"] * 4, "power": 1.0}, index=index.append(index))
    aligned = pd.DataFrame({"wind_speed_100m": [5.0, 6.0, 7.0, 8.0], "other": 0.0}, index=index)
    joined = with_reanalysis_wind_speed(original, aligned)
    assert list(joined.columns) == ["TurbineName", "power", "wind_speed_100m"]
    assert joined["wind_speed_100m"].tolist() == [5.0, 6.0, 7.0, 8.0] * 2
    assert with_reanalysis_wind_speed(original, None) is original


def test_a_cell_records_each_test_turbines_losses_and_data_screen(tmp_path: Path) -> None:
    run_dir = tmp_path / "method" / "T01" / "power_model_T01_20200101_20210101"
    run_dir.mkdir(parents=True)
    pd.DataFrame({"reason": ["usable", "rare waking configuration"], "hours_before": [100.0, 2.5]}).to_csv(
        run_dir / "power_model_T01_20200101_20210101_record_losses_20261014.csv", index=False
    )
    screen = SimpleNamespace(
        table=pd.DataFrame({"candidate": ["T02"], "correlation": [0.9], "verdict": ["kept"], "rank": [1]}),
        hours=pd.DataFrame({"n_references": [1], "hours_before": [np.nan]}),
    )
    details = measurement_details({"T01": screen, "T03": screen}, method_dir=tmp_path / "method")  # type: ignore[dict-item]
    assert details["record_losses"] == {
        "T01": [
            {"reason": "usable", "hours_before": 100.0},
            {"reason": "rare waking configuration", "hours_before": 2.5},
        ]
    }
    assert details["data_screen"]["T01"]["candidates"] == [
        {"candidate": "T02", "correlation": 0.9, "verdict": "kept", "rank": 1.0}
    ]
    assert details["data_screen"]["T03"]["hours"] == [{"n_references": 1.0, "hours_before": None}]
    json.dumps(details)


def test_a_cell_reads_the_losses_a_plots_run_writes_without_a_run_folder(tmp_path: Path) -> None:
    wtg_dir = tmp_path / "method" / "T01"
    wtg_dir.mkdir(parents=True)
    pd.DataFrame({"reason": ["usable"], "hours_before": [100.0]}).to_csv(wtg_dir / "record_losses.csv", index=False)
    screen = SimpleNamespace(table=pd.DataFrame({"candidate": []}), hours=pd.DataFrame({"n_references": []}))
    details = measurement_details({"T01": screen}, method_dir=tmp_path / "method")  # type: ignore[dict-item]
    assert details["record_losses"] == {"T01": [{"reason": "usable", "hours_before": 100.0}]}


def test_plot_cell_runs_with_plots_into_its_own_folder_and_leaves_the_study_alone(tmp_path: Path) -> None:
    study, cell = tmp_path / "study", matrix_cells(SMALL)[0]
    detail_dir(study).mkdir(parents=True)
    calls: list[bool] = []

    def execute(cell: Cell, *, detail_dir: Path, cell_dir: Path, settings: MatrixSettings, plots: bool) -> dict:  # noqa: ARG001
        calls.append(plots)
        logging.getLogger("test").warning("measuring %s", cell.cell_id)
        return {"turbines": []}

    out = plot_cell(study, cell, settings=SMALL, out_dir=tmp_path / "plots", execute=execute)
    assert out == tmp_path / "plots" / cell.cell_id
    assert calls == [True]
    assert json.loads((out / "record.json").read_text()) == {"turbines": []}
    assert "measuring" in (out / "run.log").read_text()
    assert not record_path(study, cell).exists()
