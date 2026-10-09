"""The level probe's analysis path, on toy dumps; the campaign runs themselves are drivers."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.campaigns import level_probe
from benchmarking.campaigns.level_probe import (
    PROBE_JSON,
    PROBE_LOG,
    SUMMARY_COLUMNS,
    _log_to,
    analyse_run,
    matrix_cells,
    run_all,
)
from tests.benchmarking.campaigns.test_level_analysis import _FAST, _POWER, _Q, _dump, _with_meta

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from benchmarking.campaigns.level_analysis import RowDump


def _write_run(run_dir: Path, *, arms: list[str], dumps: dict[str, RowDump]) -> None:
    run_dir.mkdir(parents=True)
    meta = {"case": "pen", "arms": arms, "test_wtgs": ["T14"], "analyse": ["T14"], "rotor_diameter_m": 82.0}
    (run_dir / PROBE_JSON).write_text(json.dumps(meta))
    for arm, dump in dumps.items():
        dump.save(run_dir / "dump" / arm / "T14")


def _arm(*, held_booleans: bool, extra_bias: float = 0.0) -> RowDump:
    rng = np.random.default_rng(5)
    base = _dump(n_base=600, n_up=400, references=("R1", "R2", "R3"), bias_kw=rng.normal(10.0, 5.0, 400))
    rows = base.rows.copy()
    held = np.zeros(len(rows), dtype=bool)
    held[np.flatnonzero(rows["upgraded"])[50:120]] = True
    rows.loc[held, f"{_POWER}{_Q}R3"] = np.nan
    if held_booleans:
        rows[f"waking_{_POWER}{_Q}R3"] = 1.0
    rows.loc[held, "counterfactual_kw"] += extra_bias
    return _with_meta(rows, references=["R1", "R2", "R3"], test_wtg="T14")


def test_analyse_run_writes_every_table_and_the_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "pen_run"
    arms = ["excl_booleans", "excl_nan"]
    _write_run(
        run_dir,
        arms=arms,
        dumps={"excl_booleans": _arm(held_booleans=True), "excl_nan": _arm(held_booleans=False, extra_bias=30.0)},
    )

    summary = analyse_run(run_dir, model_params=_FAST)

    assert list(summary.columns) == list(SUMMARY_COLUMNS)
    assert list(summary["arm"]) == arms
    assert summary["is_test"].all()
    assert (run_dir / "summary.csv").is_file()
    assert (run_dir / "summary.png").is_file()
    turbine_dir = run_dir / "analysis" / "excl_booleans" / "T14"
    for name in ("month", "wind_band", "operating_pattern", "propensity_decile", "direction_sector"):
        assert (turbine_dir / f"level_by_{name}.csv").is_file()
        assert (turbine_dir / f"level_by_{name}.png").is_file()
    # no coordinates in the toy, so the upwind grouping is skipped rather than failing the run
    assert not (turbine_dir / "level_by_upwind_offline.csv").exists()
    for name in ("propensity.csv", "propensity.png", "aipw.csv"):
        assert (turbine_dir / name).is_file()

    held = pd.read_csv(run_dir / "analysis" / "T14_channels" / "channel_difference_by_held_back.csv").set_index("group")
    observed = summary.set_index("arm")["observed_level_pp"]
    assert abs(held.loc["held back", "difference_pp"] - (observed["excl_booleans"] - observed["excl_nan"])) < 1e-9


def test_the_matrix_cases_are_their_sites_l12_placebo_cell_in_three_arms() -> None:
    assert [c.cell_id for c in matrix_cells("pen")] == [
        "pen_s00_m+0_K4_L12",
        "pen_s00_m+0_K4_L12_xbool",
        "pen_s00_m+0_K4_L12_xnan",
    ]
    assert [c.cell_id for c in matrix_cells("kel")] == [
        "kel_s00_m+0_K4_L12",
        "kel_s00_m+0_K4_L12_xbool",
        "kel_s00_m+0_K4_L12_xnan",
    ]


def test_a_failed_case_does_not_stop_the_series(tmp_path: Path) -> None:
    ran = []

    def ok(name: str) -> Path:
        ran.append(name)
        return tmp_path / name

    def boom() -> Path:
        ran.append("kel")
        msg = "no data"
        raise RuntimeError(msg)

    done = run_all({"pen": lambda: ok("pen"), "kel": boom, "hot": lambda: ok("hot")})
    assert ran == ["pen", "kel", "hot"]
    assert done == {"pen": tmp_path / "pen", "kel": None, "hot": tmp_path / "hot"}


def test_each_case_logs_to_its_own_file_alone(tmp_path: Path) -> None:
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.INFO)
    try:
        for case in ("a", "b"):
            (tmp_path / case).mkdir()
            _log_to(tmp_path / case / PROBE_LOG)
            logging.getLogger("benchmarking.test").info("case %s", case)
    finally:
        _log_to(tmp_path / PROBE_LOG)  # leave one handler, then drop it
        for h in [h for h in root.handlers if getattr(h, "name", None) == "level_probe"]:
            root.removeHandler(h)
            h.close()
        root.setLevel(level)
    assert "case b" not in (tmp_path / "a" / PROBE_LOG).read_text()
    assert "case a" not in (tmp_path / "b" / PROBE_LOG).read_text()


def test_every_turbine_overrides_the_runs_choice(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, arms=["main"], dumps={"main": _arm(held_booleans=False)})
    _arm(held_booleans=False).save(run_dir / "dump" / "main" / "R9")
    assert len(analyse_run(run_dir, model_params=_FAST)) == 1  # the run chose T14 alone
    assert len(analyse_run(run_dir, every_turbine=True, model_params=_FAST)) == 2


def test_the_command_line_reads_the_env_file_before_running(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(level_probe, "load_env", lambda: calls.append("env"))
    monkeypatch.setattr(level_probe, "run_all", lambda: calls.append("all"))
    level_probe.main(["all"])
    assert calls == ["env", "all"]
