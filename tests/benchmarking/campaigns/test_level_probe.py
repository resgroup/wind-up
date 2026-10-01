"""The level probe's analysis path, on toy dumps; the campaign runs themselves are drivers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.campaigns.level_probe import PROBE_JSON, SUMMARY_COLUMNS, analyse_run, pen_cells
from tests.benchmarking.campaigns.test_level_analysis import _FAST, _POWER, _Q, _dump, _with_meta

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.campaigns.level_analysis import RowDump


def _write_run(run_dir: Path, *, arms: list[str], dumps: dict[str, RowDump]) -> None:
    run_dir.mkdir(parents=True)
    meta = {"case": "pen", "arms": arms, "test_wtg": "T14", "rotor_diameter_m": 82.0}
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


def test_case_b_is_the_oracle_cell_in_three_arms() -> None:
    cells = pen_cells()
    assert [c.cell_id for c in cells] == ["pen_s00_m+0_K4_L12", "pen_s00_m+0_K4_L12_xbool", "pen_s00_m+0_K4_L12_xnan"]
