"""The v0 probe's pure parts and its driver on a fake estimator; the real runs are HPC drivers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from benchmarking.campaigns import v0_probe
from benchmarking.campaigns.v0_probe import (
    ESTIMATES_CSV,
    PAIRS_CSV,
    PROBE_JSON,
    SUMMARY_CSV,
    nearest_references,
    neighbourhoods,
    run_hot,
    summarise,
    v0_pair_readings,
)

if TYPE_CHECKING:
    from pathlib import Path

# A row of turbines along one parallel, 500 m apart, plus one well to the north.
COORDS = {
    "T01": (57.50, -3.080),
    "T02": (57.50, -3.0717),
    "T03": (57.50, -3.0634),
    "T04": (57.50, -3.0551),
    "T09": (57.53, -3.080),
}


def test_nearest_references_are_the_closest_by_great_circle_and_never_the_turbine_itself() -> None:
    assert nearest_references(COORDS, "T01", n=2) == ["T02", "T03"]
    assert nearest_references(COORDS, "T04", n=3) == ["T03", "T02", "T01"]
    assert "T09" not in nearest_references(COORDS, "T01", n=3)


def test_neighbourhoods_give_every_turbine_its_own_references() -> None:
    hoods = neighbourhoods(["T01", "T04"], coords=COORDS, n_refs=2)
    assert hoods == {"T01": ["T02", "T03"], "T04": ["T03", "T02"]}


def _write_v0_results(scratch: Path, name: str, rows: list[dict]) -> None:
    out = scratch / "v0_T05_x"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / name)


def test_v0_pair_readings_read_the_final_per_pair_file_in_pp_without_self_pairs(tmp_path: Path) -> None:
    rows = [
        {
            "test_wtg": "T05",
            "ref": "T01",
            "uplift_frc": -0.0019,
            "poweronly_uplift_frc": 0.0015,
            "reversed_uplift_frc": 0.0037,
            "unc_one_sigma_frc": 0.004,
        },
        {
            "test_wtg": "T01",
            "ref": "T01",
            "uplift_frc": -0.0177,
            "poweronly_uplift_frc": None,
            "reversed_uplift_frc": None,
            "unc_one_sigma_frc": 0.01,
        },
        {
            "test_wtg": "T01",
            "ref": "T02",
            "uplift_frc": -0.0112,
            "poweronly_uplift_frc": -0.0061,
            "reversed_uplift_frc": -0.0062,
            "unc_one_sigma_frc": 0.004,
        },
    ]
    _write_v0_results(tmp_path, "results_interim.csv", rows[:1])  # an interim file must lose to the final one
    _write_v0_results(tmp_path, "v0_T05_x_results_per_test_ref_20261002_090000.csv", rows)
    pairs = v0_pair_readings(tmp_path)
    assert list(pairs.columns) == ["read_turbine", "against", "variant", "reading_pp", "unc_one_sigma_pp"]
    assert len(pairs) == 2 * 3  # two real pairs, three variants each
    headline = pairs[(pairs["variant"] == "pair") & (pairs["read_turbine"] == "T05")]
    assert headline["reading_pp"].item() == pytest.approx(-0.19)
    assert not ((pairs["read_turbine"] == "T01") & (pairs["against"] == "T01")).any()


def test_v0_pair_readings_fall_back_to_the_interim_file(tmp_path: Path) -> None:
    _write_v0_results(
        tmp_path,
        "results_interim.csv",
        [
            {
                "test_wtg": "T05",
                "ref": "T01",
                "uplift_frc": 0.01,
                "poweronly_uplift_frc": 0.02,
                "reversed_uplift_frc": 0.0,
                "unc_one_sigma_frc": 0.004,
            }
        ],
    )
    assert len(v0_pair_readings(tmp_path)) == 3


def test_summarise_pivots_the_methods_and_adds_their_difference_and_a_mean_row() -> None:
    estimates = pd.DataFrame(
        {
            "test": ["T05", "T05", "T16", "T16"],
            "method": ["power_model", "v0_binned", "power_model", "v0_binned"],
            "estimate_pp": [1.28, -0.47, -0.53, -0.9],
            "references": ["T01,T02,T04,T06"] * 2 + ["T15,T17,T18,T19"] * 2,
        }
    )
    table = summarise(estimates)
    assert list(table.index) == ["T05", "T16", "mean"]
    assert table.loc["T05", "power_model_minus_v0_pp"] == pytest.approx(1.75)
    assert table.loc["mean", "power_model"] == pytest.approx(0.375)
    assert table.loc["T16", "references"] == "T15,T17,T18,T19"


def _fake_estimate(test: str, refs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test == "T03":
        msg = "v0 fell over"
        raise RuntimeError(msg)
    estimates = pd.DataFrame(
        {"method": ["power_model", "v0_binned"], "estimate_pp": [1.0, -0.5], "references": [",".join(refs)] * 2}
    )
    pairs = pd.DataFrame(
        {
            "read_turbine": [test],
            "against": [refs[0]],
            "method": ["v0_binned"],
            "variant": ["pair"],
            "reading_pp": [-0.5],
            "unc_one_sigma_pp": [0.4],
        }
    )
    return estimates, pairs


def test_run_hot_writes_progressively_logs_a_failed_turbine_and_summarises(tmp_path: Path) -> None:
    run_dir = run_hot(
        turbines=["T01", "T03", "T04"], n_refs=2, coords=COORDS, estimate=_fake_estimate, out_root=tmp_path
    )
    estimates = pd.read_csv(run_dir / ESTIMATES_CSV)
    assert sorted(estimates["test"].unique()) == ["T01", "T04"]
    assert set(estimates["method"]) == {"power_model", "v0_binned"}
    pairs = pd.read_csv(run_dir / PAIRS_CSV)
    assert list(pairs["read_turbine"]) == ["T01", "T04"]
    meta = json.loads((run_dir / PROBE_JSON).read_text())
    assert meta["neighbourhoods"] == {"T01": ["T02", "T03"], "T03": ["T02", "T04"], "T04": ["T03", "T02"]}
    assert meta["failed"] == ["T03"]
    summary = pd.read_csv(run_dir / SUMMARY_CSV, index_col=0)
    assert list(summary.index) == ["T01", "T04", "mean"]


def test_run_hot_resumes_a_run_without_redoing_finished_turbines(tmp_path: Path) -> None:
    seen = []

    def counting(test: str, refs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        seen.append(test)
        return _fake_estimate(test, refs)

    first = run_hot(turbines=["T01"], n_refs=2, coords=COORDS, estimate=counting, out_root=tmp_path)
    second = run_hot(
        turbines=["T01", "T04"], n_refs=2, coords=COORDS, estimate=counting, out_root=tmp_path, resume=first
    )
    assert second == first
    assert seen == ["T01", "T04"]
    assert sorted(pd.read_csv(first / ESTIMATES_CSV)["test"].unique()) == ["T01", "T04"]


def test_the_command_line_reads_the_env_file_before_running(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(v0_probe, "load_env", lambda: calls.append("env"))
    monkeypatch.setattr(v0_probe, "run_hot", lambda **kw: calls.append(("hot", kw["n_refs"], kw["smoke"])))
    v0_probe.main(["hot", "--n-refs", "3"])
    assert calls == ["env", ("hot", 3, False)]
