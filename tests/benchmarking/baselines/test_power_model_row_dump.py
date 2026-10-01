"""The opt-in per-row dump on ``PowerModelMethod`` (the C3 level probe's input)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.power_model import PowerModelMethod
from benchmarking.baselines.power_model.features import QUALIFIER
from benchmarking.harness.context import CampaignContext
from benchmarking.harness.method import MethodInput
from tests.benchmarking.baselines.test_power_model_method import _COLUMNS, _FAST_PARAMS, _POWER, _TURBINE, _toy_scada

if TYPE_CHECKING:
    from pathlib import Path

_N = 4000
_REFERENCES = ["R1", "R2", "R3"]


def _mi() -> MethodInput:
    idx = pd.date_range("2019-01-01", periods=_N, freq="10min", tz="UTC")
    changeover = pd.Timestamp(idx[_N // 2])
    scada = _toy_scada(_N, uplift=0.0, treated=np.asarray(idx >= changeover))
    context = CampaignContext(
        test_wtg="T1",
        timing=changeover,
        turbine_col=_TURBINE,
        candidate_references=_REFERENCES,
        wake_contributors=[],
        valid_for_uplift=pd.DataFrame(data=True, index=idx, columns=["T1", *_REFERENCES]),
    )
    return MethodInput(scada_df=scada, test_wtg="T1", campaign_context=context)


def _method(**overrides: object) -> PowerModelMethod:
    kwargs: dict[str, object] = {
        "columns": _COLUMNS,
        "baseline_rated_power_kw": 2300.0,
        "conditions": (),
        "reference_screen": False,
        "report_reference_uplifts": False,
        "model_params": _FAST_PARAMS,
        **overrides,
    }
    return PowerModelMethod(**kwargs)  # type: ignore[arg-type]


def test_the_dump_is_off_by_default_and_changes_nothing(tmp_path: Path) -> None:
    assert PowerModelMethod(columns=_COLUMNS, baseline_rated_power_kw=2300.0).row_dump_dir is None
    off = _method(out_dir=tmp_path / "off").estimate(_mi()).p50_overall
    on = _method(out_dir=tmp_path / "on", row_dump_dir=tmp_path / "dump").estimate(_mi()).p50_overall
    assert off == on
    assert not list((tmp_path / "off").rglob("rows.parquet"))


def test_the_dump_carries_every_documented_column(tmp_path: Path) -> None:
    mi = _mi()
    out = _method(out_dir=tmp_path / "diag", row_dump_dir=tmp_path / "dump").estimate(mi)

    rows = pd.read_parquet(tmp_path / "dump" / "T1" / "rows.parquet")
    sidecar = json.loads((tmp_path / "dump" / "T1" / "rows.json").read_text())

    for col in ("timestamp", "actual_kw", "counterfactual_kw", "selected", "baseline", "upgraded", "reference_mean_ws"):
        assert col in rows.columns
    # features keep their original names
    assert f"{_POWER}{QUALIFIER}R1" in rows.columns
    assert len(rows) == mi.scada_df.index.nunique()

    finite = np.isfinite(rows["counterfactual_kw"].to_numpy())
    assert (finite == (rows["selected"] & rows["upgraded"]).to_numpy()).all()

    upgraded = rows[rows["selected"] & rows["upgraded"]]
    assert sidecar["sum_actual_kw"] == pytest.approx(upgraded["actual_kw"].sum())
    assert sidecar["sum_counterfactual_kw"] == pytest.approx(upgraded["counterfactual_kw"].sum())
    assert sidecar["sum_actual_kw"] / sidecar["sum_counterfactual_kw"] - 1 == pytest.approx(out.p50_overall)
    assert sidecar["uplift"] == pytest.approx(out.p50_overall)
    assert sidecar["test_wtg"] == "T1"
    assert sidecar["references"] == _REFERENCES
    assert sidecar["power_free"] == []
    assert sidecar["n_upgraded_rows"] == len(upgraded)
    assert sidecar["n_baseline_rows"] == int((rows["selected"] & rows["baseline"]).sum())
    keys = ("wake_only", "timebase_s", "active_power_col", "northed_direction_col", "waking_threshold_kw", "coords")
    for key in keys:
        assert key in sidecar
    assert sidecar["coords"] == {}
    assert sidecar["baseline_rated_power_kw"] == 2300.0


def test_a_reference_reading_lands_in_its_own_folder(tmp_path: Path) -> None:
    _method(out_dir=tmp_path / "diag", row_dump_dir=tmp_path / "dump", report_reference_uplifts=True).estimate(_mi())
    for turbine in ("T1", *_REFERENCES):
        assert (tmp_path / "dump" / turbine / "rows.parquet").is_file()
        assert (tmp_path / "dump" / turbine / "rows.json").is_file()


def test_the_screening_clone_does_not_dump(tmp_path: Path) -> None:
    clone = _method(row_dump_dir=tmp_path / "dump")._screening_clone()  # noqa: SLF001
    assert clone.row_dump_dir is None
    assert _method(row_dump_dir=tmp_path / "dump")._reference_clone().row_dump_dir == tmp_path / "dump"  # noqa: SLF001
