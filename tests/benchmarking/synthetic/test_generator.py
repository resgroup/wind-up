"""Tests for the synthetic dataset generator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.synthetic.generator import SyntheticDataset, ToggleSchedule, generate_dataset
from benchmarking.synthetic.upgrades import ConstantCpChange
from wind_up.constants import TIMESTAMP_COL, DataColumns

if TYPE_CHECKING:
    from pathlib import Path


def _wf_df(
    *,
    turbines: tuple[str, ...] = ("T01", "T02"),
    start: str = "2020-01-01",
    periods: int = 288,
    power: float = 1000.0,
) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="10min", tz="UTC")
    frames = []
    for turbine in turbines:
        frame = pd.DataFrame(
            {
                DataColumns.turbine_name: turbine,
                DataColumns.active_power_mean: float(power),
                DataColumns.wind_speed_mean: 8.0,
                DataColumns.wind_speed_sd: 0.8,
                DataColumns.gen_rpm_mean: 1400.0,
            },
            index=index,
        )
        frames.append(frame)
    wf_df = pd.concat(frames)
    wf_df.index.name = TIMESTAMP_COL
    return wf_df


def test_prepost_modifies_only_post_rows_of_test_turbine() -> None:
    """In prepost mode only the test turbine's post-changeover power changes."""
    wf_df = _wf_df()
    timestamps = wf_df.index.unique()
    changeover = timestamps[len(timestamps) // 2]

    dataset = generate_dataset(
        scada_df=wf_df,
        test_wtgs=["T01"],
        upgrades=[ConstantCpChange(delta=0.10)],
        mode="prepost",
        upgrade_timing=changeover,
    )
    synthetic = dataset.synthetic_df

    t01 = synthetic[synthetic[DataColumns.turbine_name] == "T01"]
    pre = t01[t01.index < changeover][DataColumns.active_power_mean]
    post = t01[t01.index >= changeover][DataColumns.active_power_mean]
    assert np.allclose(pre.to_numpy(), 1000.0)
    assert np.all(post.to_numpy() > 1000.0)

    t02 = synthetic[synthetic[DataColumns.turbine_name] == "T02"][DataColumns.active_power_mean]
    assert np.allclose(t02.to_numpy(), 1000.0)


def test_generator_retains_unchanged_original() -> None:
    """The returned original_df equals the input SCADA and is not mutated."""
    wf_df = _wf_df()
    timestamps = wf_df.index.unique()
    dataset = generate_dataset(
        scada_df=wf_df,
        test_wtgs=["T01"],
        upgrades=[ConstantCpChange(delta=0.10)],
        mode="prepost",
        upgrade_timing=timestamps[len(timestamps) // 2],
    )
    pd.testing.assert_frame_equal(dataset.original_df, wf_df)


def test_toggle_modifies_alternate_blocks() -> None:
    """In toggle mode ``period`` is a full on/off cycle: half off then half on."""
    wf_df = _wf_df(periods=288)  # two days at 10-min
    dataset = generate_dataset(
        scada_df=wf_df,
        test_wtgs=["T01"],
        upgrades=[ConstantCpChange(delta=0.10)],
        mode="toggle",
        upgrade_timing=ToggleSchedule(period=pd.Timedelta(days=1)),
    )
    t01 = dataset.synthetic_df[dataset.synthetic_df[DataColumns.turbine_name] == "T01"]
    start = t01.index.min()
    half = pd.Timedelta(hours=12)
    first_half = t01[t01.index < start + half][DataColumns.active_power_mean]
    second_half = t01[(t01.index >= start + half) & (t01.index < start + 2 * half)][DataColumns.active_power_mean]
    assert np.allclose(first_half.to_numpy(), 1000.0)  # first half-period: toggle off
    assert np.all(second_half.to_numpy() > 1000.0)  # second half-period: toggle on


def test_run_metadata_records_recipe() -> None:
    """Run metadata captures the test turbines, mode, seed and upgrade descriptions."""
    wf_df = _wf_df()
    timestamps = wf_df.index.unique()
    dataset = generate_dataset(
        scada_df=wf_df,
        test_wtgs=["T01"],
        upgrades=[ConstantCpChange(delta=0.10)],
        mode="prepost",
        upgrade_timing=timestamps[len(timestamps) // 2],
        seed=7,
    )
    metadata = dataset.run_metadata
    assert metadata["test_wtgs"] == ["T01"]
    assert metadata["mode"] == "prepost"
    assert metadata["seed"] == 7
    assert metadata["upgrades"][0]["kind"] == "constant_cp"


def _prepost_constant_dataset(delta: float = 0.10) -> SyntheticDataset:
    wf_df = _wf_df()
    timestamps = wf_df.index.unique()
    return generate_dataset(
        scada_df=wf_df,
        test_wtgs=["T01"],
        upgrades=[ConstantCpChange(delta=delta)],
        mode="prepost",
        upgrade_timing=timestamps[len(timestamps) // 2],
    )


def test_dataset_true_uplift_recovers_injected_effect() -> None:
    """SyntheticDataset.true_uplift recovers the region-2-weighted injected uplift."""
    dataset = _prepost_constant_dataset(delta=0.10)
    result = dataset.true_uplift()
    # constant power 1000 kW, region-2 fraction ~0.999 -> ~+9.99%
    assert result.overall == pytest.approx(0.0999, rel=1e-3)


def test_save_writes_roundtrippable_files(tmp_path: Path) -> None:
    """save() writes synthetic, original and metadata (with ground truth) that round-trip."""
    dataset = _prepost_constant_dataset()
    dataset.save(tmp_path)

    assert (tmp_path / "synthetic.parquet").exists()
    assert (tmp_path / "original.parquet").exists()
    assert (tmp_path / "run_metadata.json").exists()

    roundtrip = pd.read_parquet(tmp_path / "synthetic.parquet")
    pd.testing.assert_frame_equal(roundtrip, dataset.synthetic_df)

    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert "ground_truth" in metadata
    assert metadata["ground_truth"]["T01"]["overall"] == pytest.approx(0.0999, rel=1e-3)
