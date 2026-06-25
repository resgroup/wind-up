"""Tests for the synthetic dataset generator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.generator import (
    SyntheticDataset,
    ToggleSchedule,
    _treated_mask,
    generate_dataset,
    treated_mask,
)
from benchmarking.synthetic.upgrades import ConstantCpChange
from wind_up.constants import TIMESTAMP_COL

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
                HOT_COLUMNS.turbine: turbine,
                HOT_COLUMNS.active_power: float(power),
                HOT_COLUMNS.wind_speed: 8.0,
                HOT_COLUMNS.wind_speed_sd: 0.8,
                HOT_COLUMNS.gen_rpm: 1400.0,
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

    t01 = synthetic[synthetic[HOT_COLUMNS.turbine] == "T01"]
    pre = t01[t01.index < changeover][HOT_COLUMNS.active_power]
    post = t01[t01.index >= changeover][HOT_COLUMNS.active_power]
    assert np.allclose(pre.to_numpy(), 1000.0)
    assert np.all(post.to_numpy() > 1000.0)

    t02 = synthetic[synthetic[HOT_COLUMNS.turbine] == "T02"][HOT_COLUMNS.active_power]
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
    t01 = dataset.synthetic_df[dataset.synthetic_df[HOT_COLUMNS.turbine] == "T01"]
    start = t01.index.min()
    half = pd.Timedelta(hours=12)
    first_half = t01[t01.index < start + half][HOT_COLUMNS.active_power]
    second_half = t01[(t01.index >= start + half) & (t01.index < start + 2 * half)][HOT_COLUMNS.active_power]
    assert np.allclose(first_half.to_numpy(), 1000.0)  # first half-period: toggle off
    assert np.all(second_half.to_numpy() > 1000.0)  # second half-period: toggle on


def test_toggle_start_leaves_pre_start_rows_untreated() -> None:
    """A ToggleSchedule.start places baseline before toggling: pre-start rows stay untreated."""
    wf_df = _wf_df(periods=288)  # two days at 10-min
    timestamps = wf_df.index.unique()
    start = timestamps[len(timestamps) // 2]  # toggling begins at the midpoint

    dataset = generate_dataset(
        scada_df=wf_df,
        test_wtgs=["T01"],
        upgrades=[ConstantCpChange(delta=0.10)],
        mode="toggle",
        upgrade_timing=ToggleSchedule(period=pd.Timedelta(hours=12), start=start),
    )
    t01 = dataset.synthetic_df[dataset.synthetic_df[HOT_COLUMNS.turbine] == "T01"]
    before = t01[t01.index < start][HOT_COLUMNS.active_power]
    assert np.allclose(before.to_numpy(), 1000.0)  # baseline before toggling: untreated

    # start_on defaults False: [start, start+6h) off, [start+6h, start+12h) on.
    half = pd.Timedelta(hours=6)
    off_block = t01[(t01.index >= start) & (t01.index < start + half)][HOT_COLUMNS.active_power]
    on_block = t01[(t01.index >= start + half) & (t01.index < start + 2 * half)][HOT_COLUMNS.active_power]
    assert np.allclose(off_block.to_numpy(), 1000.0)
    assert np.all(on_block.to_numpy() > 1000.0)


def test_public_treated_mask_infers_mode_from_timing_type() -> None:
    """treated_mask infers prepost vs toggle from the upgrade_timing type."""
    wf_df = _wf_df(turbines=("T01",), periods=288)
    index = wf_df.index
    changeover = index[len(index) // 2]

    prepost_mask = treated_mask(index, changeover)
    assert np.array_equal(prepost_mask, np.asarray(index >= changeover))

    schedule = ToggleSchedule(period=pd.Timedelta(hours=12), start=changeover)
    toggle_mask = treated_mask(index, schedule)
    assert not toggle_mask[np.asarray(index < changeover)].any()  # baseline untreated
    assert toggle_mask[np.asarray(index >= changeover)].any()  # some on-rows after start


def test_treated_mask_rejects_mode_timing_mismatch() -> None:
    """A mode that disagrees with the upgrade_timing type fails fast with a clear TypeError."""
    index = _wf_df(turbines=("T01",), periods=48).index
    changeover = index[len(index) // 2]
    schedule = ToggleSchedule(period=pd.Timedelta(hours=12), start=changeover)

    with pytest.raises(TypeError, match="prepost mode needs a changeover Timestamp"):
        _treated_mask(index, mode="prepost", upgrade_timing=schedule)
    with pytest.raises(TypeError, match="toggle mode needs a ToggleSchedule"):
        _treated_mask(index, mode="toggle", upgrade_timing=changeover)


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
