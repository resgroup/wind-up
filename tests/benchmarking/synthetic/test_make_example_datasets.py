"""Tests for the example-dataset driver (one dataset per Issue 1 profile)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.synthetic.make_example_datasets import example_profiles, generate_example_datasets
from wind_up.constants import TIMESTAMP_COL, DataColumns

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_PROFILES = {"constant_cp", "wind_speed_cp", "ti_cp", "rated_power"}


def _swept_wf_df(*, periods: int = 288, turbines: tuple[str, ...] = ("T01", "T02")) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=periods, freq="10min", tz="UTC")
    ws = np.linspace(3.0, 14.0, periods)
    power = np.clip(2300.0 * ((ws - 3.0) / 11.0) ** 3, 0.0, 2300.0)
    frames = [
        pd.DataFrame(
            {
                DataColumns.turbine_name: turbine,
                DataColumns.active_power_mean: power,
                DataColumns.wind_speed_mean: ws,
                DataColumns.wind_speed_sd: 0.1 * ws,
                DataColumns.gen_rpm_mean: 1400.0,
            },
            index=index,
        )
        for turbine in turbines
    ]
    wf_df = pd.concat(frames)
    wf_df.index.name = TIMESTAMP_COL
    return wf_df


def test_example_profiles_cover_the_four_issue1_profiles() -> None:
    """The driver defines the four Issue 1 profiles, each a non-empty upgrade list."""
    profiles = example_profiles()
    assert set(profiles) == EXPECTED_PROFILES
    assert all(len(upgrades) >= 1 for upgrades in profiles.values())


def test_generate_example_datasets_writes_one_dataset_per_profile(tmp_path: Path) -> None:
    """Each profile produces a saved dataset with a non-zero injected uplift."""
    wf_df = _swept_wf_df()
    timestamps = wf_df.index.unique()
    datasets = generate_example_datasets(
        scada_df=wf_df,
        test_wtgs=["T01"],
        mode="prepost",
        upgrade_timing=timestamps[len(timestamps) // 2],
        out_root=tmp_path,
    )

    assert set(datasets) == EXPECTED_PROFILES
    for name, dataset in datasets.items():
        assert (tmp_path / name / "synthetic.parquet").exists()
        assert (tmp_path / name / "run_metadata.json").exists()
        assert dataset.true_uplift().overall != 0.0
