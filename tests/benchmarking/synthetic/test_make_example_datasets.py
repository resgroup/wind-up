"""Tests for the example-dataset driver (one dataset per Issue 1 profile)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import benchmarking.synthetic.make_example_datasets as driver
from benchmarking.synthetic.make_example_datasets import example_profiles, generate_example_datasets, main
from wind_up.constants import TIMESTAMP_COL, DataColumns

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_PROFILES = {"constant_cp", "wind_speed_cp", "ti_cp", "rated_power"}


def _swept_wf_df(
    *, periods: int = 288, turbines: tuple[str, ...] = ("T01", "T02"), start: str = "2020-01-01"
) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="10min", tz="UTC")
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
        assert (tmp_path / name / "power_curve_T01.png").exists()
        assert dataset.true_uplift().overall != 0.0


def test_main_wires_hot_loader_to_the_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` loads SCADA via ``load_hot_scada`` and writes one dataset per profile."""
    captured: dict = {}

    def fake_load_hot_scada(**kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured.update(kwargs)
        # Span exactly the requested window, as the real loader does, so the
        # mid-window changeover lands on real rows.
        start_dt = kwargs["start_dt"]
        end_dt_excl = kwargs["end_dt_excl"]
        periods = int((end_dt_excl - start_dt) / pd.Timedelta(minutes=10))
        wf_df = _swept_wf_df(periods=periods, start=str(start_dt))
        return wf_df, pd.DataFrame({"Name": ["T01", "T02"]})

    monkeypatch.setattr(driver, "load_hot_scada", fake_load_hot_scada)

    datasets = main(out_root=tmp_path, test_wtg="T01")

    assert set(datasets) == EXPECTED_PROFILES
    for name in EXPECTED_PROFILES:
        assert (tmp_path / name / "synthetic.parquet").exists()
    # The injection changeover must land inside the loaded window so rows are treated.
    assert captured["start_dt"] < captured["end_dt_excl"]


@pytest.mark.slow
def test_main_end_to_end_downloads_real_hot_data(tmp_path: Path) -> None:
    """End-to-end: download real Hill of Towie data and produce datasets (network)."""
    datasets = main(out_root=tmp_path / "out", data_dir=tmp_path / "data", test_wtg="T01")
    assert set(datasets) == EXPECTED_PROFILES
    for name in EXPECTED_PROFILES:
        assert (tmp_path / "out" / name / "synthetic.parquet").exists()
        assert datasets[name].true_uplift().overall != 0.0
