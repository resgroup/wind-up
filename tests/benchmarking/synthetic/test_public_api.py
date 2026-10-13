"""The synthetic package exposes its main entry points at the package root."""

from __future__ import annotations

import benchmarking.synthetic as synth


def test_public_api_exports_core_entry_points() -> None:
    for name in (
        "generate_dataset",
        "SyntheticDataset",
        "ToggleSchedule",
        "treated_mask",
        "true_uplift",
        "CpCore",
        "HOT_CP_MODEL",
        "ConstantCpChange",
        "WindSpeedCpChange",
        "ConditionCpChange",
        "RatedPowerChange",
    ):
        assert hasattr(synth, name), name
