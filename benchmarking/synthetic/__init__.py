"""Synthetic upgrade-dataset generator (v1 benchmarking, WS1).

Inject a known turbine upgrade into real no-upgrade SCADA to create datasets with a
derivable ground-truth uplift, for objectively evaluating uplift methods.
"""

from __future__ import annotations

from benchmarking.synthetic.cp_core import HOT_CP_MODEL, CpCore, CpParams, cp_surface
from benchmarking.synthetic.generator import SyntheticDataset, ToggleSchedule, generate_dataset, treated_mask
from benchmarking.synthetic.ground_truth import UpliftResult, true_uplift
from benchmarking.synthetic.plots import plot_power_curve_comparison
from benchmarking.synthetic.schema import ColumnSchema
from benchmarking.synthetic.sources.hill_of_towie import (
    HOT_ACTIVE_POWER_STAT_COLS,
    HOT_COLUMNS,
    HOT_HUB_HEIGHT_M,
    HOT_RATED_POWER_KW,
)
from benchmarking.synthetic.upgrades import (
    ConditionCpChange,
    ConstantCpChange,
    RatedPowerChange,
    UpgradeEffect,
    WindSpeedCpChange,
    apply_upgrades,
)

__all__ = [
    "HOT_ACTIVE_POWER_STAT_COLS",
    "HOT_COLUMNS",
    "HOT_CP_MODEL",
    "HOT_HUB_HEIGHT_M",
    "HOT_RATED_POWER_KW",
    "ColumnSchema",
    "ConditionCpChange",
    "ConstantCpChange",
    "CpCore",
    "CpParams",
    "RatedPowerChange",
    "SyntheticDataset",
    "ToggleSchedule",
    "UpgradeEffect",
    "UpliftResult",
    "WindSpeedCpChange",
    "apply_upgrades",
    "cp_surface",
    "generate_dataset",
    "plot_power_curve_comparison",
    "treated_mask",
    "true_uplift",
]
