"""Synthetic upgrade-dataset generator (v1 benchmarking, WS1).

Inject a known turbine upgrade into real no-upgrade SCADA to create datasets with a
derivable ground-truth uplift, for objectively evaluating uplift methods.
"""

from __future__ import annotations

from benchmarking.synthetic.cp_core import HOT_CP_MODEL, CpCore, CpParams, cp_surface
from benchmarking.synthetic.faults import (
    WIND_SPEED_ROLES,
    Fault,
    NorthingStep,
    SensorGainDrift,
    SensorGainStep,
)
from benchmarking.synthetic.generator import SyntheticDataset, ToggleSchedule, generate_dataset, treated_mask
from benchmarking.synthetic.geometry import WakePair, bearing_deg, derive_wake_steering_pairs, distance_m, wrap180
from benchmarking.synthetic.ground_truth import UpliftResult, true_farm_uplift, true_net_uplift, true_uplift
from benchmarking.synthetic.plots import (
    plot_power_curve_comparison,
    plot_wake_steering_by_direction,
    plot_wake_steering_heatmaps,
    plot_wake_steering_stability,
)
from benchmarking.synthetic.schema import ColumnSchema
from benchmarking.synthetic.solar import diurnal_factor, sin_solar_elevation
from benchmarking.synthetic.sources.hill_of_towie import (
    HOT_ACTIVE_POWER_STAT_COLS,
    HOT_COLUMNS,
    HOT_HUB_HEIGHT_M,
    HOT_LAT,
    HOT_LON,
    HOT_RATED_POWER_KW,
)
from benchmarking.synthetic.upgrades import (
    ConditionCpChange,
    ConstantCpChange,
    RatedPowerChange,
    UpgradeEffect,
    WakeSteering,
    WindSpeedCpChange,
    apply_upgrades,
    north_calibrated_direction,
)

__all__ = [
    "HOT_ACTIVE_POWER_STAT_COLS",
    "HOT_COLUMNS",
    "HOT_CP_MODEL",
    "HOT_HUB_HEIGHT_M",
    "HOT_LAT",
    "HOT_LON",
    "HOT_RATED_POWER_KW",
    "WIND_SPEED_ROLES",
    "ColumnSchema",
    "ConditionCpChange",
    "ConstantCpChange",
    "CpCore",
    "CpParams",
    "Fault",
    "NorthingStep",
    "RatedPowerChange",
    "SensorGainDrift",
    "SensorGainStep",
    "SyntheticDataset",
    "ToggleSchedule",
    "UpgradeEffect",
    "UpliftResult",
    "WakePair",
    "WakeSteering",
    "WindSpeedCpChange",
    "apply_upgrades",
    "bearing_deg",
    "cp_surface",
    "derive_wake_steering_pairs",
    "distance_m",
    "diurnal_factor",
    "generate_dataset",
    "north_calibrated_direction",
    "plot_power_curve_comparison",
    "plot_wake_steering_by_direction",
    "plot_wake_steering_heatmaps",
    "plot_wake_steering_stability",
    "sin_solar_elevation",
    "treated_mask",
    "true_farm_uplift",
    "true_net_uplift",
    "true_uplift",
    "wrap180",
]
