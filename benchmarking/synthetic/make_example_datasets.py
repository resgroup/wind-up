"""Driver: produce one synthetic dataset per Issue 1 profile.

``generate_example_datasets`` is source-agnostic (give it any wind-up-format SCADA).
``main`` wires it to the Hill of Towie open data so the whole thing runs end-to-end.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from benchmarking.synthetic.generator import SyntheticDataset, ToggleSchedule, generate_dataset
from benchmarking.synthetic.upgrades import (
    ConditionCpChange,
    ConstantCpChange,
    RatedPowerChange,
    WindSpeedCpChange,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def example_profiles() -> dict[str, list]:
    """Return the four Issue 1 upgrade profiles with concrete example parameters.

    - ``constant_cp``: a flat +3% region-2 Cp change (e.g. a blade add-on).
    - ``wind_speed_cp``: a region-2 Cp change that peaks mid-region and tails to 0 at
      rated (the AeroUp shape).
    - ``ti_cp``: a Cp change that is larger at low turbulence intensity.
    - ``rated_power``: a +5% rated-power uprate.
    """
    return {
        "constant_cp": [ConstantCpChange(delta=0.03)],
        "wind_speed_cp": [WindSpeedCpChange(ws_points=(4.0, 7.0, 10.0, 13.0), deltas=(0.0, 0.04, 0.02, 0.0))],
        "ti_cp": [ConditionCpChange(by="ti", points=(0.05, 0.15), deltas=(0.04, 0.0))],
        "rated_power": [RatedPowerChange(new_rated_power_kw=2415.0)],
    }


def generate_example_datasets(
    *,
    scada_df: pd.DataFrame,
    test_wtgs: list[str],
    mode: Literal["prepost", "toggle"],
    upgrade_timing: pd.Timestamp | ToggleSchedule,
    out_root: str | Path | None = None,
    seed: int = 0,
) -> dict[str, SyntheticDataset]:
    """Generate (and optionally save) one synthetic dataset per example profile.

    :param scada_df: wind-up-format real SCADA (all turbines), the no-upgrade baseline
    :param test_wtgs: turbine name(s) to upgrade
    :param mode: ``"prepost"`` or ``"toggle"``
    :param upgrade_timing: changeover timestamp (prepost) or toggle schedule
    :param out_root: if given, each dataset is saved under ``out_root / <profile_name>``
    :param seed: top-level seed for reproducibility
    :return: mapping of profile name to its SyntheticDataset
    """
    datasets: dict[str, SyntheticDataset] = {}
    for name, upgrades in example_profiles().items():
        dataset = generate_dataset(
            scada_df=scada_df,
            test_wtgs=test_wtgs,
            upgrades=upgrades,
            mode=mode,
            upgrade_timing=upgrade_timing,
            seed=seed,
        )
        if out_root is not None:
            dataset.save(Path(out_root) / name)
        datasets[name] = dataset
    return datasets
