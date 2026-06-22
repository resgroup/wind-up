"""Driver: produce one synthetic dataset per Issue 1 profile.

``generate_example_datasets`` is source-agnostic (give it any wind-up-format SCADA).
``main`` wires it to the Hill of Towie open data so the whole thing runs end-to-end.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import pandas as pd

from benchmarking.synthetic.generator import SyntheticDataset, ToggleSchedule, generate_dataset
from benchmarking.synthetic.sources.hill_of_towie import load_hot_scada
from benchmarking.synthetic.upgrades import (
    ConditionCpChange,
    ConstantCpChange,
    RatedPowerChange,
    WindSpeedCpChange,
)

logger = logging.getLogger(__name__)

# A stable, no-upgrade Hill of Towie window: comfortably before the real T13 AeroUp
# (installed Sep 2021)
# All of 2016-2020 was previously confirmed stable for T01 and all nearby turbines
# during the setup of the Kaggle power prediction challenge.
DEFAULT_START_DT = pd.Timestamp("2020-06-01", tz="UTC")
DEFAULT_END_DT_EXCL = pd.Timestamp("2020-09-01", tz="UTC")
DEFAULT_TEST_WTG = "T01"


def default_output_root() -> Path:
    """Return the root directory example datasets are written under.

    Overridable via the ``WIND_UP_BENCHMARKING_OUTPUT_DIR`` environment variable;
    defaults to ``~/temp/wind-up-benchmarking/synthetic``.
    """
    return Path(
        os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking" / "synthetic")
    )


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
    save_plots: bool = True,
) -> dict[str, SyntheticDataset]:
    """Generate (and optionally save) one synthetic dataset per example profile.

    :param scada_df: wind-up-format real SCADA (all turbines), the no-upgrade baseline
    :param test_wtgs: turbine name(s) to upgrade
    :param mode: ``"prepost"`` or ``"toggle"``
    :param upgrade_timing: changeover timestamp (prepost) or toggle schedule
    :param out_root: if given, each dataset is saved under ``out_root / <profile_name>``
    :param seed: top-level seed for reproducibility
    :param save_plots: when saving, also write a power-curve comparison PNG per test turbine
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
            dataset_dir = Path(out_root) / name
            dataset.save(dataset_dir)
            if save_plots:
                _save_power_curve_plots(dataset, dataset_dir, profile=name)
        datasets[name] = dataset
    return datasets


def _save_power_curve_plots(dataset: SyntheticDataset, dataset_dir: Path, *, profile: str) -> None:
    """Write an original-vs-synthetic power-curve PNG for each test turbine."""
    # Imported lazily so the driver imports without matplotlib when plots aren't wanted.
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from benchmarking.synthetic.plots import plot_power_curve_comparison  # noqa: PLC0415

    for wtg in dataset.run_metadata.get("test_wtgs", []):
        uplift = dataset.true_uplift(test_wtg=wtg).overall
        fig = plot_power_curve_comparison(
            dataset.synthetic_df,
            dataset.original_df,
            test_wtg=wtg,
            save_path=dataset_dir / f"power_curve_{wtg}.png",
            title=f"{profile}: {wtg} power curve (true uplift {uplift:+.2%})",
        )
        plt.close(fig)


def main(
    *,
    out_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    start_dt: pd.Timestamp = DEFAULT_START_DT,
    end_dt_excl: pd.Timestamp = DEFAULT_END_DT_EXCL,
    test_wtg: str = DEFAULT_TEST_WTG,
    seed: int = 0,
) -> dict[str, SyntheticDataset]:
    """Produce one synthetic dataset per Issue 1 profile from real Hill of Towie data.

    Downloads (and caches) the open Hill of Towie SCADA for a stable, no-upgrade window,
    injects each example upgrade into ``test_wtg`` as a ``prepost`` changeover at the
    middle of the window, and saves the datasets under ``out_root``.

    :param out_root: dataset output root; defaults to :func:`default_output_root`
    :param data_dir: Hill of Towie data/cache dir; defaults to the package default
    :param start_dt: inclusive UTC window start
    :param end_dt_excl: exclusive UTC window end
    :param test_wtg: turbine to upgrade
    :param seed: top-level seed for reproducibility
    :return: mapping of profile name to its SyntheticDataset
    """
    out_root = Path(out_root) if out_root is not None else default_output_root()
    scada_df, _metadata_df = load_hot_scada(
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        data_dir=Path(data_dir) if data_dir is not None else None,
    )
    upgrade_timing = start_dt + (end_dt_excl - start_dt) / 2
    logger.info(
        "Generating example datasets for %s over %s..%s (changeover %s) into %s",
        test_wtg,
        start_dt,
        end_dt_excl,
        upgrade_timing,
        out_root,
    )
    return generate_example_datasets(
        scada_df=scada_df,
        test_wtgs=[test_wtg],
        mode="prepost",
        upgrade_timing=upgrade_timing,
        out_root=out_root,
        seed=seed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
