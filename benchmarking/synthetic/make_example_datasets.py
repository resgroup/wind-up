"""Driver: produce one synthetic dataset per Issue 1 profile.

``generate_example_datasets`` is source-agnostic (give it any wind-up-format SCADA).
``main`` wires it to the Hill of Towie open data so the whole thing runs end-to-end.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping

from benchmarking.synthetic.generator import SyntheticDataset, ToggleSchedule, generate_dataset
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada
from benchmarking.synthetic.upgrades import (
    ConditionCpChange,
    ConstantCpChange,
    RatedPowerChange,
    WakeSteering,
    WindSpeedCpChange,
)

logger = logging.getLogger(__name__)

# A real Hill of Towie steering cluster (the campaign's south-west pairs T03->T07 and T02->T05).
# North offsets are the latest values from the published optimized_northing_corrections.yaml
# (resgroup/hill-of-towie-open-source-analysis), valid for the 2020 example window; they are step
# corrections added to the raw nacelle position (deg).
WAKE_STEERING_CLUSTER = ("T02", "T03", "T05", "T07")
WAKE_STEERING_NORTH_OFFSETS_DEG = {"T02": 8.57, "T03": -1.90, "T05": 13.18, "T07": -16.49}

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


def generate_wake_steering_example(
    *,
    scada_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    start_dt: pd.Timestamp,
    out_root: str | Path | None = None,
    north_offsets_deg: Mapping[str, float] = WAKE_STEERING_NORTH_OFFSETS_DEG,
    cluster: tuple[str, ...] | None = None,
    seed: int = 0,
    save_plots: bool = True,
) -> SyntheticDataset:
    """Generate (and optionally save) one synthetic wake-steering dataset for a real HoT cluster.

    A 50 min on / 50 min off toggle campaign is injected into ``cluster``; steering pairs are
    derived from the turbines' coordinates. When saving, writes the dataset plus a
    steer-angle/uplift-vs-direction plot for each derived pair.

    :param scada_df: wind-up-format real SCADA (all turbines), the no-upgrade baseline
    :param metadata_df: per-turbine metadata with Name, Latitude, Longitude
    :param start_dt: campaign/toggle origin (rows before it are untreated baseline)
    :param out_root: if given, the dataset and plots are saved under ``out_root / "wake_steering"``
    :param north_offsets_deg: per-turbine north offset (deg) for every participant
    :param cluster: the participating (steering/benefitting) turbines; defaults to the keys of
        ``north_offsets_deg``. Every cluster turbine must have a north offset.
    :param seed: recorded for provenance
    :param save_plots: also write the per-pair direction plots when saving
    """
    cluster = tuple(north_offsets_deg) if cluster is None else cluster
    missing = [w for w in cluster if w not in north_offsets_deg]
    if missing:
        msg = f"north_offsets_deg is missing offsets for cluster turbine(s) {missing}"
        raise ValueError(msg)
    # Pass every turbine's coordinates (not just the cluster) so wake-blocking can see any upwind
    # turbine; only the cluster participates in steering.
    coords = {str(row.Name): (float(row.Latitude), float(row.Longitude)) for row in metadata_df.itertuples()}
    north_offsets = [(wtg, start_dt, north_offsets_deg[wtg]) for wtg in cluster]
    steering = WakeSteering(coords=coords, test_wtgs=list(cluster), north_offsets=north_offsets)
    schedule = ToggleSchedule(period=pd.Timedelta(minutes=100), start=start_dt)
    dataset = generate_dataset(
        scada_df=scada_df,
        test_wtgs=list(cluster),
        upgrades=[steering],
        mode="toggle",
        upgrade_timing=schedule,
        seed=seed,
    )
    if out_root is not None:
        dataset_dir = Path(out_root) / "wake_steering"
        dataset.save(dataset_dir)
        if save_plots:
            _save_wake_steering_plots(dataset, dataset_dir, pairs=steering.pairs)
    return dataset


def _save_wake_steering_plots(dataset: SyntheticDataset, dataset_dir: Path, *, pairs: tuple) -> None:
    """Write a steer-angle/uplift-vs-direction PNG for each derived steering pair."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from benchmarking.synthetic.plots import plot_wake_steering_by_direction  # noqa: PLC0415

    for pair in pairs:
        net = dataset.true_net_uplift(upstream=pair.upstream, downstream=pair.downstream)
        fig = plot_wake_steering_by_direction(
            dataset,
            upstream=pair.upstream,
            downstream=pair.downstream,
            save_path=dataset_dir / f"steering_{pair.upstream}_to_{pair.downstream}.png",
            title=f"Wake steering {pair.upstream} -> {pair.downstream} (net {net:+.2%})",
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


def main_wake_steering(
    *,
    out_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    start_dt: pd.Timestamp = DEFAULT_START_DT,
    end_dt_excl: pd.Timestamp = DEFAULT_END_DT_EXCL,
    seed: int = 0,
) -> SyntheticDataset:
    """Produce one synthetic wake-steering dataset from real Hill of Towie data.

    Loads (and caches) the open Hill of Towie SCADA and metadata for a stable, no-upgrade window,
    then injects a 50/50 toggle wake-steering campaign into the south-west cluster with the toggle
    starting mid-window, and saves the dataset and per-pair plots under ``out_root``.
    """
    out_root = Path(out_root) if out_root is not None else default_output_root()
    resolved_data_dir = Path(data_dir) if data_dir is not None else None
    scada_df, _ = load_hot_scada(start_dt=start_dt, end_dt_excl=end_dt_excl, data_dir=resolved_data_dir)
    metadata_df = load_hot_metadata(data_dir=resolved_data_dir)
    toggle_start = start_dt + (end_dt_excl - start_dt) / 2
    logger.info("Generating wake-steering example over %s..%s into %s", start_dt, end_dt_excl, out_root)
    return generate_wake_steering_example(
        scada_df=scada_df, metadata_df=metadata_df, start_dt=toggle_start, out_root=out_root, seed=seed
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    main_wake_steering()
