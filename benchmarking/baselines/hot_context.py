"""Shared Hill of Towie source-context for v0-style assessment methods.

``build_hot_v0_context`` assembles the source-specific inputs a wind_up assessment needs but
the harness's thin ``MethodInput`` does not carry: per-turbine metadata (lat/long), ERA5
reanalysis, and the paths to the vendored asset and northing-corrections YAMLs. It is loaded
once and reused across every campaign a method scores.

This is a benchmarking helper, not a harness-enforced contract: future methods (e.g. the
Issue 5 R-learner) reuse it by calling it from their own constructor. Keeping it here lets the
harness seam stay thin until the Issue 4 contract has two real consumers to design against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata
from wind_up.era5 import get_hot_reanalysis_datasets

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from wind_up.reanalysis_data import ReanalysisDataset

ASSETS_DIR = Path(__file__).parent / "assets"
ASSET_YAML = ASSETS_DIR / "HOT.yaml"
NORTHING_YAML = ASSETS_DIR / "optimized_northing_corrections.yaml"


@dataclass
class HotV0Context:
    """Source-specific inputs a v0-style assessment needs, loaded once and reused.

    :param metadata_df: per-turbine metadata (Name, Latitude, Longitude) in wind-up format
    :param reanalysis_datasets: ERA5 reanalysis datasets for the HoT site
    :param asset_yaml: path to the vendored asset YAML (turbine list + type)
    :param northing_yaml: path to the vendored optimized northing-corrections YAML
    """

    metadata_df: pd.DataFrame
    reanalysis_datasets: list[ReanalysisDataset]
    asset_yaml: Path = ASSET_YAML
    northing_yaml: Path = NORTHING_YAML


def build_hot_v0_context(*, data_dir: str | Path | None = None, wtg_names: Sequence[str] | None = None) -> HotV0Context:
    """Assemble the HoT v0 context: load metadata and (fetch+cache) ERA5 reanalysis once.

    :param data_dir: Hill of Towie data/cache dir; defaults to the source package default
    """
    metadata_df = load_hot_metadata(data_dir=Path(data_dir) if data_dir is not None else None, wtg_names=wtg_names)
    reanalysis_datasets = get_hot_reanalysis_datasets()
    return HotV0Context(metadata_df=metadata_df, reanalysis_datasets=reanalysis_datasets)
