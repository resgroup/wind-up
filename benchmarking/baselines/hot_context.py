"""Shared Hill of Towie source-context for v0-style assessment methods.

``build_hot_v0_context`` assembles the source-specific inputs a wind_up assessment needs but
the harness's thin ``MethodInput`` does not carry: per-turbine metadata (lat/long), ERA5
reanalysis, and the paths to the vendored asset and northing-corrections YAMLs. It is loaded
once and reused across every campaign a method scores.

This is a benchmarking helper, not a harness-enforced contract: methods reuse it by calling it
from their own constructor, keeping the harness seam thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata
from wind_up.era5 import get_era5_hourly_df
from wind_up.reanalysis_data import ReanalysisDataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

ASSETS_DIR = Path(__file__).parent / "assets"
ASSET_YAML = ASSETS_DIR / "HOT.yaml"
NORTHING_YAML = ASSETS_DIR / "optimized_northing_corrections.yaml"

HOT_LAT: float = 57.50
HOT_LON: float = -3.25
HOT_ERA5_START: str = "2000-01-01"
HOT_ERA5_END: str = "2026-05-01"


def get_hot_reanalysis_datasets() -> list[ReanalysisDataset]:
    """Return a list with one :class:`ReanalysisDataset` for the Hill of Towie site."""
    return [
        ReanalysisDataset(
            id=f"ERA5_{HOT_LAT:.2f}_{HOT_LON:.2f}",
            data=get_era5_hourly_df(lat=HOT_LAT, lon=HOT_LON, start_date=HOT_ERA5_START, end_date=HOT_ERA5_END),
        )
    ]


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
