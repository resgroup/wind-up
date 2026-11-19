"""Shared Hill of Towie source-context: metadata, ERA5 reanalysis and v0's settings for the site.

``build_hot_v0_context`` loads the per-turbine metadata and the ERA5 reanalysis once.
:func:`hot_reanalysis` and :func:`hot_north_offsets` give the reanalysis node and the vendored
northing table as facts, and :func:`hot_v0_method` gives v0_binned with the vendored turbine type
and the settings the harness benchmarks v0 with.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

from benchmarking import env
from benchmarking.baselines.v0_binned import V0BinnedMethod
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, long_to_wind_up_format
from wind_up.facts import Reanalysis
from wind_up_v0.era5 import get_era5_hourly_df
from wind_up_v0.reanalysis_data import ReanalysisDataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from wind_up.facts import NorthOffset

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
            data=get_era5_hourly_df(
                lat=HOT_LAT, lon=HOT_LON, start_date=HOT_ERA5_START, end_date=HOT_ERA5_END, cache_dir=env.cache_dir()
            ),
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


# The v0 settings the harness benchmarks v0 with: a pre period a year before, a year of long-term
# and detrend data, and a toggle detrended on its off records and not paired.
HOT_V0_OPTIONS: dict = {
    "years_offset_for_pre_period": 1,
    "years_for_lt_distribution": 1,
    "years_for_detrend": 1,
    "ws_bin_width": 1.0,
    "reanalysis_method": "node_with_best_ws_corr",
    "optimize_northing_corrections": False,
    "filter_all_test_wtgs_together": False,
    "require_ref_wake_free": False,
    "toggle": {
        "detrend_data_selection": "use_toggle_off_data",
        "pairing_filter_method": "none",
        "pairing_filter_timedelta_seconds": 3000,
    },
}


def hot_turbine_type() -> dict:
    """Return the vendored turbine type's fields beyond rotor diameter and rated power."""
    fields = yaml.safe_load((ASSETS_DIR / "turbine_type" / "SWT_2p3_82.yaml").read_text())
    return {
        k: (tuple(v) if isinstance(v, list) else v)
        for k, v in fields.items()
        if k not in ("rotor_diameter_m", "rated_power_kw")
    }


def hot_north_offsets() -> list[NorthOffset]:
    """Return the vendored northing table as the facts' north offsets."""
    rows = yaml.safe_load(NORTHING_YAML.read_text())
    return [(str(who), pd.Timestamp(when, tz="UTC"), float(offset)) for who, when, offset in rows]


def hot_reanalysis(context: HotV0Context) -> list[Reanalysis]:
    """Return the context's reanalysis datasets as the facts' reanalysis nodes, at the site's ERA5 node."""
    return [
        Reanalysis(name=d.id, latitude=HOT_LAT, longitude=HOT_LON, records=d.data) for d in context.reanalysis_datasets
    ]


def hot_v0_method(**overrides: object) -> V0BinnedMethod:
    """Return v0_binned with the vendored turbine type and :data:`HOT_V0_OPTIONS`, reading source-native records.

    :param overrides: the method's fields to set otherwise, ``options`` merged over the defaults
    """
    options = {**HOT_V0_OPTIONS, **overrides.pop("options", {})}  # type: ignore[dict-item]
    fields: dict = {
        "options": options,
        "turbine_type": hot_turbine_type(),
        "site_name": "Hill of Towie",
        "auto_choose_refs": False,
        "to_v0_records": long_to_wind_up_format,
        **overrides,
    }
    return V0BinnedMethod(**fields)
