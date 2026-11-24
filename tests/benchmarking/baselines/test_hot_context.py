"""Offline tests for the vendored HoT assets and the v0 source-context builder."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

from benchmarking import env
from benchmarking.baselines import hot_context
from benchmarking.baselines.hot_context import ASSET_YAML, NORTHING_YAML, HotV0Context, build_hot_v0_context
from wind_up_v0.yaml_loader import Loader, construct_include


def _load_yaml_with_includes(path):  # noqa: ANN001, ANN202
    yaml.add_constructor("!include", construct_include, Loader)
    with path.open() as f:
        return yaml.load(f, Loader)  # noqa: S506


class TestVendoredAssets:
    def test_asset_yaml_loads_with_turbine_type_include(self) -> None:
        asset = _load_yaml_with_includes(ASSET_YAML)
        assert asset["name"] == "Hill of Towie"
        assert asset["wtgs"] == [f"T{i:02d}" for i in range(1, 22)]
        assert asset["turbine_types"][0]["turbine_type"] == "SWT-2.3-82"
        assert asset["turbine_types"][0]["rated_power_kw"] == 2300

    def test_northing_yaml_covers_all_turbines(self) -> None:
        with NORTHING_YAML.open() as f:
            corrections = yaml.safe_load(f)
        assert isinstance(corrections, list)
        name, when, value = corrections[0]
        assert name == "T01"
        assert isinstance(when, dt.datetime)
        assert isinstance(value, float)
        assert {row[0] for row in corrections} == {f"T{i:02d}" for i in range(1, 22)}


class TestGetHotReanalysisDatasets:
    def test_wraps_era5_df_for_hot_site(self, monkeypatch) -> None:  # noqa: ANN001
        captured: dict[str, object] = {}
        era5_df = pd.DataFrame({"wind_speed_100m": [1.0, 2.0]})

        def fake_get_era5(**kwargs):  # noqa: ANN003, ANN202
            captured.update(kwargs)
            return era5_df

        monkeypatch.setattr(hot_context, "get_era5_hourly_df", fake_get_era5)

        datasets = hot_context.get_hot_reanalysis_datasets()

        assert len(datasets) == 1
        assert datasets[0].id == f"ERA5_{hot_context.HOT_LAT:.2f}_{hot_context.HOT_LON:.2f}"
        pd.testing.assert_frame_equal(datasets[0].data, era5_df)
        assert captured == {
            "lat": hot_context.HOT_LAT,
            "lon": hot_context.HOT_LON,
            "start_date": hot_context.HOT_ERA5_START,
            "end_date": hot_context.HOT_ERA5_END,
            "cache_dir": env.cache_dir(),
        }


class TestBuildHotV0Context:
    def test_wires_metadata_and_reanalysis_loaders(self, monkeypatch) -> None:  # noqa: ANN001
        metadata = pd.DataFrame({"Name": ["T01"], "Latitude": [57.5], "Longitude": [-3.25]})
        sentinel_reanalysis = [object()]
        captured: dict[str, object] = {}

        def fake_metadata(*, data_dir=None, wtg_names=None):  # noqa: ANN001, ANN202
            captured["data_dir"] = data_dir
            captured["wtg_names"] = wtg_names
            return metadata

        monkeypatch.setattr(hot_context, "load_hot_metadata", fake_metadata)
        monkeypatch.setattr(hot_context, "get_hot_reanalysis_datasets", lambda: sentinel_reanalysis)

        ctx = build_hot_v0_context(data_dir=Path("/some/dir"))

        assert isinstance(ctx, HotV0Context)
        pd.testing.assert_frame_equal(ctx.metadata_df, metadata)
        assert ctx.reanalysis_datasets is sentinel_reanalysis
        assert ctx.asset_yaml == ASSET_YAML
        assert ctx.northing_yaml == NORTHING_YAML
        assert captured["data_dir"] == Path("/some/dir")


class TestV0Facts:
    def test_the_northing_table_becomes_north_offsets(self) -> None:
        offsets = hot_context.hot_north_offsets()
        assert offsets[0][0] == "T01"
        assert offsets[0][1].tz is not None
        assert isinstance(offsets[0][2], float)

    def test_the_reanalysis_sits_at_the_sites_node(self) -> None:
        context = HotV0Context(
            metadata_df=pd.DataFrame(),
            reanalysis_datasets=[hot_context.ReanalysisDataset(id="node", data=pd.DataFrame({"x": [1.0]}))],
        )
        (node,) = hot_context.hot_reanalysis(context)
        assert (node.name, node.latitude, node.longitude) == ("node", hot_context.HOT_LAT, hot_context.HOT_LON)

    def test_the_v0_method_reads_source_native_records_with_the_vendored_type(self) -> None:
        method = hot_context.hot_v0_method(options={"ws_bin_width": 0.5}, save_plots=True)
        assert method.options["ws_bin_width"] == 0.5
        assert method.options["years_for_detrend"] == 1
        assert method.turbine_type["normal_operation_pitch_range"] == (-10, 40)
        assert method.to_v0_records is not None
        assert method.save_plots
