"""Tests for wind_up.era5 (Open-Meteo ERA5 fetch + cache).

Ported from the hill-of-towie-open-source-analysis ``test_era5_helpers.py``; adapted to
wind_up's module path and the wind_up cache-dir resolver. All offline (mocked); the real
network fetch is exercised only by callers behind a ``slow`` marker.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from wind_up_v0 import era5
from wind_up_v0.era5 import _build_era5_df


def _make_mock_response(n_hours: int = 3) -> MagicMock:
    def _make_var(val: float) -> MagicMock:
        mock_var = MagicMock()
        mock_var.ValuesAsNumpy.return_value = np.full(n_hours, val, dtype="float32")
        return mock_var

    mock_hourly = MagicMock()
    mock_hourly.Time.return_value = 1704067200  # 2024-01-01 00:00 UTC
    mock_hourly.TimeEnd.return_value = 1704067200 + 3600 * n_hours
    mock_hourly.Interval.return_value = 3600
    mock_hourly.Variables.side_effect = lambda i: _make_var(float(i))

    mock_response = MagicMock()
    mock_response.Hourly.return_value = mock_hourly
    return mock_response


class TestBuildEra5Df:
    def test_columns_match_fields(self) -> None:
        fields = ["wind_speed_10m", "wind_direction_10m"]
        df = _build_era5_df(_make_mock_response(), fields)
        assert list(df.columns) == ["wind_speed_10m", "wind_direction_10m"]

    def test_row_count_matches_hours(self) -> None:
        fields = ["wind_speed_10m"]
        df = _build_era5_df(_make_mock_response(n_hours=5), fields)
        assert len(df) == 5

    def test_index_is_utc_datetimeindex(self) -> None:
        fields = ["wind_speed_10m"]
        df = _build_era5_df(_make_mock_response(), fields)
        assert df.index.dtype == "datetime64[ns, UTC]"

    def test_timestamp_starts_at_expected_value(self) -> None:
        fields = ["wind_speed_10m"]
        df = _build_era5_df(_make_mock_response(), fields)
        assert df.index[0] == pd.Timestamp("2024-01-01", tz="UTC")

    def test_field_values_are_propagated(self) -> None:
        fields = ["wind_speed_10m", "wind_direction_10m"]
        df = _build_era5_df(_make_mock_response(), fields)
        assert df["wind_speed_10m"].iloc[0] == pytest.approx(0.0)  # index 0
        assert df["wind_direction_10m"].iloc[0] == pytest.approx(1.0)  # index 1


class TestGetEra5HourlyDf:
    def test_end_date_defaults_to_today(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_cache_path(*args: object, **_kwargs: object) -> MagicMock:
            # positional signature mirrors _era5_cache_path(lat, lon, start_date, end_date, fields)
            captured["end_date"] = args[3]
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            return mock_path

        monkeypatch.setattr(era5, "_era5_cache_path", fake_cache_path)
        monkeypatch.setattr(era5.pd, "read_parquet", lambda _p: pd.DataFrame())
        era5.get_era5_hourly_df(lat=1.0, lon=2.0)
        today = pd.Timestamp.now(tz="UTC").normalize().strftime("%Y-%m-%d")
        assert captured["end_date"] == today

    def test_reads_from_cache_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cached = pd.DataFrame({"wind_speed_100m": [1.0, 2.0]})

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        monkeypatch.setattr(era5, "_era5_cache_path", lambda *_a, **_k: mock_path)
        monkeypatch.setattr(era5.pd, "read_parquet", lambda _p: cached)

        out = era5.get_era5_hourly_df(lat=1.0, lon=2.0, start_date="2020-01-01", end_date="2020-01-02")
        pd.testing.assert_frame_equal(out, cached)


class TestCacheDir:
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("WIND_UP_CACHE_DIR", str(tmp_path))
        assert era5._resolve_cache_dir(None) == tmp_path  # noqa: SLF001

    def test_explicit_arg_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("WIND_UP_CACHE_DIR", str(tmp_path / "env"))
        explicit = tmp_path / "explicit"
        assert era5._resolve_cache_dir(explicit) == explicit  # noqa: SLF001

    def test_cache_path_is_deterministic(self, tmp_path: Path) -> None:
        args = {
            "lat": 57.5,
            "lon": -3.25,
            "start_date": "2020-01-01",
            "end_date": "2020-02-01",
            "fields": ["wind_speed_100m", "wind_direction_100m"],
        }
        p1 = era5._era5_cache_path(**args, cache_dir=tmp_path)  # noqa: SLF001
        p2 = era5._era5_cache_path(**args, cache_dir=tmp_path)  # noqa: SLF001
        assert p1 == p2
        assert p1.suffix == ".parquet"
