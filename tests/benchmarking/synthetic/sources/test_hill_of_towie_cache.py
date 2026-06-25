"""Tests for the per-(year, turbine) parquet cache in the Hill of Towie loader.

The Hill of Towie Zenodo record is fixed as one zip per year, so the loader caches each
*turbine-year* once and reuses it for any window or turbine subset. These tests fabricate
minimal year zips (so they run offline, no Zenodo) and check that:

- one parquet is written per (year, requested turbine), and only for the requested turbines;
- a repeat call reads the cache and does not re-unpack;
- growing the turbine subset only unpacks the newly requested turbines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd

from benchmarking.synthetic.sources import hill_of_towie as hot
from wind_up.constants import DataColumns

if TYPE_CHECKING:
    from pathlib import Path

_APM_FIELD = hot.WPSBackupFileField(
    alias=DataColumns.active_power_mean, field_name="wtc_ActPower_mean", table_name="tblSCTurGrid"
)
_WINDOW = {
    "start_dt": pd.Timestamp("2020-01-01", tz="UTC"),
    "end_dt_excl": pd.Timestamp("2020-03-01", tz="UTC"),
}


def _month_csv(*, year: int, month: int, serials: list[int]) -> str:
    """An end-format SCADA CSV for one month of one table, a few timestamps per turbine."""
    end_times = pd.date_range(f"{year}-{month:02d}-01 00:10", periods=4, freq="10min")
    rows = [
        {"TimeStamp": ts, "StationId": s, "wtc_ActPower_mean": 100.0 + i + s}
        for i, ts in enumerate(end_times)
        for s in serials
    ]
    return pd.DataFrame(rows).to_csv(index=False)


def _write_year_zip(*, data_dir: Path, year: int, months: list[int], serials: list[int]) -> None:
    """Write a fabricated ``{year}.zip`` holding one ``tblSCTurGrid`` CSV per month."""
    with ZipFile(data_dir / f"{year}.zip", "w") as zf:
        for m in months:
            zf.writestr(f"tblSCTurGrid_{year}_{m:02d}.csv", _month_csv(year=year, month=m, serials=serials))


def _load(*, data_dir: Path, cache_dir: Path, wtg_numbers: list[int]) -> pd.DataFrame:
    return hot.load_hot_10min_data(
        data_dir=data_dir,
        wtg_numbers=wtg_numbers,
        custom_fields=[_APM_FIELD],
        cache_dir=cache_dir,
        **_WINDOW,
    )


def _dirs(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()
    serials = [hot._HOT_SERIAL_OFFSET + n for n in (1, 3)]  # noqa: SLF001
    _write_year_zip(data_dir=data_dir, year=2020, months=[1, 2], serials=serials)
    return data_dir, cache_dir


def test_writes_one_cache_file_per_requested_turbine(tmp_path: Path) -> None:
    """Loading only T01 caches T01's year and leaves the other turbine's year un-cached."""
    data_dir, cache_dir = _dirs(tmp_path)

    wide = _load(data_dir=data_dir, cache_dir=cache_dir, wtg_numbers=[1])

    assert list(cache_dir.glob("hot10min_2020_T01_*.parquet"))
    assert not list(cache_dir.glob("hot10min_2020_T03_*.parquet"))
    assert {col[0] for col in wide.columns} == {"T01"}
    # Columns keep their source-native tag names (no v0 aliasing at load time).
    assert (("T01", _APM_FIELD.field_name)) in wide.columns


def test_repeat_call_reuses_cache_without_unpacking(tmp_path: Path) -> None:
    """A second identical call serves the parquet cache and never re-reads the zip."""
    data_dir, cache_dir = _dirs(tmp_path)
    _load(data_dir=data_dir, cache_dir=cache_dir, wtg_numbers=[1])

    # wraps= keeps the real unpack behaviour; the spy only records whether it was called.
    with patch.object(hot, "_unpack_hot_10min_year", wraps=hot._unpack_hot_10min_year) as spy:  # noqa: SLF001
        _load(data_dir=data_dir, cache_dir=cache_dir, wtg_numbers=[1])

    assert spy.call_count == 0


def test_growing_subset_only_unpacks_the_new_turbine(tmp_path: Path) -> None:
    """Adding T03 to an existing T01 cache unpacks only T03 and reuses T01."""
    data_dir, cache_dir = _dirs(tmp_path)
    _load(data_dir=data_dir, cache_dir=cache_dir, wtg_numbers=[1])

    with patch.object(hot, "_unpack_hot_10min_year", wraps=hot._unpack_hot_10min_year) as spy:  # noqa: SLF001
        wide = _load(data_dir=data_dir, cache_dir=cache_dir, wtg_numbers=[1, 3])

    spy.assert_called_once()
    assert list(spy.call_args.kwargs["serial_numbers"]) == [hot._HOT_SERIAL_OFFSET + 3]  # noqa: SLF001  only the new
    assert list(cache_dir.glob("hot10min_2020_T01_*.parquet"))
    assert list(cache_dir.glob("hot10min_2020_T03_*.parquet"))
    assert {col[0] for col in wide.columns} == {"T01", "T03"}
