"""Tests for the Greenbyte source adapter (Kelmarsh and Penmanshiel).

The published zips are hundreds of megabytes, so these build the export layout instead: nine
comment lines, a ``#``-prefixed header, then rows. What is worth pinning is the parsing contract
and the two source quirks the adapter exists to absorb -- availability published as a fraction,
and a trailing blank row in Penmanshiel's metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from zipfile import ZipFile

import pandas as pd
import pytest

from benchmarking.synthetic.sources.greenbyte import (
    AVAILABILITY,
    GREENBYTE_COLUMNS,
    KELMARSH,
    NACELLE_POSITION,
    PENMANSHIEL,
    POWER,
    TIMEBASE_S,
    TURBINE,
    load_greenbyte_metadata,
    load_greenbyte_scada,
)

if TYPE_CHECKING:
    from pathlib import Path

_PREAMBLE = "\n".join(f"# comment line {i}" for i in range(9))
_HEADER = (
    "# Date and time,Wind speed (m/s),"
    '"Wind speed, Standard deviation (m/s)",Power (kW),'
    "Nacelle position (°),Time-based System Avail.,Generator RPM (RPM)"
)


def _turbine_csv(*, rows: int, start: str, power: float, nacelle: float, availability: float) -> str:
    """One turbine's data file in the published layout."""
    index = pd.date_range(start=start, periods=rows, freq=f"{TIMEBASE_S}s")
    body = "\n".join(f"{ts:%Y-%m-%d %H:%M:%S},8.0,0.5,{power},{nacelle},{availability},1500.0" for ts in index)
    return f"{_PREAMBLE}\n{_HEADER}\n{body}\n"


def _write_zip(path: Path, *, farm: str, turbines: range, year: int, rows: int = 6, availability: float = 1.0) -> None:
    """A published SCADA zip: one data file and one status file per turbine."""
    with ZipFile(path, "w") as archive:
        for number in turbines:
            stem = f"{farm}_{number}_{year}-01-01_-_{year + 1}-01-01_{200 + number}"
            archive.writestr(
                f"Turbine_Data_{stem}.csv",
                _turbine_csv(
                    rows=rows,
                    start=f"{year}-01-01",
                    power=100.0 * number,
                    nacelle=10.0 * number,
                    availability=availability,
                ),
            )
            # status files sit alongside the data and must be ignored
            archive.writestr(f"Status_{stem}.csv", "irrelevant\n")


@pytest.fixture
def kelmarsh_dir(tmp_path: Path) -> Path:
    _write_zip(tmp_path / "Kelmarsh_SCADA_2017.zip", farm="Kelmarsh", turbines=range(1, 4), year=2017)
    return tmp_path


class TestLoadScada:
    def test_reads_every_turbine_past_the_comment_preamble(self, kelmarsh_dir: Path) -> None:
        scada = load_greenbyte_scada(KELMARSH, years=[2017], data_dir=kelmarsh_dir)
        assert sorted(scada[TURBINE].unique()) == ["T01", "T02", "T03"]
        assert len(scada) == 18  # 3 turbines x 6 rows
        assert not scada.columns.str.startswith("#").any()

    def test_turbine_numbers_are_zero_padded_to_match_the_metadata(self, kelmarsh_dir: Path) -> None:
        scada = load_greenbyte_scada(KELMARSH, years=[2017], data_dir=kelmarsh_dir)
        assert "T03" in set(scada[TURBINE])
        assert "T3" not in set(scada[TURBINE])

    def test_the_index_is_utc_timestamps(self, kelmarsh_dir: Path) -> None:
        scada = load_greenbyte_scada(KELMARSH, years=[2017], data_dir=kelmarsh_dir)
        assert isinstance(scada.index, pd.DatetimeIndex)
        assert str(scada.index.tz) == "UTC"
        assert scada.index.min() == pd.Timestamp("2017-01-01", tz="UTC")

    def test_availability_is_converted_from_a_fraction_to_seconds(self, tmp_path: Path) -> None:
        """Greenbyte publishes a fraction of the period; the rest of the layer expects seconds."""
        _write_zip(
            tmp_path / "Kelmarsh_SCADA_2017.zip",
            farm="Kelmarsh",
            turbines=range(1, 2),
            year=2017,
            availability=0.5,
        )
        scada = load_greenbyte_scada(KELMARSH, years=[2017], data_dir=tmp_path)
        assert (scada[AVAILABILITY] == TIMEBASE_S * 0.5).all()
        assert "Time-based System Avail." not in scada.columns

    def test_status_files_are_ignored(self, kelmarsh_dir: Path) -> None:
        scada = load_greenbyte_scada(KELMARSH, years=[2017], data_dir=kelmarsh_dir)
        assert scada[POWER].notna().all()

    def test_several_years_are_concatenated_in_time_order(self, kelmarsh_dir: Path) -> None:
        _write_zip(kelmarsh_dir / "Kelmarsh_SCADA_2018.zip", farm="Kelmarsh", turbines=range(1, 4), year=2018)
        scada = load_greenbyte_scada(KELMARSH, years=[2017, 2018], data_dir=kelmarsh_dir)
        assert scada.index.is_monotonic_increasing
        assert scada.index.min().year == 2017
        assert scada.index.max().year == 2018

    def test_a_year_split_across_two_zips_is_read_whole(self, tmp_path: Path) -> None:
        """Penmanshiel publishes WT01-10 and WT11-15 separately; both belong to one year."""
        _write_zip(
            tmp_path / "Penmanshiel_SCADA_2017_WT01-10.zip", farm="Penmanshiel", turbines=range(1, 11), year=2017
        )
        _write_zip(
            tmp_path / "Penmanshiel_SCADA_2017_WT11-15.zip", farm="Penmanshiel", turbines=range(11, 16), year=2017
        )
        scada = load_greenbyte_scada(PENMANSHIEL, years=[2017], data_dir=tmp_path)
        assert scada[TURBINE].nunique() == 15

    def test_the_columns_the_schema_names_are_present(self, kelmarsh_dir: Path) -> None:
        scada = load_greenbyte_scada(KELMARSH, years=[2017], data_dir=kelmarsh_dir)
        for role in ("turbine", "active_power", "wind_speed", "wind_speed_sd", "availability", "nacelle_position"):
            assert getattr(GREENBYTE_COLUMNS, role) in scada.columns, role

    def test_values_land_in_the_right_columns(self, kelmarsh_dir: Path) -> None:
        scada = load_greenbyte_scada(KELMARSH, years=[2017], data_dir=kelmarsh_dir)
        t02 = scada[scada[TURBINE] == "T02"]
        assert (t02[POWER] == 200.0).all()
        assert (t02[NACELLE_POSITION] == 20.0).all()


class TestMissingData:
    def test_an_unpublished_year_raises_naming_what_is_available(self, kelmarsh_dir: Path) -> None:
        with pytest.raises(ValueError, match="no published SCADA for 2030"):
            load_greenbyte_scada(KELMARSH, years=[2030], data_dir=kelmarsh_dir)

    def test_a_missing_download_raises_pointing_at_the_zenodo_record(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=KELMARSH.record):
            load_greenbyte_scada(KELMARSH, years=[2017], data_dir=tmp_path)


class TestMetadata:
    @staticmethod
    def _write_static(path: Path, *, farm: str, rows: int, trailing_blank: bool) -> None:
        header = (
            "Wind Farm,Title,Alternative Title,Identity,Manufacturer,Model,Rated power (kW),"
            "Hub Height (m),Rotor Diameter (m),Latitude,Longitude,Elevation (m),Country,"
            "Commercial Operations Date"
        )
        lines = [header]
        lines.extend(
            f"{farm},{farm} {n},T{n:02d},X,Senvion,MM92,2050,78.5,92,5{n}.1,-0.9{n},145,UK,15/04/2016"
            for n in range(1, rows + 1)
        )
        if trailing_blank:
            lines.append(",,,,,,,,,,,,,")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    def test_names_are_normalised_to_match_the_scada(self, tmp_path: Path) -> None:
        self._write_static(tmp_path / KELMARSH.static_file, farm="Kelmarsh", rows=6, trailing_blank=False)
        metadata = load_greenbyte_metadata(KELMARSH, data_dir=tmp_path)
        assert list(metadata["Name"]) == [f"T{n:02d}" for n in range(1, 7)]
        assert metadata["Latitude"].dtype == float

    def test_a_trailing_blank_row_is_dropped(self, tmp_path: Path) -> None:
        """Penmanshiel's published CSV ends with an all-empty row."""
        self._write_static(tmp_path / PENMANSHIEL.static_file, farm="Penmanshiel", rows=14, trailing_blank=True)
        metadata = load_greenbyte_metadata(PENMANSHIEL, data_dir=tmp_path)
        assert len(metadata) == 14
        assert metadata["Name"].is_unique


class TestFarmDefinitions:
    @pytest.mark.parametrize("farm", [KELMARSH, PENMANSHIEL])
    def test_the_published_years_are_declared(self, farm: object) -> None:
        assert farm.years == tuple(range(2016, 2022))
        assert farm.static_file.endswith("_WT_static.csv")
        assert farm.record.isdigit()
