"""Greenbyte-exported open SCADA: the Kelmarsh and Penmanshiel wind farms.

Two open datasets published on Zenodo by Cubico, exported from Greenbyte in a shared CSV layout
(nine comment lines, then a ``#``-prefixed header). Both are simpler than Hill of Towie -- six
and fourteen Senvion turbines against HoT's twenty-one -- which makes them a useful second and
third site for anything that must not be tuned to one farm.

The adapter returns the same long, source-native shape the rest of the benchmarking layer speaks,
with one normalisation: Greenbyte reports availability as a **fraction** of the period, so it is
converted to seconds here to match :data:`~benchmarking.synthetic.sources.hill_of_towie.HOT_COLUMNS`.
Source-specific knowledge belongs in the source adapter, not in the methods.

Datasets:

* Kelmarsh -- https://zenodo.org/records/5841834
* Penmanshiel -- https://zenodo.org/records/5946808
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

import pandas as pd

from benchmarking.synthetic.schema import ColumnSchema

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

TIMEBASE_S = 600
# Greenbyte writes nine comment lines before the header, which is itself commented.
_HEADER_ROW = 9
_TIMESTAMP = "# Date and time"

# The source-native names this adapter keeps. Everything else in the 299-column export is dropped.
POWER = "Power (kW)"
NACELLE_POSITION = "Nacelle position (°)"
WIND_SPEED = "Wind speed (m/s)"
WIND_SPEED_SD = "Wind speed, Standard deviation (m/s)"
GEN_RPM = "Generator RPM (RPM)"
AVAILABILITY = "availability_s"
TURBINE = "TurbineName"

GREENBYTE_COLUMNS = ColumnSchema(
    turbine=TURBINE,
    active_power=POWER,
    wind_speed=WIND_SPEED,
    wind_speed_sd=WIND_SPEED_SD,
    gen_rpm=GEN_RPM,
    availability=AVAILABILITY,
    nacelle_position=NACELLE_POSITION,
)


@dataclass(frozen=True)
class GreenbyteFarm:
    """A Zenodo-published Greenbyte export.

    :param name: short name; also the prefix of the published files and the plot title
    :param record: the Zenodo record id, so an error can say where to fetch the data
    :param years: the calendar years published for this farm
    :param rated_power_kw: the turbines' rated power
    """

    name: str
    record: str
    years: tuple[int, ...]
    rated_power_kw: float

    @property
    def static_file(self) -> str:
        """The per-turbine metadata CSV published alongside the SCADA."""
        return f"{self.name}_WT_static.csv"


# Both farms publish 2016-2021. Penmanshiel splits each year across two zips (WT01-10, WT11-15);
# the loader globs rather than naming files, so either layout works and so do the shorter names a
# manual download tends to leave behind.
KELMARSH = GreenbyteFarm(name="Kelmarsh", record="5841834", years=tuple(range(2016, 2022)), rated_power_kw=2050.0)
PENMANSHIEL = GreenbyteFarm(name="Penmanshiel", record="5946808", years=tuple(range(2016, 2022)), rated_power_kw=2050.0)

FARMS = {farm.name.lower(): farm for farm in (KELMARSH, PENMANSHIEL)}


def get_data_dir() -> Path:
    """Return the local cache directory for these datasets, creating it if needed."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_DATA_DIR", Path.home() / "temp" / "wind-up-benchmarking" / "data"))
    path = root / "zenodo"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _turbine_name(member: str) -> str:
    """``Turbine_Data_Kelmarsh_3_2017-...csv`` -> ``T03``; the farm's own numbering, zero-padded."""
    match = re.search(r"Turbine_Data_[A-Za-z]+_(\d+)_", member)
    if match is None:
        msg = f"cannot read a turbine number from {member!r}"
        raise ValueError(msg)
    return f"T{int(match.group(1)):02d}"


def load_greenbyte_metadata(farm: GreenbyteFarm, *, data_dir: Path | None = None) -> pd.DataFrame:
    """Return per-turbine ``Name``, ``Latitude`` and ``Longitude`` for ``farm``.

    Names are normalised to ``T01``-style so they match the SCADA frame.
    """
    path = (data_dir or get_data_dir()) / farm.static_file
    static = pd.read_csv(path, encoding="utf-8-sig")
    # Penmanshiel's CSV carries a trailing blank row, so rows without a turbine number are dropped
    numbers = static["Title"].astype(str).str.extract(r"(\d+)$")[0]
    keep = numbers.notna()
    return pd.DataFrame(
        {
            "Name": [f"T{int(n):02d}" for n in numbers[keep]],
            "Latitude": static.loc[keep, "Latitude"].astype(float).to_numpy(),
            "Longitude": static.loc[keep, "Longitude"].astype(float).to_numpy(),
        }
    )


def load_greenbyte_scada(
    farm: GreenbyteFarm,
    *,
    years: Sequence[int],
    data_dir: Path | None = None,
    columns: Sequence[str] = (POWER, NACELLE_POSITION, WIND_SPEED, WIND_SPEED_SD, GEN_RPM),
) -> pd.DataFrame:
    """Return long, timestamp-indexed SCADA for ``farm`` over ``years``.

    One row per turbine per 10-minute period, with :data:`TURBINE` naming the turbine and
    availability converted from Greenbyte's fraction to seconds.

    :param years: calendar years to load; each must be published for this farm
    :param data_dir: where the Zenodo zips are cached; defaults to :func:`get_data_dir`
    :param columns: the source-native value columns to keep besides availability
    :raises FileNotFoundError: if a year's zip has not been downloaded
    """
    directory = data_dir or get_data_dir()
    wanted = [_TIMESTAMP, *columns, "Time-based System Avail."]
    frames = []
    for year in years:
        if year not in farm.years:
            msg = f"{farm.name} has no published SCADA for {year}; have {list(farm.years)}"
            raise ValueError(msg)
        paths = sorted(directory.glob(f"{farm.name}*SCADA*{year}*.zip"))
        if not paths:
            msg = (
                f"no {farm.name} {year} SCADA zip in {directory}. Fetch it from "
                f"https://zenodo.org/records/{farm.record}."
            )
            raise FileNotFoundError(msg)
        for path in paths:
            frames.extend(_read_zip(path, wanted=wanted))
    scada = pd.concat(frames).sort_index()
    scada[AVAILABILITY] = scada.pop("Time-based System Avail.").astype(float) * TIMEBASE_S
    return scada


def _read_zip(path: Path, *, wanted: Sequence[str]) -> list[pd.DataFrame]:
    """Read every turbine's data file out of one published zip."""
    logger.info("reading %s", path.name)
    frames = []
    with ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            if not member.startswith("Turbine_Data"):
                continue
            raw = pd.read_csv(
                io.BytesIO(archive.read(member)),
                skiprows=_HEADER_ROW,
                usecols=lambda c, wanted=tuple(wanted): c in wanted,
                parse_dates=[_TIMESTAMP],
            )
            raw[TURBINE] = _turbine_name(member)
            frames.append(raw.set_index(_TIMESTAMP).tz_localize("UTC"))
    return frames
