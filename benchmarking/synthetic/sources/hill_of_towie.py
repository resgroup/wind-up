"""Hill of Towie open-data source adapter for the synthetic generator.

A self-contained (vendored) copy of the pieces of the
``hill-of-towie-open-source-analysis`` ``hot_open`` package needed to load
wind-up-format SCADA end to end:

- the Zenodo fetcher (``ensure_hot_data_files`` / ``download_zenodo_data``) that
  downloads and caches the Hill of Towie v2 datapack (Zenodo record ``20204946``);
- the 10-minute SCADA loader (``load_hot_10min_data``) and the wide-to-long reshape
  (``scada_wide_to_long``) that keeps source-native ``wtc_*`` tag names;
- ``load_hot_scada`` that ties them together and returns source-native long SCADA plus
  turbine metadata;
- ``long_to_wind_up_format``, the v0-only on-ramp that aliases the source columns to
  :class:`~wind_up.constants.DataColumns` names and derives ``PitchAngleMean`` /
  ``ShutdownDuration``.

Copied rather than imported so ``benchmarking`` stays hermetic and depends on
``wind_up`` only for :class:`~wind_up.constants.DataColumns` (used by the v0 on-ramp).
``requests`` and ``tqdm`` are imported lazily inside the network/IO functions so the pure
transforms import without them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from zipfile import ZipFile

import pandas as pd

from benchmarking.synthetic.schema import ColumnSchema
from wind_up.constants import DataColumns

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    import requests

logger = logging.getLogger(__name__)

TIMEBASE_S = 600
HOT_V2_RECORD_ID = "20204946"
HOT_FIRST_WTG = 1
HOT_LAST_WTG = 21
_HOT_SERIAL_OFFSET = 2304509

BYTES_IN_1MB = 1024 * 1024
CHUNK_SIZE = 10 * BYTES_IN_1MB
SMALL_FILE_THRESHOLD_BYTES = 2 * BYTES_IN_1MB

# Network resilience knobs for streamed Zenodo downloads. ``timeout`` is passed to
# ``requests.get`` as a ``(connect, read)`` tuple; with ``stream=True`` the read
# value is the budget *between* received chunks.
_CONNECT_TIMEOUT_S = 10
_READ_TIMEOUT_S = 60
_MAX_DOWNLOAD_ATTEMPTS = 5
_BACKOFF_BASE_S = 2.0
_HTTP_PARTIAL_CONTENT = 206


def get_data_dir() -> Path:
    """Return the local Hill of Towie data/cache directory, creating it if needed.

    Overridable via the ``WIND_UP_BENCHMARKING_DATA_DIR`` environment variable;
    defaults to ``~/temp/wind-up-benchmarking/data``.
    """
    path = Path(os.getenv("WIND_UP_BENCHMARKING_DATA_DIR", Path.home() / "temp" / "wind-up-benchmarking" / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------------------
# Zenodo fetch
# --------------------------------------------------------------------------------------
def download_zenodo_data(
    record_id: str,
    *,
    output_dir: Path | None = None,
    filenames: Collection[str] | None = None,
    cache_overwrite: bool = False,
) -> None:
    """Download and cache files from zenodo.org."""
    import requests  # noqa: PLC0415  (lazy: keep network deps out of the import path)

    output_dir = output_dir if output_dir is not None else get_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_fpath = output_dir / "zenodo_dataset_metadata.json"

    # One Session for the whole download so its connection pool (and every socket) is
    # closed deterministically on exit. A per-call ``requests.get`` closes its transient
    # pool before the streamed response's socket is released back to it, leaking the
    # socket until GC -- which trips ``filterwarnings = error`` via ResourceWarning.
    with requests.Session() as session:
        if not cache_overwrite and metadata_fpath.is_file():
            logger.info("Loading metadata from %s", metadata_fpath)
            with metadata_fpath.open() as f:
                content = json.load(f)
        else:
            logger.info("Fetching metadata from zenodo...")
            with session.get(
                f"https://zenodo.org/api/records/{record_id}",
                timeout=(_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S),
            ) as r:
                r.raise_for_status()
                content = r.json()
            with metadata_fpath.open("w") as f:
                json.dump(content, f)
            logger.info("Saved metadata to %s", metadata_fpath)

        remote_files: list[dict] = content["files"]
        if filenames is None:
            files_to_download: list[dict] = list(remote_files)
        else:
            files_to_download = list(_check_name_of_files_to_download(filenames, remote_files))
        required_keys = {f["key"] for f in files_to_download}
        # Auto-include any small file in the record (READMEs, deployment reports, ...).
        for rf in remote_files:
            if rf["size"] < SMALL_FILE_THRESHOLD_BYTES and rf["key"] not in required_keys:
                files_to_download.append(rf)

        downloaded_files = 0
        n_files_to_download = len(files_to_download)
        for i_file, file_to_download in enumerate(files_to_download, start=1):
            is_required = file_to_download["key"] in required_keys
            downloaded_files += _download_one_file(
                session,
                file_to_download,
                output_dir,
                cache_overwrite=cache_overwrite,
                is_required=is_required,
                progress_prefix=f"[{i_file}/{n_files_to_download}]",
            )
    logger.info("Download finished: %s new files cached at %s", downloaded_files, output_dir)


def _download_one_file(
    session: requests.Session,
    file_entry: dict,
    output_dir: Path,
    *,
    cache_overwrite: bool,
    is_required: bool,
    progress_prefix: str,
) -> int:
    """Download a single Zenodo file. Returns 1 if a new file was written, 0 otherwise.

    Uses the caller's :class:`requests.Session` so its connection pool is closed once
    by the caller, releasing every socket deterministically rather than at GC.

    Retries up to ``_MAX_DOWNLOAD_ATTEMPTS`` times on transient network errors with
    exponential backoff, resuming partial downloads via a ``Range`` header. Required
    files re-raise after exhausting retries; optional small files warn and clean up.
    """
    import requests  # noqa: PLC0415  (lazy: keep network deps out of the import path)
    from tqdm import tqdm  # noqa: PLC0415

    retryable: tuple[type[requests.RequestException], ...] = (
        requests.ConnectionError,
        requests.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )

    _file_name = file_entry["key"]
    _file_size = file_entry["size"]
    _file_url = file_entry["links"]["self"]
    dst_fpath = output_dir / _file_name

    if cache_overwrite and dst_fpath.is_file():
        dst_fpath.unlink()

    if dst_fpath.is_file() and dst_fpath.stat().st_size >= _file_size:
        logger.info("%s File %s already exists. Skipping download.", progress_prefix, dst_fpath)
        return 0

    logger.info("%s Beginning file download from Zenodo: %s...", progress_prefix, _file_name)
    for attempt in range(1, _MAX_DOWNLOAD_ATTEMPTS + 1):
        is_last_attempt = attempt == _MAX_DOWNLOAD_ATTEMPTS
        existing_size = dst_fpath.stat().st_size if dst_fpath.is_file() else 0
        headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
        try:
            result = session.get(
                _file_url,
                stream=True,
                timeout=(_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S),
                headers=headers,
            )
            result.raise_for_status()
            # If we requested a Range but the server returned 200, it ignored it.
            resume = existing_size > 0 and result.status_code == _HTTP_PARTIAL_CONTENT
            if existing_size > 0 and not resume:
                logger.info(
                    "%s Server did not honor Range request (status %s); restarting from byte 0.",
                    progress_prefix,
                    result.status_code,
                )
                existing_size = 0
            file_mode = "ab" if resume else "wb"
            remaining_bytes = max(0, _file_size - existing_size)
            with (
                result,  # close the streamed response (and its socket) deterministically, not at GC
                Path.open(dst_fpath, file_mode) as f,
                tqdm(
                    total=remaining_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {_file_name} ({_file_size / BYTES_IN_1MB:.2f} MB)",
                ) as pbar,
            ):
                for chunk in result.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))
        except retryable as e:
            if not is_last_attempt:
                partial_size = dst_fpath.stat().st_size if dst_fpath.is_file() else 0
                sleep_s = _BACKOFF_BASE_S * (2 ** (attempt - 1))
                logger.warning(
                    "%s Download attempt %d/%d for %s failed (%s). Have %d/%d bytes. Sleeping %.1fs before retrying.",
                    progress_prefix,
                    attempt,
                    _MAX_DOWNLOAD_ATTEMPTS,
                    _file_name,
                    e,
                    partial_size,
                    _file_size,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue
            return _resolve_download_failure(
                exc=e,
                dst_fpath=dst_fpath,
                file_name=_file_name,
                is_required=is_required,
                progress_prefix=progress_prefix,
            )
        except requests.RequestException as e:
            # Non-retryable (e.g. 4xx HTTPError). Resolve immediately.
            return _resolve_download_failure(
                exc=e,
                dst_fpath=dst_fpath,
                file_name=_file_name,
                is_required=is_required,
                progress_prefix=progress_prefix,
            )
        else:
            return 1

    msg = "unreachable: retry loop should have returned or raised"
    raise RuntimeError(msg)


def _resolve_download_failure(
    *,
    exc: requests.RequestException,
    dst_fpath: Path,
    file_name: str,
    is_required: bool,
    progress_prefix: str,
) -> int:
    """Re-raise for required files; warn-and-clean for optional ones."""
    if is_required:
        # Leave partial bytes on disk so a subsequent run can resume via Range.
        raise exc
    logger.warning(
        "%s Failed to download optional small file %s: %s. Continuing.",
        progress_prefix,
        file_name,
        exc,
    )
    if dst_fpath.is_file():
        dst_fpath.unlink()
    return 0


def _missing_small_files_from_cached_metadata(target_dir: Path) -> list[str]:
    """Return small-file keys absent from ``target_dir`` per cached Zenodo metadata.

    Empty list when the metadata cache is missing or unreadable; the next successful
    fetch rewrites the cache.
    """
    metadata_fpath = target_dir / "zenodo_dataset_metadata.json"
    if not metadata_fpath.is_file():
        return []
    try:
        with metadata_fpath.open() as f:
            content = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return [
        rf["key"]
        for rf in content.get("files", [])
        if rf.get("size", math.inf) < SMALL_FILE_THRESHOLD_BYTES and not (target_dir / rf["key"]).is_file()
    ]


def ensure_hot_data_files(filenames: Collection[str], *, data_dir: Path | None = None) -> None:
    """Download missing Hill of Towie v2 data files from Zenodo.

    Idempotent: makes no network call when every requested file exists locally and
    cached metadata shows no missing small files.
    """
    target_dir = data_dir if data_dir is not None else get_data_dir()
    requested = list(filenames)
    missing_requested = [f for f in requested if not (target_dir / f).is_file()]
    missing_small = _missing_small_files_from_cached_metadata(target_dir)
    if not missing_requested and not missing_small:
        logger.info(
            "ensure_hot_data_files: all %d requested files already present at %s, skipping download",
            len(requested),
            target_dir,
        )
        return
    logger.info(
        "ensure_hot_data_files: downloading from Zenodo record %s into %s "
        "(missing requested: %s; missing small files: %s)",
        HOT_V2_RECORD_ID,
        target_dir,
        missing_requested,
        missing_small,
    )
    download_zenodo_data(record_id=HOT_V2_RECORD_ID, output_dir=target_dir, filenames=missing_requested)


def _check_name_of_files_to_download(filenames: Collection[str], remote_files: Collection[dict]) -> Collection[dict]:
    requested_filenames = set(filenames)
    remote_filenames = {i["key"] for i in remote_files}
    if not requested_filenames.issubset(remote_filenames):
        msg = (
            "Could not find all files in the Zenodo record. "
            f"Missing files: {requested_filenames.difference(remote_filenames)}"
        )
        raise ValueError(msg)
    return [i for i in remote_files if i["key"] in requested_filenames]


# --------------------------------------------------------------------------------------
# 10-minute SCADA loading + wind-up-format conversion
# --------------------------------------------------------------------------------------
class WPSBackupFileField(NamedTuple):
    """Hill of Towie field and table mapping."""

    alias: str
    field_name: str
    table_name: str


# Source-native Hill of Towie tag names referenced by ``HOT_COLUMNS`` (the source-native schema
# methods see). Defined once here and reused below in ``hill_of_towie_fields`` so each tag string
# lives in exactly one place, while ``HOT_COLUMNS`` is built directly from these tags rather than
# routed through the v0 ``DataColumns`` vocabulary (which stays confined to the on-ramp aliases).
_TAG_ACTIVE_POWER_MEAN = "wtc_ActPower_mean"
_TAG_WIND_SPEED_MEAN = "wtc_AcWindSp_mean"
_TAG_WIND_SPEED_SD = "wtc_AcWindSp_stddev"
_TAG_GEN_RPM_MEAN = "wtc_GenRpm_mean"
# Diagnostics-only tags (not estimation inputs): see ``ColumnSchema`` and the shared per-run
# diagnostics. ``wtc_NacelPos_mean`` is a wind-direction proxy for plotting only.
_TAG_PITCH_MEAN = "wtc_PitcPosA_mean"
_TAG_REACTIVE_POWER_MEAN = "wtc_ReactPwr_mean"
_TAG_NACELLE_POSITION_MEAN = "wtc_NacelPos_mean"
_TAG_AMBIENT_TEMP_MEAN = "wtc_AmbieTmp_mean"
_TAG_AVAILABILITY = "wtc_ScReToOp_timeon"
# Reference active-power companion statistics (Issue 11): within-period max/min/SD of active power.
# The SD in particular is a calibration-stable, farm-sited turbulence proxy a method may opt into
# as reference features; the mean stays the primary signal.
_TAG_ACTIVE_POWER_MAX = "wtc_ActPower_max"
_TAG_ACTIVE_POWER_MIN = "wtc_ActPower_min"
_TAG_ACTIVE_POWER_SD = "wtc_ActPower_stddev"


hill_of_towie_fields = [
    WPSBackupFileField(
        alias=DataColumns.active_power_mean, field_name=_TAG_ACTIVE_POWER_MEAN, table_name="tblSCTurGrid"
    ),
    WPSBackupFileField(alias=DataColumns.active_power_sd, field_name=_TAG_ACTIVE_POWER_SD, table_name="tblSCTurGrid"),
    WPSBackupFileField(alias="ActivePowerMax", field_name=_TAG_ACTIVE_POWER_MAX, table_name="tblSCTurGrid"),
    WPSBackupFileField(alias="ActivePowerMin", field_name=_TAG_ACTIVE_POWER_MIN, table_name="tblSCTurGrid"),
    WPSBackupFileField(alias="ReactivePowerMean", field_name="wtc_ReactPwr_mean", table_name="tblSCTurGrid"),
    WPSBackupFileField(alias=DataColumns.wind_speed_mean, field_name=_TAG_WIND_SPEED_MEAN, table_name="tblSCTurbine"),
    WPSBackupFileField(alias=DataColumns.wind_speed_sd, field_name=_TAG_WIND_SPEED_SD, table_name="tblSCTurbine"),
    WPSBackupFileField(alias=DataColumns.yaw_angle_mean, field_name="wtc_NacelPos_mean", table_name="tblSCTurbine"),
    WPSBackupFileField(alias=DataColumns.yaw_angle_min, field_name="wtc_NacelPos_min", table_name="tblSCTurbine"),
    WPSBackupFileField(alias=DataColumns.yaw_angle_max, field_name="wtc_NacelPos_max", table_name="tblSCTurbine"),
    WPSBackupFileField(alias=DataColumns.gen_rpm_mean, field_name=_TAG_GEN_RPM_MEAN, table_name="tblSCTurbine"),
    WPSBackupFileField(alias="pitch_angle_a", field_name="wtc_PitcPosA_mean", table_name="tblSCTurbine"),
    WPSBackupFileField(alias="pitch_angle_b", field_name="wtc_PitcPosB_mean", table_name="tblSCTurbine"),
    WPSBackupFileField(alias="pitch_angle_c", field_name="wtc_PitcPosC_mean", table_name="tblSCTurbine"),
    WPSBackupFileField(alias=DataColumns.ambient_temp, field_name="wtc_AmbieTmp_mean", table_name="tblSCTurTemp"),
    WPSBackupFileField(
        alias="Time ready to operate in period", field_name="wtc_ScReToOp_timeon", table_name="tblSCTurFlag"
    ),
    WPSBackupFileField(alias="YawOperationCounts", field_name="wtc_ScYawOpe_counts", table_name="tblSCTurFlag"),
    WPSBackupFileField(alias="PowerReference", field_name="wtc_PowerRef_endvalue", table_name="tblSCTurbine"),
]

# The Hill of Towie source-native column schema the synthetic pipeline and methods see. The
# raw 10-min tag names (``wtc_*``) are kept as-is (no v0 aliasing); the long-format turbine
# identifier is ``TurbineName`` (assigned by :func:`scada_wide_to_long`). Built directly from
# the source-native tag constants above, so the schema carries no v0 vocabulary.
HOT_TURBINE_COL = "TurbineName"
HOT_COLUMNS = ColumnSchema(
    turbine=HOT_TURBINE_COL,
    active_power=_TAG_ACTIVE_POWER_MEAN,
    active_power_min=_TAG_ACTIVE_POWER_MIN,
    wind_speed=_TAG_WIND_SPEED_MEAN,
    wind_speed_sd=_TAG_WIND_SPEED_SD,
    gen_rpm=_TAG_GEN_RPM_MEAN,
    pitch=_TAG_PITCH_MEAN,
    reactive_power=_TAG_REACTIVE_POWER_MEAN,
    nacelle_position=_TAG_NACELLE_POSITION_MEAN,
    ambient_temp=_TAG_AMBIENT_TEMP_MEAN,
    availability=_TAG_AVAILABILITY,
)

# Baseline rated power of the Hill of Towie test turbines (kW); matches the synthetic generator's
# baseline ``rated_power_kw`` default and caps the power-model counterfactual predictions.
HOT_RATED_POWER_KW = 2300.0

# Hub height of the Hill of Towie turbines (m); feeds the ERA5 hub-height wind-speed derivation.
HOT_HUB_HEIGHT_M = 59.0

# The reference active-power companion statistics (max/min/SD) a method may opt into as features.
HOT_ACTIVE_POWER_STAT_COLS: tuple[str, ...] = (
    _TAG_ACTIVE_POWER_MAX,
    _TAG_ACTIVE_POWER_MIN,
    _TAG_ACTIVE_POWER_SD,
)


def _unpack_hot_10min_year(
    *,
    data_dir: Path,
    year: int,
    serial_numbers: Sequence[int],
    fields_to_load: Sequence[WPSBackupFileField],
) -> pd.DataFrame:
    """Unpack one full year zip into a wide, serial-keyed 10-min dataframe (the slow step).

    This is the expensive part of :func:`load_hot_10min_data` (reading and pivoting every monthly
    CSV in the year zip); it is cached per (year, turbine) by :func:`_load_unpacked_hot_10min`.
    The result covers the whole year (no window clipping), is keyed on serial numbers (not turbine
    names), and only includes the requested ``serial_numbers``. Columns keep their source-native
    tag names (the ``wtc_*`` field names); any v0 aliasing is the v0 baseline's concern.
    """
    from tqdm import tqdm  # noqa: PLC0415  (lazy: keep network deps out of the import path)

    tables_to_load = {x.table_name for x in fields_to_load}
    zip_path = data_dir / f"{year}.zip"
    logger.info("Beginning 10min data unpacking: %s", zip_path)
    with ZipFile(zip_path) as zip_file:
        year_dfs = []
        for _table in tqdm(tables_to_load, desc=f"unpacking {zip_path.stem}"):
            table_dfs = []
            for _month in range(1, 13):
                if (fname := f"{_table}_{year}_{_month:02d}.csv") not in zip_file.namelist():
                    continue
                _df = pd.read_csv(zip_file.open(fname), index_col=0, parse_dates=True)[
                    ["StationId", *[x.field_name for x in fields_to_load if x.table_name == _table]]
                ]
                if _df.index.name != "TimeStamp":
                    msg = f"unexpected index name, {_df.index.name =}"
                    raise ValueError(msg)
                if not isinstance(_df.index, pd.DatetimeIndex):
                    _df.index = pd.to_datetime(_df.index, format="ISO8601")
                    if not isinstance(_df.index, pd.DatetimeIndex):
                        msg = f"unexpected index type, {_df.index.name =} {type(_df.index)=}"
                        raise TypeError(msg)
                # convert to Start Format UTC
                _df.index = _df.index.tz_localize("UTC")  # type:ignore[attr-defined]
                _df.index = _df.index - pd.Timedelta(minutes=10)
                _df.index.name = "TimeStamp_StartFormat"
                # drop any timestamps not in this month; files overlap by 10 minutes
                _df = _df[(_df.index.year == year) & (_df.index.month == _month)]  # type:ignore[attr-defined,assignment]
                _df = _df[_df["StationId"].isin(serial_numbers)]
                pivoted_df = _df.pivot_table(
                    index=_df.index.name,
                    columns="StationId",
                    values=[x for x in _df.columns if x != "StationId"],
                ).swaplevel(axis=1)
                table_dfs.append(pivoted_df)
            table_df = pd.concat(table_dfs, verify_integrity=True, sort=True)
            year_dfs.append(table_df)
        return pd.concat(year_dfs, axis=1)


def _year_turbine_cache_path(
    *,
    year: int,
    serial_number: int,
    fields_to_load: Sequence[WPSBackupFileField],
    cache_dir: Path,
) -> Path:
    """Build the deterministic parquet path for one (year, turbine).

    The Zenodo record is a fixed one-zip-per-year layout, so a turbine-year is the stable unit of
    work: the path depends only on the year, the turbine, and the field set (folded into a short
    hash so a custom field selection gets its own files). It deliberately does **not** depend on
    the requested window or the rest of the turbine subset, so any study reuses these files.
    """
    fields_blob = json.dumps(
        {"fields": sorted(f"{x.table_name}.{x.field_name}->{x.alias}" for x in fields_to_load)},
        sort_keys=True,
    )
    fields_hash = hashlib.sha256(fields_blob.encode("utf-8")).hexdigest()[:16]
    wtg_number = serial_number - _HOT_SERIAL_OFFSET
    return cache_dir / f"hot10min_{year}_T{wtg_number:02d}_{fields_hash}.parquet"


def _load_unpacked_hot_10min(
    *,
    data_dir: Path,
    years_to_load: Sequence[int],
    serial_numbers: Sequence[int],
    fields_to_load: Sequence[WPSBackupFileField],
    cache_dir: Path,
) -> pd.DataFrame:
    """Return the full-year, serial-keyed 10-min df for the requested years and turbines.

    Caches one parquet per (year, turbine): a turbine-year is unpacked from its zip at most once
    and then reused for any window or turbine subset. When some requested turbines are not yet
    cached for a year, that year's zip is unpacked once for just those turbines and one parquet is
    written per turbine. Delete a file to force a re-unpack of that turbine-year.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    year_frames = []
    for year in years_to_load:
        paths = {
            serial: _year_turbine_cache_path(
                year=year,
                serial_number=serial,
                fields_to_load=fields_to_load,
                cache_dir=cache_dir,
            )
            for serial in serial_numbers
        }
        missing = [serial for serial in serial_numbers if not paths[serial].exists()]
        if missing:
            logger.info("HoT %d cache miss for turbines %s; unpacking", year, missing)
            unpacked = _unpack_hot_10min_year(
                data_dir=data_dir,
                year=year,
                serial_numbers=missing,
                fields_to_load=fields_to_load,
            )
            for serial in missing:
                serial_df = unpacked.loc[:, unpacked.columns.get_level_values(0) == serial]
                logger.info("Writing HoT cache: %s", paths[serial])
                serial_df.to_parquet(paths[serial])
        per_turbine = [pd.read_parquet(paths[serial]) for serial in serial_numbers]
        year_frames.append(pd.concat(per_turbine, axis=1))
    return pd.concat(year_frames, verify_integrity=True, sort=True)


def load_hot_10min_data(
    *,
    data_dir: Path,
    wtg_numbers: Sequence[int],
    start_dt: pd.Timestamp,
    end_dt_excl: pd.Timestamp,
    use_turbine_names: bool = True,
    custom_fields: Sequence[WPSBackupFileField] | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Return a wide 10-min SCADA dataframe for Hill of Towie (downloading year zips).

    Columns keep their source-native ``wtc_*`` tag names; the level-0 turbine key is the serial
    number, or the ``T01``-style turbine name when ``use_turbine_names``.

    The slow zip-unpacking step is cached as parquet under ``cache_dir`` (defaults to
    ``data_dir / "unpacked_cache"``), one file per (year, turbine). Because the Zenodo record is a
    fixed one-zip-per-year layout, a turbine-year is unpacked at most once and then reused for any
    window or turbine subset, so repeated studies over the same data skip re-reading every monthly
    CSV.
    """
    if str(start_dt.tz) != "UTC" or str(end_dt_excl.tz) != "UTC":
        msg = "start_dt and end_dt_excl must be in UTC"
        raise ValueError(msg)
    if end_dt_excl <= start_dt:
        msg = "end_dt_excl must be after start_dt"
        raise ValueError(msg)

    serial_numbers = [x + _HOT_SERIAL_OFFSET for x in wtg_numbers]
    first_year_to_load = start_dt.year
    last_year_to_load = (end_dt_excl - pd.Timedelta(seconds=TIMEBASE_S)).year
    years_to_load = list(range(first_year_to_load, last_year_to_load + 1))
    ensure_hot_data_files([f"{y}.zip" for y in years_to_load], data_dir=data_dir)
    fields_to_load = hill_of_towie_fields if custom_fields is None else custom_fields
    combined_df = _load_unpacked_hot_10min(
        data_dir=data_dir,
        years_to_load=years_to_load,
        serial_numbers=serial_numbers,
        fields_to_load=fields_to_load,
        cache_dir=cache_dir if cache_dir is not None else data_dir / "unpacked_cache",
    )
    if use_turbine_names:
        cols = combined_df.columns
        serial_to_name = {x: f"T{x - _HOT_SERIAL_OFFSET:02d}" for x in cols.get_level_values(0).unique()}
        combined_df.columns = cols.set_levels(  # type:ignore[attr-defined]
            [serial_to_name[x] for x in cols.levels[0]],  # type:ignore[attr-defined]
            level=0,
        )
    return (
        combined_df[(combined_df.index >= start_dt) & (combined_df.index < end_dt_excl)]
        .resample(pd.Timedelta(seconds=TIMEBASE_S))
        .first()
    )


def calc_shutdown_duration(wind_up_df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``ShutdownDuration`` column and return the wind-up dataframe.

    Downtime is the time *not* ready to operate in the period; additionally, stuck data —
    a turbine whose signals are unchanged from its own previous record (implausible, and a
    sign of frozen/wrong telemetry) — above a low-wind threshold is treated as a full
    period of downtime.
    """
    wind_up_df = wind_up_df.copy()
    wind_up_df[DataColumns.shutdown_duration] = TIMEBASE_S - wind_up_df["Time ready to operate in period"].fillna(
        TIMEBASE_S
    )
    signal_cols = [
        DataColumns.active_power_mean,
        DataColumns.active_power_sd,
        DataColumns.wind_speed_mean,
        DataColumns.wind_speed_sd,
        DataColumns.gen_rpm_mean,
        DataColumns.pitch_angle_mean,
        DataColumns.yaw_angle_mean,
    ]
    # Stuck (frozen) telemetry: every signal unchanged from the turbine's OWN previous
    # record. The frame holds one row per (timestamp, turbine) interleaved by timestamp,
    # so both the forward-fill and the diff must be grouped by turbine — an ungrouped diff
    # would compare adjacent rows belonging to different turbines, not a turbine over time.
    diffdf = (
        wind_up_df.groupby("TurbineName", observed=False)[signal_cols]
        .ffill()
        .fillna(0.0)
        .groupby(wind_up_df["TurbineName"], observed=False)
        .diff()
    )
    stuck_data = (diffdf == 0).all(axis=1)
    very_low_wind_threshold = 1.5
    very_low_wind = wind_up_df[DataColumns.wind_speed_mean] < very_low_wind_threshold
    stuck_filter = stuck_data & (~very_low_wind)
    wind_up_df.loc[stuck_filter, DataColumns.shutdown_duration] = TIMEBASE_S
    return wind_up_df


def scada_wide_to_long(scada_df: pd.DataFrame, *, columns: ColumnSchema = HOT_COLUMNS) -> pd.DataFrame:
    """Convert wide two-level ``scada_df`` to a narrow, source-native long frame.

    ``scada_df`` has two column levels (turbine, field); the result has one column level plus a
    ``columns.turbine`` identifier column, and keeps the source-native ``wtc_*`` field names. This
    is the method-facing layout: v0-specific aliasing and the derived ``PitchAngleMean`` /
    ``ShutdownDuration`` columns are added later by :func:`long_to_wind_up_format`, which only the
    v0 baseline needs.
    """
    # future_stack=True only exists in pandas >= 2.1; without it >= 2.1 emits a
    # FutureWarning (an error under the test config). Fall back for pandas 2.0.x.
    try:
        stacked = scada_df.stack(level=0, future_stack=True)  # noqa: PD013
    except TypeError:
        stacked = scada_df.stack(level=0, dropna=False)  # noqa: PD013
    return stacked.reset_index(level=1).rename(columns={"StationId": columns.turbine})


def long_to_wind_up_format(long_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a source-native long frame (see :func:`scada_wide_to_long`) to wind-up format.

    Renames the Hill of Towie ``wtc_*`` tag names to their v0 :class:`DataColumns` aliases, derives
    ``PitchAngleMean`` from the three per-blade pitch columns when absent, and computes
    ``ShutdownDuration``. This is the v0 baseline's on-ramp; the rest of the pipeline never needs it.
    """
    alias_by_field = {f.field_name: f.alias for f in hill_of_towie_fields}
    wind_up_df = long_df.rename(columns=alias_by_field)
    if DataColumns.pitch_angle_mean not in wind_up_df.columns:
        wind_up_df[DataColumns.pitch_angle_mean] = wind_up_df[["pitch_angle_a", "pitch_angle_b", "pitch_angle_c"]].mean(
            axis=1
        )
    return calc_shutdown_duration(wind_up_df)


# --------------------------------------------------------------------------------------
# Turbine metadata + top-level loader
# --------------------------------------------------------------------------------------
def load_hot_metadata(*, data_dir: Path | None = None, wtg_names: Sequence[str] | None = None) -> pd.DataFrame:
    """Load Hill of Towie turbine metadata (Name, Latitude, Longitude) in wind-up format."""
    data_dir = data_dir if data_dir is not None else get_data_dir()
    ensure_hot_data_files(["Hill_of_Towie_turbine_metadata.csv"], data_dir=data_dir)
    metadata_path = data_dir / "Hill_of_Towie_turbine_metadata.csv"
    logger.info("Reading: %s", metadata_path)
    return_df = (
        pd.read_csv(metadata_path)
        .loc[:, ["Turbine Name", "Latitude", "Longitude"]]
        .rename(columns={"Turbine Name": "Name"})
        .assign(TimeZone="UTC", TimeSpanMinutes=10, TimeFormat="Start")
    )
    if wtg_names is not None:
        # only return return_df rows where the turbine name is in wtg_names
        return_df = return_df[return_df["Name"].isin(wtg_names)]
    return return_df


def load_hot_scada(
    *,
    start_dt: pd.Timestamp,
    end_dt_excl: pd.Timestamp,
    wtg_numbers: Sequence[int] | None = None,
    wtg_names: Sequence[str] | None = None,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (if needed) and load source-native long Hill of Towie SCADA plus metadata.

    Downloads and caches the v2 datapack year zips from Zenodo, unpacks the requested window, and
    reshapes to a long frame with source-native ``wtc_*`` tag names (see :data:`HOT_COLUMNS`).
    Returns ``(scada_df, metadata_df)`` ready for the synthetic generator. v0-specific aliasing is
    applied later, only by the v0 baseline (see :func:`long_to_wind_up_format`).

    :param start_dt: inclusive UTC window start
    :param end_dt_excl: exclusive UTC window end
    :param wtg_numbers: turbine numbers to load; defaults to all (1..21)
    :param data_dir: data/cache directory; defaults to :func:`get_data_dir`
    """
    data_dir = data_dir if data_dir is not None else get_data_dir()
    metadata_df = load_hot_metadata(data_dir=data_dir, wtg_names=wtg_names)
    wtg_numbers = list(range(HOT_FIRST_WTG, HOT_LAST_WTG + 1)) if wtg_numbers is None else list(wtg_numbers)
    wide_scada_df = load_hot_10min_data(
        data_dir=data_dir,
        wtg_numbers=wtg_numbers,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
    )
    scada_df = scada_wide_to_long(wide_scada_df)
    return scada_df, metadata_df
