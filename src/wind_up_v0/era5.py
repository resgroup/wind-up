"""ERA5 reanalysis data fetching via the Open-Meteo archive API.

The Open-Meteo client libraries (``openmeteo_requests``, ``requests_cache``, ``retry_requests``)
are an optional dependency group (``era5``) and are imported lazily, so this module imports
without them; only the live network fetch needs them installed.

The returned DataFrame uses the Open-Meteo column names (``wind_speed_100m`` /
``wind_direction_100m``); :func:`wind_up.reanalysis_data._reanalysis_upsample` already renames
those to the wind-up reanalysis columns, so a :class:`~wind_up.reanalysis_data.ReanalysisDataset`
built from it drops straight into an analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR_ENV = "WIND_UP_CACHE_DIR"

ERA5_DEFAULT_FIELDS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "weather_code",
]


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    """Resolve the wind-up cache directory.

    Precedence: explicit ``cache_dir`` arg, then the ``WIND_UP_CACHE_DIR`` env var, then
    ``~/.cache/wind_up``. The directory is not created here (callers create it before writing).
    """
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.getenv(CACHE_DIR_ENV)
    if env:
        return Path(env)
    return Path.home() / ".cache" / "wind_up"


def _build_era5_df(response: object, fields: list[str]) -> pd.DataFrame:
    """Build a tidy hourly DataFrame from a single Open-Meteo response object."""
    hourly_data = response.Hourly()  # type: ignore[attr-defined]  # openmeteo_requests has no type stubs
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=pd.to_datetime(hourly_data.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly_data.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly_data.Interval()),
                inclusive="left",
            )
        }
        | {field: hourly_data.Variables(i).ValuesAsNumpy() for i, field in enumerate(fields)}
    ).set_index("timestamp")


def _era5_cache_path(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    fields: list[str],
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Build a deterministic parquet cache path from the request args."""
    args_blob = json.dumps(
        {"lat": lat, "lon": lon, "start_date": start_date, "end_date": end_date, "fields": list(fields)},
        sort_keys=True,
    )
    args_hash = hashlib.sha256(args_blob.encode("utf-8")).hexdigest()[:16]
    base = _resolve_cache_dir(cache_dir) / "era5_data"
    return base / f"ERA5_{lat:.2f}_{lon:.2f}_{start_date}_{end_date}_{args_hash}.parquet"


def get_era5_hourly_df(
    *,
    lat: float,
    lon: float,
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    fields: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch hourly ERA5 data from Open-Meteo for any location and return as a DataFrame.

    ``fields`` defaults to a copy of :data:`ERA5_DEFAULT_FIELDS` when ``None``. ``end_date``
    defaults to today (UTC) when ``None``. Each unique combination of arguments is cached to
    its own parquet file keyed by a hash of the arguments. Delete the cache file to force a
    refetch.
    """
    if fields is None:
        fields = list(ERA5_DEFAULT_FIELDS)
    if end_date is None:
        end_date = pd.Timestamp.now(tz="UTC").normalize().strftime("%Y-%m-%d")
    cache_path = _era5_cache_path(lat, lon, start_date, end_date, fields, cache_dir=cache_dir)
    if cache_path.exists():
        logger.info("Reading: %s", cache_path)
        return pd.read_parquet(cache_path)

    df = _fetch_era5_from_open_meteo(  # pragma: no cover - live network fetch
        lat=lat, lon=lon, start_date=start_date, end_date=end_date, fields=fields, cache_dir=cache_dir
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)  # pragma: no cover - live network fetch
    logger.info("Writing: %s", cache_path)  # pragma: no cover - live network fetch
    df.to_parquet(cache_path)  # pragma: no cover - live network fetch
    return df  # pragma: no cover - live network fetch


def _fetch_era5_from_open_meteo(  # pragma: no cover - live network fetch
    *,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    fields: list[str],
    cache_dir: str | Path | None,
) -> pd.DataFrame:
    """Fetch a single Open-Meteo archive response and build the hourly DataFrame.

    The Open-Meteo client libraries are imported here so the module imports without the
    optional ``era5`` dependency group installed.
    """
    import openmeteo_requests  # noqa: PLC0415
    import requests_cache  # noqa: PLC0415
    from retry_requests import retry  # noqa: PLC0415

    requests_cache_path = _resolve_cache_dir(cache_dir) / "openmeteo_requests_cache"
    requests_cache_path.parent.mkdir(parents=True, exist_ok=True)
    openmeteo = openmeteo_requests.Client(
        session=retry(
            requests_cache.CachedSession(str(requests_cache_path), expire_after=3600),
            retries=5,
            backoff_factor=0.2,
        )
    )
    responses = openmeteo.weather_api(
        url="https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": fields,
            "models": "era5",
            "wind_speed_unit": "ms",
        },
    )
    return _build_era5_df(responses[0], fields)
