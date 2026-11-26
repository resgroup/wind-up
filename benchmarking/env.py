"""Read the benchmarking settings from a ``.env`` file at the repository root.

Copy ``.env.example`` to ``.env`` (which git ignores) and set any of its variables; a variable
already set in the environment wins over the file. The variables are only locations on this
machine:

- ``WIND_UP_BENCHMARKING_DATA_DIR``: where Zenodo open data is downloaded, one ``<record id>``
  directory per record
- ``WIND_UP_BENCHMARKING_OUTPUT_DIR``: where studies and runs write
- ``WIND_UP_CACHE_DIR``: the reanalysis cache

Read them through :func:`data_root`, :func:`output_root` and :func:`cache_dir`, which load the file
the first time any of them is called, so no entry point needs to call :func:`load_env` itself.
"""

from __future__ import annotations

import logging
import os
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"
DATA_DIR_ENV = "WIND_UP_BENCHMARKING_DATA_DIR"
OUTPUT_DIR_ENV = "WIND_UP_BENCHMARKING_OUTPUT_DIR"
CACHE_DIR_ENV = "WIND_UP_CACHE_DIR"
DEFAULT_OUTPUT_ROOT = Path.home() / "temp" / "wind-up-benchmarking"


def load_env(path: Path = ENV_FILE) -> bool:
    """Load ``path`` into the environment without overriding what is already set; whether it exists."""
    if not path.is_file():
        return False
    load_dotenv(path, override=False)
    logger.info("Read settings from %s", path)
    return True


@cache
def _load_once() -> bool:
    return load_env(ENV_FILE)


def _setting(name: str, default: Path) -> Path:
    _load_once()
    value = os.getenv(name)
    return Path(value) if value else default


def data_root() -> Path:
    """Return where Zenodo records are downloaded: ``WIND_UP_BENCHMARKING_DATA_DIR`` or the default."""
    return _setting(DATA_DIR_ENV, DEFAULT_OUTPUT_ROOT / "data")


def output_root() -> Path:
    """Return where studies and runs write: ``WIND_UP_BENCHMARKING_OUTPUT_DIR`` or the default."""
    return _setting(OUTPUT_DIR_ENV, DEFAULT_OUTPUT_ROOT)


def cache_dir() -> Path:
    """Return the reanalysis cache: ``WIND_UP_CACHE_DIR`` or the default."""
    return _setting(CACHE_DIR_ENV, Path.home() / ".cache" / "wind_up")
