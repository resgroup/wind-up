"""Read the benchmarking settings from a ``.env`` file at the repository root.

Copy ``.env.example`` to ``.env`` (which git ignores) and set any of its variables; a variable
already set in the environment wins over the file. The variables are only locations on this
machine:

- ``WIND_UP_BENCHMARKING_DATA_DIR``: where Zenodo open data is downloaded, one ``<record id>``
  directory per record
- ``WIND_UP_BENCHMARKING_OUTPUT_DIR``: where studies and runs write
- ``WIND_UP_CACHE_DIR``: the reanalysis cache
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"


def load_env(path: Path = ENV_FILE) -> bool:
    """Load ``path`` into the environment without overriding what is already set; whether it exists."""
    if not path.is_file():
        return False
    load_dotenv(path, override=False)
    logger.info("Read settings from %s", path)
    return True
