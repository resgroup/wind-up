from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from collections.abc import Collection
logger = logging.getLogger(__name__)

BYTES_IN_MB = 1024 * 1024


def setup_logger(log_fpath: Path, level: int = logging.INFO) -> None:
    log_formatter_file = logging.Formatter("%(asctime)s [%(levelname)-8s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    file_handler = logging.FileHandler(log_fpath, mode="w")
    file_handler.setFormatter(log_formatter_file)
    root_logger.addHandler(file_handler)

    log_formatter_console = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter_console)
    root_logger.addHandler(console_handler)


def download_zenodo_data(
    record_id: str, output_dir: Path, filenames: Collection[str] | None = None, *, cache_overwrite: bool = False
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=10)
        r.raise_for_status()

    # Catch all exceptions
    except Exception as e:
        logger.info(f"Failed to download Zenodo record {record_id}. Error: {e}")

        if (output_dir / "SMARTEOLE-WFC-open-dataset.zip").exists():
            logger.info("Using cached dataset.")
            return

        raise

    files_to_download = r.json()["files"]
    if filenames is not None:
        files_to_download = [i for i in files_to_download if i["key"] in set(filenames)]
        if len(files_to_download) != len(filenames):
            msg = (
                f"Could not find all files in the Zenodo record. "
                f"Missing files: {set(filenames) - {i['key'] for i in files_to_download} }"
            )
            raise ValueError(msg)

    filepaths = []
    for file_to_download in files_to_download:
        dst_fpath = output_dir / file_to_download["key"]
        if not dst_fpath.exists() or cache_overwrite:
            logger.info(f"Beginning file download from Zenodo: {file_to_download['key']}...")
            filesize = file_to_download["size"] / BYTES_IN_MB
            result = requests.get(file_to_download["links"]["self"], stream=True, timeout=10)
            chunk_number = 0
            with Path.open(dst_fpath, "wb") as f:
                for chunk in result.iter_content(chunk_size=BYTES_IN_MB):
                    chunk_number = chunk_number + 1
                    print(f"{chunk_number} out of {math.ceil(filesize)} MB downloaded", end="\r")
                    f.write(chunk)
        else:
            logger.info(f"File {dst_fpath} already exists. Skipping download.")
        filepaths.append(dst_fpath)
