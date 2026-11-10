"""Run a campaign with the power model's per-row dump switched on."""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path

import benchmarking.campaigns.composed as composed

_original = composed.wind_up_method


def main(campaign: str, out_dir: str) -> None:
    rows = Path(out_dir) / "rows"

    def with_rows(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return dataclasses.replace(_original(*args, **kwargs), row_dump_dir=rows)

    composed.wind_up_method = with_rows
    logging.basicConfig(level=logging.INFO)
    composed.run_declaration(campaign, out_dir=Path(out_dir))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
