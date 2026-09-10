"""Run a declared campaign from the command line.

::

    python -m benchmarking.campaigns run campaign.yaml --out DIR

``--out`` defaults to ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``<the campaign's name>``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # headless: the report writes plots without a display

from benchmarking.campaigns.composed import run_declaration


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: the command, the declaration, and where to write."""
    parser = argparse.ArgumentParser(prog="python -m benchmarking.campaigns", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a campaign declaration and write its report")
    run.add_argument("declaration", type=Path, help="the campaign YAML file")
    run.add_argument("--out", type=Path, default=None, help="output directory (default: from the campaign name)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the declared campaign."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_declaration(args.declaration, out_dir=args.out)


if __name__ == "__main__":
    main()
