"""Tests for the campaign entry point: running a declaration without importing anything."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarking.campaigns.__main__ import parse_args

if TYPE_CHECKING:
    from pathlib import Path


def test_it_takes_a_declaration_to_run(tmp_path: Path) -> None:
    args = parse_args(["run", str(tmp_path / "campaign.yaml")])
    assert args.declaration == tmp_path / "campaign.yaml"
    assert args.out is None


def test_an_output_directory_can_be_given(tmp_path: Path) -> None:
    args = parse_args(["run", str(tmp_path / "campaign.yaml"), "--out", str(tmp_path / "here")])
    assert args.out == tmp_path / "here"


def test_it_rejects_an_unknown_command(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(["frobnicate", str(tmp_path / "campaign.yaml")])


def test_it_rejects_a_missing_declaration_argument() -> None:
    with pytest.raises(SystemExit):
        parse_args(["run"])
