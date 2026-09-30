"""Tests for reading benchmarking settings from a .env file."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from benchmarking.env import load_env

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

VARIABLE = "WIND_UP_TEST_ENV_VARIABLE"


def test_a_missing_file_changes_nothing(tmp_path: Path) -> None:
    assert not load_env(tmp_path / ".env")


def test_the_file_sets_variables_but_the_shell_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / ".env"
    path.write_text(f"{VARIABLE}=from-file\n# a comment\n")
    monkeypatch.delenv(VARIABLE, raising=False)
    assert load_env(path)
    assert os.environ[VARIABLE] == "from-file"
    monkeypatch.setenv(VARIABLE, "from-shell")
    load_env(path)
    assert os.environ[VARIABLE] == "from-shell"
    monkeypatch.delenv(VARIABLE)
