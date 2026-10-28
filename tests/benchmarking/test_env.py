"""Tests for reading benchmarking settings from a .env file."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from benchmarking import env
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


def test_the_locations_load_the_file_once_and_the_shell_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / ".env"
    path.write_text(f"{env.DATA_DIR_ENV}={tmp_path / 'data'}\n{env.OUTPUT_DIR_ENV}={tmp_path / 'out'}\n")
    monkeypatch.setattr(env, "ENV_FILE", path)
    for name in (env.DATA_DIR_ENV, env.OUTPUT_DIR_ENV, env.CACHE_DIR_ENV):
        monkeypatch.delenv(name, raising=False)
    env._load_once.cache_clear()  # noqa: SLF001
    try:
        monkeypatch.setenv(env.CACHE_DIR_ENV, str(tmp_path / "shell"))
        assert env.data_root() == tmp_path / "data"
        assert env.output_root() == tmp_path / "out"
        assert env.cache_dir() == tmp_path / "shell"
    finally:
        env._load_once.cache_clear()  # noqa: SLF001


def test_an_unset_location_takes_its_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "_load_once", lambda: False)
    monkeypatch.delenv(env.OUTPUT_DIR_ENV, raising=False)
    assert env.output_root() == env.DEFAULT_OUTPUT_ROOT
