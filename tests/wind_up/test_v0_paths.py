"""Importing a 0.x ``wind_up`` path fails with an error that names its ``wind_up_v0`` home."""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys

import pytest

import wind_up
import wind_up_v0
from wind_up import _v0_paths


def _module_names(package: object) -> set[str]:
    prefix = f"{package.__name__}."  # type: ignore[attr-defined]
    return {
        info.name.removeprefix(prefix)
        for info in pkgutil.walk_packages(package.__path__, prefix)  # type: ignore[attr-defined]
    }


V0_ONLY = sorted(_module_names(wind_up_v0) - _module_names(wind_up))


def test_v0_has_modules_that_v1_lacks() -> None:
    assert {"constants", "models", "main_analysis", "plots", "plots.detrend_plots"} <= set(V0_ONLY)


@pytest.mark.parametrize("name", V0_ONLY)
def test_old_path_names_its_new_home(name: str) -> None:
    with pytest.raises(ImportError, match=rf"wind_up_v0\.{name.split('.', maxsplit=1)[0]}\b") as error:
        importlib.import_module(f"wind_up.{name}")
    assert not isinstance(error.value, ModuleNotFoundError)
    assert f"wind_up.{name.split('.', maxsplit=1)[0]}" not in sys.modules


def test_from_import_of_an_old_path_names_its_new_home() -> None:
    with pytest.raises(ImportError, match=r"wind_up_v0\.constants"):
        exec("from wind_up.constants import TIMESTAMP_COL", {})  # noqa: S102


def test_unknown_module_is_still_not_found() -> None:
    assert importlib.util.find_spec("wind_up.no_such_module") is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("wind_up.no_such_module")


def test_v1_modules_are_untouched() -> None:
    for name in sorted(_module_names(wind_up) & _module_names(wind_up_v0)):
        spec = importlib.util.find_spec(f"wind_up.{name}")
        assert spec is not None
        assert not isinstance(spec.loader, _v0_paths.V0PathLoader)


def test_finder_is_installed_once() -> None:
    importlib.reload(wind_up)
    assert sum(isinstance(finder, _v0_paths.V0PathFinder) for finder in sys.meta_path) == 1


def test_finder_ignores_other_packages() -> None:
    finder = _v0_paths.V0PathFinder()
    assert finder.find_spec("wind_up_v0.constants") is None
    assert finder.find_spec("constants") is None
    assert finder.find_spec("wind_up") is None


def test_finder_finds_nothing_without_the_v0_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_v0_paths, "V0_PACKAGE", "wind_up_v0_not_installed")
    assert _v0_paths.V0PathFinder().find_spec("wind_up.constants") is None


@pytest.mark.parametrize(
    "name",
    [
        "RAW_YAWDIR_COL",
        "REANALYSIS_WD_COL",
        "WINDFARM_YAWDIR_COL",
        "add_wf_yawdir",
        "apply_northing_corrections",
        "calc_northed_col_name",
        "check_wtg_northing",
    ],
)
def test_v0_northing_name_names_its_new_home(name: str) -> None:
    with pytest.raises(ImportError, match=r"wind_up_v0\.northing"):
        exec(f"from wind_up.northing import {name}", {})  # noqa: S102


def test_v0_northing_names_all_exist_in_v0() -> None:
    v0_northing = importlib.import_module("wind_up_v0.northing")
    v1_northing = importlib.import_module("wind_up.northing")
    assert all(hasattr(v0_northing, name) for name in v1_northing.V0_NAMES)
    assert not v1_northing.V0_NAMES & set(vars(v1_northing))


def test_unknown_northing_name_is_an_attribute_error() -> None:
    v1_northing = importlib.import_module("wind_up.northing")
    with pytest.raises(AttributeError, match="no_such_name"):
        v1_northing.no_such_name  # noqa: B018
