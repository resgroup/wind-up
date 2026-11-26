"""Fail imports of 0.x ``wind_up`` module paths with an error naming their ``wind_up_v0`` home.

:func:`install` adds a finder that answers ``wind_up.<module>`` only when no such module exists
and ``wind_up_v0.<module>`` does. Importing it raises :class:`ImportError`.
"""

from __future__ import annotations

import importlib.util
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

V1_PACKAGE = "wind_up"
V0_PACKAGE = "wind_up_v0"


def moved_message(old: str, new: str) -> str:
    """Return the error text for the 0.x name ``old`` that now lives at ``new``."""
    return f"{old} is a res-wind-up 0.x path. From 1.0 the 0.x code is the {V0_PACKAGE} package: use {new}."


def _v0_module_exists(relative_name: str) -> bool:
    package = importlib.util.find_spec(V0_PACKAGE)
    if package is None or not package.submodule_search_locations:
        return False
    parts = relative_name.split(".")
    for location in package.submodule_search_locations:
        base = Path(location).joinpath(*parts)
        if base.with_suffix(".py").is_file() or (base / "__init__.py").is_file():
            return True
    return False


class V0PathLoader(Loader):
    """Loader whose module raises :class:`ImportError` naming the ``wind_up_v0`` module."""

    def __init__(self, relative_name: str) -> None:
        """Take the module's name below the package, such as ``models``."""
        self.relative_name = relative_name

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:  # noqa: ARG002
        """Use the default module creation."""
        return None

    def exec_module(self, module: ModuleType) -> None:  # noqa: ARG002
        """Raise the error."""
        raise ImportError(
            moved_message(f"{V1_PACKAGE}.{self.relative_name}", f"{V0_PACKAGE}.{self.relative_name}"),
            name=f"{V1_PACKAGE}.{self.relative_name}",
        )


class V0PathFinder(MetaPathFinder):
    """Find ``wind_up.<module>`` for a module that exists only in ``wind_up_v0``."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,  # noqa: ARG002
        target: ModuleType | None = None,  # noqa: ARG002
    ) -> ModuleSpec | None:
        """Return a spec that fails on import, or None for any other name."""
        package, _, relative_name = fullname.partition(".")
        if package != V1_PACKAGE or not relative_name or not _v0_module_exists(relative_name):
            return None
        return ModuleSpec(fullname, V0PathLoader(relative_name))


def install() -> None:
    """Add the finder after the standard ones, once."""
    if not any(isinstance(finder, V0PathFinder) for finder in sys.meta_path):
        sys.meta_path.append(V0PathFinder())
