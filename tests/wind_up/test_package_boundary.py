"""The shipped package must not depend on the repo-only benchmarking code."""

from __future__ import annotations

import ast
from pathlib import Path

import wind_up


def test_wind_up_does_not_import_benchmarking() -> None:
    offenders = []
    for path in sorted(Path(wind_up.__file__).parent.rglob("*.py")):
        tree = ast.parse(path.read_text())
        modules = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        modules |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        offenders += [f"{path.name}: {m}" for m in sorted(modules) if m.split(".", 1)[0] == "benchmarking"]
    assert not offenders
