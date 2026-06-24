from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_PRINT_BACKEND = "import wind_up, matplotlib; print(matplotlib.get_backend())"
# import pyplot (selecting a non-interactive backend) *before* wind_up, as an interactive
# notebook session would
_PYPLOT_THEN_WIND_UP = (
    "import matplotlib; matplotlib.use('pdf'); import matplotlib.pyplot;"
    " import wind_up; print(matplotlib.get_backend())"
)


def _pythonpath_without_sitecustomize_hooks() -> str | None:
    """Return PYTHONPATH with any entry that injects a ``sitecustomize`` hook removed.

    These tests check wind_up's own backend selection in a pristine interpreter, but an IDE
    can prepend a helper dir to PYTHONPATH whose ``sitecustomize.py`` runs at startup and
    forces a backend, overriding MPLBACKEND. PyCharm's "Show plots in tool window" does exactly
    this (``pycharm_matplotlib_backend`` calls ``matplotlib.use('module://backend_interagg')``).
    Such a foreign override would leak into our subprocess and defeat what we're asserting, so
    strip any PYTHONPATH dir carrying a ``sitecustomize.py``.
    """
    raw = os.environ.get("PYTHONPATH")
    if not raw:
        return None
    kept = [p for p in raw.split(os.pathsep) if p and not (Path(p) / "sitecustomize.py").is_file()]
    return os.pathsep.join(kept) if kept else None


def _backend_after(code: str, *, mplbackend: str | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    sanitized_pythonpath = _pythonpath_without_sitecustomize_hooks()
    if sanitized_pythonpath is None:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = sanitized_pythonpath
    if mplbackend is not None:
        env["MPLBACKEND"] = mplbackend
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip().lower()


def test_default_backend_is_agg_when_mplbackend_unset() -> None:
    # importing wind_up must pin a non-interactive backend so headless / SSH runs don't
    # depend on an X server (see wind_up/__init__.py)
    assert _backend_after(_PRINT_BACKEND) == "agg"


def test_explicit_mplbackend_is_respected() -> None:
    # an explicit MPLBACKEND must win over the wind_up default; "pdf" is non-interactive
    # so it is always importable in CI
    assert _backend_after(_PRINT_BACKEND, mplbackend="pdf") == "pdf"


def test_backend_left_alone_when_pyplot_already_imported() -> None:
    # if pyplot is already imported (e.g. an interactive notebook), wind_up must not switch
    # the user's backend out from under them, even with MPLBACKEND unset
    assert _backend_after(_PYPLOT_THEN_WIND_UP) == "pdf"
