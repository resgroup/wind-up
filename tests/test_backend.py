from __future__ import annotations

import os
import subprocess
import sys

_PRINT_BACKEND = "import wind_up, matplotlib; print(matplotlib.get_backend())"
# import pyplot (selecting a non-interactive backend) *before* wind_up, as an interactive
# notebook session would
_PYPLOT_THEN_WIND_UP = (
    "import matplotlib; matplotlib.use('pdf'); import matplotlib.pyplot;"
    " import wind_up; print(matplotlib.get_backend())"
)


def _backend_after(code: str, *, mplbackend: str | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
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
