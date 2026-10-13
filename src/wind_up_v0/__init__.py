"""wind_up package."""

import contextlib
import os
import sys
from importlib.metadata import version

import matplotlib as mpl

__version__ = version("res-wind-up")

# wind-up renders figures to disk (see PlotConfig); default to a non-interactive backend so
# analyses run headless on CI / SSH / batch hosts without an X server, and so a dropped
# X11-forwarding connection can't kill a long unattended run. Respect an explicit MPLBACKEND
# (checked by key presence, not truthiness) for users who want interactive display, and leave
# the backend alone if pyplot is already imported (e.g. an interactive notebook session).
if "MPLBACKEND" not in os.environ and "matplotlib.pyplot" not in sys.modules:
    # use() can raise if the backend can't be set this late; never fail `import wind_up_v0` for it.
    with contextlib.suppress(ImportError):
        mpl.use("Agg")
