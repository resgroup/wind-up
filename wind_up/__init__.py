"""wind_up package."""

import os
import sys
from importlib.metadata import version

import matplotlib as mpl

__version__ = version("res-wind-up")

# wind-up renders figures to disk (see PlotConfig); default to a non-interactive backend so
# analyses run headless on CI / SSH / batch hosts without an X server, and so a dropped
# X11-forwarding connection can't kill a long unattended run. Respect an explicit MPLBACKEND
# for users who genuinely want interactive display (show_plots=True).
if "MPLBACKEND" not in os.environ and "matplotlib.pyplot" not in sys.modules:
    try:
        mpl.use("Agg")
    except ImportError:
        # Backend can't be changed after pyplot is imported / configured.
        pass
