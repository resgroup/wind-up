"""wind-up v1 package.

The v1 uplift tool. Under construction: the composed method and public API land in a
later workstream. For now this package exists to claim the ``wind_up`` import name and
expose the distribution version.
"""

from importlib.metadata import version

__version__ = version("res-wind-up")
