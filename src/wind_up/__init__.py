"""wind-up v1 package.

The v1 uplift tool. Under construction: the composed method and public API land in a
later workstream. For now this package exposes the distribution version and the farm-level
aggregation of per-turbine uplift estimates.
"""

from importlib.metadata import version

from wind_up.farm import FarmUplift, TurbineUplift, farm_uplift

__version__ = version("res-wind-up")

__all__ = ["FarmUplift", "TurbineUplift", "__version__", "farm_uplift"]
