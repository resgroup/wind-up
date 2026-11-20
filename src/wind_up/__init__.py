"""wind-up v1 package.

The v1 uplift tool. Under construction: the composed method and public API land in a
later workstream. For now this package exposes the distribution version and the farm-level
aggregation of per-turbine uplift estimates.
"""

from importlib.metadata import version

from wind_up.analysis import AnalysisOutput, AnalysisSpec, CombinedUplift, combine, combine_subjects
from wind_up.facts import NonTurbineMeasurement, Reanalysis, Upgrade, WindFarmFacts
from wind_up.farm import FarmUplift, TurbineUplift, farm_uplift
from wind_up.labels import Label
from wind_up.method import MethodConfigurationError

__version__ = version("res-wind-up")

__all__ = [
    "AnalysisOutput",
    "AnalysisSpec",
    "CombinedUplift",
    "FarmUplift",
    "Label",
    "MethodConfigurationError",
    "NonTurbineMeasurement",
    "Reanalysis",
    "TurbineUplift",
    "Upgrade",
    "WindFarmFacts",
    "__version__",
    "combine",
    "combine_subjects",
    "farm_uplift",
]
