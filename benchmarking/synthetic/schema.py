"""The column vocabulary the synthetic pipeline and harness speak.

The benchmarking layer is deliberately independent of v0: the synthetic generator, the
ground-truth comparison and the harness all operate on **source-native** SCADA column names
(the real tag names a data source ships), never on v0's :class:`~wind_up.constants.DataColumns`
aliases. A :class:`ColumnSchema` names the handful of semantic roles those components need, so
the only place that knows a source's actual column names is the source adapter, which provides
its own :class:`ColumnSchema` (e.g. ``HOT_COLUMNS`` for Hill of Towie).

v0-specific aliasing is therefore not a pipeline concern at all: it lives entirely inside the
v0 baseline (which converts the source-native frame to wind-up format on the way in).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSchema:
    """The source-native column names for the semantic roles the pipeline reads.

    :param turbine: the turbine-identifier column of the long-format SCADA frame
    :param active_power: mean active power
    :param wind_speed: mean nacelle wind speed
    :param wind_speed_sd: nacelle wind-speed standard deviation (turbulence intensity input)
    :param gen_rpm: mean generator rpm
    """

    turbine: str
    active_power: str
    wind_speed: str
    wind_speed_sd: str
    gen_rpm: str
