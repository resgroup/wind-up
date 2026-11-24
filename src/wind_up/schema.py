"""Which column of the analyst's records holds each signal wind-up reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


# The unit of each signal role.
ROLE_UNITS: dict[str, str] = {
    "active_power": "kW",
    "wind_speed": "m/s",
    "wind_speed_sd": "m/s",
    "gen_rpm": "rpm",
    "availability": "s",
    "active_power_min": "kW",
    "pitch": "deg",
    "reactive_power": "kvar",
    "nacelle_position": "deg",
    "nacelle_position_min": "deg",
    "nacelle_position_max": "deg",
    "ambient_temp": "degC",
    "power_setpoint": "kW",
}

# The inclusive range of each role's values, in its unit; a value outside it is blanked.
ROLE_RANGES: dict[str, tuple[float, float]] = {
    "wind_speed": (0.0, 200.0),
    "wind_speed_sd": (0.0, 200.0),
    "gen_rpm": (-100.0, 5000.0),
    "pitch": (-360.0, 360.0),
    "nacelle_position": (-720.0, 720.0),
    "nacelle_position_min": (-720.0, 720.0),
    "nacelle_position_max": (-720.0, 720.0),
    "ambient_temp": (-100.0, 200.0),
}

# The inclusive range of each role's values as multiples of the turbine's rated power; a value outside it is blanked.
ROLE_RANGES_RATED: dict[str, tuple[float, float]] = {
    "active_power_min": (-1.0, 2.0),
    "reactive_power": (-2.0, 2.0),
}


@dataclass(frozen=True)
class ColumnSchema:
    """The analyst's record column for each role wind-up reads; a role left None is a signal the records lack.

    Only ``turbine`` and ``active_power`` are required. A method checks the roles it reads with
    :meth:`require_roles`.

    Each signal is in the unit :data:`ROLE_UNITS` gives. Building the facts blanks a value outside its
    role's range (:data:`ROLE_RANGES`, :data:`ROLE_RANGES_RATED`).

    :param turbine: the turbine identifier column of the long-format records
    :param active_power: mean active power, kW
    :param wind_speed: mean nacelle wind speed, m/s
    :param wind_speed_sd: nacelle wind speed standard deviation, m/s
    :param gen_rpm: mean generator speed, rpm
    :param availability: seconds of the record the turbine was ready to operate. Without it the
        operating states cannot label downtime.
    :param active_power_min: minimum active power in the record, kW
    :param pitch: blade pitch angle, degrees; one blade is enough
    :param reactive_power: mean reactive power, kvar
    :param nacelle_position: nacelle position, degrees, cumulative or not; northing calibrates it to a
        wind direction
    :param nacelle_position_min: minimum nacelle position in the record, degrees
    :param nacelle_position_max: maximum nacelle position in the record, degrees
    :param ambient_temp: ambient temperature, degrees C
    :param power_setpoint: the power setpoint at the end of each record; the synthetic upgrades taper
        and cap their power change at the higher of a record's start and end setpoints
    """

    turbine: str
    active_power: str
    wind_speed: str | None = None
    wind_speed_sd: str | None = None
    gen_rpm: str | None = None
    availability: str | None = None
    active_power_min: str | None = None
    pitch: str | None = None
    reactive_power: str | None = None
    nacelle_position: str | None = None
    nacelle_position_min: str | None = None
    nacelle_position_max: str | None = None
    ambient_temp: str | None = None
    power_setpoint: str | None = None

    def northed(self, role: str) -> str:
        """Return the name of the north-calibrated copy of ``role``'s column, e.g. ``northed_YawAngleMean``.

        Northing writes it beside the original column. Raises ``ValueError`` if ``role`` is unset.
        """
        self.require_roles([role])
        return f"northed_{getattr(self, role)}"

    def name_of(self, role: str) -> str:
        """Return the column named for ``role``. Raises ``ValueError`` if the role is unset."""
        self.require_roles([role])
        return str(getattr(self, role))

    def require_roles(self, roles: Iterable[str]) -> None:
        """Raise ``ValueError`` if any of ``roles`` is unset or blank; an unknown role raises ``AttributeError``."""
        missing = [role for role in roles if not (getattr(self, role) or "").strip()]
        if missing:
            msg = f"columns is missing required role(s) {missing}: {self}"
            raise ValueError(msg)
