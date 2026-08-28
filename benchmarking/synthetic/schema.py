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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class ColumnSchema:
    """The source-native column names for the semantic roles the pipeline reads.

    :param turbine: the turbine-identifier column of the long-format SCADA frame
    :param active_power: mean active power
    :param wind_speed: mean nacelle wind speed
    :param wind_speed_sd: nacelle wind-speed standard deviation (turbulence intensity input)
    :param gen_rpm: mean generator rpm
    :param availability: a "ready to operate" counter (e.g. seconds available in the period).
        **Required**: the methods use it for downtime filtering, which must never be silently
        skipped, so every source must supply it.
    :param active_power_min: the per-reference active-power **minimum** companion column.
        Optional on the schema, but ``PowerModelMethod`` **requires** it — each reference's
        power minimum is a standard model feature, not a per-driver opt-in — and validates its
        presence on construction. Sources without it can still drive the lighter methods.

    The remaining roles are **diagnostics-only**: they name extra signals the shared per-run
    diagnostics plot, never estimation inputs. Each defaults to ``None`` so a source that lacks a
    signal (or a caller that does not care) leaves it unset and the corresponding plots skip
    gracefully. ``nacelle_position`` in particular is a wind-direction *proxy* for plotting and
    must **not** become a model feature — it is post-treatment / not treatment-invariant
    (design-note §3).

    :param pitch: blade pitch angle (a representative single sensor is fine)
    :param reactive_power: mean reactive power
    :param nacelle_position: nacelle/yaw position (wind-direction proxy; diagnostics only)
    :param ambient_temp: ambient temperature
    :param exclude_row: names a caller-supplied **boolean** column marking rows to drop from row
        selection (e.g. special operating modes the treatment cannot affect). Optional and off by
        default; when set, a method that honours it excludes the *test* turbine's flagged rows
        alongside the downtime filter. Must be all-``False`` where unknown (never NaN) so an expanded
        time index never turns a gap into an exclusion; a method honouring the role raises on NaN.
    """

    turbine: str
    active_power: str
    wind_speed: str
    wind_speed_sd: str
    gen_rpm: str
    availability: str
    active_power_min: str | None = None
    pitch: str | None = None
    reactive_power: str | None = None
    nacelle_position: str | None = None
    ambient_temp: str | None = None
    exclude_row: str | None = None

    def require_roles(self, roles: Iterable[str]) -> None:
        """Raise ``ValueError`` if any named role is unset or blank (``None``, empty, or whitespace).

        The estimation methods take their column names *only* from a ``ColumnSchema`` (rather than
        separate per-column arguments that could disagree with it), so each validates on construction
        that the schema actually names every role it reads. ``roles`` are field names — an unknown one
        is a programming error and raises ``AttributeError`` (the call sites pass literal role names).
        """
        missing = [role for role in roles if not (getattr(self, role) or "").strip()]
        if missing:
            msg = f"columns is missing required role(s) {missing}: {self}"
            raise ValueError(msg)
