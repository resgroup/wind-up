"""Synthetic turbine-upgrade callables and their resolution.

An upgrade is a callable ``(rows) -> UpgradeEffect`` describing how it changes a test
turbine's treated rows. Upgrades compose: ``apply_upgrades`` resolves a list against the
*original* baseline in a defined order (Cp ratios multiply and are applied through the
Cp core, then a rated-power change, then a nacelle wind-speed change), so the result does
not depend on list order in surprising ways.

Phase 1 implements the four Issue 1 profiles (constant, wind-speed-dependent and
condition-dependent Cp changes, and rated-power change) plus :class:`WakeSteering`, a
geometry-driven two-turbine wake-steering upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from benchmarking.synthetic.cp_core import power_from_cp_change, region2_fraction, rpm_from_power_change
from benchmarking.synthetic.geometry import WakePair, bearing_deg, derive_wake_steering_pairs, distance_m, wrap180
from benchmarking.synthetic.solar import diurnal_factor
from benchmarking.synthetic.sources.hill_of_towie import HOT_COLUMNS, HOT_LAT, HOT_LON
from wind_up_v0.waking_state import iec_disturbed_sector_deg

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from benchmarking.synthetic.cp_core import CpCore
    from benchmarking.synthetic.schema import ColumnSchema


@dataclass
class UpgradeEffect:
    """One upgrade's contribution, resolved against the original baseline rows.

    :param cp_ratio: per-row (or scalar) multiplicative Cp ratio, e.g. 1.02 for +2% Cp
    :param ws_factor: per-row (or scalar) multiplicative change to the nacelle wind speed
    :param new_rated_power_kw: rated-power override, or None to leave rated unchanged
    :param nacelle_position_delta: per-row (or scalar) additive change to the nacelle position
        (compass degrees), applied mod 360; e.g. a wake-steering yaw offset
    """

    cp_ratio: npt.ArrayLike = 1.0
    ws_factor: npt.ArrayLike = 1.0
    new_rated_power_kw: float | None = None
    nacelle_position_delta: npt.ArrayLike = 0.0


@dataclass(frozen=True)
class ConstantCpChange:
    """A flat Cp change in region 2 (e.g. blade cleaning, fouling or add-ons)."""

    delta: float
    ws_delta: float = 0.0

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this upgrade."""
        return {"kind": "constant_cp", "delta": self.delta, "ws_delta": self.ws_delta}

    def __call__(self, rows: pd.DataFrame, columns: ColumnSchema) -> UpgradeEffect:  # noqa: ARG002
        """Return this upgrade's effect on the given rows."""
        return UpgradeEffect(cp_ratio=1.0 + self.delta, ws_factor=1.0 + self.ws_delta)


@dataclass(frozen=True)
class WindSpeedCpChange:
    """A wind-speed-dependent Cp change (e.g. the AeroUp region-2 shape).

    The Cp delta is interpolated over ``ws_points`` against the turbine's original
    wind speed; outside the point range the nearest endpoint delta is held.
    """

    ws_points: tuple[float, ...]
    deltas: tuple[float, ...]
    ws_delta: float = 0.0

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this upgrade."""
        return {
            "kind": "wind_speed_cp",
            "ws_points": list(self.ws_points),
            "deltas": list(self.deltas),
            "ws_delta": self.ws_delta,
        }

    def __call__(self, rows: pd.DataFrame, columns: ColumnSchema) -> UpgradeEffect:
        """Return this upgrade's effect on the given rows."""
        original_ws = rows[columns.wind_speed].to_numpy(dtype=float)
        delta = np.interp(original_ws, self.ws_points, self.deltas)
        return UpgradeEffect(cp_ratio=1.0 + delta, ws_factor=1.0 + self.ws_delta)


def _condition_series(rows: pd.DataFrame, by: str, columns: ColumnSchema) -> npt.NDArray[np.float64]:
    """Compute a treatment-invariant condition signal from the original rows.

    ``by="ti"`` is turbulence intensity (wind-speed SD / wind-speed mean); any other value is
    treated as the name of a column already present in ``rows``.
    """
    if by == "ti":
        ws = rows[columns.wind_speed].to_numpy(dtype=float)
        sd = rows[columns.wind_speed_sd].to_numpy(dtype=float)
        # NaN (not inf/0-division warning) for calm rows; warnings are errors in tests.
        return np.divide(sd, ws, out=np.full_like(sd, np.nan), where=ws != 0)
    return rows[by].to_numpy(dtype=float)


@dataclass(frozen=True)
class ConditionCpChange:
    """A Cp change that varies with a condition signal of the original data.

    Phase 1 supports ``by="ti"`` (turbulence intensity = WindSpeedSD / WindSpeedMean);
    the Cp delta is interpolated over ``points`` of that condition. ``by`` may also name
    any column already present in the rows.
    """

    by: str
    points: tuple[float, ...]
    deltas: tuple[float, ...]
    ws_delta: float = 0.0

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this upgrade."""
        return {
            "kind": "condition_cp",
            "by": self.by,
            "points": list(self.points),
            "deltas": list(self.deltas),
            "ws_delta": self.ws_delta,
        }

    def __call__(self, rows: pd.DataFrame, columns: ColumnSchema) -> UpgradeEffect:
        """Return this upgrade's effect on the given rows."""
        condition = _condition_series(rows, self.by, columns)
        delta = np.interp(condition, self.points, self.deltas)
        return UpgradeEffect(cp_ratio=1.0 + delta, ws_factor=1.0 + self.ws_delta)


@dataclass(frozen=True)
class RatedPowerChange:
    """A change to the turbine's rated power.

    A downrate (new < old) caps power at the new rated and leaves region-2 power
    unchanged. An uprate (new > old) lifts the region-3 fraction of power toward the new
    rated (deep region-2 power is essentially unaffected), then caps at the new rated.
    The uprate model is a documented synthetic choice rather than measured physics.
    """

    new_rated_power_kw: float

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this upgrade."""
        return {"kind": "rated_power", "new_rated_power_kw": self.new_rated_power_kw}

    def __call__(self, rows: pd.DataFrame, columns: ColumnSchema) -> UpgradeEffect:  # noqa: ARG002
        """Return this upgrade's effect on the given rows."""
        return UpgradeEffect(new_rated_power_kw=self.new_rated_power_kw)


def _north_offset_values(
    index: pd.DatetimeIndex, *, turbine: str, north_offsets: Sequence[tuple[str, pd.Timestamp, float]]
) -> npt.NDArray[np.float64]:
    """Per-row north offset (deg) for ``turbine``, step-applied from time-stamped corrections.

    Each correction holds from its timestamp until the next one for that turbine; rows before the
    first correction take the first (earliest) value. Timestamps are compared in UTC.
    """
    entries = sorted(((ts, off) for (t, ts, off) in north_offsets if t == turbine), key=lambda e: e[0])
    if not entries:
        msg = f"no northing correction for turbine {turbine!r}"
        raise KeyError(msg)
    times = _as_utc_naive(pd.DatetimeIndex([ts for ts, _ in entries]))
    offsets = np.array([off for _, off in entries], dtype=float)
    pos = np.searchsorted(times.to_numpy(), _as_utc_naive(index).to_numpy(), side="right") - 1
    return offsets[np.clip(pos, 0, len(offsets) - 1)]


def _as_utc_naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return ``index`` as a tz-naive UTC DatetimeIndex so timestamps compare consistently."""
    idx = pd.DatetimeIndex(index)
    return idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx


def north_calibrated_direction(
    index: pd.DatetimeIndex,
    nacelle_position: npt.NDArray[np.float64],
    *,
    turbine: str,
    north_offsets: Sequence[tuple[str, pd.Timestamp, float]],
) -> npt.NDArray[np.float64]:
    """North-calibrated wind direction: ``(nacelle_position + north_offset) % 360`` (deg).

    Matches ``wind_up.northing`` (offset added, then wrapped). Used both to gate the injected
    wake-steering effect and to reconstruct the calibrated direction for the diagnostic plots.
    """
    offset = _north_offset_values(index, turbine=turbine, north_offsets=north_offsets)
    return (np.asarray(nacelle_position, dtype=float) + offset) % 360.0


@dataclass(frozen=True)
class WakeSteering:
    """A geometry-driven two-turbine wake-steering upgrade (prepost or toggle).

    Applied to the participating turbines (``test_wtgs``, the same set passed to
    ``generate_dataset``). On construction it derives the directed steering pairs among the
    participants within ``max_separation_d`` rotor diameters. For each turbine it injects, from
    that turbine's north-calibrated direction and the static pair geometry, a cosine yaw **loss**
    (when it is a pair's upstream/steering turbine) and a wake-recovery **gain** (when it is a
    pair's downstream/benefitting turbine); a turbine can hold either role in different wind
    directions. The steering turbine's nacelle position is offset by the commanded yaw and its read
    wind speed is nudged (flow distortion); the benefitting turbine's nacelle wind speed is inflated
    (post-treatment). The downstream gain and ws inflation are scaled by a diurnal factor (stronger
    at night). References (turbines absent from ``test_wtgs``) are never touched.

    The yaw schedule is a trapezoid: full ``max_offset_deg`` within ``plateau_half_deg`` of the wake
    nadir, ramping linearly to zero at ``wd_width / 2``. Near the nadir a deadband applies: within
    ``crossover_half_deg`` the wind direction is too uncertain to pick a steer sign, so the applied yaw
    ramps linearly to zero at the nadir instead of flipping sharply. Because the turbine barely steers
    there, the **whole pair effect** (upstream loss, downstream gain and the reported yaw) shrinks
    toward the nadir even though the wake is strongest there. Offsets are authored in the FLORIS
    convention (CCW positive); the applied nacelle change is compass (CW positive), i.e. the negation.

    Steering availability is also gated on power: the steer magnitude is clipped to a limit that is a
    trapezoid in the **upstream** turbine's original power — zero at/below ``steer_cutin_power_kw`` (a
    parked upstream never steers), the full ``max_offset_deg`` between ``steer_low_full_power_kw`` and
    ``steer_full_power_kw``, and back to zero at ``steer_zero_power_kw`` (rated) — so the whole pair
    effect vanishes at both ends. A separate high-wind-speed gate clips it too: full at/below
    ``steer_ws_fade_start_mps``, ramping linearly to zero at ``steer_ws_zero_mps`` (and above) of the
    upstream's original wind speed. The applied envelope is the **minimum** of the direction, power and
    wind-speed gates, so the most conservative one wins and steering stops in high wind regardless of
    the reported power. The downstream gain is driven by the same upstream envelope: when
    ``generate_dataset`` runs it calls :meth:`prepare` first, which precomputes each pair's upstream
    steer signal so the benefitting turbine is gated on the **upstream's** direction and power rather
    than its own. A direct :meth:`__call__` without :meth:`prepare` falls back to per-turbine gating.

    A turbine only steers when its own inflow is wake-free: if any other turbine in ``coords`` sits
    upwind of it within the IEC 61400-12-1 disturbed sector (``iec_disturbed_sector_deg`` — wide when
    close, narrowing to zero by 20 diameters), its inflow is too turbulent to steer and the whole
    pair effect (loss and the partner's gain) is suppressed for those timestamps.

    :param coords: ``(lat, lon)`` in degrees for every turbine (participants and references)
    :param test_wtgs: participants that may steer or benefit; pairs are formed among these
    :param north_offsets: ``(turbine, timestamp, offset_deg)`` corrections, step-applied like wind_up
    :param rotor_diameter_m: rotor diameter for the proximity / disturbed-sector limits
    :param max_separation_d: maximum upstream-downstream separation, in rotor diameters
    :param wd_width: full steering sector width about each nadir (deg)
    :param plateau_half_deg: half-width of the max-amplitude plateau about the nadir (deg)
    :param max_offset_deg: peak yaw-offset magnitude (deg)
    :param crossover_half_deg: half-width (deg) of the near-nadir steering deadband. The applied yaw
        ramps linearly to zero at the nadir over ``+/-`` this angle (full past it), and the whole pair
        effect (loss, gain and reported yaw) shrinks with it, modelling the controller's inability to
        pick a steer sign when the direction is within this band of the nadir
    :param steer_cutin_power_kw: at/below this upstream power no steering happens (a parked/idling
        upstream never steers); the steer limit rises linearly from here to ``steer_low_full_power_kw``
    :param steer_low_full_power_kw: at/above this upstream power (and below ``steer_full_power_kw``) the
        full steer offset is allowed
    :param steer_full_power_kw: upper power at/below which the full steer offset is allowed
    :param steer_zero_power_kw: at/above this upstream power (rated) no steering happens; the steer
        limit falls linearly from ``max_offset_deg`` to 0 between ``steer_full_power_kw`` and here. The
        limit is a trapezoid in power and clips the steer magnitude (it does not scale the ramp)
    :param steer_ws_fade_start_mps: upstream wind speed at/below which the wind-speed gate is full; the
        gate falls linearly from here to 0 at ``steer_ws_zero_mps``
    :param steer_ws_zero_mps: upstream wind speed at/above which the wind-speed gate is 0 (no steering
        in high wind); the gate is combined with the power gate by taking the minimum of the two
    :param cos_power: exponent in the ``cos(offset)^p`` upstream-loss shape
    :param steer_loss_scale: fraction of the cosine yaw-loss actually applied (keeps the loss shape
        but scales its energy impact to realistic levels)
    :param steer_ws_gain: peak upstream read-ws fractional change (may be negative)
    :param peak_gain: peak downstream Cp gain fraction at the nadir
    :param peak_ws_gain: peak downstream nacelle-ws inflation fraction at the nadir
    :param night_factor: downstream multiplier deep at night
    :param day_factor: downstream multiplier at high sun
    :param site_lat: latitude for the solar/diurnal model (deg)
    :param site_lon: longitude for the solar/diurnal model (deg, east +)
    """

    coords: Mapping[str, tuple[float, float]]
    test_wtgs: Sequence[str]
    north_offsets: Sequence[tuple[str, pd.Timestamp, float]]
    rotor_diameter_m: float = 82.0
    max_separation_d: float = 7.0
    wd_width: float = 30.0
    plateau_half_deg: float = 5.0
    max_offset_deg: float = 20.0
    crossover_half_deg: float = 2.5
    steer_cutin_power_kw: float = 0.0
    steer_low_full_power_kw: float = 230.0
    steer_full_power_kw: float = 1610.0
    steer_zero_power_kw: float = 2300.0
    steer_ws_fade_start_mps: float = 12.0
    steer_ws_zero_mps: float = 14.0
    cos_power: float = 1.5
    steer_loss_scale: float = 0.12
    steer_ws_gain: float = -0.02
    peak_gain: float = 0.09
    peak_ws_gain: float = 0.02
    night_factor: float = 1.3
    day_factor: float = 0.6
    site_lat: float = HOT_LAT
    site_lon: float = HOT_LON
    pairs: tuple[WakePair, ...] = field(init=False, default=())
    wake_neighbours: dict[str, tuple[tuple[float, float], ...]] = field(init=False, default_factory=dict)
    # Populated by ``prepare``: (upstream, downstream) -> per-timestamp upstream steer envelope + sign.
    _upstream_steer_by_pair: dict[tuple[str, str], pd.DataFrame] = field(
        init=False, default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Derive the steering pairs and per-turbine wake neighbours; validate coords and northing."""
        missing_coords = [w for w in self.test_wtgs if w not in self.coords]
        if missing_coords:
            msg = f"coords is missing participant(s) {missing_coords}"
            raise ValueError(msg)
        powers = (
            self.steer_cutin_power_kw,
            self.steer_low_full_power_kw,
            self.steer_full_power_kw,
            self.steer_zero_power_kw,
        )
        if not (0.0 <= powers[0] < powers[1] <= powers[2] < powers[3]):
            msg = (
                "require 0 <= steer_cutin_power_kw < steer_low_full_power_kw "
                "<= steer_full_power_kw < steer_zero_power_kw"
            )
            raise ValueError(msg)
        if not 0.0 < self.steer_ws_fade_start_mps < self.steer_ws_zero_mps:
            msg = "require 0 < steer_ws_fade_start_mps < steer_ws_zero_mps"
            raise ValueError(msg)
        if not 0.0 < self.crossover_half_deg <= self.wd_width / 2.0:
            msg = "require 0 < crossover_half_deg <= wd_width / 2"
            raise ValueError(msg)
        offset_turbines = {t for (t, _, _) in self.north_offsets}
        missing_north = [w for w in self.test_wtgs if w not in offset_turbines]
        if missing_north:
            msg = f"north_offsets is missing correction(s) for participant(s) {missing_north}"
            raise ValueError(msg)
        candidates = derive_wake_steering_pairs(
            self.coords,
            test_wtgs=self.test_wtgs,
            rotor_diameter_m=self.rotor_diameter_m,
            max_separation_d=self.max_separation_d,
        )

        # Precompute, for each candidate upstream, the (bearing-to-neighbour, disturbed-half-angle) of
        # every other turbine that could wake it (IEC sector non-zero, i.e. within 20 diameters).
        neighbours: dict[str, tuple[tuple[float, float], ...]] = {}
        for wtg in {p.upstream for p in candidates}:
            entries = []
            for other, other_latlon in self.coords.items():
                if other == wtg:
                    continue
                dn = distance_m(self.coords[wtg], other_latlon) / self.rotor_diameter_m
                half_angle = float(iec_disturbed_sector_deg(dn)) / 2.0
                if half_angle > 0.0:
                    entries.append((bearing_deg(self.coords[wtg], other_latlon), half_angle))
            neighbours[wtg] = tuple(entries)
        object.__setattr__(self, "wake_neighbours", neighbours)

        # Drop pairs whose upstream is itself waked across its whole steering sector: it can never be
        # the front turbine for that partner, so the steer never happens.
        sector = np.arange(-self.wd_width / 2.0, self.wd_width / 2.0 + 1.0, 1.0)
        pairs = tuple(p for p in candidates if not self._is_waked(p.upstream, p.nadir_bearing + sector).all())
        object.__setattr__(self, "pairs", pairs)

    @property
    def description(self) -> dict:
        """Return serialisable provenance describing this upgrade."""
        return {
            "kind": "wake_steering",
            "pairs": [
                {"upstream": p.upstream, "downstream": p.downstream, "nadir_bearing": p.nadir_bearing}
                for p in self.pairs
            ],
            "north_offsets": [[t, ts, off] for (t, ts, off) in self.north_offsets],
            "wd_width": self.wd_width,
            "plateau_half_deg": self.plateau_half_deg,
            "max_offset_deg": self.max_offset_deg,
            "crossover_half_deg": self.crossover_half_deg,
            "steer_cutin_power_kw": self.steer_cutin_power_kw,
            "steer_low_full_power_kw": self.steer_low_full_power_kw,
            "steer_full_power_kw": self.steer_full_power_kw,
            "steer_zero_power_kw": self.steer_zero_power_kw,
            "steer_ws_fade_start_mps": self.steer_ws_fade_start_mps,
            "steer_ws_zero_mps": self.steer_ws_zero_mps,
            "cos_power": self.cos_power,
            "steer_loss_scale": self.steer_loss_scale,
            "steer_ws_gain": self.steer_ws_gain,
            "peak_gain": self.peak_gain,
            "peak_ws_gain": self.peak_ws_gain,
            "night_factor": self.night_factor,
            "day_factor": self.day_factor,
            "site_lat": self.site_lat,
            "site_lon": self.site_lon,
            "rotor_diameter_m": self.rotor_diameter_m,
            "max_separation_d": self.max_separation_d,
        }

    def _envelope(self, view_angle: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Trapezoidal envelope in [0, 1]: flat within the plateau, linear to 0 at the sector edge."""
        av = np.abs(view_angle)
        half = self.wd_width / 2.0
        ramp = (half - av) / (half - self.plateau_half_deg)
        env = np.where(av <= self.plateau_half_deg, 1.0, ramp)
        return np.clip(env, 0.0, 1.0)

    def _power_availability(self, power_kw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Steering availability in ``[0, 1]``: a trapezoid in the upstream turbine's original power.

        0 at/below ``steer_cutin_power_kw`` (not generating), rising to 1 by ``steer_low_full_power_kw``,
        full through ``steer_full_power_kw``, then falling to 0 at ``steer_zero_power_kw`` (rated).
        Non-finite (downtime) power counts as not generating, i.e. availability 0.
        """
        power = np.asarray(power_kw, dtype=float)
        rise = (power - self.steer_cutin_power_kw) / (self.steer_low_full_power_kw - self.steer_cutin_power_kw)
        fall = (self.steer_zero_power_kw - power) / (self.steer_zero_power_kw - self.steer_full_power_kw)
        avail = np.minimum(np.clip(rise, 0.0, 1.0), np.clip(fall, 0.0, 1.0))
        return np.where(np.isfinite(power), avail, 0.0)

    def _ws_availability(self, wind_speed_mps: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Steering availability in ``[0, 1]`` from the upstream turbine's original wind speed.

        Full at/below ``steer_ws_fade_start_mps``, ramping to 0 at ``steer_ws_zero_mps`` (and above).
        Non-finite (missing) wind speed counts as high wind, i.e. availability 0.
        """
        ws = np.asarray(wind_speed_mps, dtype=float)
        fall = (self.steer_ws_zero_mps - ws) / (self.steer_ws_zero_mps - self.steer_ws_fade_start_mps)
        return np.where(np.isfinite(ws), np.clip(fall, 0.0, 1.0), 0.0)

    def _is_waked(self, steering_wtg: str, wind_from: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Per-row mask where ``steering_wtg`` is itself inside another turbine's IEC disturbed sector."""
        waked = np.zeros(len(wind_from), dtype=bool)
        for neighbour_bearing, half_angle in self.wake_neighbours.get(steering_wtg, ()):
            waked |= np.abs(wrap180(neighbour_bearing - wind_from)) < half_angle
        return waked

    def _steer_envelope(
        self,
        index: pd.DatetimeIndex,
        *,
        nacelle: npt.NDArray[np.float64],
        power: npt.NDArray[np.float64],
        wind_speed: npt.NDArray[np.float64],
        pair: WakePair,
        turbine: str,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per-row steer envelope (magnitude) and yaw sign for ``pair`` from ``turbine``'s inputs.

        The envelope in ``[0, 1]`` is ``min(direction env, power availability, wind-speed availability)``
        reduced near the nadir by a linear deadband ramp (0 at the nadir, full past ``crossover_half_deg``):
        near the nadir the wind direction is too uncertain to pick a steer sign, so the turbine barely
        steers there and the whole pair effect shrinks. Evaluated on the upstream turbine (by
        :meth:`prepare`) it is the physical steer signal; on another turbine it is the per-turbine fallback.
        """
        cal = north_calibrated_direction(index, nacelle, turbine=turbine, north_offsets=self.north_offsets)
        view = wrap180(cal - pair.nadir_bearing)
        env = np.where(self._is_waked(pair.upstream, cal), 0.0, self._envelope(view))
        deadband = np.clip(np.abs(view) / self.crossover_half_deg, 0.0, 1.0)
        availability = np.minimum(self._power_availability(power), self._ws_availability(wind_speed))
        return np.minimum(env, availability) * deadband, np.sign(view)

    def prepare(self, scada_df: pd.DataFrame, *, columns: ColumnSchema = HOT_COLUMNS) -> None:
        """Precompute each pair's upstream steer envelope and yaw sign over the full timestamp index.

        Called once by ``generate_dataset`` before the per-turbine application so the whole pair effect
        (the upstream loss and the downstream gain) is gated on the **upstream's** direction and power.
        Without it a direct :meth:`__call__` falls back to gating each turbine on its own rows.
        """
        if columns.nacelle_position is None:
            return
        store: dict[tuple[str, str], pd.DataFrame] = {}
        for pair in self.pairs:
            upstream = scada_df[scada_df[columns.turbine] == pair.upstream]
            index = pd.DatetimeIndex(upstream.index)
            steer_env, sign = self._steer_envelope(
                index,
                nacelle=upstream[columns.nacelle_position].to_numpy(dtype=float),
                power=upstream[columns.active_power].to_numpy(dtype=float),
                wind_speed=upstream[columns.wind_speed].to_numpy(dtype=float),
                pair=pair,
                turbine=pair.upstream,
            )
            store[(pair.upstream, pair.downstream)] = pd.DataFrame(
                {"env": np.nan_to_num(steer_env), "sign": np.nan_to_num(sign)}, index=index
            )
        object.__setattr__(self, "_upstream_steer_by_pair", store)

    def _pair_steer(
        self, pair: WakePair, *, index: pd.DatetimeIndex, rows: pd.DataFrame, columns: ColumnSchema, turbine: str
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Align the pair's upstream steer envelope and yaw sign to ``rows``.

        Uses the precomputed upstream signal when :meth:`prepare` has run, else the per-turbine
        fallback computed from ``rows`` themselves.
        """
        precomputed = self._upstream_steer_by_pair.get((pair.upstream, pair.downstream))
        if precomputed is not None:
            aligned = precomputed.reindex(index)  # timestamps with no upstream row -> no steer
            return np.nan_to_num(aligned["env"].to_numpy(dtype=float)), np.nan_to_num(
                aligned["sign"].to_numpy(dtype=float)
            )
        return self._steer_envelope(
            index,
            nacelle=rows[columns.nacelle_position].to_numpy(dtype=float),
            power=rows[columns.active_power].to_numpy(dtype=float),
            wind_speed=rows[columns.wind_speed].to_numpy(dtype=float),
            pair=pair,
            turbine=turbine,
        )

    def __call__(self, rows: pd.DataFrame, columns: ColumnSchema) -> UpgradeEffect:
        """Return this turbine's combined steering loss and/or wake-recovery gain for these rows."""
        if columns.nacelle_position is None:
            msg = "WakeSteering requires columns.nacelle_position (the wind-direction gate)"
            raise ValueError(msg)
        turbine = str(rows[columns.turbine].iloc[0])
        index = pd.DatetimeIndex(rows.index)

        n = len(rows)
        cp_ratio = np.ones(n)
        ws_factor = np.ones(n)
        nacelle_delta = np.zeros(n)
        diurnal: npt.NDArray[np.float64] | None = None
        for pair in self.pairs:
            if turbine not in (pair.upstream, pair.downstream):
                continue
            # The steer envelope and sign are the UPSTREAM turbine's (via ``prepare``), so the loss and the
            # partner's gain share one arbiter: the upstream's actual steer. ``steer_env`` already folds in
            # the near-nadir deadband, so the loss, gain, ws and reported yaw all shrink together there.
            steer_env, sign = self._pair_steer(pair, index=index, rows=rows, columns=columns, turbine=turbine)
            if turbine == pair.upstream:
                yaw_magnitude = self.max_offset_deg * steer_env
                cos_deficit = 1.0 - np.maximum(np.cos(np.radians(yaw_magnitude)), 0.0) ** self.cos_power
                cp_ratio = cp_ratio * (1.0 - self.steer_loss_scale * cos_deficit)
                ws_factor = ws_factor * (1.0 + self.steer_ws_gain * steer_env)
                nacelle_delta = nacelle_delta - sign * yaw_magnitude
            if turbine == pair.downstream:
                if diurnal is None:
                    diurnal = diurnal_factor(
                        index,
                        lat=self.site_lat,
                        lon=self.site_lon,
                        night_factor=self.night_factor,
                        day_factor=self.day_factor,
                    )
                cp_ratio = cp_ratio * (1.0 + self.peak_gain * steer_env * diurnal)
                ws_factor = ws_factor * (1.0 + self.peak_ws_gain * steer_env * diurnal)
        return UpgradeEffect(cp_ratio=cp_ratio, ws_factor=ws_factor, nacelle_position_delta=nacelle_delta)


def apply_rated_change(
    power_kw: npt.NDArray[np.float64],
    *,
    original_power_kw: npt.NDArray[np.float64],
    old_rated_kw: float,
    new_rated_kw: float,
) -> npt.NDArray[np.float64]:
    """Apply a rated-power change to (already Cp-adjusted) power.

    In future this function should change other test turbine fields especially pitch angle but this is deferred for now.

    See :class:`RatedPowerChange` for the downrate/uprate behaviour.
    """
    if new_rated_kw <= old_rated_kw:
        return np.minimum(power_kw, new_rated_kw)
    region3_fraction = 1.0 - region2_fraction(original_power_kw)
    ratio = new_rated_kw / old_rated_kw
    lifted = power_kw * (1.0 + region3_fraction * (ratio - 1.0))
    return np.minimum(lifted, new_rated_kw)


def apply_upgrades(
    rows: pd.DataFrame, upgrades: list, *, cp: CpCore, columns: ColumnSchema = HOT_COLUMNS
) -> pd.DataFrame:
    """Resolve and apply a list of upgrades to one test turbine's treated rows.

    Cp ratios from all upgrades multiply together and are applied to the original power
    through the Cp core (region-2 weighted, rated-clipped); rpm tracks the resulting
    power change and the nacelle wind speed is scaled by the combined ws factor. The
    input ``rows`` is not mutated.

    :param rows: treated rows for one test turbine, carrying the original SCADA tags
    :param upgrades: upgrade callables to resolve
    :param cp: the turbine's Cp core (rated power and Cp parameters)
    :param columns: the source-native column schema ``rows`` is keyed by
    :return: a new frame with modified power, rpm and wind-speed columns
    """
    out = rows.copy()
    n = len(rows)
    baseline_power = rows[columns.active_power].to_numpy(dtype=float)
    baseline_rpm = rows[columns.gen_rpm].to_numpy(dtype=float)
    baseline_ws = rows[columns.wind_speed].to_numpy(dtype=float)

    cp_ratio = np.ones(n)
    ws_factor = np.ones(n)
    nacelle_delta = np.zeros(n)
    new_rated_power_kw: float | None = None
    for upgrade in upgrades:
        effect = upgrade(rows, columns)
        cp_ratio = cp_ratio * np.asarray(effect.cp_ratio, dtype=float)
        ws_factor = ws_factor * np.asarray(effect.ws_factor, dtype=float)
        nacelle_delta = nacelle_delta + np.asarray(effect.nacelle_position_delta, dtype=float)
        if effect.new_rated_power_kw is not None:
            new_rated_power_kw = effect.new_rated_power_kw

    new_power = power_from_cp_change(baseline_power, cp_ratio=cp_ratio, rated_power_kw=cp.rated_power_kw)
    if new_rated_power_kw is not None:
        new_power = apply_rated_change(
            new_power,
            original_power_kw=baseline_power,
            old_rated_kw=cp.rated_power_kw,
            new_rated_kw=new_rated_power_kw,
        )
    new_rpm = rpm_from_power_change(baseline_rpm=baseline_rpm, baseline_power_kw=baseline_power, new_power_kw=new_power)

    out[columns.active_power] = new_power
    out[columns.gen_rpm] = new_rpm
    out[columns.wind_speed] = baseline_ws * ws_factor
    if columns.nacelle_position is not None and np.any(nacelle_delta != 0.0):
        baseline_nacelle = rows[columns.nacelle_position].to_numpy(dtype=float)
        out[columns.nacelle_position] = (baseline_nacelle + nacelle_delta) % 360.0
    return out
