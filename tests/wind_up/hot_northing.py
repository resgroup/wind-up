"""Shared helpers for the real-data northing tests: the Hill of Towie ``north_farm`` input fixture.

``northing_farm_inputs.parquet`` is written by ``benchmarking.baselines.make_northing_farm_fixture``
from exactly the inputs ``study_northing_degradation`` northes: every turbine's yaw, power and
nacelle wind speed on its ``yaw_usable`` rows, and the ERA5 direction, for 2017-2020. Signals are
stored integer-scaled and blanked off a turbine's usable rows, so its usable mask is where its yaw
is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from wind_up.circular_math import circ_diff
from wind_up.geodesy import local_east_north
from wind_up.layout import LATITUDE_COL, LONGITUDE_COL, NAME_COL, Layout
from wind_up.northing import north_farm

NORTHING_DIR = Path(__file__).parents[1] / "test_data" / "hot" / "northing"
FIXTURE = NORTHING_DIR / "northing_farm_inputs.parquet"
METADATA = Path(__file__).parents[1] / "test_data" / "hot" / "scada" / "Hill_of_Towie_turbine_metadata.csv"
ALL_TURBINES = tuple(f"T{n:02d}" for n in range(1, 22))

# Two two-year windows rather than one four-year one: the changepoint search costs roughly the cube
# of the record length, so this covers the same events for a quarter of the runtime.
EARLY = ("2017-01-01", "2019-01-01")
LATE = ("2019-01-01", "2021-01-01")

# column suffix -> (north_farm input, stored integer per unit)
_SIGNALS = {"yaw_decideg": ("direction", 10.0), "power_kw": ("power", 1.0), "ws_cm_s": ("wind_speed", 100.0)}


def fixture_available() -> bool:
    """Whether the fixture holds real Parquet rather than an unsmudged git-lfs pointer."""
    try:
        with FIXTURE.open("rb") as handle:
            return handle.read(4) == b"PAR1"
    except OSError:
        return False


def load_fixture() -> pd.DataFrame:
    """The fixture as stored (integer-scaled, one row per 10 minutes with any turbine usable)."""
    return pd.read_parquet(FIXTURE)


@dataclass(frozen=True)
class FarmInputs:
    """``north_farm`` inputs for a set of turbines over a window, positional on ``index``."""

    index: pd.DatetimeIndex
    reference: np.ndarray
    direction: dict[str, np.ndarray]
    usable: dict[str, np.ndarray]
    power: dict[str, np.ndarray]
    wind_speed: dict[str, np.ndarray]
    turbines: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "turbines", tuple(sorted(self.direction)))

    def rotated(self, degrees: float) -> FarmInputs:
        """Every direction (each turbine's yaw and the reanalysis) rotated by ``degrees``."""
        return FarmInputs(
            index=self.index,
            reference=(self.reference + degrees) % 360.0,
            direction={t: (d + degrees) % 360.0 for t, d in self.direction.items()},
            usable=self.usable,
            power=self.power,
            wind_speed=self.wind_speed,
        )

    def knocked_out(self, missing: dict[str, np.ndarray]) -> FarmInputs:
        """The same inputs with each turbine's ``missing`` rows blanked and made unusable."""

        def blank(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            return {t: np.where(missing[t], np.nan, v) for t, v in signals.items()}

        return FarmInputs(
            index=self.index,
            reference=self.reference,
            direction=blank(self.direction),
            usable={t: u & ~missing[t] for t, u in self.usable.items()},
            power=blank(self.power),
            wind_speed=blank(self.wind_speed),
        )


def farm_inputs(
    fixture: pd.DataFrame, turbines: tuple[str, ...], start: str, end: str, *, last_days: int | None = None
) -> FarmInputs:
    """Slice the fixture to ``turbines`` over ``[start, end)``, optionally only its last ``last_days``."""
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    if last_days is not None:
        lo = max(lo, hi - pd.Timedelta(days=last_days))
    rows = fixture[(fixture.index >= lo) & (fixture.index < hi)]
    rows = rows[rows[[f"{t}_yaw_decideg" for t in turbines]].notna().any(axis=1)]
    signals: dict[str, dict[str, np.ndarray]] = {name: {} for name, _ in _SIGNALS.values()}
    for turbine in turbines:
        for suffix, (name, scale) in _SIGNALS.items():
            signals[name][turbine] = rows[f"{turbine}_{suffix}"].to_numpy(dtype=float, na_value=np.nan) / scale
    return FarmInputs(
        index=pd.DatetimeIndex(rows.index),
        reference=rows["reference_decideg"].to_numpy(dtype=float, na_value=np.nan) / 10.0,
        usable={t: np.isfinite(d) for t, d in signals["direction"].items()},
        **signals,
    )


def hot_layout(turbines: tuple[str, ...] = ALL_TURBINES) -> Layout:
    """The Hill of Towie layout for ``turbines``, with each turbine's published rotor diameter."""
    meta = pd.read_csv(METADATA, encoding="utf-8-sig")
    meta = meta[meta["Turbine Name"].isin(turbines)]
    return Layout.from_frame(
        pd.DataFrame(
            {
                "name": meta["Turbine Name"],
                "latitude": meta["Latitude"],
                "longitude": meta["Longitude"],
                "rotor_diameter_m": meta["Rotor Diameter (m)"],
            }
        )
    )


def contiguous_order(layout: Layout) -> list[str]:
    """Turbines ordered so every prefix is a spatially connected cluster (as the degradation study).

    Seeded from the westernmost turbine, then repeatedly the turbine nearest to any already chosen,
    so the first ``k`` names keep a wake pair alive for pass 4 even when ``k`` is small.
    """
    east, _ = local_east_north(latitudes=layout.frame[LATITUDE_COL], longitudes=layout.frame[LONGITUDE_COL])
    names = list(layout.frame[NAME_COL])
    order = [names[int(np.argmin(east))]]
    remaining = [n for n in names if n != order[0]]
    while remaining:
        nearest = min(
            remaining,
            key=lambda r: min(float(layout.distance_m[layout.index_of(r), layout.index_of(m)]) for m in order),
        )
        order.append(nearest)
        remaining.remove(nearest)
    return order


def north(inputs: FarmInputs, *, layout: Layout | None, pass_four: bool = True) -> dict[str, pd.DataFrame]:
    """Run ``north_farm`` as a user would: the layout given, and power and wind speed for pass 4."""
    return north_farm(
        inputs.index,
        direction_deg=inputs.direction,
        usable=inputs.usable,
        reanalysis_deg=inputs.reference,
        layout=layout,
        power=inputs.power if pass_four else None,
        wind_speed=inputs.wind_speed if pass_four else None,
    )


def changepoints(table: pd.DataFrame) -> list[tuple[pd.Timestamp, float]]:
    """Each changepoint in a north table as ``(timestamp, signed step in degrees)``."""
    offsets = table["north_offset"].to_numpy(dtype=float)
    return [(table["timestamp"].iloc[i], float(circ_diff(offsets[i], offsets[i - 1]))) for i in range(1, len(table))]


def describe(found: list[tuple[pd.Timestamp, float]]) -> str:
    """A compact, readable rendering of ``changepoints`` for assertion messages."""
    return str([(w.strftime("%Y-%m-%d"), round(s, 1)) for w, s in found])


def offset_at(table: pd.DataFrame, times: pd.DatetimeIndex) -> np.ndarray:
    """Evaluate a north table (a step function of time) at ``times``; before its first row, its first offset."""
    starts = pd.DatetimeIndex(table["timestamp"]).tz_convert("UTC")
    offsets = table["north_offset"].to_numpy(dtype=float)
    idx = np.clip(
        np.searchsorted(starts.asi8, pd.DatetimeIndex(times).tz_convert("UTC").asi8, side="right") - 1, 0, None
    )
    return offsets[idx]


def offset_errors(
    tables: dict[str, pd.DataFrame], reference: dict[str, pd.DataFrame], times: pd.DatetimeIndex
) -> dict[str, float]:
    """Per turbine, the median absolute offset difference (deg) from ``reference`` sampled at ``times``.

    One number that catches both a wrong level and a missed or mistimed changepoint -- the degradation
    study's score.
    """
    return {
        name: float(np.median(np.abs(circ_diff(offset_at(table, times), offset_at(reference[name], times)))))
        for name, table in tables.items()
    }


def monthly(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Where to sample offsets over ``index``: month starts, or six even points in a short window."""
    grid = pd.date_range(index.min().ceil("D"), index.max(), freq="MS")
    if len(grid) >= 4:
        return grid
    return pd.DatetimeIndex(np.linspace(index.min().value, index.max().value, 6).astype("datetime64[ns]")).tz_localize(
        "UTC"
    )
