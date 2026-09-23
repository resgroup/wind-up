"""Golden northing corrections for the open SCADA farms, with wake-nadir bubble-plot diagnostics.

For each farm (Hill of Towie, Kelmarsh, Penmanshiel) this runs the full v1 northing pipeline
(passes 1-2 then the pass-4 wake-nadir nudge) on the whole farm and records the result as the
**golden table** -- the per-turbine, per-changepoint northing corrections, in the same YAML layout
as v0's ``optimized_northing_corrections.yaml`` (a flat list of ``['Txx', <timestamp>, <offset deg>]``
rows). It is the best-available absolute answer, and what small-N / subset / low-data challenges are
later scored against.

It also renders a **bubble plot** of the pass-4 correction per farm -- the layout with each turbine
coloured by, sized by and labelled with its wake-nadir correction -- so a human can eyeball whether
that part is plausible. Pass-4 correctness is proven on synthetic ground truth (the recovery test);
on real data there is no ground-truth absolute northing, so the pass-4 checks here are plausibility
ones (magnitude, spatial smoothness); comparing the golden table to v0's is a review job.

Run it (reads the cached SCADA; ERA5 is fetched per site and cached)::

    uv run python -m benchmarking.baselines.study_wake_nadir_golden

It writes ``tests/test_data/hot/northing/golden_northing_corrections_<farm>.yaml`` per farm and the
bubble plots plus a plausibility summary under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``wake_nadir_golden``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless: render plots without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.harness.northing import era5_direction
from benchmarking.synthetic import HOT_RATED_POWER_KW
from benchmarking.synthetic.sources import greenbyte
from benchmarking.synthetic.sources.hill_of_towie import get_data_dir, load_hot_10min_data, load_hot_metadata
from wind_up.circular_math import circ_diff
from wind_up.geodesy import local_east_north
from wind_up.layout import LATITUDE_COL, LONGITUDE_COL, NAME_COL, Layout
from wind_up.northing import north_farm, write_north_table_yaml, yaw_usable
from wind_up_v0.era5 import get_era5_hourly_df

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
NORTHING_DIR = REPO / "tests" / "test_data" / "hot" / "northing"

TIMEBASE_S = 600.0
# Hill of Towie source-native tags.
HOT_TAGS = {
    "yaw": "wtc_NacelPos_mean",
    "power": "wtc_ActPower_mean",
    "ws": "wtc_AcWindSp_mean",
    "avail": "wtc_ScReToOp_timeon",
}


def default_output_root() -> Path:
    """Return the directory this driver writes plots under (``WIND_UP_BENCHMARKING_OUTPUT_DIR`` overrides)."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "wake_nadir_golden"


def _inputs_from_frames(
    *,
    index: pd.DatetimeIndex,
    reference: np.ndarray,
    direction: dict[str, np.ndarray],
    power: dict[str, np.ndarray],
    wind_speed: dict[str, np.ndarray],
    availability: dict[str, np.ndarray],
    rated_power_kw: float,
) -> dict:
    """Assemble north_farm inputs, deriving the ``yaw_usable`` mask, exactly as ``north_scada`` does."""
    usable = {
        turbine: yaw_usable(
            power=power[turbine],
            downtime_s=TIMEBASE_S - np.nan_to_num(availability[turbine], nan=0.0),
            reference_deg=reference,
            rated_power=rated_power_kw,
            timebase_s=TIMEBASE_S,
        )
        for turbine in direction
    }
    return {
        "index": index,
        "reference": reference,
        "direction": direction,
        "usable": usable,
        "power": power,
        "wind_speed": wind_speed,
    }


def _era5_reference(lat: float, lon: float, *, index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    """Fetch ERA5 for a site and carry its hourly wind direction onto ``index``."""
    era5_df = get_era5_hourly_df(lat=lat, lon=lon, start_date=start, end_date=end)
    return era5_direction(era5_df, index).to_numpy(dtype=float)


def hot_inputs(*, start: str, end: str) -> tuple[Layout, dict]:
    """Return the Hill of Towie layout and north_farm inputs for a window."""
    meta = load_hot_metadata()
    layout = Layout.from_frame(
        pd.DataFrame(
            {
                "name": meta["Name"],
                "latitude": meta["Latitude"],
                "longitude": meta["Longitude"],
                "rotor_diameter_m": 82.0,
                "wind_farm": "Hill of Towie",
            }
        )
    )
    turbines = list(layout.frame["name"])
    scada = load_hot_10min_data(
        data_dir=get_data_dir(),
        wtg_numbers=[int(t[1:]) for t in turbines],
        start_dt=pd.Timestamp(start, tz="UTC"),
        end_dt_excl=pd.Timestamp(end, tz="UTC"),
    )
    idx = scada.index
    index = pd.DatetimeIndex(idx[(idx >= pd.Timestamp(start, tz="UTC")) & (idx < pd.Timestamp(end, tz="UTC"))])
    era5_df = build_hot_v0_context(wtg_names=turbines).reanalysis_datasets[0].data
    reference = era5_direction(era5_df, index).to_numpy(dtype=float)
    pick = lambda tag, t: scada[(t, tag)].reindex(index).to_numpy(dtype=float)  # noqa: E731
    return layout, _inputs_from_frames(
        index=index,
        reference=reference,
        direction={t: pick(HOT_TAGS["yaw"], t) for t in turbines},
        power={t: pick(HOT_TAGS["power"], t) for t in turbines},
        wind_speed={t: pick(HOT_TAGS["ws"], t) for t in turbines},
        availability={t: pick(HOT_TAGS["avail"], t) for t in turbines},
        rated_power_kw=HOT_RATED_POWER_KW,
    )


def greenbyte_inputs(
    farm: greenbyte.GreenbyteFarm, *, rotor_diameter_m: float, start: str, end: str
) -> tuple[Layout, dict]:
    """Return a Greenbyte farm's (Kelmarsh / Penmanshiel) layout and north_farm inputs for a window."""
    meta = greenbyte.load_greenbyte_metadata(farm)
    layout = Layout.from_frame(
        pd.DataFrame(
            {
                "name": meta["Name"],
                "latitude": meta["Latitude"],
                "longitude": meta["Longitude"],
                "rotor_diameter_m": rotor_diameter_m,
                "wind_farm": farm.name,
            }
        )
    )
    turbines = list(layout.frame["name"])
    years = [
        y
        for y in farm.years
        if pd.Timestamp(f"{y}-01-01", tz="UTC") < pd.Timestamp(end, tz="UTC")
        and pd.Timestamp(f"{y + 1}-01-01", tz="UTC") > pd.Timestamp(start, tz="UTC")
    ]
    scada = greenbyte.load_greenbyte_scada(farm, years=years)
    scada = scada[(scada.index >= pd.Timestamp(start, tz="UTC")) & (scada.index < pd.Timestamp(end, tz="UTC"))]
    index = pd.DatetimeIndex(sorted(scada.index.unique()))
    reference = _era5_reference(
        float(meta["Latitude"].mean()), float(meta["Longitude"].mean()), index=index, start=start, end=end
    )
    columns = greenbyte.GREENBYTE_COLUMNS

    def by_turbine(col: str) -> dict[str, np.ndarray]:
        out = {}
        for turbine in turbines:
            rows = scada[scada[columns.turbine] == turbine]
            out[turbine] = rows[~rows.index.duplicated()].reindex(index)[col].to_numpy(dtype=float)
        return out

    return layout, _inputs_from_frames(
        index=index,
        reference=reference,
        direction=by_turbine(columns.nacelle_position),
        power=by_turbine(columns.active_power),
        wind_speed=by_turbine(columns.wind_speed),
        availability=by_turbine(columns.availability),
        rated_power_kw=farm.rated_power_kw,
    )


def golden_tables(layout: Layout, inputs: dict) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Return the golden north tables (full pipeline) and the pass-4 correction per turbine.

    Norths the farm twice on the same inputs -- without pass 4 and with it -- so the golden tables are
    the pass-4-included result and the correction is the shift pass 4 added to each turbine.
    """
    common = {
        "direction_deg": inputs["direction"],
        "usable": inputs["usable"],
        "reanalysis_deg": inputs["reference"],
        "layout": layout,
    }
    without = north_farm(inputs["index"], **common)
    with_pass4 = north_farm(inputs["index"], power=inputs["power"], wind_speed=inputs["wind_speed"], **common)
    deltas = {
        t: float(circ_diff(with_pass4[t]["north_offset"].iloc[0], without[t]["north_offset"].iloc[0])) for t in without
    }
    return with_pass4, deltas


def _layout_xy(layout: Layout) -> dict[str, tuple[float, float]]:
    """Easting and northing in metres per turbine, minimum easting and northing at 0."""
    east, north = local_east_north(latitudes=layout.frame[LATITUDE_COL], longitudes=layout.frame[LONGITUDE_COL])
    names = list(layout.frame[NAME_COL])
    return {name: (float(east[i]), float(north[i])) for i, name in enumerate(names)}


# Fixed diverging colour range so a farm whose corrections sit near 0 shows no extreme colours.
COLOUR_LIMIT_DEG = 10.0


def bubble_plot(layout: Layout, deltas: dict[str, float], *, title: str, save_path: Path) -> None:
    """Render the pass-4 wake-nadir correction across a farm layout, one bubble per turbine."""
    names = list(layout.frame[NAME_COL])
    xy = _layout_xy(layout)
    xs = [xy[t][0] for t in names]
    ys = [xy[t][1] for t in names]
    vals = np.array([deltas[t] for t in names])
    norm = TwoSlopeNorm(vmin=-COLOUR_LIMIT_DEG, vcenter=0.0, vmax=COLOUR_LIMIT_DEG)

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    sizes = 300 + 900 * np.clip(np.abs(vals) / COLOUR_LIMIT_DEG, 0.0, 1.0)
    scatter = ax.scatter(
        xs, ys, c=vals, s=sizes, cmap=plt.get_cmap("RdBu_r"), norm=norm, edgecolors="k", linewidths=0.6, zorder=3
    )
    for turbine in names:
        ax.annotate(f"{turbine}\n{deltas[turbine]:+.1f}°", xy[turbine], ha="center", va="center", fontsize=8, zorder=4)
    ax.set_aspect("equal")
    ax.margins(0.08)
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.grid(visible=True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("pass-4 wake-nadir correction [deg]")
    ax.set_title(
        f"{title} — pass-4 wake-nadir correction\n"
        f"max |Δ| {np.abs(vals).max():.1f}°, mean {np.abs(vals).mean():.1f}°"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _neighbour_spread(layout: Layout, deltas: dict[str, float]) -> list[float]:
    """Max |delta - neighbour delta| over each turbine's four nearest, for a smoothness read."""
    names = list(layout.frame["name"])
    spreads = []
    for turbine in names:
        i = layout.index_of(turbine)
        near = [
            n
            for _, n in sorted((float(layout.distance_m[i, layout.index_of(n)]), n) for n in names if n != turbine)[:4]
        ]
        if near:
            spreads.append(max(abs(float(circ_diff(deltas[turbine], deltas[n]))) for n in near))
    return spreads


def plausibility(layout: Layout, deltas: dict[str, float]) -> None:
    """Log the pass-4 magnitude and neighbour smoothness for a farm."""
    mags = np.array([abs(v) for v in deltas.values()])
    spreads = np.array(_neighbour_spread(layout, deltas))
    logger.info(
        "  pass-4 |delta|: max %.2f mean %.2f ; neighbour spread max %.2f mean %.2f",
        mags.max(),
        mags.mean(),
        spreads.max(),
        float(spreads.mean()),
    )


@dataclass(frozen=True)
class FarmRun:
    """One farm to build a golden table for."""

    slug: str
    title: str
    window: tuple[str, str]
    inputs: Callable[[], tuple[Layout, dict]]


def farm_runs() -> list[FarmRun]:
    """Return the farms this driver builds golden tables for, with their windows."""
    hot_window = ("2016-01-01", "2021-01-01")  # the full cached Hill of Towie record
    gb_window = ("2017-01-01", "2019-01-01")  # Kelmarsh and Penmanshiel: the cached 2017-2018
    return [
        FarmRun(
            "hill_of_towie", "Hill of Towie", hot_window, lambda: hot_inputs(start=hot_window[0], end=hot_window[1])
        ),
        FarmRun(
            "kelmarsh",
            "Kelmarsh",
            gb_window,
            lambda: greenbyte_inputs(greenbyte.KELMARSH, rotor_diameter_m=92.0, start=gb_window[0], end=gb_window[1]),
        ),
        FarmRun(
            "penmanshiel",
            "Penmanshiel",
            gb_window,
            lambda: greenbyte_inputs(
                greenbyte.PENMANSHIEL, rotor_diameter_m=82.0, start=gb_window[0], end=gb_window[1]
            ),
        ),
    ]


def main() -> None:
    """Generate each farm's golden northing table, wake-nadir bubble plot and plausibility summary."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = default_output_root()
    for run in farm_runs():
        logger.info("=== %s (%s..%s) ===", run.title, *run.window)
        layout, inputs = run.inputs()
        tables, deltas = golden_tables(layout, inputs)
        plausibility(layout, deltas)
        path = NORTHING_DIR / f"golden_northing_corrections_{run.slug}.yaml"
        write_north_table_yaml(tables, path=path)
        logger.info("  wrote golden table %s", path)
        bubble_plot(layout, deltas, title=run.title, save_path=out / f"wake_nadir_bubbles_{run.slug}.png")


if __name__ == "__main__":
    main()
