"""Score wind-up's northing with and without the per-turbine wake-nadir shift on a PAWP campaign."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import benchmarking.harness.northing as harness_northing
from benchmarking.campaigns.composed import _fetch_era5
from benchmarking.campaigns.loader import load_declaration
from benchmarking.campaigns.preparation import prepare_data
from benchmarking.diagnostics.context import infer_timebase
from benchmarking.harness.reanalysis import ERA5_WD_RAW, normalise_timestamps
from wind_up.northing import apply_north_table

captured: dict = {}
_original = harness_northing.add_wake_nadir_shift


def _capturing(tables, **kwargs):  # noqa: ANN001, ANN003, ANN202
    shifted, deltas = _original(tables, **kwargs)
    captured.update(unshifted=tables, shifted=shifted, deltas=deltas, index=kwargs["index"])
    return shifted, deltas


def circ(x: np.ndarray) -> np.ndarray:
    return (x + 180) % 360 - 180


def circ_median(stack: np.ndarray) -> np.ndarray:
    r = np.deg2rad(stack)
    return np.rad2deg(np.arctan2(np.nanmedian(np.sin(r), axis=0), np.nanmedian(np.cos(r), axis=0))) % 360


def score(directions: dict[str, np.ndarray], *, era5: np.ndarray, index: pd.DatetimeIndex, label: str) -> pd.DataFrame:
    names = sorted(directions)
    stack = np.vstack([directions[n] for n in names])
    rows = {}
    for i, name in enumerate(names):
        others = circ_median(np.delete(stack, i, axis=0))
        res = pd.Series(circ(stack[i] - others), index=index)
        fortnight = res.resample("14D").median().abs()
        rows[name] = {
            "farm_dev_median": float(fortnight.median()),
            "farm_dev_gt5": float((fortnight > 5).mean()),
            "era5_offset": float(np.nanmedian(circ(stack[i] - era5))),
        }
    out = pd.DataFrame(rows).T
    print(
        f"{label:28s} farm |dev| median {out.farm_dev_median.median():5.2f} deg, 14-day windows >5 deg "
        f"{out.farm_dev_gt5.mean():.3f}; offset to ERA5: farm median {out.era5_offset.median():+5.2f}, "
        f"spread (IQR) {out.era5_offset.quantile(0.75) - out.era5_offset.quantile(0.25):5.2f} deg"
    )
    return out


def main(campaign: str) -> None:
    harness_northing.add_wake_nadir_shift = _capturing
    declaration = load_declaration(campaign)
    cache = Path(campaign).parent / "nadir_capture.pkl"
    if cache.exists():
        captured.update(pickle.loads(cache.read_bytes()))
    else:
        scada = pd.read_parquet(declaration.scada_path)
        timebase = infer_timebase(pd.DatetimeIndex(scada.index.unique()).sort_values())
        scada = normalise_timestamps(scada, timestamps=declaration.timestamps, timebase=timebase)
        index = pd.DatetimeIndex(scada.index.unique()).sort_values()
        prepared = prepare_data(
            scada,
            spec=declaration.spec,
            columns=declaration.columns,
            era5_hourly_df=_fetch_era5(declaration, index=index),
            operating_state=declaration.operating_state,
            cell_selection=declaration.cell_selection,
            out_dir=None,
        )
        captured["era5"] = prepared.aligned_era5[ERA5_WD_RAW].reindex(captured["index"]).to_numpy(dtype=float)
        captured["direction"] = {
            w: g[declaration.columns.nacelle_position].reindex(captured["index"]).to_numpy(dtype=float)
            for w, g in scada.groupby(declaration.columns.turbine)
        }
        captured["running"] = {
            w: (g[declaration.columns.active_power].reindex(captured["index"]) > 0.05 * 2000).to_numpy()
            for w, g in scada.groupby(declaration.columns.turbine)
        }
        cache.write_bytes(pickle.dumps(captured))

    index, deltas = captured["index"], captured["deltas"]
    era5 = captured["era5"]
    farm_shift = float(np.median(list(deltas.values())))
    print(f"per-turbine shifts: median {farm_shift:+.2f}, min {min(deltas.values()):+.2f}, max {max(deltas.values()):+.2f}")

    def northed(tables: dict[str, pd.DataFrame], extra: float = 0.0) -> dict[str, np.ndarray]:
        out = {}
        for w, table in tables.items():
            d = apply_north_table(index, captured["direction"][w], north_table=table) + extra
            out[w] = np.where(captured["running"][w], d % 360, np.nan)
        return out

    variants = {
        "consensus only": northed(captured["unshifted"]),
        "per-turbine nadir (current)": northed(captured["shifted"]),
        "one farm-wide nadir shift": northed(captured["unshifted"], farm_shift),
    }
    results = {k: score(v, era5=era5, index=index, label=k) for k, v in variants.items()}
    table = pd.concat({k: v for k, v in results.items()}, axis=1)
    table[("delta", "")] = pd.Series(deltas)
    table.round(2).to_csv(Path(campaign).parent / "nadir_variants.csv")


if __name__ == "__main__":
    main(sys.argv[1])
