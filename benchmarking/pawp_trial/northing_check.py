"""How well wind-up's discovered northing makes each turbine agree with the rest of the farm."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yaml

from benchmarking.pawp_trial.read import WORK_DIR
from benchmarking.synthetic.sources import greenbyte as gb


def circ_diff(a: pd.DataFrame | pd.Series, b: pd.DataFrame | pd.Series) -> pd.DataFrame:
    return (a - b + 180) % 360 - 180


def circ_median_excluding(frame: pd.DataFrame) -> pd.DataFrame:
    """Per turbine, the circular median (via median sin/cos) of every other turbine."""
    s, c = np.sin(np.deg2rad(frame)), np.cos(np.deg2rad(frame))
    out = {}
    for t in frame.columns:
        others = frame.columns != t
        out[t] = np.rad2deg(np.arctan2(s.loc[:, others].median(axis=1), c.loc[:, others].median(axis=1))) % 360
    return pd.DataFrame(out)


def main(name: str) -> None:
    pd.set_option("display.width", 250)
    run_dir = WORK_DIR / name
    scada = pd.read_parquet(run_dir / "scada.parquet")
    raw = scada.pivot_table(index=scada.index, columns=gb.TURBINE, values=gb.NACELLE_POSITION, aggfunc="first")
    raw.index = raw.index - pd.Timedelta(minutes=10)  # period start, as wind-up indexes it
    running = scada.pivot_table(index=scada.index, columns=gb.TURBINE, values=gb.POWER, aggfunc="first").gt(50)
    running.index = raw.index
    raw = raw.where(running)
    corr = yaml.safe_load((run_dir / "out/A_data_preparation/04_northing/northing_corrections.yaml").read_text())
    entries = corr if isinstance(corr, list) else next(iter(corr.values()))
    offset = pd.DataFrame(np.nan, index=raw.index, columns=raw.columns)
    for turbine, when, deg in entries:
        offset.loc[pd.Timestamp(when, tz="UTC") :, turbine] = deg
    offset = offset.ffill().fillna(0.0)
    fixed = (raw + offset) % 360

    def score(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        res = circ_diff(frame, circ_median_excluding(frame))
        days = res.resample("14D").median()
        absd = days.abs()
        summary = pd.DataFrame(
            {
                "median_abs_dev": absd.median(),
                "share_14d_gt5": (absd > 5).mean(),
                "share_14d_gt10": (absd > 10).mean(),
                "worst_14d": absd.max(),
            }
        )
        print(
            f"\n{label}: farm median |14-day residual| {float(absd.stack().median()):.2f} deg; "
            f"14-day windows > 5 deg: {float((absd > 5).stack().mean()):.3f}, > 10 deg: {float((absd > 10).stack().mean()):.3f}"
        )
        return summary

    before = score(raw, "raw nacelle position")
    after = score(fixed, "after wind-up northing")
    both = before.add_suffix("_raw").join(after.add_suffix("_fixed"))
    print("\nturbines still off after correction (any 14-day window > 5 deg):")
    print(both[both.share_14d_gt5_fixed > 0].round(2).sort_values("worst_14d_fixed", ascending=False).to_string())
    both.to_csv(run_dir / "northing_check.csv")


if __name__ == "__main__":
    main(sys.argv[1])
