"""First-pass profile of the PAWP files: timestamps, ranges, sentinels, availability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarking.pawp_trial.read import read_wide

VARIABLES = [
    "Power_mean", "Power_min", "Power_max", "Power_std",
    "WindSpeed_mean", "WindSpeed_max", "WindSpeed_std",
    "GeneratorSpeed_mean", "RotorSpeed_mean", "PitchAngle_mean", "PitchAngle_min",
    "NacelleDirection_mean", "WindDirection_mean",
    "AmbientTemperature_mean", "NacelleTemperature_mean", "SpinnerTemperature_mean",
    "OperatingState", "FirstAlarm10min",
]  # fmt: skip


def describe_index(idx: pd.DatetimeIndex) -> str:
    steps = pd.Series(idx[1:] - idx[:-1]).value_counts().head(5)
    return (
        f"n={len(idx)} first={idx[0]} last={idx[-1]} dup={idx.duplicated().sum()} "
        f"monotonic={idx.is_monotonic_increasing} steps={dict(steps)}"
    )


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    frames = {}
    for v in VARIABLES:
        f = read_wide(v)
        frames[v] = f
        vals = f.to_numpy().ravel()
        vals = vals[~np.isnan(vals)]
        q = np.quantile(vals, [0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]) if len(vals) else []
        print(f"\n== {v}: {describe_index(f.index)}")
        print(f"   non-NaN frac={len(vals) / f.size:.4f} quantiles(0,.1%,1%,50%,99%,99.9%,100%)={np.round(q, 3)}")
        top = pd.Series(vals).value_counts().head(6)
        print(f"   most common values: {dict(top.round(3))}")

    p = frames["Power_mean"]
    print("\nPower_mean > 2100 kW count:", int((p > 2100).sum().sum()), " > 3000:", int((p > 3000).sum().sum()))
    for v in ["WindDirection_mean", "NacelleDirection_mean"]:
        f = frames[v]
        print(f"{v}: ==1000 count {int((f == 1000).sum().sum())}, >360 count {int((f > 360).sum().sum())}, <0 {int((f < 0).sum().sum())}")
    st = frames["OperatingState"]
    non_int = (st.notna() & (st != st.round())).sum().sum()
    print(f"OperatingState non-integer values: {int(non_int)} of {int(st.notna().sum().sum())}")
    print("OperatingState value counts:", dict(pd.Series(st.to_numpy().ravel()).value_counts().head(15)))
    al = frames["FirstAlarm10min"]
    print("FirstAlarm value counts:", dict(pd.Series(al.to_numpy().ravel()).value_counts().head(15)))

    # NaN pattern agreement between variables (same 'communication loss' mask?)
    base = p.notna()
    for v in VARIABLES:
        f = frames[v].reindex(index=p.index)
        print(f"  notna agreement Power_mean vs {v}: {(f.notna() == base).to_numpy().mean():.4f}")

    # availability of power by turbine and year
    avail = p.notna().groupby(p.index.year).mean()
    print("\nPower_mean data availability by year (rows) x turbine (cols) — min / median per year:")
    print(avail.T.describe().T[["min", "50%"]].round(3))
    print("\nTurbines with any year < 0.9 (2009-2017):")
    sub = avail.loc[2009:2017]
    print(sub.loc[:, (sub < 0.9).any()].round(2).T)
    avail.to_csv("F:/pawp_work/power_availability_by_year.csv")
    monthly = p.notna().groupby([p.index.year, p.index.month]).mean()
    monthly.to_csv("F:/pawp_work/power_availability_by_month.csv")


if __name__ == "__main__":
    main()
