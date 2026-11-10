"""Check what the operating state, alarm and sentinel values mean in practice."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarking.pawp_trial.read import read_wide


def long(v: str) -> pd.Series:
    return read_wide(v).stack()


def main() -> None:
    pd.set_option("display.width", 200)
    df = pd.concat(
        {v: long(v) for v in ["Power_mean", "WindSpeed_mean", "OperatingState", "FirstAlarm10min", "PitchAngle_mean", "WindDirection_mean", "NacelleDirection_mean"]},
        axis=1,
    )
    df["state_bin"] = np.select(
        [df.OperatingState == 0, df.OperatingState.between(0.5, 3.5), df.OperatingState >= 4],
        ["0 idle", "1-3 run", "4-5 park"],
        "other",
    )
    df["state_raw"] = df.OperatingState.round(3)
    print("\nPower/wind by integer state (and fractional 4.x):")
    print(df.groupby("state_raw").agg(n=("Power_mean", "size"), p_mean=("Power_mean", "mean"), p_gt100=("Power_mean", lambda s: (s > 100).mean()),
                                      ws=("WindSpeed_mean", "mean"), pitch=("PitchAngle_mean", "median"),
                                      alarm_nonzero=("FirstAlarm10min", lambda s: (s > 0).mean())).sort_values("n", ascending=False).head(20).round(3))
    print("\nTop alarm codes: share of records with power > 100 kW, mean power, mean state")
    print(df[df.FirstAlarm10min > 0].groupby("FirstAlarm10min").agg(n=("Power_mean", "size"), p_gt100=("Power_mean", lambda s: (s > 100).mean()),
                                                                     p=("Power_mean", "mean"), state=("OperatingState", "mean")).sort_values("n", ascending=False).head(15).round(3))
    t = df.index.get_level_values(0)
    print("\nWindDirection==1000 share by year:")
    print((df.WindDirection_mean == 1000).groupby(t.year).mean().round(3).to_dict())
    print("NacelleDirection==1000 share by year:")
    print((df.NacelleDirection_mean == 1000).groupby(t.year).mean().round(3).to_dict())
    print("WindDirection==1000 share by turbine (top 10):")
    print((df.WindDirection_mean == 1000).groupby(level=1).mean().sort_values(ascending=False).head(10).round(3).to_dict())
    wd1000 = df.WindDirection_mean == 1000
    print("When WindDirection==1000: power mean / share running:", round(df.Power_mean[wd1000].mean(), 1), round((df.state_bin[wd1000] == "1-3 run").mean(), 3))
    print("\nWind speed > 40 m/s count:", int((df.WindSpeed_mean > 40).sum()), "by year", (df.WindSpeed_mean > 40).groupby(t.year).sum().to_dict())
    print("Power > 2100 by year:", (df.Power_mean > 2100).groupby(t.year).sum().to_dict())
    print("Running (state 1-3) with power < 0 share:", round((df.Power_mean[df.state_bin == "1-3 run"] < 0).mean(), 4))
    print("Parked (state>=4) with power > 500 share:", round((df.Power_mean[df.state_bin == "4-5 park"] > 500).mean(), 4))


if __name__ == "__main__":
    main()
