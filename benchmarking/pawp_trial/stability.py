"""Year-to-year stability of each turbine's nacelle power curve and power relative to the farm."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarking.pawp_trial.read import WORK_DIR, read_wide


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    p = read_wide("Power_mean")
    ws = read_wide("WindSpeed_mean")
    st = read_wide("OperatingState")
    running = st.ge(0.5) & st.le(3.5)
    p_run, ws_run = p.where(running), ws.where(running)

    # 1. Mean power in the 7-9 m/s nacelle wind speed band, per turbine per half-year.
    band = ws_run.ge(7) & ws_run.lt(9)
    half = p.index.year.astype(str) + "H" + np.where(p.index.month <= 6, "1", "2")
    pc = p_run.where(band).groupby(half).mean()
    rel = pc.div(pc.median(axis=1), axis=0)
    print("Power in 7-9 m/s band relative to farm median (per half-year): turbines with any deviation > 5% after 2009")
    sub = rel.loc["2010H1":]
    print(sub.loc[:, (sub.sub(1).abs() > 0.05).any()].round(3).T.to_string())
    rel.to_csv(WORK_DIR / "pc_band_rel_by_half.csv")

    # 2. Farm median of the 7-9 m/s band power over time (controller update 2015-10-08).
    print("\nFarm-median power in 7-9 m/s band by half-year:")
    print(pc.median(axis=1).round(1).to_dict())
    month = p.index.to_period("M")
    pcm = p_run.where(band).groupby(month).mean().median(axis=1)
    print("\nMonthly farm-median power (7-9 m/s) Jun 2015 - Mar 2016:")
    print(pcm.loc["2015-06":"2016-03"].round(1).to_dict())
    # cut-in: share of running records at 3-4 m/s, and mean power, before/after
    lo = ws_run.ge(3) & ws_run.lt(4)
    for lab, sl in [("2014-10..2015-09", slice("2014-10-01", "2015-09-30")), ("2015-10-15..2016-09", slice("2015-10-15", "2016-09-30"))]:
        print(f"{lab}: mean power at 3-4 m/s running = {float(np.nanmean(p_run.where(lo).loc[sl].to_numpy())):.1f} kW, "
              f"share running at 3-4 m/s = {float(running.where(ws.ge(3) & ws.lt(4)).loc[sl].stack().mean()):.3f}")

    # 3. Nacelle wind speed relative to farm median by half-year (anemometer / transfer-function steps).
    wsr = ws_run.where(ws_run.gt(4)).groupby(half).mean()
    wsrel = wsr.div(wsr.median(axis=1), axis=0)
    sub = wsrel.loc["2010H1":]
    print("\nNacelle wind speed (running, >4 m/s) relative to farm median: turbines with half-year swing > 4%")
    swing = sub.max() - sub.min()
    print(sub.loc[:, swing > 0.04].round(3).T.to_string())
    wsrel.to_csv(WORK_DIR / "ws_rel_by_half.csv")

    # 4. Energy ratio to farm median (power only, all records with data) by half-year.
    er = p.groupby(half).mean()
    errel = er.div(er.median(axis=1), axis=0).loc["2010H1":]
    print("\nPower relative to farm median, max-min swing over half-years (top 10):")
    print((errel.max() - errel.min()).sort_values(ascending=False).head(10).round(3).to_dict())


if __name__ == "__main__":
    main()
