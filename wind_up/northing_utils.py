import pandas as pd

from wind_up.constants import RAW_DOWNTIME_S_COL, RAW_POWER_COL

YAW_OK_PW_FRACTION = 0.05


def add_ok_yaw_col(
    wf_or_wtg_df: pd.DataFrame, *, new_col_name: str, wd_col: str, rated_power: float, timebase_s: int
) -> pd.DataFrame:
    yaw_of_downtime_s = timebase_s * 1 / 4
    wf_or_wtg_df[new_col_name] = (
        wf_or_wtg_df[wd_col].notna()
        & (wf_or_wtg_df[RAW_POWER_COL] > rated_power * YAW_OK_PW_FRACTION)
        & (wf_or_wtg_df[RAW_DOWNTIME_S_COL].fillna(0) < yaw_of_downtime_s)
    )
    return wf_or_wtg_df
