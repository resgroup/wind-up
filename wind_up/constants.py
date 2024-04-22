from pathlib import Path

import pandas as pd

PROJECTROOT_DIR = Path(__file__).parents[1]
CONFIG_DIR = Path(__file__).parents[1] / "config"
TURBINE_DATA_DIR = Path(__file__).parents[1] / "input_data/turbine_data"
REANALYSIS_DIR = Path(__file__).parents[1] / "input_data/reanalysis"
TOGGLE_DIR = Path(__file__).parents[1] / "input_data/toggle"
OUTPUT_DIR = Path(__file__).parents[1] / "output"

RANDOM_SEED = 0
SCATTER_S = 1
SCATTER_ALPHA = 0.2

TIMEBASE_PD_TIMEDELTA = pd.Timedelta("10min")
TIMEBASE_S = TIMEBASE_PD_TIMEDELTA.total_seconds()
ROWS_PER_HOUR = 3600 / TIMEBASE_S
ROWS_PER_DAY = 24 * ROWS_PER_HOUR


class DataColumns:
    turbine_name = "TurbineName"
    active_power_mean = "ActivePowerMean"
    active_power_sd = "ActivePowerSD"
    wind_speed_mean = "WindSpeedMean"
    wind_speed_sd = "WindSpeedSD"
    yaw_angle_mean = "YawAngleMean"
    yaw_angle_min = "YawAngleMin"
    yaw_angle_max = "YawAngleMax"
    pitch_angle_mean = "PitchAngleMean"
    gen_rpm_mean = "GenRpmMean"
    ambient_temp = "AmbientTemp"
    shutdown_duration = "ShutdownDuration"

    @classmethod
    def all(cls: type["DataColumns"]) -> list[str]:
        return [v for k, v in vars(cls).items() if not k.startswith("_") and k != "all"]


HOURS_PER_YEAR = 8766
DEFAULT_AIR_DENSITY = 1.22

TIMESTAMP_COL = "TimeStamp_StartFormat"
RAW_WINDSPEED_COL = "raw_WindSpeedMean"
RAW_POWER_COL = "raw_ActivePowerMean"
RAW_DOWNTIME_S_COL = "raw_ShutdownDuration"
RAW_YAWDIR_COL = "raw_YawAngleMean"

REANALYSIS_WS_COL = "reanalysis_ws"
REANALYSIS_WD_COL = "reanalysis_wd"
WINDFARM_YAWDIR_COL = "wf_yawdir"
