import pandas as pd

from wind_up.constants import TIMESTAMP_COL, TOGGLE_DIR
from wind_up.models import WindUpConfig


def check_toggle_files_exist(cfg: WindUpConfig) -> None:
    if cfg.toggle is not None:
        toggle_df = pd.read_parquet(TOGGLE_DIR / cfg.asset.name / cfg.toggle.toggle_filename)
        if (
            not toggle_df.groupby(level="TurbineName").idxmin()[TIMESTAMP_COL] <= cfg.analysis_first_dt_utc_start
            and toggle_df.groupby(level="TurbineName").idxmax()[TIMESTAMP_COL] >= cfg.analysis_last_dt_utc_start
        ):
            msg = (
                f"toggle file {cfg.toggle.toggle_filename} does not cover the analysis period "
                f"{cfg.analysis_first_dt_utc_start} to {cfg.analysis_last_dt_utc_start}"
            )
            raise RuntimeError(msg)


def check_input_data(cfg: WindUpConfig) -> None:
    print(f"\nchecking input data for {cfg.assessment_name}")
    # see if all the input files needed for this cfg exists
    check_toggle_files_exist(cfg=cfg)
