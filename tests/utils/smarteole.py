from pathlib import Path

import pandas as pd

from examples.smarteole_utils import SmartEoleExtractor
from wind_up.interface import AssessmentInputs
from wind_up.models import Asset, PlotConfig, Toggle, Turbine, WindUpConfig
from wind_up.reanalysis_data import ReanalysisDataset


def _define_wind_up_cfg(analysis_timebase_s: int, analysis_output_dir: Path) -> WindUpConfig:
    wtg_map = {
        f"SMV{i}": Turbine.model_validate(
            {
                "name": f"SMV{i}",
                "turbine_type": {
                    "turbine_type": "Senvion-MM82-2050",
                    "rotor_diameter_m": 82.0,
                    "rated_power_kw": 2050.0,
                    "cutout_ws_mps": 25,
                    "normal_operation_pitch_range": (-10.0, 35.0),
                    "normal_operation_genrpm_range": (250.0, 2000.0),
                    "rpm_v_pw_margin_factor": 0.05,
                    "pitch_to_stall": False,
                },
            }
        )
        for i in range(1, 7 + 1)
    }
    northing_corrections_utc = [
        ("SMV1", pd.Timestamp("2020-02-17 16:30:00+0000"), 5.750994540354649),
        ("SMV2", pd.Timestamp("2020-02-17 16:30:00+0000"), 5.690999999999994),
        ("SMV3", pd.Timestamp("2020-02-17 16:30:00+0000"), 5.558000000000042),
        ("SMV4", pd.Timestamp("2020-02-17 16:30:00+0000"), 5.936999999999996),
        ("SMV5", pd.Timestamp("2020-02-17 16:30:00+0000"), 6.797253350869262),
        ("SMV6", pd.Timestamp("2020-02-17 16:30:00+0000"), 5.030130916842758),
        ("SMV7", pd.Timestamp("2020-02-17 16:30:00+0000"), 4.605999999999972),
    ]

    wd_filter_margin = 3 + 7 * analysis_timebase_s / 600
    return WindUpConfig(
        assessment_name="smarteole_example",
        timebase_s=analysis_timebase_s,
        require_ref_wake_free=True,
        detrend_min_hours=12,
        ref_wd_filter=[207 - wd_filter_margin, 236 + wd_filter_margin],  # steer is from 207-236
        filter_all_test_wtgs_together=True,
        use_lt_distribution=False,
        out_dir=analysis_output_dir,
        test_wtgs=[wtg_map["SMV6"], wtg_map["SMV5"]],
        ref_wtgs=[wtg_map["SMV7"]],
        ref_super_wtgs=[],
        non_wtg_ref_names=[],
        analysis_first_dt_utc_start=pd.Timestamp("2020-02-17 16:30:00+0000"),
        upgrade_first_dt_utc_start=pd.Timestamp("2020-02-17 16:30:00+0000"),
        analysis_last_dt_utc_start=pd.Timestamp("2020-05-25 00:00:00+0000") - pd.Timedelta(seconds=analysis_timebase_s),
        lt_first_dt_utc_start=pd.Timestamp("2020-02-17 16:30:00+0000"),
        lt_last_dt_utc_start=pd.Timestamp("2020-05-25 00:00:00+0000") - pd.Timedelta(seconds=analysis_timebase_s),
        detrend_first_dt_utc_start=pd.Timestamp("2020-02-17 16:30:00+0000"),
        detrend_last_dt_utc_start=pd.Timestamp("2020-05-25 00:00:00+0000") - pd.Timedelta(seconds=analysis_timebase_s),
        years_for_lt_distribution=0,
        years_for_detrend=0,
        ws_bin_width=1.0,
        asset=Asset.model_validate(
            {
                "name": "Sole du Moulin Vieux",
                "wtgs": list(wtg_map.values()),
                "masts_and_lidars": [],
            }
        ),
        northing_corrections_utc=northing_corrections_utc,
        toggle=Toggle.model_validate(
            {
                "name": "wake steering",
                "toggle_file_per_turbine": False,
                "toggle_filename": "SMV_offset_active_toggle_df.parquet",
                "detrend_data_selection": "use_toggle_off_data",
                "pairing_filter_method": "any_within_timedelta",
                "pairing_filter_timedelta_seconds": 3600,
                "toggle_change_settling_filter_seconds": 120,
            }
        ),
    )


def create_config(tmp_path: Path, smarteole_data_dir: Path, cache_dir: Path) -> AssessmentInputs:
    analysis_timebase_s = 600
    reanalysis_file_path = smarteole_data_dir / "ERA5T_50.00N_2.75E_100m_1hr_20200201_20200531.parquet"

    input_data_cache_dir = cache_dir / "test-smarteole"
    input_data_cache_dir.mkdir(exist_ok=True, parents=True)
    smarteole_extractor = SmartEoleExtractor(
        data_dir=smarteole_data_dir,
        analysis_timebase_s=analysis_timebase_s,
    )
    scada_df = smarteole_extractor.unpack_smarteole_scada()
    metadata_df = smarteole_extractor.unpack_smarteole_metadata()
    toggle_df = smarteole_extractor.unpack_smarteole_toggle_data()

    cache_subdir = tmp_path / f"timebase_{analysis_timebase_s}"
    cache_subdir.mkdir(parents=True, exist_ok=True)

    # augment scada_df for plotting reasons
    toggle_df_no_tz = toggle_df.copy()
    toggle_df_no_tz.index = toggle_df_no_tz.index.tz_localize(None)
    scada_df = scada_df.merge(toggle_df_no_tz["yaw_offset_command"], left_index=True, right_index=True, how="left")
    scada_df["yaw_offset_command"] = scada_df["yaw_offset_command"].where(scada_df["TurbineName"] == "SMV6", 0)
    del toggle_df_no_tz

    cfg = _define_wind_up_cfg(analysis_timebase_s=analysis_timebase_s, analysis_output_dir=tmp_path)
    reanalysis_dataset = ReanalysisDataset(
        id="ERA5T_50.00N_2.75E_100m_1hr",
        data=pd.read_parquet(reanalysis_file_path),
    )

    plot_cfg = PlotConfig(show_plots=False, save_plots=True, plots_dir=cfg.out_dir / "plots")
    return AssessmentInputs.from_cfg(
        cfg=cfg,
        plot_cfg=plot_cfg,
        toggle_df=toggle_df,
        scada_df=scada_df,
        metadata_df=metadata_df,
        reanalysis_datasets=[reanalysis_dataset],
        cache_dir=cache_subdir,
    )
