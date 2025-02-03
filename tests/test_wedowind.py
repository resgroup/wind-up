"""Tests for the `WeDoWind` example wind farm."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import wind_up
from examples.helpers import setup_logger
from examples.wedowind_example import main_wedowind_analysis
from tests.conftest import TEST_DATA_FLD


@pytest.fixture
def wedowind_expected_pitch_angle_results() -> pd.DataFrame:
    shutil.rmtree(TEST_DATA_FLD / "wedowind/Pitch Angle", ignore_errors=True)
    return pd.DataFrame(
        [
            {
                "preprocess_warning_counts": 0,
                "test_warning_counts": 6,
                "test_ref_warning_counts": 3,
                "test_wtg": "Test",
                "test_pw_col": "test_ActivePowerMean",
                "ref": "Ref",
                "ref_ws_col": "ref_ws_est_blend",
                "uplift_frc": 0.030473380515171976,
                "unc_one_sigma_frc": 0.006068651042272032,
                "uplift_p95_frc": 0.02049133783758799,
                "uplift_p5_frc": 0.04045542319275596,
                "pp_data_coverage": 0.07030065689742294,
                "distance_m": 1727.532358659697,
                "bearing_deg": 80.01879303945559,
                "unc_one_sigma_noadj_frc": 0.004194965381090924,
                "unc_one_sigma_lowerbound_frc": 0.0043787000647438,
                "unc_one_sigma_bootstrap_frc": 0.006068651042272032,
                "ref_max_northing_error_v_reanalysis": np.nan,
                "ref_max_northing_error_v_wf": np.nan,
                "ref_max_ws_drift": np.nan,
                "ref_max_ws_drift_pp_period": np.nan,
                "ref_powercurve_shift": -0.03498701513899527,
                "ref_rpm_shift": 0.0,
                "ref_pitch_shift": 0.0,
                "detrend_pre_r2_improvement": 6.304604371898392e-05,
                "detrend_post_r2_improvement": 5.719700032302821e-05,
                "mean_power_pre": 827.0693190786338,
                "mean_power_post": 884.7969322535646,
                "mean_test_yaw_offset_pre": 0.0,
                "mean_test_yaw_offset_post": 0.0,
                "t_value_one_sigma": 1.000403388430281,
                "missing_bins_unc_scale_factor": 1.0001853241828869,
                "pp_valid_hours_pre": 206.83333333333334,
                "pp_valid_hours_post": 349.66666666666663,
                "pp_valid_hours": 556.5,
                "pp_invalid_bin_count": 8,
                "uplift_noadj_frc": 0.034852080579915776,
                "poweronly_uplift_frc": 0.03616453743718513,
                "reversed_uplift_frc": 0.027407137307697532,
                "reversal_error": -0.0087574001294876,
                "lt_wtg_hours_raw": 3536.3333333333335,
                "lt_wtg_hours_filt": 3536.333333333333,
                "test_max_ws_drift": np.nan,
                "test_max_ws_drift_pp_period": np.nan,
                "test_powercurve_shift": 0.0861442820830347,
                "test_rpm_shift": 0.0,
                "test_pitch_shift": 0.0,
            }
        ],
        index=[0],
    )


@pytest.fixture
def wedowind_expected_vg_results() -> pd.DataFrame:
    shutil.rmtree(TEST_DATA_FLD / "wedowind/Vortex Generator", ignore_errors=True)
    return pd.DataFrame(
        [
            {
                "preprocess_warning_counts": 0,
                "test_warning_counts": 2,
                "test_ref_warning_counts": 2,
                "test_wtg": "Test",
                "test_pw_col": "test_pw_clipped",
                "ref": "Ref",
                "ref_ws_col": "ref_ws_est_blend",
                "uplift_frc": 0.023046823561716365,
                "unc_one_sigma_frc": 0.005119990647449296,
                "uplift_p95_frc": 0.014625188375301775,
                "uplift_p5_frc": 0.03146845874813096,
                "pp_data_coverage": 0.0736418776371308,
                "distance_m": 1727.532358659697,
                "bearing_deg": 80.01879303945559,
                "unc_one_sigma_noadj_frc": 0.004255707445544146,
                "unc_one_sigma_lowerbound_frc": 0.0020446262880886657,
                "unc_one_sigma_bootstrap_frc": 0.005119990647449296,
                "ref_max_northing_error_v_reanalysis": np.nan,
                "ref_max_northing_error_v_wf": np.nan,
                "ref_max_ws_drift": np.nan,
                "ref_max_ws_drift_pp_period": np.nan,
                "ref_powercurve_shift": 0.005226755665862948,
                "ref_rpm_shift": 0.0,
                "ref_pitch_shift": 0.0,
                "detrend_pre_r2_improvement": 9.392745002734237e-05,
                "detrend_post_r2_improvement": 0.00013574805792260225,
                "mean_power_pre": 456.00674788268043,
                "mean_power_post": 492.0387530327367,
                "mean_test_yaw_offset_pre": 0.0,
                "mean_test_yaw_offset_post": 0.0,
                "t_value_one_sigma": 1.0002396357467755,
                "missing_bins_unc_scale_factor": 1.030719576586684,
                "pp_valid_hours_pre": 396.66666666666663,
                "pp_valid_hours_post": 348.0,
                "pp_valid_hours": 744.6666666666666,
                "pp_invalid_bin_count": 12,
                "uplift_noadj_frc": 0.0210021972736277,
                "poweronly_uplift_frc": 0.01271095866103132,
                "reversed_uplift_frc": 0.01680021123720865,
                "reversal_error": 0.004089252576177331,
                "lt_wtg_hours_raw": 3498.6666666666665,
                "lt_wtg_hours_filt": 3498.6666666666665,
                "test_max_ws_drift": np.nan,
                "test_max_ws_drift_pp_period": np.nan,
                "test_powercurve_shift": 0.017597853713183786,
                "test_rpm_shift": 0.0,
                "test_pitch_shift": 0.0,
            }
        ],
        index=[0],
    )


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
def test_wedowind_pitch_angle(wedowind_expected_pitch_angle_results: pd.DataFrame, tmp_path: Path) -> None:
    """Test the WeDoWind Pitch Angle example."""
    cache_dir = TEST_DATA_FLD / "wedowind"
    setup_logger(tmp_path / "analysis.log")
    actual = main_wedowind_analysis(
        analysis_name="Pitch Angle",
        generate_custom_plots=False,
        save_plots=False,
        cache_dir=cache_dir,
        analysis_output_dir=tmp_path,
        bootstrap_runs_override=40,  # to speed up test
    )
    assert actual["wind_up_version"].squeeze() == wind_up.__version__
    pd.testing.assert_frame_equal(
        actual.drop(columns=["time_calculated", "wind_up_version"]), wedowind_expected_pitch_angle_results
    )

    shutil.rmtree(cache_dir / "Pitch Angle", ignore_errors=True)  # clean up cache


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
def test_wedowind_vortex_generators(wedowind_expected_vg_results: pd.DataFrame, tmp_path: Path) -> None:
    """Test the WeDoWind Vortex Generators example."""
    cache_dir = TEST_DATA_FLD / "wedowind"
    setup_logger(tmp_path / "analysis.log")
    actual = main_wedowind_analysis(
        analysis_name="Vortex Generator",
        generate_custom_plots=False,
        save_plots=False,
        cache_dir=cache_dir,
        analysis_output_dir=tmp_path,
        bootstrap_runs_override=40,  # to speed up test
    )
    assert actual["wind_up_version"].squeeze() == wind_up.__version__
    pd.testing.assert_frame_equal(
        actual.drop(columns=["time_calculated", "wind_up_version"]), wedowind_expected_vg_results
    )

    shutil.rmtree(cache_dir / "Vortex Generator", ignore_errors=True)  # clean up
