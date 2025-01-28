"""Tests for the `wedowind` example wind farm."""

import numpy as np
import pandas as pd
import pytest

from examples.helpers import setup_logger
from examples.wedowind_example import main as main_wedowind_analysis
from tests.conftest import TEST_DATA_FLD


@pytest.fixture
def wedowind_expected_pitch_angle_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "wind_up_version": "0.1.11",
                "preprocess_warning_counts": 1,
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


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
def test_wedowind_pitch_angle(wedowind_expected_pitch_angle_results: pd.DataFrame) -> None:
    """Test the `wedowind` example wind farm."""

    output_dir = TEST_DATA_FLD / "wedowind"
    setup_logger(output_dir / "analysis.log")

    actual = main_wedowind_analysis(
        "Pitch Angle",
        generate_custom_plots=False,
        save_plots=False,
        cache_dir=output_dir,
        analysis_output_dir=output_dir,
        bootstrap_runs_override=40,  # to speed up test
    )

    pd.testing.assert_frame_equal(actual.drop(columns=["time_calculated"]), wedowind_expected_pitch_angle_results)
