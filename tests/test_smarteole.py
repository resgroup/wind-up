"""Tests running the Smarteole dataset through wind-up analysis."""

import copy

import pandas as pd
import pytest

from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis


def _prep_results(input_df: pd.DataFrame, dp: int = 1) -> pd.DataFrame:
    input_df = input_df[
        [
            "test_wtg",
            "ref",
            "uplift_frc",
            "unc_one_sigma_frc",
            "uplift_p95_frc",
            "uplift_p5_frc",
            "pp_valid_hours_pre",
            "pp_valid_hours_post",
            "mean_power_post",
        ]
    ]
    for i, col in enumerate(x for x in input_df.columns if x.endswith("_frc")):
        if i == 0:
            output_df = input_df.assign(**{col: (input_df[col] * 100).round(dp).astype(str) + "%"})
        else:
            output_df = output_df.assign(**{col: (input_df[col] * 100).round(dp).astype(str) + "%"})
        output_df = output_df.rename(columns={col: col.replace("_frc", "_pct")})
    output_df["mean_power_post"] = output_df["mean_power_post"].round(0).astype("int64")
    return output_df.rename(
        columns={
            "test_wtg": "turbine",
            "ref": "reference",
            "uplift_pct": "energy uplift",
            "unc_one_sigma_pct": "uplift uncertainty",
            "uplift_p95_pct": "uplift P95",
            "uplift_p5_pct": "uplift P5",
            "pp_valid_hours_pre": "valid hours toggle off",
            "pp_valid_hours_post": "valid hours toggle on",
            "mean_power_post": "mean power toggle on",
        }
    )


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
def test_smarteole_analysis(smarteole_assessment_inputs: AssessmentInputs) -> None:
    """Test running the Smarteole analysis."""

    expected_print_df = pd.DataFrame(
        {
            "turbine": ["SMV6", "SMV5"],
            "reference": ["SMV7", "SMV7"],
            "energy uplift": ["-1.0%", "3.1%"],
            "uplift uncertainty": ["0.6%", "1.2%"],
            "uplift P95": ["-2.0%", "1.2%"],
            "uplift P5": ["-0.1%", "5.0%"],
            "valid hours toggle off": [132 + 3 / 6, 133 + 0 / 6],
            "valid hours toggle on": [136 + 0 / 6, 137 + 1 / 6],
            "mean power toggle on": [1148, 994],
        },
        index=[0, 1],
    )

    assessment_inputs = copy.deepcopy(smarteole_assessment_inputs)
    results_per_test_ref_df = run_wind_up_analysis(assessment_inputs)
    actual_df = _prep_results(results_per_test_ref_df)

    pd.testing.assert_frame_equal(actual_df, expected_print_df)
