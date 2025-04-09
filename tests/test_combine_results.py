from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wind_up.combine_results import (
    _CombinedResultsCols,
    calculate_total_uplift_of_test_and_ref_turbines,
    combine_results,
)


def calc_expected_combine_results(trdf: pd.DataFrame) -> pd.DataFrame:
    p50_uplift = (trdf["uplift_frc"] * (1 / trdf["unc_one_sigma_frc"] ** 2)).sum() / (
        1 / trdf["unc_one_sigma_frc"] ** 2
    ).sum()
    sigma_correlated = (1 / trdf["unc_one_sigma_frc"]).sum() / (1 / trdf["unc_one_sigma_frc"] ** 2).sum()
    sigma_independent = (
        (
            (
                trdf["unc_one_sigma_frc"]
                * 1
                / (trdf["unc_one_sigma_frc"] ** 2)
                / (1 / (trdf["unc_one_sigma_frc"] ** 2)).sum()
            )
            ** 2
        ).sum()
    ) ** 0.5
    sigma_test = (sigma_independent + sigma_correlated) / 2
    return pd.DataFrame(
        data={
            "test_wtg": ["test1"],
            "p50_uplift": [p50_uplift],
            "sigma": [sigma_test],
            "sigma_test": [sigma_test],
            "sigma_uncorr": [sigma_independent],
            "sigma_corr": [sigma_correlated],
        },
    )


def test_combine_two_refs() -> None:
    trdf = pd.DataFrame(
        data={
            "test_wtg": ["test1", "test1"],
            "ref": ["ref1", "ref2"],
            "uplift_frc": [0.02, 0.02],
            "unc_one_sigma_frc": [0.02, 0.02],
        },
    )
    edf = calc_expected_combine_results(trdf)
    tdf = combine_results(trdf=trdf, auto_choose_refs=False)
    tdf = tdf[edf.columns.tolist()]
    assert_frame_equal(edf, tdf)


def test_combine_three_refs() -> None:
    trdf = pd.DataFrame(
        data={
            "test_wtg": ["test1", "test1", "test1"],
            "ref": ["ref1", "ref2", "ref3"],
            "uplift_frc": [0.01, 0.02, 0.03],
            "unc_one_sigma_frc": [0.03, 0.02, 0.01],
        },
    )
    edf = calc_expected_combine_results(trdf)

    tdf = combine_results(trdf=trdf, auto_choose_refs=False)

    tdf = tdf[edf.columns.tolist()]
    assert_frame_equal(edf, tdf)


def test_brt_t16_pitch() -> None:
    trdf = pd.read_csv(Path(__file__).parents[0] / "test_data/trdf_BRT_T16_pitch_Sep23.csv", index_col=0)
    edf = pd.read_csv(Path(__file__).parents[0] / "test_data/tdf_BRT_T16_pitch.csv", index_col=0)
    tdf = combine_results(trdf=trdf)
    tdf = tdf[edf.columns.tolist()]
    assert_frame_equal(edf, tdf)


def test_brt_t16_pitch_no_auto_choose() -> None:
    trdf = pd.read_csv(Path(__file__).parents[0] / "test_data/trdf_BRT_T16_pitch_Sep23.csv", index_col=0)
    edf = pd.read_csv(Path(__file__).parents[0] / "test_data/tdf_BRT_T16_pitch_no_auto_choose.csv", index_col=0)
    tdf = combine_results(trdf=trdf, auto_choose_refs=False)
    tdf = tdf[edf.columns.tolist()]
    assert_frame_equal(edf, tdf)


def test_brt_t16_pitch_exclude_refs() -> None:
    trdf = pd.read_csv(Path(__file__).parents[0] / "test_data/trdf_BRT_T16_pitch_Sep23.csv", index_col=0)
    edf = pd.read_csv(Path(__file__).parents[0] / "test_data/tdf_BRT_T16_pitch_exclude_refs.csv", index_col=0)
    # all but one ref excluded
    tdf = combine_results(trdf=trdf, auto_choose_refs=False, exclude_refs=["BRT_T02", "BRT_T03", "BRT_T04", "BRT_T14"])
    tdf = tdf[edf.columns.tolist()]
    assert_frame_equal(edf, tdf)


class TestTotalTestTurbinesUplift:
    def test_not_enough_test_turbines_to_combine(self) -> None:
        _df = pd.DataFrame(
            data={
                _CombinedResultsCols.test_wtg: ["test1", "ref1", "ref2"],
                _CombinedResultsCols.is_ref: [False, True, True],
                _CombinedResultsCols.p50_uplift: [0.1, 0.0, 0.0],
                _CombinedResultsCols.uncertainty_one_sigma: [0.1, 0.0, 0.0],
            },
        )
        with pytest.raises(ValueError, match="combined_results_df must have more than one turbine"):
            calculate_total_uplift_of_test_and_ref_turbines(_df)

    def test_successfully_calculates_uplift(self) -> None:
        test_turbine_1 = "test_01"
        test_turbine_2 = "test_02"

        _df = pd.DataFrame(
            data={
                _CombinedResultsCols.test_wtg: [test_turbine_1, test_turbine_2, "ref_02", "ref_03", "ref_01"],
                _CombinedResultsCols.is_ref: [False, False, True, True, True],
                _CombinedResultsCols.p50_uplift: [-0.31, 1.5, 0.0, 0.0, 0.0],
                _CombinedResultsCols.uncertainty_one_sigma: [0.75, 0.36, 0.0, 0.0, 0.0],
            },
        )
        actual = calculate_total_uplift_of_test_and_ref_turbines(_df)
        expected = pd.DataFrame(
            [
                pd.Series(
                    {
                        "p50_uplift": 1.161066319895969,
                        "sigma_uncorr": 0.3245483006885785,
                        "sigma_corr": 0.4330299089726917,
                        "wtg_count": 2.0,
                        "wtg_list": f"{test_turbine_1}, {test_turbine_2}",
                        "sigma": 0.3787891048306351,
                        "p95_uplift": 0.5380136869655971,
                        "p5_uplift": 1.7841189528263406,
                    }
                ),
                pd.Series(
                    {
                        "p50_uplift": 0.0,
                        "sigma_uncorr": 0.0,
                        "sigma_corr": 0.0,
                        "wtg_count": 3.0,
                        "wtg_list": "ref_01, ref_02, ref_03",
                        "sigma": 0.0,
                        "p95_uplift": 0.0,
                        "p5_uplift": 0.0,
                    }
                ),
            ],
            index=pd.Index(["test", "ref"], name="role"),
        )

        pd.testing.assert_frame_equal(actual, expected)
