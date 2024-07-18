from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from wind_up.combine_results import combine_results


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
