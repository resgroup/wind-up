"""Tests for _cook_pp in pp_analysis.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_up.constants import DataColumns
from wind_up.pp_analysis import _cook_pp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WS_BIN_WIDTH = 1.0
RATED_POWER = 1300.0  # kW
CUTOUT_WS = 25.0


def _make_pp_raw_df(
    bin_mids: list[float], pw_means: list[float | None], hours: list[float], pre_or_post: str = "pre"
) -> pd.DataFrame:
    """Build a minimal pp_df as would be produced by _pp_raw_df."""
    n = len(bin_mids)
    assert len(pw_means) == n
    assert len(hours) == n

    pw_col = f"pw_mean_{pre_or_post}"
    ws_col = f"ws_mean_{pre_or_post}"
    hours_col = f"hours_{pre_or_post}"
    pw_std_col = f"pw_std_{pre_or_post}"
    ws_std_col = f"ws_std_{pre_or_post}"
    pw_sem_col = f"pw_sem_{pre_or_post}"
    ws_sem_col = f"ws_sem_{pre_or_post}"
    count_col = f"count_{pre_or_post}"

    counts = [max(1, round(h * 3600 / 600)) for h in hours]  # assume 10-min timebase

    df = pd.DataFrame(
        {
            "bin_mid": bin_mids,
            "bin_left": [m - WS_BIN_WIDTH / 2 for m in bin_mids],
            "bin_right": [m + WS_BIN_WIDTH / 2 for m in bin_mids],
            "bin_closed_right": [True] * n,
            pw_col: pw_means,
            ws_col: bin_mids,
            hours_col: hours,
            pw_std_col: [50.0 if p is not None else np.nan for p in pw_means],
            ws_std_col: [0.2] * n,
            count_col: counts,
            pw_sem_col: [50.0 / max(1, c) ** 0.5 if p is not None else np.nan for p, c in zip(pw_means, counts)],
            ws_sem_col: [0.2 / max(1, c) ** 0.5 for c in counts],
        }
    )
    return df.set_index("bin_mid", drop=False, verify_integrity=True)


def _make_site_mean_pc_df(bin_mids: list[float] | None = None, rated_power: float = RATED_POWER) -> pd.DataFrame:
    """Build a simple site mean power curve DataFrame."""
    if bin_mids is None:
        bin_mids = list(np.arange(0.5, CUTOUT_WS + 0.5, 1.0))

    def _simple_pc(ws: float) -> float:
        if ws < 3.0:
            return 0.0
        if ws >= 12.0:
            return rated_power
        # linear ramp from 3 to 12 m/s
        return rated_power * (ws - 3.0) / (12.0 - 3.0)

    pw_clipped = [_simple_pc(m) for m in bin_mids]
    return pd.DataFrame({"bin_mid": bin_mids, DataColumns.wind_speed_mean: bin_mids, "pw_clipped": pw_clipped})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_data_pp_df() -> pd.DataFrame:
    """pp_df with enough data in every bin to be valid."""
    bin_mids = list(range(1, 26))  # 1..25 m/s

    # simple power curve: 0 below 3, ramp 3-12, rated above 12
    def pc(ws: float) -> float:
        if ws < 3:
            return 0.0
        if ws >= 12:
            return RATED_POWER
        return RATED_POWER * (ws - 3) / 9

    pw_means = [pc(m) for m in bin_mids]
    hours = [50.0] * len(bin_mids)  # well above validity threshold
    return _make_pp_raw_df(bin_mids, pw_means, hours)


@pytest.fixture
def sparse_high_ws_pp_df() -> pd.DataFrame:
    """pp_df where data only reaches 700 kW (at ~8 m/s), sparse above that."""
    bin_mids = list(range(1, 26))

    def pc(ws: float) -> float:
        if ws < 3:
            return 0.0
        if ws >= 8:
            return 700.0
        return 700.0 * (ws - 3) / 5

    pw_means = [pc(m) for m in bin_mids]
    # bins above 8 m/s have insufficient data (below validity threshold)
    hours = [50.0 if m <= 8 else 1.0 for m in bin_mids]
    return _make_pp_raw_df(bin_mids, pw_means, hours)


# ---------------------------------------------------------------------------
# Basic output structure tests
# ---------------------------------------------------------------------------


class TestCookPpOutputStructure:
    def test_returns_dataframe(self, full_data_pp_df: pd.DataFrame) -> None:
        result = _cook_pp(
            full_data_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self, full_data_pp_df: pd.DataFrame) -> None:
        result = _cook_pp(
            full_data_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
        )
        assert "pw_mean_pre" in result.columns
        assert "pw_at_mid_pre" in result.columns
        assert "pw_sem_at_mid_pre" in result.columns
        assert "pre_valid" in result.columns
        assert "pw_mean_pre_raw" in result.columns

    def test_does_not_mutate_input(self, full_data_pp_df: pd.DataFrame) -> None:
        original = full_data_pp_df.copy()
        _cook_pp(
            full_data_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
        )
        pd.testing.assert_frame_equal(full_data_pp_df, original)

    def test_no_nans_in_output(self, full_data_pp_df: pd.DataFrame) -> None:
        result = _cook_pp(
            full_data_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
        )
        key_cols = ["pw_mean_pre", "pw_at_mid_pre", "pw_sem_at_mid_pre"]
        assert not result[key_cols].isna().any().any()


# ---------------------------------------------------------------------------
# Validity flagging tests
# ---------------------------------------------------------------------------


class TestValidityFlagging:
    def test_bins_with_sufficient_hours_are_valid(self, full_data_pp_df: pd.DataFrame) -> None:
        result = _cook_pp(
            full_data_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
        )
        # With 50 hours per bin and bin_width=1, threshold is 3 hrs -> all valid
        assert result["pre_valid"].all()

    def test_bins_below_hours_threshold_are_invalid(self, sparse_high_ws_pp_df: pd.DataFrame) -> None:
        result = _cook_pp(
            sparse_high_ws_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
        )
        # bins above 8 m/s have only 1 hour (< 3 hour threshold)
        assert not result.loc[result["bin_mid"] > 8, "pre_valid"].any()

    def test_raw_pw_col_preserves_original_values(self, sparse_high_ws_pp_df: pd.DataFrame) -> None:
        result = _cook_pp(
            sparse_high_ws_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
        )
        # raw column should be unchanged even where bins are invalid
        assert result["pw_mean_pre_raw"].notna().any()


# ---------------------------------------------------------------------------
# clip_to_rated tests
# ---------------------------------------------------------------------------


class TestClipToRated:
    def test_clip_to_rated_caps_pw_mean(self, full_data_pp_df: pd.DataFrame) -> None:
        result = _cook_pp(
            full_data_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=True,
        )
        assert (result["pw_mean_pre"] <= RATED_POWER).all()

    def test_clip_to_rated_caps_pw_at_mid(self, full_data_pp_df: pd.DataFrame) -> None:
        result = _cook_pp(
            full_data_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=True,
        )
        assert (result["pw_at_mid_pre"] <= RATED_POWER).all()

    def test_no_clip_allows_values_at_rated(self, full_data_pp_df: pd.DataFrame) -> None:
        result = _cook_pp(
            full_data_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
        )
        assert (result["pw_at_mid_pre"] >= 0).all()

    def test_power_clipped_to_zero_at_low_ws(self, full_data_pp_df: pd.DataFrame) -> None:
        result = _cook_pp(
            full_data_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
        )
        assert (result["pw_mean_pre"] >= 0).all()


# ---------------------------------------------------------------------------
# site_mean_pc_df gap-filling tests
# ---------------------------------------------------------------------------


class TestSiteMeanPcGapFilling:
    def test_without_site_mean_invalid_bins_clip_at_max_measured(self, sparse_high_ws_pp_df: pd.DataFrame) -> None:
        result = _cook_pp(
            sparse_high_ws_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
        )
        # Without site mean curve, high-ws bins should be filled at ~700 kW
        high_ws_pw = result.loc[result["bin_mid"] > 12, "pw_at_mid_pre"]
        assert (high_ws_pw <= 700.0 * 1.05).all()  # allow small tolerance

    def test_with_site_mean_invalid_bins_reach_rated_power(self, sparse_high_ws_pp_df: pd.DataFrame) -> None:
        site_mean_pc_df = _make_site_mean_pc_df()
        result = _cook_pp(
            sparse_high_ws_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
            site_mean_pc_df=site_mean_pc_df,
        )
        # With site mean curve, bins at rated ws should reach ~1300 kW
        high_ws_pw = result.loc[result["bin_mid"] >= 12, "pw_at_mid_pre"]
        assert (high_ws_pw >= RATED_POWER * 0.95).all()

    def test_with_site_mean_valid_bins_are_unchanged(self, sparse_high_ws_pp_df: pd.DataFrame) -> None:
        site_mean_pc_df = _make_site_mean_pc_df()
        result_with = _cook_pp(
            sparse_high_ws_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
            site_mean_pc_df=site_mean_pc_df,
        )
        result_without = _cook_pp(
            sparse_high_ws_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
            site_mean_pc_df=None,
        )
        # valid bins (<=8 m/s) should be identical regardless of site_mean_pc_df
        valid_mask = result_with["pre_valid"]
        pd.testing.assert_series_equal(
            result_with.loc[valid_mask, "pw_at_mid_pre"],
            result_without.loc[valid_mask, "pw_at_mid_pre"],
        )

    def test_site_mean_none_behaviour_unchanged(self, full_data_pp_df: pd.DataFrame) -> None:
        result_none = _cook_pp(
            full_data_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
            site_mean_pc_df=None,
        )
        result_no_arg = _cook_pp(
            full_data_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
        )
        pd.testing.assert_frame_equal(result_none, result_no_arg)

    def test_power_curve_monotonically_non_decreasing_with_site_mean(self, sparse_high_ws_pp_df: pd.DataFrame) -> None:
        site_mean_pc_df = _make_site_mean_pc_df()
        result = _cook_pp(
            sparse_high_ws_pp_df,
            pre_or_post="pre",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
            site_mean_pc_df=site_mean_pc_df,
        )
        pw = result["pw_at_mid_pre"].to_numpy()
        # allow tiny floating point diffs
        diffs = np.diff(pw)
        assert (diffs >= -1.0).all(), f"Power curve decreased by more than 1 kW: {diffs.min():.2f}"


# ---------------------------------------------------------------------------
# post period tests
# ---------------------------------------------------------------------------


class TestPostPeriod:
    def test_post_period_columns_named_correctly(self) -> None:
        post_df = _make_pp_raw_df(
            bin_mids=list(range(1, 26)),
            pw_means=[min(RATED_POWER, max(0, RATED_POWER * (m - 3) / 9)) for m in range(1, 26)],
            hours=[50.0] * 25,
            pre_or_post="post",
        )
        result = _cook_pp(
            post_df,
            pre_or_post="post",
            ws_bin_width=WS_BIN_WIDTH,
            rated_power=RATED_POWER,
            clip_to_rated=False,
        )
        assert "pw_mean_post" in result.columns
        assert "pw_at_mid_post" in result.columns
        assert "post_valid" in result.columns
