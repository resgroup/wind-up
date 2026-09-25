import pandas as pd

from wind_up_v0.constants import TIMESTAMP_COL
from wind_up_v0.interface import PrePostSplitter
from wind_up_v0.models import WindUpConfig


def _toggle_df(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """2-level (wtg_name, timestamp) toggle_df with distinct per-turbine patterns.

    MRG_T01 (test wtg): on for the last 3 bins.
    MRG_T06 (test wtg): on for the last 2 bins (its own, different pattern).
    MRG_T02 (ref wtg):  never on (the crashing case).
    """

    def mk(on_mask: list[bool]) -> pd.DataFrame:
        return pd.DataFrame({"toggle_on": on_mask, "toggle_off": [not x for x in on_mask]}, index=idx)

    return pd.concat(
        {
            "MRG_T01": mk([False, False, False, True, True, True]),
            "MRG_T06": mk([False, False, False, False, True, True]),
            "MRG_T02": mk([False] * 6),
        },
        names=["wtg_name", TIMESTAMP_COL],
    )


def _splitter(cfg: WindUpConfig, borrow_from: str | None) -> tuple[PrePostSplitter, pd.DataFrame, pd.DatetimeIndex]:
    cfg.toggle.toggle_change_settling_filter_seconds = 0  # keep every bin so counts are exact
    cfg.toggle.borrow_toggle_from_wtg = borrow_from
    idx = pd.date_range(cfg.analysis_first_dt_utc_start, periods=6, freq="10min", name=TIMESTAMP_COL)
    splitter = PrePostSplitter(cfg=cfg, toggle_df=_toggle_df(idx))
    wtg_scada = pd.DataFrame({"some_col": range(6)}, index=idx)
    return splitter, wtg_scada, idx


def test_split_borrows_toggle_for_non_test_wtg(test_marge_config: WindUpConfig) -> None:
    splitter, wtg_scada, idx = _splitter(test_marge_config, borrow_from="MRG_T01")
    _test_df, _pre_df, post_df = splitter.split(wtg_scada, test_wtg_name="MRG_T02")
    # MRG_T02 has no toggle_on of its own; it borrows MRG_T01's 3 on-bins.
    assert list(post_df.index) == list(idx[3:6])


def test_split_uses_own_toggle_for_real_test_wtg(test_marge_config: WindUpConfig) -> None:
    splitter, wtg_scada, idx = _splitter(test_marge_config, borrow_from="MRG_T01")
    _test_df, _pre_df, post_df = splitter.split(wtg_scada, test_wtg_name="MRG_T06")
    # MRG_T06 is a test wtg, so it keeps its OWN 2 on-bins, not MRG_T01's 3.
    assert list(post_df.index) == list(idx[4:6])


def test_split_without_borrow_leaves_non_test_wtg_post_empty(test_marge_config: WindUpConfig) -> None:
    splitter, wtg_scada, _idx = _splitter(test_marge_config, borrow_from=None)
    _test_df, _pre_df, post_df = splitter.split(wtg_scada, test_wtg_name="MRG_T02")
    # Regression demo: this is the crashing case — no borrow, no post data.
    assert post_df.empty
