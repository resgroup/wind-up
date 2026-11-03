import pandas as pd

from wind_up.constants import TIMESTAMP_COL
from wind_up.interface import PrePostSplitter, _preprocess_cache_key
from wind_up.models import WindUpConfig
from wind_up.reanalysis_data import ReanalysisDataset
from wind_up.smart_data import add_smart_lat_long_to_cfg


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


def test_preprocess_cache_key_changes_with_inputs(test_lsa_t13_config: WindUpConfig) -> None:
    wtg_names = [wtg.name for wtg in test_lsa_t13_config.asset.wtgs]
    metadata_df = pd.DataFrame(
        {"Name": wtg_names, "Latitude": [52.4 + i / 100 for i in range(len(wtg_names))], "Longitude": -0.9}
    )
    inputs = {
        "cfg": test_lsa_t13_config,
        "scada_df": pd.DataFrame({"ActivePowerMean": [1.0, 2.0]}),
        "metadata_df": metadata_df,
        "reanalysis_datasets": [ReanalysisDataset(id="ERA5", data=pd.DataFrame({"100_m_hws_mean_mps": [5.0]}))],
    }
    key = _preprocess_cache_key(**inputs)
    assert key == _preprocess_cache_key(**inputs)
    new_cfg = test_lsa_t13_config.model_copy(update={"exclusion_periods_utc": []})
    assert key != _preprocess_cache_key(**{**inputs, "cfg": new_cfg})
    new_scada_df = pd.DataFrame({"ActivePowerMean": [1.0, 3.0]})
    assert key != _preprocess_cache_key(**{**inputs, "scada_df": new_scada_df})
    # turbine coordinates are copied into cfg from metadata_df after preprocessing,
    # also into test_wtgs and ref_wtgs, which share their Turbine instances with asset.wtgs
    test_ref_wtgs = test_lsa_t13_config.test_wtgs + test_lsa_t13_config.ref_wtgs
    coords_before = [(wtg.latitude, wtg.longitude) for wtg in test_ref_wtgs]
    add_smart_lat_long_to_cfg(md=metadata_df, cfg=test_lsa_t13_config)
    assert [(wtg.latitude, wtg.longitude) for wtg in test_ref_wtgs] != coords_before
    assert key == _preprocess_cache_key(**inputs)


def test_preprocess_cache_key_changes_with_dtypes(test_lsa_t13_config: WindUpConfig) -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="10min")
    inputs = {
        "cfg": test_lsa_t13_config,
        "scada_df": pd.DataFrame({"ActivePowerMean": [1, 0]}, index=idx),
        "metadata_df": pd.DataFrame({"Name": ["LSA_T13"]}),
        "reanalysis_datasets": [],
    }
    key = _preprocess_cache_key(**inputs)
    bool_scada_df = pd.DataFrame({"ActivePowerMean": [True, False]}, index=idx)
    assert key != _preprocess_cache_key(**{**inputs, "scada_df": bool_scada_df})
    utc_scada_df = pd.DataFrame({"ActivePowerMean": [1, 0]}, index=idx.tz_localize("UTC"))
    assert key != _preprocess_cache_key(**{**inputs, "scada_df": utc_scada_df})
