"""Offline tests for the Hill of Towie source adapter.

These cover the wind-up-format contract of the pure SCADA transforms. The network
path (Zenodo download via ``load_hot_scada``) is exercised separately by a
``slow``-marked test that is excluded from the default offline suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarking.synthetic.sources.hill_of_towie import scada_df_to_wind_up_df
from wind_up.constants import TIMESTAMP_COL, DataColumns

TIMEBASE_S = 600


def _wide_scada_df(*, turbines: tuple[str, ...] = ("T01", "T02"), periods: int = 6) -> pd.DataFrame:
    """Build a fabricated wide two-level SCADA frame as ``load_hot_10min_data`` emits it.

    Level 0 of the columns is the turbine name (index name ``StationId``); level 1 is
    the wind-up field alias. The row index is the start-format timestamp.
    """
    index = pd.date_range("2020-01-01", periods=periods, freq="10min", tz="UTC")
    index.name = TIMESTAMP_COL

    fields = {
        DataColumns.active_power_mean: np.linspace(100.0, 600.0, periods),
        DataColumns.active_power_sd: np.full(periods, 10.0),
        "ReactivePowerMean": np.full(periods, 5.0),
        DataColumns.wind_speed_mean: np.linspace(5.0, 10.0, periods),
        DataColumns.wind_speed_sd: np.full(periods, 1.0),
        DataColumns.yaw_angle_mean: np.full(periods, 180.0),
        DataColumns.yaw_angle_min: np.full(periods, 175.0),
        DataColumns.yaw_angle_max: np.full(periods, 185.0),
        DataColumns.gen_rpm_mean: np.linspace(1000.0, 1500.0, periods),
        "pitch_angle_a": np.full(periods, 1.0),
        "pitch_angle_b": np.full(periods, 2.0),
        "pitch_angle_c": np.full(periods, 3.0),
        DataColumns.ambient_temp: np.full(periods, 8.0),
        "Time ready to operate in period": np.full(periods, float(TIMEBASE_S)),
    }
    # Both turbines share identical values at each timestamp but vary over time. Stuck
    # detection must compare each turbine to its OWN previous record (grouped by turbine),
    # so two turbines that merely match each other are not mistaken for frozen data.
    per_turbine = {turbine: pd.DataFrame(fields, index=index) for turbine in turbines}
    wide = pd.concat(per_turbine, axis=1)
    wide.columns = wide.columns.set_names(["StationId", None])
    return wide


def test_scada_df_to_wind_up_df_emits_wind_up_format() -> None:
    """The wide two-level frame becomes a single-level wind-up frame with TurbineName."""
    wind_up_df = scada_df_to_wind_up_df(_wide_scada_df())

    assert wind_up_df.columns.nlevels == 1
    assert wind_up_df.index.name == TIMESTAMP_COL
    assert set(wind_up_df[DataColumns.turbine_name].unique()) == {"T01", "T02"}
    for col in (
        DataColumns.active_power_mean,
        DataColumns.wind_speed_mean,
        DataColumns.wind_speed_sd,
        DataColumns.gen_rpm_mean,
    ):
        assert col in wind_up_df.columns


def test_scada_df_to_wind_up_df_computes_mean_pitch() -> None:
    """PitchAngleMean is the mean of the three per-blade pitch columns."""
    wind_up_df = scada_df_to_wind_up_df(_wide_scada_df())
    assert np.allclose(wind_up_df[DataColumns.pitch_angle_mean], 2.0)


def test_calc_shutdown_duration_zero_when_fully_available() -> None:
    """Two turbines that share values per timestamp but vary over time are all available.

    Stuck detection compares each turbine to its own previous record, so turbines that
    merely match each other at the same timestamp are not flagged as frozen.
    """
    wind_up_df = scada_df_to_wind_up_df(_wide_scada_df())
    assert DataColumns.shutdown_duration in wind_up_df.columns
    assert np.allclose(wind_up_df[DataColumns.shutdown_duration], 0.0)


def test_calc_shutdown_duration_flags_stuck_data() -> None:
    """Repeated (stuck) rows above the low-wind threshold are flagged as full downtime."""
    wide = _wide_scada_df(turbines=("T01",), periods=4)
    # Freeze T01 to identical values across all rows (stuck), at a wind speed above 1.5 m/s.
    for field in wide.columns.get_level_values(1).unique():
        wide.loc[:, ("T01", field)] = wide.iloc[0].loc[("T01", field)]
    wind_up_df = scada_df_to_wind_up_df(wide)
    # First row has no prior to diff against; subsequent stuck rows are full downtime.
    assert np.allclose(wind_up_df[DataColumns.shutdown_duration].iloc[1:], TIMEBASE_S)


def test_calc_shutdown_duration_flags_only_the_frozen_turbine() -> None:
    """In a multi-turbine frame, only the turbine whose own data is frozen is flagged."""
    wide = _wide_scada_df(turbines=("T01", "T02"), periods=4)
    # Freeze T01 across all rows (stuck); T02 keeps varying over time.
    for field in wide.columns.get_level_values(1).unique():
        wide.loc[:, ("T01", field)] = wide.iloc[0].loc[("T01", field)]
    wind_up_df = scada_df_to_wind_up_df(wide)
    t01 = wind_up_df.loc[wind_up_df[DataColumns.turbine_name] == "T01", DataColumns.shutdown_duration].to_numpy()
    t02 = wind_up_df.loc[wind_up_df[DataColumns.turbine_name] == "T02", DataColumns.shutdown_duration].to_numpy()
    assert np.allclose(t01[1:], TIMEBASE_S)  # frozen turbine -> downtime after the first row
    assert np.allclose(t02, 0.0)  # varying turbine -> available throughout
