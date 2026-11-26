import numpy as np
import pandas as pd

from wind_up_v0.plots.yaw_direction_plots import _hours_of_data


def _binned(series: pd.Series, *, bin_width: float) -> pd.Series:
    edges = np.arange(0, 40 + bin_width, bin_width)
    return pd.cut(series, bins=edges).apply(lambda x: x.mid)


def test_hours_of_data_leaves_out_bins_without_data() -> None:
    plot_df = pd.DataFrame({"ws": [7.0] * 6 + [9.0] * 3 + [11.0], "wd": [12.0] * 6 + [17.0] * 3 + [17.0]})
    plot_df["ws_bin_centre"] = _binned(plot_df["ws"], bin_width=2)
    plot_df["wd_bin_centre"] = _binned(plot_df["wd"], bin_width=5)
    plot_df["yaw_offset"] = [1.0] * 9 + [np.nan]

    hours = _hours_of_data(plot_df, signal_name="yaw_offset")

    assert list(hours.index) == [7.0, 9.0]
    assert list(hours.columns) == [12.5, 17.5]
    np.testing.assert_array_equal(hours.to_numpy(), [[1.0, np.nan], [np.nan, 0.5]])
