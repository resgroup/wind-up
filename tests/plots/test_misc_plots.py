import math

import pandas as pd
import pytest

from wind_up.geodesy import distance_and_bearing
from wind_up_v0.models import WindUpConfig
from wind_up_v0.plots.misc_plots import bubble_plot_calcs


def test_bubble_plot_positions_are_true_east_north(test_homer_config: WindUpConfig) -> None:
    cfg = test_homer_config
    names = [wtg.name for wtg in cfg.asset.wtgs]
    series = pd.Series([1.0, 2.0], index=names)
    x, y, *_ = bubble_plot_calcs(cfg=cfg, series=series, sizes=series)
    t1, t2 = cfg.asset.wtgs[0], cfg.asset.wtgs[1]
    distance_m, bearing_deg = distance_and_bearing((t1.latitude, t1.longitude), (t2.latitude, t2.longitude))
    assert x[1] - x[0] == pytest.approx(distance_m * math.sin(math.radians(bearing_deg)), abs=0.01)
    assert y[1] - y[0] == pytest.approx(distance_m * math.cos(math.radians(bearing_deg)), abs=0.01)


def test_bubble_plot_positions_are_centred_on_the_median(test_homer_config: WindUpConfig) -> None:
    cfg = test_homer_config
    names = [wtg.name for wtg in cfg.asset.wtgs]
    series = pd.Series([1.0, 2.0], index=names)
    x, y, *_ = bubble_plot_calcs(cfg=cfg, series=series, sizes=series)
    assert sorted(x) == pytest.approx([-sorted(x)[1], sorted(x)[1]])
    assert sorted(y) == pytest.approx([-sorted(y)[1], sorted(y)[1]])
