from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import pandas as pd

    from wind_up.models import PlotConfig


def plot_rolling_windspeed_diff_one_wtg(
    *,
    ser: pd.Series,
    wtg_name: str,
    ws_col: str,
    plot_cfg: PlotConfig,
    sub_dir: str | None,
) -> None:
    plt.figure()
    plt.plot(ser)
    plot_title = f"{wtg_name} rolling {ws_col} diff to reanalysis"
    plt.title(plot_title)
    plt.xlabel("datetime")
    plt.ylabel("rolling_windspeed_diff [m/s]")
    plt.grid()
    plt.tight_layout()
    if plot_cfg.show_plots:
        plt.show()
    if plot_cfg.save_plots:
        _sub_dir = wtg_name if sub_dir is None else sub_dir
        (plot_cfg.plots_dir / _sub_dir).mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_cfg.plots_dir / _sub_dir / f"{plot_title}.png")
    plt.close()
