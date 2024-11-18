from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utm

if TYPE_CHECKING:
    from pathlib import Path

    from wind_up.models import WindUpConfig

logger = logging.getLogger(__name__)


def rescale_vector(
    vector: pd.Series | np.ndarray,
    lowthres: float = 0,
    highthres: float = 1,
) -> pd.Series | np.ndarray:
    if isinstance(vector, list):
        vector = np.array(vector)

    delta = highthres - lowthres
    tozero = vector - vector.min()
    if not (tozero != 0).any():
        tozero += 1
    zeroone = tozero / (tozero.max())
    scaled = zeroone * delta
    return scaled + lowthres


def bubble_plot_calcs(
    *,
    cfg: WindUpConfig,
    series: pd.Series,
    sizes: pd.Series,
    cbarunits: str | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float],
    tuple[float, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    list[str],
    list[str],
]:
    x = np.array([])
    y = np.array([])
    wtg_id = []
    for wtg in cfg.asset.wtgs:
        c = utm.from_latlon(wtg.latitude, wtg.longitude)
        x = np.append(x, c[0])
        y = np.append(y, c[1])
        wtg_id.append(wtg.name)
    x = x - np.median(x)
    y = y - np.median(y)

    xmar = 500
    ymar = 500
    xlims = (x.min() - xmar, x.max() + xmar)
    ylims = (y.min() - ymar, y.max() + ymar)

    cbarstr = "" if cbarunits is None else cbarunits

    # negative numbers in the series
    if any(series < 0):
        # color
        logger.warning("any(series<0)=True, perhaps some the sizes are weird")
        s = series + abs(series.min())
        sc = 0.1 + 0.8 * s / s.max()
        midtick = (-series.min() * 0.8) / (series.max() - series.min()) + 0.1
        ticks = np.array([sc.min(), midtick, sc.max()])
        ticklabs = [
            f"{series.dropna().min():3.1f}\n{cbarstr}",
            f"{0:3.1f}\n{cbarstr}",
            f"{series.dropna().max():3.1f}\n{cbarstr}",
        ]
        # size
        si = sizes.abs() / sizes.abs().max()

    # all positive numbers
    else:
        # color
        s = rescale_vector(series, lowthres=0.01, highthres=0.99)
        sc = s.copy()
        ticks = np.linspace(sc.min(), sc.max(), 3)
        ticklabs = [
            f"{series.min():3.1f}\n{cbarstr}",
            f"{(series.max() + series.min()) * 0.5:3.1f}\n{cbarstr}",
            f"{series.max():3.1f}\n{cbarstr}",
        ]
        # size
        si = rescale_vector(sizes, lowthres=0, highthres=2)

    park_wtgids = [wtg.name for wtg in cfg.asset.wtgs]
    series_wtgs = series.index.tolist()
    missing_wtgs = [wtg.name for wtg in cfg.asset.wtgs if wtg.name not in series_wtgs]

    if missing_wtgs:
        misseries = pd.Series(index=missing_wtgs)
        sc = pd.concat([sc, misseries], axis=0, sort=True)
        sc = sc.loc[park_wtgids].copy()  # right order
        si = pd.concat([si, misseries], axis=0, sort=True)
        si = si.loc[park_wtgids].copy()  # right order

    if len(sc) != len(park_wtgids):
        msg = "The length of 'series' does not match with the number of turbines"
        raise ValueError(msg)

    if len(si) != len(park_wtgids):
        msg = "The length of 'sizes' does not match with the number of turbines"
        raise ValueError(msg)

    return x, y, xlims, ylims, sc, si, ticks, ticklabs, wtg_id, park_wtgids, series_wtgs


def bubble_plot(
    cfg: WindUpConfig,
    series: pd.Series,
    *,
    sizes: pd.Series | None = None,
    cmap: str = "coolwarm",
    fontsize: int = 12,
    idfontsize: int = 10,
    bubblesize: int = 1000,
    movetext: tuple[float, float] = (-0.0017, -0.001),
    txt: bool = True,
    title: str | None = None,
    cbarunits: str | None = None,
    save_path: Path | None = None,
    display_wtgids: bool = True,
    cbar_label: str | None = None,
    text_rotation: int = 0,
    figuresize: tuple[float, float] | None = None,
    show_plot: bool = False,
) -> None:
    if isinstance(series, type(pd.DataFrame())):
        msg = "series should be a pandas series"
        raise TypeError(msg)

    if sizes is None:
        sizes = series

    x, y, xlims, ylims, sc, si, ticks, ticklabs, wtg_id, park_wtgids, series_wtgs = bubble_plot_calcs(
        cfg=cfg,
        series=series,
        sizes=sizes,
        cbarunits=cbarunits,
    )

    # meters will have the same dimension in both axes
    ratio = (ylims[1] - ylims[0]) / (xlims[1] - xlims[0])
    hsize = 10
    if figuresize is None:
        f, ax = plt.subplots(1, 1, figsize=(hsize * 1.2, hsize * ratio))
    else:
        f, ax = plt.subplots(1, 1, figsize=figuresize)

    csa = ax.scatter(x, y, c=sc, s=bubblesize * si + 100, cmap=cmap, edgecolors=None)

    cbar = f.colorbar(csa, ticks=ticks, pad=0.02, fraction=0.046)  # cax=cax)
    if cbar_label is not None:
        cbar.set_label(cbar_label, fontsize=fontsize)
    cbar.ax.set_yticklabels(ticklabs, fontsize=fontsize)

    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.tick_params(axis="x", labelsize=fontsize)
    plt.tick_params(axis="y", labelsize=fontsize)

    plt.ylabel("Y coord [m]", fontsize=fontsize)
    plt.xlabel("X coord [m]", fontsize=fontsize)
    if title is None:
        plt.title(cfg.asset.name, fontsize=int(fontsize * 1.1))
    else:
        plt.title(title, fontsize=int(fontsize * 1.1))
    plt.tight_layout()

    if txt:
        for i in range(len(park_wtgids)):
            if display_wtgids:
                if wtg_id[i] in series_wtgs:
                    plt.text(
                        x[i] + movetext[0],
                        y[i] + movetext[1],
                        wtg_id[i],
                        color="k",
                        weight=600,
                        fontsize=idfontsize,
                        rotation=text_rotation,
                    )
                else:
                    plt.text(
                        x[i] + movetext[0],
                        y[i] + movetext[1],
                        wtg_id[i],
                        color="grey",
                        fontsize=idfontsize,
                        rotation=text_rotation,
                    )

    plt.grid()
    if show_plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path, dpi=120)
    plt.close()
