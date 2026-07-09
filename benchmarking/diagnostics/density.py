"""Scatter plots coloured by data density.

wind-up datasets are large, so plain scatter plots saturate and the eye cannot tell where the
bulk of the data sits versus a handful of outliers. Colouring each point by the local 2-D
histogram density fixes that. Adapted from ``tuneup-ml``'s ``plotting/density.py`` (the user's
own code, offered for reuse), with a fallback for small/degenerate inputs so it never raises.

The density field is a fine 2-D histogram smoothed with a gaussian before it is sampled back onto
the points, giving a KDE-like gradient at O(n) cost (no per-point ``gaussian_kde``, which is
O(n^2) and prohibitively slow on wind-up-sized data).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import numpy.typing as npt

# splinef2d needs a few points along each axis; below this fall back to a flat colour.
_MIN_POINTS_FOR_DENSITY = 8
_MIN_UNIQUE_PER_AXIS = 2
_DEFAULT_BINS = 120  # fine histogram; the gaussian smoothing below keeps the gradient continuous
_DEFAULT_SMOOTH_SIGMA = 2.0  # gaussian smoothing (in bins) for a KDE-like gradient


def density_scatter(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    ax: plt.Axes,
    bins: int = _DEFAULT_BINS,
    smooth_sigma: float = _DEFAULT_SMOOTH_SIGMA,
    sort: bool = True,
    colorbar: bool = True,
    **kwargs: Any,  # noqa: ANN401
) -> plt.Axes:
    """Scatter ``y`` vs ``x`` on ``ax``, colouring points by smoothed 2-D histogram density.

    Non-finite pairs are dropped. With too few points to estimate a density the points are still
    plotted (flat colour). The densest points are drawn last so they are visible on top.

    :param ax: the axes to draw on (required; callers manage the figure/layout)
    :param bins: 2-D histogram bin count per axis
    :param smooth_sigma: gaussian smoothing of the histogram, in bins (0 disables); a KDE-like gradient
    :param sort: draw densest points last
    :param colorbar: attach a "data density" colourbar to ``ax``
    """
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    finite = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[finite], yv[finite]
    if len(xv) == 0:
        return ax

    z = _density(xv, yv, bins=bins, smooth_sigma=smooth_sigma)
    if sort:
        order = z.argsort()
        xv, yv, z = xv[order], yv[order], z[order]

    scatter = ax.scatter(xv, yv, c=z, **kwargs)
    if colorbar:
        cbar = ax.figure.colorbar(scatter, ax=ax)
        # The density scale is arbitrary, so hide the numeric ticks. Use the colorbar API
        # (set_ticks) rather than set_yticklabels([]), which trips Matplotlib's FixedFormatter
        # warning — fatal under the tests' warnings-as-errors config.
        cbar.set_ticks([])
        cbar.ax.set_ylabel("data density")
    return ax


def _density(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], *, bins: int, smooth_sigma: float = _DEFAULT_SMOOTH_SIGMA
) -> npt.NDArray[np.float64]:
    """Per-point density via a smoothed 2-D histogram interpolated back onto the points (0 on failure)."""
    if (
        len(x) < _MIN_POINTS_FOR_DENSITY
        or len(np.unique(x)) < _MIN_UNIQUE_PER_AXIS
        or len(np.unique(y)) < _MIN_UNIQUE_PER_AXIS
    ):
        return np.zeros(len(x))
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    if smooth_sigma > 0:
        data = gaussian_filter(data, sigma=smooth_sigma)  # KDE-like smooth gradient over the fine bins
    # Linear (not splinef2d): points at the data edges fall just outside the bin-centre grid, and
    # splinef2d refuses to extrapolate. Linear with a 0 fill is robust and visually equivalent.
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    z = np.asarray(z, dtype=float)
    z[~np.isfinite(z)] = 0.0
    return z
