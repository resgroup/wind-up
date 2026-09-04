"""Reference-validity screening: find candidate references that are outliers for this farm.

Each candidate reference is estimated as if it were a test turbine, against the other candidates.
The screen then removes clear outliers from what is normal for this farm over this analysis
period -- it is not looking for turbines that changed.

That distinction is the whole design. Turbines are not expected to stay the same: references may
drift or step, and the assumption is only that most of them do so the way the test turbine would
have. A pack all losing 1% a year still reads ~0 on each other, and correctly so. What biases an
estimate is a reference that moves unlike the rest, which is why the statistic is deviation from
the pack's median and the rule is an absolute threshold on it.

Only the worst reference is dropped per pass: a bad reference infects every other estimate, so the
pool is re-judged after each removal.

The loop is estimator-agnostic -- it takes an ``estimate_one`` callable -- so the rule and the
guards are separable from the model that supplies the numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

# Fewer candidate references than this and there is no majority to rule with, so nothing is screened.
MIN_POOL_TO_SCREEN = 3


@dataclass(frozen=True)
class ScreenResult:
    """What the screen concluded, and the per-pass detail behind it.

    :param screened: the references ruled out, in the order they were dropped
    :param passes: one row per turbine per pass -- ``pass``, ``turbine``, ``estimate``,
        ``deviation`` (from that pass's median) and ``dropped``
    :param screenable: whether the pool was large enough to screen at all
    """

    screened: tuple[str, ...]
    passes: pd.DataFrame
    screenable: bool


def rank_by_deviation(estimates: Mapping[str, float]) -> list[tuple[str, float]]:
    """Return ``(turbine, deviation from the median)`` worst first, ties breaking on name.

    A non-finite estimate ranks worst: a reference the screen could not estimate is not thereby
    shown to be good.
    """
    finite = {name: float(v) for name, v in estimates.items() if np.isfinite(v)}
    median = float(np.median(list(finite.values()))) if finite else float("nan")
    ranked = [(name, abs(v - median)) for name, v in finite.items()]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    unestimated = sorted(name for name in estimates if name not in finite)
    return [(name, float("inf")) for name in unestimated] + ranked


def worst_outlier(estimates: Mapping[str, float], *, floor: float) -> str | None:
    """Return the reference furthest from the median if it is at least ``floor`` away, else None."""
    ranked = rank_by_deviation(estimates)
    if not ranked:
        return None
    name, deviation = ranked[0]
    return name if deviation >= floor else None


def max_screenable(pool_size: int) -> int:
    """How many references may be ruled out of a pool of ``pool_size`` and still leave a majority."""
    return (pool_size - 1) // 2


def screen_references(
    pool: Sequence[str],
    *,
    estimate_one: Callable[[str, list[str]], float],
    floor: float,
) -> ScreenResult:
    """Rule out outlying candidate references one per pass, re-screening until none stand out.

    :param pool: the candidate references
    :param estimate_one: ``(target, references) -> uplift`` for one screening estimate
    :param floor: deviation from the pack's median at which a reference is ruled out
    :raises ValueError: when a reference still stands out after a majority's worth have been ruled
        out -- a farm-wide problem rather than a reference-validity one
    """
    remaining = list(pool)
    if len(remaining) < MIN_POOL_TO_SCREEN:
        logger.info(
            "reference screen skipped: %d candidate reference(s), fewer than the %d needed to form a majority",
            len(remaining),
            MIN_POOL_TO_SCREEN,
        )
        return ScreenResult(screened=(), passes=_empty_passes(), screenable=False)

    allowance = max_screenable(len(remaining))
    screened: list[str] = []
    rows: list[dict[str, object]] = []
    for n_pass in range(1, allowance + 2):
        if len(remaining) < MIN_POOL_TO_SCREEN:
            # A drop can leave too few to vote. Two references are always equidistant from their
            # own midpoint, so the rule would flag one of them arbitrarily.
            logger.info(
                "reference screen stopping: %d reference(s) remain, fewer than the %d needed to form a majority",
                len(remaining),
                MIN_POOL_TO_SCREEN,
            )
            break
        estimates = {wtg: estimate_one(wtg, [r for r in remaining if r != wtg]) for wtg in remaining}
        worst = worst_outlier(estimates, floor=floor)
        rows.extend(_pass_rows(estimates, dropped=worst, n_pass=n_pass))
        if worst is None:
            break
        if len(screened) >= allowance:
            msg = (
                f"{worst!r} still stands out after ruling out {sorted(screened)}, which is already the "
                f"{allowance} a majority of {len(pool)} references can outvote. More references look unlike each "
                f"other than the screen can attribute to any of them, which points to a farm-wide problem that "
                f"has probably reached the test turbine too, so no estimate is offered."
            )
            raise ValueError(msg)
        screened.append(worst)
        remaining = [r for r in remaining if r != worst]
        logger.info("reference screen pass %d ruled out %s; %d reference(s) remain", n_pass, worst, len(remaining))
    return ScreenResult(screened=tuple(screened), passes=pd.DataFrame(rows), screenable=True)


def _pass_rows(estimates: Mapping[str, float], *, dropped: str | None, n_pass: int) -> list[dict[str, object]]:
    """Return one row per screened turbine recording what this pass saw."""
    deviations = dict(rank_by_deviation(estimates))
    return [
        {
            "pass": n_pass,
            "turbine": wtg,
            "estimate": value,
            "deviation": deviations[wtg],
            "dropped": wtg == dropped,
        }
        for wtg, value in estimates.items()
    ]


def _empty_passes() -> pd.DataFrame:
    """Return the empty per-pass frame, with the columns a screen that never ran would have."""
    return pd.DataFrame(columns=["pass", "turbine", "estimate", "deviation", "dropped"])
