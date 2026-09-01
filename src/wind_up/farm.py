"""Combine per-turbine uplift estimates into one farm headline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class TurbineUplift:
    """One turbine's uplift estimate and the treated-period energy behind it.

    :param turbine: turbine name
    :param uplift: the turbine's P50 uplift, as an energy-ratio fraction
    :param treated_energy: observed treated-period energy — the sum of finite active power over
        the treated records
    :param n_records: how many records that sum covers
    :param rated_power_kw: the rating the capacity-factor cap uses; where the rating changed over
        the campaign, pass the higher of the pre- and post-change values
    """

    turbine: str
    uplift: float
    treated_energy: float
    n_records: int
    rated_power_kw: float


@dataclass(frozen=True)
class FarmUplift:
    """The farm headline and the per-turbine detail behind it.

    :param uplift: the headline, ``(sum treated energy) / (sum counterfactual energy) - 1`` over the
        used turbines; NaN when none are usable
    :param turbines: one row per input turbine with ``turbine``, ``uplift``, ``treated_energy``,
        ``n_records``, ``rated_power_kw``, ``counterfactual_energy``, ``used`` and ``guard``
        (``""`` when no guard fired)
    :param uplift_spread: the max-min of the used turbines' uplifts; NaN below two used turbines
    """

    uplift: float
    turbines: pd.DataFrame
    uplift_spread: float


def farm_uplift(turbines: Sequence[TurbineUplift]) -> FarmUplift:
    """Aggregate per-turbine uplifts into one energy-weighted farm headline.

    Each turbine's counterfactual energy is estimated as ``treated_energy / (1 + uplift)`` and
    guarded: a turbine is dropped when its uplift is non-finite or ``<= -1``, when its treated
    energy is negative, or when it has no records; a counterfactual implying a mean power above
    ``rated_power_kw`` is clipped to that rating.
    """
    if not turbines:
        msg = "farm_uplift needs at least one turbine"
        raise ValueError(msg)

    frame = pd.DataFrame([_evaluate(t) for t in turbines])
    used = frame[frame["used"]]

    counterfactual_total = float(used["counterfactual_energy"].sum())
    treated_total = float(used["treated_energy"].sum())
    uplift = treated_total / counterfactual_total - 1.0 if counterfactual_total else float("nan")

    spreads = used["uplift"]
    spread = float(spreads.max() - spreads.min()) if len(spreads) > 1 else float("nan")
    return FarmUplift(uplift=uplift, turbines=frame, uplift_spread=spread)


def _evaluate(turbine: TurbineUplift) -> dict[str, object]:
    """Return one turbine's row: its counterfactual energy, whether it is used, and any guard."""
    base: dict[str, object] = {
        "turbine": turbine.turbine,
        "uplift": turbine.uplift,
        "treated_energy": turbine.treated_energy,
        "n_records": turbine.n_records,
        "rated_power_kw": turbine.rated_power_kw,
    }
    guard = _drop_reason(turbine)
    if guard:
        return {**base, "counterfactual_energy": float("nan"), "used": False, "guard": guard}

    counterfactual = max(turbine.treated_energy / (1.0 + turbine.uplift), 0.0)
    cap = turbine.rated_power_kw * turbine.n_records
    if counterfactual > cap:
        return {**base, "counterfactual_energy": cap, "used": True, "guard": "capacity_cap"}
    return {**base, "counterfactual_energy": counterfactual, "used": True, "guard": ""}


def _drop_reason(turbine: TurbineUplift) -> str:
    """Name the guard that removes ``turbine`` from the weighting, or ``""`` to keep it."""
    if not math.isfinite(turbine.uplift):
        return "non_finite_uplift"
    if turbine.n_records <= 0:
        return "no_records"
    if turbine.treated_energy < 0:
        return "negative_energy"
    if 1.0 + turbine.uplift <= 0:
        return "negative_counterfactual"
    return ""
