"""The shared upgrade-profile set for the longer (overnight) prepost and toggle studies.

Defined once so the prepost and toggle runs score an identical set. Seven profiles spanning
sign, magnitude and shape:

* constant Cp: -10%, 0% (placebo), +3%, +10%
* wind-speed-dependent Cp: +10% held below 5 m/s, fading linearly to 0% by 12 m/s
* TI-dependent Cp: +10% held below 10% turbulence intensity, fading to 0% by 30% TI
* rated-power uprate: +5% (HoT rated = 2300 kW -> 2415 kW)
"""

from __future__ import annotations

from benchmarking.synthetic import (
    HOT_RATED_POWER_KW,
    ConditionCpChange,
    ConstantCpChange,
    RatedPowerChange,
    WindSpeedCpChange,
)


def overnight_profiles() -> dict[str, list]:
    """Return the shared {name -> list of upgrade effects} mapping for the overnight studies."""
    return {
        "cp_minus_10pct": [ConstantCpChange(delta=-0.10)],
        "cp_0pct": [ConstantCpChange(delta=0.0)],
        "cp_plus_3pct": [ConstantCpChange(delta=0.03)],
        "cp_plus_10pct": [ConstantCpChange(delta=0.10)],
        # +10% Cp held below 5 m/s, fading linearly to 0% by 12 m/s (endpoints held outside range)
        "ws_dependent_cp": [WindSpeedCpChange(ws_points=(5.0, 12.0), deltas=(0.10, 0.0))],
        # +10% Cp held below 10% TI, fading linearly to 0% by 30% TI (TI = ws_sd / ws, a fraction)
        "ti_dependent_cp": [ConditionCpChange(by="ti", points=(0.10, 0.30), deltas=(0.10, 0.0))],
        "rated_plus_5pct": [RatedPowerChange(new_rated_power_kw=HOT_RATED_POWER_KW * 1.05)],
    }
