"""Vectorised Cp-space physics core for synthetic upgrade injection.

The Cp surface is an analytic, parameterised model of Cp as a function of tip-speed
ratio (TSR) and blade pitch. It makes a made-up surface that is internally self-consistent for power<->Cp
conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class CpParams:
    """Parameters of the analytic ``Cp(TSR, pitch)`` surface."""

    cp_max: float
    opt_pitch: float
    pitch_scale: float
    opt_tsr: float
    tsr_scale: float
    banana_factor: float


# Hill of Towie Cp model: log-symmetric TSR form, coefficients least-squares fitted to
# the turbine's Cp(TSR, pitch) table, weighted (1 + Cp)^2 to favour the high-Cp region.
HOT_CP_MODEL = CpParams(
    cp_max=0.4509,
    opt_pitch=-1.2085,
    pitch_scale=0.0067,
    opt_tsr=6.6255,
    tsr_scale=1.5558,
    banana_factor=42.2606,
)


def cp_surface(
    *,
    tsr: npt.ArrayLike,
    pitch: npt.ArrayLike,
    params: CpParams = HOT_CP_MODEL,
) -> npt.NDArray[np.float64]:
    """Evaluate the analytic Cp surface at the given TSR and pitch.

    :param tsr: tip-speed ratio (scalar or array)
    :param pitch: blade pitch in degrees (scalar or array)
    :param params: Cp surface parameters
    :return: Cp value(s)
    """
    tsr_arr = np.asarray(tsr, dtype=float)
    pitch_arr = np.asarray(pitch, dtype=float)

    # banana_factor shifts the effective optimal pitch as TSR departs from optimal,
    # producing the characteristic curved ("banana") Cp contours.
    effective_opt_pitch = params.opt_pitch + params.banana_factor * np.abs(1.0 / params.opt_tsr - 1.0 / tsr_arr)
    pitch_factor = np.maximum(0.0, 1.0 - params.pitch_scale * (effective_opt_pitch - pitch_arr) ** 2)
    # The TSR falloff is log-symmetric (penalises tsr/opt_tsr + opt_tsr/tsr - 2), so it
    # decays gently up the high-TSR side as real turbines do, unlike a symmetric
    # quadratic.
    tsr_ratio = tsr_arr / params.opt_tsr
    tsr_factor = np.maximum(0.0, 1.0 - params.tsr_scale * (tsr_ratio + 1.0 / tsr_ratio - 2.0))
    return params.cp_max * pitch_factor * tsr_factor


# Defaults for the power-based region-2 fraction, ported from baby-yoda's HoT turbine
# model (sigmoid midpoint 2000 kW, steepness 1/130, for a 2300 kW rated turbine).
REGION2_MIDPOINT_KW = 2000.0
REGION2_STEEPNESS = 1.0 / 130.0

# A Cp change is not applied to records at or above this fraction of rated power: such
# points are effectively at pure rated and should be left unchanged (the region-2 sigmoid
# only tails towards zero there, so without this guard a tiny change would still leak in).
NEAR_RATED_POWER_FRACTION = 0.995


def region2_fraction(
    power_kw: npt.ArrayLike,
    *,
    midpoint_kw: float = REGION2_MIDPOINT_KW,
    steepness: float = REGION2_STEEPNESS,
) -> npt.NDArray[np.float64]:
    """Estimate the fraction of a 10-min period spent in region 2 from mean power.

    A squared logistic that is ~1 deep in region 2 and tails to ~0 as power approaches
    rated, so a Cp change applied in region 2 fades out near rated power.

    :param power_kw: mean active power (scalar or array)
    :param midpoint_kw: power at which the underlying logistic is 0.5
    :param steepness: logistic steepness
    :return: region-2 fraction in [0, 1]
    """
    power_arr = np.asarray(power_kw, dtype=float)
    logistic = 1.0 / (1.0 + np.exp(steepness * (power_arr - midpoint_kw)))
    return logistic**2


def power_from_cp_change(
    baseline_power_kw: npt.ArrayLike,
    *,
    cp_ratio: npt.ArrayLike,
    rated_power_kw: float,
) -> npt.NDArray[np.float64]:
    """Apply a Cp ratio to baseline power, weighted by the region-2 fraction.

    Only the region-2 fraction of the period responds to the Cp change; the remainder
    (near/at rated) is unchanged, and the result is clipped so it never exceeds the
    larger of rated power and the original power. A Cp change is applied only to records
    that are *producing and below pure rated*: non-producing records (zero or negative
    power: idling, self-consumption, curtailment) and virtually-rated records (at or above
    ``NEAR_RATED_POWER_FRACTION`` of rated) are left untouched.

    :param baseline_power_kw: original mean active power (scalar or array)
    :param cp_ratio: ratio of new Cp to baseline Cp (e.g. 1.02 for +2%)
    :param rated_power_kw: rated power used for the upper clip
    :return: modified mean active power
    """
    baseline = np.asarray(baseline_power_kw, dtype=float)
    ratio = np.asarray(cp_ratio, dtype=float)
    fraction = region2_fraction(baseline)
    new_power = baseline * (1.0 + fraction * (ratio - 1.0))
    upper_clip = np.maximum(baseline, rated_power_kw)
    clipped = np.minimum(new_power, upper_clip)
    modifiable = (baseline > 0.0) & (baseline < rated_power_kw * NEAR_RATED_POWER_FRACTION)
    return np.where(modifiable, clipped, baseline)


# Generator-speed vs power curve for the HoT turbine, ported from baby-yoda
# (operating curves, sco_hot).
_RPM_VS_POWER_X = [0.0, 200.0, 350.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 1750.0, 2000.0, 2300.0]
_RPM_VS_POWER_Y = [720.0, 755.0, 956.0, 1090.0, 1262.0, 1392.0, 1471.0, 1512.0, 1534.0, 1545.0, 1552.0]


def rpm_from_power(power_kw: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Look up generator rpm from mean power on the ported operating curve."""
    return np.interp(np.asarray(power_kw, dtype=float), _RPM_VS_POWER_X, _RPM_VS_POWER_Y)


def rpm_from_power_change(
    *,
    baseline_rpm: npt.ArrayLike,
    baseline_power_kw: npt.ArrayLike,
    new_power_kw: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Scale baseline rpm by the operating-curve rpm ratio implied by a power change.

    rpm tracks power along the operating curve, so a change in power drags rpm by the
    ratio of curve rpm at the new vs baseline power. Where the new power is not
    positive the baseline rpm is left unchanged.

    :param baseline_rpm: original generator rpm
    :param baseline_power_kw: original mean active power
    :param new_power_kw: modified mean active power
    :return: modified generator rpm
    """
    baseline_rpm_arr = np.asarray(baseline_rpm, dtype=float)
    new_power = np.asarray(new_power_kw, dtype=float)
    ratio = rpm_from_power(new_power) / rpm_from_power(baseline_power_kw)
    scaled = baseline_rpm_arr * ratio
    return np.where(new_power > 0.0, scaled, baseline_rpm_arr)


@dataclass(frozen=True)
class CpCore:
    """Per-turbine Cp-space physics, the integration point upgrades operate through.

    Bundles the rated power and Cp surface parameters so an upgrade can turn a desired
    Cp ratio into modified power and rpm without knowing turbine-specific constants.
    """

    rated_power_kw: float = 2300.0
    cp_params: CpParams = HOT_CP_MODEL

    def apply_cp_ratio(self, baseline_power_kw: npt.ArrayLike, *, cp_ratio: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Apply a Cp ratio to baseline power using this core's rated-power clip."""
        return power_from_cp_change(baseline_power_kw, cp_ratio=cp_ratio, rated_power_kw=self.rated_power_kw)

    def rpm_after(
        self,
        *,
        baseline_rpm: npt.ArrayLike,
        baseline_power_kw: npt.ArrayLike,
        new_power_kw: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Return generator rpm after a power change, tracking the operating curve."""
        return rpm_from_power_change(
            baseline_rpm=baseline_rpm,
            baseline_power_kw=baseline_power_kw,
            new_power_kw=new_power_kw,
        )
