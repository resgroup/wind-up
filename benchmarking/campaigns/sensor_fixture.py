"""The R2 sensor fixture: what does an unstable anemometer cost wind-up as it ships today.

A placebo campaign -- T06 plus its three nearest turbines whose northing is stable over the
period, with no upgrade injected, so the truth is exactly zero and any movement in an estimate is
the fault's doing. Each arm re-runs it with one anemometer gain fault:

* **shape** -- a step at the moment the contrast is measured across, or a drift ramping over the
  whole record (the shape least likely to cancel under toggle's alternation);
* **gain** -- ``x1.5`` and ``x0.5``, the worst case that still happens in real data;
* **target** -- the test turbine (whose anemometer feeds the conditional axes) or the nearest
  reference (whose wind speed feeds only the ERA5 lag sync).

An **exposed** arm repeats the clean cell and the steps with reference anemometry deliberately
carried as model features, which is the configuration the standing exclusion rules out. Comparing
the two prices that exclusion.

Both fault shapes scale mean and SD together, so turbulence intensity is unchanged and only the
wind-speed axis moves.

Run it::

    uv run python -m benchmarking.campaigns.sensor_fixture

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``sensor_fixture``/``<timestamp>/``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl

mpl.use("Agg")  # headless: the report writes plots without a display

import pandas as pd

from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.campaigns.methods import carried_forward_methods
from benchmarking.campaigns.placebo import (
    PLACEBO_CAMPAIGN_START,
    placebo_analysis_period,
    placebo_campaign,
)
from benchmarking.campaigns.runner import CampaignRunner
from benchmarking.harness.northing import era5_direction
from benchmarking.synthetic import HOT_COLUMNS, SensorGainDrift, SensorGainStep
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarking.campaigns.declaration import SyntheticCampaign
    from benchmarking.campaigns.runner import CampaignResult
    from benchmarking.synthetic import Fault

logger = logging.getLogger(__name__)

# T06 is the measured-best fixture turbine; its three nearest neighbours whose northing is stable
# over 2017-2018 are the references (T05 is nearer but carries real northing steps of its own).
FIXTURE_TEST_WTG = "T06"
FIXTURE_REFERENCES = ("T15", "T10", "T08")
FIXTURE_TURBINES = (FIXTURE_TEST_WTG, *FIXTURE_REFERENCES)

# The nearest reference, so a reference-side fault lands on the most influential turbine.
FAULT_REFERENCE = "T15"
FAULT_TARGETS = (FIXTURE_TEST_WTG, FAULT_REFERENCE)

# Worst case first: if a half-scale reading does not move the headline, no smaller one will.
GAINS = (1.5, 0.5)

# The reference anemometry the exposed arm carries as features, which the default never does.
EXPOSED_STAT_COLS = (HOT_COLUMNS.wind_speed, HOT_COLUMNS.wind_speed_sd)

# Movement in the headline at or above this many percentage points counts as material.
MATERIAL_PP = 0.25

_MODES = ("prepost", "toggle")


@dataclass(frozen=True)
class Arm:
    """One cell of the fixture: a fault to inject (or none) and the method configuration to use.

    :param name: the arm's label in the output tables
    :param fault: the sensor fault to inject; ``None`` for a clean cell
    :param exposed: carry reference anemometry as model features
    """

    name: str
    fault: Fault | None = None
    exposed: bool = False


def analysis_period(mode: Literal["prepost", "toggle"]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the whole record the methods see for ``mode``: the baseline plus the campaign."""
    return placebo_analysis_period(mode)


def fault_time(mode: Literal["prepost", "toggle"]) -> pd.Timestamp:
    """When an injected step lands: at the changeover (prepost) or mid-campaign (toggle)."""
    if mode == "prepost":
        return PLACEBO_CAMPAIGN_START
    _, end = analysis_period(mode)
    return PLACEBO_CAMPAIGN_START + (end - PLACEBO_CAMPAIGN_START) / 2


def fixture_arms(mode: Literal["prepost", "toggle"]) -> list[Arm]:
    """Return every arm of the fixture for ``mode``: the clean cell, the faults, and the exposed set."""
    if mode not in _MODES:
        msg = f"unknown mode {mode!r}; expected 'prepost' or 'toggle'"
        raise ValueError(msg)
    at = fault_time(mode)
    arms = [Arm(name="clean")]
    for gain in GAINS:
        for target in FAULT_TARGETS:
            arms.append(  # noqa: PERF401 - a comprehension over three nested loops reads worse
                Arm(name=f"step_x{gain:g}_{target}", fault=SensorGainStep(turbine=target, at=at, gain=gain))
            )
    for gain in GAINS:
        for target in FAULT_TARGETS:
            arms.append(  # noqa: PERF401 - as above
                Arm(name=f"drift_x{gain:g}_{target}", fault=SensorGainDrift(turbine=target, gain=gain))
            )
    arms.append(Arm(name="exposed_clean", exposed=True))
    for gain in GAINS:
        for target in FAULT_TARGETS:
            arms.append(  # noqa: PERF401 - as above
                Arm(
                    name=f"exposed_step_x{gain:g}_{target}",
                    fault=SensorGainStep(turbine=target, at=at, gain=gain),
                    exposed=True,
                )
            )
    return arms


def _coords(turbines: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Hill of Towie coordinates for ``turbines``."""
    metadata = load_hot_metadata()
    return {
        str(row.Name): (float(row.Latitude), float(row.Longitude))
        for row in metadata.itertuples()
        if str(row.Name) in set(turbines)
    }


def fixture_campaign(
    mode: Literal["prepost", "toggle"],
    *,
    arm: Arm,
    coords: dict[str, tuple[float, float]] | None = None,
) -> SyntheticCampaign:
    """Declare one arm of the fixture: the placebo on the fixture turbines, plus the arm's fault.

    :param mode: ``"prepost"`` or ``"toggle"``
    :param arm: the cell being declared
    :param coords: turbine coordinates; a placeholder is used when omitted
    """
    return placebo_campaign(
        mode,
        upgraded=[FIXTURE_TEST_WTG],
        turbines=list(FIXTURE_TURBINES),
        coords=coords,
        faults=[arm.fault] if arm.fault is not None else [],
    )


def run_cell(
    *,
    mode: Literal["prepost", "toggle"],
    arm: Arm,
    scada_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    out_dir: Path,
    include_power_model: bool = True,
) -> CampaignResult:
    """Run one arm and return its result."""
    campaign = fixture_campaign(mode, arm=arm, coords=_coords(FIXTURE_TURBINES))
    dataset = campaign.generate(scada_df)
    spec = campaign.spec()
    index = pd.DatetimeIndex(dataset.synthetic_df.index.unique()).sort_values()
    runner = CampaignRunner(
        spec,
        dataset,
        build_methods=lambda wtg: carried_forward_methods(
            spec,
            out_dir=out_dir / wtg,
            era5_hourly_df=era5_df if include_power_model else None,
            include_power_model=include_power_model,
            reference_stat_cols=EXPOSED_STAT_COLS if arm.exposed else (),
        ),
        era5_wd=era5_direction(era5_df, index),
    )
    return runner.run()


def _conditional_rows(result: CampaignResult, *, mode: str, arm: Arm) -> list[dict[str, object]]:
    """Flatten each method's per-condition estimates for ``result`` into tidy rows."""
    rows: list[dict[str, object]] = []
    for (method, _turbine), output in result.outputs.items():
        by_condition = output.p50_by_condition
        if by_condition is None:
            continue
        rows.extend(
            {
                "mode": mode,
                "method": method,
                "arm": arm.name,
                "exposed": arm.exposed,
                "faulted": arm.fault is not None,
                "condition": row.condition,
                "condition_bin": row.condition_bin,
                "p50_uplift": row.p50_uplift,
            }
            for row in by_condition.itertuples()
        )
    return rows


def run_fixture(
    *,
    modes: Sequence[str] = _MODES,
    include_power_model: bool = True,
    out_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every arm for each mode and return the headline and per-condition tables.

    :return: ``(headline, conditional)`` -- one headline row per ``(mode, method, arm)`` and one
        conditional row per ``(mode, method, arm, condition, condition_bin)``
    """
    root = Path(out_root) if out_root is not None else default_output_root()
    run_dir = root / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    era5_df = build_hot_v0_context(wtg_names=list(FIXTURE_TURBINES)).reanalysis_datasets[0].data
    headline_rows: list[dict[str, object]] = []
    conditional_rows: list[dict[str, object]] = []
    for mode in modes:
        period = analysis_period(mode)  # type: ignore[arg-type]
        logger.info("loading Hill of Towie SCADA %s..%s for %s", *period, list(FIXTURE_TURBINES))
        scada_df, _ = load_hot_scada(
            start_dt=period[0],
            end_dt_excl=period[1],
            wtg_numbers=[int(w[1:]) for w in FIXTURE_TURBINES],
            wtg_names=list(FIXTURE_TURBINES),
        )
        for arm in fixture_arms(mode):  # type: ignore[arg-type]
            logger.info("running %s %s", mode, arm.name)
            result = run_cell(
                mode=mode,  # type: ignore[arg-type]
                arm=arm,
                scada_df=scada_df,
                era5_df=era5_df,
                out_dir=run_dir / f"{mode}_{arm.name}",
                include_power_model=include_power_model,
            )
            headline_rows.extend(
                {
                    "mode": mode,
                    "method": row.method,
                    "arm": arm.name,
                    "exposed": arm.exposed,
                    "faulted": arm.fault is not None,
                    "estimate": row.estimate,
                    "truth": row.truth,
                    "signed_error": row.signed_error,
                }
                for row in result.farm.itertuples()
            )
            conditional_rows.extend(_conditional_rows(result, mode=mode, arm=arm))

    headline = pd.DataFrame(headline_rows)
    conditional = pd.DataFrame(conditional_rows)
    headline.to_csv(run_dir / "headline.csv", index=False)
    conditional.to_csv(run_dir / "conditional.csv", index=False)
    impact_table(headline).to_csv(run_dir / "headline_impact.csv", index=False)
    conditional_impact_table(conditional).to_csv(run_dir / "conditional_impact.csv", index=False)
    logger.info("wrote the fixture results to %s", run_dir)
    return headline, conditional


def default_output_root() -> Path:
    """Return the directory this driver writes under (``WIND_UP_BENCHMARKING_OUTPUT_DIR`` overrides)."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "sensor_fixture"


def impact_table(headline: pd.DataFrame) -> pd.DataFrame:
    """How far each fault moved the headline estimate, in percentage points.

    Movement is measured against the clean cell of the same mode, method and exposure, so an
    exposed arm is never charged for its own configuration's bias. A group with no clean cell is
    skipped rather than half judged.
    """
    if headline.empty:
        return headline
    rows: list[dict[str, object]] = []
    for (mode, method, exposed), group in headline.groupby(["mode", "method", "exposed"]):
        clean = group[~group["faulted"]]
        if len(clean) != 1:
            continue
        base = float(clean["estimate"].iloc[0])
        rows.extend(
            {
                "mode": mode,
                "method": method,
                "exposed": exposed,
                "arm": row.arm,
                "clean_pct": base * 100,
                "estimate_pct": float(row.estimate) * 100,
                "moved_pp": (float(row.estimate) - base) * 100,
            }
            for row in group[group["faulted"]].itertuples()
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["material"] = out["moved_pp"].abs() >= MATERIAL_PP
    return out


def conditional_impact_table(conditional: pd.DataFrame) -> pd.DataFrame:
    """How far each fault moved every per-condition estimate, in percentage points.

    Each bin moves against its own clean value. A bin the clean run never produced is dropped:
    a gain fault re-bins rows, so a faulted run can report bins that have nothing to compare to.
    """
    if conditional.empty:
        return conditional
    keys = ["mode", "method", "exposed", "condition", "condition_bin"]
    clean = conditional[~conditional["faulted"]][[*keys, "p50_uplift"]].rename(columns={"p50_uplift": "clean"})
    faulted = conditional[conditional["faulted"]]
    merged = faulted.merge(clean, on=keys, how="inner")
    if merged.empty:
        return merged
    merged["clean_pct"] = merged["clean"] * 100
    merged["estimate_pct"] = merged["p50_uplift"] * 100
    merged["moved_pp"] = (merged["p50_uplift"] - merged["clean"]) * 100
    merged["material"] = merged["moved_pp"].abs() >= MATERIAL_PP
    return merged[[*keys, "arm", "clean_pct", "estimate_pct", "moved_pp", "material"]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    headline_table, conditional_table = run_fixture()
    impact = impact_table(headline_table)
    print(impact.to_string(index=False))  # noqa: T201 - a driver's whole point is its printed summary
    moved = int(impact["material"].sum()) if not impact.empty else 0
    print(f"\n{moved} of {len(impact)} headline cells moved by >= {MATERIAL_PP} pp")  # noqa: T201
