"""The R3 reference fixture: what does an undeclared reference change cost wind-up as it ships today.

A placebo campaign -- T06 plus nearby turbines whose northing is stable over the period, with no
upgrade injected, so the truth is exactly zero and any movement in an estimate is the injected
change's doing. Each arm re-runs it with one or two references carrying a Cp change of their own:

* pool -- three candidate references with one bad, or five with two bad, so the good references
  stay in the majority in both;
* sign -- ``+5%`` (an improving reference drags the estimate down) and ``-4%`` (a degrading one
  lifts it). The degradation is the realistic case -- a reference is far likelier to pick up a
  problem of its own than an unannounced improvement -- so the five-reference arm uses it.

An injected change stays on once it lands. In prepost it lands at the changeover -- a reference
retrofitted in the same programme as the test turbine, the maximally confounded case. In toggle it
lands halfway through the test period, because a change predating a toggle test is common-mode
across every on and off block and largely cancels; what corrupts a toggle result is a reference
changing *during* the test.

Run it::

    uv run python -m benchmarking.campaigns.reference_fixture

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``reference_fixture``/``<timestamp>/``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
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
from benchmarking.synthetic import ReferenceCpChange
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarking.campaigns.declaration import SyntheticCampaign
    from benchmarking.campaigns.runner import CampaignResult

logger = logging.getLogger(__name__)

# T06 is the measured-best fixture turbine. Its three nearest neighbours whose northing is stable
# over 2017-2018 are the small pool (T05 is nearer but carries real northing steps of its own);
# the large pool adds the next two nearest.
FIXTURE_TEST_WTG = "T06"
REFERENCES_3 = ("T15", "T10", "T08")
REFERENCES_5 = (*REFERENCES_3, "T04", "T02")
_POOLS: dict[int, tuple[str, ...]] = {3: REFERENCES_3, 5: REFERENCES_5}

# The nearest references, so an injected change lands on the most influential turbines.
BAD_REFERENCE = REFERENCES_3[0]
SECOND_BAD_REFERENCE = REFERENCES_3[1]

# The Cp changes injected into a bad reference, and when they land. A reference is far more likely
# to pick up an accidental problem of its own than an unannounced improvement, so the degradation is
# the realistic case and the multi-reference arm uses it. The improvement is also the harder of the
# two to see, since the Cp core clips at rated and attenuates it.
UP_DELTA = 0.05
DOWN_DELTA = -0.04


def _pct(delta: float) -> str:
    """Spell a Cp change for an arm name; arm names reach output paths, so no "%" or sign."""
    return f"{'up' if delta > 0 else 'down'}{round(abs(delta) * 100)}pct"


# Movement in the headline at or above this many percentage points counts as material.
MATERIAL_PP = 0.25

_MODES = ("prepost", "toggle")


@dataclass(frozen=True)
class Arm:
    """One cell of the fixture: a reference pool size and the changes injected into it.

    :param name: the arm's label in the output tables
    :param pool: how many candidate references the campaign declares
    :param changes: the reference Cp changes to inject; empty for a clean cell
    """

    name: str
    pool: int
    changes: tuple[ReferenceCpChange, ...] = field(default_factory=tuple)


def references(pool: int) -> tuple[str, ...]:
    """Return the candidate references of a pool size."""
    if pool not in _POOLS:
        msg = f"unknown pool size {pool!r}; expected one of {sorted(_POOLS)}"
        raise ValueError(msg)
    return _POOLS[pool]


def analysis_period(mode: Literal["prepost", "toggle"]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the whole record the methods see for ``mode``: the baseline plus the campaign."""
    return placebo_analysis_period(mode)


def change_time(mode: Literal["prepost", "toggle"]) -> pd.Timestamp:
    """When an injected reference change lands, for ``mode``.

    Prepost puts it at the changeover -- the maximally confounded case, a reference retrofitted in
    the same programme as the test turbine. Toggle puts it halfway through the test period, because
    a change predating a toggle test is common-mode across every on and off block and largely
    cancels; what corrupts a toggle result is a reference changing *during* the test.
    """
    if mode not in _MODES:
        msg = f"unknown mode {mode!r}; expected 'prepost' or 'toggle'"
        raise ValueError(msg)
    if mode == "prepost":
        return PLACEBO_CAMPAIGN_START
    _, end = analysis_period(mode)
    return PLACEBO_CAMPAIGN_START + (end - PLACEBO_CAMPAIGN_START) / 2


def fixture_arms(mode: Literal["prepost", "toggle"]) -> list[Arm]:
    """Return every arm of the fixture for ``mode``: the clean cells and the injected changes."""
    at = change_time(mode)
    return [
        Arm(name="3ref_clean", pool=3),
        Arm(
            name=f"3ref_{BAD_REFERENCE}_{_pct(UP_DELTA)}",
            pool=3,
            changes=(ReferenceCpChange(turbine=BAD_REFERENCE, at=at, delta=UP_DELTA),),
        ),
        Arm(
            name=f"3ref_{BAD_REFERENCE}_{_pct(DOWN_DELTA)}",
            pool=3,
            changes=(ReferenceCpChange(turbine=BAD_REFERENCE, at=at, delta=DOWN_DELTA),),
        ),
        Arm(name="5ref_clean", pool=5),
        Arm(
            name=f"5ref_{BAD_REFERENCE}_{SECOND_BAD_REFERENCE}_{_pct(DOWN_DELTA)}",
            pool=5,
            changes=(
                ReferenceCpChange(turbine=BAD_REFERENCE, at=at, delta=DOWN_DELTA),
                ReferenceCpChange(turbine=SECOND_BAD_REFERENCE, at=at, delta=DOWN_DELTA),
            ),
        ),
    ]


def _coords(turbines: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Hill of Towie coordinates for ``turbines``."""
    metadata = load_hot_metadata()
    return {
        str(row.Name): (float(row.Latitude), float(row.Longitude))
        for row in metadata.itertuples()
        if str(row.Name) in set(turbines)
    }


def fixture_turbines(pool: int) -> tuple[str, ...]:
    """Every turbine one arm's campaign declares: the test turbine and its candidate references."""
    return (FIXTURE_TEST_WTG, *references(pool))


def fixture_campaign(
    mode: Literal["prepost", "toggle"],
    *,
    arm: Arm,
    coords: dict[str, tuple[float, float]] | None = None,
) -> SyntheticCampaign:
    """Declare one arm of the fixture: the placebo on the arm's turbines, plus its reference changes.

    :param mode: ``"prepost"`` or ``"toggle"``
    :param arm: the cell being declared
    :param coords: turbine coordinates; a placeholder is used when omitted
    """
    return placebo_campaign(
        mode,
        upgraded=[FIXTURE_TEST_WTG],
        turbines=list(fixture_turbines(arm.pool)),
        coords=coords,
        faults=list(arm.changes),
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
    turbines = fixture_turbines(arm.pool)
    campaign = fixture_campaign(mode, arm=arm, coords=_coords(turbines))
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
                "pool": arm.pool,
                "faulted": bool(arm.changes),
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

    # The largest pool covers every arm's turbines, so the SCADA and ERA5 are loaded once per mode.
    all_turbines = fixture_turbines(max(_POOLS))
    era5_df = build_hot_v0_context(wtg_names=list(all_turbines)).reanalysis_datasets[0].data
    headline_rows: list[dict[str, object]] = []
    conditional_rows: list[dict[str, object]] = []
    for mode in modes:
        period = analysis_period(mode)  # type: ignore[arg-type]
        logger.info("loading Hill of Towie SCADA %s..%s for %s", *period, list(all_turbines))
        scada_df, _ = load_hot_scada(
            start_dt=period[0],
            end_dt_excl=period[1],
            wtg_numbers=[int(w[1:]) for w in all_turbines],
            wtg_names=list(all_turbines),
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
                    "pool": arm.pool,
                    "faulted": bool(arm.changes),
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
    return root / "reference_fixture"


def impact_table(headline: pd.DataFrame) -> pd.DataFrame:
    """How far each injected change moved the headline estimate, in percentage points.

    Movement is measured against the clean cell of the same mode, method and pool size, so an arm
    is never charged for the difference between a 3-reference and a 5-reference estimate. A group
    with no clean cell is skipped rather than half judged.
    """
    if headline.empty:
        return headline
    rows: list[dict[str, object]] = []
    for (mode, method, pool), group in headline.groupby(["mode", "method", "pool"]):
        clean = group[~group["faulted"]]
        if len(clean) != 1:
            continue
        base = float(clean["estimate"].iloc[0])
        rows.extend(
            {
                "mode": mode,
                "method": method,
                "pool": pool,
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
    """How far each injected change moved every per-condition estimate, in percentage points.

    Each bin moves against its own clean value, within its own pool size. A bin the clean run
    never produced is dropped: it has nothing to compare to.
    """
    if conditional.empty:
        return conditional
    keys = ["mode", "method", "pool", "condition", "condition_bin"]
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
