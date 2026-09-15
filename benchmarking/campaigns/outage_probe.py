"""R4 stage 1: which missing-data shapes crash wind-up, and which return a number in silence.

An information-gathering probe, not a fixture: it asks only whether an estimate is *reached*, and
where it stops when it is not. What a surviving estimate is *worth* is stage 2's question, measured
against a clean control on the placebo.

The arms cover the field failure modes -- a whole-farm SCADA outage, a whole-turbine logging
outage, a single-signal outage, ERA5 columns lost to a download or mapping error, and ERA5 holes --
each in the two shapes that take different code paths:

* **absent** -- the column is not in the frame at all (a mapping error, a renamed tag);
* **empty** -- the column is there and its values are NaN (a logging outage), or the rows are gone.

The distinction is the point: an absent column raises, while a NaN one is handled natively by
LightGBM and passes through without comment.

Each arm runs the real campaign path, so the shared northing step sees the fault too. A failure is
attributed to the deepest repo frame in its traceback, which separates a northing failure from a
`power_model` one.

Run it::

    uv run python -m benchmarking.campaigns.outage_probe

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``outage_probe``/``<timestamp>/``.
"""

from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl

mpl.use("Agg")  # headless: the campaign path writes plots without a display

import pandas as pd

from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.power_model import CURATED_ERA5_EXCLUDE, TUNED_MODEL_PARAMS, PowerModelMethod
from benchmarking.campaigns.placebo import (
    PLACEBO_CAMPAIGN_START,
    placebo_analysis_period,
    placebo_campaign,
)
from benchmarking.campaigns.runner import CampaignRunner
from benchmarking.harness.northing import era5_direction
from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from benchmarking.harness import Method

logger = logging.getLogger(__name__)

# The R2/R3 fixture turbines: T06 and the three nearest whose northing is stable over 2017-2018.
PROBE_TEST_WTG = "T06"
PROBE_REFERENCES = ("T15", "T10", "T08")
PROBE_TURBINES = (PROBE_TEST_WTG, *PROBE_REFERENCES)

# The nearest reference, so a reference-side outage lands on the most influential turbine.
OUTAGE_REFERENCE = "T15"

# Outage windows, named for where they sit relative to the changeover. Stage 1 runs the baseline
# one; stage 2 is where position becomes the variable that matters.
_OUTAGE_DAYS = 30
BASELINE_OUTAGE = (PLACEBO_CAMPAIGN_START - pd.Timedelta(days=120), PLACEBO_CAMPAIGN_START - pd.Timedelta(days=90))
UPGRADED_OUTAGE = (PLACEBO_CAMPAIGN_START + pd.Timedelta(days=90), PLACEBO_CAMPAIGN_START + pd.Timedelta(days=120))


def _role(role: str) -> str:
    """Return the Hill of Towie column naming ``role``; every role the probe faults must be named."""
    HOT_COLUMNS.require_roles([role])
    return str(getattr(HOT_COLUMNS, role))


# The per-turbine value columns a logging outage takes out together.
_TURBINE_SIGNALS = tuple(
    _role(r)
    for r in (
        "active_power",
        "active_power_min",
        "wind_speed",
        "wind_speed_sd",
        "gen_rpm",
        "availability",
        "nacelle_position",
    )
)

# ERA5 columns that are load-bearing rather than incidental: the sync's own two, and the matching
# axes the conditional step bins on.
ERA5_SYNC_COLS = ("wind_speed_100m", "wind_direction_100m")
ERA5_MATCHING_COL = "wind_gusts_10m"
ERA5_INCIDENTAL_COL = "temperature_2m"


def _identity(df: pd.DataFrame) -> pd.DataFrame:
    return df


@dataclass(frozen=True)
class Arm:
    """One cell of the probe: a named data fault applied to the SCADA frame, the ERA5 frame, or both.

    :param name: the cell's label in the results table
    :param what: one line describing the fault, carried into the table so the CSV reads alone
    :param shape: ``absent`` (the column is gone) or ``empty`` (present but NaN, or rows removed)
    :param scada: transform applied to the SCADA frame before the campaign generates from it
    :param era5: transform applied to the hourly ERA5 frame
    """

    name: str
    what: str
    shape: Literal["clean", "absent", "empty"] = "clean"
    scada: Callable[[pd.DataFrame], pd.DataFrame] = _identity
    era5: Callable[[pd.DataFrame], pd.DataFrame] = _identity


@dataclass
class ArmOutcome:
    """What one arm did: an estimate, or where it stopped.

    :param reached_estimate: whether the campaign returned a number at all
    :param estimate: the power model's farm estimate when it did
    :param error_type: the exception class name when it did not
    :param error_message: the exception's first line
    :param failed_in: ``module:function:line`` of the deepest repo frame in the traceback
    """

    reached_estimate: bool
    estimate: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    failed_in: str | None = None


def _window_mask(index: pd.DatetimeIndex, window: tuple[pd.Timestamp, pd.Timestamp]) -> pd.Series:
    """Boolean over ``index``: is this timestamp inside the half-open outage window."""
    start, end = window
    return pd.Series((index >= start) & (index < end), index=index)


def null_signals(
    df: pd.DataFrame,
    *,
    turbines: Sequence[str],
    columns: Sequence[str],
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    """Return ``df`` with ``columns`` set to NaN for ``turbines`` inside ``window`` (a logging outage)."""
    out = df.copy()
    rows = (
        _window_mask(pd.DatetimeIndex(out.index), window).to_numpy()
        & out[HOT_COLUMNS.turbine].isin(list(turbines)).to_numpy()
    )
    present = [c for c in columns if c in out.columns]
    out.loc[rows, present] = float("nan")
    return out


def drop_rows(df: pd.DataFrame, *, turbines: Sequence[str], window: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """Return ``df`` with ``turbines``' records inside ``window`` removed outright (no row at all)."""
    rows = (
        _window_mask(pd.DatetimeIndex(df.index), window).to_numpy()
        & df[HOT_COLUMNS.turbine].isin(list(turbines)).to_numpy()
    )
    return df[~rows]


def drop_columns(df: pd.DataFrame, *, columns: Sequence[str]) -> pd.DataFrame:
    """Return ``df`` without ``columns`` (a mapping error, or a tag that was never delivered)."""
    return df.drop(columns=[c for c in columns if c in df.columns])


def drop_turbine(df: pd.DataFrame, *, turbine: str) -> pd.DataFrame:
    """Return ``df`` without ``turbine``'s records at all (a turbine missing from the whole delivery)."""
    return df[df[HOT_COLUMNS.turbine] != turbine]


def null_era5(df: pd.DataFrame, *, window: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """Return hourly ERA5 with every column NaN inside ``window`` (a reanalysis hole)."""
    out = df.copy()
    rows = _window_mask(pd.DatetimeIndex(out.index), window).to_numpy()
    out.loc[rows, :] = float("nan")
    return out


def drop_era5_rows(df: pd.DataFrame, *, window: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """Return hourly ERA5 with the timestamps inside ``window`` missing from the index entirely."""
    return df[~_window_mask(pd.DatetimeIndex(df.index), window).to_numpy()]


def probe_arms() -> list[Arm]:
    """Return every cell of the stage-1 matrix, clean control first."""
    ref = OUTAGE_REFERENCE
    test = PROBE_TEST_WTG
    return [
        Arm(name="clean", what="no fault"),
        # 1. whole-farm SCADA outage
        Arm(
            name="farm_empty",
            what=f"every turbine, every signal NaN for {_OUTAGE_DAYS}d of baseline",
            shape="empty",
            scada=lambda df: null_signals(
                df, turbines=PROBE_TURBINES, columns=_TURBINE_SIGNALS, window=BASELINE_OUTAGE
            ),
        ),
        Arm(
            name="farm_rows_gone",
            what=f"every turbine's records absent for {_OUTAGE_DAYS}d of baseline",
            shape="empty",
            scada=lambda df: drop_rows(df, turbines=PROBE_TURBINES, window=BASELINE_OUTAGE),
        ),
        # 2. whole-turbine logging outage, reference and test turbine
        Arm(
            name="ref_empty",
            what=f"{ref} (nearest reference), every signal NaN for {_OUTAGE_DAYS}d of baseline",
            shape="empty",
            scada=lambda df: null_signals(df, turbines=[ref], columns=_TURBINE_SIGNALS, window=BASELINE_OUTAGE),
        ),
        Arm(
            name="ref_rows_gone",
            what=f"{ref}'s records absent for {_OUTAGE_DAYS}d of baseline",
            shape="empty",
            scada=lambda df: drop_rows(df, turbines=[ref], window=BASELINE_OUTAGE),
        ),
        Arm(
            name="ref_absent_entirely",
            what=f"{ref} missing from the delivery altogether",
            shape="absent",
            scada=lambda df: drop_turbine(df, turbine=ref),
        ),
        Arm(
            name="test_empty",
            what=f"{test} (test turbine), every signal NaN for {_OUTAGE_DAYS}d of baseline",
            shape="empty",
            scada=lambda df: null_signals(df, turbines=[test], columns=_TURBINE_SIGNALS, window=BASELINE_OUTAGE),
        ),
        Arm(
            name="test_empty_upgraded",
            what=f"{test}, every signal NaN for {_OUTAGE_DAYS}d of the upgraded period",
            shape="empty",
            scada=lambda df: null_signals(df, turbines=[test], columns=_TURBINE_SIGNALS, window=UPGRADED_OUTAGE),
        ),
        # 3. single turbine-signal outage
        Arm(
            name="ref_power_empty",
            what=f"{ref}'s active power alone NaN for {_OUTAGE_DAYS}d of baseline",
            shape="empty",
            scada=lambda df: null_signals(df, turbines=[ref], columns=[_role("active_power")], window=BASELINE_OUTAGE),
        ),
        Arm(
            name="power_min_absent",
            what="the active-power minimum column missing farm-wide",
            shape="absent",
            scada=lambda df: drop_columns(df, columns=[_role("active_power_min")]),
        ),
        Arm(
            name="availability_absent",
            what="the availability column missing farm-wide",
            shape="absent",
            scada=lambda df: drop_columns(df, columns=[_role("availability")]),
        ),
        Arm(
            name="nacelle_position_absent",
            what="the nacelle position column missing farm-wide (northing has nothing to correct)",
            shape="absent",
            scada=lambda df: drop_columns(df, columns=[_role("nacelle_position")]),
        ),
        Arm(
            name="ref_direction_empty",
            what=f"{ref}'s nacelle position alone NaN for the whole record",
            shape="empty",
            scada=lambda df: null_signals(
                df,
                turbines=[ref],
                columns=[_role("nacelle_position")],
                window=(pd.Timestamp.min.tz_localize("UTC"), pd.Timestamp.max.tz_localize("UTC")),
            ),
        ),
        # 4. ERA5 columns lost to a download or mapping error
        Arm(
            name="era5_incidental_absent",
            what=f"ERA5 {ERA5_INCIDENTAL_COL} column missing",
            shape="absent",
            era5=lambda df: drop_columns(df, columns=[ERA5_INCIDENTAL_COL]),
        ),
        Arm(
            name="era5_matching_absent",
            what=f"ERA5 {ERA5_MATCHING_COL} column missing (a conditional matching axis)",
            shape="absent",
            era5=lambda df: drop_columns(df, columns=[ERA5_MATCHING_COL]),
        ),
        Arm(
            name="era5_sync_absent",
            what=f"ERA5 {' + '.join(ERA5_SYNC_COLS)} missing (what the lag sync locks onto)",
            shape="absent",
            era5=lambda df: drop_columns(df, columns=ERA5_SYNC_COLS),
        ),
        # 5. ERA5 holes
        Arm(
            name="era5_hole_1mo",
            what=f"every ERA5 column NaN for {_OUTAGE_DAYS}d of baseline",
            shape="empty",
            era5=lambda df: null_era5(df, window=BASELINE_OUTAGE),
        ),
        Arm(
            name="era5_hole_3mo",
            what="every ERA5 column NaN for 3 months of baseline",
            shape="empty",
            era5=lambda df: null_era5(df, window=(BASELINE_OUTAGE[0], BASELINE_OUTAGE[0] + pd.Timedelta(days=90))),
        ),
        Arm(
            name="era5_rows_gone",
            what=f"ERA5 timestamps absent from the index for {_OUTAGE_DAYS}d of baseline",
            shape="empty",
            era5=lambda df: drop_era5_rows(df, window=BASELINE_OUTAGE),
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


def _power_model_only(spec: object, *, out_dir: Path, era5_hourly_df: pd.DataFrame, seed: int) -> list[Method]:
    """Build just the power model for one turbine: it is the only method R4 is asking about."""
    return [
        PowerModelMethod(
            columns=HOT_COLUMNS,
            baseline_rated_power_kw=spec.rated_power_kw,  # type: ignore[attr-defined]
            era5_hourly_df=era5_hourly_df,
            seed=seed,
            era5_exclude=CURATED_ERA5_EXCLUDE,
            availability_feature=False,
            model_params=dict(TUNED_MODEL_PARAMS),
            out_dir=out_dir / "power_model",
            save_plots=True,
        )
    ]


def _deepest_repo_frame(exc: BaseException) -> str | None:
    """Return ``module:function:line`` of the deepest traceback frame in this repo's own code.

    Installed packages are skipped: a bare ``KeyError`` raised inside pandas is attributed to the
    repo line that asked for the missing column, which is the one worth reading.
    """
    root = Path(__file__).resolve().parents[2]
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        path = Path(frame.filename).resolve()
        if root in path.parents and "site-packages" not in path.parts and ".venv" not in path.parts:
            return f"{path.relative_to(root)}:{frame.name}:{frame.lineno}"
    return None


def run_arm(
    arm: Arm,
    *,
    mode: Literal["prepost", "toggle"],
    scada_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    out_dir: Path,
    seed: int = 0,
) -> ArmOutcome:
    """Apply ``arm``'s fault, run the campaign, and record the estimate or where it stopped.

    ``seed`` drives the power model's holdout split and LightGBM state, so repeating an arm across
    seeds gives the estimator-noise floor a fault's movement has to clear to mean anything.
    """
    faulted_scada = arm.scada(scada_df)
    faulted_era5 = arm.era5(era5_df)
    try:
        campaign = placebo_campaign(
            mode,
            upgraded=[PROBE_TEST_WTG],
            turbines=list(PROBE_TURBINES),
            coords=_coords(PROBE_TURBINES),
        )
        dataset = campaign.generate(faulted_scada)
        spec = campaign.spec()
        index = pd.DatetimeIndex(dataset.synthetic_df.index.unique()).sort_values()
        runner = CampaignRunner(
            spec,
            dataset,
            build_methods=lambda wtg: _power_model_only(
                spec, out_dir=out_dir / wtg, era5_hourly_df=faulted_era5, seed=seed
            ),
            era5_wd=era5_direction(faulted_era5, index),
        )
        result = runner.run()
    except Exception as exc:  # noqa: BLE001 - the probe's whole purpose is to see what escapes
        logger.info("%s %s stopped: %s: %s", mode, arm.name, type(exc).__name__, exc)
        return ArmOutcome(
            reached_estimate=False,
            error_type=type(exc).__name__,
            error_message=str(exc).splitlines()[0] if str(exc) else "",
            failed_in=_deepest_repo_frame(exc),
        )
    estimate = float(result.farm.set_index("method").loc["power_model", "estimate"])
    logger.info("%s %s reached an estimate of %+.3f%%", mode, arm.name, estimate * 100)
    return ArmOutcome(reached_estimate=True, estimate=estimate)


def run_probe(
    *,
    modes: Sequence[str] = ("prepost",),
    arms: Sequence[str] | None = None,
    seeds: Sequence[int] = (0,),
    out_root: str | Path | None = None,
) -> pd.DataFrame:
    """Run the matrix and return one row per ``(mode, arm, seed)``.

    :param modes: the campaign modes to run; stage 1 defaults to prepost alone
    :param arms: run only these arm names (the clean control is always worth keeping); all when None
    :param seeds: repeat every arm once per seed, which turns the table into a noise floor plus the
        movement each fault adds to it
    :param out_root: where the run folder is written; the driver's default root when None
    """
    root = Path(out_root) if out_root is not None else default_output_root()
    run_dir = root / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    era5_df = build_hot_v0_context(wtg_names=list(PROBE_TURBINES)).reanalysis_datasets[0].data
    selected = [a for a in probe_arms() if arms is None or a.name in set(arms)]
    rows: list[dict[str, object]] = []
    for mode in modes:
        period = placebo_analysis_period(mode)  # type: ignore[arg-type]
        logger.info("loading Hill of Towie SCADA %s..%s for %s", *period, list(PROBE_TURBINES))
        scada_df, _ = load_hot_scada(
            start_dt=period[0],
            end_dt_excl=period[1],
            wtg_numbers=[int(w[1:]) for w in PROBE_TURBINES],
            wtg_names=list(PROBE_TURBINES),
        )
        for seed in seeds:
            for arm in selected:
                logger.info("running %s %s seed=%d -- %s", mode, arm.name, seed, arm.what)
                started = pd.Timestamp.now()
                outcome = run_arm(
                    arm,
                    mode=mode,  # type: ignore[arg-type]
                    scada_df=scada_df,
                    era5_df=era5_df,
                    out_dir=run_dir / f"{mode}_{arm.name}_seed{seed}",
                    seed=seed,
                )
                rows.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "arm": arm.name,
                        "shape": arm.shape,
                        "what": arm.what,
                        "reached_estimate": outcome.reached_estimate,
                        "estimate": outcome.estimate,
                        "error_type": outcome.error_type,
                        "error_message": outcome.error_message,
                        "failed_in": outcome.failed_in,
                        "seconds": (pd.Timestamp.now() - started).total_seconds(),
                    }
                )
                pd.DataFrame(rows).to_csv(run_dir / "outcomes.csv", index=False)
    logger.info("wrote the probe results to %s", run_dir)
    return pd.DataFrame(rows)


def default_output_root() -> Path:
    """Return the directory this driver writes under (``WIND_UP_BENCHMARKING_OUTPUT_DIR`` overrides)."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "outage_probe"


def main() -> None:
    """Run the stage-1 matrix and log the outcome table."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    outcomes = run_probe()
    logger.info("\n%s", outcomes[["mode", "arm", "shape", "reached_estimate", "estimate", "failed_in"]].to_string())


if __name__ == "__main__":
    main()
