"""Rung 1 of the bottom-up ladder: a toggle campaign disguised as a prepost campaign.

One window of Hill of Towie is cut into alternating day blocks. The toggle leg reads it as it
stands, with the odd blocks toggled on. The prepost leg reads it disguised: every timestamp is
remapped, SCADA and reanalysis together, so the even blocks run first as a contiguous pre period
and the odd blocks follow as a contiguous post period.

Both legs therefore contrast the same two halves of the same rows. Nothing is injected, so every
turbine's truth is 0. The halves are built from interleaved days, so over the default window they
hold the same months record for record and meet the same weather, the same seasons and the same
wear. A prepost reading away from 0 here cannot be blamed on what the two periods met; it belongs
to the prepost path.

Every step that reads the whole timeline at once -- the reanalysis lag match, northing discovery --
is settled before the disguise and handed to both legs, so that rearranging the timeline cannot
move it.

The whole farm is read the way a campaign report reads it: one test turbine, every other turbine
estimated against the rest, with the reference screen off so nothing is dropped before the mean.

Run it::

    uv run python -m benchmarking.campaigns.disguise_probe            # both legs
    uv run python -m benchmarking.campaigns.disguise_probe --leg prepost

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``disguise_probe``/``<timestamp>/``.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl

mpl.use("Agg")  # headless: the report writes plots without a display

import numpy as np
import pandas as pd

from benchmarking.baselines.era5_sync import sync_era5
from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.baselines.power_model.features import reference_mean_wind_speed
from benchmarking.campaigns.methods import carried_forward_methods
from benchmarking.campaigns.placebo import PLACEBO_TURBINES, placebo_campaign
from benchmarking.campaigns.runner import CampaignRunner, per_turbine_table
from benchmarking.diagnostics.context import era5_source_label
from benchmarking.harness.northing import era5_direction, north_scada
from benchmarking.synthetic import HOT_COLUMNS, HOT_LAT, HOT_LON, ToggleSchedule
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarking.campaigns.declaration import SyntheticCampaign
    from benchmarking.campaigns.runner import CampaignResult
    from benchmarking.synthetic import ColumnSchema
    from wind_up.layout import Layout

logger = logging.getLogger(__name__)

# Two whole years, so each half is a whole year: the disguise then deals the same number of records
# of every calendar month to the pre period and to the post period.
DISGUISE_WINDOW = (pd.Timestamp("2017-09-01", tz="UTC"), pd.Timestamp("2019-09-01", tz="UTC"))
# One day: long enough that a block holds a whole diurnal cycle, short enough that consecutive
# blocks meet the same weather.
DISGUISE_BLOCK = pd.Timedelta("1D")
DISGUISE_TEST_WTG = "T13"

Leg = Literal["prepost", "toggle"]
LEGS: tuple[Leg, ...] = ("prepost", "toggle")


def _coords(turbines: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Hill of Towie coordinates for ``turbines``."""
    metadata = load_hot_metadata()
    return {
        str(row.Name): (float(row.Latitude), float(row.Longitude))
        for row in metadata.itertuples()
        if str(row.Name) in set(turbines)
    }


def block_count(
    *, window: tuple[pd.Timestamp, pd.Timestamp] = DISGUISE_WINDOW, block: pd.Timedelta = DISGUISE_BLOCK
) -> int:
    """Return how many blocks ``window`` holds.

    :raises ValueError: if the window does not run forwards, is not a whole number of blocks, holds
        an odd number of them, or the block is not a whole number of hours. Each would leave the
        two halves unequal, or move hourly reanalysis off the hour.
    """
    start, end = window
    span = end - start
    if span <= pd.Timedelta(0):
        msg = f"the window must run forwards, got {start} .. {end}"
        raise ValueError(msg)
    if block % pd.Timedelta(hours=1) != pd.Timedelta(0):
        msg = f"a block must be a whole number of hours so hourly reanalysis stays on the hour, got {block}"
        raise ValueError(msg)
    if span % block != pd.Timedelta(0):
        msg = f"the window {start} .. {end} is not a whole number of {block} blocks"
        raise ValueError(msg)
    n_blocks = int(span // block)
    if n_blocks % 2:
        msg = f"the window {start} .. {end} holds {n_blocks} blocks, an odd number, so its halves would differ"
        raise ValueError(msg)
    return n_blocks


def disguised_changeover(
    *, window: tuple[pd.Timestamp, pd.Timestamp] = DISGUISE_WINDOW, block: pd.Timedelta = DISGUISE_BLOCK
) -> pd.Timestamp:
    """Return the changeover the disguised record appears to have: the middle of the window."""
    return window[0] + (block_count(window=window, block=block) // 2) * block


def disguise_timestamps(
    index: pd.DatetimeIndex,
    *,
    window: tuple[pd.Timestamp, pd.Timestamp] = DISGUISE_WINDOW,
    block: pd.Timedelta = DISGUISE_BLOCK,
) -> pd.DatetimeIndex:
    """Map the alternating blocks of ``index`` onto a contiguous first half and second half.

    A block goes to the first half when its number is even and to the second when it is odd.
    Blocks keep their order, and every record keeps its offset within its block, so time of day
    survives untouched.

    :param index: timestamps inside ``window``
    :param window: ``(start, end)`` of the record being disguised, end exclusive
    :param block: the length of one block
    :raises ValueError: if any timestamp falls outside the window
    """
    start, end = window
    n_blocks = block_count(window=window, block=block)
    if len(index) == 0:
        return index
    if index.min() < start or index.max() >= end:
        msg = f"timestamps {index.min()} .. {index.max()} reach outside the window {start} .. {end}"
        raise ValueError(msg)
    number = np.asarray((index - start) // block).astype(int)
    within = index - (start + pd.to_timedelta(number * block))
    moved = np.where(number % 2 == 0, number // 2, n_blocks // 2 + number // 2)
    return pd.DatetimeIndex(start + pd.to_timedelta(moved * block) + within, name=index.name)


def undisguise_timestamps(
    index: pd.DatetimeIndex,
    *,
    window: tuple[pd.Timestamp, pd.Timestamp] = DISGUISE_WINDOW,
    block: pd.Timedelta = DISGUISE_BLOCK,
) -> pd.DatetimeIndex:
    """Return the timestamps :func:`disguise_timestamps` would have moved to ``index``.

    The inverse of the disguise, for reading a disguised run's output back against the record it
    was made from.

    :raises ValueError: if any timestamp falls outside the window
    """
    start, end = window
    n_blocks = block_count(window=window, block=block)
    if len(index) == 0:
        return index
    if index.min() < start or index.max() >= end:
        msg = f"timestamps {index.min()} .. {index.max()} reach outside the window {start} .. {end}"
        raise ValueError(msg)
    number = np.asarray((index - start) // block).astype(int)
    within = index - (start + pd.to_timedelta(number * block))
    half = n_blocks // 2
    source = np.where(number < half, 2 * number, 2 * (number - half) + 1)
    return pd.DatetimeIndex(start + pd.to_timedelta(source * block) + within, name=index.name)


def disguise_frame(
    df: pd.DataFrame,
    *,
    window: tuple[pd.Timestamp, pd.Timestamp] = DISGUISE_WINDOW,
    block: pd.Timedelta = DISGUISE_BLOCK,
) -> pd.DataFrame:
    """Return ``df`` trimmed to ``window``, its timestamps disguised, in the new timestamp order.

    Works on any timestamp-indexed frame, long SCADA and hourly reanalysis alike; pass the same
    window and block to both so they stay aligned record for record.
    """
    inside = df[(df.index >= window[0]) & (df.index < window[1])].copy()
    inside.index = disguise_timestamps(pd.DatetimeIndex(inside.index), window=window, block=block)
    return inside.sort_index(kind="stable")


def matching_toggle(
    *, window: tuple[pd.Timestamp, pd.Timestamp] = DISGUISE_WINDOW, block: pd.Timedelta = DISGUISE_BLOCK
) -> ToggleSchedule:
    """Return the schedule whose on-blocks are exactly the blocks the disguise makes the post period."""
    block_count(window=window, block=block)
    return ToggleSchedule(period=2 * block, start=window[0], start_on=False)


def leg_campaign(
    leg: Leg,
    *,
    test_wtg: str = DISGUISE_TEST_WTG,
    turbines: Sequence[str] | None = None,
    window: tuple[pd.Timestamp, pd.Timestamp] = DISGUISE_WINDOW,
    block: pd.Timedelta = DISGUISE_BLOCK,
) -> SyntheticCampaign:
    """Declare one leg: the whole farm over ``window``, nothing injected, ``test_wtg`` the test turbine.

    The prepost leg changes over at the middle of the window, which is what the disguised record
    shows; the toggle leg toggles every block from the start of it.
    """
    participating = list(PLACEBO_TURBINES if turbines is None else turbines)
    if leg not in LEGS:
        msg = f"unknown leg {leg!r}; expected one of {LEGS}"
        raise ValueError(msg)
    campaign = placebo_campaign(
        leg,
        upgraded=[test_wtg],
        turbines=participating,
        coords=_coords(participating),
    )
    timing: pd.Timestamp | ToggleSchedule = (
        disguised_changeover(window=window, block=block)
        if leg == "prepost"
        else matching_toggle(window=window, block=block)
    )
    # Northing is settled once on the undisguised record by source_northed_scada; an empty table
    # leaves the shared step applying zero and discovering nothing.
    return replace(campaign, upgrade_timing=timing, analysis_period=window, north_offsets=[])


def _readings(result: CampaignResult, *, test_wtg: str) -> pd.DataFrame:
    """Return one row per (method, turbine): the test turbine's estimate and every reference's."""
    per_turbine = per_turbine_table(result)
    tested = per_turbine[["method", "test_wtg", "estimate"]].rename(
        columns={"test_wtg": "turbine", "estimate": "uplift"}
    )
    frames = [tested.assign(role="test", screened=False)]
    stability = result.report.reference_stability
    if not stability.empty:
        frames.append(stability[["method", "turbine", "uplift", "screened"]].assign(role="reference"))
    readings = pd.concat(frames, ignore_index=True)
    readings["reading_pp"] = readings["uplift"] * 100
    readings["test_wtg"] = test_wtg
    readings["truth"] = 0.0
    return readings.drop(columns="uplift")


def source_aligned_era5(
    era5_df: pd.DataFrame,
    *,
    scada_df: pd.DataFrame,
    references: Sequence[str],
    columns: ColumnSchema = HOT_COLUMNS,
) -> pd.DataFrame:
    """Return reanalysis on the SCADA grid, lag-matched to the site over the undisguised record.

    The power model matches reanalysis to the site by the whole-series row shift that best
    correlates its wind speed with the reference mean, and it runs that sweep in timestamp order.
    Once blocks are interleaved every shift straddles block boundaries, which costs more
    correlation than the shallow true peak is worth, so a disguised record picks a different lag
    and its reanalysis features stop lining up with the record's own. Matching once here, before
    the disguise, and handing the result to both legs leaves the model's sweep nothing to shift.

    :param era5_df: hourly reanalysis for the site
    :param scada_df: the undisguised SCADA the lag is matched against
    :param references: the turbines whose mean wind speed is the site signal
    :param columns: the SCADA column schema
    """
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    reference_ws = reference_mean_wind_speed(
        scada_df,
        references=list(references),
        turbine_col=columns.turbine,
        wind_speed_col=columns.wind_speed,
    )
    return sync_era5(era5_df, target_index=index, reference_ws=reference_ws).aligned


def source_northed_scada(
    scada_df: pd.DataFrame,
    *,
    era5_df: pd.DataFrame,
    rated_power_kw: float,
    layout: Layout | None,
    columns: ColumnSchema = HOT_COLUMNS,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Return the SCADA north-calibrated once, over the undisguised record.

    Northing discovery searches a turbine's nacelle position along the timeline for steps.
    Interleaving blocks rearranges that timeline, so the search meets different steps and can
    settle on a different table, leaving the two legs correcting the same rows by different
    amounts. Discovering once here and writing the correction into the nacelle position itself
    hands both legs the same directions; each leg then declares an empty offsets table, so the
    shared step applies zero and discovers nothing.

    :param scada_df: the undisguised SCADA the corrections are discovered on
    :param era5_df: reanalysis on the SCADA grid, the anchor discovery measures against
    :param rated_power_kw: turbine rating, for deciding which rows are usable for northing
    :param layout: the farm layout for the neighbour consensus and the wake-nadir shift, or ``None``
    :param columns: the SCADA column schema
    :param out_dir: where the discovered table and its plots are written
    """
    index = pd.DatetimeIndex(pd.unique(scada_df.index)).sort_values()
    northed = north_scada(
        scada_df,
        columns=columns,
        north_offsets=None,
        rated_power_kw=rated_power_kw,
        layout=layout,
        era5_wd=era5_direction(era5_df, index),
        out_dir=out_dir,
    )
    companion = columns.northed("nacelle_position")
    northed[columns.nacelle_position] = northed[companion]
    return northed.drop(columns=companion)


def source_record(
    *,
    scada_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    references: Sequence[str],
    rated_power_kw: float,
    layout: Layout | None,
    window: tuple[pd.Timestamp, pd.Timestamp] = DISGUISE_WINDOW,
    columns: ColumnSchema = HOT_COLUMNS,
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the ``(scada, reanalysis)`` both legs read: trimmed to ``window``, every shared step settled.

    Two steps read the whole timeline at once, and on an interleaved record each would settle
    somewhere else: the reanalysis lag match and northing discovery. Both run here, once, on the
    undisguised record, so what the legs are left differing by is the prepost path itself.

    :param scada_df: SCADA covering at least ``window``
    :param era5_df: hourly reanalysis for the site
    :param references: the turbines whose mean wind speed the reanalysis lag is matched against
    :param rated_power_kw: turbine rating, for deciding which rows are usable for northing
    :param layout: the farm layout northing discovery uses, or ``None``
    :param window: ``(start, end)`` of the record both legs share, end exclusive
    :param columns: the SCADA column schema
    :param out_dir: where the discovered northing table and its plots are written
    """
    inside = scada_df[(scada_df.index >= window[0]) & (scada_df.index < window[1])]
    aligned = source_aligned_era5(era5_df, scada_df=inside, references=references, columns=columns)
    northed = source_northed_scada(
        inside,
        era5_df=aligned,
        rated_power_kw=rated_power_kw,
        layout=layout,
        columns=columns,
        out_dir=out_dir,
    )
    return northed, aligned


def run_leg(
    leg: Leg,
    *,
    scada_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    out_dir: Path,
    test_wtg: str = DISGUISE_TEST_WTG,
    turbines: Sequence[str] | None = None,
    window: tuple[pd.Timestamp, pd.Timestamp] = DISGUISE_WINDOW,
    block: pd.Timedelta = DISGUISE_BLOCK,
) -> pd.DataFrame:
    """Run one leg end-to-end and return every turbine's reading, each with a truth of 0.

    The prepost leg is handed the disguised record, SCADA and reanalysis both remapped; the toggle
    leg is handed the record as it stands.

    Both frames must be the prepared record :func:`source_record` returns: every step that reads
    the whole timeline at once -- the reanalysis lag match and northing discovery -- is settled
    there, on the undisguised record, so that the disguise cannot move it. Handing this raw SCADA
    and hourly reanalysis instead lets each leg settle those steps for itself, and the legs then
    differ for reasons that have nothing to do with the prepost path.

    :param leg: ``"prepost"`` or ``"toggle"``
    :param scada_df: the prepared SCADA, trimmed to ``window`` and north-calibrated
    :param era5_df: the prepared reanalysis, on the SCADA grid and lag-matched
    :param out_dir: where the run writes
    :param test_wtg: the one turbine declared upgraded; every other turbine is a candidate reference
    :param turbines: every participating turbine; the placebo farm when ``None``
    :param window: ``(start, end)`` of the record both legs share, end exclusive
    :param block: the length of one interleaved block
    """
    campaign = leg_campaign(leg, test_wtg=test_wtg, turbines=turbines, window=window, block=block)
    if leg == "prepost":
        scada_df = disguise_frame(scada_df, window=window, block=block)
        era5_df = disguise_frame(era5_df, window=window, block=block)
    dataset = campaign.generate(scada_df)
    spec = campaign.spec()
    runner = CampaignRunner(
        spec,
        dataset,
        build_methods=lambda wtg: carried_forward_methods(
            spec,
            out_dir=out_dir / wtg,
            era5_hourly_df=era5_df,
            era5_label=era5_source_label(HOT_LAT, HOT_LON),
            reference_screen=False,
            report_reference_uplifts=True,
        ),
    )
    return _readings(runner.run(), test_wtg=test_wtg)


def default_output_root() -> Path:
    """Return the directory this driver writes under (``WIND_UP_BENCHMARKING_OUTPUT_DIR`` overrides)."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "disguise_probe"


def run_disguise_probe(
    *,
    legs: Sequence[Leg] = LEGS,
    test_wtg: str = DISGUISE_TEST_WTG,
    turbines: Sequence[str] | None = None,
    window: tuple[pd.Timestamp, pd.Timestamp] = DISGUISE_WINDOW,
    block: pd.Timedelta = DISGUISE_BLOCK,
    out_root: str | Path | None = None,
) -> pd.DataFrame:
    """Run the requested legs over one loading of the data and return one row per (leg, method, turbine)."""
    participating = list(PLACEBO_TURBINES if turbines is None else turbines)
    root = Path(out_root) if out_root is not None else default_output_root()
    run_dir = root / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    era5_df = build_hot_v0_context(wtg_names=participating).reanalysis_datasets[0].data
    logger.info("loading Hill of Towie SCADA %s..%s for %s", *window, participating)
    scada_df, _ = load_hot_scada(
        start_dt=window[0],
        end_dt_excl=window[1],
        wtg_numbers=[int(w[1:]) for w in participating],
        wtg_names=participating,
    )
    logger.info(
        "%d blocks of %s: the disguised record changes over at %s",
        block_count(window=window, block=block),
        block,
        disguised_changeover(window=window, block=block),
    )
    declared = leg_campaign(LEGS[0], test_wtg=test_wtg, turbines=participating, window=window, block=block)
    scada_df, era5_df = source_record(
        scada_df=scada_df,
        era5_df=era5_df,
        references=declared.candidate_references,
        rated_power_kw=declared.rated_power_kw,
        layout=declared.layout,
        window=window,
        out_dir=run_dir / "northing",
    )

    frames: list[pd.DataFrame] = []
    for leg in legs:
        logger.info("running the %s leg", leg)
        readings = run_leg(
            leg,
            scada_df=scada_df,
            era5_df=era5_df,
            out_dir=run_dir / leg,
            test_wtg=test_wtg,
            turbines=participating,
            window=window,
            block=block,
        )
        readings.insert(0, "leg", leg)
        frames.append(readings)
        pd.concat(frames).to_csv(run_dir / "readings.csv", index=False)
        for method, group in readings.groupby("method"):
            logger.info(
                "%s leg, %s: %d turbines read mean %+.3f pp, median %+.3f pp, sd %.3f pp (truth 0)",
                leg,
                method,
                len(group),
                group["reading_pp"].mean(),
                group["reading_pp"].median(),
                group["reading_pp"].std(),
            )
    all_readings = pd.concat(frames).reset_index(drop=True)
    leg_summary(all_readings).to_csv(run_dir / "leg_summary.csv", index=False)
    logger.info("wrote the disguise probe results to %s", run_dir)
    return all_readings


def leg_summary(readings: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (leg, method): how the whole farm read when every turbine's truth was 0."""
    if readings.empty:
        return readings
    summary = readings.groupby(["leg", "method"], as_index=False).agg(
        n=("reading_pp", "size"),
        mean_pp=("reading_pp", "mean"),
        median_pp=("reading_pp", "median"),
        sd_pp=("reading_pp", "std"),
    )
    worst = readings.assign(abs_pp=readings["reading_pp"].abs()).groupby(["leg", "method"], as_index=False)
    return summary.merge(worst.agg(max_abs_pp=("abs_pp", "max")), on=["leg", "method"], how="left")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leg",
        default="both",
        choices=(*LEGS, "both"),
        help="which leg to run; both by default, so the two readings can be set side by side",
    )
    parser.add_argument("--test-wtg", default=DISGUISE_TEST_WTG, help="the one turbine declared upgraded")
    parser.add_argument(
        "--block",
        default=None,
        help=(
            "the length of one interleaved block, as a pandas offset such as 1D or 4380h; the window must hold "
            "an even, whole number of them. Longer blocks deal the two halves fewer, longer pieces, so at the "
            "long end each half is a season rather than a shuffle of them."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    chosen: tuple[Leg, ...] = LEGS if args.leg == "both" else (args.leg,)
    chosen_block = DISGUISE_BLOCK if args.block is None else pd.Timedelta(args.block)
    probe_readings = run_disguise_probe(legs=chosen, test_wtg=args.test_wtg, block=chosen_block)
    print(leg_summary(probe_readings).to_string(index=False))  # noqa: T201 - a driver's point is its summary
