"""Inspect one synthetic wake-steering toggle case across every method, with plots on.

The wake-steering analogue of :mod:`benchmarking.baselines.inspect_prepost_hard_case`. It builds
one synthetic dataset in which **T01 steers to benefit T04** (the single geometry-derived pair that
survives full-farm wake-blocking; nadir ~222 deg, 5 rotor diameters apart), with T02/T03/T05/T07
present as references, then runs naive_ratio + power_model (+ v0) **unmodified** on the identical
``MethodInput`` for each of the two participating turbines, ``save_plots=True``, into its own
subfolder of one timestamped run dir. T01 is the steering turbine (an energy *loss*); T04 the
benefitting turbine (a *gain*), so the run reports upwind and downwind uplift side by side against
the injected truth. No net (pair) comparison — the methods have no net-uplift code.

Data window is 2016-01-01..2019-01-01 with a 100-min (50 on / 50 off) toggle test starting
2018-01-01, so 2016-2017 are untreated pre-upgrade baseline. Wind direction and stability are not
harness conditional axes (only ws/ti/power), so the true direction/stability/wind-speed structure is
surfaced via the synthetic ground-truth wake plots under ``ground_truth/`` for eyeballing against
each method's ws/ti/power conditional plots.

``wd_filter=True`` turns on a **temporary** script-level hack: keep only timestamps whose calibrated
*original* (unmodified) T01 yaw sits within the steering sector (``nadir +/- (wd_width/2 + margin)``),
so the toggle contrast is concentrated on the data where steering actually happens. Both the method
input and the comparison ground-truth target flow from the filtered dataset; the ``ground_truth/``
structural plots stay on the full dataset. The run directory name gets a ``_wdfilter`` postfix so the
output confirms the hack was used. This is not the real per-method direction filtering (follow-up
work) — just a way to see ahead.

Run it::

    uv run python -m benchmarking.baselines.inspect_wake_steering_case

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``inspect_wake_steering``/``<timestamp>[_wdfilter]/``
(``T01/`` and ``T04/`` each with ``naive/`` ``power_model/`` ``v0/`` subfolders, a ``ground_truth/``
folder, a ``comparison_summary.csv`` and ``run.log``). The first run downloads + caches the Hill of
Towie SCADA (Zenodo) and ERA5 (Open-Meteo); the ``ml`` group is needed for the power model and v0
needs the wind_up pipeline.

``include_v0=True`` (the function default) runs a full wind_up toggle assessment per turbine — tens of
minutes each — so the bare ``python -m`` entry point runs the two fast methods only; pass
``include_v0=True`` for the full three-method comparison.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless: the structural plots need no display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from benchmarking.baselines.hot_context import NORTHING_YAML, build_hot_v0_context
from benchmarking.baselines.inspect_prepost_hard_case import conditional_truth_vs_estimate
from benchmarking.baselines.naive_ratio import NaiveRatioMethod
from benchmarking.baselines.power_model import CURATED_ERA5_EXCLUDE, TUNED_MODEL_PARAMS, PowerModelMethod
from benchmarking.baselines.v0_binned import V0BinnedMethod
from benchmarking.harness import (
    CONDITIONS,
    Method,
    MethodInput,
    MethodOutput,
    condition_bins,
    plot_conditional_uplift,
)
from benchmarking.harness.northing import era5_direction, north_scada
from benchmarking.synthetic import (
    HOT_COLUMNS,
    HOT_RATED_POWER_KW,
    SyntheticDataset,
    ToggleSchedule,
    WakeSteering,
    generate_dataset,
    north_calibrated_direction,
    plot_wake_steering_by_direction,
    plot_wake_steering_heatmaps,
    plot_wake_steering_stability,
    treated_mask,
)
from benchmarking.synthetic.geometry import wrap180
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarking.baselines.hot_context import HotV0Context

logger = logging.getLogger(__name__)

# The T01 -> T04 steering pair plus the four references, and their Hill of Towie turbine numbers.
UPSTREAM = "T01"
DOWNSTREAM = "T04"
PARTICIPANTS = (UPSTREAM, DOWNSTREAM)
REFERENCES = ("T02", "T03", "T05", "T07")
SUBSET = ("T01", "T02", "T03", "T04", "T05", "T07")
WTG_NUMBERS = [1, 2, 3, 4, 5, 7]

# 2016-2017 pre-upgrade baseline; a 100-min (50 on / 50 off) toggle test through 2018.
START_DT = pd.Timestamp("2016-01-01", tz="UTC")
END_DT_EXCL = pd.Timestamp("2019-01-01", tz="UTC")
TOGGLE_START = pd.Timestamp("2018-01-01", tz="UTC")
TOGGLE_PERIOD = pd.Timedelta(minutes=100)

DEFAULT_WD_MARGIN_DEG = 3.0


def default_output_root() -> Path:
    """Return the directory this driver writes its outputs under.

    Overridable via ``WIND_UP_BENCHMARKING_OUTPUT_DIR``; defaults to
    ``~/temp/wind-up-benchmarking/inspect_wake_steering``.
    """
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "inspect_wake_steering"


def _load_north_offsets(turbines: Sequence[str]) -> list[tuple[str, pd.Timestamp, float]]:
    """Step-applied north offsets for ``turbines`` from the vendored optimized-northing YAML (UTC)."""
    data = yaml.safe_load(NORTHING_YAML.read_text())
    return [
        (str(name), pd.Timestamp(ts, tz="UTC"), float(offset))
        for (name, ts, offset) in data
        if str(name) in set(turbines)
    ]


def _build_dataset(scada_df: pd.DataFrame, metadata_df: pd.DataFrame) -> tuple[SyntheticDataset, WakeSteering]:
    """Inject the T01 -> T04 wake-steering toggle campaign and return the dataset and upgrade."""
    coords = {str(row.Name): (float(row.Latitude), float(row.Longitude)) for row in metadata_df.itertuples()}
    steering = WakeSteering(
        coords=coords, test_wtgs=list(PARTICIPANTS), north_offsets=_load_north_offsets(PARTICIPANTS)
    )
    schedule = ToggleSchedule(period=TOGGLE_PERIOD, start=TOGGLE_START)
    dataset = generate_dataset(
        scada_df=scada_df,
        test_wtgs=list(PARTICIPANTS),
        upgrades=[steering],
        mode="toggle",
        upgrade_timing=schedule,
        seed=0,
    )
    return dataset, steering


def _pair_sector(steering: WakeSteering, *, margin_deg: float) -> tuple[float, float]:
    """Return the (nadir, half-width) of the T01 -> T04 steering sector, half = wd_width/2 + margin."""
    pair = next(p for p in steering.pairs if p.upstream == UPSTREAM and p.downstream == DOWNSTREAM)
    return pair.nadir_bearing, steering.wd_width / 2.0 + margin_deg


def _in_sector_timestamps(dataset: SyntheticDataset, *, nadir: float, half_width: float) -> pd.DatetimeIndex:
    """Timestamps whose calibrated ORIGINAL (unsteered) T01 yaw lies within the steering sector.

    Uses the untouched ``original_df`` nacelle position, so the direction is treatment-invariant by
    construction (steering only alters the synthetic frame's treated rows).
    """
    up = dataset.original_df[dataset.original_df[HOT_COLUMNS.turbine] == UPSTREAM]
    index = pd.DatetimeIndex(up.index)
    direction = north_calibrated_direction(
        index,
        up[HOT_COLUMNS.nacelle_position].to_numpy(dtype=float),
        turbine=UPSTREAM,
        north_offsets=_load_north_offsets([UPSTREAM]),
    )
    in_sector = np.abs(wrap180(direction - nadir)) <= half_width
    return index[in_sector]


def _filter_dataset(dataset: SyntheticDataset, keep: pd.DatetimeIndex) -> SyntheticDataset:
    """Restrict both frames of ``dataset`` to the ``keep`` timestamps (method input + truth stay aligned)."""
    keep_set = pd.Index(keep)
    return replace(
        dataset,
        synthetic_df=dataset.synthetic_df[dataset.synthetic_df.index.isin(keep_set)],
        original_df=dataset.original_df[dataset.original_df.index.isin(keep_set)],
    )


def _power_model(out_dir: Path, era5_hourly_df: pd.DataFrame) -> PowerModelMethod:
    """Construct the HoT-configured power model (accepted tuned defaults; conditional uplift on)."""
    return PowerModelMethod(
        columns=HOT_COLUMNS,
        baseline_rated_power_kw=HOT_RATED_POWER_KW,
        era5_hourly_df=era5_hourly_df,
        availability_feature=False,
        era5_exclude=CURATED_ERA5_EXCLUDE,
        model_params=dict(TUNED_MODEL_PARAMS),
        out_dir=out_dir,
        save_plots=True,
    )


def _fast_methods(out_dir: Path, era5_hourly_df: pd.DataFrame) -> list[Method]:
    """Return the fast per-turbine methods (naive + power_model), each writing plots into a subfolder."""
    return [
        NaiveRatioMethod(columns=HOT_COLUMNS, out_dir=out_dir / "naive", save_plots=True),
        _power_model(out_dir / "power_model", era5_hourly_df),
    ]


def _run_v0_pair(
    context: HotV0Context,
    *,
    dataset: SyntheticDataset,
    schedule: ToggleSchedule,
    truths: dict[str, float],
    out_dir: Path,
) -> pd.DataFrame:
    """Run v0 **once** over both steering participants together and return their tidy comparison rows.

    Both turbines are the run's test turbines (so neither is a reference for the other), with the
    wake-steering settings on: ``filter_all_test_wtgs_together`` (shared filtered timebase),
    ``require_ref_wake_free`` (a reference behind a steering turbine would bias the contrast), and a
    realistic on/off-block pairing filter matched to the toggle half-period.
    """
    v0 = V0BinnedMethod(
        context,
        scratch_dir=out_dir,
        save_plots=True,
        filter_all_test_wtgs_together=True,
        require_ref_wake_free=True,
        pairing_filter_method="any_within_timedelta",
        pairing_filter_timedelta_seconds=int(TOGGLE_PERIOD.total_seconds() // 2),
    )
    mi = MethodInput(
        scada_df=dataset.synthetic_df,
        test_wtg=PARTICIPANTS[0],
        upgrade_timing=schedule,
        turbine_col=HOT_COLUMNS.turbine,
    )
    start = time.perf_counter()
    outputs = v0.estimate_multi(mi, test_wtgs=list(PARTICIPANTS))
    wall_time_s = time.perf_counter() - start  # a single shared run; recorded on each turbine's row
    rows = []
    for wtg in PARTICIPANTS:
        estimate = outputs[wtg].p50_overall
        truth = truths[wtg]
        logger.info(
            "%-12s %s estimate %+.3f%%  truth %+.3f%%  error %+.3f%%  (single co-run, %.1fs)",
            v0.name,
            wtg,
            100 * estimate,
            100 * truth,
            100 * (estimate - truth),
            wall_time_s,
        )
        rows.append(
            {
                "test_wtg": wtg,
                "method": v0.name,
                "estimate": estimate,
                "truth": truth,
                "signed_error": estimate - truth,
                "wall_time_s": wall_time_s,
            }
        )
    return pd.DataFrame(rows)


def _truth(dataset: SyntheticDataset, *, test_wtg: str, schedule: ToggleSchedule) -> float:
    """Overall ground-truth uplift for ``test_wtg`` over its toggle-treated rows in ``dataset``."""
    test_index = pd.DatetimeIndex(dataset.synthetic_df[dataset.synthetic_df[HOT_COLUMNS.turbine] == test_wtg].index)
    mask = treated_mask(test_index, schedule)
    return dataset.true_uplift(test_wtg=test_wtg, mask=mask).overall


def _run_methods(
    methods: list[Method], *, mi: MethodInput, truth: float
) -> tuple[pd.DataFrame, dict[str, MethodOutput]]:
    """Run every method on the identical input; return the tidy comparison and each method's output."""
    rows = []
    outputs: dict[str, MethodOutput] = {}
    for method in methods:
        start = time.perf_counter()
        output = method.estimate(mi)
        wall_time_s = time.perf_counter() - start
        outputs[method.name] = output
        estimate = output.p50_overall
        logger.info(
            "%-12s %s estimate %+.3f%%  truth %+.3f%%  error %+.3f%%  (%.1fs)",
            method.name,
            mi.test_wtg,
            100 * estimate,
            100 * truth,
            100 * (estimate - truth),
            wall_time_s,
        )
        rows.append(
            {
                "test_wtg": mi.test_wtg,
                "method": method.name,
                "estimate": estimate,
                "truth": truth,
                "signed_error": estimate - truth,
                "wall_time_s": wall_time_s,
            }
        )
    return pd.DataFrame(rows), outputs


def _plot_conditional_uplift(
    dataset: SyntheticDataset,
    output: MethodOutput,
    *,
    test_wtg: str,
    schedule: ToggleSchedule,
    out_dir: Path,
) -> None:
    """Overlay the power_model per-condition estimate against per-condition truth for one turbine."""
    if output.p50_by_condition is None:
        logger.warning("power_model returned no p50_by_condition for %s; skipping conditional plots", test_wtg)
        return
    test_index = pd.DatetimeIndex(dataset.synthetic_df[dataset.synthetic_df[HOT_COLUMNS.turbine] == test_wtg].index)
    mask = treated_mask(test_index, schedule)
    truth_by_condition = {
        c: dataset.true_uplift(
            test_wtg=test_wtg, mask=mask, by=c, bins=condition_bins(c, rated_power_kw=HOT_RATED_POWER_KW)
        ).by_condition
        for c in CONDITIONS
    }
    clean = {c: df for c, df in truth_by_condition.items() if df is not None}
    if not clean:
        return
    frame = conditional_truth_vs_estimate(output, clean, method_name="power_model")
    for c in clean:
        plot_conditional_uplift(
            frame,
            condition=c,
            save_path=out_dir / f"conditional_uplift_{c}.png",
            title=f"Conditional uplift ({c}) — {test_wtg} (power_model vs truth)",
        )


def _plot_truth_uplift_vs_ws(dataset: SyntheticDataset, *, test_wtg: str, save_path: Path) -> None:
    """Write a ground-truth-only uplift-vs-wind-speed plot for one turbine (changed records)."""
    table = dataset.true_uplift(test_wtg=test_wtg, by="ws", bins=condition_bins("ws")).by_condition
    assert table is not None  # by="ws" always returns a table  # noqa: S101
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, table["true_uplift"].to_numpy(dtype=float) * 100.0, marker="o", color="k")
    ax.set_xticks(x)
    ax.set_xticklabels(table["condition_bin"].astype(str), rotation=45, ha="right")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("wind speed [m/s]")
    ax.set_ylabel("true uplift [%]")
    # Over steering-active (changed) records only, so this is the undiluted physical effect shape --
    # larger than the all-treated-rows campaign truth the methods are scored against.
    ax.set_title(f"Ground-truth uplift vs wind speed — {test_wtg} (steering-active records)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _write_ground_truth_plots(dataset: SyntheticDataset, *, out_dir: Path) -> None:
    """Write the pair's direction/heatmap/stability wake plots and each turbine's uplift-vs-ws plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    net = dataset.true_net_uplift(upstream=UPSTREAM, downstream=DOWNSTREAM)
    suffix = f"{UPSTREAM}_to_{DOWNSTREAM}"
    plt.close(
        plot_wake_steering_by_direction(
            dataset,
            upstream=UPSTREAM,
            downstream=DOWNSTREAM,
            save_path=out_dir / f"steering_{suffix}.png",
            title=f"Wake steering {UPSTREAM} -> {DOWNSTREAM} (net {net:+.2%})",
        )
    )
    plt.close(
        plot_wake_steering_heatmaps(
            dataset,
            upstream=UPSTREAM,
            downstream=DOWNSTREAM,
            save_path=out_dir / f"heatmaps_{suffix}.png",
            title=f"Wake steering {UPSTREAM} -> {DOWNSTREAM}: heat maps (net {net:+.2%})",
        )
    )
    plt.close(
        plot_wake_steering_stability(
            dataset,
            upstream=UPSTREAM,
            downstream=DOWNSTREAM,
            save_path=out_dir / f"stability_{suffix}.png",
            title=f"Wake steering {UPSTREAM} -> {DOWNSTREAM}: stability modulation (net {net:+.2%})",
        )
    )
    for wtg in PARTICIPANTS:
        _plot_truth_uplift_vs_ws(dataset, test_wtg=wtg, save_path=out_dir / f"uplift_vs_ws_{wtg}.png")


def _run_dir(out_root: Path, *, wd_filter: bool) -> Path:
    """Create and return a timestamped run dir (a ``_wdfilter`` postfix when the hack is on)."""
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    name = f"wake_steering_case_{ts}" + ("_wdfilter" if wd_filter else "")
    run_dir = out_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _add_log_file(run_dir: Path) -> logging.Handler:
    """Attach a run.log file handler to the root logger and return it."""
    handler = logging.FileHandler(run_dir / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def inspect_wake_steering_case(
    *,
    wd_filter: bool = False,
    wd_margin_deg: float = DEFAULT_WD_MARGIN_DEG,
    include_v0: bool = True,
    out_root: str | Path | None = None,
) -> pd.DataFrame:
    """Run every method on the T01 -> T04 wake-steering toggle case (both turbines) with plots on.

    :param wd_filter: keep only timestamps whose calibrated original T01 yaw is in the steering
        sector (a temporary script-level hack; postfixes the run dir with ``_wdfilter``)
    :param wd_margin_deg: extra half-width added either side of the ``wd_width/2`` sector edge
    :param include_v0: also run the slow v0 baseline (a full wind_up toggle run per turbine)
    :param out_root: base output dir; defaults to :func:`default_output_root`
    :return: per-turbine per-method ``estimate``/``truth``/``signed_error``/``wall_time_s``
    """
    output_root = Path(out_root) if out_root is not None else default_output_root()
    run_dir = _run_dir(output_root, wd_filter=wd_filter)
    handler = _add_log_file(run_dir)
    try:
        logger.info("Loading Hill of Towie SCADA %s..%s for %s", START_DT, END_DT_EXCL, list(SUBSET))
        scada_df, _ = load_hot_scada(
            start_dt=START_DT, end_dt_excl=END_DT_EXCL, wtg_numbers=WTG_NUMBERS, wtg_names=list(SUBSET)
        )
        metadata_df = load_hot_metadata()
        context = build_hot_v0_context(wtg_names=list(SUBSET))
        era5 = context.reanalysis_datasets[0].data

        full_dataset, steering = _build_dataset(scada_df, metadata_df)
        # The shared northing step, discovering as it does on every other path -- the methods read
        # a north-calibrated direction wind-up worked out, not the table the injection was gated on.
        full_dataset = replace(
            full_dataset,
            synthetic_df=north_scada(
                full_dataset.synthetic_df,
                columns=HOT_COLUMNS,
                north_offsets=None,
                rated_power_kw=HOT_RATED_POWER_KW,
                era5_wd=era5_direction(era5, pd.DatetimeIndex(full_dataset.synthetic_df.index.unique()).sort_values()),
            ),
        )
        schedule = ToggleSchedule(period=TOGGLE_PERIOD, start=TOGGLE_START)

        nadir, half_width = _pair_sector(steering, margin_deg=wd_margin_deg)
        logger.info(
            "T01 -> T04 nadir %.1f deg; steering sector [%.1f, %.1f] (wd_filter=%s)",
            nadir,
            nadir - half_width,
            nadir + half_width,
            wd_filter,
        )

        analysis_dataset = full_dataset
        if wd_filter:
            keep = _in_sector_timestamps(full_dataset, nadir=nadir, half_width=half_width)
            analysis_dataset = _filter_dataset(full_dataset, keep)
            logger.info(
                "wd_filter kept %d of %d timestamps",
                len(keep),
                len(pd.unique(full_dataset.synthetic_df.index)),
            )

        # Structural ground-truth plots always show the full (unfiltered) dataset.
        _write_ground_truth_plots(full_dataset, out_dir=run_dir / "ground_truth")

        # Fast methods run per turbine; v0 runs once over BOTH participants (single co-run below).
        summaries = []
        truths: dict[str, float] = {}
        for wtg in PARTICIPANTS:
            out_dir = run_dir / wtg
            mi = MethodInput(
                scada_df=analysis_dataset.synthetic_df,
                test_wtg=wtg,
                upgrade_timing=schedule,
                turbine_col=HOT_COLUMNS.turbine,
            )
            truths[wtg] = _truth(analysis_dataset, test_wtg=wtg, schedule=schedule)
            summary, outputs = _run_methods(_fast_methods(out_dir, era5), mi=mi, truth=truths[wtg])
            _plot_conditional_uplift(
                analysis_dataset,
                outputs["power_model"],
                test_wtg=wtg,
                schedule=schedule,
                out_dir=out_dir / "power_model",
            )
            summaries.append(summary)

        if include_v0:
            summaries.append(
                _run_v0_pair(
                    context, dataset=analysis_dataset, schedule=schedule, truths=truths, out_dir=run_dir / "v0"
                )
            )

        combined = pd.concat(summaries, ignore_index=True)
        summary_path = run_dir / "comparison_summary.csv"
        combined.to_csv(summary_path, index=False)
        logger.info("Wrote %s\n%s", summary_path, combined.to_string(index=False))
        return combined
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # v0 is a slow full wind_up run per turbine; the bare module-run does the fast methods only.
    inspect_wake_steering_case(include_v0=False)
