"""The shift probe: what the placebo's prepost offset owes to the pre-to-post weather shift.

Every arm runs the same prepost placebo -- a full year of baseline, then a six-month campaign, with
nothing injected, so the truth is 0 and every estimate is bias -- and slides the changeover across
the 2018 calendar. Only the weather the campaign meets changes: ERA5 mean wind speed over the
campaign runs from 0.77x the baseline's (a spring changeover, so a calm summer campaign) to 1.18x
(an autumn changeover, so a windy winter campaign), which brackets the shift of the real Hill of
Towie AeroUp window in both directions.

Each arm is paired with five measures of its own shift: the ERA5 mean and mean-cubed wind-speed
ratios of campaign to baseline, and, per test turbine, the CEM upgraded-row retention, the share of
campaign rows sitting in weather cells the baseline covers more thinly than they occupy, and the
implied per-bin shrinkage.

If the bias scales with the shift -- positive into a windier campaign, negative into a calmer one --
the offset is the counterfactual model shrinking toward its training mean under covariate shift. If
it is flat, that mechanism is out and the search moves to the fit itself.

The reference screen is off, and the per-reference uplift report with it, so every arm sees the same
reference pool and one arm differs from the next only in the window it runs over.

Run it::

    uv run python -m benchmarking.campaigns.shift_probe             # the sweep
    uv run python -m benchmarking.campaigns.shift_probe references  # the whole-farm reference arms

The reference arms re-read the two ends of the dose the way a campaign report reads its references:
one test turbine, every other turbine estimated against the rest. That is the same table a real
campaign's reference mean comes off, so the two numbers can be set side by side.

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``shift_probe``/``<timestamp>/``.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless: the report writes plots without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.baselines.hot_context import build_hot_v0_context
from benchmarking.campaigns.methods import carried_forward_methods
from benchmarking.campaigns.placebo import (
    PLACEBO_TURBINES,
    PLACEBO_UPGRADED,
    placebo_analysis_period,
    placebo_campaign,
)
from benchmarking.campaigns.runner import CampaignRunner, per_turbine_table
from benchmarking.diagnostics.context import era5_source_label
from benchmarking.diagnostics.style import apply_grid, save_fig
from benchmarking.harness.northing import era5_direction
from benchmarking.synthetic import HOT_LAT, HOT_LON
from benchmarking.synthetic.sources.hill_of_towie import load_hot_metadata, load_hot_scada

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarking.campaigns.declaration import SyntheticCampaign
    from benchmarking.campaigns.runner import CampaignResult

logger = logging.getLogger(__name__)

# One year of baseline and a six-month campaign, the same for every arm: a full-year baseline is
# season-complete whatever month it ends in, and a six-month campaign is short enough to meet a
# distinctly different season without falling under the screen's own campaign-length floor.
PROBE_BASELINE_MONTHS = 12
PROBE_CAMPAIGN_MONTHS = 6

# Changeover months chosen so the arms step evenly through the shift: 0.77, 0.85, 0.92, 1.02, 1.12
# and 1.18 times the baseline mean wind speed.
PROBE_CHANGEOVER_MONTHS = (3, 5, 6, 7, 8, 9)
PROBE_YEAR = 2018

ERA5_WIND_SPEED = "wind_speed_100m"


def probe_changeovers() -> tuple[pd.Timestamp, ...]:
    """Return the changeover of every arm, in calendar order."""
    return tuple(pd.Timestamp(year=PROBE_YEAR, month=m, day=1, tz="UTC") for m in PROBE_CHANGEOVER_MONTHS)


def arm_name(changeover: pd.Timestamp) -> str:
    """Return the label an arm's output folder and table rows carry."""
    return f"changeover_{changeover:%Y%m}"


def analysis_period(changeover: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(start, end)`` of the whole record an arm's methods see, end exclusive."""
    return placebo_analysis_period(
        "prepost",
        campaign_start=changeover,
        baseline_months=PROBE_BASELINE_MONTHS,
        campaign_months=PROBE_CAMPAIGN_MONTHS,
    )


def probe_span(changeovers: Sequence[pd.Timestamp] | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the record every arm together covers, so the SCADA is loaded once."""
    periods = [analysis_period(c) for c in (changeovers if changeovers is not None else probe_changeovers())]
    return min(p[0] for p in periods), max(p[1] for p in periods)


def _coords(turbines: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Hill of Towie coordinates for ``turbines``."""
    metadata = load_hot_metadata()
    return {
        str(row.Name): (float(row.Latitude), float(row.Longitude))
        for row in metadata.itertuples()
        if str(row.Name) in set(turbines)
    }


def probe_campaign(
    changeover: pd.Timestamp,
    *,
    turbines: Sequence[str] | None = None,
    upgraded: Sequence[str] | None = None,
    coords: dict[str, tuple[float, float]] | None = None,
) -> SyntheticCampaign:
    """Declare one arm: the placebo turbines over this arm's window, nothing injected."""
    participating = list(PLACEBO_TURBINES if turbines is None else turbines)
    return placebo_campaign(
        "prepost",
        upgraded=list(PLACEBO_UPGRADED if upgraded is None else upgraded),
        turbines=participating,
        coords=coords if coords is not None else _coords(participating),
        campaign_start=changeover,
        baseline_months=PROBE_BASELINE_MONTHS,
        campaign_months=PROBE_CAMPAIGN_MONTHS,
    )


def weather_shift(era5_df: pd.DataFrame, *, changeover: pd.Timestamp) -> dict[str, float]:
    """Return an arm's ERA5 wind shift: campaign-to-baseline ratios of mean and mean-cubed speed.

    :param era5_df: hourly ERA5 carrying :data:`ERA5_WIND_SPEED`
    :param changeover: the arm's changeover
    """
    start, end = analysis_period(changeover)
    ws = era5_df[ERA5_WIND_SPEED]
    pre = ws[(ws.index >= start) & (ws.index < changeover)]
    post = ws[(ws.index >= changeover) & (ws.index < end)]
    return {
        "pre_era5_ws": float(pre.mean()),
        "post_era5_ws": float(post.mean()),
        "ws_ratio": float(post.mean() / pre.mean()),
        "energy_ratio": float((post**3).mean() / (pre**3).mean()),
    }


def _latest(run_dir: Path, pattern: str) -> Path | None:
    """Return the newest file matching ``pattern`` anywhere under ``run_dir``, or ``None``."""
    matches = sorted(run_dir.rglob(pattern))
    return matches[-1] if matches else None


def cem_doses(turbine_dir: Path) -> dict[str, float]:
    """Return the shift measures a power-model run wrote under ``turbine_dir``.

    Keys are ``retained_fraction_upgraded``, ``thin_cell_share`` (the share of campaign rows in
    weather cells holding fewer baseline rows than campaign rows) and ``implied_shrinkage``. A
    measure whose CSV is absent comes back as ``NaN``.
    """
    doses: dict[str, float] = {
        "retained_fraction_upgraded": float("nan"),
        "thin_cell_share": float("nan"),
        "implied_shrinkage": float("nan"),
    }
    balance = _latest(turbine_dir, "*_cem_balance_*.csv")
    if balance is not None:
        doses["retained_fraction_upgraded"] = float(pd.read_csv(balance)["retained_fraction_upgraded"].iloc[0])
    cells = _latest(turbine_dir, "*_cem_cells_*.csv")
    if cells is not None:
        cell_df = pd.read_csv(cells)
        thin = cell_df[cell_df["n_baseline"] < cell_df["n_upgraded"]]
        doses["thin_cell_share"] = float(thin["n_upgraded"].sum() / cell_df["n_upgraded"].sum())
    overall = _latest(turbine_dir, "*_conditional_overall_*.csv")
    if overall is not None:
        doses["implied_shrinkage"] = float(pd.read_csv(overall)["implied_shrinkage"].iloc[0])
    return doses


def run_arm(
    *,
    changeover: pd.Timestamp,
    scada_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    out_dir: Path,
    turbines: Sequence[str] | None = None,
    include_power_model: bool = True,
) -> CampaignResult:
    """Run one arm end-to-end and return its result; every estimate has a truth of 0."""
    participating = list(PLACEBO_TURBINES if turbines is None else turbines)
    campaign = probe_campaign(changeover, turbines=participating)
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
            era5_label=era5_source_label(HOT_LAT, HOT_LON),
            reference_screen=False,
            report_reference_uplifts=False,
        ),
        era5_wd=era5_direction(era5_df, index),
        northing_out_dir=out_dir / "northing",
    )
    return runner.run()


def default_output_root() -> Path:
    """Return the directory this driver writes under (``WIND_UP_BENCHMARKING_OUTPUT_DIR`` overrides)."""
    root = Path(os.getenv("WIND_UP_BENCHMARKING_OUTPUT_DIR", Path.home() / "temp" / "wind-up-benchmarking"))
    return root / "shift_probe"


# The reference arms read the whole farm the way a campaign report does: one test turbine, every
# other turbine a candidate reference estimated against the rest. That is the table a real campaign
# reads its references off, so its mean is comparable with a real campaign's.
REFERENCE_ARM_TEST_WTG = "T13"


def run_reference_arm(
    *,
    changeover: pd.Timestamp,
    scada_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    out_dir: Path,
    test_wtg: str = REFERENCE_ARM_TEST_WTG,
    turbines: Sequence[str] | None = None,
    reference_screen: bool = False,
) -> pd.DataFrame:
    """Run one arm as a whole-farm campaign and return its per-reference readings, truth 0.

    :param changeover: the arm's changeover
    :param scada_df: SCADA covering at least the arm's window
    :param era5_df: hourly ERA5 for the site
    :param out_dir: where the run writes
    :param test_wtg: the one turbine declared upgraded; every other turbine is a candidate reference
    :param turbines: every participating turbine; the placebo farm when ``None``
    :param reference_screen: run the screen as a campaign ships it, and mark what it ruled out
    """
    participating = list(PLACEBO_TURBINES if turbines is None else turbines)
    campaign = probe_campaign(changeover, turbines=participating, upgraded=[test_wtg])
    dataset = campaign.generate(scada_df)
    spec = campaign.spec()
    index = pd.DatetimeIndex(dataset.synthetic_df.index.unique()).sort_values()
    runner = CampaignRunner(
        spec,
        dataset,
        build_methods=lambda wtg: carried_forward_methods(
            spec,
            out_dir=out_dir / wtg,
            era5_hourly_df=era5_df,
            era5_label=era5_source_label(HOT_LAT, HOT_LON),
            reference_screen=reference_screen,
            report_reference_uplifts=True,
        ),
        era5_wd=era5_direction(era5_df, index),
        northing_out_dir=out_dir / "northing",
    )
    result = runner.run()
    stability = result.report.reference_stability
    readings = stability[stability["method"] == "power_model"][["turbine", "uplift", "screened"]].copy()
    readings["reading_pp"] = readings["uplift"] * 100
    readings["test_wtg"] = test_wtg
    return readings.drop(columns="uplift").reset_index(drop=True)


def run_reference_probe(
    *,
    changeovers: Sequence[pd.Timestamp] | None = None,
    test_wtg: str = REFERENCE_ARM_TEST_WTG,
    turbines: Sequence[str] | None = None,
    reference_screen: bool = False,
    out_root: str | Path | None = None,
) -> pd.DataFrame:
    """Run the reference arms and return one row per (arm, reference), each with a truth of 0.

    Defaults to the calmest and the windiest arm of the sweep, the two ends of the dose.
    """
    arms = list(changeovers if changeovers is not None else (probe_changeovers()[0], probe_changeovers()[-1]))
    participating = list(PLACEBO_TURBINES if turbines is None else turbines)
    root = Path(out_root) if out_root is not None else default_output_root()
    suffix = "_screened" if reference_screen else ""
    run_dir = root / f"references{suffix}_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    era5_df = build_hot_v0_context(wtg_names=participating).reanalysis_datasets[0].data
    span = probe_span(arms)
    logger.info("loading Hill of Towie SCADA %s..%s for %s", *span, participating)
    scada_df, _ = load_hot_scada(
        start_dt=span[0],
        end_dt_excl=span[1],
        wtg_numbers=[int(w[1:]) for w in participating],
        wtg_names=participating,
    )

    frames: list[pd.DataFrame] = []
    for changeover in arms:
        name = arm_name(changeover)
        logger.info("running reference arm %s", name)
        readings = run_reference_arm(
            changeover=changeover,
            scada_df=scada_df,
            era5_df=era5_df,
            out_dir=run_dir / name,
            test_wtg=test_wtg,
            turbines=participating,
            reference_screen=reference_screen,
        )
        readings.insert(0, "arm", name)
        readings["changeover"] = changeover
        for key, value in weather_shift(era5_df, changeover=changeover).items():
            readings[key] = value
        frames.append(readings)
        pd.concat(frames).to_csv(run_dir / "reference_readings.csv", index=False)
        kept = readings[~readings["screened"]]
        logger.info(
            "%s: %d references read mean %+.3f pp, median %+.3f pp, sd %.3f pp; %d kept read mean %+.3f pp",
            name,
            len(readings),
            readings["reading_pp"].mean(),
            readings["reading_pp"].median(),
            readings["reading_pp"].std(),
            len(kept),
            kept["reading_pp"].mean(),
        )
    logger.info("wrote the reference arms to %s", run_dir)
    return pd.concat(frames).reset_index(drop=True)


def run_probe(
    *,
    changeovers: Sequence[pd.Timestamp] | None = None,
    turbines: Sequence[str] | None = None,
    include_power_model: bool = True,
    out_root: str | Path | None = None,
) -> pd.DataFrame:
    """Run every arm and return one tidy row per (arm, method, test turbine).

    Each row carries the estimate, its truth of 0, and the arm's shift measures.
    """
    arms = list(changeovers if changeovers is not None else probe_changeovers())
    participating = list(PLACEBO_TURBINES if turbines is None else turbines)
    root = Path(out_root) if out_root is not None else default_output_root()
    run_dir = root / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    era5_df = build_hot_v0_context(wtg_names=participating).reanalysis_datasets[0].data
    span = probe_span(arms)
    logger.info("loading Hill of Towie SCADA %s..%s for %s", *span, participating)
    scada_df, _ = load_hot_scada(
        start_dt=span[0],
        end_dt_excl=span[1],
        wtg_numbers=[int(w[1:]) for w in participating],
        wtg_names=participating,
    )

    rows: list[dict[str, object]] = []
    for changeover in arms:
        name = arm_name(changeover)
        logger.info("running %s", name)
        arm_dir = run_dir / name
        result = run_arm(
            changeover=changeover,
            scada_df=scada_df,
            era5_df=era5_df,
            out_dir=arm_dir,
            turbines=participating,
            include_power_model=include_power_model,
        )
        shift = weather_shift(era5_df, changeover=changeover)
        for row in per_turbine_table(result).itertuples():
            doses = cem_doses(arm_dir / str(row.test_wtg)) if row.method == "power_model" else {}
            rows.append(
                {
                    "arm": name,
                    "changeover": changeover,
                    "method": row.method,
                    "test_wtg": row.test_wtg,
                    "estimate_pp": float(row.estimate) * 100,
                    "truth": float(row.truth),
                    **shift,
                    **doses,
                }
            )
        pd.DataFrame(rows).to_csv(run_dir / "estimates.csv", index=False)

    estimates = pd.DataFrame(rows)
    arm_table = arm_summary(estimates)
    arm_table.to_csv(run_dir / "arms.csv", index=False)
    dose_table = dose_response(estimates)
    dose_table.to_csv(run_dir / "dose_response.csv", index=False)
    plot_dose_response(estimates, out_dir=run_dir)
    logger.info("wrote the probe results to %s", run_dir)
    return estimates


def arm_summary(estimates: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (method, arm): the arm's bias across its test turbines, in pp."""
    if estimates.empty:
        return estimates
    grouped = estimates.groupby(["method", "arm"], as_index=False).agg(
        changeover=("changeover", "first"),
        ws_ratio=("ws_ratio", "first"),
        energy_ratio=("energy_ratio", "first"),
        retained_fraction_upgraded=("retained_fraction_upgraded", "mean"),
        thin_cell_share=("thin_cell_share", "mean"),
        implied_shrinkage=("implied_shrinkage", "mean"),
        mean_pp=("estimate_pp", "mean"),
        median_pp=("estimate_pp", "median"),
        sd_pp=("estimate_pp", "std"),
        n=("estimate_pp", "size"),
    )
    return grouped.sort_values(["method", "changeover"]).reset_index(drop=True)


# Dose scalars regressed against the bias. Each is centred on its no-shift value, so a slope is
# "pp of bias per unit of shift" and an intercept is the bias a shift-free campaign would carry.
_DOSE_ORIGINS = {
    "ws_ratio": 1.0,
    "energy_ratio": 1.0,
    "retained_fraction_upgraded": 1.0,
    "thin_cell_share": 0.0,
    "implied_shrinkage": 1.0,
}


def dose_response(estimates: pd.DataFrame) -> pd.DataFrame:
    """Return the straight-line fit of arm bias against each dose, one row per (method, dose).

    ``slope_pp`` is percentage points of bias per unit of dose, ``intercept_pp`` the bias at the
    dose's no-shift value, and ``r`` the correlation across the arms. A dose no arm measured, or a
    method with fewer than three arms, is left out.
    """
    arms = arm_summary(estimates)
    if arms.empty:
        return arms
    rows: list[dict[str, object]] = []
    for method, group in arms.groupby("method"):
        for dose, origin in _DOSE_ORIGINS.items():
            usable = group[["mean_pp", dose]].dropna()
            if len(usable) < 3 or usable[dose].nunique() < 2:  # noqa: PLR2004 - two points fit any line
                continue
            x = usable[dose].to_numpy() - origin
            y = usable["mean_pp"].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            # a bias identical in every arm has no correlation to report, only a flat slope
            r = float(np.corrcoef(x, y)[0, 1]) if y.std() > 0 else float("nan")
            rows.append(
                {
                    "method": method,
                    "dose": dose,
                    "n_arms": len(usable),
                    "slope_pp": float(slope),
                    "intercept_pp": float(intercept),
                    "r": r,
                    "dose_range": float(usable[dose].max() - usable[dose].min()),
                }
            )
    return pd.DataFrame(rows)


def plot_dose_response(estimates: pd.DataFrame, *, out_dir: Path, dose: str = "ws_ratio") -> Path:
    """Plot each arm's bias against ``dose``, one panel per method, and return the file written.

    Every test turbine's estimate is drawn behind its arm's mean, so the spread within an arm can be
    read against the movement between arms; the straight line is :func:`dose_response`'s fit.
    """
    arms = arm_summary(estimates)
    fits = dose_response(estimates).set_index(["method", "dose"])
    methods = sorted(arms["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 4.5), squeeze=False)
    for ax, method in zip(axes[0], methods, strict=True):
        rows = estimates[estimates["method"] == method]
        ax.scatter(rows[dose], rows["estimate_pp"], s=14, color="C0", alpha=0.4, label="test turbine")
        means = arms[arms["method"] == method]
        ax.scatter(means[dose], means["mean_pp"], s=60, color="C1", marker="D", label="arm mean")
        if (method, dose) in fits.index:
            fit = fits.loc[(method, dose)]
            x = np.linspace(float(means[dose].min()), float(means[dose].max()), 2)
            ax.plot(x, fit["slope_pp"] * (x - _DOSE_ORIGINS[dose]) + fit["intercept_pp"], color="C3")
            ax.set_title(f"{method}: {fit['slope_pp']:+.2f} pp per unit, r={fit['r']:.2f}")
        else:
            ax.set_title(str(method))
        ax.axhline(0.0, color="k", linewidth=1)
        ax.set_xlabel(f"{dose} (campaign / baseline)")
        ax.set_ylabel("estimate [pp], truth 0")
        ax.legend(fontsize="small")
        apply_grid(ax)
    path = out_dir / f"dose_response_{dose}.png"
    save_fig(fig, path)
    return path


def reference_summary(readings: pd.DataFrame) -> pd.DataFrame:
    """Return one row per reference arm: how the whole farm's references read, truth 0."""
    if readings.empty:
        return readings
    summary = readings.groupby(["arm", "ws_ratio"], as_index=False).agg(
        mean_pp=("reading_pp", "mean"),
        median_pp=("reading_pp", "median"),
        sd_pp=("reading_pp", "std"),
        n=("reading_pp", "size"),
    )
    if "screened" not in readings.columns:
        return summary
    kept = (
        readings[~readings["screened"]]
        .groupby(["arm", "ws_ratio"], as_index=False)
        .agg(
            kept_mean_pp=("reading_pp", "mean"),
            kept_median_pp=("reading_pp", "median"),
            n_kept=("reading_pp", "size"),
        )
    )
    return summary.merge(kept, on=["arm", "ws_ratio"], how="left")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="sweep",
        choices=("sweep", "references"),
        help="sweep: every arm's test turbines; references: the whole farm's references at the dose ends",
    )
    parser.add_argument(
        "--screen", action="store_true", help="references mode: run the reference screen as a campaign ships it"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.mode == "sweep":
        probe_estimates = run_probe()
        print(arm_summary(probe_estimates).to_string(index=False))  # noqa: T201 - a driver's point is its summary
        print()  # noqa: T201
        print(dose_response(probe_estimates).to_string(index=False))  # noqa: T201
    else:
        probe_readings = run_reference_probe(reference_screen=args.screen)
        print(reference_summary(probe_readings).to_string(index=False))  # noqa: T201
