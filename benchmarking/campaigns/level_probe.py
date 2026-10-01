"""The level probe: where a power-model placebo reading's level comes from.

A placebo reading (a turbine that did not change, truth 0) carries a level error of +0.1 to +1 pp
on contiguous prepost periods. This probe runs two campaigns with the power model's per-row dump
on and decomposes each reading's level by month, direction, wind band, which references were
running, how many upwind turbines were not waking, and how well the baseline covers each row
(propensity). It then compares the level with what the augmented inverse-propensity term predicts
from covariate shift and model error alone. See :mod:`benchmarking.campaigns.level_analysis`.

Cases:

* ``hot`` -- the Hill of Towie whole-farm reference arm at changeover 2018-09-01 (12 months of
  baseline into a 6-month campaign, T13 the declared test turbine, every other turbine a candidate
  reference, screen off). Every reading has a truth of 0.
* ``pen`` -- the prepost matrix cell ``pen_s00_m+0_K4_L12`` in its ``main``, ``excl_booleans`` and
  ``excl_nan`` arms. T14 is analysed in each, plus the difference between the two exclusion arms,
  which drop identical rows and so must agree.

Run it::

    uv run python -m benchmarking.campaigns.level_probe hot [--turbines T17 T13 ...] [--all] [--smoke]
    uv run python -m benchmarking.campaigns.level_probe pen
    uv run python -m benchmarking.campaigns.level_probe analyse <run_dir> [--turbines ...] [--all]

Outputs land under ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``level_probe``/``<case>_<timestamp>/``:
``dump/<arm>/<turbine>/`` (the rows), ``analysis/<arm>/<turbine>/`` (the tables and figures),
``summary.csv``/``summary.png`` and ``probe.log``. ``analyse`` re-reads an existing dump, so the
campaign fits run once.
"""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl

mpl.use("Agg")  # headless: the probe writes plots without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarking.baselines.power_model import TUNED_MODEL_PARAMS
from benchmarking.campaigns.composed import output_root
from benchmarking.campaigns.level_analysis import (
    RowDump,
    aipw_level,
    baseline_predictions,
    channel_difference_by,
    label_direction_sector,
    label_held_back,
    label_month,
    label_operating_pattern,
    label_propensity_decile,
    label_upwind_offline,
    label_wind_band,
    level_by,
    propensity,
    propensity_summary,
    top_group,
)
from benchmarking.diagnostics.style import apply_grid, save_fig

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

HOT_CHANGEOVER = pd.Timestamp("2018-09-01", tz="UTC")
HOT_TEST_WTG = "T13"
# The large contiguous-arm movers in CF24, then three that barely moved.
HOT_DEFAULT_TURBINES = ("T17", "T13", "T12", "T07", "T06", "T16")
# A smoke run: four turbines, two months into one, the test turbine alone analysed.
HOT_SMOKE_TURBINES = ("T12", "T13", "T14", "T15")
HOT_SMOKE_MONTHS = (2, 1)
PEN_TEST_WTG = "T14"
PEN_ARMS = ("main", "excl_booleans", "excl_nan")
CHANNEL_ARMS = ("excl_booleans", "excl_nan")
PROBE_JSON = "probe.json"
PROBE_LOG = "probe.log"
SUMMARY_COLUMNS = (
    "case",
    "arm",
    "turbine",
    "is_test",
    "observed_level_pp",
    "predicted_level_pp",
    "in_sample_control_pp",
    "share_rows_m_gt_0.95",
    "share_level_m_gt_0.95",
    "ess_fraction",
    "trimmed_rows",
    "top_month",
    "top_direction_sector",
    "top_operating_pattern",
)


def probe_output_root() -> Path:
    """Return where probe runs are written: ``WIND_UP_BENCHMARKING_OUTPUT_DIR``/``level_probe``."""
    return output_root() / "level_probe"


def analysis_model_params() -> dict[str, Any]:
    """Return the LightGBM parameters the probe's own fits use: the headline's."""
    return dict(TUNED_MODEL_PARAMS)


# --- one turbine -----------------------------------------------------------------------------------


def grouping_labels(dump: RowDump, *, rotor_diameter_m: float) -> dict[str, pd.Series | None]:
    """Return every grouping but propensity's, ``None`` where its inputs are absent."""
    return {
        "month": label_month(dump),
        "direction_sector": label_direction_sector(dump),
        "wind_band": label_wind_band(dump),
        "operating_pattern": label_operating_pattern(dump),
        "upwind_offline": label_upwind_offline(dump, rotor_diameter_m=rotor_diameter_m),
    }


def analyse_turbine(
    dump: RowDump, *, out_dir: Path, rotor_diameter_m: float, model_params: Mapping[str, Any]
) -> dict[str, Any]:
    """Decompose one reading's level, fit its propensity and AIPW term, and return its summary row."""
    out_dir.mkdir(parents=True, exist_ok=True)
    turbine = str(dump.meta["test_wtg"])
    m_hat = propensity(dump, model_params=model_params)
    labels = grouping_labels(dump, rotor_diameter_m=rotor_diameter_m)
    labels["propensity_decile"] = label_propensity_decile(dump, m_hat)
    tops: dict[str, float | str] = {}
    for name, grouping in labels.items():
        if grouping is None:
            tops[name] = np.nan
            continue
        table = level_by(dump, grouping)
        table.to_csv(out_dir / f"level_by_{name}.csv", index=False)
        plot_level(table, turbine=turbine, uplift=float(dump.meta["uplift"]), grouping=name, path=out_dir)
        tops[name] = top_group(table)

    selected = dump.baseline | dump.upgraded
    pd.DataFrame(
        {
            "timestamp": dump.rows.loc[selected, "timestamp"],
            "upgraded": dump.upgraded[selected],
            "m_hat": m_hat[selected],
        }
    ).to_csv(out_dir / "propensity.csv", index=False)
    plot_propensity(dump, m_hat, path=out_dir / "propensity.png")

    g_oof, g_in = baseline_predictions(dump, model_params=model_params)
    aipw = aipw_level(dump, m_hat, g_oof=g_oof, g_in_sample=g_in)
    pd.DataFrame([aipw]).to_csv(out_dir / "aipw.csv", index=False)
    summary = propensity_summary(dump, m_hat)
    logger.info(
        "%s: observed %+.3f pp, AIPW-predicted %+.3f pp (in-sample control %+.3f), %.1f%% of rows at m>0.95 "
        "carry %.0f%% of the level, ESS %.2f",
        turbine,
        aipw["observed_level_pp"],
        aipw["predicted_level_pp"],
        aipw["in_sample_control_pp"],
        100 * summary["share_rows_m_gt_0.95"],
        100 * summary["share_level_m_gt_0.95"],
        aipw["ess_fraction"],
    )
    return {
        "turbine": turbine,
        **aipw,
        **summary,
        "top_month": tops["month"],
        "top_direction_sector": tops["direction_sector"],
        "top_operating_pattern": tops["operating_pattern"],
    }


def analyse_channels(
    dump_a: RowDump, dump_b: RowDump, *, out_dir: Path, rotor_diameter_m: float
) -> dict[str, pd.DataFrame]:
    """Split the difference between two exclusion arms' readings of one turbine; write and return the tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        **grouping_labels(dump_a, rotor_diameter_m=rotor_diameter_m),
        "held_back": label_held_back(dump_a, dump_b),
    }
    tables = {}
    for name, grouping in labels.items():
        if grouping is None:
            continue
        table = channel_difference_by(dump_a, dump_b, grouping)
        table.to_csv(out_dir / f"channel_difference_by_{name}.csv", index=False)
        tables[name] = table
    held = tables["held_back"].set_index("group")
    logger.info(
        "%s channels: %+.3f pp between the arms, %+.3f pp of it on held-back rows",
        dump_a.meta["test_wtg"],
        float(tables["held_back"]["difference_pp"].sum()),
        float(held["difference_pp"].get("held back", 0.0)),
    )
    return tables


# --- a run -----------------------------------------------------------------------------------------


def dumped_turbines(run_dir: Path, arm: str) -> list[str]:
    """Return the turbines dumped under ``arm``, sorted."""
    return sorted(p.name for p in (run_dir / "dump" / arm).iterdir() if (p / "rows.json").is_file())


def analyse_run(
    run_dir: Path,
    *,
    turbines: Sequence[str] | None = None,
    model_params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Analyse the dumps of a probe run and write the tables, figures and ``summary.csv``.

    :param run_dir: a run's directory, holding ``probe.json`` and ``dump/``
    :param turbines: the turbines to analyse in each arm; every dumped turbine when ``None``
    :param model_params: LightGBM parameters for the probe's own fits; the headline's when ``None``
    """
    meta = json.loads((run_dir / PROBE_JSON).read_text(encoding="utf-8"))
    params = dict(model_params) if model_params is not None else analysis_model_params()
    rows = []
    for arm in meta["arms"]:
        wanted = dumped_turbines(run_dir, arm) if turbines is None else list(turbines)
        for turbine in wanted:
            dump = RowDump.load(run_dir / "dump" / arm / turbine)
            logger.info("analysing %s %s", arm, turbine)
            row = analyse_turbine(
                dump,
                out_dir=run_dir / "analysis" / arm / turbine,
                rotor_diameter_m=meta["rotor_diameter_m"],
                model_params=params,
            )
            rows.append({"case": meta["case"], "arm": arm, "is_test": turbine == meta["test_wtg"], **row})
    summary = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    summary.to_csv(run_dir / "summary.csv", index=False)
    plot_summary(summary, path=run_dir / "summary.png")
    if set(CHANNEL_ARMS) <= set(meta["arms"]):
        test = meta["test_wtg"]
        analyse_channels(
            *(RowDump.load(run_dir / "dump" / arm / test) for arm in CHANNEL_ARMS),
            out_dir=run_dir / "analysis" / f"{test}_channels",
            rotor_diameter_m=meta["rotor_diameter_m"],
        )
    logger.info("wrote the level probe analysis to %s", run_dir)
    return summary


def write_probe_meta(run_dir: Path, **meta: object) -> None:
    """Record what the run is, with the commit it ran from, in ``probe.json``."""
    from benchmarking.baselines.prepost_matrix.study import git_state  # noqa: PLC0415

    commit, dirty = git_state()
    payload = {**meta, "commit": commit, "dirty": dirty, "started": f"{pd.Timestamp.now():%Y-%m-%dT%H:%M:%S}"}
    (run_dir / PROBE_JSON).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("level probe %s", payload)


def _new_run_dir(case: str) -> Path:
    run_dir = probe_output_root() / f"{case}_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _log_to(run_dir / PROBE_LOG)
    return run_dir


def _log_to(path: Path) -> None:
    handler = logging.FileHandler(path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)


# --- the cases -------------------------------------------------------------------------------------


def run_hot(*, turbines: Sequence[str] | None = None, analyse_all: bool = False, smoke: bool = False) -> Path:
    """Run case A, the Hill of Towie whole-farm reference arm, then analyse it; return the run directory."""
    from benchmarking.baselines.hot_context import build_hot_v0_context  # noqa: PLC0415
    from benchmarking.campaigns.placebo import PLACEBO_TURBINES  # noqa: PLC0415
    from benchmarking.campaigns.shift_probe import analysis_period, arm_name, run_reference_arm  # noqa: PLC0415
    from benchmarking.synthetic.sources.hill_of_towie import HOT_ROTOR_DIAMETER_M, load_hot_scada  # noqa: PLC0415

    participating = list(HOT_SMOKE_TURBINES if smoke else PLACEBO_TURBINES)
    baseline_months, campaign_months = HOT_SMOKE_MONTHS if smoke else (12, 6)
    run_dir = _new_run_dir("hot_smoke" if smoke else "hot")
    arm = arm_name(HOT_CHANGEOVER)
    write_probe_meta(
        run_dir,
        case="hot",
        arms=[arm],
        test_wtg=HOT_TEST_WTG,
        rotor_diameter_m=HOT_ROTOR_DIAMETER_M,
        changeover=HOT_CHANGEOVER,
        baseline_months=baseline_months,
        campaign_months=campaign_months,
        turbines=participating,
    )
    era5_df = build_hot_v0_context(wtg_names=participating).reanalysis_datasets[0].data
    start, end = analysis_period(HOT_CHANGEOVER, baseline_months=baseline_months, campaign_months=campaign_months)
    scada_df, _ = load_hot_scada(
        start_dt=start, end_dt_excl=end, wtg_numbers=[int(w[1:]) for w in participating], wtg_names=participating
    )
    readings = run_reference_arm(
        changeover=HOT_CHANGEOVER,
        scada_df=scada_df,
        era5_df=era5_df,
        out_dir=run_dir / "campaign",
        test_wtg=HOT_TEST_WTG,
        turbines=participating,
        row_dump_dir=run_dir / "dump" / arm,
        baseline_months=baseline_months,
        campaign_months=campaign_months,
    )
    readings.to_csv(run_dir / "reference_readings.csv", index=False)
    analysed = None if analyse_all else ([HOT_TEST_WTG] if smoke else list(turbines or HOT_DEFAULT_TURBINES))
    analyse_run(run_dir, turbines=analysed)
    return run_dir


def pen_cells() -> list[Any]:
    """Return case B's cells: ``pen_s00_m+0_K4_L12`` in each of its three arms."""
    from benchmarking.baselines.prepost_matrix.cells import Cell  # noqa: PLC0415

    return [Cell(arm=arm, site="pen", seed_index=0, multiplier=0, k=4, post_months=12) for arm in PEN_ARMS]  # type: ignore[arg-type]


def run_pen(*, execute: Callable[..., dict[str, Any]] | None = None) -> Path:
    """Run case B, the Penmanshiel oracle cell in its three arms, then analyse T14; return the run directory."""
    from benchmarking.baselines.prepost_matrix.cells import SIZES  # noqa: PLC0415
    from benchmarking.baselines.prepost_matrix.execute import execute_cell, prefetch  # noqa: PLC0415
    from benchmarking.baselines.prepost_matrix.study import detail_dir, git_state, open_study, run_cell  # noqa: PLC0415
    from benchmarking.campaigns.rollout import penmanshiel_site  # noqa: PLC0415

    run_dir = _new_run_dir("pen")
    cells = pen_cells()
    write_probe_meta(
        run_dir,
        case="pen",
        arms=list(PEN_ARMS),
        test_wtg=PEN_TEST_WTG,
        rotor_diameter_m=penmanshiel_site().rotor_diameter_m,
        cells=[c.cell_id for c in cells],
    )
    # The small size's matrix, so the draw (seed, exclusions) is the one CF26 reports.
    settings = SIZES["small"].settings
    study = run_dir / "study"
    commit, dirty = git_state()
    open_study(study, settings, size="small", commit=commit, dirty=dirty)
    prefetch(detail_dir(study), cells)
    run = execute if execute is not None else execute_cell
    for cell in cells:
        logger.info("running %s", cell.cell_id)
        record = run_cell(
            study,
            cell,
            settings=settings,
            execute=partial(run, method_overrides={"row_dump_dir": run_dir / "dump" / cell.arm}),
        )
        if record["status"] != "ok":
            msg = f"{cell.cell_id} failed:\n{record.get('traceback', '')}"
            raise RuntimeError(msg)
        estimate = {t["turbine"]: t["estimate"] for t in record["turbines"]}.get(PEN_TEST_WTG)
        logger.info("%s: %s reads %s", cell.cell_id, PEN_TEST_WTG, estimate)
    analyse_run(run_dir, turbines=[PEN_TEST_WTG])
    return run_dir


# --- figures ---------------------------------------------------------------------------------------


def plot_level(table: pd.DataFrame, *, turbine: str, uplift: float, grouping: str, path: Path) -> Path:
    """Bar the groups' contributions, with their row share as a line on a second axis."""
    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(table) + 2), 4))
    x = np.arange(len(table))
    ax.bar(x, table["contribution_pp"], color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, table["group"], rotation=90 if len(table) > 8 else 0)  # noqa: PLR2004
    ax.set_ylabel("contribution to the level (pp)")
    ax.set_xlabel(grouping)
    apply_grid(ax)
    share = ax.twinx()
    share.plot(x, table["row_share"], color="tab:orange", marker="o")
    share.set_ylabel("share of upgraded rows", color="tab:orange")
    share.set_ylim(bottom=0)
    ax.set_title(f"{turbine}: headline {100 * uplift:+.3f} pp, by {grouping}")
    out = path / f"level_by_{grouping}.png"
    save_fig(fig, out)
    return out


def plot_propensity(dump: RowDump, m_hat: pd.Series, *, path: Path) -> Path:
    """Histogram the out-of-fold propensity of the baseline and the upgraded rows."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(m_hat[dump.baseline], bins=bins, alpha=0.6, label="baseline", density=True)
    ax.hist(m_hat[dump.upgraded], bins=bins, alpha=0.6, label="upgraded", density=True)
    ax.set_xlabel("out-of-fold P(upgraded | X)")
    ax.set_ylabel("density")
    ax.set_title(f"{dump.meta['test_wtg']}: propensity")
    ax.legend()
    apply_grid(ax)
    save_fig(fig, path)
    return path


def plot_summary(summary: pd.DataFrame, *, path: Path) -> Path:
    """Scatter the AIPW-predicted level against the observed one, one point per reading."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(summary["observed_level_pp"], summary["predicted_level_pp"])
    one_arm = bool((summary["arm"] == summary["arm"].iloc[0]).all()) if len(summary) else True
    for row in summary.itertuples():
        label = row.turbine if one_arm else f"{row.turbine} {row.arm}"
        ax.annotate(label, (row.observed_level_pp, row.predicted_level_pp), fontsize="small")
    values = summary[["observed_level_pp", "predicted_level_pp"]].to_numpy(dtype=float)
    lo, hi = (float(np.nanmin(values)), float(np.nanmax(values))) if np.isfinite(values).any() else (-1.0, 1.0)
    pad = 0.1 * max(hi - lo, 0.1)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=0.8, label="identity")
    ax.set_xlabel("observed level (pp)")
    ax.set_ylabel("AIPW-predicted level (pp)")
    ax.legend()
    apply_grid(ax)
    save_fig(fig, path)
    return path


# --- command line ----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Run a case, or re-run the analysis of an existing run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    hot = sub.add_parser("hot", help="the Hill of Towie whole-farm reference arm")
    hot.add_argument("--turbines", nargs="+", default=None)
    hot.add_argument("--all", action="store_true", help="analyse every reading")
    hot.add_argument("--smoke", action="store_true", help="four turbines, three months, T13 alone")
    sub.add_parser("pen", help="the Penmanshiel oracle cell in its three arms")
    analyse = sub.add_parser("analyse", help="re-run the analysis of an existing run")
    analyse.add_argument("run_dir", type=Path)
    analyse.add_argument("--turbines", nargs="+", default=None)
    analyse.add_argument("--all", action="store_true", help="analyse every dumped reading")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "hot":
        run_hot(turbines=args.turbines, analyse_all=args.all, smoke=args.smoke)
    elif args.command == "pen":
        run_pen()
    else:
        _log_to(args.run_dir / PROBE_LOG)
        meta = json.loads((args.run_dir / PROBE_JSON).read_text(encoding="utf-8"))
        default = [meta["test_wtg"]] if meta["case"] == "pen" else list(HOT_DEFAULT_TURBINES)
        analyse_run(args.run_dir, turbines=None if args.all else (args.turbines or default))


if __name__ == "__main__":
    main()
