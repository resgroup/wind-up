"""Reproduce the published Hill of Towie v0 analyses through the v1 interface with ``v0_binned``.

The analyses are those of the hill-of-towie-open-source-analysis repo: AeroUp on T13
(``scripts/uplift_analysis_2025/aero_up.py``) and Dynamic Yaw, collective control and wake steering
(``scripts/wfc_analysis_2026/uplift_cc.py`` and ``uplift_ws.py``). Their data is too large for CI, so
they are checked by hand, in two steps on the machine the reference ran on:

1. In that repo's environment, dump the inputs its run read, from its own caches, with
   ``dump_hot_v0_inputs`` (see that module).

2. In wind-up's environment, run the same analyses through the v1 interface::

       python -m benchmarking.baselines.reproduce_hot_v0 t13 INPUTS HOT_REPO OUT
       python -m benchmarking.baselines.reproduce_hot_v0 dy INPUTS HOT_REPO OUT [cc] [ws]

   ``HOT_REPO`` is the checkout whose YAML configured the reference. Each analysis writes its
   per test-reference results to ``OUT/results/<name>_per_reference.csv``, to compare with the
   reference's ``*_results_per_test_ref_*.csv``. A run that raises is skipped, as the reference
   skipped it, and a run whose results are already written is not run again. With ``--config-only`` each run's v0
   configuration is written to ``OUT/configs`` instead, to compare with the reference's
   ``config_*.json``.
"""

from __future__ import annotations

import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from benchmarking.baselines.dump_hot_v0_inputs import STEER_TABLE, STEERING_USED_LABEL
from benchmarking.baselines.v0_binned import V0BinnedMethod
from benchmarking.synthetic.sources.hill_of_towie import hill_of_towie_fields
from wind_up.analysis import AnalysisSpec
from wind_up.facts import NonTurbineMeasurement, Reanalysis, Upgrade, WindFarmFacts
from wind_up.labels import Label
from wind_up.schema import ColumnSchema

TIMEBASE = pd.Timedelta(minutes=10)
# The mean of the three blade pitch columns, the pitch the reference runs read.
PITCH_MEAN_COL = "mean_wtc_PitcPosA_B_C"
BLADE_PITCH_COLS = ("wtc_PitcPosA_mean", "wtc_PitcPosB_mean", "wtc_PitcPosC_mean")
SOURCE_FIELD_OF_ALIAS = {f.alias: f.field_name for f in hill_of_towie_fields}
COLUMNS = ColumnSchema(
    turbine="TurbineName",
    active_power="wtc_ActPower_mean",
    wind_speed="wtc_AcWindSp_mean",
    wind_speed_sd="wtc_AcWindSp_stddev",
    gen_rpm="wtc_GenRpm_mean",
    pitch=PITCH_MEAN_COL,
    nacelle_position="wtc_NacelPos_mean",
    nacelle_position_min="wtc_NacelPos_min",
    nacelle_position_max="wtc_NacelPos_max",
    reactive_power="wtc_ReactPwr_mean",
    ambient_temp="wtc_AmbieTmp_mean",
)
TURBINE_TYPE = {
    "turbine_type": "SWT-2.3-82",
    "normal_operation_pitch_range": (-10, 40),
    "normal_operation_genrpm_range": (500, 1600),
}
RATED_POWER_KW = 2300.0
ROTOR_DIAMETER_M = 82.0
ERA5_NODE = ("ERA5_57.50_-3.25", 57.5, -3.25)
LIDARS = ("ZX300_2428", "ZTM_5060")


def utc(value: object) -> pd.Timestamp:
    """Return ``value`` as a UTC timestamp."""
    return pd.Timestamp(value, tz="UTC")


def _records(inputs: Path) -> pd.DataFrame:
    """Return the dumped SCADA with the Hill of Towie source field names and the mean blade pitch.

    The dump has the v0 names the Hill of Towie repo gives its fields; ``ShutdownDuration`` keeps the
    name of the Hill of Towie file it comes from.
    """
    records = pd.read_parquet(inputs / "scada_df.parquet").drop(columns="PitchAngleMean")
    records = records.rename(columns=SOURCE_FIELD_OF_ALIAS)
    records[PITCH_MEAN_COL] = records[list(BLADE_PITCH_COLS)].mean(axis=1)
    return records


def _turbines(inputs: Path) -> pd.DataFrame:
    meta = pd.read_parquet(inputs / "metadata_df.parquet")
    return pd.DataFrame(
        {
            "name": meta["Name"],
            "latitude": meta["Latitude"],
            "longitude": meta["Longitude"],
            "rated_power_kw": RATED_POWER_KW,
            "rotor_diameter_m": ROTOR_DIAMETER_M,
        }
    )


def _reanalysis(inputs: Path) -> list[Reanalysis]:
    name, lat, lon = ERA5_NODE
    return [
        Reanalysis(
            name=name, latitude=lat, longitude=lon, records=pd.read_parquet(inputs / f"reanalysis_{name}.parquet")
        )
    ]


def _exclusions(rows: list) -> list[tuple]:
    return [(None if who == "ALL" else who, utc(a), utc(b)) for who, a, b in rows]


def _north_offsets(rows: list) -> list[tuple]:
    return [(who, utc(when), float(offset)) for who, when, offset in rows]


def _run(
    method: V0BinnedMethod, facts: WindFarmFacts, analyses: list[AnalysisSpec], *, out: Path, config_only: bool
) -> None:
    for analysis in analyses:
        if config_only:
            config = method._config(facts, analysis, use_lt_distribution="aep" in analysis.report)  # noqa: SLF001
            config.save_json(out / "configs" / f"{analysis.name}.json")
            continue
        if (out / "results" / f"{analysis.name}_per_turbine.csv").exists():
            print(analysis.name, "already done", flush=True)  # noqa: T201
            continue
        try:
            (output,) = method.estimate(facts, [analysis])
        except Exception:  # noqa: BLE001
            print(analysis.name, "skipping due to exception", flush=True)  # noqa: T201
            traceback.print_exc()
            continue
        (out / "results").mkdir(parents=True, exist_ok=True)
        if output.per_reference is not None:
            output.per_reference.to_csv(out / "results" / f"{analysis.name}_per_reference.csv", index=False)
        if output.per_turbine is not None:
            output.per_turbine.to_csv(out / "results" / f"{analysis.name}_per_turbine.csv", index=False)
        print(analysis.name, "AEP", output.aep_uplift, "campaign", output.campaign_uplift, flush=True)  # noqa: T201


def t13(inputs: Path, *, hot_repo: Path, out: Path, config_only: bool) -> None:
    """Run AeroUp on T13 against its four references, AEP uplift, as ``aero_up.py`` configures it."""
    cfg = hot_repo / "scripts/uplift_analysis_2025/wind_up_config"
    raw = yaml.safe_load((cfg / "HoT_AeroUp_T13.yaml").read_text().replace("!include", "#"))
    facts = WindFarmFacts(
        records=_records(inputs),
        columns=COLUMNS,
        timebase=TIMEBASE,
        turbines=_turbines(inputs),
        upgrades=[Upgrade(name="AeroUp", turbines=["T13"], timing=utc(raw["upgrade_first_dt_utc_start"]))],
        exclusions=_exclusions(raw["exclusion_periods_utc"]),
        north_offsets=_north_offsets(
            yaml.safe_load((cfg / "northing/optimized_northing_corrections.yaml").read_text())
        ),
        reanalysis=_reanalysis(inputs),
    )
    analysis = AnalysisSpec(
        name=raw["assessment_name"],
        upgrade="AeroUp",
        subjects=["T13"],
        references=raw["ref_wtgs"],
        period=(None, utc(raw["analysis_last_dt_utc_start"]) + TIMEBASE),
        report="aep",
    )
    options = {
        k: raw[k]
        for k in (
            "years_offset_for_pre_period",
            "years_for_lt_distribution",
            "use_test_wtg_lt_distribution",
            "years_for_detrend",
            "ws_bin_width",
            "reanalysis_method",
            "optimize_northing_corrections",
        )
    }
    method = V0BinnedMethod(
        options=options,
        turbine_type=TURBINE_TYPE,
        site_name="Hill of Towie",
        out_dir=out / "v0",
    )
    _run(method, facts, [analysis], out=out, config_only=config_only)


def dy(inputs: Path, *, hot_repo: Path, out: Path, which: set[str], config_only: bool) -> None:
    """Run Dynamic Yaw: collective control of the controlled turbines, and each steer against each reference."""
    cfg = hot_repo / "scripts/wfc_analysis_2026/wind_up_config"
    raw = yaml.safe_load((cfg / "HOT_dynamic_yaw.yaml").read_text().replace("!include", "#"))
    scada = _records(inputs)
    scada[STEERING_USED_LABEL] = pd.read_parquet(inputs / "steering_used.parquet")[STEERING_USED_LABEL].to_numpy()
    toggle = pd.read_parquet(inputs / "toggle_df.parquet")
    lidars = []
    for name in LIDARS:
        declared = yaml.safe_load((cfg / f"asset/masts_and_lidars/{name}.yaml").read_text())
        lidars.append(
            NonTurbineMeasurement(
                name=name,
                latitude=declared["latitude"],
                longitude=declared["longitude"],
                records=pd.read_parquet(inputs / f"lidar_{name}.parquet"),
                columns={
                    "wind_speed": declared["wind_speed_column"],
                    "wind_direction": declared["wind_direction_column"],
                },
            )
        )
    steers = pd.read_csv(inputs / STEER_TABLE)
    timing = {
        "timing": toggle,
        "settling_time": pd.Timedelta(seconds=raw["toggle"]["toggle_change_settling_filter_seconds"]),
        "upgrade_state_overrides": [(None, toggle.index.min(), utc(raw["upgrade_first_dt_utc_start"]), "neither")],
    }
    controlled = [t for t in raw["test_wtgs"] if t != "T07"]
    steered = sorted({*steers["upwind_wtg"], *steers["downwind_wtg"]})
    facts = WindFarmFacts(
        records=scada,
        columns=COLUMNS,
        timebase=TIMEBASE,
        turbines=_turbines(inputs),
        upgrades=[
            Upgrade(name="dynamic yaw", turbines=controlled, **timing),
            Upgrade(name="wake steering", turbines=steered, **timing),
        ],
        exclusions=_exclusions(yaml.safe_load((cfg / "exclusion_periods/HOT_DY.yaml").read_text())),
        labels=[Label(name=STEERING_USED_LABEL, column=STEERING_USED_LABEL)],
        north_offsets=_north_offsets(
            yaml.safe_load((cfg / "northing_corrections/HOT_DY_Design_northing.yaml").read_text())
        ),
        non_turbine_measurements=lidars,
        reanalysis=_reanalysis(inputs),
    )
    references = [*raw["ref_wtgs"], *raw["non_wtg_ref_names"]]
    end = utc(raw["analysis_last_dt_utc_start"]) + TIMEBASE
    options = {
        **{
            k: raw[k]
            for k in (
                "write_pp_df_parquets",
                "require_ref_wake_free",
                "years_for_lt_distribution",
                "years_for_detrend",
                "ws_bin_width",
                "reanalysis_method",
                "optimize_northing_corrections",
            )
        },
        "toggle": {
            k: raw["toggle"][k]
            for k in ("name", "detrend_data_selection", "pairing_filter_method", "pairing_filter_timedelta_seconds")
        },
    }
    common: dict[str, Any] = {
        "turbine_type": TURBINE_TYPE,
        "site_name": "Hill of Towie",
        "auto_choose_refs": False,
        "out_dir": out / "v0",
    }
    if "cc" in which:
        cc = AnalysisSpec(
            name="HOT_dynamic_yaw_CC_only",
            upgrade="dynamic yaw",
            subjects=controlled,
            references=references,
            exclude_labels=[STEERING_USED_LABEL],
            period=(None, end),
            report="aep",
        )
        # 0.7.1 dropped the steering records before preprocessing, so they leave the long-term distribution too.
        method = V0BinnedMethod(options=options, analysis_label_stage="preprocessing", **common)
        _run(method, facts, [cc], out=out, config_only=config_only)
    if "ws" in which:
        order = list(facts.turbine_names)
        for steer in steers.itertuples():
            subjects = [t for t in order if t in (steer.upwind_wtg, steer.downwind_wtg)]
            sector = [steer.first_wdir % 360, steer.last_wdir % 360]
            method = V0BinnedMethod(options={**options, "ref_wd_filter": sector}, **common)
            analyses = [
                AnalysisSpec(
                    name=f"{steer.upwind_wtg}_{steer.downwind_wtg}_{ref}",
                    upgrade="wake steering",
                    subjects=subjects,
                    references=[ref],
                    period=(None, end),
                    report="campaign",
                )
                for ref in references
            ]
            _run(method, facts, analyses, out=out, config_only=config_only)


def main(argv: list[str]) -> None:
    """Run the command ``argv`` names; see the module's documentation."""
    warnings.filterwarnings("ignore")
    flags = {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    command, rest = args[0], args[1:]
    config_only = "--config-only" in flags
    if command == "t13":
        t13(Path(rest[0]), hot_repo=Path(rest[1]), out=Path(rest[2]), config_only=config_only)
    elif command == "dy":
        which = set(rest[3:]) or {"cc", "ws"}
        dy(Path(rest[0]), hot_repo=Path(rest[1]), out=Path(rest[2]), which=which, config_only=config_only)
    else:
        msg = f"unknown command {command!r}; see the module's documentation"
        raise SystemExit(msg)


if __name__ == "__main__":
    main(sys.argv[1:])
