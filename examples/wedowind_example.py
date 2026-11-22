# Dataset citation: Ding, Y. (2021). Turbine Upgrade Dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5516556
# description of challenge https://relight.cloud/doc/turbine-upgrade-dataset-9zw1vl/turbineperformance
# description of challenge text copied below:
#
# Turbine Upgrade Dataset
# This turbine upgrade dataset includes three sets, one corresponding to an actual vortex generator installation and
# two corresponding to an artificial pitch angle adjustment. Two pairs of wind turbines from the same inland
# wind farm, as used in Chapter 5 of the Data Science for Wind Energy book, are chosen to provide the data, each pair
# consisting of two wind turbines, together with a nearby met mast. The turbine that undergoes an upgrade in a pair
# is referred to as the experimental turbine, the reference turbine, or the test turbine, whereas the one that does
# not have the upgrade is referred to as the control turbine. In both pairs, the test turbine and the control turbine
# are practically identical and were put into service at the same time. This wind farm is on a reasonably flat terrain.
#
# The power output, y, is measured on individual turbines, whereas the  environmental variables in x (i.e., the weather
# covariates) are measured  by sensors at the nearby mast. For this dataset, there are five variables in x and they
# are the same as those in the Inland Wind Farm Dataset1. For the vortex generator installation pair, there are 14
# months' worth of data in the period before the upgrade and around eight weeks of data after the upgrade. For the
# pitch angle adjustment pair, there are about eight months of data before the upgrade and eight and a half weeks
# after the upgrade.
#
# Note that the pitch angle adjustment is not physically carried out, but rather simulated on the respective test
# turbine. The following data modification is done to the test turbine data. The actual test turbine data, including
# both power production data and environmental measurements, are taken from the actual turbine pair operation. Then,
# the power production from the designated test turbine on the range of wind speed over 9 m/s is increased by 5%,
# namely multiplied by a factor of 1.05, while all other variables are kept the same. No data modification of any
# kind is done to the data affiliated with the control turbine in the pitch angle adjustment pair.
#
# The third column of a respective dataset is the upgrade status variable, of which a zero means the test turbine is
# not modified yet, while a one means that the test turbine is modified. The upgrade status has no impact on the
# control turbine, as the control turbine remains unmodified throughout. The vortex generator installation takes
# effect on June 20, 2011, and the pitch angle adjustment takes effect on April 25, 2011.

from __future__ import annotations

import logging
import sys
import zipfile
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field

from wind_up import AnalysisSpec, Label, NonTurbineMeasurement, Reanalysis, Upgrade, WindFarmFacts
from wind_up.schema import ColumnSchema
from wind_up_v0.constants import OUTPUT_DIR, PROJECTROOT_DIR, DataColumns
from wind_up_v0.wind_funcs import calc_cp

sys.path.append(str(PROJECTROOT_DIR))
from benchmarking.baselines.v0_binned import V0BinnedMethod, v0_records
from examples.helpers import download_zenodo_data, format_and_print_results_table, setup_logger

if TYPE_CHECKING:
    from pathlib import Path

    from wind_up import AnalysisOutput

CACHE_DIR = PROJECTROOT_DIR / "cache" / "wedowind_example"
ASSESSMENT_NAME = "wedowind_example"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / ASSESSMENT_NAME
ZIP_FILENAME = "Turbine_Upgrade_Dataset.zip"
TIMEBASE = pd.Timedelta(minutes=10)
RATED_POWER_KW = 1500
ROTOR_DIAMETER_M = 80
SITE_ORIGIN = (40.0, -89.0)
TEST_TURBINE = "Test"
REF_TURBINE = "Ref"
WIND_SPEED = "V"
WIND_DIRECTION = "D"
UPGRADE_STATUS = "upgradestatus"
COLUMNS = ColumnSchema(turbine="turbine", active_power="power_kw", availability="available_s")
# This farm-wide exclusion applies to reference records as well as the subject.
UNUSABLE_DIRECTION = Label(name="unusable wind direction", column="unusable_direction", withholds=frozenset({"uplift"}))

logger = logging.getLogger(__name__)


class WeDoWindAnalysisConf(BaseModel):
    scada_file_name: str = Field(description="e.g. 'Turbine Upgrade Dataset(Pitch Angle Pair).csv'")
    unusable_wd_ranges: list[tuple[int, int]] = Field(description="directions to exclude from entire analysis")
    ref_wd_filter: list[int] = Field(
        description="directions to include in power performance analysis", default=[0, 360]
    )
    clip_rated_power_pp: bool
    use_rated_invalid_bins: bool = Field(
        description="use rated power bins which have been extrapolated in uplift calculation", default=False
    )


def download_wedowind_data_from_zenodo(cache_dir: Path = CACHE_DIR) -> None:
    logger.info("Downloading example data from Zenodo")
    if not (cache_dir / ZIP_FILENAME).exists():
        # https://zenodo.org/records/5516556
        download_zenodo_data(record_id="5516556", output_dir=cache_dir, filenames={ZIP_FILENAME})


def read_wedowind_pair(path: Path, filename: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive, archive.open(filename) as csv:
        raw = pd.read_csv(csv, parse_dates=["time"], index_col="time")
    raw.index = pd.to_datetime(raw.index, utc=True).rename("timestamp")
    raw.columns = raw.columns.str.replace(" ", "")
    return raw


def create_illustrative_layout() -> pd.DataFrame:
    # Offsets are based on Figure 5.6 of Data Science for Wind Energy (Yu Ding 2020).
    # The site's identity and the actual turbine positions are unavailable.
    coords_df = pd.DataFrame(
        {
            "name": [TEST_TURBINE, REF_TURBINE, "WT3", "WT4", "Mast", "MAST2"],
            "east_m": [500, 2200, 9836, 7571, 0, 9571],
            "north_m": [9136, 9436, 0, 2050, 9836, 50],
        }
    ).set_index("name")
    offsets = (coords_df - coords_df.mean()).loc[[TEST_TURBINE, REF_TURBINE, "Mast"]]
    points = [
        Geodesic.WGS84.Direct(*SITE_ORIGIN, float(np.degrees(np.arctan2(east, north))), float(np.hypot(east, north)))
        for east, north in offsets.itertuples(index=False, name=None)
    ]
    return pd.DataFrame(
        [(point["lat2"], point["lon2"]) for point in points], columns=["latitude", "longitude"], index=offsets.index
    )


def create_synthetic_reanalysis(start: pd.Timestamp) -> Reanalysis:
    # The legacy method requires a reanalysis input. This seeded placeholder keeps the
    # example reproducible without assigning real weather data to the unknown site.
    rng = np.random.default_rng(0)
    rows = 100
    return Reanalysis(
        name="synthetic reanalysis placeholder",
        latitude=SITE_ORIGIN[0],
        longitude=SITE_ORIGIN[1],
        records=pd.DataFrame(
            data={
                "100_m_hws_mean_mps": rng.uniform(5, 10, rows),
                "100_m_hwd_mean_deg-n_true": rng.uniform(0, 360, rows),
            },
            index=pd.date_range(start=start, periods=rows, freq="h"),
        ),
    )


def make_turbine_records(raw: pd.DataFrame, config: WeDoWindAnalysisConf) -> pd.DataFrame:
    unusable = pd.Series(data=False, index=raw.index)
    for start, end in config.unusable_wd_ranges:
        unusable |= raw[WIND_DIRECTION].between(start, end)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "turbine": turbine,
                    "power_kw": raw[power_column] * RATED_POWER_KW,
                    "available_s": TIMEBASE.total_seconds(),
                    UNUSABLE_DIRECTION.column: unusable,
                },
                index=raw.index,
            )
            for turbine, power_column in [(TEST_TURBINE, "y_test(normalized)"), (REF_TURBINE, "y_ctrl(normalized)")]
        ]
    )


def adapt_records_for_binned_method(records: pd.DataFrame, *, mast: NonTurbineMeasurement) -> pd.DataFrame:
    # The legacy preprocessor needs turbine wind and operating columns. Add these
    # proxies at its boundary. The facts retain the mast as a separate measurement.
    wind = mast.records.reindex(records.index)
    return v0_records(records, columns=COLUMNS, timebase=TIMEBASE).assign(
        **{
            DataColumns.wind_speed_mean: wind[mast.columns["wind_speed"]].to_numpy(),
            DataColumns.yaw_angle_mean: wind[mast.columns["wind_direction"]].to_numpy(),
            DataColumns.pitch_angle_mean: 0,
            DataColumns.gen_rpm_mean: 1000,
        }
    )


def generate_custom_exploratory_plots(
    scada_df: pd.DataFrame, assumed_rated_power_kw: float, rotor_diameter_m: int, out_dir: Path
) -> Path:
    """These custom plots are to help with SCADA data exploration.
    It was created because it was unclear how the SCADA data is related to the metadata so helped in looking for wakes
    in the data.

    Returns: path to directory containing the plots
    """
    custom_plots_dir_root = out_dir / "custom_plots"
    custom_plots_dir_timeseries = custom_plots_dir_root / "timeseries"

    custom_plots_dir_root.mkdir(exist_ok=True, parents=True)
    custom_plots_dir_timeseries.mkdir(exist_ok=True)

    for name, df in scada_df.groupby("turbine"):
        for col in df.columns:
            plt.figure()
            plt.scatter(df.index, df[col], s=1)
            title = f"{name} {col}"
            plt.xlabel("timestamp (UTC)")
            plt.ylabel(col)
            plt.xticks(rotation=90)
            plt.grid()
            plt.tight_layout()
            plt.savefig(custom_plots_dir_timeseries / f"{title}.png")
            plt.close()

    region2_power_margin = 0.2
    region2_df = scada_df[
        (scada_df["normalized_power"] > region2_power_margin)
        & (scada_df["normalized_power"] < (1 - region2_power_margin))
    ]

    binned_by_turbine = {}
    for name, df in region2_df.groupby("turbine"):
        plt.figure()
        plt.scatter(
            df[WIND_DIRECTION],
            calc_cp(
                power_kw=df["normalized_power"] * assumed_rated_power_kw,
                ws_ms=df[WIND_SPEED],
                air_density_kgpm3=1.2,
                rotor_diameter_m=rotor_diameter_m,
            ),
            s=1,
            alpha=0.7,
        )
        plt.ylim(0, 1.5)
        title = f"{name} Cp vs {WIND_DIRECTION}"
        plt.title(title)
        plt.xlabel(WIND_DIRECTION)
        plt.ylabel("Cp")
        plt.xticks(rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.savefig(custom_plots_dir_root / f"{title}.png")
        plt.close()
        # find mean normalized_power and V binned by D
        _df = df.copy()
        _df["direction_bin"] = pd.cut(_df[WIND_DIRECTION], bins=range(0, 361, 5))
        binned = _df.groupby("direction_bin", observed=False)[[WIND_DIRECTION, "normalized_power", WIND_SPEED]].mean()
        binned_by_turbine[name] = binned
        plt.figure()
        plt.plot(
            binned[WIND_DIRECTION],
            calc_cp(
                power_kw=binned["normalized_power"] * assumed_rated_power_kw,
                ws_ms=binned[WIND_SPEED],
                air_density_kgpm3=1.2,
                rotor_diameter_m=rotor_diameter_m,
            ),
            marker=".",
        )
        title = f"{name} mean Cp vs {WIND_DIRECTION}"
        plt.title(title)
        plt.xlabel(WIND_DIRECTION)
        plt.ylabel("Cp")
        plt.xticks(rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.savefig(custom_plots_dir_root / f"{title}.png")
        plt.close()

    plt.figure()
    for name, binned in binned_by_turbine.items():
        plt.plot(
            binned[WIND_DIRECTION],
            calc_cp(
                power_kw=binned["normalized_power"] * assumed_rated_power_kw,
                ws_ms=binned[WIND_SPEED],
                air_density_kgpm3=1.2,
                rotor_diameter_m=rotor_diameter_m,
            ),
            label=name,
            marker=".",
        )
    plt.ylim(0.2, 0.7)
    title = f"mean Cp vs {WIND_DIRECTION}"
    plt.title(title)
    plt.xlabel(WIND_DIRECTION)
    plt.ylabel("Cp")
    plt.xticks(rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.savefig(custom_plots_dir_root / f"{title}.png")
    plt.close()

    logger.info("Custom plots saved to directory: %s", custom_plots_dir_root)
    return custom_plots_dir_root


ANALYSIS_SPECIFIC_CONFIG = {
    "Pitch Angle": WeDoWindAnalysisConf(
        scada_file_name="Turbine Upgrade Dataset(Pitch Angle Pair).csv",
        unusable_wd_ranges=[(70, 140), (150, 180)],  # determined by inspecting custom Cp vs D plots
        ref_wd_filter=[160, 240],  # apparent wake free sector determined by inspecting detrending plots
        clip_rated_power_pp=False,  # rated power is apparently higher after the upgrade
        use_rated_invalid_bins=True,  # use extrapolated rated power results
    ),
    "Vortex Generator": WeDoWindAnalysisConf(
        scada_file_name="Turbine Upgrade Dataset(VG Pair).csv",
        unusable_wd_ranges=[(25, 115), (240, 315)],  # determined by inspecting custom Cp vs D plots
        ref_wd_filter=[115, 240],  # apparent wake free sector determined by inspecting detrending plots
        clip_rated_power_pp=True,  # Vortex Generators are not expected to increase rated power
    ),
}


def main_wedowind_analysis(
    analysis_name: str,
    *,
    generate_custom_plots: bool = True,
    save_plots: bool = True,
    cache_dir: Path = CACHE_DIR,
    analysis_output_dir: Path = ANALYSIS_OUTPUT_DIR,
    bootstrap_runs_override: int | None = None,
) -> AnalysisOutput:
    """Run a WeDoWind case through the v1 method interface and write its pairwise CSV."""
    if analysis_name not in ANALYSIS_SPECIFIC_CONFIG:
        msg = f"analysis_name must be one of {list(ANALYSIS_SPECIFIC_CONFIG)}"
        raise ValueError(msg)
    analysis_conf = ANALYSIS_SPECIFIC_CONFIG[analysis_name]

    download_wedowind_data_from_zenodo(cache_dir)
    raw = read_wedowind_pair(cache_dir / ZIP_FILENAME, analysis_conf.scada_file_name)
    records = make_turbine_records(raw, analysis_conf)
    if generate_custom_plots:
        plot_records = records[["turbine"]].assign(normalized_power=records["power_kw"] / RATED_POWER_KW)
        generate_custom_exploratory_plots(
            scada_df=plot_records.join(raw[[UPGRADE_STATUS, WIND_SPEED, WIND_DIRECTION, "rho", "S", "I"]]),
            assumed_rated_power_kw=RATED_POWER_KW,
            rotor_diameter_m=ROTOR_DIAMETER_M,
            out_dir=analysis_output_dir / analysis_name,
        )

    layout = create_illustrative_layout()
    mast_columns = {WIND_SPEED: "wind_speed_mps", WIND_DIRECTION: "wind_direction_deg"}
    mast = NonTurbineMeasurement(
        name="Mast",
        latitude=float(layout.loc["Mast", "latitude"]),
        longitude=float(layout.loc["Mast", "longitude"]),
        records=raw[list(mast_columns)].rename(columns=mast_columns),
        columns={"wind_speed": mast_columns[WIND_SPEED], "wind_direction": mast_columns[WIND_DIRECTION]},
    )
    upgraded = raw.index[raw[UPGRADE_STATUS] > 0]
    post_first, post_last = upgraded.min(), upgraded.max()
    history_start = max(raw.index.min(), post_first - pd.DateOffset(years=1))
    upgrade = Upgrade(name=analysis_name, turbines=[TEST_TURBINE], timing=post_first)

    # Rated power and rotor diameter are based on Table 5.1 of Data Science for Wind Energy (Yu Ding 2020).
    facts = WindFarmFacts(
        records=records,
        columns=COLUMNS,
        timebase=TIMEBASE,
        turbines=layout.loc[[TEST_TURBINE, REF_TURBINE]]
        .reset_index()
        .assign(
            rated_power_kw=RATED_POWER_KW,
            rotor_diameter_m=ROTOR_DIAMETER_M,
        ),
        upgrades=[upgrade],
        labels=[UNUSABLE_DIRECTION],
        non_turbine_measurements=[mast],
        reanalysis=[create_synthetic_reanalysis(history_start)],
    )
    analysis = AnalysisSpec(
        name=analysis_name,
        upgrade=upgrade.name,
        references=[REF_TURBINE],
        period=(history_start, post_last + TIMEBASE),
        exclude_labels=[UNUSABLE_DIRECTION.name],
        report="aep",
    )
    method = V0BinnedMethod(
        options={
            "ref_wd_filter": analysis_conf.ref_wd_filter,
            "years_offset_for_pre_period": 1,
            "years_for_lt_distribution": 1,
            "years_for_detrend": 1,
            "ws_bin_width": 1.0,
            "use_test_wtg_lt_distribution": True,
            "lt_first_dt_utc_start": history_start,
            "lt_last_dt_utc_start": history_start + pd.DateOffset(years=1) - TIMEBASE,
            "detrend_first_dt_utc_start": history_start,
            "detrend_last_dt_utc_start": post_first - pd.Timedelta(weeks=1) - TIMEBASE,
            # Preserve the original inclusive pre/post bounds for the legacy estimator.
            "prepost": {
                "pre_first_dt_utc_start": history_start,
                "pre_last_dt_utc_start": history_start + (post_last - post_first) - TIMEBASE,
                "post_first_dt_utc_start": post_first,
                "post_last_dt_utc_start": post_last,
            },
            "missing_scada_data_fields": [DataColumns.yaw_angle_min, DataColumns.yaw_angle_max],
            "optimize_northing_corrections": False,
            "clip_rated_power_pp": analysis_conf.clip_rated_power_pp,
            "use_rated_invalid_bins": analysis_conf.use_rated_invalid_bins,
            "bootstrap_runs_override": bootstrap_runs_override,
        },
        turbine_type={
            "turbine_type": "unknown turbine type",
            "cutout_ws_mps": 20,
            "normal_operation_pitch_range": (-10.0, 35.0),
            "normal_operation_genrpm_range": (0, 2000.0),
        },
        site_name="Mystery Wind Farm",
        auto_choose_refs=False,  # The analysis has a single declared reference.
        to_v0_records=partial(adapt_records_for_binned_method, mast=mast),
        out_dir=analysis_output_dir,
        save_plots=save_plots,
        cache_dir=cache_dir / analysis_name,
    )

    (output,) = method.estimate(facts, [analysis])
    if output.per_reference is None:
        messages = " | ".join(warning.message for warning in output.warnings)
        msg = f"WeDoWind {analysis_name} produced no pairwise results: {messages}"
        raise RuntimeError(msg)
    pairs = output.per_reference.drop(columns="report")
    pairs.to_csv(analysis_output_dir / analysis_name / "results_per_test_ref.csv", index=False)
    format_and_print_results_table(pairs)
    return output


if __name__ == "__main__":
    setup_logger(ANALYSIS_OUTPUT_DIR / "analysis.log")
    main_wedowind_analysis("Pitch Angle")
    main_wedowind_analysis("Vortex Generator")
