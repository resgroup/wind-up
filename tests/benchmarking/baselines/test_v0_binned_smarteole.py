"""The SMARTEOLE analysis of ``tests/test_smarteole.py`` through the v1 interface, with v0_binned."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from benchmarking.baselines.v0_binned import V0BinnedMethod
from examples.smarteole_utils import SmartEoleExtractor
from tests.conftest import TEST_DATA_FLD
from tests.test_smarteole import _prep_results
from wind_up.analysis import AnalysisSpec
from wind_up.facts import Reanalysis, Upgrade, WindFarmFacts
from wind_up.schema import ColumnSchema

if TYPE_CHECKING:
    from pathlib import Path

    from wind_up.analysis import AnalysisOutput

SMARTEOLE_DIR = TEST_DATA_FLD / "smarteole"
TIMEBASE = pd.Timedelta(minutes=10)
FIRST = pd.Timestamp("2020-02-17 16:30:00+0000")
END = pd.Timestamp("2020-05-25 00:00:00+0000")
LAST = END - TIMEBASE
NORTHING = {
    "SMV1": 5.750994540354649,
    "SMV2": 5.690999999999994,
    "SMV3": 5.558000000000042,
    "SMV4": 5.936999999999996,
    "SMV5": 6.797253350869262,
    "SMV6": 5.030130916842758,
    "SMV7": 4.605999999999972,
}
COLUMNS = ColumnSchema(
    turbine="turbine",
    active_power="active_power_avg",
    wind_speed="wind_speed_avg",
    wind_speed_sd="wind_speed_std",
    gen_rpm="generator_speed_avg",
    pitch="blade_1_pitch_angle_avg",
    nacelle_position="nacelle_position_avg",
    ambient_temp="temperature_avg",
    availability="fully_available_s",
)
WD_FILTER_MARGIN = 3 + 7 * TIMEBASE.total_seconds() / 600


def smarteole_facts() -> tuple[WindFarmFacts, AnalysisSpec]:
    """Return the facts of the SMARTEOLE example and its analysis: SMV6 and SMV5 against SMV7."""
    extractor = SmartEoleExtractor(data_dir=SMARTEOLE_DIR, analysis_timebase_s=int(TIMEBASE.total_seconds()))
    scada = extractor.unpack_smarteole_scada()
    scada["fully_available_s"] = TIMEBASE.total_seconds()  # as wind-up 0.7, which ignored the derate flag
    metadata = extractor.unpack_smarteole_metadata()
    toggle = extractor.unpack_smarteole_toggle_data()
    scada = scada.merge(toggle["yaw_offset_command"], left_index=True, right_index=True, how="left")
    scada["yaw_offset_command"] = scada["yaw_offset_command"].where(scada["turbine"] == "SMV6", 0)
    turbines = pd.DataFrame(
        {
            "name": metadata["Turbine"],
            "latitude": metadata["Latitude"],
            "longitude": metadata["Longitude"],
            "rated_power_kw": 2050.0,
            "rotor_diameter_m": 82.0,
        }
    )
    reanalysis = Reanalysis(
        name="ERA5T_50.00N_2.75E_100m_1hr",
        latitude=50.0,
        longitude=2.75,
        records=pd.read_parquet(SMARTEOLE_DIR / "ERA5T_50.00N_2.75E_100m_1hr_20200201_20200531.parquet"),
    )
    facts = WindFarmFacts(
        records=scada,
        columns=COLUMNS,
        timebase=TIMEBASE,
        turbines=turbines,
        upgrades=[
            Upgrade(
                name="wake steering",
                turbines=["SMV6", "SMV5"],
                timing=toggle,
                settling_time=pd.Timedelta(seconds=120),
            )
        ],
        north_offsets=[(wtg, FIRST, offset) for wtg, offset in NORTHING.items()],
        reanalysis=[reanalysis],
    )
    analysis = AnalysisSpec(
        name="smarteole_example",
        upgrade="wake steering",
        subjects=["SMV6", "SMV5"],
        references=["SMV7"],
        period=(FIRST, END),
        report="campaign",
    )
    return facts, analysis


def smarteole_method(out_dir: Path) -> V0BinnedMethod:
    """Return v0_binned with the SMARTEOLE example's own v0 settings."""
    return V0BinnedMethod(
        options={
            "require_ref_wake_free": True,
            "detrend_min_hours": 12,
            "ref_wd_filter": [207 - WD_FILTER_MARGIN, 236 + WD_FILTER_MARGIN],
            "filter_all_test_wtgs_together": True,
            "ref_super_wtgs": [],
            "analysis_first_dt_utc_start": FIRST,
            "lt_first_dt_utc_start": FIRST,
            "lt_last_dt_utc_start": LAST,
            "detrend_first_dt_utc_start": FIRST,
            "detrend_last_dt_utc_start": LAST,
            "years_for_lt_distribution": 0,
            "years_for_detrend": 0,
            "ws_bin_width": 1.0,
            "toggle": {
                "name": "wake steering",
                "detrend_data_selection": "use_toggle_off_data",
                "pairing_filter_method": "any_within_timedelta",
                "pairing_filter_timedelta_seconds": 3600,
            },
        },
        turbine_type={
            "turbine_type": "Senvion-MM82-2050",
            "cutout_ws_mps": 25,
            "normal_operation_pitch_range": (-10.0, 35.0),
            "normal_operation_genrpm_range": (250.0, 2000.0),
            "rpm_v_pw_margin_factor": 0.05,
            "pitch_to_stall": False,
        },
        site_name="Sole du Moulin Vieux",
        auto_choose_refs=False,
        out_dir=out_dir,
    )


def smarteole_output(out_dir: Path) -> AnalysisOutput:
    """Return the SMARTEOLE analysis answered by v0_binned."""
    facts, analysis = smarteole_facts()
    (output,) = smarteole_method(out_dir).estimate(facts, [analysis])
    return output


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
def test_the_smarteole_analysis_gives_v0s_results(tmp_path: Path) -> None:
    output = smarteole_output(tmp_path)

    assert output.per_reference is not None
    expected = pd.DataFrame(
        {
            "turbine": ["SMV6", "SMV5"],
            "reference": ["SMV7", "SMV7"],
            "energy uplift": ["-1.0%", "3.1%"],
            "uplift uncertainty": ["0.6%", "1.2%"],
            "uplift P95": ["-2.0%", "1.2%"],
            "uplift P5": ["-0.1%", "5.0%"],
            "valid hours toggle off": [132 + 3 / 6, 133 + 0 / 6],
            "valid hours toggle on": [136 + 0 / 6, 137 + 1 / 6],
            "mean power toggle on": [1148, 994],
        },
        index=[0, 1],
    )
    pd.testing.assert_frame_equal(_prep_results(output.per_reference.drop(columns="report")), expected)
