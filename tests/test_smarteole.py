"""Tests running the Smarteole dataset through wind-up analysis."""

from pathlib import Path

import pytest

from examples.smarteole_example import (
    SmarteoleData,
    main_smarteole_analysis,
    unpack_smarteole_metadata,
    unpack_smarteole_scada,
    unpack_smarteole_toggle_data,
)

SMARTEOLE_DATA_DIR = Path(__file__).parents[1] / "tests/test_data/smarteole"


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
def test_smarteole_analysis(tmp_path: Path) -> None:
    """Test running the Smarteole analysis."""

    timebase_s = 600
    cache_subdir = tmp_path / f"timebase_{timebase_s}"
    cache_subdir.mkdir(parents=True, exist_ok=True)

    scada_df = unpack_smarteole_scada(
        timebase_s=timebase_s, scada_data_file=SMARTEOLE_DATA_DIR / "SMARTEOLE_WakeSteering_SCADA_1minData.csv"
    )
    metadata_df = unpack_smarteole_metadata(
        timebase_s=timebase_s, metadata_file=SMARTEOLE_DATA_DIR / "SMARTEOLE_WakeSteering_Coordinates_staticData.csv"
    )
    toggle_df = unpack_smarteole_toggle_data(
        timebase_s=timebase_s, toggle_file=SMARTEOLE_DATA_DIR / "SMARTEOLE_WakeSteering_ControlLog_1minData.csv"
    )
    smarteole_data = SmarteoleData(scada_df=scada_df, metadata_df=metadata_df, toggle_df=toggle_df)

    main_smarteole_analysis(
        smarteole_data=smarteole_data,
        reanalysis_file_path=SMARTEOLE_DATA_DIR / "ERA5T_50.00N_2.75E_100m_1hr_20200201_20200531.parquet",
        analysis_timebase_s=timebase_s,
        check_results=True,  # asserts expected results
        analysis_output_dir=tmp_path,
        cache_sub_dir=cache_subdir,
    )
