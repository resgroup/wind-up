"""Tests running the Smarteole dataset through wind-up analysis."""

import copy
from pathlib import Path

import pytest

from examples.smarteole_example import SmarteoleData, main_smarteole_analysis
from tests.conftest import SMARTEOLE_CACHE_DIR
from wind_up.interface import AssessmentInputs

SMARTEOLE_DATA_DIR = Path(__file__).parents[1] / "tests/test_data/smarteole"


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
def test_smarteole_analysis(
    smarteole_assessment_inputs: tuple[AssessmentInputs, SmarteoleData], tmp_path: Path
) -> None:
    """Test running the Smarteole analysis."""
    smarteole_data = copy.deepcopy(smarteole_assessment_inputs[1])

    main_smarteole_analysis(
        smarteole_data=smarteole_data,
        reanalysis_file_path=SMARTEOLE_DATA_DIR / "ERA5T_50.00N_2.75E_100m_1hr_20200201_20200531.parquet",
        analysis_timebase_s=600,
        check_results=True,  # asserts expected results
        analysis_output_dir=tmp_path,
        cache_sub_dir=SMARTEOLE_CACHE_DIR,
    )

    assert (tmp_path / "plots/input_data_timeline_fig.png").exists()
