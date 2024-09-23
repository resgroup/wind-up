import re

import pytest

from wind_up.plots import scada_funcs_plots


class TestAxisLabelFromFieldName:
    @staticmethod
    @pytest.mark.parametrize(
        "field_name_unsupported", ["UnsupportedFieldName", "test_UnsupportedFieldName", "raw_UnsupportedFieldName"]
    )
    def test_unsupported_field_name(field_name_unsupported: str) -> None:
        fname_unsupported_lean = re.sub(r"^.*?_", "", field_name_unsupported)
        msg = (
            f"Failed to construct axis label for field '{field_name_unsupported}' "
            f"because {fname_unsupported_lean} does not have a default unit defined"
        )

        with pytest.raises(ValueError, match=msg):
            scada_funcs_plots._axis_label_from_field_name(field_name=field_name_unsupported)  # noqa: SLF001

    @staticmethod
    @pytest.mark.parametrize(
        ("field_name", "expected"),
        [
            ("ActivePowerMean", "ActivePowerMean [kW]"),
            ("test_ActivePowerMean", "test_ActivePowerMean [kW]"),
            ("raw_ActivePowerMean", "raw_ActivePowerMean [kW]"),
            ("WindSpeedMean", "WindSpeedMean [m/s]"),
            ("YawAngleMean", "YawAngleMean [deg]"),
            ("PitchAngleMean", "PitchAngleMean [deg]"),
            ("GenRpmMean", "GenRpmMean [RPM]"),
            ("AmbientTemp", "AmbientTemp [degC]"),
        ],
    )
    def test_supported_field_name(field_name: str, expected: str) -> None:
        actual = scada_funcs_plots._axis_label_from_field_name(field_name=field_name)  # noqa: SLF001
        assert actual == expected
