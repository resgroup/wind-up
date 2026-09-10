"""Tests for the R1 northing fixture's declaration and its bites/fixed verdicts.

The fixture's actual runs are a driver (they need the Hill of Towie download and the power
model); what is unit-tested here is that each cell of the 2x2 is declared as intended and that
the verdict arithmetic says what the acceptance thresholds mean.
"""

from __future__ import annotations

import pandas as pd
import pytest

from benchmarking.campaigns.northing_fixture import (
    CAMPAIGN_START,
    FAULT_OFFSET_DEG,
    FAULT_TURBINE,
    FIXTURE_REFERENCES,
    FIXTURE_TEST_WTG,
    analysis_period,
    fault_time,
    fixture_campaign,
    verdict_table,
)


class TestFixtureCampaign:
    @pytest.mark.parametrize("mode", ["prepost", "toggle"])
    def test_the_test_turbine_is_never_a_reference(self, mode: str) -> None:
        campaign = fixture_campaign(mode, faulted=True, northing=True)
        assert campaign.upgraded_turbines == [FIXTURE_TEST_WTG]
        assert FIXTURE_TEST_WTG not in campaign.candidate_references
        assert set(campaign.candidate_references) == set(FIXTURE_REFERENCES)

    def test_t05_is_not_in_the_fixture(self) -> None:
        """T05 is T06's nearest neighbour but carries real northing steps over 2017-2018."""
        campaign = fixture_campaign("prepost", faulted=False, northing=True)
        assert "T05" not in campaign.turbines

    @pytest.mark.parametrize("mode", ["prepost", "toggle"])
    def test_faulted_injects_one_step_on_a_reference(self, mode: str) -> None:
        campaign = fixture_campaign(mode, faulted=True, northing=True)
        assert len(campaign.faults) == 1
        fault = campaign.faults[0]
        # the fault must land on a reference: both v0 and power_model key on reference direction
        assert fault.turbine in FIXTURE_REFERENCES
        assert fault.turbine == FAULT_TURBINE
        assert fault.offset_deg == FAULT_OFFSET_DEG
        assert fault.at == fault_time(mode)

    @pytest.mark.parametrize("mode", ["prepost", "toggle"])
    def test_clean_injects_no_fault(self, mode: str) -> None:
        assert fixture_campaign(mode, faulted=False, northing=True).faults == []

    def test_northing_on_leaves_offsets_undeclared_so_they_are_discovered(self) -> None:
        assert fixture_campaign("prepost", faulted=True, northing=True).north_offsets is None

    def test_northing_off_declares_an_empty_table_rather_than_none(self) -> None:
        """An empty list means "apply exactly these" -- so the northed column is an uncorrected copy."""
        assert fixture_campaign("prepost", faulted=True, northing=False).north_offsets == []

    def test_the_fault_never_reaches_the_public_spec(self) -> None:
        spec = fixture_campaign("prepost", faulted=True, northing=True).spec()
        assert not hasattr(spec, "faults")
        assert spec.north_offsets is None

    def test_prepost_faults_at_the_changeover(self) -> None:
        assert fault_time("prepost") == CAMPAIGN_START

    def test_toggle_faults_mid_campaign(self) -> None:
        _, end = analysis_period("toggle")
        assert CAMPAIGN_START < fault_time("toggle") < end

    def test_an_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            fixture_campaign("sideways", faulted=False, northing=True)


def _table(errors: dict[tuple[bool, bool], float]) -> pd.DataFrame:
    """A one-mode, one-method 2x2 from ``{(faulted, northing): signed_error_fraction}``."""
    return pd.DataFrame(
        [
            {
                "mode": "prepost",
                "method": "power_model",
                "faulted": faulted,
                "northing": northing,
                "signed_error": error,
            }
            for (faulted, northing), error in errors.items()
        ]
    )


class TestVerdictTable:
    def test_a_fault_that_bites_and_is_fixed_passes_all_three(self) -> None:
        verdicts = verdict_table(
            _table({(False, False): 0.003, (True, False): 0.030, (False, True): 0.003, (True, True): 0.004})
        )
        row = verdicts.iloc[0]
        assert row["bites_pp"] == pytest.approx(2.7)
        assert bool(row["bites"])
        assert bool(row["fixed"])
        assert bool(row["no_harm"])

    def test_a_fault_too_small_to_bite_is_reported_as_such(self) -> None:
        verdicts = verdict_table(
            _table({(False, False): 0.003, (True, False): 0.008, (False, True): 0.003, (True, True): 0.003})
        )
        assert not bool(verdicts.iloc[0]["bites"])

    def test_northing_that_does_not_close_the_gap_fails_fixed(self) -> None:
        verdicts = verdict_table(
            _table({(False, False): 0.003, (True, False): 0.030, (False, True): 0.003, (True, True): 0.020})
        )
        assert bool(verdicts.iloc[0]["bites"])
        assert not bool(verdicts.iloc[0]["fixed"])

    def test_northing_that_hurts_clean_data_fails_no_harm(self) -> None:
        verdicts = verdict_table(
            _table({(False, False): 0.003, (True, False): 0.030, (False, True): 0.012, (True, True): 0.004})
        )
        assert not bool(verdicts.iloc[0]["no_harm"])

    def test_the_sign_of_the_error_does_not_matter_only_its_size(self) -> None:
        verdicts = verdict_table(
            _table({(False, False): 0.003, (True, False): -0.030, (False, True): 0.003, (True, True): -0.004})
        )
        assert bool(verdicts.iloc[0]["bites"])
        assert bool(verdicts.iloc[0]["fixed"])

    def test_an_incomplete_two_by_two_is_skipped_rather_than_half_judged(self) -> None:
        partial = _table({(False, False): 0.003, (True, False): 0.030})
        assert verdict_table(partial).empty
