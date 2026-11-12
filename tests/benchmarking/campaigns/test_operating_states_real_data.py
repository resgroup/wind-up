"""Operating-state regression tests on six months of real Hill of Towie SCADA (Aug 2018 - Jan 2019), all 21 turbines.

Synthetic tests pin each rule; only real SCADA shows what the rules do together: availability counters that
disagree with power, stops before and after the Balancing Mechanism start, noise modes, icing, derates. The
fixture (``benchmarking.baselines.make_operating_state_fixture``) spans the BM start, so every Hill of Towie
label is exercised.

What is held is what an analyst reads off ``02_operating_states``:

* **the hours per turbine, month and state**, with each state's validity for northing, waking and uplift,
  against the golden table recorded with the labels the analysts signed off -- any change to the labelling
  shows up here, as the rows that moved;
* **the label checks** -- hours of a parked rotor counted as running and of a running rotor counted as down,
  the two mislabels that move a result most -- against their golden values;
* **the site's rules as an analyst states them** -- no BM curtailment before the BM start, noise mode only on
  the noise-mode turbines, a stop before the BM start never normal operation, a parked rotor never running.

Regenerate the golden tables only on purpose, from the code whose labels are the new reference.
"""

from __future__ import annotations

import pandas as pd
import pytest

from benchmarking.baselines.make_operating_state_fixture import (
    FIXTURE,
    GOLDEN_CHECKS,
    GOLDEN_HOURS,
    label_checks_table,
    load_fixture,
    state_hours_table,
)
from benchmarking.campaigns.real import (
    BM_CURTAILMENT,
    HOT_OPERATING_STATE,
    ICING,
    NOISE_MODE,
    label_hot_operating_states,
)
from benchmarking.harness.operating_state import NORMAL_OPERATION, STATE_COL
from benchmarking.synthetic.sources.hill_of_towie import (
    HOT_BM_START,
    HOT_COLUMNS,
    HOT_NOISE_SETPOINTS_KW,
    HOT_POWER_SETPOINT_COL,
    HOT_STOP_SETPOINT_KW,
    HOT_TURBINE_COL,
)


def _fixture_available() -> bool:
    """Whether the fixture holds real Parquet rather than an unsmudged git-lfs pointer."""
    try:
        with FIXTURE.open("rb") as handle:
            return handle.read(4) == b"PAR1"
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _fixture_available(), reason="Hill of Towie operating-state fixture not available (git-lfs not pulled)"
)


@pytest.fixture(scope="module")
def labelled() -> pd.DataFrame:
    """The fixture with the Hill of Towie labels, labelled once for the module."""
    return label_hot_operating_states(load_fixture())


def _moved_rows(actual: pd.DataFrame, golden: pd.DataFrame, *, keys: list[str], value: str) -> pd.DataFrame:
    """The rows whose ``value`` differs from the golden table's, with both values side by side."""
    merged = actual.merge(golden, on=keys, how="outer", suffixes=("_now", "_golden")).fillna(
        {f"{value}_now": 0.0, f"{value}_golden": 0.0}
    )
    moved = merged[(merged[f"{value}_now"] - merged[f"{value}_golden"]).abs() > 1e-9]
    return moved.sort_values(keys).reset_index(drop=True)


def test_hours_per_state_match_the_signed_off_labels(labelled: pd.DataFrame) -> None:
    keys = ["turbine", "month", STATE_COL, "valid_northing", "waking_state", "valid_uplift"]
    golden = pd.read_csv(GOLDEN_HOURS, dtype=dict.fromkeys(keys, str))
    moved = _moved_rows(state_hours_table(labelled), golden, keys=keys, value="hours")
    assert moved.empty, f"{len(moved)} turbine-month-state rows moved (hours now vs golden):\n{moved.head(30)}"


def test_label_checks_match_the_signed_off_labels(labelled: pd.DataFrame) -> None:
    actual = label_checks_table(labelled).set_index("turbine")
    golden = pd.read_csv(GOLDEN_CHECKS).set_index("turbine")
    diff = (actual - golden).abs()
    assert (diff < 1e-9).all().all(), (
        f"label checks moved (hours now, golden):\n{actual.join(golden, rsuffix='_golden')}"
    )


def test_the_site_labels_occur(labelled: pd.DataFrame) -> None:
    """The fixture exercises the Hill of Towie labels, so a rule that stops firing is noticed.

    High wind derate (20 m/s and over) does not occur in this window.
    """
    missing = {NOISE_MODE, BM_CURTAILMENT, ICING} - set(labelled[STATE_COL])
    assert not missing, f"no records labelled {sorted(missing)}"


def test_no_bm_curtailment_before_the_bm_start(labelled: pd.DataFrame) -> None:
    before = labelled.index < HOT_BM_START
    assert not (labelled.loc[before, STATE_COL] == BM_CURTAILMENT).any()


def test_noise_mode_only_on_the_noise_mode_turbines(labelled: pd.DataFrame) -> None:
    noisy = set(labelled.loc[labelled[STATE_COL] == NOISE_MODE, HOT_TURBINE_COL])
    assert noisy, "no noise mode in the fixture"
    assert noisy <= set(HOT_NOISE_SETPOINTS_KW)


def test_a_stop_before_the_bm_start_is_never_normal_operation(labelled: pd.DataFrame) -> None:
    stopped = (labelled[HOT_POWER_SETPOINT_COL] == HOT_STOP_SETPOINT_KW) & (labelled.index < HOT_BM_START)
    assert stopped.any(), "no stops before the BM start in the fixture"
    assert not (labelled.loc[stopped, STATE_COL] == NORMAL_OPERATION).any()


def test_a_parked_rotor_is_never_normal_operation(labelled: pd.DataFrame) -> None:
    parked = labelled[HOT_COLUMNS.pitch] > HOT_OPERATING_STATE["parked_pitch_above_deg"]
    assert parked.any(), "no parked records in the fixture"
    assert not (labelled.loc[parked, STATE_COL] == NORMAL_OPERATION).any()
