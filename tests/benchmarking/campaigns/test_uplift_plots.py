"""Tests for the campaign-level uplift plots and the reference aggregate behind them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from benchmarking.campaigns.uplift_plots import per_reference_uplifts, write_uplift_plots

if TYPE_CHECKING:
    from pathlib import Path

_PER_TURBINE = pd.DataFrame({"method": ["wind-up"] * 2, "test_wtg": ["T1", "T2"], "estimate": [0.02, 0.03]})
_FARM = pd.DataFrame({"method": ["wind-up"], "estimate": [0.025], "uplift_spread": [0.01], "n_guarded": [0]})


def _stability(uplifts: dict[str, float], *, screened: tuple[str, ...] = ()) -> pd.DataFrame:
    """The per-(test turbine, reference) table the report writes, two test turbines deep."""
    rows = [
        {
            "method": "wind-up",
            "test_wtg": test,
            "turbine": reference,
            "uplift": uplift,
            "actual_energy": 1_000_000.0,
            "n_records": 50_000,
            "screened": reference in screened,
        }
        for test in ("T1", "T2")
        for reference, uplift in uplifts.items()
    ]
    return pd.DataFrame(rows)


class TestOneRowPerReference:
    def test_the_repeated_readings_collapse_to_one(self) -> None:
        refs = per_reference_uplifts(_stability({"R1": 0.001, "R2": -0.002}))
        assert list(refs["turbine"]) == ["R1", "R2"]
        assert list(refs["uplift"]) == [0.001, -0.002]

    def test_a_reference_screened_for_any_test_turbine_counts_as_screened(self) -> None:
        refs = per_reference_uplifts(_stability({"R1": 0.05}, screened=("R1",))).set_index("turbine")
        assert bool(refs.loc["R1", "screened"])

    def test_an_empty_table_gives_an_empty_frame_with_the_columns(self) -> None:
        refs = per_reference_uplifts(pd.DataFrame())
        assert refs.empty
        assert "screened" in refs.columns


class TestWhatIsDrawn:
    def test_all_three_plots_are_written(self, tmp_path: Path) -> None:
        written = write_uplift_plots(
            tmp_path,
            per_turbine=_PER_TURBINE,
            stability=_stability({"R1": 0.001, "R2": -0.002, "R3": 0.05}, screened=("R3",)),
            rated_power_kw=2300.0,
            farm=_FARM,
        )
        assert [p.name for p in written] == [
            "uplift_distributions.png",
            "per_turbine_uplift.png",
            "farm_uplift.png",
        ]
        assert all(p.exists() for p in written)

    def test_a_campaign_with_no_references_still_draws(self, tmp_path: Path) -> None:
        # the estimates exist even when nothing was offered to compare them against
        written = write_uplift_plots(
            tmp_path, per_turbine=_PER_TURBINE, stability=pd.DataFrame(), rated_power_kw=2300.0, farm=_FARM
        )
        assert len(written) == 3

    def test_nothing_is_drawn_without_a_test_turbine(self, tmp_path: Path) -> None:
        written = write_uplift_plots(
            tmp_path, per_turbine=pd.DataFrame(), stability=pd.DataFrame(), rated_power_kw=2300.0, farm=_FARM
        )
        assert written == []
        assert not list(tmp_path.glob("*.png"))


@pytest.mark.parametrize("screened", [(), ("R3",)])
def test_the_screened_reference_is_drawn_either_way(tmp_path: Path, screened: tuple[str, ...]) -> None:
    """It is kept out of the spread, not out of the picture: a reader should see what was rejected."""
    written = write_uplift_plots(
        tmp_path,
        per_turbine=_PER_TURBINE,
        stability=_stability({"R1": 0.001, "R3": 0.05}, screened=screened),
        rated_power_kw=2300.0,
        farm=_FARM,
    )
    assert len(written) == 3
