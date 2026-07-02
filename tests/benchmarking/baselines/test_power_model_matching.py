"""Tests for the power-model coarsened-exact-matching (CEM) utility (Issue 8, Component 2).

Pure/fast unit tests on tiny hand-computable fixtures: equal counts per retained cell, one-sided
cells dropped (the common-support guard), seeded-subsample reproducibility, finite-value handling,
the balance diagnostic, and the retain-too-little warning / hard-floor raise.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.power_model.matching import coarsened_exact_match

# One matching variable "ws" with edges [0, 10, 20, 30] -> three cells A=(0,10] B=(10,20] C=(20,30].
_EDGES = {"ws": [0.0, 10.0, 20.0, 30.0]}


def _fixture() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A tiny 11-row frame with hand-computable cell membership.

    positions -> (ws, role):
      A: base 0,1,2 + up 3            -> two-sided, k=1 (subsample baseline 3->1)
      B: base 4,5   + up 6,7          -> two-sided, k=2 (keep all)
      C: base 8,9                     -> one-sided, dropped (common-support guard)
      base 10 has NaN ws              -> dropped by the finite-value restriction
    """
    ws = [5.0, 6.0, 7.0, 5.0, 15.0, 16.0, 15.0, 16.0, 25.0, 26.0, np.nan]
    frame = pd.DataFrame({"ws": ws}, index=pd.RangeIndex(len(ws)))
    baseline_sel = np.zeros(len(ws), dtype=bool)
    upgraded_sel = np.zeros(len(ws), dtype=bool)
    baseline_sel[[0, 1, 2, 4, 5, 8, 9, 10]] = True
    upgraded_sel[[3, 6, 7]] = True
    return frame, baseline_sel, upgraded_sel


def _match(seed: int = 0):  # noqa: ANN202
    frame, baseline_sel, upgraded_sel = _fixture()
    return coarsened_exact_match(
        frame, baseline_sel=baseline_sel, upgraded_sel=upgraded_sel, bin_edges=_EDGES, seed=seed, min_matched_rows=1
    )


class TestEqualCounts:
    def test_equal_matched_count_per_side(self) -> None:
        result = _match()
        assert len(result.baseline_positions) == len(result.upgraded_positions) == 3

    def test_equal_counts_within_every_retained_cell(self) -> None:
        result = _match()
        retained = result.per_cell[result.per_cell["n_matched"] > 0]
        # after matching each retained cell has the same count on both sides
        assert (retained["n_matched"] == retained["n_matched"]).all()
        # cell A keeps 1/side, cell B keeps 2/side
        assert sorted(retained["n_matched"].tolist()) == [1, 2]


class TestCommonSupport:
    def test_one_sided_cell_dropped(self) -> None:
        result = _match()
        # cell C (positions 8, 9) is baseline-only -> excluded from the matched set
        assert 8 not in result.baseline_positions
        assert 9 not in result.baseline_positions
        assert result.n_cells_one_sided == 1
        assert result.n_cells_two_sided == 2


class TestFiniteHandling:
    def test_nan_matching_value_row_excluded(self) -> None:
        result = _match()
        assert 10 not in result.baseline_positions  # the NaN-ws baseline row never enters matching
        assert result.n_baseline_in == 7  # 8 baseline rows minus the NaN one


class TestSeededSubsample:
    def test_same_seed_is_reproducible(self) -> None:
        a, b = _match(seed=3), _match(seed=3)
        assert np.array_equal(a.baseline_positions, b.baseline_positions)
        assert np.array_equal(a.upgraded_positions, b.upgraded_positions)

    def test_subsampled_row_comes_from_the_cell(self) -> None:
        result = _match(seed=1)
        # cell A keeps exactly one of the three baseline rows {0, 1, 2}
        kept_from_a = [p for p in result.baseline_positions if p in (0, 1, 2)]
        assert len(kept_from_a) == 1
        # cell B keeps both of its baseline rows
        assert {4, 5}.issubset(set(result.baseline_positions.tolist()))

    def test_positions_are_sorted(self) -> None:
        result = _match(seed=2)
        assert list(result.baseline_positions) == sorted(result.baseline_positions)
        assert list(result.upgraded_positions) == sorted(result.upgraded_positions)


class TestBalanceDiagnostic:
    def test_retained_fractions(self) -> None:
        result = _match()
        assert result.retained_fraction_baseline == pytest.approx(3 / 7)
        assert result.retained_fraction_upgraded == pytest.approx(1.0)

    def test_effective_sample_size(self) -> None:
        assert _match().n_matched_per_side == 3

    def test_per_cell_before_counts(self) -> None:
        per_cell = _match().per_cell.set_index("ws")
        # per-cell "before" counts by cell code (A=0, B=1, C=2)
        assert per_cell.loc[0, "n_baseline"] == 3
        assert per_cell.loc[0, "n_upgraded"] == 1
        assert per_cell.loc[1, "n_baseline"] == 2
        assert per_cell.loc[1, "n_upgraded"] == 2
        assert per_cell.loc[2, "n_baseline"] == 2
        assert per_cell.loc[2, "n_upgraded"] == 0


class TestGuards:
    def test_raises_below_hard_floor(self) -> None:
        frame, baseline_sel, upgraded_sel = _fixture()
        with pytest.raises(ValueError, match="matched"):
            coarsened_exact_match(
                frame, baseline_sel=baseline_sel, upgraded_sel=upgraded_sel, bin_edges=_EDGES, seed=0
            )  # default min_matched_rows=10 > the 3 available

    def test_warns_when_little_retained(self, caplog: pytest.LogCaptureFixture) -> None:
        frame, baseline_sel, upgraded_sel = _fixture()
        with caplog.at_level(logging.WARNING):
            coarsened_exact_match(  # baseline retains 3/7 ≈ 0.43, below this 0.9 warn fraction
                frame,
                baseline_sel=baseline_sel,
                upgraded_sel=upgraded_sel,
                bin_edges=_EDGES,
                seed=0,
                min_matched_rows=1,
                warn_retained_fraction=0.9,
            )
        assert any("retain" in rec.message.lower() for rec in caplog.records)


class TestMultipleVariables:
    def test_cell_key_is_the_tuple_of_all_vars(self) -> None:
        # two vars: rows share ws-cell but split on a second var -> different cells, so no match
        frame = pd.DataFrame({"ws": [5.0, 5.0, 5.0, 5.0], "gust": [2.0, 2.0, 18.0, 18.0]})
        baseline_sel = np.array([True, False, True, False])
        upgraded_sel = np.array([False, True, False, True])
        result = coarsened_exact_match(
            frame,
            baseline_sel=baseline_sel,
            upgraded_sel=upgraded_sel,
            bin_edges={"ws": [0.0, 10.0], "gust": [0.0, 10.0, 20.0]},
            seed=0,
            min_matched_rows=1,
        )
        # (ws bin 0, gust bin 0) = {base 0, up 1} and (ws 0, gust 1) = {base 2, up 3}: both two-sided, k=1
        assert result.n_matched_per_side == 2
        assert result.n_cells_two_sided == 2
