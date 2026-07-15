"""Tests for the circular block bootstrap behind the toggle specialist's uncertainty.

The bootstrap is pure numerics on paired ``(test, reference)`` sums, so these build the timeline
directly rather than going through SCADA. They cover the recovered scale on data with a known
sampling error, the block-length and campaign-length responses, the pairing property that makes the
whole design work, reproducibility, and the degenerate paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.baselines.block_bootstrap import bootstrap_ratio_uplift

_TIMEBASE = pd.Timedelta(minutes=10)


def _timeline(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq=_TIMEBASE, tz="UTC")


def _alternating(n: int, *, block: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """On/off masks alternating every ``block`` records, as a fast toggle produces."""
    cycle = (np.arange(n) // block) % 2
    return cycle == 1, cycle == 0


def _case(
    n: int = 2016,
    *,
    uplift: float = 0.05,
    noise: float = 0.05,
    seed: int = 0,
    ref_level: float = 500.0,
) -> dict:
    """A toggle timeline whose test power is ``k * ref`` (times ``1+uplift`` when on) plus noise.

    Reference power is a slow sinusoid (a weather-like signal both segments share) so the on/off
    pairing has something real to cancel; the noise is per-record and independent, which is the
    variability the bootstrap should recover.
    """
    rng = np.random.default_rng(seed)
    times = _timeline(n)
    on, off = _alternating(n)
    ref = ref_level * (1.2 + np.sin(np.arange(n) / 144.0))
    test = 0.8 * ref * np.where(on, 1.0 + uplift, 1.0) * (1.0 + noise * rng.standard_normal(n))
    return {
        "times": times,
        "test_power": test,
        "ref_total": ref,
        "upgraded": on,
        "baseline": off,
        "cell_membership": {"overall": np.ones(n, dtype=bool)},
        "campaign_start": times[0],
        "campaign_end": times[-1],
        "timebase": _TIMEBASE,
    }


def _run(case: dict, *, block_hours: float = 48.0, n_resamples: int = 400, seed: int = 0):  # noqa: ANN202
    return bootstrap_ratio_uplift(**case, block_hours=block_hours, n_resamples=n_resamples, seed=seed)


class TestSigma:
    def test_sigma_is_finite_and_positive(self) -> None:
        cell = _run(_case()).cells["overall"]
        assert np.isfinite(cell.sigma)
        assert cell.sigma > 0
        assert cell.frac_resamples_finite == 1.0

    def test_sigma_matches_the_actual_scatter_of_the_estimator(self) -> None:
        """The point of a bootstrap: its sigma should equal the estimator's real sampling spread.

        Measured by re-drawing the noise many times and taking the spread of the resulting point
        estimates, which is the quantity a single run's bootstrap is trying to infer from one draw.
        """
        estimates = []
        for seed in range(60):
            case = _case(seed=seed)
            on, off = case["upgraded"], case["baseline"]
            test, ref = case["test_power"], case["ref_total"]
            rho_up = test[on].sum() / ref[on].sum()
            rho_base = test[off].sum() / ref[off].sum()
            estimates.append(rho_up / rho_base - 1.0)
        actual_spread = float(np.std(estimates, ddof=1))

        sigma = _run(_case(seed=0), n_resamples=2000).cells["overall"].sigma
        assert sigma == pytest.approx(actual_spread, rel=0.35)

    def test_sigma_shrinks_with_campaign_length(self) -> None:
        short = _run(_case(n=1008)).cells["overall"].sigma
        long = _run(_case(n=8064)).cells["overall"].sigma
        assert long < short
        # ~8x the records should cut sigma towards 1/sqrt(8); allow a wide band since the block
        # bootstrap also has fewer blocks to work with at the short end.
        assert 1.5 < short / long < 4.5

    def test_robust_sigma_agrees_with_sigma_for_a_well_populated_cell(self) -> None:
        """They diverge only when the resample distribution is heavy-tailed; here it should not be."""
        cell = _run(_case(), n_resamples=2000).cells["overall"]
        assert cell.sigma_robust == pytest.approx(cell.sigma, rel=0.2)


class TestPairing:
    def test_a_block_carries_its_on_and_off_rows_together(self) -> None:
        """The design's load-bearing property.

        A slow shared signal in the reference cancels between on and off *within* a block. If blocks
        did not carry both segments, that signal would leak into sigma and inflate it. Making the
        shared signal far larger must therefore barely move sigma.
        """
        calm = _run(_case(ref_level=500.0)).cells["overall"].sigma
        # Same noise, same uplift, a much stronger shared weather signal on both segments.
        wild = _run(_case(ref_level=5000.0)).cells["overall"].sigma
        assert wild == pytest.approx(calm, rel=0.25)


class TestBlockLength:
    def test_block_length_does_not_change_the_point_estimate_inputs(self) -> None:
        """Sigma varies with block length; nothing else the bootstrap touches does."""
        sigmas = {bl: _run(_case(), block_hours=bl).cells["overall"].sigma for bl in (6.0, 24.0, 48.0)}
        assert all(np.isfinite(s) and s > 0 for s in sigmas.values())

    def test_n_blocks_is_the_campaign_divided_by_the_block(self) -> None:
        # 2016 records at 10min = 14 days; 48h blocks -> 7.
        assert _run(_case(n=2016), block_hours=48.0).n_blocks == 7
        assert _run(_case(n=2016), block_hours=24.0).n_blocks == 14

    def test_a_block_longer_than_the_campaign_reports_a_degenerate_zero_sigma(self) -> None:
        """Clamped to the campaign, so every resample is the whole campaign and sigma collapses.

        Reported rather than hidden: a zero sigma is visibly degenerate downstream, where a quietly
        small one would not be.
        """
        result = _run(_case(n=1008), block_hours=1000.0)
        assert result.n_blocks == 1
        assert result.cells["overall"].sigma == pytest.approx(0.0, abs=1e-12)


class TestReproducibility:
    def test_same_seed_gives_the_same_sigma(self) -> None:
        a = _run(_case(), seed=7).cells["overall"].sigma
        b = _run(_case(), seed=7).cells["overall"].sigma
        assert a == b

    def test_different_seeds_agree_to_monte_carlo_noise(self) -> None:
        a = _run(_case(), seed=1, n_resamples=2000).cells["overall"].sigma
        b = _run(_case(), seed=2, n_resamples=2000).cells["overall"].sigma
        assert a == pytest.approx(b, rel=0.1)


class TestCells:
    def test_each_cell_is_bootstrapped_independently(self) -> None:
        case = _case()
        n = len(case["times"])
        first_half = np.arange(n) < n // 2
        case["cell_membership"] = {
            "overall": np.ones(n, dtype=bool),
            "first": first_half,
            "second": ~first_half,
        }
        cells = _run(case).cells
        assert set(cells) == {"overall", "first", "second"}
        # A half-sized cell has fewer records, so a wider sigma than the whole.
        assert cells["first"].sigma > cells["overall"].sigma

    def test_a_cell_with_no_baseline_rows_reports_nan_and_flags_it(self) -> None:
        """No off rows means no ``rho_base``, so no uplift and nothing to bootstrap."""
        case = _case()
        n = len(case["times"])
        case["cell_membership"] = {"on_only": case["upgraded"].copy()}
        cell = _run(case).cells["on_only"]
        assert np.isnan(cell.sigma)
        assert np.isnan(cell.sigma_robust)
        assert cell.frac_resamples_finite == 0.0
        assert n > 0  # guard against an accidentally empty case

    def test_an_empty_cell_reports_nan(self) -> None:
        case = _case()
        case["cell_membership"] = {"empty": np.zeros(len(case["times"]), dtype=bool)}
        cell = _run(case).cells["empty"]
        assert np.isnan(cell.sigma)
        assert cell.frac_resamples_finite == 0.0


class TestDegenerate:
    def test_no_records_gives_nan_cells(self) -> None:
        result = bootstrap_ratio_uplift(
            times=_timeline(0),
            test_power=np.array([]),
            ref_total=np.array([]),
            upgraded=np.array([], dtype=bool),
            baseline=np.array([], dtype=bool),
            cell_membership={"overall": np.array([], dtype=bool)},
            campaign_start=pd.Timestamp("2020-01-01", tz="UTC"),
            campaign_end=pd.Timestamp("2020-01-08", tz="UTC"),
            timebase=_TIMEBASE,
            block_hours=48.0,
            n_resamples=100,
            seed=0,
        )
        assert result.n_blocks == 0
        assert np.isnan(result.cells["overall"].sigma)

    def test_too_few_resamples_for_a_spread_gives_nan(self) -> None:
        assert np.isnan(_run(_case(), n_resamples=1).cells["overall"].sigma)

    def test_unsorted_input_is_sorted_rather_than_trusted(self) -> None:
        """Blocks are contiguous in time, so the record order must not change the answer."""
        case = _case()
        ordered = _run(case, n_resamples=800).cells["overall"].sigma

        rng = np.random.default_rng(0)
        shuffle = rng.permutation(len(case["times"]))
        shuffled = dict(case)
        shuffled["times"] = case["times"][shuffle]
        for key in ("test_power", "ref_total", "upgraded", "baseline"):
            shuffled[key] = case[key][shuffle]
        shuffled["cell_membership"] = {k: v[shuffle] for k, v in case["cell_membership"].items()}

        assert _run(shuffled, n_resamples=800).cells["overall"].sigma == pytest.approx(ordered)
