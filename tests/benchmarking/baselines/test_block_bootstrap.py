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

from benchmarking.baselines.block_bootstrap import bootstrap_ratio_uplift, relative_scatter

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


def _run(case: dict, *, block_hours: float = 48.0, n_resamples: int = 400, seed: int = 0, **kw: object):  # noqa: ANN202
    return bootstrap_ratio_uplift(**case, block_hours=block_hours, n_resamples=n_resamples, seed=seed, **kw)


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

    def test_a_block_longer_than_the_campaign_falls_back_rather_than_claiming_certainty(self) -> None:
        """One block spans the campaign, so no resample can vary: the bootstrap has nothing to say.

        It used to return ~1e-15 — float residue, which a reader sees as "+/-0.0 pp" and reads as
        certainty. Now the bootstrap reports NaN and the fallback carries the cell.
        """
        result = _run(_case(n=1008), block_hours=1000.0)
        assert result.n_blocks == 1
        cell = result.cells["overall"]
        assert np.isnan(cell.sigma_bootstrap)
        assert np.isfinite(cell.sigma)
        assert cell.sigma == cell.sigma_fallback


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


class TestTooFewRecordsToFallBackOn:
    """A cell too sparse to bootstrap must still get an honest, wide sigma — not 0, and not NaN (F33).

    Resampling draws whole *blocks*, so if a cell's records sit in one block, every resample scales
    numerator and denominator together (rho = k*test / k*ref) and the ratio never moves: the bootstrap
    returns near-total confidence in a number built from one or two points. Measured on real
    campaigns: coverage 0.158 at one record per side and 0.237 at two (target 0.683), with one cell
    reporting sigma exactly 0 while being 14 pp wrong. The fallback is what covers that regime.
    """

    def _sparse_case(self, n_per_side: int) -> dict:
        """A case whose 'sparse' cell holds exactly ``n_per_side`` records on each side."""
        case = _case(n=2016)
        n = len(case["times"])
        sparse = np.zeros(n, dtype=bool)
        sparse[np.flatnonzero(case["upgraded"])[:n_per_side]] = True
        sparse[np.flatnonzero(case["baseline"])[:n_per_side]] = True
        case["cell_membership"] = {"overall": np.ones(n, dtype=bool), "sparse": sparse}
        return case

    def test_a_one_record_cell_gets_a_finite_sigma_from_the_fallback(self) -> None:
        """The headline fix: 0 or NaN would both be worse answers than a wide number.

        The bootstrap still *reports* its collapsed value rather than hiding it behind NaN — that is
        an honest diagnostic, and ``max`` is what stops it reaching the caller.
        """
        cells = _run(self._sparse_case(1)).cells
        sparse = cells["sparse"]
        assert sparse.sigma_bootstrap < sparse.sigma_fallback, "the bootstrap collapses on one record"
        assert np.isfinite(sparse.sigma)
        assert sparse.sigma == sparse.sigma_fallback

    def test_the_sparse_sigma_is_much_wider_than_the_well_populated_one(self) -> None:
        cells = _run(self._sparse_case(1)).cells
        assert cells["sparse"].sigma > 10 * cells["overall"].sigma

    def test_sigma_narrows_as_records_are_added(self) -> None:
        """The fallback scales as sqrt(1/n_on + 1/n_off), so more data must mean less uncertainty."""
        sigmas = [_run(self._sparse_case(n)).cells["sparse"].sigma for n in (1, 2, 4, 8)]
        assert sigmas == sorted(sigmas, reverse=True)

    def test_a_well_populated_cell_keeps_its_bootstrap(self) -> None:
        """The fallback must not disturb the regime the bootstrap already covers (F32)."""
        cell = _run(self._sparse_case(1)).cells["overall"]
        assert np.isfinite(cell.sigma_bootstrap)
        assert cell.sigma == cell.sigma_bootstrap
        assert cell.sigma_bootstrap > cell.sigma_fallback, "the bootstrap sees structure the fallback cannot"

    def test_both_components_are_always_reported(self) -> None:
        """So a blend rule can be re-judged from a saved sweep without re-running it."""
        for cell in _run(self._sparse_case(1)).cells.values():
            assert np.isfinite(cell.sigma_fallback)

    def test_the_reported_sigma_is_the_larger_of_the_two(self) -> None:
        for cell in _run(self._sparse_case(4)).cells.values():
            expected = np.nanmax([cell.sigma_bootstrap, cell.sigma_fallback])
            assert cell.sigma == pytest.approx(expected)

    def test_the_fallback_never_reduces_the_bootstrap(self) -> None:
        """max() is the safe direction for a reliability indicator."""
        for cell in _run(_case()).cells.values():
            if np.isfinite(cell.sigma_bootstrap):
                assert cell.sigma >= cell.sigma_bootstrap

    def test_the_finite_fraction_is_still_reported_so_the_reason_is_visible(self) -> None:
        cell = _run(self._sparse_case(1)).cells["sparse"]
        assert np.isfinite(cell.frac_resamples_finite)


class TestRelativeScatter:
    def test_it_recovers_a_known_per_record_noise_level(self) -> None:
        """The fallback is only as good as this: it is the campaign's own measured scatter."""
        case = _case(noise=0.05)
        s_rel = relative_scatter(
            case["test_power"], case["ref_total"], upgraded=case["upgraded"], baseline=case["baseline"]
        )
        assert s_rel == pytest.approx(0.05, rel=0.15)

    def test_it_tracks_the_noise(self) -> None:
        cases = [_case(noise=x) for x in (0.02, 0.05, 0.10)]
        scatters = [
            relative_scatter(c["test_power"], c["ref_total"], upgraded=c["upgraded"], baseline=c["baseline"])
            for c in cases
        ]
        assert scatters == sorted(scatters)

    def test_it_survives_reference_power_near_zero(self) -> None:
        """A per-record ratio would explode near cut-in; the ratio-of-sums form does not."""
        case = _case()
        case["ref_total"][:100] = 1e-9  # a near-cut-in stretch
        s_rel = relative_scatter(
            case["test_power"], case["ref_total"], upgraded=case["upgraded"], baseline=case["baseline"]
        )
        assert np.isfinite(s_rel)


class TestPerfectDataMayReportZero:
    """Nothing may preclude a zero sigma by construction.

    With noiseless, perfectly-matched data the uplift really is determined, so 0 is the correct
    answer and a model that cannot express it is mis-specified. There is no irreducible floor to
    justify one either: F31 tested exactly that hypothesis over campaigns up to a year and found
    sigma kept shrinking and kept tracking the error down to 0.135 pp.

    An earlier version NaN-ed a zero spread to trap the 1-record artefact. That punished this
    legitimate case to catch that one; the fallback traps the artefact without the collateral.
    """

    def _noiseless(self) -> dict:
        case = _case(noise=0.0)
        # exactly k * ref, times (1 + uplift) when on: no scatter at all
        case["test_power"] = 0.8 * case["ref_total"] * np.where(case["upgraded"], 1.03, 1.0)
        return case

    def test_both_components_vanish_on_noiseless_data(self) -> None:
        cell = _run(self._noiseless()).cells["overall"]
        assert cell.sigma_bootstrap == pytest.approx(0.0, abs=1e-9)
        assert cell.sigma_fallback == pytest.approx(0.0, abs=1e-9)

    def test_the_reported_sigma_reaches_zero(self) -> None:
        cell = _run(self._noiseless()).cells["overall"]
        assert cell.sigma == pytest.approx(0.0, abs=1e-9)
        assert not np.isnan(cell.sigma), "0 is the right answer here, not 'cannot estimate'"

    def test_the_scatter_measure_itself_vanishes(self) -> None:
        """The fallback is proportional to s_rel, so it can only reach 0 if this does."""
        case = self._noiseless()
        s_rel = relative_scatter(
            case["test_power"], case["ref_total"], upgraded=case["upgraded"], baseline=case["baseline"]
        )
        assert s_rel == pytest.approx(0.0, abs=1e-9)

    def test_sigma_scales_down_smoothly_as_noise_falls(self) -> None:
        """Approaching zero continuously, not hitting a floor."""
        sigmas = []
        for noise in (0.10, 0.05, 0.02, 0.005):
            case = _case(noise=noise)
            sigmas.append(_run(case).cells["overall"].sigma)
        assert sigmas == sorted(sigmas, reverse=True)
        assert sigmas[-1] < sigmas[0] / 10, "no floor is arresting the descent"
