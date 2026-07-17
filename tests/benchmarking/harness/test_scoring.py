"""Tests for the scoring orchestrator and the two-part fairness guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarking.harness.campaign import campaign_windows
from benchmarking.harness.replicates import StudyConfig, build_replicates
from benchmarking.harness.scoring import _merge_diagnostics, score_one, score_study, truth_mask
from benchmarking.synthetic import HOT_COLUMNS, ConstantCpChange
from wind_up.constants import TIMESTAMP_COL

from .stubs import BiasedMethod, ConditionalOracleMethod, OracleMethod, RecordingMethod, UncertainMethod

PROFILE = [ConstantCpChange(delta=0.05)]


def _base_scada(turbines: tuple[str, ...] = ("T1", "T3", "T4", "T7")) -> pd.DataFrame:
    index = pd.date_range("2016-01-01", "2018-12-31", freq="1D", tz="UTC")
    frames = [
        pd.DataFrame(
            {
                HOT_COLUMNS.turbine: turbine,
                HOT_COLUMNS.active_power: 1000.0,
                HOT_COLUMNS.wind_speed: 8.0,
                HOT_COLUMNS.wind_speed_sd: 0.8,
                HOT_COLUMNS.gen_rpm: 1400.0,
            },
            index=index,
        )
        for turbine in turbines
    ]
    wf_df = pd.concat(frames)
    wf_df.index.name = TIMESTAMP_COL
    return wf_df


def _study(mode: str = "prepost", n_replicates: int = 4, seed: int = 0) -> StudyConfig:
    return StudyConfig(
        mode=mode,
        turbine_subset=["T1", "T3", "T4", "T7"],
        treatment_start_range=(pd.Timestamp("2017-01-01", tz="UTC"), pd.Timestamp("2017-12-31", tz="UTC")),
        min_pre_months=12,
        campaign_months=[3, 6],
        toggle_period=pd.Timedelta(days=14),
        n_replicates=n_replicates,
        seed=seed,
    )


def test_results_have_one_row_per_method_replicate_and_campaign_length() -> None:
    base = _base_scada()
    results = score_study(base, profile=PROFILE, methods=[OracleMethod(base)], study=_study(n_replicates=4))
    # 4 replicates x 2 campaign lengths x 1 method
    assert len(results) == 4 * 2
    assert set(results["campaign_months"]) == {3, 6}
    expected_columns = {"method", "profile", "replicate", "campaign_months", "estimate", "truth", "signed_error"}
    assert set(results.columns) >= expected_columns


def test_results_record_wall_time_per_run() -> None:
    base = _base_scada()
    results = score_study(base, profile=PROFILE, methods=[OracleMethod(base)], study=_study(n_replicates=1))
    assert "wall_time_s" in results.columns
    assert results["wall_time_s"].notna().all()
    assert (results["wall_time_s"] >= 0).all()


def test_results_record_the_window_boundaries_per_replicate() -> None:
    base = _base_scada()
    results = score_study(base, profile=PROFILE, methods=[OracleMethod(base)], study=_study(n_replicates=1))
    for col in ("treatment_start", "baseline_start", "activity_end"):
        assert col in results.columns
        assert results[col].notna().all()
    # baseline_start = treatment_start - min_pre_months (12), activity_end = treatment_start + campaign_months
    row3 = results[results["campaign_months"] == 3].iloc[0]
    assert row3["baseline_start"] == row3["treatment_start"] - pd.DateOffset(months=12)
    assert row3["activity_end"] == row3["treatment_start"] + pd.DateOffset(months=3)


def test_oracle_method_has_near_zero_signed_error() -> None:
    base = _base_scada()
    results = score_study(base, profile=PROFILE, methods=[OracleMethod(base)], study=_study())
    assert np.allclose(results["signed_error"].to_numpy(), 0.0, atol=1e-9)


def test_biased_method_signed_error_equals_offset() -> None:
    base = _base_scada()
    results = score_study(base, profile=PROFILE, methods=[BiasedMethod(base, offset=0.02)], study=_study())
    assert np.allclose(results["signed_error"].to_numpy(), 0.02, atol=1e-9)


def test_truth_is_recomputed_per_campaign_length_not_shared() -> None:
    base = _base_scada()
    results = score_study(base, profile=PROFILE, methods=[OracleMethod(base)], study=_study())
    # constant power -> uplift is ~equal across lengths, but truth must be present per row
    assert results["truth"].notna().all()
    assert (results["truth"] > 0).all()  # +5% Cp raises region-2 power


def test_profile_name_is_recorded() -> None:
    base = _base_scada()
    results = score_study(
        base, profile=PROFILE, methods=[OracleMethod(base)], study=_study(), profile_name="constant_cp_5pct"
    )
    assert set(results["profile"]) == {"constant_cp_5pct"}


def test_fairness_every_method_sees_identical_method_inputs() -> None:
    base = _base_scada()
    rec_a = RecordingMethod(name="a")
    rec_b = RecordingMethod(name="b")
    score_study(base, profile=PROFILE, methods=[rec_a, rec_b], study=_study())

    assert len(rec_a.seen) == len(rec_b.seen) > 0
    for mi_a, mi_b in zip(rec_a.seen, rec_b.seen, strict=True):
        assert mi_a.test_wtg == mi_b.test_wtg
        assert mi_a.upgrade_timing == mi_b.upgrade_timing
        pd.testing.assert_frame_equal(mi_a.scada_df, mi_b.scada_df)


def test_fairness_shorter_campaign_input_is_a_prefix_of_the_longer() -> None:
    base = _base_scada()
    rec = RecordingMethod()
    score_study(base, profile=PROFILE, methods=[rec], study=_study(n_replicates=1))
    # one replicate, lengths [3, 6] in order
    short_mi, long_mi = rec.seen[0], rec.seen[1]
    assert short_mi.scada_df.index.min() == long_mi.scada_df.index.min()  # same baseline start
    assert short_mi.scada_df.index.max() <= long_mi.scada_df.index.max()  # shorter activity
    assert len(short_mi.scada_df) < len(long_mi.scada_df)


def test_toggle_study_scores_without_error() -> None:
    base = _base_scada()
    results = score_study(base, profile=PROFILE, methods=[OracleMethod(base)], study=_study(mode="toggle"))
    assert len(results) == 4 * 2
    assert np.allclose(results["signed_error"].to_numpy(), 0.0, atol=1e-9)


def test_two_methods_scored_side_by_side() -> None:
    base = _base_scada()
    results = score_study(
        base, profile=PROFILE, methods=[OracleMethod(base), BiasedMethod(base, offset=0.01)], study=_study()
    )
    assert set(results["method"]) == {"oracle", "biased"}
    oracle_err = results[results["method"] == "oracle"]["signed_error"].to_numpy()
    biased_err = results[results["method"] == "biased"]["signed_error"].to_numpy()
    assert np.allclose(oracle_err, 0.0, atol=1e-9)
    assert np.allclose(biased_err, 0.01, atol=1e-9)


def test_on_method_complete_fires_per_method_with_only_that_methods_rows() -> None:
    base = _base_scada()
    seen: list[tuple[str, pd.DataFrame]] = []
    methods = [OracleMethod(base), BiasedMethod(base, offset=0.01)]
    results = score_study(
        base,
        profile=PROFILE,
        methods=methods,
        study=_study(),
        on_method_complete=lambda name, df: seen.append((name, df)),
    )
    # one call per method, in method order, each carrying only that method's rows
    assert [name for name, _ in seen] == ["oracle", "biased"]
    for name, df in seen:
        assert set(df["method"]) == {name}
        assert len(df) == 4 * 2  # 4 replicates x 2 campaign lengths
    # the callback never changes the returned frame: the slices concatenate back to it exactly
    rebuilt = pd.concat([df for _, df in seen], ignore_index=True)
    pd.testing.assert_frame_equal(rebuilt, results)


def test_on_method_complete_is_optional() -> None:
    base = _base_scada()
    with_cb = score_study(
        base, profile=PROFILE, methods=[OracleMethod(base)], study=_study(), on_method_complete=lambda _n, _d: None
    )
    without_cb = score_study(base, profile=PROFILE, methods=[OracleMethod(base)], study=_study())
    # wall_time_s is measured per run, so it differs between two runs; the callback must not change
    # anything else.
    drop = ["wall_time_s"]
    pd.testing.assert_frame_equal(with_cb.drop(columns=drop), without_cb.drop(columns=drop))


def test_conditional_rows_are_emitted_with_truth_and_near_zero_error() -> None:
    base = _base_scada()
    results = score_study(base, profile=PROFILE, methods=[ConditionalOracleMethod(base)], study=_study(n_replicates=1))
    assert "condition_bin" in results.columns
    overall = results[results["condition"] == "overall"]
    assert (overall["condition_bin"] == "overall").all()
    cond = results[results["condition"].isin(["ws", "ti", "power"])]
    assert set(cond["condition"]) == {"ws", "ti", "power"}
    # populated bins: a per-bin oracle must match per-bin truth (power too, via the rating-scaled edges)
    populated = cond[cond["truth"].notna() & cond["estimate"].notna()]
    assert len(populated) > 0
    assert (populated["condition"] == "power").any()
    assert np.allclose(populated["signed_error"].to_numpy(), 0.0, atol=1e-9)


def test_overall_only_method_emits_no_conditional_rows() -> None:
    base = _base_scada()
    results = score_study(base, profile=PROFILE, methods=[OracleMethod(base)], study=_study(n_replicates=1))
    assert set(results["condition"]) == {"overall"}


# --- uncertainty passthrough -------------------------------------------------------------------


def test_sigma_is_nan_for_a_method_that_reports_no_uncertainty() -> None:
    base = _base_scada()
    results = score_study(base, profile=PROFILE, methods=[OracleMethod(base)], study=_study(n_replicates=1))
    assert "sigma" in results.columns
    assert results["sigma"].isna().all()


def test_sigma_reaches_the_results_for_overall_and_per_bin_rows() -> None:
    base = _base_scada()
    results = score_study(
        base, profile=PROFILE, methods=[UncertainMethod(base, sigma=0.03)], study=_study(n_replicates=1)
    )
    overall = results[results["condition"] == "overall"]
    per_bin = results[results["condition"] != "overall"]
    assert (overall["sigma"] == 0.03).all()
    # the stub reports twice the overall sigma per bin, so the two channels are told apart
    assert (per_bin["sigma"] == 0.06).all()


def test_uncertainty_diagnostics_are_carried_to_every_row() -> None:
    base = _base_scada()
    results = score_study(
        base,
        profile=PROFILE,
        methods=[UncertainMethod(base, diagnostic_columns={"n_blocks": 7.0, "frac_resamples_finite": 0.5})],
        study=_study(n_replicates=1),
    )
    assert (results["n_blocks"] == 7.0).all()
    assert (results["frac_resamples_finite"] == 0.5).all()


def test_a_diagnostics_column_clashing_with_a_harness_column_raises() -> None:
    """Silently overwriting `estimate` or `truth` would corrupt the very thing the row reports."""
    base = _base_scada()
    with pytest.raises(ValueError, match="clash with columns the harness owns"):
        score_study(
            base,
            profile=PROFILE,
            methods=[UncertainMethod(base, diagnostic_columns={"estimate": 0.0})],
            study=_study(n_replicates=1),
        )


def test_diagnostics_missing_a_key_column_raises_a_clear_error() -> None:
    """A method returning a frame not keyed by (condition, condition_bin) gets a contract error,
    not an opaque KeyError from the row lookup.
    """
    diagnostics = pd.DataFrame([{"condition": "overall", "n_blocks": 7}])  # no condition_bin
    with pytest.raises(ValueError, match="must be keyed by"):
        _merge_diagnostics([], diagnostics)


# --- score_one is the unit score_study is built from -------------------------------------------


def test_score_one_reproduces_score_study_row_for_row() -> None:
    base = _base_scada()
    study = _study(n_replicates=2)
    method = UncertainMethod(base)
    from_study = score_study(base, profile=PROFILE, methods=[method], study=study, profile_name="p")

    replicates = build_replicates(base, profile=PROFILE, study=study)
    rows: list[dict] = []
    for replicate in replicates:
        for window in campaign_windows(
            replicate.treatment_start,
            min_pre_months=study.min_pre_months,
            campaign_months=study.campaign_months,
            campaign_weeks=study.campaign_weeks,
            data_start=base.index.min(),
            data_end=base.index.max(),
        ):
            mask = truth_mask(replicate, window)
            truth = replicate.true_uplift(mask=mask).overall
            rows.extend(score_one(method, replicate=replicate, window=window, truth=truth, mask=mask, profile_name="p"))
    from_one = pd.DataFrame(rows)

    drop = ["wall_time_s"]  # measured per call, so it differs between two runs
    pd.testing.assert_frame_equal(
        from_study.drop(columns=drop).reset_index(drop=True), from_one.drop(columns=drop).reset_index(drop=True)
    )


def test_duplicate_diagnostics_keys_raise_rather_than_silently_win() -> None:
    """Keeping the last row would drop a diagnostic without a word; the frame is keyed, so say so."""
    diagnostics = pd.DataFrame(
        [
            {"condition": "power", "condition_bin": "(0, 100]", "n_blocks": 7},
            {"condition": "power", "condition_bin": "(0, 100]", "n_blocks": 9},
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        _merge_diagnostics([], diagnostics)
