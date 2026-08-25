"""Orchestration: turn real SCADA + an upgrade recipe into a synthetic dataset.

A stable, no-upgrade window of real SCADA is taken as the baseline. The chosen test
turbine(s) have a known upgrade injected into their treated rows (post-changeover in
``prepost`` mode); references and untreated rows stay real. The untouched original is
retained alongside so the true uplift can always be derived by comparison.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from benchmarking.synthetic.cp_core import HOT_CP_MODEL, CpCore, CpParams
from benchmarking.synthetic.ground_truth import UpliftResult, true_net_uplift, true_uplift
from benchmarking.synthetic.sources.hill_of_towie import HOT_COLUMNS
from benchmarking.synthetic.upgrades import apply_upgrades

if TYPE_CHECKING:
    import pandas as pd

    from benchmarking.synthetic.schema import ColumnSchema

_GROUND_TRUTH_WS_BINS = list(np.arange(0.0, 26.0, 1.0))


@dataclass(frozen=True)
class ToggleSchedule:
    """A simple regular toggle: the upgrade alternates on/off over each ``period``.

    :param period: length of one full on/off cycle (half off, half on)
    :param start_on: whether the first block is treated (on); default off (pre-like)
    :param start: when toggling begins; rows before it are untreated baseline and it is
        the toggle origin. ``None`` (default) keeps the data's first timestamp as origin
        with no baseline.
    """

    period: pd.Timedelta
    start_on: bool = False
    start: pd.Timestamp | None = None


@dataclass
class SyntheticDataset:
    """A generated synthetic dataset plus its untouched ground-truth reference."""

    synthetic_df: pd.DataFrame
    original_df: pd.DataFrame
    run_metadata: dict = field(default_factory=dict)
    columns: ColumnSchema = HOT_COLUMNS

    def true_uplift(
        self,
        *,
        test_wtg: str | None = None,
        mask: np.ndarray | None = None,
        by: str | None = None,
        bins: list | None = None,
    ) -> UpliftResult:
        """Derive the true uplift by comparing synthetic to original.

        Defaults ``test_wtg`` to the first test turbine in the run metadata.
        """
        if test_wtg is None:
            test_wtg = self.run_metadata["test_wtgs"][0]
        return true_uplift(
            self.synthetic_df, self.original_df, test_wtg=test_wtg, mask=mask, by=by, bins=bins, columns=self.columns
        )

    def true_net_uplift(self, *, upstream: str, downstream: str, mask: np.ndarray | None = None) -> float:
        """Derive the production-weighted net uplift of a wake-steering pair (synthetic vs original)."""
        return true_net_uplift(
            self.synthetic_df,
            self.original_df,
            upstream=upstream,
            downstream=downstream,
            mask=mask,
            columns=self.columns,
        )

    def save(self, out_dir: str | Path) -> Path:
        """Write synthetic.parquet, original.parquet and run_metadata.json to ``out_dir``.

        The metadata file includes a full-record ground-truth summary (overall uplift and
        a per-original-wind-speed-bin breakdown) for each test turbine.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.synthetic_df.to_parquet(out_path / "synthetic.parquet")
        self.original_df.to_parquet(out_path / "original.parquet")

        ground_truth = {}
        for wtg in self.run_metadata.get("test_wtgs", []):
            result = self.true_uplift(test_wtg=wtg, by="ws", bins=_GROUND_TRUTH_WS_BINS)
            assert result.by_condition is not None  # always set when ``by`` is given  # noqa: S101
            by_condition = result.by_condition.copy()
            by_condition["condition_bin"] = by_condition["condition_bin"].astype(str)
            ground_truth[wtg] = {
                "overall": result.overall,
                "by_wind_speed": by_condition.to_dict(orient="records"),
            }

        metadata = {**self.run_metadata, "ground_truth": ground_truth}
        (out_path / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        return out_path


def _treated_mask(
    index: pd.DatetimeIndex,
    *,
    mode: Literal["prepost", "toggle"],
    upgrade_timing: pd.Timestamp | ToggleSchedule,
) -> np.ndarray:
    """Boolean mask over ``index`` selecting the rows where the upgrade is active."""
    if mode == "prepost":
        if isinstance(upgrade_timing, ToggleSchedule):
            msg = "prepost mode needs a changeover Timestamp, got a ToggleSchedule"
            raise TypeError(msg)
        return np.asarray(index >= upgrade_timing)
    if mode == "toggle":
        if not isinstance(upgrade_timing, ToggleSchedule):
            msg = f"toggle mode needs a ToggleSchedule, got {type(upgrade_timing).__name__}"
            raise TypeError(msg)
        schedule = upgrade_timing
        origin = schedule.start if schedule.start is not None else index.min()
        # ``period`` is a full on/off cycle, so each on/off block is half a period.
        block = (index - origin) // (schedule.period / 2)
        on_parity = 0 if schedule.start_on else 1
        treated = np.asarray((np.asarray(block) % 2) == on_parity)
        if schedule.start is not None:
            # Rows before toggling begins are untreated baseline (and their negative block
            # index must not be allowed to alias onto an "on" parity).
            treated &= np.asarray(index >= schedule.start)
        return treated
    msg = f"unknown mode {mode!r}"
    raise ValueError(msg)


def treated_mask(index: pd.DatetimeIndex, upgrade_timing: pd.Timestamp | ToggleSchedule) -> np.ndarray:
    """Boolean mask of the rows an upgrade treats, with the mode inferred from ``upgrade_timing``.

    A ``ToggleSchedule`` selects toggle mode; any other value (a changeover timestamp) selects
    prepost mode. Shared by the generator, methods and the scoring truth path so they all agree
    on which rows are treated.
    """
    mode: Literal["prepost", "toggle"] = "toggle" if isinstance(upgrade_timing, ToggleSchedule) else "prepost"
    return _treated_mask(index, mode=mode, upgrade_timing=upgrade_timing)


def generate_dataset(
    *,
    scada_df: pd.DataFrame,
    test_wtgs: list[str],
    upgrades: list,
    mode: Literal["prepost", "toggle"],
    upgrade_timing: pd.Timestamp | ToggleSchedule,
    cp_params: CpParams = HOT_CP_MODEL,
    rated_power_kw: float = 2300.0,
    columns: ColumnSchema = HOT_COLUMNS,
    seed: int = 0,
) -> SyntheticDataset:
    """Generate a synthetic dataset by injecting an upgrade into the test turbine(s).

    :param scada_df: source-native long real SCADA (all turbines), the no-upgrade baseline
    :param test_wtgs: turbine name(s) to upgrade
    :param upgrades: upgrade callables applied to each test turbine's treated rows
    :param mode: ``"prepost"`` (changeover date) or ``"toggle"``
    :param upgrade_timing: changeover timestamp (prepost) or toggle schedule
    :param cp_params: Cp surface parameters for the test turbines
    :param rated_power_kw: baseline rated power for the test turbines
    :param columns: the source-native column schema ``scada_df`` is keyed by
    :param seed: recorded in run metadata for provenance; generation is fully
        deterministic and does not otherwise consume it
    :return: the synthetic dataset, original reference and run metadata
    """
    original_df = scada_df.copy()
    synthetic_df = scada_df.copy()
    modified_columns = [columns.active_power, columns.gen_rpm, columns.wind_speed]
    if columns.nacelle_position is not None and columns.nacelle_position in scada_df.columns:
        # Wake steering also steers the nacelle; other upgrades leave it unchanged (a no-op write-back).
        modified_columns.append(columns.nacelle_position)

    treated = _treated_mask(synthetic_df.index, mode=mode, upgrade_timing=upgrade_timing)
    for wtg in test_wtgs:
        is_test = (synthetic_df[columns.turbine] == wtg).to_numpy()
        mask = is_test & treated
        if not mask.any():
            continue
        cp = CpCore(rated_power_kw=rated_power_kw, cp_params=cp_params)
        modified = apply_upgrades(synthetic_df.loc[mask], upgrades, cp=cp, columns=columns)
        for col in modified_columns:
            synthetic_df.loc[mask, col] = modified[col].to_numpy()

    run_metadata = {
        "test_wtgs": list(test_wtgs),
        "mode": mode,
        "upgrade_timing": str(upgrade_timing),
        "upgrades": [u.description for u in upgrades],
        "rated_power_kw": rated_power_kw,
        "cp_params": asdict(cp_params),
        "seed": seed,
    }
    return SyntheticDataset(
        synthetic_df=synthetic_df, original_df=original_df, run_metadata=run_metadata, columns=columns
    )
