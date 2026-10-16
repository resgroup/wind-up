"""The data screen: drop a candidate reference whose data cannot support the measurement.

Each candidate (a power reference or reserve of the plan) is paired with the test turbine alone. The
pair's usable records follow the record selection of method step 11. A candidate fails when its usable
hours before or after fall below a fraction of the median over every candidate, or its power
correlation with the test turbine falls more than a margin below the median. The first ``k`` passers in
rank order are the power references; the other passers stay reserves; failures become wake contributors.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benchmarking.baselines.power_model.screening import MIN_POOL_TO_SCREEN
from benchmarking.campaigns.plans import changed_works, treated_activity
from benchmarking.diagnostics.context import infer_timebase
from benchmarking.harness.operating_state import VALID_UPLIFT_COL, OperatingStateConfig, label_operating_states
from benchmarking.harness.record_selection import PAIR_CATEGORIES, USABLE, changed_times, pair_reasons

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from benchmarking.campaigns.declaration import CampaignSpec
    from benchmarking.harness.record_selection import ChangedTimes
    from benchmarking.synthetic import ColumnSchema
    from wind_up.analysis_period import AnalysisPlan
    from wind_up.layout import Layout

logger = logging.getLogger(__name__)

KEPT = "kept"
DROPPED = "dropped"
SPARE = "spare"
TABLE_COLUMNS = (
    "test_wtg",
    "candidate",
    "rank",
    "distance_d",
    "hours_before",
    "hours_after",
    "correlation",
    "median_hours_before",
    "median_hours_after",
    "median_correlation",
    "verdict",
    "reason",
)
HOURS_COLUMNS = ("test_wtg", "n_references", "hours_before", "hours_after", "references")
LOSS_COLUMNS = ("test_wtg", "candidate", "reason", "records")

_MIN_CORRELATION_RECORDS = 2
_SEPARATOR = "; "


@dataclass(frozen=True)
class DataScreenSettings:
    """How the data screen judges a candidate.

    :param hours_fraction: the fewest usable hours a candidate may have in each period, as a fraction
        of the median over every candidate
    :param correlation_margin: how far a candidate's power correlation may fall below the median
    """

    hours_fraction: float = 0.5
    correlation_margin: float = 0.2


DEFAULT_DATA_SCREEN = DataScreenSettings()


@dataclass(frozen=True)
class DataScreen:
    """One test turbine's data screen.

    :param plan: the screened plan
    :param table: one row per candidate, in rank order, with :data:`TABLE_COLUMNS`
    :param hours: the joint usable hours of the nearest 1..N screened references, with :data:`HOURS_COLUMNS`
    :param losses: each candidate's span records per loss reason, with :data:`LOSS_COLUMNS`
    :param changed: each changed turbine and the times it counts as changed
    :param k: power references sought
    :param settings: how the candidates were judged
    """

    plan: AnalysisPlan
    table: pd.DataFrame
    hours: pd.DataFrame
    losses: pd.DataFrame
    changed: ChangedTimes
    k: int
    settings: DataScreenSettings = DEFAULT_DATA_SCREEN


def screen_plan(
    plan: AnalysisPlan,
    frame: pd.DataFrame,
    *,
    spec: CampaignSpec,
    columns: ColumnSchema,
    settings: DataScreenSettings = DEFAULT_DATA_SCREEN,
) -> DataScreen:
    """Screen ``plan``'s power references and reserves on their data and return the screened plan.

    :param plan: the test turbine's plan
    :param frame: long SCADA cut to the plan's span
    :param spec: the campaign's public facts
    :param columns: the schema ``frame`` is keyed by
    :param settings: how a candidate is judged
    """
    test = plan.turbine
    candidates = [*plan.power_references, *plan.reserves]
    index = pd.DatetimeIndex(pd.unique(frame.index)).sort_values()
    timebase = infer_timebase(index)
    scada = frame[frame[spec.turbine_col].astype(str).isin([test, *candidates])]
    if VALID_UPLIFT_COL not in scada.columns:
        scada = label_operating_states(
            scada,
            columns=columns,
            config=OperatingStateConfig(),
            timebase=timebase,
            rated_power_kw=spec.rated_power_kw,
        )
    present = sorted({str(t) for t in scada[spec.turbine_col].unique()} | {test, *candidates})
    usable = pd.DataFrame({t: spec.usable_mask(t, index) for t in present}, index=index, dtype=bool)
    changed = changed_times(
        test_wtg=test,
        works=changed_works(spec),
        changeover=plan.works,
        start=plan.start,
        end=plan.end,
        test_wake_changed=not spec.wake_unchanged,
    )
    reasons = pair_reasons(
        scada,
        subject=test,
        candidates=candidates,
        changed=changed,
        turbine_col=spec.turbine_col,
        usable=usable,
        index=index,
    )
    after = treated_activity(index, timing=spec.timing_for(test), window=plan.after)
    power = _power(scada, turbine_col=spec.turbine_col, power_col=columns.active_power, index=index)
    hours_per_record = timebase / pd.Timedelta(hours=1)

    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        used = np.asarray(reasons[candidate] == USABLE)
        rows.append(
            {
                "test_wtg": test,
                "candidate": candidate,
                "rank": rank,
                "distance_d": plan.distances_d.get(candidate, np.nan),
                "hours_before": float(np.count_nonzero(used & ~after)) * hours_per_record,
                "hours_after": float(np.count_nonzero(used & after)) * hours_per_record,
                "correlation": _correlation(power[test], power[candidate], used=used),
            }
        )
    table = pd.DataFrame(rows)
    table["median_hours_before"] = float(np.median(table["hours_before"]))
    table["median_hours_after"] = float(np.median(table["hours_after"]))
    table["median_correlation"] = (
        float(np.nanmedian(table["correlation"])) if table["correlation"].notna().any() else np.nan
    )
    table["reason"] = [_failure(row, settings=settings) for row in table.itertuples()]
    passers = [c for c, r in zip(table["candidate"], table["reason"], strict=True) if not r]
    kept, spares = passers[: plan.k], passers[plan.k :]
    table["verdict"] = [KEPT if c in kept else SPARE if c in spares else DROPPED for c in table["candidate"]]
    table = table[list(TABLE_COLUMNS)]

    failed = set(candidates) - set(passers)
    _log(plan, table=table, kept=kept, failed=failed)
    screened = dataclasses.replace(plan, power_references=tuple(kept), reserves=tuple(spares))
    return DataScreen(
        plan=screened,
        table=table,
        hours=_joint_hours(reasons, order=[*kept, *spares], after=after, test=test, hours_per_record=hours_per_record),
        losses=_losses(reasons, test=test),
        changed=changed,
        k=plan.k,
        settings=settings,
    )


def write_data_screen(
    screens: dict[str, DataScreen], *, layout: Layout, scada_df: pd.DataFrame, turbine_col: str, out_dir: Path
) -> None:
    """Write ``reference_data.csv``, ``reference_hours.csv`` and each test turbine's plots under ``out_dir``.

    :param screens: each test turbine's data screen
    :param layout: the farm layout, for the maps
    :param scada_df: the SCADA the screens ran on, for the wind rose of the waking direction
    :param turbine_col: the turbine-identifier column of ``scada_df``
    :param out_dir: where to write
    """
    from benchmarking.campaigns.data_screen_plots import write_data_screen_plots  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    reference_data_table(screens).to_csv(out_dir / "reference_data.csv", index=False)
    reference_hours_table(screens).to_csv(out_dir / "reference_hours.csv", index=False)
    for screen in screens.values():
        write_data_screen_plots(screen, layout=layout, scada_df=scada_df, turbine_col=turbine_col, out_dir=out_dir)


def reference_data_table(screens: dict[str, DataScreen]) -> pd.DataFrame:
    """Every test turbine's candidates, test turbines in name order."""
    return _stack([screens[t].table for t in sorted(screens)], columns=TABLE_COLUMNS)


def reference_hours_table(screens: dict[str, DataScreen]) -> pd.DataFrame:
    """Every test turbine's joint usable hours against the number of references."""
    return _stack([screens[t].hours for t in sorted(screens)], columns=HOURS_COLUMNS)


def _stack(frames: list[pd.DataFrame], *, columns: tuple[str, ...]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns))


def _power(scada: pd.DataFrame, *, turbine_col: str, power_col: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Timestamps x turbines active power."""
    wide = scada.pivot_table(index=scada.index, columns=turbine_col, values=power_col, aggfunc="first", dropna=False)
    wide.columns = wide.columns.astype(str)
    return wide.reindex(index=index)


def _correlation(test: pd.Series, candidate: pd.Series, *, used: npt.NDArray[np.bool_]) -> float:
    """Pearson correlation over the used records, NaN with too few or no variation."""
    x = test.to_numpy(dtype=float)[used]
    y = candidate.to_numpy(dtype=float)[used]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < _MIN_CORRELATION_RECORDS or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _failure(row: object, *, settings: DataScreenSettings) -> str:
    """Why a candidate fails, or empty when it passes."""
    reasons = []
    for period in ("before", "after"):
        hours = getattr(row, f"hours_{period}")
        median = getattr(row, f"median_hours_{period}")
        if hours < settings.hours_fraction * median:
            reasons.append(f"hours {period} {hours:.0f} < {settings.hours_fraction:g} x median {median:.0f}")
    correlation = getattr(row, "correlation")  # noqa: B009
    median = getattr(row, "median_correlation")  # noqa: B009
    if np.isnan(correlation):
        reasons.append("no correlation")
    elif correlation < median - settings.correlation_margin:
        reasons.append(f"correlation {correlation:.3f} < median {median:.3f} - {settings.correlation_margin:g}")
    return _SEPARATOR.join(reasons)


def _joint_hours(
    reasons: pd.DataFrame, *, order: list[str], after: npt.NDArray[np.bool_], test: str, hours_per_record: float
) -> pd.DataFrame:
    """Joint usable hours before and after for the nearest 1..N of ``order``."""
    rows = []
    joint = np.ones(len(reasons), dtype=bool)
    for n, candidate in enumerate(order, start=1):
        joint &= np.asarray(reasons[candidate] == USABLE)
        rows.append(
            {
                "test_wtg": test,
                "n_references": n,
                "hours_before": float(np.count_nonzero(joint & ~after)) * hours_per_record,
                "hours_after": float(np.count_nonzero(joint & after)) * hours_per_record,
                "references": _SEPARATOR.join(order[:n]),
            }
        )
    return pd.DataFrame(rows, columns=list(HOURS_COLUMNS))


def _losses(reasons: pd.DataFrame, *, test: str) -> pd.DataFrame:
    """Each candidate's records per loss reason, every reason listed."""
    rows = [
        {
            "test_wtg": test,
            "candidate": candidate,
            "reason": reason,
            "records": int((reasons[candidate] == reason).sum()),
        }
        for candidate in reasons.columns
        for reason in PAIR_CATEGORIES
    ]
    return pd.DataFrame(rows, columns=list(LOSS_COLUMNS))


def _log(plan: AnalysisPlan, *, table: pd.DataFrame, kept: list[str], failed: set[str]) -> None:
    """Log the screen's table and warn when it breaks the pool rule."""
    logger.info(
        "%s: data screen of the candidate references:\n%s",
        plan.turbine,
        table.drop(columns=["test_wtg"]).to_string(index=False),
    )
    if failed:
        logger.info("%s: the data screen dropped %s", plan.turbine, sorted(failed))
    if len(kept) < plan.k:
        logger.warning(
            "%s: %d of %d power references pass the data screen, so the pool rule is broken",
            plan.turbine,
            len(kept),
            plan.k,
        )
    if len(kept) < MIN_POOL_TO_SCREEN:
        logger.warning(
            "%s: only %d power references pass the data screen; the reference screen cannot judge them",
            plan.turbine,
            len(kept),
        )
