"""The level probe's analysis: where a placebo reading's level lives in a power-model row dump.

A power-model placebo reading (a turbine that did not change, truth 0) carries a level error. The
double machine learning framing splits the naive counterfactual's bias into covariate shift times
model error, ``E[P(post | X) (g - ĝ)(X)]``, and drift in ``Y | X`` between the periods that ``X``
does not carry. These functions decide which term a reading's level belongs to and where in the
data it sits, from the per-row frame ``PowerModelMethod(row_dump_dir=...)`` writes. Nothing here
changes an estimate.

* :func:`level_by` splits the headline ``Σ(actual - counterfactual) / Σ counterfactual`` over the
  upgraded rows into per-group contributions that sum to it exactly. The ``label_*`` functions
  supply the groups: month, direction sector, wind band, which power references were running,
  how many upwind turbines were not waking, and propensity decile.
* :func:`propensity` is an out-of-fold ``P(upgraded | X)``; :func:`ess_fraction` reads how well the
  baseline covers the campaign off it.
* :func:`aipw_level` is the augmented inverse-propensity correction's estimate of the covariate
  shift bias. For a placebo the observed level is pure bias, so the two agreeing says the shift
  term explains it.
* :func:`channel_difference_by` splits the difference between two dumps of the same rows (the two
  ``exclusion_channels``) the same way.

A label function returns ``None``, with a logged warning, when its inputs are absent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from benchmarking.baselines.power_model.features import QUALIFIER
from benchmarking.baselines.power_model.fitting import (
    make_outcome_model,
    make_propensity_model,
    model_safe_features,
    purged_time_block_folds,
)
from benchmarking.baselines.power_model.method import _clip_predictions
from wind_up.circular_math import circ_diff
from wind_up.geodesy import distance_and_bearing
from wind_up.layout import iec_disturbed_sector_deg

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("timestamp", "actual_kw", "counterfactual_kw", "selected", "baseline", "upgraded")
# Columns of a dump that are not model features.
NON_FEATURE_COLUMNS = (*REQUIRED_COLUMNS, "reference_mean_ws")
MISSING = "(missing)"
OTHER = "other"
UNKNOWN = "unknown"
HELD_BACK = "held back"
BOTH = "both"
NEAR_CERTAIN = 0.95
NEAR_CERTAIN_LABEL = "0.95-1.00"
TRIM = 0.99
SECTOR_DEG = 30.0
WIND_BAND_MS = 2.0
ERA5_DIRECTION = "wind_direction_100m"
MIN_UPGRADED_ROWS = 20
N_FOLDS = 5
EMBARGO = pd.Timedelta("1D")
_CLOSURE_TOL_PP = 1e-9
_ROWS_FILE = "rows.parquet"
_META_FILE = "rows.json"


@dataclass(frozen=True)
class RowDump:
    """One estimate's dumped rows (the parquet frame) and its sidecar (the json)."""

    rows: pd.DataFrame
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        """Refuse a dump without the columns every analysis reads."""
        missing = [c for c in REQUIRED_COLUMNS if c not in self.rows.columns]
        if missing:
            msg = f"the row dump is missing required column(s) {missing}; it carries {list(self.rows.columns)}"
            raise ValueError(msg)

    @classmethod
    def load(cls, directory: str | Path) -> RowDump:
        """Read the dump ``PowerModelMethod`` wrote to ``directory``."""
        directory = Path(directory)
        rows = pd.read_parquet(directory / _ROWS_FILE)
        return cls(rows=rows, meta=json.loads((directory / _META_FILE).read_text(encoding="utf-8")))

    def save(self, directory: str | Path) -> None:
        """Write the dump as ``PowerModelMethod`` does."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.rows.to_parquet(directory / _ROWS_FILE, engine="pyarrow", index=False)
        (directory / _META_FILE).write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @property
    def upgraded(self) -> pd.Series:
        """The rows the headline sums over: selected and upgraded."""
        return self.rows["selected"].astype(bool) & self.rows["upgraded"].astype(bool)

    @property
    def baseline(self) -> pd.Series:
        """The rows the counterfactual is fitted on: selected and baseline."""
        return self.rows["selected"].astype(bool) & self.rows["baseline"].astype(bool)

    @property
    def features(self) -> pd.DataFrame:
        """The feature matrix under its original names."""
        return self.rows.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in self.rows.columns])

    @property
    def power_references(self) -> list[str]:
        """The references whose power is a feature, sorted."""
        return sorted(set(self.meta["references"]) - set(self.meta["power_free"]))

    def power_col(self, turbine: str) -> str:
        """Return the feature carrying ``turbine``'s active power."""
        return f"{self.meta['active_power_col']}{QUALIFIER}{turbine}"

    def waking_col(self, turbine: str) -> str:
        """Return the feature carrying ``turbine``'s waking boolean."""
        return f"waking_{self.meta['active_power_col']}{QUALIFIER}{turbine}"


# --- the decomposition ---------------------------------------------------------------------------


def level_by(dump: RowDump, labels: pd.Series) -> pd.DataFrame:
    """Split the headline over the groups ``labels`` names; a NaN label is the ``(missing)`` group.

    ``contribution_pp`` sums to ``100 * uplift`` exactly, and this raises when it does not
    reproduce the sidecar's uplift, since that means the grouping lost or duplicated rows.
    """
    up = dump.upgraded
    frame = pd.DataFrame(
        {
            "group": _labels_or_missing(labels[up]),
            "actual": dump.rows.loc[up, "actual_kw"].to_numpy(dtype=float),
            "counterfactual": dump.rows.loc[up, "counterfactual_kw"].to_numpy(dtype=float),
        }
    )
    total = frame["counterfactual"].sum()
    table = frame.groupby("group", sort=True).agg(
        n_rows=("actual", "size"), sum_actual_kw=("actual", "sum"), sum_counterfactual_kw=("counterfactual", "sum")
    )
    table["row_share"] = table["n_rows"] / len(frame)
    table["contribution_pp"] = 100 * (table["sum_actual_kw"] - table["sum_counterfactual_kw"]) / total
    table["local_bias_pct"] = 100 * (table["sum_actual_kw"] / table["sum_counterfactual_kw"] - 1)
    headline = 100 * float(dump.meta["uplift"])
    if abs(table["contribution_pp"].sum() - headline) > _CLOSURE_TOL_PP:
        msg = (
            f"the groups sum to {table['contribution_pp'].sum():.12f} pp but the sidecar's uplift is "
            f"{headline:.12f} pp: the grouping lost or duplicated rows"
        )
        raise ValueError(msg)
    columns = ["n_rows", "row_share", "sum_actual_kw", "sum_counterfactual_kw", "contribution_pp", "local_bias_pct"]
    return table.reset_index()[["group", *columns]]


def top_group(table: pd.DataFrame, column: str = "contribution_pp") -> str:
    """Return the group with the largest ``|column|``."""
    return str(table.loc[table[column].abs().idxmax(), "group"])


def _labels_or_missing(labels: pd.Series) -> np.ndarray:
    return labels.astype(object).where(labels.notna(), MISSING).astype(str).to_numpy()


# --- groupings -------------------------------------------------------------------------------------


def label_month(dump: RowDump) -> pd.Series:
    """Calendar month of each row, ``YYYY-MM``."""
    return pd.to_datetime(dump.rows["timestamp"]).dt.strftime("%Y-%m")


def row_direction(dump: RowDump) -> pd.Series | None:
    """Each row's wind direction: the references' circular mean, else reanalysis, else ``None``.

    The circular mean is over the unit vectors of every reference's northed-direction pair, so a
    reference that is missing on a row drops out of that row alone.
    """
    direction = dump.meta.get("northed_direction_col")
    sin_cols = [f"{direction}_sin{QUALIFIER}{r}" for r in dump.meta["references"]] if direction else []
    sin_cols = [c for c in sin_cols if c in dump.rows.columns]
    if sin_cols:
        sin = dump.rows[sin_cols].to_numpy(dtype=float)
        cos = dump.rows[[c.replace("_sin" + QUALIFIER, "_cos" + QUALIFIER) for c in sin_cols]].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"), _quiet_nanmean():
            deg = np.rad2deg(np.arctan2(np.nanmean(sin, axis=1), np.nanmean(cos, axis=1))) % 360
        return pd.Series(deg, index=dump.rows.index)
    if ERA5_DIRECTION in dump.rows.columns:
        return dump.rows[ERA5_DIRECTION].astype(float) % 360
    return None


def label_direction_sector(dump: RowDump) -> pd.Series | None:
    """30° sectors centred on north (``000`` spans 345° to 15°), labelled by their centre."""
    deg = row_direction(dump)
    if deg is None:
        logger.warning("%s: no reference or reanalysis direction, so the direction grouping is skipped", _who(dump))
        return None
    centre = (np.floor(((deg + SECTOR_DEG / 2) % 360) / SECTOR_DEG) * SECTOR_DEG).astype("Int64")
    return centre.map(lambda c: f"{int(c):03d}", na_action="ignore")


def label_wind_band(dump: RowDump) -> pd.Series | None:
    """2 m/s bands of the references' mean wind speed, ``04-06`` style."""
    ws = dump.rows.get("reference_mean_ws")
    if ws is None or ws.isna().all():
        logger.warning("%s: no reference wind speed, so the wind-band grouping is skipped", _who(dump))
        return None
    lo = (np.floor(ws / WIND_BAND_MS) * WIND_BAND_MS).astype("Int64")
    return lo.map(lambda v: f"{int(v):02d}-{int(v + WIND_BAND_MS):02d}", na_action="ignore")


def label_operating_pattern(dump: RowDump, *, min_rows: int = 50) -> pd.Series:
    """Which power references have power on each row, as a bitmask over them sorted.

    A pattern on fewer than ``min_rows`` upgraded rows collapses into ``other``.
    """
    refs = dump.power_references
    bits = np.column_stack(
        [
            np.isfinite(dump.rows[dump.power_col(r)].to_numpy(dtype=float))
            if dump.power_col(r) in dump.rows.columns
            else np.zeros(len(dump.rows), dtype=bool)
            for r in refs
        ]
    )
    labels = pd.Series(["".join("1" if b else "0" for b in row) for row in bits], index=dump.rows.index)
    counts = labels[dump.upgraded].value_counts()
    rare = counts[counts < min_rows].index
    if len(rare):
        logger.info(
            "%s: %d operating pattern(s) on %d upgraded row(s) collapse into %r",
            _who(dump),
            len(rare),
            int(counts[rare].sum()),
            OTHER,
        )
    return labels.where(~labels.isin(rare), OTHER)


def label_upwind_offline(dump: RowDump, *, rotor_diameter_m: float) -> pd.Series | None:
    """How many turbines in the test turbine's disturbed sector are not waking, per row.

    A neighbour is in the sector when the row's direction lies within half the IEC 61400-12-1
    disturbed-sector width of its bearing from the test turbine. Its waking state is its
    ``waking`` boolean when it has one (a power-free reference, a wake-only turbine, or a held-back
    reference under the ``booleans`` channel), else whether its power reaches the waking threshold.
    A row with any sector neighbour of unknown state is ``unknown``.
    """
    coords = dump.meta.get("coords") or {}
    test = dump.meta["test_wtg"]
    if not coords or test not in coords:
        logger.warning("%s: no turbine coordinates, so the upwind-offline grouping is skipped", _who(dump))
        return None
    deg = row_direction(dump)
    if deg is None:
        logger.warning("%s: no direction, so the upwind-offline grouping is skipped", _who(dump))
        return None
    deg_arr = deg.to_numpy(dtype=float)
    offline = np.zeros(len(dump.rows))
    unknown = np.zeros(len(dump.rows), dtype=bool)
    for turbine in sorted({*dump.meta["references"], *dump.meta["wake_only"]}):
        if turbine not in coords:
            continue
        waking = _waking_state(dump, turbine)
        if waking is None:
            continue
        distance, bearing = distance_and_bearing(tuple(coords[test]), tuple(coords[turbine]))
        half_width = float(iec_disturbed_sector_deg(distance / rotor_diameter_m)) / 2
        in_sector = np.abs(circ_diff(deg_arr, bearing)) <= half_width
        unknown |= in_sector & np.isnan(waking)
        offline += in_sector & (waking == 0)
    labels = pd.Series(offline.astype(int).astype(str), index=dump.rows.index, dtype=object)
    labels[unknown] = UNKNOWN
    labels[np.isnan(deg_arr)] = np.nan
    return labels


def _waking_state(dump: RowDump, turbine: str) -> np.ndarray | None:
    """1 waking, 0 not, NaN unknown; ``None`` when the dump carries nothing about ``turbine``."""
    if dump.waking_col(turbine) in dump.rows.columns:
        return dump.rows[dump.waking_col(turbine)].to_numpy(dtype=float)
    if dump.power_col(turbine) in dump.rows.columns:
        power = dump.rows[dump.power_col(turbine)].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            return np.where(np.isnan(power), np.nan, (power >= dump.meta["waking_threshold_kw"]).astype(float))
    return None


def label_propensity_decile(dump: RowDump, m_hat: pd.Series) -> pd.Series:
    """Deciles of ``m_hat`` over the upgraded rows, with ``0.95-1.00`` always its own top group."""
    up = dump.upgraded
    labels = pd.Series(np.nan, index=dump.rows.index, dtype=object)
    near = up & (m_hat > NEAR_CERTAIN)
    labels[near] = NEAR_CERTAIN_LABEL
    rest = up & (m_hat <= NEAR_CERTAIN)
    if rest.any():
        bins = pd.qcut(m_hat[rest], q=10, duplicates="drop")
        labels[rest] = bins.map(lambda b: f"{max(b.left, 0.0):.2f}-{b.right:.2f}").astype(str)
    return labels


# --- propensity and the AIPW term ----------------------------------------------------------------


def propensity(
    dump: RowDump,
    *,
    n_folds: int = N_FOLDS,
    embargo: pd.Timedelta = EMBARGO,
    model_params: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Out-of-fold ``P(upgraded | X)`` over the selected rows, NaN elsewhere.

    Fitted over ``n_folds`` folds, each holding out one contiguous time block of the baseline rows
    and one of the upgraded rows, its training set purged of ``embargo`` either side of both. The
    blocks are cut per period because the period is time: a plain time block at either end would
    hold one class out almost whole.
    """
    _check_sizes(dump, n_folds=n_folds)
    rows = dump.baseline | dump.upgraded
    x = model_safe_features(dump.features[rows])
    label = dump.upgraded[rows].to_numpy(dtype=int)
    ts = pd.DatetimeIndex(dump.rows.loc[rows, "timestamp"])
    out = np.full(len(x), np.nan)
    for train, test in purged_time_block_folds(ts, n_folds=n_folds, embargo=embargo, strata=label):
        classes = np.unique(label[train])
        if len(classes) == 1:  # a fold whose training set holds one period predicts it for certain
            out[test] = float(classes[0])
            continue
        model = make_propensity_model(**(model_params or {}))
        model.fit(x.iloc[train], label[train])
        out[test] = model.predict_proba(x.iloc[test])[:, 1]
    m_hat = pd.Series(np.nan, index=dump.rows.index)
    m_hat[rows] = out
    return m_hat


def ess_fraction(dump: RowDump, m_hat: pd.Series, *, trim: float = TRIM) -> float:
    """Effective sample size of the odds weights ``m/(1-m)`` on baseline rows, over their count."""
    m = m_hat[dump.baseline].to_numpy(dtype=float)
    w = _odds(m[m <= trim])
    return float(w.sum() ** 2 / (w**2).sum() / len(m)) if len(w) else float("nan")


def propensity_summary(dump: RowDump, m_hat: pd.Series) -> dict[str, float]:
    """Return the share of upgraded rows the baseline barely covers, the share of level they carry, and ESS."""
    up = dump.upgraded
    near = up & (m_hat > NEAR_CERTAIN)
    headline = 100 * float(dump.meta["uplift"])
    contribution = (
        100
        * (dump.rows.loc[near, "actual_kw"].sum() - dump.rows.loc[near, "counterfactual_kw"].sum())
        / float(dump.rows.loc[up, "counterfactual_kw"].sum())
    )
    return {
        "share_rows_m_gt_0.95": float(near.sum() / up.sum()),
        "share_level_m_gt_0.95": float(contribution / headline) if headline else float("nan"),
        "ess_fraction": ess_fraction(dump, m_hat),
    }


def baseline_predictions(
    dump: RowDump,
    *,
    n_folds: int = N_FOLDS,
    embargo: pd.Timedelta = EMBARGO,
    model_params: Mapping[str, Any] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return the outcome model's out-of-fold and in-sample predictions on the baseline rows.

    The out-of-fold ones come from ``n_folds`` purged contiguous time blocks of the baseline rows;
    the in-sample ones from one fit on all of them. Both are clipped as the headline's are; NaN off
    the baseline.
    """
    base = dump.baseline
    x = model_safe_features(dump.features[base])
    y = dump.rows.loc[base, "actual_kw"].to_numpy(dtype=float)
    ts = pd.DatetimeIndex(dump.rows.loc[base, "timestamp"])
    rated = float(dump.meta["baseline_rated_power_kw"])
    oof = np.full(len(x), np.nan)
    for train, test in purged_time_block_folds(ts, n_folds=n_folds, embargo=embargo):
        model = make_outcome_model(**(model_params or {}))
        model.fit(x.iloc[train], y[train])
        oof[test] = _clip_predictions(model.predict(x.iloc[test]), y_train=y[train], rated_power_kw=rated)
    model = make_outcome_model(**(model_params or {}))
    model.fit(x, y)
    in_sample = _clip_predictions(model.predict(x), y_train=y, rated_power_kw=rated)
    return _on_rows(dump, base, oof), _on_rows(dump, base, in_sample)


def aipw_level(
    dump: RowDump, m_hat: pd.Series, *, g_oof: pd.Series, g_in_sample: pd.Series, trim: float = TRIM
) -> dict[str, float]:
    """Estimate the level that covariate shift x model error causes, by augmented inverse propensity.

    The baseline residuals, reweighted by the odds ``w = m/(1-m)`` to look like the campaign, say
    how far the counterfactual misses on campaign-like rows::

        correction_kw   = n_upgraded * Σ_baseline w (actual - ĝ) / Σ_baseline w
        predicted_level = 100 * correction_kw / Σ_upgraded counterfactual

    The weights are self-normalised: ``Σ_baseline w`` already estimates ``n_upgraded``, so a further
    row-count ratio would count the campaign twice. A positive correction is a counterfactual that
    reads low, so it is the level a placebo shows: the two are compared directly. Baseline rows
    with ``m > trim`` are left out. The in-sample control repeats this with in-sample residuals,
    which an overfitted model drives to zero whatever the bias.
    """
    base = dump.baseline
    m = m_hat[base].to_numpy(dtype=float)
    keep = m <= trim
    actual = dump.rows.loc[base, "actual_kw"].to_numpy(dtype=float)
    w = _odds(m[keep])
    n_up = int(dump.upgraded.sum())
    sum_counter = float(dump.rows.loc[dump.upgraded, "counterfactual_kw"].sum())

    def level(g: pd.Series) -> float:
        residual = (actual - g[base].to_numpy(dtype=float))[keep]
        return float(100 * n_up * (w * residual).sum() / w.sum() / sum_counter)

    return {
        "observed_level_pp": 100 * float(dump.meta["uplift"]),
        "predicted_level_pp": level(g_oof),
        "in_sample_control_pp": level(g_in_sample),
        "trimmed_rows": int((~keep).sum()),
        "ess_fraction": ess_fraction(dump, m_hat, trim=trim),
    }


def _odds(m: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return m / (1 - m)


def _on_rows(dump: RowDump, mask: pd.Series, values: np.ndarray) -> pd.Series:
    out = pd.Series(np.nan, index=dump.rows.index)
    out[mask] = values
    return out


def _check_sizes(dump: RowDump, *, n_folds: int) -> None:
    n_up, n_base = int(dump.upgraded.sum()), int(dump.baseline.sum())
    if n_up < MIN_UPGRADED_ROWS:
        msg = f"{_who(dump)}: {n_up} upgraded selected rows, fewer than {MIN_UPGRADED_ROWS}; the probe is for campaigns"
        raise ValueError(msg)
    if n_base < 5 * n_folds:
        msg = f"{_who(dump)}: {n_base} baseline selected rows, fewer than {5 * n_folds}; the probe is for campaigns"
        raise ValueError(msg)


# --- the channel difference ----------------------------------------------------------------------


def channel_difference_by(dump_a: RowDump, dump_b: RowDump, labels: pd.Series) -> pd.DataFrame:
    """Split ``uplift_a - uplift_b`` over the groups ``labels`` names, for two dumps of the same rows.

    With the actuals shared, ``uplift_a - uplift_b = Σa (Σc_b - Σc_a) / (Σc_a Σc_b)``, which is
    linear in the per-row counterfactual difference, so ``difference_pp`` sums to it exactly.
    """
    up = _shared_upgraded(dump_a, dump_b)
    actual = dump_a.rows.loc[up, "actual_kw"].to_numpy(dtype=float)
    c_a = dump_a.rows.loc[up, "counterfactual_kw"].to_numpy(dtype=float)
    c_b = dump_b.rows.loc[up, "counterfactual_kw"].to_numpy(dtype=float)
    scale = 100 * actual.sum() / (c_a.sum() * c_b.sum())
    frame = pd.DataFrame({"group": _labels_or_missing(labels[up]), "delta": c_a - c_b})
    table = frame.groupby("group", sort=True).agg(
        n_rows=("delta", "size"), sum_counterfactual_a_minus_b_kw=("delta", "sum")
    )
    table["row_share"] = table["n_rows"] / len(frame)
    table["difference_pp"] = -scale * table["sum_counterfactual_a_minus_b_kw"]
    return table.reset_index()[["group", "n_rows", "row_share", "sum_counterfactual_a_minus_b_kw", "difference_pp"]]


def label_held_back(dump_a: RowDump, dump_b: RowDump) -> pd.Series:
    """``held back`` where one dump knows a reference's operating state and the other knows nothing of it.

    Under the ``booleans`` channel a held-back reference carries its booleans over its whole record,
    so they differ from the ``nan`` channel on every row; they carry information the other dump
    lacks only where that dump has no power for the reference either. Those rows are labelled.
    """
    _shared_upgraded(dump_a, dump_b)
    prefixes = ("waking_", "normal_operation_")
    refs = set(dump_a.meta["references"]) | set(dump_b.meta["references"])
    columns = {
        c
        for d in (dump_a, dump_b)
        for c in d.rows.columns
        if c.startswith(prefixes) and c.rsplit(QUALIFIER, 1)[-1] in refs
    }
    differs = np.zeros(len(dump_a.rows), dtype=bool)
    for col in sorted(columns):
        ref = col.rsplit(QUALIFIER, 1)[-1]
        known_a, known_b = _known(dump_a, col), _known(dump_b, col)
        blind_a = ~known_a & ~_known(dump_a, dump_a.power_col(ref))
        blind_b = ~known_b & ~_known(dump_b, dump_b.power_col(ref))
        differs |= (known_a & blind_b) | (known_b & blind_a)
    return pd.Series(np.where(differs, HELD_BACK, BOTH), index=dump_a.rows.index, dtype=object)


def _known(dump: RowDump, col: str) -> np.ndarray:
    if col not in dump.rows.columns:
        return np.zeros(len(dump.rows), dtype=bool)
    return np.isfinite(dump.rows[col].to_numpy(dtype=float))


def _shared_upgraded(dump_a: RowDump, dump_b: RowDump) -> pd.Series:
    up_a, up_b = dump_a.upgraded, dump_b.upgraded
    same = (
        len(dump_a.rows) == len(dump_b.rows)
        and (up_a.to_numpy() == up_b.to_numpy()).all()
        and (dump_a.rows["timestamp"].to_numpy() == dump_b.rows["timestamp"].to_numpy()).all()
    )
    if not same:
        msg = (
            "the two dumps do not share their upgraded selected rows; the channel oracle rests on their being identical"
        )
        raise ValueError(msg)
    return up_a


def _who(dump: RowDump) -> str:
    return str(dump.meta.get("test_wtg", "?"))


class _quiet_nanmean:  # noqa: N801
    """Silence numpy's all-NaN-slice warning, which a row with no reference direction raises."""

    def __enter__(self) -> None:
        import warnings  # noqa: PLC0415

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc: object) -> None:
        self._ctx.__exit__(*exc)
