"""Coarsened exact matching (CEM) for the power-model bias-cancellation correction (Issue 8).

The bias-cancellation correction trains/predicts in two symmetric directions and relies on a *common
per-bin multiplicative shrinkage* cancelling between them — which only holds if the baseline and
upgraded periods share a covariate distribution within each reporting bin. This module makes that
matching explicit and model-free: bin the matching variables into cells, keep only cells present on
*both* sides (the common-support guard — the exact failure that sank the R-learner in prepost, F1),
and seeded-subsample the larger side down to the smaller within every retained cell so the two sides
carry equal weight per cell.

Deliberately pure: it takes a matching frame + boolean side masks and returns matched index positions
plus a balance/coverage diagnostic, with no dependency on the method. The matching axis (ERA5) is kept
separate from the reporting/binning axis (test-turbine ws/TI); this utility is generic over whatever
``bin_edges`` it is handed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Hard floor on matched rows per side; mirrors PowerModelMethod._MIN_BASELINE_ROWS so decimated
# matching fails loudly rather than silently producing a noisy estimate.
_MIN_MATCHED_ROWS = 10
# Below this retained fraction on either side, warn (matching is throwing most of the data away).
_WARN_RETAINED_FRACTION = 0.1


@dataclass
class MatchResult:
    """Matched positions per side plus the CEM balance/coverage diagnostic.

    :param baseline_positions: sorted integer positions (into the analysis index) of the matched
        baseline rows; equal count to ``upgraded_positions`` within every retained cell
    :param upgraded_positions: sorted integer positions of the matched upgraded rows
    :param per_cell: one row per cell with the per-var bin code(s), the before counts (``n_baseline`` /
        ``n_upgraded``) and the after count per side (``n_matched``, 0 for dropped one-sided cells)
    :param n_baseline_in: baseline rows entering matching (finite in every matching var)
    :param n_upgraded_in: upgraded rows entering matching
    """

    baseline_positions: np.ndarray
    upgraded_positions: np.ndarray
    per_cell: pd.DataFrame
    n_baseline_in: int
    n_upgraded_in: int

    @property
    def n_matched_per_side(self) -> int:
        """Matched rows per side (the effective sample size the two directions are estimated on)."""
        return len(self.baseline_positions)

    @property
    def retained_fraction_baseline(self) -> float:
        """Fraction of the entering baseline rows kept after matching."""
        return self.n_matched_per_side / self.n_baseline_in if self.n_baseline_in else float("nan")

    @property
    def retained_fraction_upgraded(self) -> float:
        """Fraction of the entering upgraded rows kept after matching."""
        return self.n_matched_per_side / self.n_upgraded_in if self.n_upgraded_in else float("nan")

    @property
    def n_cells_two_sided(self) -> int:
        """Cells present on both sides (retained)."""
        return int(((self.per_cell["n_baseline"] > 0) & (self.per_cell["n_upgraded"] > 0)).sum())

    @property
    def n_cells_one_sided(self) -> int:
        """Cells present on only one side (dropped by the common-support guard)."""
        return int(((self.per_cell["n_baseline"] == 0) ^ (self.per_cell["n_upgraded"] == 0)).sum())


def coarsened_exact_match(
    matching_frame: pd.DataFrame,
    *,
    baseline_sel: np.ndarray,
    upgraded_sel: np.ndarray,
    bin_edges: dict[str, list[float]],
    seed: int,
    min_matched_rows: int = _MIN_MATCHED_ROWS,
    warn_retained_fraction: float = _WARN_RETAINED_FRACTION,
) -> MatchResult:
    """Match baseline and upgraded rows on coarsened cells of the matching variables.

    :param matching_frame: the matching columns on the analysis index (row *position* is the identity
        the returned positions refer to); must contain every key of ``bin_edges``
    :param baseline_sel: boolean mask over the frame's rows selecting normally-operating baseline rows
    :param upgraded_sel: boolean mask over the frame's rows selecting normally-operating upgraded rows
    :param bin_edges: ``{var: edges}`` per matching variable; a row outside a var's edges (``pd.cut``
        returns NaN) is unmatchable and dropped, just like a non-finite value
    :param seed: seed for the per-cell subsample of the larger side
    :param min_matched_rows: hard floor; raise if fewer rows per side survive matching
    :param warn_retained_fraction: warn if either side retains less than this fraction
    """
    variables = list(bin_edges)
    missing = [v for v in variables if v not in matching_frame.columns]
    if missing:
        msg = f"matching_frame is missing matching columns {missing}; have {list(matching_frame.columns)}"
        raise ValueError(msg)

    n_rows = len(matching_frame)
    for name, sel in (("baseline_sel", baseline_sel), ("upgraded_sel", upgraded_sel)):
        if np.asarray(sel).shape != (n_rows,):
            msg = f"{name} shape {np.asarray(sel).shape} does not align to matching_frame's {n_rows} rows"
            raise ValueError(msg)

    codes = _cell_codes(matching_frame, bin_edges)
    valid_cell = np.all(codes >= 0, axis=1)  # -1 marks a NaN/out-of-range value in some var
    base = np.flatnonzero(np.asarray(baseline_sel, dtype=bool) & valid_cell)
    up = np.flatnonzero(np.asarray(upgraded_sel, dtype=bool) & valid_cell)
    n_baseline_in = int(base.size)
    n_upgraded_in = int(up.size)

    per_cell, matched_base, matched_up = _match_cells(codes, variables=variables, base=base, up=up, seed=seed)

    result = MatchResult(
        baseline_positions=np.sort(matched_base),
        upgraded_positions=np.sort(matched_up),
        per_cell=per_cell,
        n_baseline_in=n_baseline_in,
        n_upgraded_in=n_upgraded_in,
    )
    _check_coverage(result, min_matched_rows=min_matched_rows, warn_retained_fraction=warn_retained_fraction)
    return result


def _cell_codes(matching_frame: pd.DataFrame, bin_edges: dict[str, list[float]]) -> np.ndarray:
    """Integer cell code per (row, var); -1 where the value is NaN or outside the var's edges."""
    columns = []
    for var, edges in bin_edges.items():
        # include_lowest so a value exactly on the lowest edge (e.g. 0° direction, calm wind) lands in
        # the first cell rather than becoming NaN and being dropped as if out of range.
        cut = pd.cut(matching_frame[var], bins=edges, labels=False, include_lowest=True)  # NaN outside / on NaN
        columns.append(cut.to_numpy(dtype=float))
    stacked = np.column_stack(columns)
    return np.where(np.isfinite(stacked), stacked, -1.0).astype(int)


def _match_cells(
    codes: np.ndarray, *, variables: list[str], base: np.ndarray, up: np.ndarray, seed: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Group the two sides by cell, drop one-sided cells, subsample the larger side per two-sided cell."""
    rng = np.random.default_rng(seed)
    base_by_cell = _positions_by_cell(codes, base)
    up_by_cell = _positions_by_cell(codes, up)

    rows: list[dict] = []
    matched_base: list[int] = []
    matched_up: list[int] = []
    for cell in sorted(set(base_by_cell) | set(up_by_cell)):
        b = base_by_cell.get(cell, np.empty(0, dtype=int))
        u = up_by_cell.get(cell, np.empty(0, dtype=int))
        k = min(len(b), len(u))  # 0 for a one-sided cell -> nothing kept
        if k:
            matched_base.extend(_subsample(b, k, rng).tolist())
            matched_up.extend(_subsample(u, k, rng).tolist())
        row = dict(zip(variables, cell, strict=True))
        row |= {"n_baseline": len(b), "n_upgraded": len(u), "n_matched": k}
        rows.append(row)

    per_cell = pd.DataFrame(rows, columns=[*variables, "n_baseline", "n_upgraded", "n_matched"])
    return per_cell, np.asarray(matched_base, dtype=int), np.asarray(matched_up, dtype=int)


def _positions_by_cell(codes: np.ndarray, positions: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
    """Map each cell key (tuple of per-var codes) to the given positions that fall in it."""
    out: dict[tuple[int, ...], list[int]] = {}
    for pos in positions:
        out.setdefault(tuple(codes[pos].tolist()), []).append(int(pos))
    return {cell: np.asarray(v, dtype=int) for cell, v in out.items()}


def _subsample(positions: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Keep all ``k`` positions when the side is already at ``k``, else seeded-choose ``k`` without replacement."""
    if len(positions) == k:
        return positions
    return rng.choice(positions, size=k, replace=False)


def _check_coverage(result: MatchResult, *, min_matched_rows: int, warn_retained_fraction: float) -> None:
    """Warn on thin retention; raise below the hard floor so decimated matching fails loudly."""
    if result.n_matched_per_side < min_matched_rows:
        msg = (
            f"CEM retained only {result.n_matched_per_side} matched rows per side (< {min_matched_rows}); "
            f"the matching cells barely overlap. Coarsen the bins or widen the campaign."
        )
        raise ValueError(msg)
    for side, frac in (
        ("baseline", result.retained_fraction_baseline),
        ("upgraded", result.retained_fraction_upgraded),
    ):
        if frac < warn_retained_fraction:
            logger.warning(
                "CEM retained only %.1f%% of the %s rows (%d of the entering set); the matched estimate rests "
                "on a small, possibly unrepresentative slice.",
                100 * frac,
                side,
                result.n_matched_per_side,
            )
