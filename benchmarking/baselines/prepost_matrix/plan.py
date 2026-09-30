"""Project a matrix's cost before running it: CPU-hours, wall hours on N workers, and peak memory.

Costs are :data:`~benchmarking.baselines.prepost_matrix.cells.CELL_COST_S`, or a finished study's
measured ones, so a size can be fitted to a time budget without a trial run.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from benchmarking.baselines.prepost_matrix.cells import CELL_COST_S, PEAK_RSS_MB, cell_cost_s, matrix_cells

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from benchmarking.baselines.prepost_matrix.cells import MatrixSettings

_S_PER_H = 3600
_MB_PER_GB = 1024


@dataclass(frozen=True)
class Plan:
    """What a matrix is expected to cost on ``workers`` workers."""

    n_cells: dict[str, int]
    workers: int
    cpu_hours: float
    wall_hours: float
    peak_memory_gb: float

    def describe(self) -> str:
        """Return the plan as a few readable lines."""
        sites = ", ".join(f"{site} {n}" for site, n in sorted(self.n_cells.items()))
        return (
            f"{sum(self.n_cells.values())} cells ({sites})\n"
            f"{self.cpu_hours:.1f} CPU-hours; about {self.wall_hours:.1f} h on {self.workers} worker(s)\n"
            f"peak memory up to {self.peak_memory_gb:.0f} GB (the largest cell's peak RSS times the workers)"
        )


def makespan_s(durations: Sequence[float], *, workers: int) -> float:
    """Return when the last cell ends if each, in order, goes to the least-loaded worker."""
    loads = [0.0] * workers
    for duration in durations:
        heapq.heapreplace(loads, loads[0] + duration)
    return max(loads)


def measured_costs(study_dir: Path) -> tuple[dict[tuple[str, int], float], dict[str, float]]:
    """Return a study's ok cells' mean wall seconds per (site, K) and largest peak RSS (MB) per site."""
    table = pd.read_csv(study_dir / "cells.csv")
    ok = table[table["status"] == "ok"]
    costs = {(str(site), int(k)): float(v) for (site, k), v in ok.groupby(["site", "k"])["wall_time_s"].mean().items()}
    rss = {str(site): float(v) for site, v in ok.groupby("site")["peak_rss_mb"].max().items()}
    return costs, rss


def plan_study(settings: MatrixSettings, *, workers: int, measured: Path | None = None) -> Plan:
    """Project ``settings``' cost on ``workers`` workers, from ``measured``'s costs where it has them."""
    costs, rss = dict(CELL_COST_S), dict(PEAK_RSS_MB)
    if measured is not None:
        found_costs, found_rss = measured_costs(measured)
        costs |= found_costs
        rss |= found_rss
    cells = matrix_cells(settings)
    durations = sorted((cell_cost_s(c, costs) for c in cells), reverse=True)
    peak_mb = max((rss[c.site] for c in cells), default=0.0)
    return Plan(
        n_cells=dict(Counter(c.site for c in cells)),
        workers=workers,
        cpu_hours=sum(durations) / _S_PER_H,
        wall_hours=makespan_s(durations, workers=workers) / _S_PER_H,
        peak_memory_gb=peak_mb * min(workers, len(cells)) / _MB_PER_GB,
    )
