"""Shared per-run diagnostics for the v1 benchmarking methods.

A method adapts its internals to a :class:`~benchmarking.diagnostics.context.DiagnosticContext`
and calls :func:`~benchmarking.diagnostics.common.write_common_diagnostics` (plus
:func:`~benchmarking.diagnostics.config_dump.write_run_config`) to emit a consistent, v0-grade
set of diagnostic plots and a run-config file. Project plotting conventions live in
:mod:`~benchmarking.diagnostics.style` (grid on by default) and
:mod:`~benchmarking.diagnostics.density` (density-coloured scatter).
"""

from __future__ import annotations

from benchmarking.diagnostics.common import write_common_diagnostics
from benchmarking.diagnostics.config_dump import write_run_config
from benchmarking.diagnostics.context import DiagnosticContext, infer_timebase
from benchmarking.diagnostics.density import density_scatter
from benchmarking.diagnostics.style import apply_grid, save_fig

__all__ = [
    "DiagnosticContext",
    "apply_grid",
    "density_scatter",
    "infer_timebase",
    "save_fig",
    "write_common_diagnostics",
    "write_run_config",
]
