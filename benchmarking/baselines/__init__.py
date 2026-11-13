"""v0 binned power-curve baseline for the benchmarking harness.

Wires wind_up's pre/post power-performance pipeline behind the harness's ``Method`` seam so
the existing v0 method can be scored against synthetic ground truth, establishing the
baseline every new method must beat.
"""

from __future__ import annotations

from benchmarking.baselines.hot_context import HotV0Context, build_hot_v0_context
from benchmarking.baselines.v0_binned import V0BinnedMethod

__all__ = ["HotV0Context", "V0BinnedMethod", "build_hot_v0_context"]
