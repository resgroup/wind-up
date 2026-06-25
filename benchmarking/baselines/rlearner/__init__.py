"""Cross-fit R-learner uplift method (v1 Issue 5).

A pluggable, v0-independent treatment-effect estimator behind the harness ``Method`` seam.
See ``docs/superpowers/specs/2026-06-25-rlearner-method-design.md``.
"""

from __future__ import annotations

from benchmarking.baselines.rlearner.method import RLearnerMethod

__all__ = ["RLearnerMethod"]
