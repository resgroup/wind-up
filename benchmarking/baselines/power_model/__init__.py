"""Counterfactual power-model uplift method (v1 — simplest-possible ML).

A pluggable, v0-independent counterfactual power model behind the harness ``Method`` seam: learn
the test turbine's normal power from curated reference-only (weather + wake) features over the
baseline, predict the counterfactual over the upgraded window, and take the energy ratio. See
``docs/superpowers/specs/2026-06-29-power-model-design.md``.
"""

from __future__ import annotations

from benchmarking.baselines.power_model.method import PowerModelMethod

__all__ = ["PowerModelMethod"]
