# wind-up v1 — related tools & prior art

A living list of related open-source tools and key methodology references worth
deeper investigation as v1 progresses. Complements the reading list already in
`wind-up-ml-uplift-design-note.md` (causal-ML / treatment-effect literature).

## Related open-source tools

### FLASC — FLORIS-based Analysis for SCADA data (NREL)
<https://github.com/NatLabRockies/flasc>

SCADA filtering, analysis, model validation, and field-experiment design/monitoring,
integrated with the FLORIS wake model. Relevant to v1:
- **Synthetic / artificial data** — has `examples_artificial_data`; this is the
  precedent (P. Fleming) for generating SCADA-like data with a known answer.
  **Investigate for WS1** (synthetic upgrade-dataset generator): how they construct
  artificial data and inject known effects.
- **Energy-ratio methodology** for quantifying wake effects in both synthetic and
  historical data — a comparison point for uplift/energy-ratio metrics.
- Time-synchronization and filtering utilities for multi-turbine SCADA.

### OpenOA — Operational Assessment (NREL)
<https://github.com/NatLabRockies/OpenOA>

Wind-plant operational assessment from SCADA + met + reanalysis. Relevant to v1:
- **`MonteCarloAEP`** — long-term AEP from 1–3 yr records with reanalysis (ERA5 /
  MERRA-2) and **Monte-Carlo uncertainty**. Comparison/inspiration for **WS4**
  (uncertainty) and long-term extrapolation (G4 step 3).
- **`PlantData` schema** — a standardized, validated data structure integrating
  SCADA / met towers / revenue meters / reanalysis. **Prior art for WS3** (the
  assessment data contract) — worth studying before fixing our schema.
- Utilities: power-curve fitting, outlier/range filtering, imputation,
  met processing, plotting.
- Other analyses (`TurbineLongTermGrossEnergy`, `ElectricalLosses`, `WakeLosses`,
  `EYAGapAnalysis`, `StaticYawMisalignment`) — context, not directly uplift.

### DSWE — Data Science for Wind Energy (Y. Ding et al.)
R and Python. Implements the three-step `funGP` performance-comparison pipeline
(covariate matching → power model → functional-GP comparison with confidence bands),
explicitly framed as treatment-effect estimation. A natural **WS2 candidate /
cross-check**. See design-note §5.

## Key methodology references

### Kanev, *AWC validation methodology*, TNO 2020 R11300 (Aug 2020)
The reference for the **multi-dimensional binning** candidate (WS2). Directly
relevant beyond binning:
- **Multi-dimensional binning** — bin by wind speed *and* wind direction; keep ws
  bins fixed at 1 m/s and **adaptively widen the wind-direction bin** until the
  normalized standard error of farm power per bin drops below a target. Enables a
  power curve *per direction sector* (a route to G3 conditional uplift).
- **Toggle-period study (§4.8)** — quantifies how toggle/campaign length affects
  the mean power ratio and its 95% CI. **Directly relevant to G2** (shorter
  campaigns) and to designing the WS1 short-campaign sweep. Notable cautionary
  result: a **12-hour toggle period** badly biases results because each data set
  then samples only one half of the **diurnal cycle** (different atmospheric
  stability) — reinforces G3 (day/night conditioning) and is a pitfall the
  synthetic-data harness should be able to reproduce.
- **Consensus wind speed/direction** from farm-wide nacelle anemometry — a
  treatment-invariant reference-construction idea relevant to WS3 features.
- **Filtering rules** for unavailability / curtailment / power-boosting (exclude
  affected turbines and those in their wakes, rather than dropping whole records).
- **Uncertainty quantification (§4.9)** — power-ratio CIs; context for WS4.

### Astolfi et al. — multivariate-linear before/after method
- Astolfi, Castellani & Terzi (2018), *Wind Turbine Power Curve Upgrades*,
  Energies 11(5):1300.
- Astolfi, Castellani, Fravolini, Cascianelli & Terzi (2018), *Computing the real
  impact of wind turbine power curve upgrades: a SCADA-based multivariate linear
  method and a vortex generator test case* (preprints.org 2018060082).

The source of the **WS2 "Astolfi multivariate-linear" candidate**. Key ideas:
- Don't compare raw before/after energy under non-stationary conditions; instead
  compare post-upgrade production to a **data-driven model of pre-upgrade
  production under the same conditions** (a before/after residual method) — the
  same logic as wind-up's pre/post approach.
- **Multivariate linear** power model. Practical tip (design note §5): feeding
  **min/max/std of the inputs, not just the means**, cut their error metrics by
  roughly a third — directly relevant to feature engineering in **WS1/WS3**.
- Reported upgrade magnitudes (~1.3% yaw optimisation, ~2.5% control re-powering)
  are the same order as Hill of Towie (~0.7–1.7%) — useful sanity range.

### Ding, Barber & Hammer (2022) — funGP performance comparison
*Data-Driven wind turbine performance assessment and quantification using SCADA
data and field measurements*, Frontiers in Energy Research 10:1050342.

The methodology paper behind the **DSWE / funGP** tool above: a three-step
pipeline — covariate matching → data-driven power model → **functional-GP
(`funGP`) comparison with confidence bands** — explicitly framed as
treatment-effect estimation. A **WS2 candidate / cross-check**.

### Further reading
See `wind-up-ml-uplift-design-note.md` §5 for the causal-ML reading list (R-learner,
DML, metalearners; Ding/Astolfi wind-specific work; conformal prediction; block
bootstrap) and §7 for the proposed dependencies.
