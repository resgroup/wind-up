# Migrating from wind-up 0.x

From 1.0 the `res-wind-up` distribution holds two packages:

- `wind_up` is the v1 interface: facts, analyses and methods ([the facts API](facts-api.md)).
- `wind_up_v0` is the 0.x code, unchanged. Everything that 0.x called `wind_up` is here.

## What breaks, and the fix

**Every 0.x import path changes.** Replace `wind_up` with `wind_up_v0`:

```python
# 0.x
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig

# 1.0
from wind_up_v0.interface import AssessmentInputs
from wind_up_v0.main_analysis import run_wind_up_analysis
from wind_up_v0.models import PlotConfig, WindUpConfig
```

Nothing else about a 0.x analysis changes: the configuration, the YAML files and the results are
the same.

**An old path fails on import, with the new path in the message.** Importing a 0.x module path
such as `wind_up.models` or `wind_up.plots.detrend_plots` raises an `ImportError`:

```
ImportError: wind_up.models is a res-wind-up 0.x path. From 1.0 the 0.x code is the wind_up_v0 package: use wind_up_v0.models.
```

Three modules exist in both packages: `wind_up.circular_math`, `wind_up.era5` and
`wind_up.northing`. The first two hold every 0.x name, so imports from them keep working. A 0.x
name of `wind_up.northing` raises the same `ImportError`, naming `wind_up_v0.northing`.

**A package that depends on 0.x without an upper bound installs 1.0.** A requirement such as
`res-wind-up>=0.7` picks up 1.0 on a fresh install, and the package then fails at its first
`wind_up` import. Until the package is migrated, pin it:

```
res-wind-up<1
```

**Type checking sees the real annotations.** 1.0 ships a `py.typed` marker, so mypy and other type
checkers no longer treat the package as untyped. A downstream type check can report mismatches in
code that has not changed. They were there before; they were not checked.

## Running the 0.x method from the v1 interface

The 0.x method, the binned power curve, also runs from v1 facts and analyses as `V0BinnedMethod`,
so one declaration of a wind farm can be answered by it and by the v1 methods. The
[SMARTEOLE notebook](../examples/smarteole_example.ipynb) and
[`examples/wedowind_example.py`](../examples/wedowind_example.py) do this. Until it moves to
`wind_up.methods` before 1.0, it is imported from the repository
(`benchmarking.baselines.v0_binned`), not from the installed package.

On this route some things that were configuration in 0.x come from the facts instead:

- **Downtime.** v0 filters downtime by a `ShutdownDuration` column. When the records have none,
  it is derived as the record's period less the availability counter, and a missing counter reading
  counts as fully available. To keep a downtime definition of your own, give the records a
  `ShutdownDuration` column before building the facts; it is then used as given.
- **Northing.** The facts' declared `north_offsets` are used, not v0's own northing corrections.
- **Dates, test and reference turbines, toggle signal and settling time** come from the upgrade and
  the analysis.

The method's own options are the remaining fields of v0's `WindUpConfig` and `TurbineType`.
Results can differ a little from a 0.x run of the same data for these reasons; the notebook states
the differences it sees.
