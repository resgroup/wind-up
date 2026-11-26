# wind-up

A Python package for assessing wind turbine yield uplift from operational data.

[![PyPI version](https://img.shields.io/pypi/v/res-wind-up.svg)](https://pypi.python.org/pypi/res-wind-up)
[![License](https://img.shields.io/pypi/l/res-wind-up.svg)](https://github.com/resgroup/wind-up/blob/main/LICENSE.txt)
[![Python versions](https://img.shields.io/pypi/pyversions/res-wind-up.svg)](https://pypi.python.org/pypi/res-wind-up)
[![Lint & Format: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![Typing: mypy](https://img.shields.io/badge/typing-mypy-yellow.svg)](https://github.com/python/mypy)
[![Lint and test](https://github.com/resgroup/wind-up/actions/workflows/lint-and-test.yaml/badge.svg)](https://github.com/resgroup/wind-up/actions/workflows/lint-and-test.yaml)

`wind-up` compares turbine performance before and after a change to estimate
energy-yield uplift. It is designed for wind-farm SCADA analysis where the
signal of interest is small, operational data is messy, and a credible result
needs more than a simple before/after power-curve plot.

The package is published on PyPI as `res-wind-up`. It holds two Python packages:
`wind_up` is the v1 interface, which is in progress, and `wind_up_v0` is the 0.x code,
unchanged.

> [!IMPORTANT]
> Coming from wind-up 0.x? Every `wind_up.<module>` import becomes `wind_up_v0.<module>`.
> See [migrating from 0.x](docs/migrating-from-0.x.md).

## What it does

`wind-up` provides a complete analysis workflow for pre/post or toggle-style
uplift assessments:

- prepares and filters wind-farm SCADA data
- builds SCADA-derived power curves by turbine type
- adds reanalysis, mast, or LiDAR reference data
- applies yaw direction northing corrections
- estimates wind speed and waking state
- detrends test turbine performance against reference wind direction
- performs pre/post power performance analysis with reversal checks and bootstrapped uncertainty
- combines per-reference results into turbine-level and fleet-level uplift estimates
- writes result tables and diagnostic plots for review

The public examples cover several realistic analysis shapes, including
SMARTEOLE toggle data and WeDoWind challenge-style pre/post assessments.

Further example useage can be seen in the [Hill of Towie open analysis repo](https://github.com/resgroup/hill-of-towie-open-source-analysis).

## Methodology

The validation methodology that `wind-up` implements is described in detail in
[**wind-up uplift validation methodology v3**](docs/wind-up%20uplift%20validation%20methodology%20v3.pdf)
in the [`docs`](docs) folder.

## Installation

Install the released package with your Python environment manager of choice.
Python 3.10 to 3.13 is supported.

Using `uv`:

```bash
uv add res-wind-up
```

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install res-wind-up
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

To run the bundled notebooks and example scripts, install the optional example
dependencies as well:

```bash
uv add "res-wind-up[examples]"
# or
pip install "res-wind-up[examples]"
```

Check that the package imports correctly:

```python
import wind_up

print(wind_up.__version__)
```

## Ways to run an analysis

There are three, and a new analysis should start from the first.

| Entry point | What it is | Use it when |
| --- | --- | --- |
| **The Python facts API** (`wind_up`) | You describe the wind farm as `WindFarmFacts`, ask a question as an `AnalysisSpec` and pass both to a method: `method.estimate(facts, analyses)` | Starting a new analysis. You choose the method, can run several methods on the same facts, and work with the results in Python |
| **The YAML campaign run** | One YAML file declares the campaign and one command runs every step of [the v1 method](docs/v1/method.md), from data checks and northing to a farm uplift with a report | You want the v1 power model method end to end without writing Python. It runs from a clone of this repository, not from the installed package |
| **The 0.x interface** (`wind_up_v0`) | `WindUpConfig` and `run_wind_up_analysis`, as in wind-up 0.7 | Maintaining an existing 0.x analysis |

Where each is documented:

- The facts API: [`examples/smarteole_example.ipynb`](examples/smarteole_example.ipynb) is a
  walkthrough on open data and the best place to start, and
  [the facts API reference](docs/facts-api.md) describes every object.
  [`examples/wedowind_example.py`](examples/wedowind_example.py) runs pre/post analyses the same
  way.
- The YAML campaign run: [running a campaign](docs/running-a-campaign.md), and
  [designing a campaign](docs/designing-a-campaign.md) for choosing the turbines to upgrade.
- The 0.x interface: the sketch under [The 0.x interface](#the-0x-interface) below, and
  [migrating from 0.x](docs/migrating-from-0.x.md).

Which method each can use:

| Method | Facts API | YAML campaign run |
| --- | --- | --- |
| Toggle energy ratio (`wind_up.methods.ToggleEnergyRatioMethod`) | yes | no |
| Binned power curve, the 0.x method (`V0BinnedMethod`) | yes, from a clone | no |
| Power model, the v1 method (`PowerModelMethod`) | yes, from a clone | yes; it is the only method the run uses |

> [!NOTE]
> The `benchmarking/` folder is the evaluation platform that v1 methods are developed and scored
> on. It is not part of the product and will not be in the 1.0 package. Two methods and the YAML
> campaign run still live there and move into `wind_up` before 1.0: until then
> `V0BinnedMethod` is imported from `benchmarking.baselines.v0_binned`, `PowerModelMethod` from
> `benchmarking.baselines.power_model`, and the campaign run is
> `python -m benchmarking.campaigns run`. Code that imports from `benchmarking` will need its
> imports changed at 1.0.

## The 0.x interface

A 0.x analysis has four stages:

```python
from wind_up_v0.interface import AssessmentInputs
from wind_up_v0.main_analysis import run_wind_up_analysis
from wind_up_v0.models import PlotConfig, WindUpConfig
from wind_up_v0.reanalysis_data import ReanalysisDataset

# 1. Build a WindUpConfig describing the asset, test turbines,
#    reference turbines, analysis dates, filters, and output folder.
cfg = WindUpConfig(...)

# 2. Configure diagnostic plot output.
plot_cfg = PlotConfig(
    show_plots=False,
    save_plots=True,
    plots_dir=cfg.out_dir / "plots",
)

# 3. Prepare inputs from SCADA, metadata, and reference datasets.
assessment_inputs = AssessmentInputs.from_cfg(
    cfg=cfg,
    plot_cfg=plot_cfg,
    scada_df=scada_df,
    metadata_df=metadata_df,
    reanalysis_datasets=[ReanalysisDataset(id="ERA5", data=reanalysis_df)],
    cache_dir=cache_dir,
)

# 4. Run the assessment and save the per-test/per-reference results.
results_per_test_ref_df = run_wind_up_analysis(assessment_inputs)
results_per_test_ref_df.to_csv(cfg.out_dir / "results_per_test_ref.csv", index=False)
```

For a complete, executable version of this pattern, see `tests/test_smarteole.py`.
The configuration is intentionally explicit: wind-resource assessments are
sensitive to turbine metadata, analysis periods, filters, and reference choices,
so those assumptions should be visible in code.

## Inputs and outputs

At a high level, an assessment needs:

| Input | Purpose |
| --- | --- |
| SCADA time series | turbine power, wind speed, yaw, pitch, RPM, downtime, and related operational signals |
| turbine metadata | turbine names, turbine type, rated power, rotor diameter, and location where available |
| analysis configuration | test/reference turbines, pre/post or toggle periods, filters, long-term settings, and output paths |
| reference data | reanalysis, mast, LiDAR, or reference-turbine signals used for detrending and validation |

The main analysis returns a `pandas.DataFrame` with per-test/per-reference uplift
results, uncertainty columns, warning counts, and supporting diagnostic metrics.
When plot saving is enabled, diagnostic figures are written under
`PlotConfig.plots_dir`; CSV results are written under the configured assessment
output directory.

> [!NOTE]
> On import, `wind-up` selects the non-interactive matplotlib `Agg` backend
> unless the `MPLBACKEND` environment variable is already set. This lets analyses
> run headless (CI, SSH, batch servers) without an X server. Set `MPLBACKEND`
> yourself before importing `wind_up_v0` if you want an interactive backend.

## Analysis features

`wind-up` includes utilities for the parts of an uplift study that usually need
careful handling:

- SCADA cleaning and filtering for unavailable or implausible operating data
- power curve estimation by turbine type
- wind speed estimation from turbine power (preferred for reference turbines) and measured wind speed
- waking state calculation per timestep using turbine coordinates and SCADA
- yaw direction northing checks and optional optimized northing corrections
- long-term distribution calculations
- wind speed drift checks
- pre/post and toggle-based splitting
- reference selection and combined uplift calculations
- diagnostic plots for input data, detrending, power curves, yaw direction,
  reanalysis comparison, waking state, long-term distributions, and final results

## Development

This project uses `uv` for dependency management, `poethepoet` for task running,
Ruff for formatting/linting, mypy for type checking, and pytest with coverage for
tests.

> [!IMPORTANT]
> The test data (SCADA, reanalysis) is stored with [Git LFS](https://git-lfs.com/).
> Install it **before cloning** — or, in an existing clone, run
> `git lfs install && git lfs pull` once — otherwise those files check out as small
> pointer stubs and the data-backed tests and example notebooks will fail.

Create the development environment:

```bash
uv sync --all-extras --dev
```

> [!TIP]
> For normal local development and pre-push checks, start with the fast suite.
> It runs formatting, linting, type checking, and tests that are not marked as
> slow.

```bash
uv run poe all-fast
```

Run the full local verification suite before releases or when you need the slow
regression tests as well:

```bash
uv run poe all
```

> [!NOTE]
> `uv run poe all` includes tests marked as slow. On a typical local machine it
> can take 15 minutes or more; `uv run poe all-fast` is much quicker for everyday
> iteration.

Individual tasks are also available:

```bash
uv run poe lint-check   # formatter, linter, and mypy checks
uv run poe test-fast    # tests excluding the slow marker
uv run poe test         # full test suite with coverage report
uv run poe jupy         # start JupyterLab for example exploration
```

> [!WARNING]
> `uv run poe jupy` starts a local JupyterLab server and keeps running until you
> stop it. Use it when you want an interactive notebook session, not as a
> one-shot verification command.

The GitHub Actions workflow runs linting and tests on Python 3.10 and 3.13.

## Project status

The package is marked as beta in the Python package metadata. Interfaces and
configuration options may still evolve, but the repository includes a substantial
test suite and example analyses for regression coverage.

## License

`wind-up` is released under the BSD 3-Clause License. See
[`LICENSE.txt`](LICENSE.txt) for the full license text.

## Contact

For questions about the package, contact Alex Clerc at
[Alex.Clerc@res-group.com](mailto:Alex.Clerc@res-group.com).
