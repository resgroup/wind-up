# W0 — `src/` layout + rename legacy to `wind_up_v0`

**Status:** design approved 2026-08-28. Branch: `v1-W0`.

## Goal

The new v1 tool claims the `wind_up` import name while the legacy tool is retained
under `wind_up_v0`. Done **early** so all later v1 code lands in the new layout. This
is a behaviour-preserving restructure: the legacy pipeline must produce byte-identical
results before and after.

## Sequencing (read first)

**The example baseline MUST be captured BEFORE ANY change is made** — before moving any
file, before touching a single import, before editing `pyproject.toml`. The very first
action of implementation is to run the runnable examples on the pristine, unchanged tree
and save their outputs (see Done-when #6). If any change lands before the baseline is
captured, the baseline is invalid and the whole equivalence check is worthless — in that
case `git stash` / reset back to a clean tree and capture it first. Every other step in
this spec happens only after the baseline exists.

## Target layout

```
wind-up/
  src/
    wind_up/          # v1 — new skeleton (claims the import name; W1 fills it)
      __init__.py     # package docstring + __version__ = version("res-wind-up")
      py.typed
    wind_up_v0/       # legacy — verbatim move of today's wind_up/
  benchmarking/       # unchanged location (stays OUT of src/)
  tests/              # unchanged location
  examples/           # unchanged location
  config/ input_data/ cache/   # PROJECTROOT_DIR targets; stay at root for W0
```

Distribution name **stays `res-wind-up`**; only import names change (`wind_up` = v1,
`wind_up_v0` = legacy).

## The rename rule

A blanket token rename **`wind_up` → `wind_up_v0`** applied to every reference to the
**legacy package**:

- internal imports inside the moved package (~40 files under `src/wind_up_v0/`);
- `benchmarking/` importers: `baselines/v0_binned.py`, `baselines/hot_context.py`,
  `synthetic/sources/hill_of_towie.py`, `synthetic/upgrades.py`;
- `examples/`: `kelmarsh_kaggle.py`, `smarteole_utils.py`, `wedowind_example.py`,
  and `smarteole_example.ipynb`;
- `tests/`: every importer, `tests/conftest.py`, and `tests/test_data/hot/*`;
- the self-import at `main_analysis.py:13` (`import wind_up` → `import wind_up_v0`)
  and the bare-attribute usages in `tests/test_wedowind.py` and `tests/test_version.py`.

**Two things that must NOT change** (behaviour-preserving):

1. The output dict **key** `"wind_up_version"` in `main_analysis.py` — it is part of
   v0's result schema. Only the `wind_up.__version__` expression feeding it changes;
   the value (same `res-wind-up` version string) is identical.
2. Prose / comments that name "wind_up" as the tool stay as-is unless actively
   misleading. No churn on documentation wording.

## Path-root fix

`src/wind_up_v0/constants.py`: `Path(__file__).parents[1]` → `parents[2]` for the
repo-root-relative constants (`PROJECTROOT_DIR`, `CONFIG_DIR`,
`TURBINE_DATA_DIR`, `REANALYSIS_DIR`, `TOGGLE_DIR`), so they keep resolving to the
repo root after gaining the `src/` level. `OUTPUT_DIR` (`Path.home() / ...`) is
unaffected. This keeps `tests/test_version.py`'s `PROJECTROOT_DIR / "pyproject.toml"`
and `tests/conftest.py`'s `CACHE_DIR = PROJECTROOT_DIR / "cache"` working.

## v1 skeleton

`src/wind_up/__init__.py` = package docstring + `__version__ = version("res-wind-up")`;
plus `src/wind_up/py.typed`. This keeps `tests/test_version.py`
(`import wind_up; wind_up.__version__ == pyproject version`) green unchanged, since
both packages share the `res-wind-up` distribution version. W1 fills the real composed
method.

## Packaging & config repointing (`pyproject.toml` and CI)

- `[tool.setuptools.packages.find]`: `where = ["src", "."]`,
  `include = ["wind_up*", "wind_up_v0*", "benchmarking*"]`. `src/` (no `__init__.py`)
  is not itself a package, so find yields `wind_up`, `wind_up_v0` from `src/` and
  `benchmarking` from the root without duplication; `tests`/`examples` are excluded by
  not matching `include`. Rewrite the existing comment to explain `benchmarking` is
  still packaged **temporarily** (external `toggle_specialist` use) and point at the
  W2 cleanup.
- poe `test` / `test-fast` coverage source: `--source wind_up` → `--source wind_up_v0`
  (coverage stays scoped to the legacy code, as today; skeleton/benchmarking uncovered).
- `[tool.coverage.report] omit`: `wind_up/plots/*.py` → `src/wind_up_v0/plots/*.py`.
- ruff `[tool.ruff.lint.per-file-ignores]`: repoint the `wind_up/...` keys to
  `src/wind_up_v0/...` (`models.py`, `smart_data.py`, `plots/*.py`). The `**/__init__.py`,
  `tests/**`, and `examples/**` globs are unchanged.
- mypy: `mypy .` names src modules correctly (it walks up from a file and stops at
  `src/`, which has no `__init__.py`, yielding `wind_up_v0.*`); no config change
  expected. Verify during implementation; if needed, add `mypy_path = "src"`.
- `.github/CODEOWNERS`: `wind_up/*` → `src/wind_up_v0/*` (and add `src/wind_up/*`).
- CI workflows drive everything through `poe` tasks and a generic `build`; no hardcoded
  `wind_up` paths, so no workflow edits beyond the above.
- Reinstall the editable env (`uv sync`) after the move so both import names resolve.

## Examples

- **Runnable examples** (at minimum `smarteole_example.ipynb`; also
  `kelmarsh_kaggle.py`, `wedowind_example.py` if their data / network is available):
  refresh the import paths to `wind_up_v0` and **re-execute** so the stored notebook
  outputs are regenerated against the renamed package. These land in the working tree
  for the user to commit.
- **Non-runnable examples** (data no longer available, network required, etc.): do not
  attempt to force-run. Add a short docstring / top-of-notebook note stating the example
  is not currently runnable and a brief reason. The user decides later what to do with
  them.

## benchmarking + deferred cleanup (recorded on W2)

benchmarking **stays packaged for now** (external `toggle_specialist` dependency). Add
a scope bullet to **W2** in `docs/v1/issues_campaigns.md`: once that external dependency
is gone,

- drop `benchmarking*` from packaging and confirm it is excluded from the v1.0.0
  release artifact; and
- delete the `config/`, `input_data/`, `cache/` root folders — legacy artefacts from
  before env-vars / `Path.home()` were used — reworking `constants.py`'s path handling
  accordingly (env vars / `Path.home()` instead of `PROJECTROOT_DIR`-relative).

## Verification / done-when

1. Repo builds and `poe all-fast` passes under the new layout.
2. `wind_up_v0` runs the legacy pipeline; its **committed benchmark is unchanged**.
3. No importer references the old `wind_up` path for the legacy tool.
4. `import wind_up` resolves to the new v1 skeleton and exposes `__version__`.
5. Runnable examples re-execute cleanly with refreshed imports; non-runnable ones carry
   a docstring note explaining why.

### Done-when #6 — one-off before/after example benchmark (NOT a test)

A manual, one-off equivalence check. No new or committed test artefacts; scratchpad only.

- **STEP ZERO — before any rename or config edit, on the pristine tree** (`git status`
  clean of W0 changes): run each runnable example with `OUTPUT_DIR` pointed at
  `<scratchpad>/baseline/`. This must happen before the first file is moved. Capture,
  per example:
  - all saved figure PNGs, and
  - a text file of stdout + the repr of any result dataframes / uplift numbers.
- **After** the rename + config changes, run the same examples with `OUTPUT_DIR` pointed
  at `<scratchpad>/after/`.
- Compare:
  - **Numbers:** `diff` the captured text — must be empty (exact equality).
  - **Plots:** load each `baseline/` vs `after/` PNG pair with PIL/numpy and assert the
    **pixel arrays are equal** (metadata-insensitive; stronger than a human glance).
  - **Human backstop:** open a couple of before/after PNG pairs directly to sanity-check,
    and report which examples ran vs were skipped.
- Any non-identical pixel array or numeric diff is a real signal to investigate, not to
  wave through — a pure import rename is expected to be exactly equal.
