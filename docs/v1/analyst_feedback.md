# wind-up v1 — analyst feedback log

What the W3 dry-run analysts asked wind-up to tell them and it did not. This is a triage inbox,
not a decision log: entries are raised here, then promoted to an issue in
[issues_campaigns.md](issues_campaigns.md) or rejected. Evidence for why a run reached its verdict
belongs in [findings_campaigns.md](findings_campaigns.md); this file records the *request*.

Entries are numbered **AF*n***. Each names the runs that raised it, because an ask raised
independently by analysts in both modes is worth more than one raised once.

**Status** is one of `untriaged`, `issue <n>`, or `rejected: <reason>`.

## The runs

| run | mode | campaign | verdict | truth |
|---|---|---|---|---|
| instance-f | prepost | Hill of Towie, 21 turbines, 9 test, changeover 2018-06-01, a year either side | "cannot distinguish from zero", class *nothing at all* | 0 |
| instance-g | toggle | Hill of Towie, 21 turbines, 9 test, 100-min blocks from 2018-03-01, 12-month baseline | "cannot distinguish from zero" | 0 |

Both are placebos on real SCADA (2026-09-12), each given one run, the documentation, and the
campaign design. Both reached the right answer, so these asks are about the *cost* of getting
there, not about a wrong result.

---

## AF1 — Put the reference-relative estimate, with a standard error, in `farm_uplift.csv`

**Raised by:** instance-f (its number 1), instance-g (its number 1, as an automatic placebo row).
**Status:** untriaged.

`farm_uplift.csv` leads with `estimate`, which is the quantity that moves when the reference set
changes. The prepost analyst measured how much: re-running against seven references instead of
twelve moved the headline **1.25 pp** (+0.70% to +1.95%), while the gap between the upgraded mean
and the reference mean moved 0.50 pp and stayed insignificant in both runs (t = −0.05, then 1.25).
The gap is the answer; the headline is not. Both analysts assembled it by hand from
`per_turbine.csv` and `reference_stability.csv`.

Requested columns, per analyst: `reference_mean`, `reference_sd`, `estimate_vs_reference_mean`,
`estimate_vs_reference_se`, `t_stat`. The toggle analyst asked for the same thing from the other
end — an automatic A-A/placebo row giving the campaign's own resolution (`resolution_pp`), which on
its run was **0.84 pp** against a reference spread of 2.94 pp, so the documented yardstick was
three times too pessimistic.

Both analysts said the same thing about what this would have bought: the prepost one would have
written "indistinguishable from zero" in one line **and would not have been tempted by a
post-hoc reference subgroup** that phase 2 later proved wrong.

## AF2 — Resolve reference stability in time: `reference_stability_by_month.csv`

**Raised by:** instance-f (its number 2), instance-g (its number 4).
**Status:** untriaged.

`reference_stability.csv` gives one number per reference for the whole campaign. T17 read +5.42%
(prepost) and +1.72% (toggle) and neither analyst could tell whether it stepped at the changeover,
drifted, or carried a one-off outage — which is the difference between a sensor fault, a repair,
and something that happened to the whole site. The existing columns plus a `month` column would do
it. The prepost analyst called this "the single output that would have told me most" after AF1, and
noted it would also have exposed that its apparently well-behaved reference subgroup was not
stable either.

## AF3 — An `anemometer/` directory mirroring `northing/`

**Raised by:** instance-f (its number 3), instance-g (its number 8).
**Status:** untriaged, and **downgraded** — Alex, 2026-09-12: anemometer faults are coming out of
the brief's failure-mode list and will not be injected into campaigns, because CF11 measured them
at 0.21 pp worst case in prepost and exactly zero in toggle. What remains of this ask is that the
campaign cannot see a real fault in the data, not that the fault changes the answer.

The brief lists "an anemometer gain step or drift" among the known failure modes and **no output
speaks to wind speed sensors at all**, while `northing/` covers the direction sensor thoroughly
(changepoints with timestamps and magnitudes, per-turbine plots, a farm overview). Both analysts
built a substitute in ~30 lines of pandas — each turbine's `wtc_AcWindSp_mean` over the farm
median, by month — and it found two faults the runs never mentioned:

- **T01**: a drift from +0.7% to −5.1% over roughly Oct 2017 – Jan 2018, net −3.9 pp, entirely
  inside the baseline period. T01 was a trusted reference in both prepost runs.
- **T16**: a step of about −9% in August 2017 — the same month as the T16 nacelle-position
  changepoint wind-up found at 2017-08-09 13:20. wind-up found and corrected one half of a
  combined sensor event and said nothing about the other half.

Requested shape: `anemometer/farm_anemometer.png` plus `anemometer_corrections.yaml`, listing
changepoints the way the northing pair does. Note that reference nacelle wind speed is deliberately
not a model feature; it reaches the estimate only through ERA5 time-alignment, so the exposure of
the *estimate* to this class of fault is smaller than the analysts assumed. That does not answer
the ask: a campaign report that cannot see a 9% sensor step is missing a failure mode the brief
names.

## AF4 — Write the run's log into `--out`, as v0 does

**Raised by:** a human analyst, 2026-09-12.
**Status:** untriaged.

`python -m benchmarking.campaigns run` configures logging with `logging.basicConfig`, which installs
a console handler and nothing else, so the run leaves no log behind. v0's `setup_logger`
(`examples/helpers.py`) adds a `FileHandler` beside the console one and the examples write
`analysis.log` into the analysis output directory.

This matters more here than it looks, because stdout is load-bearing in this tool and nothing else
records it: the reference screen's thresholds, its minimum pool size, and the fact that it stopped
early exist only in the log, and when it stops `reference_stability.csv` comes back header-only and
silent. An analyst who closes the terminal has lost the only account of how the pool was judged.

A campaign run is long enough that nobody watches it live, so the log is read afterwards or not at
all. `--out` already exists and is the natural home.

---

## Also raised, not yet ranked

Kept so nothing is lost. Same status field; promote into a numbered entry when triaged.

| ask | raised by | status |
|---|---|---|
| Say which directory to run from. §2 gives the command and never says. (The `data:` paths resolve against the declaration's own directory, so only the path to `campaign.yaml` is relative to where you stand — which the reader also cannot tell) | a human analyst, 2026-09-12 | untriaged |
| The `timing` block's commented `# prepost:` / `# toggle:` labels read as required keys, so the first attempt nested `prepost:` and `changeover:` under `mode:`. `mode` is then a mapping and the error reports its repr — `unknown timing.mode "{'prepost': None, ...}"` — rather than saying `mode` must be one of two strings | a human analyst, 2026-09-12 | untriaged |
| `conditional.csv` extended to the reference turbines, so per-bin shape has a noise floor | instance-f | untriaged |
| A wind-direction axis in `conditional.csv` (30° bins) — the axis wake steering lives on | instance-g | untriaged |
| A yaw-offset table: median (nacelle position − reference direction), ON minus OFF, by sector | instance-g | untriaged |
| Leave-one-reference-out sensitivity on the headline | instance-f | untriaged |
| Baseline-vs-treated availability for *references*, not just test turbines | instance-f, instance-g | untriaged |
| ON-vs-OFF diagnostic plots: pre-period and OFF blocks are both labelled "baseline", so every plot is season-confounded | instance-g | untriaged |
| Document the screen's rule (2.5 pp from the pool median) — it appears only in the run log | instance-f | untriaged |
| Quantify "a healthy campaign reads near 0%", and warn that the reference spread is a property of the reference set, not of the campaign | instance-f, instance-g | untriaged |
| `rated_power_kw` is undocumented and unvalidated against the data | instance-f | untriaged |
| `northing_corrections.yaml` has no header, units or sign convention; the "(1 cp)" plot title is load-bearing | instance-f | untriaged |
| Wake steering has no path through the tooling: the designer says it is "not appropriate", nothing says what is | instance-f | untriaged |
