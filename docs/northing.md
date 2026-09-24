# How wind-up norths a farm

A turbine's nacelle position is only as good as its north calibration. Sensors are installed with
an offset, drift, and get re-zeroed during maintenance, and every direction-dependent calculation
inherits the error. `wind_up.northing.north_farm` finds those offsets and returns one **north
table** per turbine: a list of `(timestamp, north_offset)` rows, each offset holding until the next.
Offsets are absolute: adding one to the raw signal norths it. `apply_north_table` applies a table,
and `write_north_table_yaml` writes the tables in the same format v0's
`optimized_northing_corrections.yaml` uses, so a table can be hand edited and supplied back.

`north_farm` runs up to four steps. Which run depends on the farm size and on what the caller
supplies.

| step | runs when | reference | finds changepoints |
|---|---|---|---|
| reanalysis-anchor | always | reanalysis | no |
| changepoints-v-consensus | 3 or more turbines | nearest-neighbour consensus | yes |
| changepoints-v-reanalysis | fewer than 3 turbines | reanalysis | yes, coarser |
| wake-nadir-shift | a layout and power are supplied | wake geometry | no |

## The changepoint search

Every step that finds changepoints uses `estimate_north_table`. It takes the circular difference
between a turbine's direction and a reference over the rows where the turbine is generating and
available (`yaw_usable`), and searches that residual for step changes by exact dynamic programming
on a daily grid, then pins each step to native resolution. The residual is first normalised per
direction sector, so site veer (a turbine reading differently from the reference depending on where
the wind comes from) is not mistaken for a step. Each segment's offset is the circular median of
its raw residual, so the correction stays absolute.

The search then prunes what it found. A small step the record later undoes is wander, not a
recalibration, and is dropped. A step with little record either side of it must be larger to
count, because a short segment's level is uncertain. `NorthingSettings` holds the knobs: the
smallest step reported, the shortest segment, the changepoint budget per year, and so on.

## reanalysis-anchor

Each turbine gets one constant offset that nulls its whole-record direction against reanalysis
(ERA5). This step fixes the farm in absolute terms: without it, a farm that is uniformly wrong
looks self-consistent. It finds no changepoints, because reanalysis is unreliable over short
periods: an unusual weather spell moves every turbine's residual together, and correcting that
would write the excursion into the consensus the next step trusts.

## changepoints-v-consensus

For a farm of three or more turbines, each turbine is northed against a consensus of the others,
and this step finds all the changepoints. The consensus is the per-timestamp circular median of
the anchored directions of the turbine's four nearest neighbours by geodesic distance from the
`Layout`. A turbine shares wind with its neighbours, not with the far side of a large farm, and a
miscalibrated turbine elsewhere cannot pull its reference. A consensus stands at a timestamp only
when a strict majority of its turbines report, and never fewer than three. With `layout=None`,
every turbine is northed against one whole-farm consensus instead.

A neighbour's own step is still in the anchored signals the first consensus is built from, and a
four-turbine median moves when one member steps, which would hand the turbine a matching false
step. So the step repeats: each round rebuilds the consensus from the previous round's tables,
re-northing only turbines whose neighbours changed, until no table moves by more than half a
degree or shifts a changepoint by more than a day.

A turbine whose consensus never overlaps its own usable rows keeps its reanalysis anchor.

## changepoints-v-reanalysis

A farm of one or two turbines cannot form a consensus, so each turbine is northed directly against
reanalysis with changepoints. Reanalysis wanders, so this step needs a larger step (10 deg) and a
longer gap between changepoints (30 days) before it attributes a step to the turbine
(`against_reanalysis`).

## wake-nadir-shift

The steps above tie the farm's absolute north to reanalysis, so any error in reanalysis's own
direction remains in every table. When `north_farm` is given a layout and power, it shifts
each turbine's table by one constant, read from where its wake lands
(`wind_up.wake_nadir.wake_nadir_offsets`).

A turbine's wake reaches a downstream neighbour when the wind blows along the bearing between them.
For each such pair within 10 rotor diameters, the downstream turbine's power (and, when supplied,
nacelle wind speed) relative to the upstream turbine's is binned against the upstream turbine's
northed direction, in 1 deg bins across a 30 deg sector centred on the geometric bearing. A
quadratic fit locates the dip. How far the dip sits from the bearing is the upstream turbine's
residual error. A dip that is unbracketed, not convex, too shallow or too thinly sampled is
rejected.

A turbine's pairs are combined by their circular median, so a majority of consistent pairs
outvotes one biased by terrain or wake deflection. A turbine with no resolvable pair takes the
circular median of its nearest resolved turbines, or zero if there are none. The shift moves every
offset in the turbine's table and never changes which rows are valid or where the changepoints are.

The campaign runner writes a map of these shifts (`wake_nadir_bubble.png`) with the other northing
plots.
