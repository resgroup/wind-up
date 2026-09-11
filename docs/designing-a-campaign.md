# Designing a campaign with wind-up

Before a campaign runs, someone has to choose which turbines to upgrade. wind-up's campaign
design utility makes that choice for you, or checks one you made by hand, against the reference
rules of the wind-up validation methodology (`docs/wind-up uplift validation methodology v3.pdf`,
step 2): *"A set of reference turbines is chosen for each test turbine. At least three reference
turbines are used where possible."*

It picks **as many test turbines as the site allows** while every one keeps three nearby
references, and while the test turbines occupy a fair share of front-row positions.

## 1. Describe the site

One row per turbine, as a table (CSV or a pandas DataFrame). Column names are matched
case-insensitively.

| Column | Required | Meaning |
|---|---|---|
| `latitude`, `longitude` | yes | WGS84 degrees |
| `name` | for turbines of the farm under design | Neighbouring turbines you can see in aerial imagery but cannot name may be left blank |
| `rotor_diameter_m` | no | A missing diameter takes the largest one given, which widens that turbine's wake sector (the worst case) |
| `wind_farm` | no | Which farm a turbine belongs to. Neighbouring farms' turbines are drawn in their own colour and only block wakes |

Include neighbouring farms' turbines: they decide which of your turbines are front row.

The same file can serve as the campaign declaration's `turbines.csv` (see
[running a campaign](running-a-campaign.md)); rows without a name are skipped there.

## 2. Say what each turbine may do

Every turbine of the farm under design holds at most one of these roles:

| Argument | The turbine is | Example reasons |
|---|---|---|
| `test_priority` | a test candidate, ranked: first is highest priority | Turbines with the most expected uplift, or a spread of expected uplifts |
| `reference_only` | never tested, but may be a reference | The upgrade cannot be fitted to it; the owner will not change it |
| `excluded` | neither tested nor a reference | Offline, part of another campaign, a known performance issue |

A turbine you do not list is a test candidate too. Unlisted candidates follow the listed ones,
chosen to keep references as near as possible, with `seed` settling any remaining choice. If you
have no way to rank turbines, pass nothing.

## 3. The rules

A set of test turbines complies when:

1. **Three references within reach.** Among each test turbine's 4 nearest reference-eligible
   turbines (`reference_pool_size`), at least 3 are not test turbines, and they lie within 20 rotor
   diameters of it (`max_reference_distance_d`). Its references are the 3 nearest of those.
2. **The nearest turbine is always a reference.** A test turbine's nearest neighbour is never
   another test turbine.
3. **A fair share of the front row.** A turbine is front row when at least 90° of wind directions
   (`front_row_min_clear_deg`) reach it with no other turbine upwind, by the IEC 61400-12-1
   disturbed-sector rule. If `F` of the `A` available turbines are front row and `n` are tested,
   the front-row test turbines number `n × F / A` rounded to the nearest whole number (either
   neighbour on an exact half). Too many or too few both fail. Excluded turbines do not count in
   `A` or `F`; reference-only turbines do.
4. **Roles are respected.** Excluded and reference-only turbines are never tested.

A turbine may be a reference for any number of test turbines.

## 4. Design

```python
import pandas as pd
from wind_up.campaign_design import design_campaign, write_design

layout = pd.read_csv("turbines.csv")
design = design_campaign(
    layout,
    wind_farm="My Farm",                 # needed only when the layout names several farms
    test_priority=["T07", "T12", "T03"],
    reference_only=["T17"],
    excluded=["T09"],
    seed=0,
)
write_design(design, out_dir="design")
```

wind-up first finds the most test turbines any compliant design allows (`n_test` asks for fewer).
It then walks `test_priority` and keeps each turbine for which a compliant design of that size
still exists, so a higher priority always wins over a lower one, but never at the cost of a test
slot. A turbine that gives way is listed with the reason.

The turbines you did not list then fill the remaining slots with the nearest references possible:
wind-up finds the smallest reference distance that still allows a design of that size alongside
the turbines already kept, and walks the unlisted turbines, in the order drawn from `seed`, under
that limit. `summary.yaml` reports it as `reference_limit_d`. A fully listed priority is never
overridden this way; to draw random designs, shuffle the whole priority list rather than rely on
`seed`.

## 5. Check a design you already have

```python
from wind_up.campaign_design import check_design, write_compliance

report = check_design(layout, test_turbines=["T02", "T04", "T05"], excluded=["T09"])
print(report.compliant)
for problem in report.problems:
    print(problem)
write_compliance(report, out_dir="check")
```

Non-compliance is reported, never raised, so a failing design can still be inspected.

## 6. Read the outputs

| File | What it holds |
|---|---|
| `compliance.csv` | One row per test turbine: its three references, their distances in metres and rotor diameters, how near each one ranks, which are front row |
| `summary.yaml` | Whether the design complies, any problems, the counts, the front-row share against its target, the most test turbines possible, the reference distance limit the unlisted turbines were chosen under |
| `turbines.csv` | Every farm turbine: its role, the test turbines it serves as a reference, its priority, and why it was or was not tested |
| `roles.yaml` | A `turbines:` block to paste into the campaign declaration. Every available non-test turbine is offered as a reference, because the analysis uses them all |
| `design_map.png` | The layout in metres east and north of the site's south-west corner |
| `design_map_latlon.png` | The same in latitude and longitude |

On the maps, test turbines are red and joined to their references (blue); other available
turbines are grey, reference-only turbines hollow, excluded turbines crosses, and front-row
turbines ringed in black.

## A worked example: Hill of Towie

The 21 turbines of Hill of Towie (82 m rotors, 4–5 rotor diameters apart), with T17 reference-only
and no priority, design as 10 test turbines. 14 of the 21 are front row, so the fair share for 10
test turbines is 6.67, and 7 of the 10 are front row. Every test turbine's references are among its
four nearest turbines, the furthest about 12 rotor diameters away.

## Limits

- Distance stands in for how well two turbines' power correlates, which is what the methodology
  ranks references by. On complex terrain, check the chosen references against your data.
- Front row is judged from geometry alone, not from the site's wind rose.
- Distances and bearings are computed on the WGS84 ellipsoid.
