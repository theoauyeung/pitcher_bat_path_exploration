# Physical Miss + Decision Cost — Design & Implementation Notes

## Motivation

`disruption_tax` is computed as `xRV(realized swing) − xRV(intended swing)` conditional on swinging. This creates two problems:

1. **Positive disruption_tax on whiffs** — When a batter adapts mechanics to a pitch outside the zone (e.g. pulling swing up toward a high fastball), the outcome models can rate that adaptation as *better* than the intention baseline, yielding positive disruption_tax even on strikeouts. This is a model artifact from the intention model's baseline being misspecified at extreme pitch locations.

2. **No swing-decision signal** — The metric is silent on whether the batter should have swung at all. A batter chasing a sweeper 6 inches off the plate might show zero disruption_tax if their mechanics were well-adapted to that location.

**Solution**: Add two new output columns without removing existing ones.

---

## New columns in `results/xrv_causal.parquet`

| Column | Meaning | Sign |
|--------|---------|------|
| `miss_distortion_tax` | Run-value cost of movement-caused increase in physical bat-to-ball miss | Negative = pitcher advantage |
| `decision_cost` | Opportunity cost of swinging vs. taking at the **projected** plate location | Positive = taking was better |
| `adjusted_disruption_tax` | Total batter burden vs. optimal action at projected location — `disruption_tax − max(0, decision_cost)` | Negative = pitcher advantage |

`disruption_tax`, `distortion_tax`, `selection_tax`, `spatial_distortion_tax`, `distortion_share` are **unchanged**.

`adjusted_disruption_tax` is additive: when swinging was optimal (`decision_cost ≤ 0`), it equals `disruption_tax`. When taking was better (`decision_cost > 0`), the baseline shifts to `take_xrv` and the full swing cost is captured. Population mean ≈ −0.012 runs vs. −0.002 for raw `disruption_tax`; 16.2% of swings have `decision_cost > 0`.

---

## `miss_distortion_tax`

Physically grounded alternative to `disruption_tax` that uses directly measured bat-to-ball distance rather than an intention model baseline. Immune to the positive-disruption artifact on whiffs.

**Two OLS miss models** (in `03_causal_models.fit_miss_models`):

- **Whiff model**: `ball_bat_miss ~ pc_dev_x + pc_dev_z + x_proj + z_proj + angular_devs + balls + strikes`
  - `ball_bat_miss` is directly measured bat-to-ball separation in inches (Hawk-Eye)
  - Fit on whiff rows only (~91% coverage, ~175k rows)

- **Contact model**: `contact_miss ~ same predictors`
  - `contact_miss = sqrt(offset_z_in² + offset_x_in²)` — geometric distance from bat sweet spot
  - Fit on contacts where `|offset_x_in| ≤ 20` (~89% coverage)

**Miss-to-xRV conversion**: a thin OLS `delta_run_exp ~ contact_miss + count_dummies` on contact rows gives the sensitivity `d(runs)/d(inch)` ≈ −0.014 runs/inch. This slopes negative — more off-center contact → fewer runs.

**Per-swing attribution**:

For whiffs (where physical miss is binary — any whiff is a whiff regardless of magnitude):
```
movement_miss = a_x × pc_dev_x + a_z × pc_dev_z
movement_miss_frac = clip(movement_miss / ball_bat_miss, 0, 1)
miss_distortion_tax = movement_miss_frac × whiff_rv[count]
```

For contacts (continuous — more off-center = worse outcome):
```
miss_distortion_tax = movement_miss × miss_rv_slope
```

**Caveats**:
- `miss_distortion_tax` for whiffs requires `ball_bat_miss`; NaN on the 9% of whiff rows where it's missing
- `miss_rv_slope` is estimated from a second OLS step, adding noise relative to a direct counterfactual

---

## `decision_cost`

Opportunity cost of swinging at the pitch as opposed to taking it, evaluated at the **projected** plate location (x_proj, z_proj) — the pre-commit location the batter's decision was based on — not the actual post-movement plate_x / plate_z.

```
take_xRV = P_strike(x_proj, z_proj) × called_strike_rv[count]
         + (1 − P_strike) × ball_rv[count]

decision_cost = take_xRV − xrv_intended
```

Where `xrv_intended` is the counterfactual swing value at projected location with zero angular deviations (already computed in `disruption_tax_split`).

**Strike probability**: uses a smooth parametric strike zone (logistic sigmoid on zone boundaries, k=8) rather than an empirical model from takes/called-strikes. An empirical model would be more accurate near zone edges but requires non-swing pitches not currently in `swings_precommit.parquet` (would need a separate DB pull).

**Count transitions from `count_values.csv`**:
- `called_strike_rv[(b, s)]` = ERV(b, s+1) − ERV(b, s) for s<2; 0 − ERV(b, 2) for s=2
- `ball_rv[(b, s)]` = ERV(b+1, s) − ERV(b, s) for b<3; 0.33 − ERV(3, s) for b=3 (walk approximation)

**Interpretation**:
- `decision_cost > 0`: taking was better than swinging (batter chased a bad ball or swung at a tunneled pitch that looked like a strike)
- `decision_cost < 0`: swinging was correct (batter attacked a hittable pitch at the projected location)
- Population mean ≈ −0.08 runs, which is expected — batters self-select into swinging at pitches where swinging is correct

**Caveats**:
- Positive `decision_cost` doesn't attribute the bad decision to movement — could be pitch-type deception, count-based gambling, etc. Movement attribution would require the full Option 5 causal graph (out of scope).
- Parametric strike zone is less accurate at edges than an empirical model from non-swing data.

---

## Implementation

**`03_causal_models.py`** — new functions in sections 6 and 7:
- `fit_miss_models(df, commit_ms=150)` → `(whiff_miss_model, contact_miss_model, miss_rv_slope)`
- `compute_miss_distortion_tax(df, whiff_miss_model, contact_miss_model, miss_rv_slope, whiff_rv, commit_ms=150)` → Series
- `compute_decision_cost(df, count_values_path, commit_ms=150, xrv_intended=None)` → Series

`disruption_tax_split` exposes `_xrv_intended` as an internal column (stripped before save) so `compute_decision_cost` doesn't need to recompute it.

**`04_run_pipeline.py`** — wired after Stage 4 (outcome models) and Stage 5 (disruption tax):
- Miss models fit in parallel with outcome models
- `miss_distortion_tax` and `decision_cost` computed immediately after disruption tax split
- Both columns added to `tax_cols`, CSV aggregations (`mean_miss_distortion_tax`, `mean_decision_cost`), and `causal_models.joblib`

---

## Verification checklist

- [x] Pipeline runs with `--skip-phase-a` and produces both columns
- [x] `miss_distortion_tax` negative mean (−0.003 runs) — movement costs batters runs
- [x] `miss_rv_slope` negative (−0.014 runs/inch) — more off-center = fewer runs
- [x] `decision_cost` negative mean (−0.081 runs) — batters swing at good pitches on average
- [ ] `miss_distortion_tax` is larger (more negative) for high-movement pitch types (ST, SL, FS) vs. FF
- [ ] `decision_cost` is positive for pitches tunneled away from the projected strike zone
- [ ] Split-half reliability check on `miss_distortion_tax`
