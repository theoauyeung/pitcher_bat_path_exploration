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
| `adjusted_disruption_tax` | Total batter burden vs. optimal action at projected location | Negative = pitcher advantage |

`disruption_tax`, `distortion_tax`, `selection_tax`, `spatial_distortion_tax`, `distortion_share` are **unchanged**.

---

## 1. Miss models

### Whiff miss model

Predicts total bat-to-ball separation (inches) on whiff swings from post-commit movement and other factors:

```
ball_bat_miss_i = α₀
               + α₁ · pc_dev_x_i
               + α₂ · pc_dev_z_i
               + α₃ · x_proj_i
               + α₄ · z_proj_i
               + α₅ · vert_attack_angle_dev_i
               + α₆ · horz_attack_angle_dev_i
               + α₇ · swing_path_tilt_dev_i
               + α₈ · balls_i
               + α₉ · strikes_i
               + εᵢ
```

Fit on whiff rows where `ball_bat_miss` is non-null (~91% coverage, ~175k rows). OLS with HC1 robust standard errors.

### Contact miss model

Predicts geometric off-center contact distance (inches) on contacts:

```
contact_miss_i = sqrt(offset_z_in_i²  +  offset_x_in_i²)

contact_miss_i = α₀
               + α₁ · pc_dev_x_i
               + α₂ · pc_dev_z_i
               + α₃ · x_proj_i
               + α₄ · z_proj_i
               + α₅ · vert_attack_angle_dev_i
               + α₆ · horz_attack_angle_dev_i
               + α₇ · swing_path_tilt_dev_i
               + α₈ · balls_i
               + α₉ · strikes_i
               + εᵢ
```

Fit on contact rows where `|offset_x_in| ≤ 20` (~89% coverage). OLS with HC1 robust standard errors.

### Variable definitions (both miss models)

| Symbol | Column | Units | Description |
|--------|--------|-------|-------------|
| `ball_bat_miss_i` | `ball_bat_miss` | inches | Bat-to-ball separation at contact (Hawk-Eye measured) |
| `contact_miss_i` | derived | inches | Euclidean distance from bat sweet spot to ball |
| `offset_z_in_i` | `offset_z_in` | inches | Vertical offset from bat center at contact |
| `offset_x_in_i` | `offset_x_in` | inches | Horizontal offset from bat center at contact |
| `pc_dev_x_i` | `pc{ms}_dev_x` | feet | Post-commit horizontal movement |
| `pc_dev_z_i` | `pc{ms}_dev_z` | feet | Post-commit vertical movement |
| `x_proj_i` | `pc{ms}_x_proj` | feet | Pre-commit projected plate x |
| `z_proj_i` | `pc{ms}_z_proj` | feet | Pre-commit projected plate z |
| `{metric}_dev_i` | `{metric}_dev` | degrees | Angular swing deviation from Phase A |
| `balls_i`, `strikes_i` | `balls`, `strikes` | count | Count state |

### Miss-to-xRV slope

A second OLS step converts contact miss to run value:

```
delta_run_exp_i = γ₀  +  γ₁ · contact_miss_i  +  Σ_c γ_c · I(count_i = c)  +  εᵢ

miss_rv_slope = γ₁   (≈ −0.014 runs/inch)
```

`miss_rv_slope` is negative — more off-center contact produces fewer runs for the batter.

---

## 2. Per-swing miss distortion tax

### Whiff rows

Movement-caused miss is the projection of post-commit movement onto the miss dimension via the whiff model coefficients:

```
movement_miss_i = a_x · pc_dev_x_i  +  a_z · pc_dev_z_i

movement_miss_frac_i = clip(movement_miss_i / ball_bat_miss_i,  0,  1)

miss_distortion_tax_i = movement_miss_frac_i × whiff_rv[balls_i, strikes_i]
```

| Symbol | Description |
|--------|-------------|
| `a_x` | `whiff_miss_model.params['pc{ms}_dev_x']` — inches of bat-to-ball miss per foot of horizontal movement |
| `a_z` | `whiff_miss_model.params['pc{ms}_dev_z']` — inches of bat-to-ball miss per foot of vertical movement |
| `movement_miss_i` | Portion of bat-to-ball miss attributable to post-commit movement (inches) |
| `movement_miss_frac_i` | Fraction of total measured miss caused by movement; clipped to [0, 1] |
| `whiff_rv[b, s]` | Count-adjusted run value of a whiff at (balls=b, strikes=s) |

NaN when `ball_bat_miss` is missing (~9% of whiff rows).

### Contact rows

For contacts, the miss is continuous and priced directly through the run-value slope:

```
movement_miss_i = a_x · pc_dev_x_i  +  a_z · pc_dev_z_i

miss_distortion_tax_i = movement_miss_i × miss_rv_slope
```

| Symbol | Description |
|--------|-------------|
| `a_x` | `contact_miss_model.params['pc{ms}_dev_x']` |
| `a_z` | `contact_miss_model.params['pc{ms}_dev_z']` |
| `movement_miss_i` | Movement-caused increase in contact distance (inches) |
| `miss_rv_slope` | ≈ −0.014 runs/inch; negative means more miss → fewer runs |

---

## 3. Decision cost

### Strike probability (parametric strike zone)

```
P_strike_i = σ(k·(0.83 − x_proj_i)) × σ(k·(0.83 + x_proj_i)) × σ(k·(z_proj_i − sz_bot_i)) × σ(k·(sz_top_i − z_proj_i))
```

| Symbol | Description |
|--------|-------------|
| `σ(·)` | Logistic sigmoid: σ(x) = 1 / (1 + exp(−x)) |
| `k = 8.0` | Sharpness parameter — gives ~5%→95% transition over ≈0.3 ft around zone boundary |
| `x_proj_i` | Pre-commit projected plate x (feet) — the location the batter's decision was based on |
| `z_proj_i` | Pre-commit projected plate z (feet) |
| `sz_top_i` | Top of the strike zone for this batter (feet); default 3.5 if missing |
| `sz_bot_i` | Bottom of the strike zone (feet); default 1.5 if missing |
| `0.83` | Half-width of the strike zone in feet (~9.95 inches, accounting for ball diameter) |

### Count transition run values

```
cs_rv[(b, s)]   = ERV(b, s+1) − ERV(b, s)     for s < 2   [cost of falling behind in count]
                = 0  −  ERV(b, 2)              for s = 2   [cost of strikeout from two-strike count]

ball_rv[(b, s)] = ERV(b+1, s) − ERV(b, s)     for b < 3
                = WALK_RV  −  ERV(3, s)        for b = 3   [walk approximation]
```

| Symbol | Description |
|--------|-------------|
| `ERV(b, s)` | Expected run value of count (balls=b, strikes=s), from `count_values.csv` (RE24 framework) |
| `cs_rv[(b,s)]` | Run value change of a called strike at count (b, s) |
| `ball_rv[(b,s)]` | Run value change of a ball at count (b, s) |
| `WALK_RV = 0.33` | Approximate run value of a walk |

### Take value

```
take_xRV_i = P_strike_i × cs_rv[balls_i, strikes_i]  +  (1 − P_strike_i) × ball_rv[balls_i, strikes_i]
```

Evaluated at the **projected** plate location (`x_proj`, `z_proj`) — the pre-commit location the batter's decision was based on, not the actual post-movement plate crossing.

### Decision cost formula

```
decision_cost_i = take_xRV_i  −  xrv_intended_i
```

| Symbol | Description |
|--------|-------------|
| `take_xRV_i` | Expected run value of taking the pitch at its projected location |
| `xrv_intended_i` | Expected run value of swinging with intended mechanics at projected location (from `disruption_tax_split`; dropped from final parquet output) |
| `decision_cost_i` | Positive = taking was better; negative = swinging was correct |

**Interpretation**:
- `decision_cost > 0`: batter chased a bad ball or swung at a tunneled pitch that looked like a strike
- `decision_cost < 0`: batter attacked a hittable pitch at the projected location (self-selected correctly)
- Population mean ≈ −0.08 runs — batters self-select into swinging at pitches where swinging is correct

---

## 4. Adjusted disruption tax

```
adjusted_disruption_tax_i = disruption_tax_i  −  max(0,  decision_cost_i)
```

| Symbol | Description |
|--------|-------------|
| `disruption_tax_i` | xRV cost of post-commit disruption vs. intended swing at projected location |
| `decision_cost_i` | Opportunity cost of the swing decision itself |
| `adjusted_disruption_tax_i` | Total batter burden vs. the optimal available action at commit time |

**Properties**:
- When `decision_cost ≤ 0`: swinging was correct → `adjusted_disruption_tax = disruption_tax`
- When `decision_cost > 0`: taking was better → baseline shifts to `take_xRV`; full swing cost is captured
- Population mean ≈ −0.012 runs vs. −0.002 for raw `disruption_tax`; 16.2% of swings have `decision_cost > 0`
- Additive: `adjusted_disruption_tax ≤ disruption_tax` always (equality when swinging was optimal)

---

## Caveats

- `miss_distortion_tax` for whiffs requires `ball_bat_miss`; NaN on ~9% of whiff rows where Hawk-Eye data is missing
- `miss_rv_slope` is estimated from a second OLS step, adding noise relative to a direct counterfactual
- Positive `decision_cost` doesn't attribute the bad decision to movement — could be pitch-type deception, count-based gambling, etc. Movement attribution would require a fuller causal graph (out of scope)
- Parametric strike zone is less accurate at edges than an empirical model from non-swing data

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
