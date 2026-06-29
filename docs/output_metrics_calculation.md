# Output Metrics — How Each Column Is Calculated

This document traces every column in `results/xrv_causal.parquet` from raw model outputs to final values. It is a calculation reference, not a design doc — see the individual design docs for motivation and identification strategy.

---

## Pipeline overview

```
Phase A (02_intention_model.py)
  → intended_{metric}         (Bayesian LMM posterior mean prediction)
  → {metric}_dev              (realized − intended)

Phase B (03_causal_models.py)
  → Mediator models           (one LME per angular axis)
  → Outcome models            (3× XGBoost: BIP, foul, xwOBA)
  → xrv_realized, xrv_spatial, xrv_intended  (three counterfactual evaluations)
  → disruption_tax            (primary decomposition)
  → distortion_tax, selection_tax, spatial_distortion_tax, distortion_share
  → miss_distortion_tax       (physical bat-to-ball miss channel)
  → decision_cost             (swing vs. take opportunity cost)
  → adjusted_disruption_tax   (composite burden metric)
```

---

## Step 1 — Intended swing shape (Phase A outputs)

For each of the five swing-shape responses, the Bayesian LMM produces a posterior-mean prediction:

```
intended_{metric}_i = X_i @ β̄  +  Z_batter_i @ ū_batter
```

- `X_i` includes count, pitch location (batter frame), height quadratic, timing, and handedness interaction — everything the batter could have planned for
- `Z_batter_i` = `[1, z(strikes_i)]` for the batter-specific intercept and count-pressure slope
- Pitcher random effects always excluded at prediction time

**Deviation residual (Phase B input)**:

```
{metric}_dev_i = {metric}_i  −  intended_{metric}_i
```

This is the swing deviation used as the mediator: how much the batter's realized mechanics differed from their intention.

---

## Step 2 — Three counterfactual xRV evaluations

All three use the same composite xRV formula, evaluated at different input conditions:

```
P(foul_i)  = (1 − P(BIP_i))  ×  P(foul_i | not BIP_i)
P(whiff_i) = (1 − P(BIP_i))  ×  (1 − P(foul_i | not BIP_i))

xRV_i = P(BIP_i) × E[xwOBA_i | BIP_i]
      + P(foul_i) × foul_rv[balls_i, strikes_i]
      + P(whiff_i) × whiff_rv[balls_i, strikes_i]
```

The three evaluations differ only in what is substituted into the `OUTCOME_FEATURES` vector:

| Evaluation | `plate_x` / `plate_z` | `{metric}_dev` |
|------------|-----------------------|----------------|
| `xrv_realized` | actual `plate_x`, `plate_z` | actual deviations |
| `xrv_spatial` | actual `plate_x`, `plate_z` | 0° for all three axes |
| `xrv_intended` | projected `x_proj`, `z_proj` | 0° for all three axes |

`xrv_intended` is never written to the output parquet — it is computed internally in `disruption_tax_split` and passed to `compute_decision_cost`, then dropped.

---

## Step 3 — Primary disruption decomposition

### disruption_tax

```
disruption_tax_i = xrv_realized_i  −  xrv_intended_i
```

Total run-value cost vs. a world where the pitch stayed at its projected location and the batter executed their intended swing. Negative = pitcher advantage.

### spatial_distortion_tax

```
spatial_distortion_tax_i = xrv_spatial_i  −  xrv_intended_i
```

Run-value cost attributable solely to the spatial shift in where the ball crossed the plate, holding swing mechanics at intention (zero deviation). Captures the "target moved" channel even when the batter executes perfectly.

### angular_disruption (internal, not in parquet)

```
angular_disruption_i = xrv_realized_i  −  xrv_spatial_i
```

Run-value cost from swing mechanics deviating from intention, on top of spatial displacement. Positive when the deviation hurt the batter; negative when it helped.

---

## Step 4 — Angular distortion / selection split

The mediator model coefficients give the causal leverage of movement on each angular axis:

```
distortion_dev_m_i = a_x_m × pc_dev_x_i  +  a_z_m × pc_dev_z_i
```

where `a_x_m` and `a_z_m` are the treatment coefficients from the mediator model for axis m.

The share of angular deviation explained by movement across all three axes uses a squared-norm decomposition:

```
angular_distortion_share_i = ( Σ_m distortion_dev_m_i² )  /  ( Σ_m m_dev_i² )
```

Clipped to [0, 1]. NaN when the total angular deviation is near zero (Σ_m m_dev² < 1e-8).

### distortion_tax

```
distortion_tax_i = spatial_distortion_tax_i  +  angular_disruption_i × angular_distortion_share_i
```

Total run-value cost attributable to post-commit pitch movement: the full spatial channel plus the movement-caused fraction of the angular channel.

### selection_tax

```
selection_tax_i = angular_disruption_i  ×  (1 − angular_distortion_share_i)
```

The angular disruption the batter cannot blame on movement — their own swing decision component.

### distortion_share

```
distortion_share_i = distortion_tax_i  /  disruption_tax_i
```

Fraction of total disruption attributable to movement. Clipped to [0, 1]; NaN when `|disruption_tax| < 1e-8`.

**Additive invariant**: `distortion_tax + selection_tax = disruption_tax` holds exactly for all non-NaN rows.

---

## Step 5 — Physical miss channel (miss_distortion_tax)

An independent run-value estimate using directly measured bat-to-ball distance rather than the intention model baseline.

### Movement-caused miss

From the appropriate miss model's treatment coefficients:

```
movement_miss_i = a_x × pc_dev_x_i  +  a_z × pc_dev_z_i
```

`a_x`, `a_z` come from the whiff miss model (for whiff rows) or contact miss model (for contact rows).

### miss_distortion_tax calculation

**For whiff rows** (requires `ball_bat_miss` to be non-null):

```
movement_miss_frac_i = clip(movement_miss_i / ball_bat_miss_i,  0,  1)
miss_distortion_tax_i = movement_miss_frac_i × whiff_rv[balls_i, strikes_i]
```

`whiff_rv[b, s]` is the empirical mean `delta_run_exp` on whiffs at count (b, s).

**For contact rows**:

```
miss_distortion_tax_i = movement_miss_i × miss_rv_slope
```

`miss_rv_slope ≈ −0.014 runs/inch`, estimated from OLS of `delta_run_exp` on `contact_miss`.

Negative values mean movement caused worse contact for the batter.

---

## Step 6 — Decision cost (decision_cost)

### Strike probability at projected location

```
P_strike_i = σ(8·(0.83 − x_proj_i)) × σ(8·(0.83 + x_proj_i)) × σ(8·(z_proj_i − sz_bot_i)) × σ(8·(sz_top_i − z_proj_i))
```

`σ(x) = 1 / (1 + exp(−x))`. Evaluated at the pre-commit projected location — the information the batter had when deciding to swing.

### Count transition run values (from count_values.csv)

```
cs_rv[(b, s)]   = ERV(b, s+1) − ERV(b, s)     s < 2
                = 0  −  ERV(b, 2)              s = 2

ball_rv[(b, s)] = ERV(b+1, s) − ERV(b, s)     b < 3
                = 0.33 − ERV(3, s)             b = 3
```

### Take value

```
take_xRV_i = P_strike_i × cs_rv[balls_i, strikes_i]  +  (1 − P_strike_i) × ball_rv[balls_i, strikes_i]
```

### decision_cost

```
decision_cost_i = take_xRV_i  −  xrv_intended_i
```

Positive: taking was better than swinging. Negative: the swing was correct.

---

## Step 7 — Adjusted disruption tax (adjusted_disruption_tax)

```
adjusted_disruption_tax_i = disruption_tax_i  −  max(0,  decision_cost_i)
```

- `decision_cost ≤ 0` (swinging was correct): `adjusted_disruption_tax = disruption_tax`
- `decision_cost > 0` (should have taken): the baseline shifts to `take_xRV`; the full cost of swinging at a bad pitch is captured

This is the comprehensive per-swing burden metric: how much did the batter lose, versus the best action available given only pre-commit information?

---

## Summary table — all output columns in xrv_causal.parquet

| Column | Sign (negative = pitcher advantage) | Formula |
|--------|-------------------------------------|---------|
| `disruption_tax` | Negative | `xrv_realized − xrv_intended` |
| `spatial_distortion_tax` | Negative | `xrv_spatial − xrv_intended` |
| `distortion_tax` | Negative | `spatial_distortion_tax + angular_disruption × angular_distortion_share` |
| `selection_tax` | Negative when deviation hurts | `angular_disruption × (1 − angular_distortion_share)` |
| `distortion_share` | — | `distortion_tax / disruption_tax`, clipped [0, 1] |
| `miss_distortion_tax` | Negative | Whiff: `movement_miss_frac × whiff_rv`; Contact: `movement_miss × miss_rv_slope` |
| `decision_cost` | Positive = should have taken | `take_xRV − xrv_intended` |
| `adjusted_disruption_tax` | Negative | `disruption_tax − max(0, decision_cost)` |

**Additive invariants**:
- `disruption_tax = distortion_tax + selection_tax`
- `adjusted_disruption_tax = disruption_tax` when `decision_cost ≤ 0`
- `adjusted_disruption_tax ≤ disruption_tax` always

---

## Leaderboard aggregation (pitcher/batter CSVs)

Each leaderboard aggregates over swings where `distortion_tax` is non-null, minimum 50 swings:

```
mean_disruption_tax          = mean(disruption_tax)
mean_distortion_tax          = mean(distortion_tax)
mean_selection_tax           = mean(selection_tax)
mean_adjusted_disruption_tax = mean(adjusted_disruption_tax)
mean_miss_distortion_tax     = mean(miss_distortion_tax)
mean_decision_cost           = mean(decision_cost)
mean_distortion_share        = mean(distortion_share)
n_swings                     = count
```

All means are per-swing (runs/swing), not per-plate-appearance. More negative `mean_distortion_tax` = pitcher induced more disruption per swing.
