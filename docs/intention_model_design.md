# Intention Model (Phase A) — Design & Implementation Notes

## Motivation

To separate distortion from selection, we need a baseline: *what swing would this batter have produced if the pitch had gone exactly where they expected?* The deviation from that baseline — `realized − intended` — is the mediator that Phase B prices in run value.

Without a well-specified intention model, any swing deviation gets split arbitrarily between distortion and selection. The model must condition on everything the batter could plausibly have planned for (count, location, timing, handedness matchup) and nothing they couldn't (post-commit movement).

---

## What the model estimates

For each of five swing-shape responses, a Bayesian linear mixed-effects model predicts what the batter *intended* to do on each swing:

| Response | Role | Why it matters |
|----------|------|----------------|
| `vert_attack_angle` | Primary distortion signal | Most sensitive to late vertical movement (splitters, sinkers) |
| `horz_attack_angle` | Primary distortion signal | Most sensitive to late horizontal movement (sweepers) |
| `swing_path_tilt` | Primary distortion signal | Plane of the barrel through the zone |
| `bat_speed` | Secondary (effort) | Captures hold-back / half-swing under pressure |
| `swing_length` | Secondary (effort) | Shortening the swing is a common two-strike adjustment |

Each response is fit with a separate Gaussian LMM. A true joint model (correlated batter random effects across all five responses) would require brms/Stan; the separate-Bambi approximation loses only the cross-response batter RE covariance structure, which is recovered post-hoc if needed.

---

## Model formulas

### Angular responses (vert_attack_angle, horz_attack_angle, swing_path_tilt)

```
y_i = β₀
    + β₁ · z(balls_i)
    + β₂ · z(strikes_i)
    + β₃ · z(plate_x_bat_i)
    + β₄ · z(plate_z_i)
    + β₅ · z(plate_z_i²)
    + β₆ · z(offset_y_ms_i)
    + β₇ · offset_y_ms_missing_i
    + β₈ · pitcher_throws_L_i
    + β₉ · z(plate_x_bat_i) · pitcher_throws_L_i
    + u₀ⱼ + u₁ⱼ · z(strikes_i)   [batter random intercept + strikes slope]
    + v₀ₖ                          [pitcher random intercept, excluded at predict time]
    + εᵢ,  εᵢ ~ N(0, σ²)
```

### Effort responses (bat_speed, swing_length)

```
y_i = β₀
    + β₁ · z(balls_i)
    + β₂ · z(strikes_i)
    + β₃ · z(plate_x_bat_i)
    + β₄ · z(plate_z_i)
    + β₅ · pitcher_throws_L_i
    + u₀ⱼ + u₁ⱼ · z(strikes_i)
    + v₀ₖ
    + εᵢ,  εᵢ ~ N(0, σ²)
```

### Variable definitions

| Symbol | Column | Units | Description |
|--------|--------|-------|-------------|
| `y_i` | response | degrees or mph or ft | Swing-shape outcome for swing i |
| `z(x)` | — | dimensionless | Standardized: (x − μ_train) / σ_train |
| `balls_i` | `balls` | count 0–3 | Balls in count before this pitch |
| `strikes_i` | `strikes` | count 0–2 | Strikes in count before this pitch |
| `plate_x_bat_i` | derived | feet | Pitch location in batter's frame: `plate_x × (−1 if RHB, +1 if LHB)`. Inside = positive regardless of handedness |
| `plate_z_i` | `plate_z` | feet | Pitch height above ground at plate crossing |
| `plate_z_i²` | derived | feet² | Squared height — captures the nonlinear relationship between pitch altitude and swing plane |
| `offset_y_ms_i` | `offset_y_ms` | ms | Contact timing offset: negative = early, positive = late. Imputed to training mean when missing |
| `offset_y_ms_missing_i` | derived | 0/1 | 1 if `offset_y_ms` was missing for this swing; absorbs the mean shift for timing-missing rows |
| `pitcher_throws_L_i` | derived | 0/1 | 1 if pitcher is left-handed; absorbs spin-direction reversal across platoon matchups |
| `u₀ⱼ` | — | — | Per-batter random intercept for batter j |
| `u₁ⱼ` | — | — | Per-batter random slope on z(strikes) — captures individual count-pressure adjustment |
| `v₀ₖ` | — | — | Per-pitcher random intercept for pitcher k; always zero at prediction time |
| `εᵢ` | — | — | Residual error |

---

## Prediction formula

The prediction uses only posterior means (no MCMC uncertainty propagated downstream):

```
ŷ_i = X_i @ β̄  +  Z_batter_i @ ū_batter
```

| Symbol | Description |
|--------|-------------|
| `β̄` | Posterior mean of fixed-effect coefficients |
| `ū_batter` | Posterior mean of batter random effects (intercept + strikes slope) |
| `Z_batter_i` | Design row for batter j: `[1, z(strikes_i)]` |
| Pitcher RE | Always zero — intention baseline strips mound quality |
| Unseen batters | RE = 0 (population mean) |

Valid for Gaussian LMM because E[Xβ + Zu] = X·E[β] + Z·E[u].

---

## Deviation residual

```
{metric}_dev_i  =  realized_i  −  intended_i
```

| Symbol | Description |
|--------|-------------|
| `realized_i` | Observed swing-shape value (e.g. `vert_attack_angle`) |
| `intended_i` | Model prediction `ŷ_i` — what the batter planned |
| `{metric}_dev_i` | Phase B mediator: positive = batter executed above intention, negative = below |

---

## Formula design decisions

**Angular responses** condition on more predictors than effort responses because attack angle is more sensitive to location geometry:

- `plate_x_bat` — pitch location in the batter's own frame (inside = positive for both hands). A pitch on the inner half mechanically produces a different attack angle than the same pitch outside; conditioning on this removes it from the deviation residual.
- `plate_z` + `plate_z²` (quadratic) — batters tilt their swing plane to match pitch height. Without the quadratic term, the model mislabels appropriate low-ball plane adaptation as execution error.
- `offset_y_ms` — contact timing (early/on-time/late). Attack angle at contact depends on where in the swing arc the bat was measured; conditioning on timing removes the arc-sampling artifact from the deviation (Powers-Yurko effect).
- `pitcher_throws_L` + interaction with `plate_x_bat` — absorbs spin-direction reversal across platoon matchups. A left-hander's breaking ball breaks opposite to a right-hander's; conflating them would contaminate the location fixed effect.

**Missing timing** (`offset_y_ms`) is frequent on whiffs (~40%) and is not missing at random — it correlates with contact quality. Imputed to the training-data mean with a `offset_y_ms_missing` indicator so the timing coefficient is identified only from observed rows.

**Batter random effects**: intercept + strikes slope per response. Location slopes (`plate_x_bat`, `plate_z`) were tested as random effects but produce a degenerate LKJ correlation prior in NUTS geometry (max_treedepth warnings, R-hat > 1.01 for tail batters). Location effects stay as fixed effects; the random slope captures each batter's individual adjustment under count pressure, which is the core intention signal.

**Pitcher random effects**: intercept only, excluded from all predictions. The intention baseline is "what would this batter do against a neutral pitcher" — including pitcher REs would bake in mound-quality effects that belong in the outcome model, not the intention.

---

## Inference

Default: `method="vi"` (ADVI, 50k iterations, ~2 min). Only posterior means are used downstream — for this application, VI and MCMC produce functionally equivalent point estimates. Use `method="mcmc"` only if posterior uncertainty or chain diagnostics are needed.

Subsample default: `n_subsample=75_000`. The `(1 | pitcher_id)` term in Bambi's formulae backend allocates a dense `(n_rows × n_pitchers)` contrast matrix; the full ~763k dataset OOMs at ~4.9 GB. The subsample is sufficient for stable fixed-effect and batter RE estimates.

**Custom prediction** (`_posterior_mean_predict`): `model.predict()` materializes `(n_obs × n_groups × n_draws)` arrays and OOMs at 763k observations. The custom predictor applies posterior-mean fixed effects and batter REs as a direct linear combination: `ŷ = X @ β̄ + Z_batter @ ū_batter`. Unseen batters (not in the training subsample) receive RE = 0 (population mean).

---

## Outputs

For each swing:
- `intended_{metric}` — the posterior-mean predicted swing shape
- `{metric}_dev` = `realized − intended` — the Phase B mediator

Both are written to `models/intended_df.parquet` and joined back to the full swing frame in `04_run_pipeline.py`. Bambi model objects cannot be pickled on Python 3.14 (FrameLocalsProxy in formulae.Environment); only `idata` and `data` are persisted to `models/intention_result.joblib`.

---

## Implementation

**`02_intention_model.py`**:
- `fit(df, ...)` — prepares data and fits one Bambi model per response
- `predict_intended(result, swings)` — applies posterior-mean linear predictor to the full swing frame
- `swing_deviations(swings, intended_df)` — adds `{metric}_dev` columns
- `calibrate(swings_with_devs)` — within-batter bias/RMSE/correlation summary

**`04_run_pipeline.py`** — Phase A runs first; output cached to `models/intended_df.parquet`. Use `--skip-phase-a` to reload without refitting when iterating on Phase B.

---

## Verification checklist

- [ ] `intended_vert_attack_angle` distribution narrower than `vert_attack_angle` — conditioning removes structured variation
- [ ] Mean deviation near 0 for each response across all batters — the batter intercept absorbs systematic bias
- [ ] VAA deviation heatmap near-zero across plate zones — residuals by zone flag under-specified location fixed effects
- [ ] `calibrate()` output: mean |bias| < 0.5° for angular responses; corr > 0.3 for most batters
- [x] Custom predictor (`_posterior_mean_predict`) matches `model.predict()` on held-out subsample
- [x] Pipeline runs with `method="vi"` in < 5 minutes per response
