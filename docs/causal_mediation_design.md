# Causal Mediation (Phase B) — Design & Implementation Notes

## Motivation

Phase A gives us `{metric}_dev` — how far each batter's swing deviated from their intention. Phase B answers: *how much did that deviation cost in run value, and how much of it was mechanically forced by post-commit movement?*

The core identification problem: post-commit movement affects run value through two distinct channels:

1. **Angular** — movement knocks the swing plane off-target, producing a deviation the outcome models can price directly
2. **Spatial** — movement shifts where the ball crosses the plate, so a perfectly-executed intended swing still misses because the ball ended up somewhere different than the batter projected

An angular-only model misses channel 2 entirely. A breaking ball dropping 6" after commit costs the batter real runs even if their swing plane is exactly what they intended — the target moved.

---

## Three-scenario counterfactual

The central design is evaluating `xRV` three times per swing to decompose disruption into spatial and angular components:

| Scenario | Angular deviations | Plate location | Purpose |
|----------|--------------------|----------------|---------|
| `xrv_realized` | actual `{metric}_dev` | actual `plate_x`, `plate_z` | What actually happened |
| `xrv_spatial` | zero (0°) | actual `plate_x`, `plate_z` | Cost of spatial displacement alone, with a perfect swing |
| `xrv_intended` | zero (0°) | projected `x_proj`, `z_proj` | Batter's information set — ball at pre-commit location, perfect swing |

```
disruption_tax     = xrv_realized − xrv_intended     [negative = pitcher advantage]
spatial_distortion = xrv_spatial  − xrv_intended
angular_disruption = xrv_realized − xrv_spatial
```

**Why this works**: `xrv_intended` represents a world where the ball stayed at the projected plate location and the batter executed their intended swing perfectly. Any gap from that baseline is disruption. Splitting it into spatial vs. angular lets us attribute disruption to its source.

**Option C was rejected**: an earlier approach added `pc_dev_x`/`pc_dev_z` directly as regressors in the outcome models to decompose location. This produced backward regression signs because `pc150_dev_z` always absorbs gravity (always negative), making the coefficient direction opposite to the causal direction. The predict-twice counterfactual (Option B) avoids this by substituting `x_proj`/`z_proj` directly into the plate-location slots the outcome models were trained on.

---

## 1. Mediator models

### Formula

One linear mixed-effects model per angular deviation axis m ∈ {`vert_attack_angle_dev`, `horz_attack_angle_dev`, `swing_path_tilt_dev`}:

```
m_dev_i = α₀
        + α₁ · pc_dev_x_i
        + α₂ · pc_dev_z_i
        + α₃ · x_proj_i
        + α₄ · z_proj_i
        + α₅ · release_speed_i
        + α₆ · balls_i
        + α₇ · strikes_i
        + α₈ · offset_y_ms_i
        + u₀ⱼ                   [per-batter random intercept]
        + εᵢ,  εᵢ ~ N(0, σ²)
```

### Variable definitions

| Symbol | Column | Units | Description |
|--------|--------|-------|-------------|
| `m_dev_i` | `{metric}_dev` | degrees | Angular swing deviation = realized − intended from Phase A |
| `pc_dev_x_i` | `pc{ms}_dev_x` | feet | Post-commit horizontal movement: actual `plate_x` − projected `x_proj` |
| `pc_dev_z_i` | `pc{ms}_dev_z` | feet | Post-commit vertical movement: actual `plate_z` − projected `z_proj` |
| `x_proj_i` | `pc{ms}_x_proj` | feet | Pre-commit projected plate x — where the ball was heading at commit time |
| `z_proj_i` | `pc{ms}_z_proj` | feet | Pre-commit projected plate z |
| `release_speed_i` | `release_speed` | mph | Pitch velocity at release |
| `balls_i` | `balls` | count 0–3 | Balls in count |
| `strikes_i` | `strikes` | count 0–2 | Strikes in count |
| `offset_y_ms_i` | `offset_y_ms` | ms | Contact timing offset |
| `u₀ⱼ` | — | — | Per-batter random intercept |

The pre-commit projected location (`x_proj`, `z_proj`) is a required control — without it, any correlation between typical pitch location and typical swing deviation would contaminate the treatment estimate. Conditioning on projection makes post-commit deviation exogenous to the swing decision (conditional ignorability).

Treatment coefficients **`a_x_m = α₁`** and **`a_z_m = α₂`** give the causal leverage: degrees of swing deviation per foot of post-commit horizontal or vertical movement, respectively.

---

## 2. Outcome models

### Feature set

All three outcome models share the same feature vector (column order is fixed):

```
OUTCOME_FEATURES = [
    vert_attack_angle_dev,    # degrees — swing plane deviation
    horz_attack_angle_dev,    # degrees — horizontal direction deviation
    swing_path_tilt_dev,      # degrees — barrel tilt deviation
    plate_x,                  # feet — actual plate x at crossing (or x_proj in counterfactual)
    plate_z,                  # feet — actual plate z at crossing (or z_proj in counterfactual)
    balls,                    # count
    strikes,                  # count
]
```

### Three XGBoost models

| Model | Target `y` | Training sample | Architecture |
|-------|-----------|-----------------|--------------|
| `bip_model` | `P(BIP_i)` = P(ball in play) | All swings | XGBoost classifier |
| `foul_model` | `P(foul_i \| not BIP)` = P(foul given swing not in play) | Non-BIP swings only | XGBoost classifier |
| `xwoba_model` | `E[xwOBA_i \| BIP]` = expected wOBA on balls in play | BIP only | XGBoost regressor |

Hyperparameters: `n_estimators=400`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`.

**Why XGBoost**: linear models extrapolate in the wrong direction at extreme plate locations (pitches 9" above the zone were assigned 38% BIP probability by logistic regression; XGBoost assigns ~6%, consistent with the empirical base rate in that sparse region). Tree models also handle extreme angular deviation outliers (HAA_dev = −99°) without inflating xRV unrealistically.

**Why foul and whiff are separate**: at two strikes, a foul keeps the at-bat alive (run-value delta = 0); a whiff ends it (delta = −ERV(balls, 2)). Conflating them biases xRV for high-disruption swings where foul rate is elevated.

### Composite xRV formula

```
P(foul_i)  = (1 − P(BIP_i))  ×  P(foul_i | not BIP_i)

P(whiff_i) = (1 − P(BIP_i))  ×  (1 − P(foul_i | not BIP_i))

xRV_i = P(BIP_i) × E[xwOBA_i | BIP_i]
      + P(foul_i) × foul_rv[balls_i, strikes_i]
      + P(whiff_i) × whiff_rv[balls_i, strikes_i]
```

| Symbol | Source | Description |
|--------|--------|-------------|
| `P(BIP_i)` | `bip_model.predict_proba()[:, 1]` | Probability swing results in a ball in play |
| `P(foul_i \| not BIP_i)` | `foul_model.predict_proba()[:, 1]` | Probability of foul given not BIP |
| `E[xwOBA_i \| BIP_i]` | `xwoba_model.predict()` | Expected xwOBA conditional on BIP |
| `foul_rv[b, s]` | `count_values.csv` | Run value of a foul at count (b, s); = 0 at s=2 |
| `whiff_rv[b, s]` | Empirical mean of `delta_run_exp` on whiffs by count | Run value of a whiff at count (b, s) |

---

## 3. Disruption tax decomposition

### Three xRV evaluations

The same `_xrv_from_shape()` function is called three times with different inputs:

| Call | `zero_angular` | `zero_spatial` | Plate location used | Angular deviations used |
|------|----------------|----------------|---------------------|------------------------|
| `xrv_realized` | False | False | `plate_x`, `plate_z` | actual `{metric}_dev` |
| `xrv_spatial` | True | False | `plate_x`, `plate_z` | 0° for all three axes |
| `xrv_intended` | True | True | `x_proj`, `z_proj` | 0° for all three axes |

### Primary decomposition

```
disruption_tax_i     = xrv_realized_i − xrv_intended_i
spatial_distortion_i = xrv_spatial_i  − xrv_intended_i
angular_disruption_i = xrv_realized_i − xrv_spatial_i
```

### Angular distortion attribution

For each angular axis m, the mediator model gives the portion of deviation caused by movement:

```
distortion_dev_m_i = a_x_m × pc_dev_x_i  +  a_z_m × pc_dev_z_i
selection_dev_m_i  = m_dev_i  −  distortion_dev_m_i
```

Squared-norm decomposition across all three angular axes:

```
angular_distortion_share_i = Σ_m(distortion_dev_m_i²) / Σ_m(m_dev_i²)
```

| Symbol | Description |
|--------|-------------|
| `a_x_m` | `mediator_models[m].params['pc{ms}_dev_x']` — causal leverage: degrees of deviation per foot of horizontal movement |
| `a_z_m` | `mediator_models[m].params['pc{ms}_dev_z']` — causal leverage per foot of vertical movement |
| `m_dev_i` | Total angular deviation on axis m for swing i |
| `angular_distortion_share_i` | Fraction of angular deviation explained by movement; clipped to [0, 1]; NaN when Σ_m(m_dev²) < 1e-8 |

The squared-norm ratio is used rather than a raw L2 ratio so that the result is always in [0, 1] regardless of whether distortion and selection components point in the same or opposite directions.

### Final tax split

```
distortion_tax_i = spatial_distortion_i  +  angular_disruption_i × angular_distortion_share_i

selection_tax_i  = angular_disruption_i  ×  (1 − angular_distortion_share_i)

distortion_share_i = distortion_tax_i / disruption_tax_i      [clipped to [0, 1]]
```

**Invariant**: `disruption_tax = distortion_tax + selection_tax` holds exactly for all non-NaN rows.

**Spatial disruption is 100% attributed to distortion** by construction — late movement is the only cause of the ball arriving somewhere different than projected.

---

## Validation controls

**Negative control**: four-seam fastballs with `pc150_dev_total < 1/12 ft` (near-straight) should show `disruption_tax ≈ 0`. Nonzero mean on this subset means the pre/post split is leaking selection into the treatment — the commit-time is effectively too early or the projection model is miscalibrated.

**Positive control**: pitch types with the most post-commit movement (ST, FS, SL) should show the largest distortion tax. If sweepers show less distortion than four-seamers, the mediator models are not identifying the causal path.

---

## Implementation

**`03_causal_models.py`**:
- `fit_mediator_models(df, commit_ms)` → dict of MixedLMResults per angular deviation axis
- `fit_outcome_models(df, commit_ms)` → `(bip_model, foul_model, xwoba_model, whiff_rv)`
- `_xrv_from_shape(df, ..., zero_angular, zero_spatial)` — evaluates one counterfactual scenario
- `disruption_tax_split(df, ...)` → df with all tax columns + internal `_xrv_intended`
- `indirect_effect(...)` → numerical finite-difference cross-check (central difference, eps=0.5°; replaces analytical product-of-coefficients which required statsmodels `.params` attributes)
- `negative_control_check`, `positive_control_check` — built-in validation

**`04_run_pipeline.py`** — Phase B runs after Phase A. Mediator and outcome models fit sequentially; disruption tax computed immediately after. `_xrv_intended` is passed to `compute_decision_cost` then dropped before save.

---

## Verification checklist

- [x] Negative control: FF with dev_total < 1" shows disruption_tax near 0
- [x] Positive control: ST/FS/SL ranked higher distortion tax than FF
- [x] `disruption_tax = distortion_tax + selection_tax` holds for all non-NaN rows
- [x] `spatial_distortion_tax` negative mean — ball displacement costs batters runs
- [x] `angular_distortion_share` ∈ [0, 1] for all rows
- [x] `adjusted_disruption_tax ≤ disruption_tax` for all rows with `decision_cost > 0`
- [x] `adjusted_disruption_tax == disruption_tax` for all rows with `decision_cost ≤ 0`
- [x] XGBoost P(BIP) at extreme z (>4 ft above zone) attenuates near empirical base rate (~6%), not 38% as logistic predicted
- [ ] Indirect effect (numerical finite-difference) directionally consistent with counterfactual tax
- [ ] Robustness grid: commit_ms 125/150/175/200 produce consistent pitcher/batter leaderboard rankings
- [ ] `distortion_share` by pitch type: sweepers > sinkers > four-seamers
