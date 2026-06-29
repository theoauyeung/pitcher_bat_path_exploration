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

| Scenario | Swing angles | Plate location | Purpose |
|----------|-------------|----------------|---------|
| `xrv_realized` | actual deviations | actual (post-movement) | what actually happened |
| `xrv_spatial` | zero deviations | actual (post-movement) | cost of location shift alone, perfect swing |
| `xrv_intended` | zero deviations | projected (pre-movement) | batter's information set, perfect swing |

```
disruption_tax     = xrv_realized − xrv_intended  [negative = pitcher advantage]
spatial_distortion = xrv_spatial  − xrv_intended
angular_disruption = xrv_realized − xrv_spatial
```

**Why this works**: `xrv_intended` represents a world where the ball stayed at the projected plate location and the batter executed their intended swing perfectly. Any gap from that baseline is disruption. Splitting it into spatial vs. angular lets us attribute disruption to its source.

**Option C was rejected**: an earlier approach added `pc_dev_x`/`pc_dev_z` directly as regressors in the outcome models to decompose location. This produced backward regression signs because `pc150_dev_z` always absorbs gravity (always negative), making the coefficient direction opposite to the causal direction. The predict-twice counterfactual (Option B) avoids this by substituting `x_proj`/`z_proj` directly into the plate-location slots the outcome models were trained on.

---

## Mediator models

One linear mixed-effects model per angular deviation axis:

```
{metric}_dev ~ pc{ms}_dev_x + pc{ms}_dev_z     ← treatment (post-commit movement)
             + pc{ms}_x_proj + pc{ms}_z_proj    ← pre-commit projected location (control)
             + release_speed + balls + strikes + offset_y_ms
             + (1 | batter_id)
```

The pre-commit projected location is a required control — without it, any correlation between typical pitch location and typical swing deviation would contaminate the treatment estimate. Conditioning on projection makes post-commit deviation exogenous to the swing decision (conditional ignorability).

Treatment coefficients `a_x`, `a_z` per axis give the causal leverage: degrees of swing deviation caused by one foot of post-commit horizontal or vertical movement.

Minimum sample thresholds: batters with < 20 swings and pitchers with < 10 swings are excluded. Pitcher ID is not in the model (crossed random effects add cost with minimal benefit — the pre-commit control already absorbs most pitch-quality variation).

---

## Outcome models (three channels)

Outcome models use **actual** `plate_x`/`plate_z`. Spatial disruption is priced through the counterfactual (substituting `x_proj`/`z_proj` at predict time), not through additional regressors.

| Model | Target | Sample | Method |
|-------|--------|--------|--------|
| `bip_model` | P(ball in play) | all swings | XGBoost classifier |
| `foul_model` | P(foul \| not BIP) | non-BIP swings only | XGBoost classifier |
| `xwoba_model` | E[xwOBA \| BIP] | BIP only | XGBoost regressor |

**Why XGBoost, not logistic/OLS**: linear models extrapolate in the wrong direction at extreme plate locations (pitches 9" above the zone were assigned 38% BIP probability by logistic regression; XGBoost assigns ~6%, consistent with the empirical base rate in that sparse region). Tree models also handle extreme angular deviation outliers (HAA_dev = −99°) without inflating e_xwoba unrealistically. All three models share the same feature set: `ANGULAR_DEVS + ["plate_x", "plate_z", "balls", "strikes"]` — column order is fixed by `OUTCOME_FEATURES` in `03_causal_models.py` and must match between fit and predict.

**Why foul and whiff are separate**: at two strikes, a foul keeps the at-bat alive (run-value delta = 0); a whiff ends it (delta = −ERV(balls, 2)). Conflating them biases xRV for high-disruption swings where foul rate is elevated — the model would understate the cost of a whiff relative to a foul at the same disruption level.

**Composite xRV**:
```
P(foul)  = (1 − P(BIP)) × P(foul | not BIP)
P(whiff) = (1 − P(BIP)) × (1 − P(foul | not BIP))
xRV = P(BIP) × E[xwOBA|BIP]  +  P(foul) × foul_rv[count]  +  P(whiff) × whiff_rv[count]
```

Count-transition run values come from `count_values.csv` (RE24 framework). Foul at (b, s=2) → delta = 0; whiff at (b, s=2) → delta = −ERV(b, 2).

---

## Distortion / selection attribution

The angular disruption is further split by how much was mechanically caused by movement vs. the batter's own decision. For each swing, the mediator model provides the movement-caused component of each deviation:

```
distortion_dev_m = a_x_m × pc_dev_x + a_z_m × pc_dev_z
selection_dev_m  = {metric}_dev_m − distortion_dev_m
```

Attribution uses squared-norm decomposition across the three angular axes:
```
angular_distortion_share = ||distortion_dev||² / ||total_dev||²   (clipped to [0,1])
```

The raw L2-norm ratio is wrong when distortion and selection components point in opposite directions (e.g. movement pushed VAA down but batter also pulled it down intentionally). Squared-norm gives a clean [0,1] proportion regardless of sign alignment.

Spatial disruption is 100% attributed to distortion by construction — late movement is the only cause of the ball arriving somewhere different than projected.

```
distortion_tax = spatial_distortion + angular_disruption × angular_distortion_share
selection_tax  = angular_disruption × (1 − angular_distortion_share)
distortion_share = distortion_tax / disruption_tax   (clipped to [0,1])
```

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
