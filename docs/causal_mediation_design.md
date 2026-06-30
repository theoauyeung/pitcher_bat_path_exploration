# Causal Mediation (Phase B) — Design & Implementation Notes

## Motivation

Phase A gives us $\Delta_i^{(m)}$ (`{metric}_dev`) — how far each batter's swing deviated from their intention. Phase B answers: *how much did that deviation cost in run value, and how much of it was mechanically forced by post-commit movement?*

The core identification problem: post-commit movement affects run value through two distinct channels:

1. **Angular** — movement knocks the swing plane off-target, producing a deviation the outcome models can price directly
2. **Spatial** — movement shifts where the ball crosses the plate, so a perfectly-executed intended swing still misses because the ball ended up somewhere different than the batter projected

An angular-only model misses channel 2 entirely. A breaking ball dropping 6" after commit costs the batter real runs even if their swing plane is exactly what they intended — the target moved.

---

## Three-scenario counterfactual

The central design is evaluating $\widehat{\text{xRV}}$ three times per swing to decompose disruption into spatial and angular components:

| Scenario | Angular deviations $\Delta_i^{(m)}$ | Plate location | Purpose |
|----------|-------------------------------------|----------------|---------|
| `xrv_realized` | actual deviations | actual $x_i$, $z_i$ | What actually happened |
| `xrv_spatial` | $0°$ for all axes | actual $x_i$, $z_i$ | Cost of spatial displacement alone, with a perfect swing |
| `xrv_intended` | $0°$ for all axes | projected $\tilde{x}_i$, $\tilde{z}_i$ | Batter's information set — ball at pre-commit location, perfect swing |

$$
\begin{aligned}
\tau_\text{total}   &= \widehat{\text{xRV}}_i^\text{realized} - \widehat{\text{xRV}}_i^\text{intended} \\
\tau_\text{spatial} &= \widehat{\text{xRV}}_i^\text{spatial}  - \widehat{\text{xRV}}_i^\text{intended} \\
\tau_\text{angular} &= \widehat{\text{xRV}}_i^\text{realized} - \widehat{\text{xRV}}_i^\text{spatial}
\end{aligned}
$$

Negative values = pitcher advantage. $\tau_\text{total}$ is `disruption_tax`; $\tau_\text{spatial}$ is `spatial_distortion_tax`; $\tau_\text{angular}$ is an internal intermediate (not in the output parquet).

**Why this works**: $\widehat{\text{xRV}}_i^\text{intended}$ represents a world where the ball stayed at the projected plate location and the batter executed their intended swing perfectly. Any gap from that baseline is disruption. Splitting it into spatial vs. angular lets us attribute disruption to its source.

**Option C was rejected**: an earlier approach added $d_{x,i}$ / $d_{z,i}$ directly as regressors in the outcome models to decompose location. This produced backward regression signs because `pc150_dev_z` always absorbs gravity (always negative), making the coefficient direction opposite to the causal direction. The predict-twice counterfactual (Option B) avoids this by substituting $\tilde{x}_i$ / $\tilde{z}_i$ directly into the plate-location slots the outcome models were trained on.

---

## 1. Mediator models

### Symbol definitions

| Symbol | Column | Units | Description |
|--------|--------|-------|-------------|
| $\Delta_i^{(m)}$ | `{metric}_dev` | ° | Angular swing deviation = realized − intended (Phase A output) |
| $d_{x,i}$ | `pc{ms}_dev_x` | ft | Post-commit horizontal movement: actual $x_i$ − projected $\tilde{x}_i$ |
| $d_{z,i}$ | `pc{ms}_dev_z` | ft | Post-commit vertical movement: actual $z_i$ − projected $\tilde{z}_i$ |
| $\tilde{x}_i$ | `pc{ms}_x_proj` | ft | Pre-commit projected plate x — where the ball was heading at commit time |
| $\tilde{z}_i$ | `pc{ms}_z_proj` | ft | Pre-commit projected plate z |
| $v_i$ | `release_speed` | mph | Pitch velocity at release |
| $b_i$, $s_i$ | `balls`, `strikes` | count | Count state |
| $t_i$ | `offset_y_ms` | ms | Contact timing offset |
| $u_{0j}$ | — | — | Per-batter random intercept |

### Formula

One linear mixed-effects model per angular axis $m \in \{\text{VAA},\, \text{HAA},\, \text{tilt}\}$:

$$
\begin{aligned}
\Delta_i^{(m)} &= \alpha_0 \\
               &+ \alpha_1\, d_{x,i} + \alpha_2\, d_{z,i} \\
               &+ \alpha_3\, \tilde{x}_i + \alpha_4\, \tilde{z}_i \\
               &+ \alpha_5\, v_i + \alpha_6\, b_i + \alpha_7\, s_i + \alpha_8\, t_i \\
               &+ u_{0j} + \varepsilon_i
\end{aligned}
$$

The pre-commit projected location $(\tilde{x}_i, \tilde{z}_i)$ is a required control — without it, any correlation between typical pitch location and typical swing deviation would contaminate the treatment estimate. Conditioning on projection makes post-commit deviation exogenous to the swing decision (conditional ignorability).

The **treatment coefficients** are:

$$
a_{x,m} = \hat{\alpha}_1, \qquad a_{z,m} = \hat{\alpha}_2
$$

These give the causal leverage: degrees of swing deviation per foot of post-commit horizontal or vertical movement, respectively. They are used in Step 4 of the disruption decomposition.

---

## 2. Outcome models

### Feature vector

All three outcome models share the same ordered feature vector:

$$
\mathbf{f}_i = \bigl[\Delta_i^{(\text{VAA})},\; \Delta_i^{(\text{HAA})},\; \Delta_i^{(\text{tilt})},\; x_i,\; z_i,\; b_i,\; s_i\bigr]
$$

Column order is fixed by `OUTCOME_FEATURES` in `03_causal_models.py` and must match between fit and predict.

### Three XGBoost models

| Model | Target | Training sample |
|-------|--------|-----------------|
| `bip_model` | $\hat{p}_{\text{BIP},i} = P(\text{BIP}_i \mid \mathbf{f}_i)$ | All swings |
| `foul_model` | $\hat{p}_{\text{foul}|{\lnot\text{BIP}},i} = P(\text{foul}_i \mid \lnot\text{BIP}_i,\, \mathbf{f}_i)$ | Non-BIP swings only |
| `xwoba_model` | $\hat{e}_{\text{xwOBA},i} = \mathbb{E}[\text{xwOBA}_i \mid \text{BIP}_i,\, \mathbf{f}_i]$ | BIP only |

Hyperparameters: `n_estimators=400`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`.

**Why XGBoost**: linear models extrapolate in the wrong direction at extreme plate locations (pitches 9" above the zone were assigned 38% BIP probability by logistic regression; XGBoost assigns ~6%, consistent with the empirical base rate in that sparse region). Tree models also handle extreme angular deviation outliers without inflating $\widehat{\text{xRV}}$ unrealistically.

**Why foul and whiff are separate**: at two strikes, a foul keeps the at-bat alive (run-value delta = 0); a whiff ends it (delta $= -\text{ERV}(b, 2)$). Conflating them biases $\widehat{\text{xRV}}$ for high-disruption swings where foul rate is elevated.

### Composite xRV formula

$$
\begin{aligned}
P(\text{foul})_i  &= \bigl(1 - \hat{p}_{\text{BIP},i}\bigr) \cdot \hat{p}_{\text{foul}|\lnot\text{BIP},i} \\[4pt]
P(\text{whiff})_i &= \bigl(1 - \hat{p}_{\text{BIP},i}\bigr) \cdot \bigl(1 - \hat{p}_{\text{foul}|\lnot\text{BIP},i}\bigr) \\[6pt]
\widehat{\text{xRV}}_i &= \hat{p}_{\text{BIP},i} \cdot \hat{e}_{\text{xwOBA},i}
                           + P(\text{foul})_i \cdot r^{\text{foul}}_{b_i,s_i}
                           + P(\text{whiff})_i \cdot r^{\text{whiff}}_{b_i,s_i}
\end{aligned}
$$

| Symbol | Source | Description |
|--------|--------|-------------|
| $\hat{p}_{\text{BIP},i}$ | `bip_model.predict_proba()[:,1]` | P(ball in play) |
| $\hat{p}_{\text{foul}|\lnot\text{BIP},i}$ | `foul_model.predict_proba()[:,1]` | P(foul given not BIP) |
| $\hat{e}_{\text{xwOBA},i}$ | `xwoba_model.predict()` | Expected xwOBA conditional on BIP |
| $r^{\text{foul}}_{b,s}$ | `count_values.csv` | Run-value delta of a foul at count $(b, s)$; $= 0$ at $s=2$ |
| $r^{\text{whiff}}_{b,s}$ | Empirical mean `delta_run_exp` on whiffs by count | Run-value delta of a whiff at count $(b, s)$ |

---

## 3. Disruption tax decomposition

### Step 3a — Three xRV evaluations

The same composite formula above is evaluated three times. What differs is the input feature vector $\mathbf{f}_i$:

- **realized**: use actual $\Delta_i^{(m)}$, actual $x_i$ / $z_i$
- **spatial**: set all $\Delta_i^{(m)} = 0$, use actual $x_i$ / $z_i$
- **intended**: set all $\Delta_i^{(m)} = 0$, substitute $\tilde{x}_i$ / $\tilde{z}_i$ for $x_i$ / $z_i$

### Step 3b — Primary decomposition

**`disruption_tax`:**

$$
\tau_{\text{total},i} = \widehat{\text{xRV}}_i^\text{realized} - \widehat{\text{xRV}}_i^\text{intended}
$$

**`spatial_distortion_tax`:**

$$
\tau_{\text{spatial},i} = \widehat{\text{xRV}}_i^\text{spatial} - \widehat{\text{xRV}}_i^\text{intended}
$$

Internal intermediate (not written to parquet):

$$
\tau_{\text{angular},i} = \widehat{\text{xRV}}_i^\text{realized} - \widehat{\text{xRV}}_i^\text{spatial}
$$

### Step 3c — Angular distortion attribution

For each axis $m$, the mediator model gives the movement-caused portion of deviation (`distortion_dev_{m}`):

$$
\hat{\Delta}_i^{(m)} = a_{x,m}\, d_{x,i} + a_{z,m}\, d_{z,i}
$$

The fraction of total angular deviation explained by movement, using a squared-norm decomposition across all three axes (`angular_distortion_share`):

$$
\rho_i = \text{clip}\!\left(\frac{\displaystyle\sum_{m} \bigl(\hat{\Delta}_i^{(m)}\bigr)^2}{\displaystyle\sum_{m} \bigl(\Delta_i^{(m)}\bigr)^2},\; 0,\; 1\right)
$$

$\rho_i = \text{NaN}$ when $\sum_m (\Delta_i^{(m)})^2 < 10^{-8}$ (near-zero total deviation). The squared-norm ratio is used so that $\rho_i \in [0, 1]$ regardless of whether distortion and selection components point in the same or opposite directions.

### Step 3d — Final tax split

**`distortion_tax`:**

$$
\tau_{\text{dist},i} = \tau_{\text{spatial},i} + \tau_{\text{angular},i} \cdot \rho_i
$$

**`selection_tax`:**

$$
\tau_{\text{sel},i} = \tau_{\text{angular},i} \cdot (1 - \rho_i)
$$

**`distortion_share`:**

$$
\phi_i = \text{clip}\!\left(\frac{\tau_{\text{dist},i}}{\tau_{\text{total},i}},\; 0,\; 1\right)
$$

**Additive invariant**: $\tau_{\text{dist},i} + \tau_{\text{sel},i} = \tau_{\text{total},i}$ holds exactly for all non-NaN rows.

**Spatial disruption is 100% attributed to distortion** by construction — late movement is the only cause of the ball arriving somewhere different than projected.

---

## Validation controls

**Negative control**: four-seam fastballs with $d_{\text{total},i} < \tfrac{1}{12}$ ft (near-straight) should show $\tau_{\text{total},i} \approx 0$. Nonzero mean on this subset means the pre/post split is leaking selection into the treatment — the commit-time is effectively too early or the projection model is miscalibrated.

**Positive control**: pitch types with the most post-commit movement (ST, FS, SL) should show the largest $\tau_{\text{dist}}$. If sweepers show less distortion than four-seamers, the mediator models are not identifying the causal path.

---

## Implementation

**`03_causal_models.py`**:
- `fit_mediator_models(df, commit_ms)` → dict of MixedLMResults per angular deviation axis
- `fit_outcome_models(df, commit_ms)` → `(bip_model, foul_model, xwoba_model, whiff_rv)`
- `_xrv_from_shape(df, ..., zero_angular, zero_spatial)` — evaluates one counterfactual scenario
- `disruption_tax_split(df, ...)` → df with all tax columns + internal `_xrv_intended`
- `indirect_effect(...)` → numerical finite-difference cross-check (central difference, $\varepsilon = 0.5°$)
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
