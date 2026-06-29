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

## Symbol index

| Symbol | Column | Description |
|--------|--------|-------------|
| $i$ | — | Swing index |
| $j$ | — | Batter index |
| $m$ | — | Angular axis index: VAA, HAA, tilt |
| $b_i$, $s_i$ | `balls`, `strikes` | Count state |
| $x_i$, $z_i$ | `plate_x`, `plate_z` | Actual plate crossing (post-movement) |
| $\tilde{x}_i$, $\tilde{z}_i$ | `pc{ms}_x_proj`, `pc{ms}_z_proj` | Pre-commit projected plate location |
| $d_{x,i}$, $d_{z,i}$ | `pc{ms}_dev_x`, `pc{ms}_dev_z` | Post-commit movement: $x_i - \tilde{x}_i$ and $z_i - \tilde{z}_i$ |
| $\Delta_i^{(m)}$ | `{metric}_dev` | Angular swing deviation from Phase A |
| $\hat{\Delta}_i^{(m)}$ | `distortion_dev_{m}` | Movement-caused component of $\Delta_i^{(m)}$ |
| $a_{x,m}$, $a_{z,m}$ | — | Mediator model treatment coefficients for axis $m$ |
| $\rho_i$ | `angular_distortion_share` | Fraction of angular deviation caused by movement |
| $\sigma(\cdot)$ | — | Logistic sigmoid: $\sigma(x) = (1+e^{-x})^{-1}$ |
| $\kappa$ | — | `miss_rv_slope` $\approx -0.014$ runs/inch |
| $R_i^\text{take}$ | internal `take_xRV` | Expected run value of taking the pitch at projected location |
| $c_i$ | `decision_cost` | Opportunity cost of swinging vs. taking |

---

## Step 1 — Intended swing shape

The Bayesian LMM (Phase A) produces a posterior-mean prediction for each swing:

$$
\hat{y}_i = \mathbf{x}_i^\top \bar{\boldsymbol{\beta}} + \mathbf{z}_{\text{batter},i}^\top \bar{\mathbf{u}}_j
$$

$\mathbf{x}_i$ contains count, batter-frame location, timing, and platoon terms. $\mathbf{z}_{\text{batter},i} = [1,\, z(s_i)]$ captures the batter-specific intercept and count-pressure slope. Pitcher random effects are always zero at prediction time.

**Deviation residual** — the Phase B mediator:

$$
\Delta_i^{(m)} = y_i^\text{realized} - \hat{y}_i^\text{intended}
$$

Positive = batter executed above their intention on axis $m$.

---

## Step 2 — Composite xRV from outcome models

All three counterfactual evaluations use the same formula. Inputs to the three XGBoost models are drawn from feature vector $\mathbf{f}_i$:

$$
\mathbf{f}_i = \bigl[\Delta_i^{(\text{VAA})},\; \Delta_i^{(\text{HAA})},\; \Delta_i^{(\text{tilt})},\; x_i,\; z_i,\; b_i,\; s_i\bigr]
$$

$$
\begin{aligned}
P(\text{foul})_i  &= \bigl(1 - \hat{p}_{\text{BIP},i}\bigr) \cdot \hat{p}_{\text{foul}|\lnot\text{BIP},i} \\[4pt]
P(\text{whiff})_i &= \bigl(1 - \hat{p}_{\text{BIP},i}\bigr) \cdot \bigl(1 - \hat{p}_{\text{foul}|\lnot\text{BIP},i}\bigr) \\[6pt]
\widehat{\text{xRV}}_i &= \hat{p}_{\text{BIP},i} \cdot \hat{e}_{\text{xwOBA},i}
                           + P(\text{foul})_i \cdot r^\text{foul}_{b_i,s_i}
                           + P(\text{whiff})_i \cdot r^\text{whiff}_{b_i,s_i}
\end{aligned}
$$

The three counterfactual evaluations differ only in what is placed into $\mathbf{f}_i$:

| Evaluation | $\Delta_i^{(m)}$ values | Plate location |
|------------|------------------------|----------------|
| **realized** | actual deviations | actual $x_i$, $z_i$ |
| **spatial** | $0$ for all axes | actual $x_i$, $z_i$ |
| **intended** | $0$ for all axes | projected $\tilde{x}_i$, $\tilde{z}_i$ |

$\widehat{\text{xRV}}_i^\text{intended}$ is never written to the output parquet — it is computed internally and passed to `compute_decision_cost`, then dropped.

---

## Step 3 — disruption_tax and spatial_distortion_tax

**`disruption_tax`** — total run-value cost vs. the intended-swing-at-projected-location baseline:

$$
\tau_{\text{total},i} = \widehat{\text{xRV}}_i^\text{realized} - \widehat{\text{xRV}}_i^\text{intended}
$$

**`spatial_distortion_tax`** — cost of spatial displacement alone, with a perfect swing:

$$
\tau_{\text{spatial},i} = \widehat{\text{xRV}}_i^\text{spatial} - \widehat{\text{xRV}}_i^\text{intended}
$$

Internal intermediate (not written to parquet):

$$
\tau_{\text{angular},i} = \widehat{\text{xRV}}_i^\text{realized} - \widehat{\text{xRV}}_i^\text{spatial}
$$

Negative = pitcher advantage in all three.

---

## Step 4 — Angular distortion / selection attribution

**Movement-caused deviation** on each axis, from the mediator model treatment coefficients:

$$
\hat{\Delta}_i^{(m)} = a_{x,m}\, d_{x,i} + a_{z,m}\, d_{z,i}
$$

**`angular_distortion_share`** — fraction of total angular deviation explained by movement:

$$
\rho_i = \text{clip}\!\left(\frac{\displaystyle\sum_{m} \bigl(\hat{\Delta}_i^{(m)}\bigr)^2}{\displaystyle\sum_{m} \bigl(\Delta_i^{(m)}\bigr)^2},\; 0,\; 1\right)
$$

$\rho_i = \text{NaN}$ when $\sum_m (\Delta_i^{(m)})^2 < 10^{-8}$.

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

**Additive invariant**: $\tau_{\text{dist},i} + \tau_{\text{sel},i} = \tau_{\text{total},i}$ exactly.

---

## Step 5 — miss_distortion_tax

Movement-caused miss from the appropriate miss model's treatment coefficients:

$$
\mu_i^\text{mvt} = a_x\, d_{x,i} + a_z\, d_{z,i}
$$

**Whiff rows** $\tau_{\text{miss},i}$ (requires non-null `ball_bat_miss`):

$$
f_i = \text{clip}\!\left(\frac{\mu_i^\text{mvt}}{\mu_i^\text{whiff}},\; 0,\; 1\right), \qquad
\tau_{\text{miss},i} = f_i \cdot r^\text{whiff}_{b_i,s_i}
$$

**Contact rows** $\tau_{\text{miss},i}$:

$$
\tau_{\text{miss},i} = \mu_i^\text{mvt} \cdot \kappa
$$

where $\kappa \approx -0.014$ runs/inch (estimated from OLS of `delta_run_exp` on `contact_miss`). Negative result = pitcher advantage.

---

## Step 6 — decision_cost

**Parametric strike probability at projected location:**

$$
P_{\text{str},i} = \sigma\!\bigl(k(0.83-\tilde{x}_i)\bigr)\cdot\sigma\!\bigl(k(0.83+\tilde{x}_i)\bigr)\cdot\sigma\!\bigl(k(\tilde{z}_i-z_{\text{bot},i})\bigr)\cdot\sigma\!\bigl(k(z_{\text{top},i}-\tilde{z}_i)\bigr)
$$

$k = 8$, $\sigma(x) = (1+e^{-x})^{-1}$.

**Count transition run values** (from `count_values.csv`, RE24 framework):

$$
r^\text{cs}_{b,s} = \begin{cases} \text{ERV}(b,s+1) - \text{ERV}(b,s) & s < 2 \\ -\,\text{ERV}(b,2) & s = 2 \end{cases}
$$

$$
r^\text{ball}_{b,s} = \begin{cases} \text{ERV}(b+1,s) - \text{ERV}(b,s) & b < 3 \\ 0.33 - \text{ERV}(3,s) & b = 3 \end{cases}
$$

**Take value** $R_i^\text{take}$:

$$
R_i^\text{take} = P_{\text{str},i} \cdot r^\text{cs}_{b_i,s_i} + (1-P_{\text{str},i}) \cdot r^\text{ball}_{b_i,s_i}
$$

**`decision_cost`:**

$$
c_i = R_i^\text{take} - \widehat{\text{xRV}}_i^\text{intended}
$$

$c_i > 0$: taking was better. $c_i < 0$: swinging was correct.

---

## Step 7 — adjusted_disruption_tax

**`adjusted_disruption_tax`:**

$$
\tau_{\text{adj},i} = \tau_{\text{total},i} - \max(0,\; c_i)
$$

When $c_i \leq 0$: $\tau_{\text{adj},i} = \tau_{\text{total},i}$. When $c_i > 0$: the baseline shifts to $R_i^\text{take}$ and the full cost of swinging at a bad pitch is captured.

$\tau_{\text{adj},i} \leq \tau_{\text{total},i}$ always.

---

## Summary — all output columns

| Column | Sign convention | Formula |
|--------|----------------|---------|
| `disruption_tax` | Negative = pitcher advantage | $\widehat{\text{xRV}}^\text{realized} - \widehat{\text{xRV}}^\text{intended}$ |
| `spatial_distortion_tax` | Negative = pitcher advantage | $\widehat{\text{xRV}}^\text{spatial} - \widehat{\text{xRV}}^\text{intended}$ |
| `distortion_tax` | Negative = pitcher advantage | $\tau_\text{spatial} + \tau_\text{angular} \cdot \rho$ |
| `selection_tax` | Negative when deviation hurts | $\tau_\text{angular} \cdot (1 - \rho)$ |
| `distortion_share` | — | $\text{clip}(\tau_\text{dist} / \tau_\text{total},\, 0,\, 1)$ |
| `miss_distortion_tax` | Negative = pitcher advantage | Whiff: $f \cdot r^\text{whiff}$; Contact: $\mu^\text{mvt} \cdot \kappa$ |
| `decision_cost` | Positive = should have taken | $R^\text{take} - \widehat{\text{xRV}}^\text{intended}$ |
| `adjusted_disruption_tax` | Negative = pitcher advantage | $\tau_\text{total} - \max(0, c)$ |

**Additive invariants:**

- $\tau_\text{dist} + \tau_\text{sel} = \tau_\text{total}$
- $\tau_\text{adj} = \tau_\text{total}$ when $c \leq 0$
- $\tau_\text{adj} \leq \tau_\text{total}$ always

---

## Leaderboard aggregation (pitcher/batter CSVs)

Aggregated over swings where `distortion_tax` is non-null, minimum 50 swings per entity:

| Leaderboard column | Formula |
|--------------------|---------|
| `mean_disruption_tax` | $\bar{\tau}_\text{total}$ |
| `mean_distortion_tax` | $\bar{\tau}_\text{dist}$ |
| `mean_selection_tax` | $\bar{\tau}_\text{sel}$ |
| `mean_adjusted_disruption_tax` | $\bar{\tau}_\text{adj}$ |
| `mean_miss_distortion_tax` | $\bar{\tau}_\text{miss}$ |
| `mean_decision_cost` | $\bar{c}$ |
| `mean_distortion_share` | $\bar{\phi}$ |
| `n_swings` | count |

All means are per-swing (runs/swing), not per-plate-appearance. More negative `mean_distortion_tax` = pitcher induced more disruption per swing.
