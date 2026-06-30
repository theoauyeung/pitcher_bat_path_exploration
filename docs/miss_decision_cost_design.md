# Physical Miss + Decision Cost — Design & Implementation Notes

## Motivation

`disruption_tax` is computed as $\widehat{\text{xRV}}(\text{realized}) - \widehat{\text{xRV}}(\text{intended})$ conditional on swinging. This creates two problems:

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

### Symbol definitions

| Symbol | Column | Units | Description |
|--------|--------|-------|-------------|
| $\mu_i^\text{whiff}$ | `ball_bat_miss` | in | Bat-to-ball separation at contact (Hawk-Eye measured); whiff rows only |
| $o_{z,i}$ | `offset_z_in` | in | Vertical bat-offset from sweet spot at contact |
| $o_{x,i}$ | `offset_x_in` | in | Horizontal bat-offset from sweet spot at contact |
| $\mu_i^\text{contact}$ | derived | in | Geometric off-center distance: $\sqrt{o_{z,i}^2 + o_{x,i}^2}$ |
| $d_{x,i}$ | `pc{ms}_dev_x` | ft | Post-commit horizontal movement |
| $d_{z,i}$ | `pc{ms}_dev_z` | ft | Post-commit vertical movement |
| $\tilde{x}_i$, $\tilde{z}_i$ | `pc{ms}_x_proj`, `pc{ms}_z_proj` | ft | Pre-commit projected plate location |
| $\Delta_i^{(m)}$ | `{metric}_dev` | ° | Angular swing deviations from Phase A |
| $b_i$, $s_i$ | `balls`, `strikes` | count | Count state |
| $\Delta r_i$ | `delta_run_exp` | runs | Change in run expectancy for this swing outcome |
| $R_i^\text{take}$ | internal | runs | Expected run value of taking the pitch at projected location |
| $\tau_{\text{miss},i}$ | `miss_distortion_tax` | runs | Run-value cost of movement-caused miss |

### Whiff miss model

Predicts total bat-to-ball separation on whiff rows. Fit on swings with non-null `ball_bat_miss` (~91% of whiffs, ~175k rows). OLS with HC1 robust standard errors.

$$
\begin{aligned}
\mu_i^\text{whiff} &= \alpha_0 \\
                   &+ \alpha_1\, d_{x,i} + \alpha_2\, d_{z,i} \\
                   &+ \alpha_3\, \tilde{x}_i + \alpha_4\, \tilde{z}_i \\
                   &+ \alpha_5\, \Delta_i^{(\text{VAA})} + \alpha_6\, \Delta_i^{(\text{HAA})} + \alpha_7\, \Delta_i^{(\text{tilt})} \\
                   &+ \alpha_8\, b_i + \alpha_9\, s_i + \varepsilon_i
\end{aligned}
$$

### Contact miss model

Predicts geometric off-center contact distance on contact rows where $|o_{x,i}| \leq 20$. Same right-hand side as the whiff model. OLS with HC1 robust standard errors.

$$
\mu_i^\text{contact} = \sqrt{o_{z,i}^2 + o_{x,i}^2}
$$

$$
\begin{aligned}
\mu_i^\text{contact} &= \alpha_0 \\
                     &+ \alpha_1\, d_{x,i} + \alpha_2\, d_{z,i} \\
                     &+ \alpha_3\, \tilde{x}_i + \alpha_4\, \tilde{z}_i \\
                     &+ \alpha_5\, \Delta_i^{(\text{VAA})} + \alpha_6\, \Delta_i^{(\text{HAA})} + \alpha_7\, \Delta_i^{(\text{tilt})} \\
                     &+ \alpha_8\, b_i + \alpha_9\, s_i + \varepsilon_i
\end{aligned}
$$

### Miss-to-xRV slope

A second OLS step on contact rows converts contact miss distance into run-value units:

$$
\Delta r_i = \gamma_0 + \gamma_1\, \mu_i^\text{contact} + \sum_c \gamma_c\, \mathbf{1}[\text{count}_i = c] + \varepsilon_i
$$

$$
\kappa = \hat{\gamma}_1 \approx -0.014 \text{ runs/inch}
$$

$\kappa < 0$: more off-center contact → fewer runs for the batter.

---

## 2. Per-swing miss distortion tax

### Movement-caused miss

For both miss model types, the movement-caused component is the inner product of post-commit movement with the model's treatment coefficients:

$$
\mu_i^\text{mvt} = a_{x}\, d_{x,i} + a_{z}\, d_{z,i}
$$

where $a_x$ and $a_z$ come from the whiff miss model (whiff rows) or contact miss model (contact rows).

### Whiff rows

$$
f_i = \text{clip}\!\left(\frac{\mu_i^\text{mvt}}{\mu_i^\text{whiff}},\; 0,\; 1\right)
$$

$$
\tau_{\text{miss},i} = f_i \cdot r^\text{whiff}_{b_i, s_i}
$$

| Symbol | Description |
|--------|-------------|
| $f_i$ | Fraction of total bat-to-ball miss attributable to post-commit movement; clipped to $[0,1]$ |
| $r^\text{whiff}_{b,s}$ | Count-adjusted run value of a whiff at count $(b, s)$ |

NaN when `ball_bat_miss` is missing (~9% of whiff rows).

### Contact rows

$$
\tau_{\text{miss},i} = \mu_i^\text{mvt} \cdot \kappa
$$

$\kappa \approx -0.014$ runs/inch, so positive $\mu_i^\text{mvt}$ (movement increased miss) → negative tax (pitcher advantage).

---

## 3. Decision cost

### Strike probability at projected location

$$
P_{\text{str},i} = \sigma\!\bigl(k(0.83 - \tilde{x}_i)\bigr) \cdot \sigma\!\bigl(k(0.83 + \tilde{x}_i)\bigr) \cdot \sigma\!\bigl(k(\tilde{z}_i - z_{\text{bot},i})\bigr) \cdot \sigma\!\bigl(k(z_{\text{top},i} - \tilde{z}_i)\bigr)
$$

where $\sigma(x) = (1 + e^{-x})^{-1}$ is the logistic sigmoid and $k = 8$.

| Symbol | Column | Description |
|--------|--------|-------------|
| $\tilde{x}_i$ | `pc{ms}_x_proj` | Pre-commit projected plate x (ft) |
| $\tilde{z}_i$ | `pc{ms}_z_proj` | Pre-commit projected plate z (ft) |
| $z_{\text{top},i}$ | `sz_top` | Top of batter's strike zone (ft); default 3.5 if missing |
| $z_{\text{bot},i}$ | `sz_bot` | Bottom of batter's strike zone (ft); default 1.5 if missing |
| $0.83$ | — | Half-width of the strike zone in feet (~9.95 in, accounting for ball diameter) |
| $k = 8$ | — | Sharpness; ~5%→95% transition over ≈0.3 ft around zone boundary |

Evaluated at the **projected** location $(\tilde{x}_i, \tilde{z}_i)$ — the pre-commit location the batter's decision was based on, not the actual post-movement plate crossing.

### Count transition run values

$$
r^\text{cs}_{b,s} = \begin{cases}
\text{ERV}(b,\, s+1) - \text{ERV}(b,\, s) & s < 2 \\
0 - \text{ERV}(b,\, 2) & s = 2
\end{cases}
$$

$$
r^\text{ball}_{b,s} = \begin{cases}
\text{ERV}(b+1,\, s) - \text{ERV}(b,\, s) & b < 3 \\
0.33 - \text{ERV}(3,\, s) & b = 3
\end{cases}
$$

where $\text{ERV}(b, s)$ is the expected run value of count $(b, s)$ from `count_values.csv` (RE24 framework) and $0.33$ approximates the mean run value of a walk.

### Take value

**`take_xRV`** — $R_i^\text{take}$:

$$
R_i^\text{take} = P_{\text{str},i} \cdot r^\text{cs}_{b_i, s_i} + (1 - P_{\text{str},i}) \cdot r^\text{ball}_{b_i, s_i}
$$

### Decision cost

**`decision_cost`** — $c_i$:

$$
c_i = R_i^\text{take} - \widehat{\text{xRV}}_i^\text{intended}
$$

| Symbol | Description |
|--------|-------------|
| $R_i^\text{take}$ | Expected run value of taking the pitch at its projected location |
| $\widehat{\text{xRV}}_i^\text{intended}$ | Expected run value of swinging with intended mechanics at projected location (internal `_xrv_intended` from `disruption_tax_split`) |
| $c_i$ | Positive = taking was better; negative = swinging was correct |

Population mean $\approx -0.08$ runs — batters self-select into swinging at pitches where swinging is correct.

---

## 4. Adjusted disruption tax

**`adjusted_disruption_tax`** — $\tau_{\text{adj},i}$:

$$
\tau_{\text{adj},i} = \tau_{\text{total},i} - \max(0,\; c_i)
$$

| Case | Result |
|------|--------|
| $c_i \leq 0$ (swinging was correct) | $\tau_{\text{adj},i} = \tau_{\text{total},i}$ |
| $c_i > 0$ (should have taken) | $\tau_{\text{adj},i} = \tau_{\text{total},i} - c_i$ — baseline shifts to $R_i^\text{take}$ |

Population mean $\approx -0.012$ runs vs. $-0.002$ for raw $\tau_\text{total}$; 16.2% of swings have $c_i > 0$.

$\tau_{\text{adj},i} \leq \tau_{\text{total},i}$ always. When swinging was optimal ($c_i \leq 0$), equality holds.

---

## Caveats

- `miss_distortion_tax` for whiffs requires `ball_bat_miss`; NaN on ~9% of whiff rows where Hawk-Eye data is missing
- $\kappa$ is estimated from a second OLS step, adding noise relative to a direct counterfactual
- Positive $c_i$ doesn't attribute the bad decision to movement — could be pitch-type deception, count-based gambling, etc. Movement attribution would require a fuller causal graph (out of scope)
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
