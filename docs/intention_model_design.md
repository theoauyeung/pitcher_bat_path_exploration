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

### Symbol definitions

| Symbol | Column | Units | Description |
|--------|--------|-------|-------------|
| $y_i$ | response | ° or mph or ft | Swing-shape outcome for swing $i$ |
| $z(\cdot)$ | — | dimensionless | Standardize: $(x - \mu_\text{train}) / \sigma_\text{train}$ |
| $b_i$ | `balls` | 0–3 | Balls in count before pitch |
| $s_i$ | `strikes` | 0–2 | Strikes in count before pitch |
| $x_i$ | `plate_x_bat` | ft | Pitch location in batter's frame: `plate_x` × (−1 if RHB, +1 if LHB); inside = positive for both hands |
| $h_i$ | `plate_z` | ft | Pitch height above ground at plate crossing |
| $t_i$ | `offset_y_ms` | ms | Contact timing offset; negative = early, positive = late; imputed to training mean when missing |
| $\delta_i$ | `offset_y_ms_missing` | 0/1 | 1 if timing was missing; absorbs the mean shift for timing-missing rows |
| $L_i$ | `pitcher_throws_L` | 0/1 | 1 if pitcher is left-handed |
| $u_{0j}$ | — | — | Per-batter random intercept for batter $j$ |
| $u_{1j}$ | — | — | Per-batter random slope on $z(s_i)$; captures individual count-pressure adjustment |
| $v_{0k}$ | — | — | Per-pitcher random intercept for pitcher $k$; always **zero at prediction time** |
| $\varepsilon_i$ | — | — | Residual error, $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ |

### Angular responses (`vert_attack_angle`, `horz_attack_angle`, `swing_path_tilt`)

$$
\begin{aligned}
y_i &= \beta_0 \\
    &+ \beta_1\, z(b_i) + \beta_2\, z(s_i) \\
    &+ \beta_3\, z(x_i) + \beta_4\, z(h_i) + \beta_5\, z(h_i^2) \\
    &+ \beta_6\, z(t_i) + \beta_7\, \delta_i \\
    &+ \beta_8\, L_i + \beta_9\, z(x_i) \cdot L_i \\
    &+ u_{0j} + u_{1j}\, z(s_i) + v_{0k} + \varepsilon_i
\end{aligned}
$$

### Effort responses (`bat_speed`, `swing_length`)

$$
\begin{aligned}
y_i &= \beta_0 \\
    &+ \beta_1\, z(b_i) + \beta_2\, z(s_i) \\
    &+ \beta_3\, z(x_i) + \beta_4\, z(h_i) \\
    &+ \beta_5\, L_i \\
    &+ u_{0j} + u_{1j}\, z(s_i) + v_{0k} + \varepsilon_i
\end{aligned}
$$

---

## Prediction formula

The prediction uses only posterior means (no MCMC uncertainty propagated downstream):

$$
\hat{y}_i = \mathbf{x}_i^\top \bar{\boldsymbol{\beta}} \;+\; \mathbf{z}_{\text{batter},i}^\top \bar{\mathbf{u}}_j
$$

| Symbol | Description |
|--------|-------------|
| $\bar{\boldsymbol{\beta}}$ | Posterior mean fixed-effect coefficients |
| $\bar{\mathbf{u}}_j$ | Posterior mean batter random effects for batter $j$: $[\bar{u}_{0j},\, \bar{u}_{1j}]$ |
| $\mathbf{z}_{\text{batter},i}$ | Batter design row: $[1,\, z(s_i)]$ |
| $v_{0k}$ | Always 0 at predict time — intention baseline strips mound quality |
| Unseen batters | $\bar{\mathbf{u}}_j = \mathbf{0}$ (population mean) |

Valid for Gaussian LMM because $\mathbb{E}[X\boldsymbol{\beta} + Z\mathbf{u}] = X\,\mathbb{E}[\boldsymbol{\beta}] + Z\,\mathbb{E}[\mathbf{u}]$.

---

## Deviation residual

$$
\Delta_i^{(m)} = y_i^{\text{realized}} - \hat{y}_i^{\text{intended}}
$$

| Symbol | Column | Description |
|--------|--------|-------------|
| $y_i^\text{realized}$ | `{metric}` | Observed swing-shape value |
| $\hat{y}_i^\text{intended}$ | `intended_{metric}` | Model prediction $\hat{y}_i$ |
| $\Delta_i^{(m)}$ | `{metric}_dev` | Phase B mediator; positive = batter executed above intention |

---

## Formula design decisions

**Angular responses** condition on more predictors than effort responses because attack angle is more sensitive to location geometry:

- $z(x_i)$ — pitch location in the batter's own frame. A pitch on the inner half mechanically produces a different attack angle than the same pitch outside; conditioning on this removes it from the deviation residual.
- $z(h_i) + z(h_i^2)$ (quadratic) — batters tilt their swing plane to match pitch height. Without the quadratic term, the model mislabels appropriate low-ball plane adaptation as execution error.
- $z(t_i)$ — contact timing. Attack angle at contact depends on where in the swing arc the bat was measured; conditioning on timing removes the arc-sampling artifact from the deviation (Powers-Yurko effect).
- $L_i + z(x_i) \cdot L_i$ — absorbs spin-direction reversal across platoon matchups. A left-hander's breaking ball breaks opposite to a right-hander's; conflating them would contaminate the location fixed effect.

**Missing timing** ($t_i$) is frequent on whiffs (~40%) and is not missing at random — it correlates with contact quality. Imputed to the training-data mean with indicator $\delta_i$ so the timing coefficient is identified only from observed rows.

**Batter random effects**: intercept + strikes slope per response. Location slopes ($x_i$, $h_i$) were tested as random effects but produce a degenerate LKJ correlation prior in NUTS geometry (max_treedepth warnings, R-hat > 1.01 for tail batters). Location effects stay as fixed effects; the random slope captures each batter's individual adjustment under count pressure, which is the core intention signal.

**Pitcher random effects**: intercept only, excluded from all predictions. The intention baseline is "what would this batter do against a neutral pitcher" — including pitcher REs would bake in mound-quality effects that belong in the outcome model, not the intention.

---

## Inference

Default: `method="vi"` (ADVI, 50k iterations, ~2 min). Only posterior means are used downstream — for this application, VI and MCMC produce functionally equivalent point estimates. Use `method="mcmc"` only if posterior uncertainty or chain diagnostics are needed.

Subsample default: `n_subsample=75_000`. The `(1 | pitcher_id)` term in Bambi's formulae backend allocates a dense $(n_\text{rows} \times n_\text{pitchers})$ contrast matrix; the full ~763k dataset OOMs at ~4.9 GB. The subsample is sufficient for stable fixed-effect and batter RE estimates.

**Custom prediction** (`_posterior_mean_predict`): `model.predict()` materializes $(n_\text{obs} \times n_\text{groups} \times n_\text{draws})$ arrays and OOMs at 763k observations. The custom predictor applies posterior-mean fixed effects and batter REs as the direct linear combination above. Unseen batters receive RE = 0 (population mean).

---

## Outputs

For each swing:
- `intended_{metric}` — the posterior-mean predicted swing shape $\hat{y}_i$
- `{metric}_dev` — the deviation $\Delta_i^{(m)}$, the Phase B mediator

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
