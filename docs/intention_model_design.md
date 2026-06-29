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
