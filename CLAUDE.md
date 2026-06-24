# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```powershell
# Activate the project venv (always use this Python, not system or miniforge)
.venv\Scripts\python.exe 00_pull_data.py          # pulls from mlb_db → data/swings_2023_2025.csv
.venv\Scripts\python.exe 01_precommit_split.py    # adds pc{ms}_* trajectory columns → data/swings_precommit.parquet
.venv\Scripts\python.exe run_values.py            # builds RE24, count values, linear weights → results/*.csv
.venv\Scripts\python.exe 04_run_pipeline.py       # full Phase A + Phase B → results/xrv_causal.parquet
```

The numbered scripts must run in order. `04_run_pipeline.py` is the entry point for everything downstream; it imports `02_intention_model` and `03_causal_models` directly via `importlib`.

---

## Architecture: two independent workstreams

**Workstream 1 — Causal mediation pipeline** (the main research, files `00–04`):

Estimates per-swing run-value disruption tax and splits it into distortion (mechanically caused by late pitch movement) vs. selection (batter decision). The causal chain is:

```
post-commit movement → swing-shape deviation → run value
    (treatment)            (mediator)           (outcome)
```

**Workstream 2 — Swing shape autoresearch** (`experiments/swing_shape/`):

Separate XGBoost hyperparameter search optimizing cross-validated RMSE on 15 swing-shape prediction models (5 targets × 3 pitch classes). Best result is exp28: `mean_rmse=5.485`. Only `config.py` is modified between experiments; `train.py` is read-only.

---

## Data source

All pitch data comes from `mlb_db` MySQL at `10.200.200.107`. Key column names as they appear in the DB and the saved CSVs/parquets:

- Pitcher handedness: `pitcher_throws` (values `"R"` / `"L"`) — **not** `pitcher_hand` or `p_throws`
- Ball in play: `is_bip` — distinct from `is_contact` (which includes fouls)
- Trajectory: `x0/y0/z0, vx0/vy0/vz0, ax/ay/az` at y=50 ft reference point
- Timing: `offset_y_ms` — early/on-time/late in milliseconds; frequently missing on whiffs

---

## Script-by-script process guide

### `00_pull_data.py` — Data pull + sequence lags

**Why pull ALL pitches (not just swings):** Sequence lag features (`prev_pitch_type`, `velo_delta`, `plate_x_delta`) require knowing what the previous pitch was — including takes, balls, and called strikes that would be invisible in a swing-only pull. All pitches are pulled first, lags computed, then non-swings are dropped.

**Inputs (from mlb_db):**
- Tables: `pbp_raw`, `pbp_descriptions`, `pbp_calculations`
- Filter: `level_id=1` (MLB), `game_type='R'` (regular season), `game_year IN (2023, 2024, 2025)`
- ~2.1M total pitch rows before swing filter

**Columns pulled:**
- Trajectory params: `x0/y0/z0, vx0/vy0/vz0, ax/ay/az` (at y=50 ft reference; required for pre/post-commit split)
- Swing shape: `vert_attack_angle, horz_attack_angle, bat_speed, swing_length, swing_path_tilt`
- Contact geometry: `ball_bat_intercept_y, ball_bat_miss, offset_y_ms, offset_z_in, offset_x_in`
- Zone: `plate_x, plate_z, sz_top, sz_bot`
- Count/context: `balls, strikes, outs_when_up, inning, count_group`
- Release: `release_pos_x/y/z, release_extension, arm_angle, release_speed, pfx_x, pfx_z`
- Labels: `is_swing, is_whiff, is_contact, is_bip, is_same_side_matchup, is_single/double/triple/home_run`
- Run value: `delta_run_exp`

**Sequence lags computed:**
- `prev_pitch_type, prev_release_speed, prev_plate_x, prev_plate_z`
- `velo_delta, plate_z_delta, plate_x_delta` (current − previous)
- `prev_outcome` (ball / whiff / foul / called_strike)

**Swing quality filters (applied after lags):**
- `is_swing == 1`
- `bat_speed > 40 mph` — removes sub-human / tracker garbage
- `vert_attack_angle ∈ [−45, 45]°` — removes 0.1% extreme outliers

**Output:** `data/swings_2023_2025.csv` (~760k rows). Bat-tracking coverage starts 2H 2023; 2023 rows will have NaN for bat-tracking columns on pre-rollout pitches.

**Considered but rejected:**
- Pulling swings only from the start — would miss prior-pitch context for lag features; first pitch of each PA would always appear out-of-sequence

---

### `01_precommit_split.py` — Pre/post-commit trajectory split

**Identification strategy (proj_desc §6):** A batter cannot select on information they do not have. The swing is committed early; post-commit movement is the gap between where the ball actually crossed the plate and where it would have crossed under constant-acceleration extrapolation from commit time. Conditioning on the full pre-commit trajectory makes post-commit deviation exogenous to the swing decision (conditional ignorability).

**Trajectory model (Zobrist 9-parameter, anchored at release):**
```
x(t) = R_x + V_x·t + ½·A_x·t²
y(t) = R_y + V_y·t + ½·A_y·t²   (y=0 at front of plate)
z(t) = R_z + V_z·t + ½·A_z·t²
```
Parameters reconstructed from `(release_pos_x, release_extension, release_pos_z, pfx_x, pfx_z, release_speed, plate_x, plate_z)`.

Key identity: `dev_x = pfx_x·(commit_s/t_plate)²` — post-commit deviation is an exact fractional rescaling of total horizontal break, with the fraction set by how early the batter commits. Same for `dev_z`.

**Inputs per row:** `release_pos_x, release_extension, release_pos_z, pfx_x, pfx_z, release_speed, plate_x, plate_z`

**Columns added (prefix `pc{commit_ms}_`) for commit grid `[125, 150, 175, 200]` ms:**
- `dev_x, dev_z, dev_total` — post-commit deviation at plate (ft)
- `x_proj, z_proj` — pre-commit projected plate crossing (ft); `x_proj + dev_x = plate_x` identically
- `x_commit, y_commit, z_commit` — ball position at commit time (ft)
- `vx_commit, vy_commit, vz_commit` — velocity at commit time (ft/s)
- `t_plate` — total flight time from release to plate (s); valid range ≈ [0.35, 0.60]s
- `R_x/y/z, V_x/y/z, A_x/y/z` — 9-parameter trajectory params (used by visualizations in `06_kinematic_diagram.py`)

**Why 150 ms is the default commit time:** Conservative choice gives a lower bound on distortion. Misfiling pre-commit movement as distortion contaminates the treatment regressor; misfiling distortion as pre-commit movement only absorbs it into the control. The asymmetry means an early setting biases toward less distortion — so any surviving effect is a conservative estimate. The robustness grid over 125/150/175/200 ms demotes commit-time uncertainty to a sensitivity check.

**Validation built in:** `_validate()` asserts `(x_proj + dev_x − plate_x).abs().max() < 1e-6` (algebraic identity to floating-point precision) and `t_plate ∈ [0.30, 0.65]s` for ≥98% of rows.

**Output:** `data/swings_precommit.parquet` (~760k rows, original columns + all pc{ms}_* columns)

**Considered but rejected:**
- Using `ax/ay/az` columns directly from Statcast instead of pfx reconstruction — pfx is more stable under missing/noisy ax values; Zobrist's original implementation uses pfx
- Latent changepoint model for commit time (mentioned in proj_desc §6) — would require lab swing-initiation capture as an informative prior; the MLB swing-shape likelihood is near-flat in commit time and can't identify it alone

---

### `02_intention_model.py` — Phase A: batter intended swing

**Purpose:** Estimates each batter's intended swing shape conditional on count, pitch location, platoon, and timing. The deviation residual `{metric}_dev = realized − intended` is the mediator that Phase B prices in run value.

**Responses — 5 total:**
- Angular (primary distortion signals): `vert_attack_angle, horz_attack_angle, swing_path_tilt`
- Effort (secondary): `bat_speed, swing_length`

Each response gets its own Bambi/PyMC Gaussian LMM (separate fits, not a joint multivariate model — see note below).

**Model formula (angular responses):**
```
{resp} ~ scale(balls) + scale(strikes)
       + scale(x_proj_bat) + x_proj_missing
       + scale(z_proj) + scale(z_proj_sq)
       + scale(offset_y_ms) + offset_y_ms_missing
       + pitcher_throws_L + pitcher_throws_L:scale(x_proj_bat)
       + (1 + scale(strikes) | batter_id)
       + (1 | pitcher_id)
```

**Model formula (effort responses):**
```
{resp} ~ scale(balls) + scale(strikes)
       + scale(x_proj_bat) + scale(z_proj)
       + pitcher_throws_L
       + (1 + scale(strikes) | batter_id)
       + (1 | pitcher_id)
```

**What each term does and why it's there:**

- `x_proj_bat`: projected plate crossing at commit time (pc150_x_proj) in batter's own frame — `x_proj × {−1 for RHB, +1 for LHB}`. Uses the projected location rather than `plate_x` because `plate_x = x_proj + dev_x`; conditioning on `plate_x` would encode post-commit movement into the intention baseline, partially canceling it out of the mediator and attenuating the distortion estimate. Commit time fixed at 150 ms (`COMMIT_MS` module constant).
- `x_proj_missing` indicator: trajectory reconstruction fails for ~2–4% of rows; mean-imputed with this indicator absorbing the systematic shift.
- `z_proj` / `z_proj_sq`: quadratic smooth on projected height (pc150_z_proj) — batters mechanically tilt their swing to match the pitch plane; under-modeling height mislabels appropriate plane adaptation as execution error, which contaminates the mediator.
- `offset_y_ms`: timing axis (early/on-time/late in ms) — attack angle at contact depends on where in the swing arc the bat was sampled; conditioning on timing removes the Powers-Yurko arc-sampling artifact from the deviation residual.
- `offset_y_ms_missing` indicator + mean imputation: missingness is systematic (correlated with whiff rate and contact quality), not MCAR. The indicator absorbs the average shift for missing rows; the timing coefficient is identified only from observed rows.
- `pitcher_throws_L`: absorbs spin-direction reversal across platoon matchups. Left-handed pitchers' breaking balls break the opposite horizontal direction, creating a mechanical location effect that is not a batter intention signal.
- `pitcher_throws_L:scale(x_proj_bat)`: interaction — the platoon-handedness effect is not uniform across location; a left-hander's ball to an inside corner on an RHB looks like a breaking ball to the outside corner of an LHB.
- `(1 + scale(strikes) | batter_id)`: per-batter intercept + count-pressure slope. The strikes slope captures how each batter adjusts their swing as the count worsens — this is the core "intention" signal.
- `(1 | pitcher_id)`: partial out mound quality from the intention baseline. Excluded from predictions — the counterfactual intended swing is what the batter would do against a neutral pitcher.

**Inference options:**
- `method="vi"` (default): ADVI, 50k iterations, ~2 minutes. Only posterior means are used downstream, so VI is functionally equivalent to MCMC for this pipeline.
- `method="mcmc"`: NUTS, `draws=1000, tune=1000, chains=4, max_treedepth=15`. Takes hours; use only if posterior uncertainty is needed.

**Custom prediction (`_posterior_mean_predict`):** Bypasses `model.predict()` which materializes `(n_obs × n_groups × n_draws)` arrays and OOMs at ~763k observations. Manually applies posterior-mean fixed effects and batter REs as a linear predictor: `ŷ = X @ β̄ + Z_batter @ ū_batter`. Pitcher REs are always excluded (intention baseline strips mound quality). Unseen batters get RE = 0.

**Outputs per swing:**
- `intended_{metric}` — posterior-mean predicted swing shape (the intention baseline)
- `{metric}_dev` — `realized − intended` (the Phase B mediator)
- `angular_mahal` — Mahalanobis distance in joint 3D angular deviation space using the empirical residual covariance matrix

**Considered but rejected:**
- Per-batter random slopes on `x_proj_bat` and `z_proj`: creates a 4×4 LKJ correlation prior that causes degenerate NUTS geometry (max_treedepth warnings, R-hat > 1.01 for tail batters regardless of tuning steps). Confirmed across autoresearch experiments. Location effects stay as fixed effects only.
- True joint `mvbind` fit with correlated batter RE across all 5 responses (the original plan in proj_desc §8, which requires `brms`/Stan): approximated with separate Bambi fits + empirical residual covariance for the Mahalanobis metric. The only material concession vs. brms is that the joint batter RE covariance structure is recovered post-hoc rather than jointly identified.
- Skew-normal family for `bat_speed` and `swing_length` (proj_desc §8 specifies this to handle hold-back asymmetry): Bambi 0.18 lacks skewnormal; Gaussian used for all responses.
- `n_subsample=None` (full dataset): the `(1 | pitcher_id)` term causes Bambi's `formulae` backend to allocate a dense `(n_rows × n_pitchers)` contrast matrix that OOMs at ~4.9 GB on the full dataset. Default is 75k rows.
- Adding `velo_delta`, `prev_pitch_type` sequence features to the intention formula — considered for count_group conditioning; dropped because sequence context is a pitch-selection predictor, not an intention-of-swing predictor

### Phase A MCMC known issues

- **Batter random slopes must be intercept + strikes only.** Adding `x_proj_bat` or `z_proj` to the random effects creates a 4×4 LKJ correlation prior with degenerate NUTS geometry.
- **Use `n_subsample=75_000` (the default).** The full dataset OOMs the pitcher_id contrast matrix at ~4.9 GB. Use `n_subsample=None` only with ≥16 GB RAM free.
- R-hat > 1.01 for some tail batters is expected with MCMC on this model — not a bug. `tune=1000, max_treedepth=15` are the minimum viable settings.

---

### `03_causal_models.py` — Phase B: run-value mediation

**DAG:**
```
post-commit movement  →  angular deviation  →  run value
    (treatment)             (mediator, §A)      (outcome)
```

#### 1. Mediator models (Linear Mixed-Effects, `statsmodels.MixedLM`)

One model per angular deviation axis:
```
{metric}_dev ~ pc{ms}_dev_x + pc{ms}_dev_z        ← treatment (post-commit movement)
             + pc{ms}_x_proj + pc{ms}_z_proj        ← pre-commit projected location (control)
             + release_speed + balls + strikes + offset_y_ms
```
Groups: `batter_id` (random intercept). Fit on REML=False.

Minimum inclusion thresholds: `batter_id ≥ 20 swings`, `pitcher_id ≥ 10 swings` (pitcher_id not in model; used for filtering only).

The treatment coefficients `a_x, a_z` per axis are the causal leverage: how many degrees of swing deviation does one foot of post-commit horizontal/vertical movement cause.

#### 2. Outcome models — three channels

- `bip_model`: logistic P(BIP | swing) ~ angular deviations + plate_x + plate_z + balls + strikes (HC1 SEs)
- `foul_model`: logistic P(foul | not BIP) ~ same predictors, fit on non-BIP swings only
- `xwoba_model`: OLS xwOBAcon ~ same predictors, BIP only

**Why foul and whiff are separate:** At 2 strikes, a foul keeps the count; a whiff ends the PA. Conflating them biases xRV at moderate disruption levels where foul rate is elevated.

**Composite xRV:**
```
P(foul)  = (1 − P(BIP)) × P(foul | not BIP)
P(whiff) = (1 − P(BIP)) × (1 − P(foul | not BIP))
xRV = P(BIP)×E[xwOBA|BIP] + P(foul)×foul_rv[count] + P(whiff)×whiff_rv[count]
```

#### 3. Disruption tax (predict-twice counterfactual)

- `xrv_realized`: predict xRV at the realized angular deviations
- `xrv_intended`: predict xRV with all angular deviations set to 0 (the counterfactual intended swing)
- `disruption_tax = xrv_realized − xrv_intended` (negative = pitcher cost batter runs)

#### 4. Distortion attribution (squared-norm decomposition)

Per angular axis m:
```
distortion_dev_m = a_x_m × pc_dev_x + a_z_m × pc_dev_z   (movement-caused component)
selection_dev_m  = {metric}_dev_m − distortion_dev_m        (residual)
```

```
distortion_share = ||distortion_dev||² / ||total_dev||²
```

**Why squared-norm:** The raw L2-norm ratio (`dist_norm / (dist_norm + sel_norm)`) is wrong when distortion and selection components point in opposite directions — the denominator becomes larger than the total. Squared-norm decomposition gives a clean [0,1] proportion and is algebraically correct.

```
distortion_tax = disruption_tax × distortion_share
selection_tax  = disruption_tax × (1 − distortion_share)
```

#### 5. Indirect effect (analytical, product-of-coefficients)

For each axis m and each outcome channel:
```
indirect_m = a_m × ∂xRV/∂m
```
`∂xRV/∂m` accounts for all three channels via the logistic gradient. Consistent with the counterfactual tax under linearity; differs only through nonlinearity in the contact channel.

#### Validation controls

- **Negative control:** FF with `pc150_dev_total < 1/12 ft` should show `disruption_tax ≈ 0`. Nonzero means the pre/post split is leaking selection.
- **Positive control:** high-movement pitches (sweepers, sinkers) should show the largest distortion tax.

**Considered but rejected:**
- Crossed pitcher random effects in mediator models (commented in `fit_mediator_models`): slow on large datasets and pitcher effect is minor for the mediator — the pre-commit control already absorbs most pitch-quality variation
- Mediation sensitivity analysis for unmeasured mediator-outcome confounding (proj_desc §9) — not yet implemented

---

### `04_run_pipeline.py` — Orchestrator

Runs Phase A → Phase B in sequence. Key behaviors:

**`competitive_swings()` filter:** `bat_speed ≥ 50 mph` (stricter than the 40 mph pull filter in `00_pull_data.py`). Removes borderline tracked swings that cleared the data pull but are mechanically implausible for full-effort competitive swings.

**Phase A cache:** `models/intended_df.parquet` (plain DataFrame of predicted intents). Bambi `Model` objects cannot be pickled on Python 3.14 due to `FrameLocalsProxy` in `formulae.Environment`. The cache stores the outputs, not the model objects.

**`--skip-phase-a`:** Loads `intended_df.parquet` from cache without refitting Phase A. Useful when iterating on Phase B without changing the intention model.

**`foul_rv` vs `whiff_rv`:**
- Whiff at (b, s < 2): `delta = ERV(b, s+1) − ERV(b, s)` (count advances)
- Whiff at (b, 2): `delta = 0 − ERV(b, 2)` (strikeout)
- Foul at (b, s < 2): same as whiff — count advances
- Foul at (b, 2): `delta = 0` — count stays, PA continues

**`xwoba` assignment:** Hit outcomes checked in descending order of value (`HR → 3B → 2B → 1B → out_in_play`) to handle any double-coding edge cases in source data.

---

## Key invariants

- `disruption_tax = xRV(realized) − xRV(intended)` where "intended" means all angular deviations = 0. Negative = pitcher cost batter runs.
- `distortion_share` uses squared-norm decomposition: `||distortion_dev||² / ||total_dev||²`, giving a clean [0,1] proportion. The raw L2-norm ratio (`dist_norm / (dist_norm + sel_norm)`) is wrong when components point in opposite directions.
- `foul_rv` differs from `whiff_rv` at 2-strike counts: foul at 2K → delta = 0 (count stays); whiff at 2K → delta = −ERV(b, 2) (strikeout).
- The pitcher_id sentinel (`-1`) in `predict_intended` zeros out pitcher random effects, leaving only batter RE in the intention baseline. Pitcher handedness (`pitcher_throws_L`) is a fixed effect and must still be populated.

---

## Outputs

| File | Contents |
|------|----------|
| `results/xrv_causal.parquet` | Per-swing disruption/distortion/selection tax |
| `results/distortion_pitcher.csv` | Aggregated by pitcher (≥50 swings) |
| `results/distortion_batter.csv` | Aggregated by batter (≥50 swings) |
| `models/intention_result.joblib` | Phase A Bambi models + idata |
| `models/intended_df.parquet` | Phase A per-swing intended swing shape (prediction cache) |
| `models/causal_models.joblib` | Phase B models (bip_model, foul_model, xwoba_model, whiff_rv, foul_rv) |
| `results/linear_weights.csv` | Required input for xwOBA calculation |
| `results/count_values.csv` | Required input for whiff_rv / foul_rv |
