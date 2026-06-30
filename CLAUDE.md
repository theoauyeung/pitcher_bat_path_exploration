# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```powershell
# Always use the project venv, not system Python
.venv\Scripts\python.exe 00_pull_data.py          # data/swings_2023_2025.csv
.venv\Scripts\python.exe 01_precommit_split.py    # data/swings_precommit.parquet
.venv\Scripts\python.exe run_values.py            # results/linear_weights.csv, count_values.csv
.venv\Scripts\python.exe 04_run_pipeline.py       # results/xrv_causal.parquet + leaderboards
```

Scripts must run in order. `04_run_pipeline.py` is the entry point for Phase A + Phase B; it imports `02_intention_model` and `03_causal_models` directly.

Use `--skip-phase-a` to reload the cached Phase A output (`models/intended_df.parquet`) without refitting — useful when iterating on Phase B alone.

---

## Architecture

**Workstream 1 — Causal mediation pipeline** (`00–04`):

Estimates how much of each swing's run-value cost is caused by post-commit pitch movement (distortion) vs. the batter's own swing decision (selection). The causal chain:

```
post-commit movement → swing-shape deviation → run value
    (treatment)            (mediator)           (outcome)
```

**Workstream 2 — Swing shape autoresearch** (`experiments/swing_shape/`):

Separate XGBoost hyperparameter search over 15 swing-shape prediction models. Only `config.py` changes between experiments; `train.py` is read-only.

---

## Data source
- Pitcher handedness: `pitcher_throws` (`"R"` / `"L"`) — **not** `pitcher_hand` or `p_throws`
- Ball in play: `is_bip` — distinct from `is_contact` (which includes fouls)
- Timing offset: `offset_y_ms` — frequently missing on whiffs; the model handles this with imputation + a missing indicator

---

## What each script does

### `00_pull_data.py` — Data pull

Pulls all pitches (not just swings) so sequence lag features (`prev_pitch_type`, `velo_delta`, location deltas) capture what the batter saw on the previous pitch, including takes. After lags are computed, non-swings are dropped.

Output: `data/swings_2023_2025.csv` (~760k swing rows). Bat-tracking columns will be NaN for 2023 pitches before mid-season rollout.

### `01_precommit_split.py` — Pre/post-commit trajectory split

Reconstructs each pitch's full flight path from release parameters, then computes where the ball *would have* crossed the plate if it had continued on a constant trajectory from the batter's commit time (~150 ms pre-contact). The gap between that projected location and the actual plate crossing is the post-commit deviation — movement the batter had no time to react to.

Default commit time is 150 ms. This is deliberately conservative (understates distortion), so the robustness grid over 125/150/175/200 ms treats commit-time uncertainty as a sensitivity check rather than the main analysis.

Output: `data/swings_precommit.parquet` with `pc{ms}_dev_x/z`, `pc{ms}_x_proj/z_proj`, and 9-parameter trajectory columns for each commit time in the grid.

### `02_intention_model.py` — Phase A: batter intended swing (imported by 04)

Fits a Bayesian LMM per swing-shape response (VAA, HAA, swing path tilt, bat speed, swing length) using count, pitch location, contact timing, and platoon handedness as predictors, with per-batter random effects. The model captures what each batter *intended* to do given the information available at swing time.

The residual `realized − intended` is the swing deviation mediator that Phase B uses to price distortion.

Key behavior: `method="vi"` (ADVI) is the default — only posterior means are used downstream so it's equivalent to MCMC and takes ~2 min instead of hours. Phase A output is cached to `models/intended_df.parquet`; Bambi model objects cannot be pickled on Python 3.14.

### `03_causal_models.py` — Phase B: run-value mediation (imported by 04)

Two sets of models:

**Mediator models** (one per angular deviation axis): estimate how much of each swing deviation is mechanically caused by post-commit movement. The treatment coefficients give the causal leverage — degrees of swing deviation per foot of late movement.

**Outcome models** (three channels): price swing deviation in run value using XGBoost gradient-boosted trees. Trees prevent the linear extrapolation artifacts that logistic/OLS models produce at extreme plate locations (e.g. pitches 9" above the zone) or extreme angular deviations.
- `bip_model` — P(ball in play)
- `foul_model` — P(foul | not BIP), fit only on non-BIP swings. Kept separate from whiff because at two strikes a foul keeps the PA alive; a whiff ends it.
- `xwoba_model` — E[xwOBA | BIP]

Feature set for all three: `OUTCOME_FEATURES = ANGULAR_DEVS + ["plate_x", "plate_z", "balls", "strikes"]`. Column order matters — always slice with `df[OUTCOME_FEATURES]` before predicting.

Also fits **miss models** (statsmodels OLS) to measure physical bat-to-ball miss on whiffs and contacts, and computes **decision cost** — the opportunity cost of swinging vs. taking at the projected plate location. **`adjusted_disruption_tax`** combines these: `disruption_tax − max(0, decision_cost)`, giving total batter burden vs. the optimal available action.

**Disruption tax** uses three counterfactual scenarios:

| Scenario | Swing angles | Plate location |
|----------|-------------|----------------|
| Realized | actual deviations | actual (post-movement) |
| Spatial only | zero deviations | actual (post-movement) |
| Intended | zero deviations | projected (pre-movement) |

```
disruption_tax          = xrv_realized − xrv_intended
spatial_distortion      = xrv_spatial  − xrv_intended
angular_disruption      = xrv_realized − xrv_spatial
distortion_tax          = spatial_distortion + angular_disruption × angular_distortion_share
selection_tax           = angular_disruption × (1 − angular_distortion_share)
adjusted_disruption_tax = disruption_tax − max(0, decision_cost)
```

`angular_distortion_share` uses squared-norm decomposition across the three angular axes. Spatial disruption is fully attributed to distortion by construction. `adjusted_disruption_tax` is additive — equals `disruption_tax` when swinging was correct; shifts baseline to `take_xrv` when taking was better.

### `04_run_pipeline.py` — Orchestrator

Runs Phase A → Phase B in sequence and writes all outputs. Key flag: `--skip-phase-a` loads cached Phase A output. Use `method="vi"` for fast iteration.

---

## Outputs

| File | Contents |
|------|----------|
| `results/xrv_causal.parquet` | Per-swing: `disruption_tax`, `adjusted_disruption_tax`, `distortion_tax`, `selection_tax`, `spatial_distortion_tax`, `distortion_share`, `miss_distortion_tax`, `decision_cost` |
| `results/distortion_pitcher.csv` | Pitcher-level leaderboard (≥50 swings) |
| `results/distortion_batter.csv` | Batter-level leaderboard (≥50 swings) |
| `models/intention_result.joblib` | Phase A idata + training data |
| `models/intended_df.parquet` | Phase A per-swing intended swing shape (cache) |
| `models/causal_models.joblib` | Phase B models |
