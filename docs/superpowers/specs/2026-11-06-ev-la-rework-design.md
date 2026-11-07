# EV/LA Pipeline Rework Design

## Overview

Replace the DB-dependent data layer with an R + sabRmetrics pull, and rework the Phase B BIP outcome channel to predict continuous exit velocity and launch angle rather than discrete-bucket xwOBA. The goal is a mechanistically grounded causal chain:

> post-commit movement → swing deviation → exit velocity / launch angle → run value

The distortion_tax metric definition and all downstream outputs are preserved. The rework makes the BIP channel continuous and interpretable, and adds `ev_suppression` as a plain-English summary metric.

---

## Architecture

### Data layer (R → Python handoff)

**New:** `src/data/pull_savant.R`
Replaces `pull_data.py` and `run_values.py` entirely.

1. Pulls 2023, 2024, 2025 from Baseball Savant using `sabRmetrics::download_baseballsavant()` with parallel cluster
2. Binds years into one frame, writes `data/savant_raw.csv`
3. Computes RE24 and count-state run values using sabRmetrics built-ins, writes:
   - `results/linear_weights.csv`
   - `results/count_values.csv`

Pull pattern (adapted from prior projects):
```r
library(sabRmetrics)
library(tidyverse)

cluster <- parallel::makeCluster(parallel::detectCores())
years <- list(
  "2023" = download_baseballsavant("2023-01-01", "2023-12-31", cl = cluster),
  "2024" = download_baseballsavant("2024-01-01", "2024-12-31", cl = cluster),
  "2025" = download_baseballsavant("2025-01-01", "2025-12-31", cl = cluster)
)
parallel::stopCluster(cluster)
raw <- bind_rows(years)
write.csv(raw, "data/savant_raw.csv", row.names = FALSE)
```

**Risk:** `download_baseballsavant()` returns "all available columns" from Baseball Savant. Standard bat tracking columns (`vert_attack_angle`, `horz_attack_angle`, `swing_path_tilt`, `launch_speed`, `launch_angle`) are confirmed in the Statcast schema. More granular Hawk-Eye columns (`ball_bat_miss`, `offset_y_ms`, `offset_z_in`, `offset_x_in`) are present in the Baseball Savant bulk download as of 2023 but should be verified on first run. If missing, `fit_miss_models()` degrades gracefully (NaN rows are dropped).

**New:** `src/data/process_savant.py`
Replaces the Python portion of `pull_data.py`. Reads `data/savant_raw.csv`, applies column renames to match the existing schema, computes sequence lag features (`prev_pitch_type`, `velo_delta`, location deltas), filters to swings (`is_swing == 1`), writes `data/swings_2023_2025.csv`.

Output schema is identical to what `pull_data.py` produced, plus `launch_speed` and `launch_angle` columns.

**Deleted:** `src/data/pull_data.py`, `src/data/run_values.py`

**Unchanged:** `src/data/precommit_split.py` — reads from CSV/parquet, no DB dependency.

### Pipeline order

```
src/data/pull_savant.R          → data/savant_raw.csv
                                   results/linear_weights.csv
                                   results/count_values.csv
src/data/process_savant.py      → data/swings_2023_2025.csv
src/data/precommit_split.py     → data/swings_precommit.parquet
src/models/run_pipeline.py      → results/xrv_causal.parquet + leaderboards
```

---

## Phase B rework — `src/models/causal_models.py`

### What changes

**`fit_outcome_models()`** — signature changes from returning `(bip_model, foul_model, xwoba_model, whiff_rv)` to `(bip_model, foul_model, exit_velo_model, launch_angle_model, ev_la_rv_model, whiff_rv)`.

Three new XGBoost models replace `xwoba_model`:

**`exit_velo_model`** — `XGBRegressor` on BIP swings (`is_bip == 1`, `launch_speed.notna()`). Features: `OUTCOME_FEATURES` (unchanged — `angular_devs + plate_x + plate_z + balls + strikes`). Target: `launch_speed` (mph).

**`launch_angle_model`** — `XGBRegressor` on BIP swings. Same features. Target: `launch_angle` (degrees).

**`ev_la_rv_model`** — `XGBRegressor` on all BIP in the dataset. Features: `[launch_speed, launch_angle]`. Target: `delta_run_exp`. This maps the continuous EV/LA space to run value — a smooth surface capturing that 105 mph at −5° is a groundout and 105 mph at 28° is a homer, without binning.

### Updated `_xrv_from_shape()`

BIP channel changes from:

```python
# old
e_xwoba = xwoba_model.predict(feat)
return p_bip * e_xwoba + p_foul * foul_vals + p_whiff * whiff_vals
```

to:

```python
# new
ev_pred = exit_velo_model.predict(feat)
la_pred = launch_angle_model.predict(feat)
ev_la_feat = np.column_stack([ev_pred, la_pred])
e_rv_bip = ev_la_rv_model.predict(ev_la_feat)
return p_bip * e_rv_bip + p_foul * foul_vals + p_whiff * whiff_vals
```

Whiff and foul channels (`bip_model`, `foul_model`, `whiff_rv`, `foul_rv`) are unchanged.

### New output columns in `disruption_tax_split()`

Added to `xrv_causal.parquet` (NaN on non-BIP rows):

| Column | Definition |
|--------|-----------|
| `ev_realized` | Actual `launch_speed` from raw data |
| `la_realized` | Actual `launch_angle` from raw data |
| `ev_intended` | `exit_velo_model.predict(intended_features)` |
| `la_intended` | `launch_angle_model.predict(intended_features)` |
| `ev_suppression` | `ev_realized − ev_intended` (negative = pitcher suppressed EV) |

The `disruption_tax`, `distortion_tax`, `selection_tax`, `spatial_distortion_tax`, `distortion_share`, `adjusted_disruption_tax`, `miss_distortion_tax`, `decision_cost` columns are preserved with identical definitions.

### `run_pipeline.py` changes

- Update `fit_outcome_models()` call site to unpack 6 returns instead of 4
- Pass `exit_velo_model`, `launch_angle_model`, `ev_la_rv_model` through to `disruption_tax_split()` and `indirect_effect()` — both functions' signatures change from `xwoba_model` to `exit_velo_model, launch_angle_model, ev_la_rv_model`
- Join `launch_speed` and `launch_angle` from raw swing data onto `xrv_causal` output for `ev_realized` / `la_realized` columns
- Update leaderboard aggregations to include `mean_ev_suppression`, `mean_la_change`

---

## Leaderboard updates

`results/distortion_pitcher.csv` and `results/distortion_batter.csv` get two new columns:

- `mean_ev_suppression` — mean of `ev_suppression` over qualifying BIP swings (negative = pitcher suppresses EV)
- `mean_la_change` — mean of `la_realized − la_intended` over qualifying BIP swings

All existing columns unchanged.

---

## Visualization updates — `src/viz/generate_results.ipynb`

One new cell added after the existing validation section:

**EV suppression scatter** — pitcher-level scatter of `mean_ev_suppression` (x) vs. `mean_distortion_tax` (y), annotating the top-10 pitchers by distortion tax. Narrative anchor: shows the two metrics track together, grounding the abstract run-value distortion tax in the legible unit of mph.

Existing 8 analysis cells are unchanged — all use `distortion_tax` directly and are unaffected by how the BIP channel is computed.

---

## Phase 2 (deferred) — `src/viz/validation_eda.ipynb`

A standalone EDA notebook providing visual empirical justification for every major design decision. Not part of the Phase 1 implementation. Sections planned:

1. **Why EV/LA over discrete xwOBA buckets** — R² comparison of continuous EV/LA model vs. 5-bucket linear weights; EV/LA distribution shift under realized vs. intended scenarios
2. **Why 150ms commit threshold** — distortion signal (correlation of post-commit deviation with swing deviation) plotted across 100–200ms grid; shows peak near 150ms
3. **Why causal split (distortion vs. selection)** — batters with similar `disruption_tax` but different `distortion/selection` ratios; show the ratio predicts different outcome profiles
4. **Why Bayesian LMM for intended swing** — residual vs. fitted for OLS baseline vs. LMM; per-batter random effect distribution showing shrinkage is non-trivial
5. **Why XGBoost for outcome models** — OLS/logistic extrapolation artifacts at extreme plate locations vs. XGBoost stability; shown as predicted probability surface
6. **Why breaking balls for convergent validity** — pitch-mix confound visualization (all pitches vs. breaking-ball-only correlation with post-commit deviation)

---

## Global constraints

- R ≥ 4.0, sabRmetrics installed; Python pipeline unchanged in dependencies
- `savant_raw.csv` is the single source of truth for raw data — no DB connection required anywhere
- `launch_speed` and `launch_angle` columns must be present in `swings_2023_2025.csv` output of `process_savant.py`
- `OUTCOME_FEATURES` list in `causal_models.py` is unchanged — EV/LA models use the same feature set as the old `xwoba_model`
- All existing column names in `xrv_causal.parquet` are preserved; new columns are additive only
- `--skip-phase-a` flag behavior unchanged
- `MIN_SWINGS = 200` filter unchanged in all leaderboard outputs
- `CLAUDE.md` updated to reflect new pipeline order and deleted scripts
