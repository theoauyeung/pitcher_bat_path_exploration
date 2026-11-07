# EV/LA Pipeline Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken DB-dependent data layer with an R + sabRmetrics pull and rework the Phase B BIP channel to predict continuous exit velocity and launch angle instead of discrete-bucket xwOBA, adding `ev_suppression` as a plain-English summary metric.

**Architecture:** An R script pulls 2023–2025 Statcast data via sabRmetrics, computes linear weights and count values from `delta_run_exp`, and writes three files that the Python pipeline reads unchanged. Inside Phase B, `xwoba_model` is replaced by three XGBoost models chained as swing features → EV/LA → run value; whiff and foul channels are untouched and the `distortion_tax` output definition is preserved.

**Tech Stack:** R ≥ 4.0, sabRmetrics, tidyverse, arrow (R parquet); Python, XGBoost, pandas, numpy, joblib; Jupyter.

## Global Constraints

- R ≥ 4.0, sabRmetrics and tidyverse installed
- `data/savant_raw.csv` is the single source of truth for raw data — no DB connection required anywhere
- `launch_speed` and `launch_angle` columns must be present in `data/swings_2023_2025.csv`
- `OUTCOME_FEATURES = list(ANGULAR_DEVS) + ["plate_x", "plate_z", "balls", "strikes"]` is unchanged in `causal_models.py`
- All existing column names in `results/xrv_causal.parquet` are preserved; new columns (`ev_realized`, `la_realized`, `ev_intended`, `la_intended`, `ev_suppression`) are additive only
- `--skip-phase-a` flag behavior unchanged
- `MIN_SWINGS = 200` filter unchanged in all leaderboard outputs
- `validation_eda.ipynb` is explicitly deferred — do NOT create it in this plan

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/data/pull_savant.R` | Pull 2023-2025 Statcast + compute RE24/LW in R |
| Create | `src/data/process_savant.py` | Column renames, lag features, swing filter → `swings_2023_2025.csv` |
| Delete | `src/data/pull_data.py` | Replaced by pull_savant.R + process_savant.py |
| Delete | `src/data/run_values.py` | Replaced by pull_savant.R |
| Modify | `src/models/causal_models.py` | Add EV/LA models, update Phase B signatures |
| Modify | `src/models/run_pipeline.py` | Update call sites, save EV columns, update leaderboards |
| Modify | `src/viz/generate_results.ipynb` | Add EV suppression scatter cell |
| Modify | `CLAUDE.md` | Update pipeline order, delete references to removed scripts |

---

### Task 1: R data pull — `src/data/pull_savant.R`

**Files:**
- Create: `src/data/pull_savant.R`
- Delete: `src/data/pull_data.py`, `src/data/run_values.py`

**Interfaces:**
- Consumes: Baseball Savant API via `sabRmetrics::download_baseballsavant()`
- Produces:
  - `data/savant_raw.csv` — one row per pitch, all Statcast columns including `launch_speed`, `launch_angle`, `delta_run_exp`
  - `results/linear_weights.csv` — columns: `outcome_type`, `lw`
  - `results/count_values.csv` — columns: `balls`, `strikes`, `expected_run_value`

- [ ] **Step 1: Create `src/data/pull_savant.R`**

```r
library(sabRmetrics)
library(tidyverse)

# ── 1. Pull 2023–2025 Statcast ───────────────────────────────────────────────
cluster <- parallel::makeCluster(parallel::detectCores())

data_2023 <- sabRmetrics::download_baseballsavant(
  start_date = "2023-01-01",
  end_date   = "2023-12-31",
  cl         = cluster
)
data_2024 <- sabRmetrics::download_baseballsavant(
  start_date = "2024-01-01",
  end_date   = "2024-12-31",
  cl         = cluster
)
data_2025 <- sabRmetrics::download_baseballsavant(
  start_date = "2025-01-01",
  end_date   = "2025-12-31",
  cl         = cluster
)

parallel::stopCluster(cluster)

raw <- bind_rows(data_2023, data_2024, data_2025)
cat("Total rows:", nrow(raw), "\n")
cat("Columns:", paste(names(raw), collapse = ", "), "\n")

# ── 2. Write raw data ─────────────────────────────────────────────────────────
dir.create("data", showWarnings = FALSE)
write.csv(raw, "data/savant_raw.csv", row.names = FALSE)
cat("Saved data/savant_raw.csv\n")

# ── 3. Linear weights from delta_run_exp ─────────────────────────────────────
# For PA-ending pitches, delta_run_exp equals the linear weight of the event.
# We map Statcast event names to the pipeline's outcome_type keys.
outcome_map <- c(
  "single"                        = "single",
  "double"                        = "double",
  "triple"                        = "triple",
  "home_run"                      = "home_run",
  "strikeout"                     = "strikeout",
  "strikeout_double_play"         = "strikeout",
  "field_out"                     = "out_in_play",
  "grounded_into_double_play"     = "out_in_play",
  "force_out"                     = "out_in_play",
  "double_play"                   = "out_in_play",
  "fielders_choice"               = "out_in_play",
  "fielders_choice_out"           = "out_in_play",
  "sac_fly"                       = "out_in_play",
  "sac_fly_double_play"           = "out_in_play",
  "sac_bunt"                      = "out_in_play",
  "walk"                          = "walk",
  "hit_by_pitch"                  = "hit_by_pitch",
  "intent_walk"                   = "walk"
)

lw_raw <- raw %>%
  filter(!is.na(events), events != "") %>%
  mutate(outcome_type = outcome_map[events]) %>%
  filter(!is.na(outcome_type)) %>%
  group_by(outcome_type) %>%
  summarise(lw = mean(delta_run_exp, na.rm = TRUE), n = n()) %>%
  filter(n >= 50) %>%
  select(outcome_type, lw)

dir.create("results", showWarnings = FALSE)
write.csv(lw_raw, "results/linear_weights.csv", row.names = FALSE)
cat("Saved results/linear_weights.csv\n")
print(lw_raw)

# ── 4. Count values from delta_run_exp ───────────────────────────────────────
# Count value = mean remaining run expectancy from this count forward within a PA.
count_vals <- raw %>%
  filter(!is.na(delta_run_exp), !is.na(balls), !is.na(strikes)) %>%
  arrange(game_pk, at_bat_number, pitch_number) %>%
  group_by(game_pk, at_bat_number) %>%
  mutate(remaining_rv = rev(cumsum(rev(delta_run_exp)))) %>%
  ungroup() %>%
  group_by(balls, strikes) %>%
  summarise(expected_run_value = mean(remaining_rv, na.rm = TRUE), n = n()) %>%
  filter(n >= 100) %>%
  select(balls, strikes, expected_run_value)

write.csv(count_vals, "results/count_values.csv", row.names = FALSE)
cat("Saved results/count_values.csv\n")
print(count_vals)
```

- [ ] **Step 2: Run the script and verify outputs**

```bash
Rscript src/data/pull_savant.R
```

Expected (after ~20–40 min):
```
Total rows: ~2000000
Saved data/savant_raw.csv
Saved results/linear_weights.csv
Saved results/count_values.csv
```

Verify:
```bash
# Check all three files exist
ls -lh data/savant_raw.csv results/linear_weights.csv results/count_values.csv

# Check columns printed to stdout contain: launch_speed, launch_angle, delta_run_exp
# Check linear_weights.csv has rows for single, double, triple, home_run, out_in_play, strikeout
# Check count_values.csv has 12 rows (balls 0-3 × strikes 0-2)
```

- [ ] **Step 3: Delete obsolete scripts**

```bash
git rm src/data/pull_data.py src/data/run_values.py
```

- [ ] **Step 4: Commit**

```bash
git add src/data/pull_savant.R data/savant_raw.csv results/linear_weights.csv results/count_values.csv
git commit -m "feat: replace DB pull with sabRmetrics R script; compute LW and count values from delta_run_exp"
```

---

### Task 2: Python swing processing — `src/data/process_savant.py`

**Files:**
- Create: `src/data/process_savant.py`

**Interfaces:**
- Consumes: `data/savant_raw.csv` (from Task 1) — standard Statcast column names
- Produces: `data/swings_2023_2025.csv` — same schema as the old `pull_data.py` output plus `launch_speed` and `launch_angle` columns

**Column name notes:** Statcast CSV from Baseball Savant uses `batter` (int ID), `pitcher` (int ID), `player_name` (pitcher name as "Last, First"), `stand` (batter hand), `p_throws` (pitcher hand), `type` (B/S/X pitch outcome). These get renamed to match the pipeline convention. `batter_full_name` is derived from a player lookup (see Step 1).

- [ ] **Step 1: Create `src/data/process_savant.py`**

```python
"""
Read data/savant_raw.csv from the R pull, rename columns to pipeline convention,
derive outcome flags, compute within-PA lag features, filter to tracked swings.

Output: data/swings_2023_2025.csv
  Same schema as old pull_data.py output plus launch_speed and launch_angle.
"""
from pathlib import Path
import pandas as pd
import numpy as np

RAW_PATH = "data/savant_raw.csv"
OUT_PATH = "data/swings_2023_2025.csv"

print(f"Reading {RAW_PATH}...")
df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"Raw shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")  # inspect on first run to verify rename map

# ── 1. Column renames — Statcast → pipeline convention ───────────────────────
# Verify these on first run against the printed column list above.
rename_map = {
    "batter":        "batter_id",
    "pitcher":       "pitcher_id",
    "player_name":   "pitcher_full_name",   # Statcast player_name = pitcher
    "stand":         "batter_stand",
    "p_throws":      "pitcher_throws",
    "type":          "pitch_outcome",        # B=ball, S=strike, X=in-play
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# batter_full_name: Statcast doesn't include batter name in bulk pull.
# Build from a player lookup derived from the data itself.
# mlb_id → name via pybaseball or a cached table if available.
# Fallback: cast batter_id to string so notebook annotations degrade gracefully.
if "batter_full_name" not in df.columns:
    try:
        from pybaseball import playerid_reverse_lookup
        batter_ids = df["batter_id"].dropna().unique().tolist()
        lookup = playerid_reverse_lookup(batter_ids, key_type="mlbam")
        lookup["batter_full_name"] = lookup["name_first"] + " " + lookup["name_last"]
        lookup = lookup.rename(columns={"key_mlbam": "batter_id"})[["batter_id", "batter_full_name"]]
        df = df.merge(lookup, on="batter_id", how="left")
    except Exception as e:
        print(f"WARNING: could not resolve batter names ({e}). Setting batter_full_name = batter_id.")
        df["batter_full_name"] = df["batter_id"].astype(str)

# ── 2. Derive outcome flags ───────────────────────────────────────────────────
swing_desc = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip",
    "missed_bunt", "bunt_foul_tip",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "foul", "foul_bunt",
}
whiff_desc = {
    "swinging_strike", "swinging_strike_blocked", "missed_bunt", "bunt_foul_tip",
}
contact_desc = {
    "foul", "foul_tip", "foul_bunt",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
}
bip_desc = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}

df["is_swing"]   = df["description"].isin(swing_desc).astype(int)
df["is_whiff"]   = df["description"].isin(whiff_desc).astype(int)
df["is_contact"] = df["description"].isin(contact_desc).astype(int)
df["is_bip"]     = df["description"].isin(bip_desc).astype(int)

df["is_home_run"] = (df["events"] == "home_run").astype(int)
df["is_triple"]   = (df["events"] == "triple").astype(int)
df["is_double"]   = (df["events"] == "double").astype(int)
df["is_single"]   = (df["events"] == "single").astype(int)

def _count_group(balls, strikes):
    if strikes == 2:
        return "two_strike"
    if balls == 3:
        return "three_ball"
    return "other"

df["count_group"] = [
    _count_group(b, s)
    for b, s in zip(df["balls"].fillna(0).astype(int), df["strikes"].fillna(0).astype(int))
]

df["is_same_side_matchup"] = (df["pitcher_throws"] == df["batter_stand"]).astype(int)

# ── 3. Within-PA sequence lag features ───────────────────────────────────────
print("Computing sequence lags...")
df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
g = df.groupby(["game_pk", "at_bat_number"])

df["prev_pitch_type"]    = g["pitch_type"].shift(1)
df["prev_release_speed"] = g["release_speed"].shift(1)
df["prev_plate_x"]       = g["plate_x"].shift(1)
df["prev_plate_z"]       = g["plate_z"].shift(1)
df["velo_delta"]         = df["release_speed"] - df["prev_release_speed"]
df["plate_z_delta"]      = df["plate_z"] - df["prev_plate_z"]
df["plate_x_delta"]      = df["plate_x"] - df["prev_plate_x"]

cur_outcome = pd.Series("other", index=df.index)
cur_outcome[df["pitch_outcome"] == "B"] = "ball"
cur_outcome[(df["pitch_outcome"] == "S") & (df["is_swing"] == 1) & (df["is_whiff"] == 1)] = "whiff"
cur_outcome[(df["pitch_outcome"] == "S") & (df["is_swing"] == 1) & (df["is_whiff"] == 0)] = "foul"
cur_outcome[(df["pitch_outcome"] == "S") & (df["is_swing"] == 0)] = "called_strike"
df["cur_outcome"]  = cur_outcome
df["prev_outcome"] = g["cur_outcome"].shift(1)

# ── 4. Filter to tracked swings ───────────────────────────────────────────────
df = df[
    (df["is_swing"] == 1) &
    df["bat_speed"].notna() &
    (df["bat_speed"] > 40) &
    df["vert_attack_angle"].notna() &
    df["vert_attack_angle"].between(-45, 45)
].copy()

print(f"After swing + tracking filter: {len(df):,} rows")

# ── 5. Verify EV/LA columns are present ──────────────────────────────────────
assert "launch_speed" in df.columns, "launch_speed missing — check savant_raw.csv columns"
assert "launch_angle" in df.columns, "launch_angle missing — check savant_raw.csv columns"
bip_rows = df[df["is_bip"] == 1]
ev_coverage = bip_rows["launch_speed"].notna().mean()
print(f"EV coverage on BIP rows: {ev_coverage:.1%}")
assert ev_coverage > 0.80, f"Expected >80% EV coverage on BIP, got {ev_coverage:.1%}"

# ── 6. Save ───────────────────────────────────────────────────────────────────
Path("data").mkdir(exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {OUT_PATH}")
print(f"Shape: {df.shape}")
print(f"Years: {sorted(df['game_year'].unique())}")
print(f"Batters: {df['batter_id'].nunique():,}")
print(f"Pitchers: {df['pitcher_id'].nunique():,}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
```

- [ ] **Step 2: Install pybaseball if not present**

```bash
.venv/bin/pip install pybaseball
```

- [ ] **Step 3: Run and verify**

```bash
.venv/bin/python src/data/process_savant.py
```

Expected output:
```
Raw shape: (~2000000, N)
After swing + tracking filter: ~760000 rows
EV coverage on BIP rows: >80%
Saved data/swings_2023_2025.csv
```

Verify the output CSV has `launch_speed` and `launch_angle`:
```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('data/swings_2023_2025.csv', nrows=5)
assert 'launch_speed' in df.columns
assert 'launch_angle' in df.columns
assert 'batter_id' in df.columns
assert 'pitcher_throws' in df.columns
print('All required columns present')
print(df[['launch_speed','launch_angle','batter_id','pitcher_throws']].dtypes)
"
```

- [ ] **Step 4: Commit**

```bash
git add src/data/process_savant.py data/swings_2023_2025.csv
git commit -m "feat: add process_savant.py; derive outcome flags, lags, swing filter from savant_raw.csv"
```

---

### Task 3: Phase B EV/LA outcome models — `src/models/causal_models.py`

**Files:**
- Modify: `src/models/causal_models.py`

**Interfaces:**
- Consumes: `df` with `is_bip`, `launch_speed`, `launch_angle`, `delta_run_exp`, `OUTCOME_FEATURES` columns
- Produces:
  - `fit_outcome_models(df)` now returns `(bip_model, foul_model, exit_velo_model, launch_angle_model, ev_la_rv_model, whiff_rv)` — 6 items instead of 4
  - `_xrv_from_shape(df, bip_model, foul_model, exit_velo_model, launch_angle_model, ev_la_rv_model, whiff_rv, foul_rv, ...)` — updated signature
  - `disruption_tax_split(df, bip_model, foul_model, exit_velo_model, launch_angle_model, ev_la_rv_model, whiff_rv, foul_rv, mediator_models, ...)` — updated signature; adds `ev_realized`, `la_realized`, `ev_intended`, `la_intended`, `ev_suppression` to output DataFrame
  - `indirect_effect(mediator_models, bip_model, foul_model, exit_velo_model, launch_angle_model, ev_la_rv_model, whiff_rv, foul_rv, df, ...)` — updated signature

- [ ] **Step 1: Replace `xwoba_model` with EV/LA models in `fit_outcome_models()`**

Find `fit_outcome_models` (line 97). Replace the full function:

```python
def fit_outcome_models(df, commit_ms=150):
    """Fit all outcome components. Returns 6-tuple:
    (bip_model, foul_model, exit_velo_model, launch_angle_model, ev_la_rv_model, whiff_rv).

    BIP channel is now a two-stage chain:
      swing features → (exit_velo_model, launch_angle_model) → ev_la_rv_model → run value
    This makes the causal chain explicit: post-commit movement → swing deviation →
    exit velocity / launch angle → run value.

    Whiff and foul channels are unchanged.
    """
    _xgb_kw = dict(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )

    d = df.copy()
    if "is_foul" not in d.columns:
        d["is_foul"] = ((d["is_contact"] == 1) & (d["is_bip"] != 1)).astype(int)

    # BIP / foul probability models — unchanged
    d_bip = d[OUTCOME_FEATURES + ["is_bip"]].dropna()
    bip_model = XGBClassifier(**_xgb_kw).fit(
        d_bip[OUTCOME_FEATURES], d_bip["is_bip"], verbose=False
    )

    d_foul = d.loc[d["is_bip"] == 0, OUTCOME_FEATURES + ["is_foul"]].dropna()
    foul_model = XGBClassifier(**_xgb_kw).fit(
        d_foul[OUTCOME_FEATURES], d_foul["is_foul"], verbose=False
    )

    # Exit velocity model — predicts launch_speed (mph) on BIP swings
    d_ev = df.loc[
        (df["is_bip"] == 1) & df["launch_speed"].notna(),
        OUTCOME_FEATURES + ["launch_speed"],
    ].dropna()
    exit_velo_model = XGBRegressor(**_xgb_kw).fit(
        d_ev[OUTCOME_FEATURES], d_ev["launch_speed"], verbose=False
    )

    # Launch angle model — predicts launch_angle (degrees) on BIP swings
    d_la = df.loc[
        (df["is_bip"] == 1) & df["launch_angle"].notna(),
        OUTCOME_FEATURES + ["launch_angle"],
    ].dropna()
    launch_angle_model = XGBRegressor(**_xgb_kw).fit(
        d_la[OUTCOME_FEATURES], d_la["launch_angle"], verbose=False
    )

    # EV/LA → run value model — smooth surface from (launch_speed, launch_angle) to delta_run_exp
    d_rv = df.loc[
        (df["is_bip"] == 1) & df["launch_speed"].notna() & df["delta_run_exp"].notna(),
        ["launch_speed", "launch_angle", "delta_run_exp"],
    ].dropna()
    ev_la_rv_model = XGBRegressor(**_xgb_kw).fit(
        d_rv[["launch_speed", "launch_angle"]], d_rv["delta_run_exp"], verbose=False
    )

    whiffs = df[(df["is_swing"] == 1) & (df["is_whiff"] == 1) &
                df["delta_run_exp"].notna()]
    whiff_rv = whiffs.groupby(["balls", "strikes"])["delta_run_exp"].mean().to_dict()
    whiff_rv["default"] = whiffs["delta_run_exp"].mean()

    return bip_model, foul_model, exit_velo_model, launch_angle_model, ev_la_rv_model, whiff_rv
```

- [ ] **Step 2: Update `_xrv_from_shape()` signature and BIP channel**

Find `_xrv_from_shape` (line 142). Replace the full function:

```python
def _xrv_from_shape(df, bip_model, foul_model, exit_velo_model, launch_angle_model,
                    ev_la_rv_model, whiff_rv, foul_rv,
                    commit_ms=150, zero_angular=False, zero_spatial=False, empirical=True):
    """Predict per-swing xRV for one of three counterfactual scenarios.

    BIP channel: swing features → predicted EV/LA → ev_la_rv_model → E[run value | BIP].
    Whiff and foul channels unchanged.
    """
    prefix = f"pc{commit_ms}"
    emp    = "emp_" if empirical else ""
    x_proj = f"{prefix}_{emp}x_proj"
    z_proj = f"{prefix}_{emp}z_proj"

    score = df[["balls", "strikes"]].copy()
    score["plate_x"] = df[x_proj] if zero_spatial else df["plate_x"]
    score["plate_z"] = df[z_proj] if zero_spatial else df["plate_z"]
    for dev in ANGULAR_DEVS:
        score[dev] = 0.0 if zero_angular else df[dev]

    feat = score[OUTCOME_FEATURES]
    p_bip            = pd.Series(bip_model.predict_proba(feat)[:, 1],  index=df.index)
    p_foul_given_not = pd.Series(foul_model.predict_proba(feat)[:, 1], index=df.index)

    ev_pred      = exit_velo_model.predict(feat)
    la_pred      = launch_angle_model.predict(feat)
    ev_la_feat   = np.column_stack([ev_pred, la_pred])
    e_rv_bip     = pd.Series(ev_la_rv_model.predict(ev_la_feat), index=df.index)

    def _count_lookup(rv_table):
        return df.apply(
            lambda r: rv_table.get((int(r["balls"]), int(r["strikes"])),
                                   rv_table["default"]),
            axis=1,
        )

    whiff_vals = _count_lookup(whiff_rv)
    foul_vals  = _count_lookup(foul_rv)

    p_foul  = (1 - p_bip) * p_foul_given_not
    p_whiff = (1 - p_bip) * (1 - p_foul_given_not)

    return p_bip * e_rv_bip + p_foul * foul_vals + p_whiff * whiff_vals
```

- [ ] **Step 3: Update `disruption_tax_split()` signature and add EV output columns**

Find `disruption_tax_split` (line 195). Replace the full function signature and the `args` tuple, and add EV column computation before `return`:

```python
def disruption_tax_split(df, bip_model, foul_model, exit_velo_model, launch_angle_model,
                         ev_la_rv_model, whiff_rv, foul_rv,
                         mediator_models, commit_ms=150, empirical=True):
    """Full disruption tax with two-channel distortion/selection decomposition.
    Signature updated: xwoba_model replaced by exit_velo_model, launch_angle_model, ev_la_rv_model.
    New output columns: ev_realized, la_realized, ev_intended, la_intended, ev_suppression.
    All existing output columns (disruption_tax, distortion_tax, selection_tax, ...) unchanged.
    """
    prefix    = f"pc{commit_ms}"
    emp       = "emp_" if empirical else ""
    dev_x_col = f"{prefix}_{emp}dev_x"
    dev_z_col = f"{prefix}_{emp}dev_z"

    args = (df, bip_model, foul_model, exit_velo_model, launch_angle_model, ev_la_rv_model,
            whiff_rv, foul_rv)

    xrv_realized = _xrv_from_shape(*args, commit_ms=commit_ms,
                                    zero_angular=False, zero_spatial=False, empirical=empirical)
    xrv_spatial  = _xrv_from_shape(*args, commit_ms=commit_ms,
                                    zero_angular=True,  zero_spatial=False, empirical=empirical)
    xrv_intended = _xrv_from_shape(*args, commit_ms=commit_ms,
                                    zero_angular=True,  zero_spatial=True,  empirical=empirical)

    disruption_tax     = xrv_realized - xrv_intended
    spatial_distortion = xrv_spatial  - xrv_intended
    angular_disruption = xrv_realized - xrv_spatial

    attr_cols = {}
    for dev_col, model in mediator_models.items():
        a_x = model.params.get(dev_x_col, 0.0)
        a_z = model.params.get(dev_z_col, 0.0)
        dist_dev = a_x * df[dev_x_col] + a_z * df[dev_z_col]
        attr_cols[f"distortion_dev_{dev_col}"] = dist_dev
        attr_cols[f"selection_dev_{dev_col}"]  = df[dev_col] - dist_dev

    attr = pd.DataFrame(attr_cols, index=df.index)
    distortion_sq = sum(attr[f"distortion_dev_{d}"] ** 2 for d in ANGULAR_DEVS)
    total_sq      = sum(df[d] ** 2 for d in ANGULAR_DEVS)
    angular_distortion_share = np.where(
        total_sq > 1e-8,
        np.clip(distortion_sq / total_sq, 0.0, 1.0),
        np.nan,
    )

    distortion_tax = spatial_distortion + angular_disruption * angular_distortion_share
    selection_tax  = angular_disruption * (1 - angular_distortion_share)
    distortion_share = np.where(
        disruption_tax.abs() > 1e-8,
        np.clip(distortion_tax / disruption_tax, 0.0, 1.0),
        np.nan,
    )

    out = df.copy()
    out["disruption_tax"]         = disruption_tax
    out["spatial_distortion_tax"] = spatial_distortion
    out["distortion_tax"]         = distortion_tax
    out["selection_tax"]          = selection_tax
    out["distortion_share"]       = distortion_share
    out["_xrv_intended"]          = xrv_intended.values

    # ── EV/LA counterfactual columns (BIP only; NaN on whiffs/fouls) ─────────
    bip_mask = df["is_bip"] == 1

    # Realized EV/LA — from raw data
    out["ev_realized"] = np.where(bip_mask, df["launch_speed"],  np.nan)
    out["la_realized"] = np.where(bip_mask, df["launch_angle"],  np.nan)

    # Intended EV/LA — model prediction under intended swing (zero angular dev, zero spatial)
    intended_feat = df[["balls", "strikes"]].copy()
    prefix_emp    = f"pc{commit_ms}_{'emp_' if empirical else ''}x_proj"
    prefix_emp_z  = f"pc{commit_ms}_{'emp_' if empirical else ''}z_proj"
    intended_feat["plate_x"] = df[prefix_emp]
    intended_feat["plate_z"] = df[prefix_emp_z]
    for dev in ANGULAR_DEVS:
        intended_feat[dev] = 0.0

    ev_int = exit_velo_model.predict(intended_feat[OUTCOME_FEATURES])
    la_int = launch_angle_model.predict(intended_feat[OUTCOME_FEATURES])
    out["ev_intended"] = np.where(bip_mask, ev_int,  np.nan)
    out["la_intended"] = np.where(bip_mask, la_int,  np.nan)

    # EV suppression: realized − intended (negative = pitcher suppressed EV)
    out["ev_suppression"] = np.where(
        bip_mask,
        out["ev_realized"] - out["ev_intended"],
        np.nan,
    )

    return out
```

- [ ] **Step 4: Update `indirect_effect()` signature**

Find `indirect_effect` (line 282). Update the function signature and the `_mean_xrv` helper:

```python
def indirect_effect(mediator_models, bip_model, foul_model, exit_velo_model,
                    launch_angle_model, ev_la_rv_model, whiff_rv, foul_rv,
                    df, commit_ms=150, eps=0.5, empirical=True):
    """Numerical indirect effect — signature updated for EV/LA chain."""
    prefix = f"pc{commit_ms}"
    emp    = "emp_" if empirical else ""
    dev_x  = f"{prefix}_{emp}dev_x"
    dev_z  = f"{prefix}_{emp}dev_z"

    d = df[OUTCOME_FEATURES + [dev_x, dev_z]].dropna()

    def _mean_xrv(score_df):
        feat  = score_df[OUTCOME_FEATURES]
        p_bip = bip_model.predict_proba(feat)[:, 1]
        p_fc  = foul_model.predict_proba(feat)[:, 1]
        ev_pred    = exit_velo_model.predict(feat)
        la_pred    = launch_angle_model.predict(feat)
        e_rv_bip   = ev_la_rv_model.predict(np.column_stack([ev_pred, la_pred]))
        keys  = list(zip(score_df["balls"].astype(int), score_df["strikes"].astype(int)))
        w_rv  = np.array([whiff_rv.get(k, whiff_rv["default"]) for k in keys])
        f_rv  = np.array([foul_rv.get(k,  foul_rv["default"])  for k in keys])
        p_f   = (1 - p_bip) * p_fc
        p_w   = (1 - p_bip) * (1 - p_fc)
        return (p_bip * e_rv_bip + p_f * f_rv + p_w * w_rv).mean()

    rows = []
    for dev_col, med_model in mediator_models.items():
        a_x = med_model.params.get(dev_x, np.nan)
        a_z = med_model.params.get(dev_z, np.nan)

        score_up = d[OUTCOME_FEATURES].copy(); score_up[dev_col] += eps
        score_dn = d[OUTCOME_FEATURES].copy(); score_dn[dev_col] -= eps

        grad_xrv = (_mean_xrv(score_up) - _mean_xrv(score_dn)) / (2 * eps)

        rows.append({
            "dev_col":    dev_col,
            "a_x":        a_x,
            "a_z":        a_z,
            "grad_xrv":   grad_xrv,
            "indirect_x": a_x * grad_xrv,
            "indirect_z": a_z * grad_xrv,
        })

    return pd.DataFrame(rows).set_index("dev_col")
```

- [ ] **Step 5: Smoke-test the reworked functions with synthetic data**

```bash
.venv/bin/python -c "
import numpy as np, pandas as pd
from src.models.causal_models import fit_outcome_models, disruption_tax_split, OUTCOME_FEATURES, ANGULAR_DEVS

rng = np.random.default_rng(0)
n = 500
df = pd.DataFrame({
    'vert_attack_angle_dev': rng.normal(0, 2, n),
    'horz_attack_angle_dev': rng.normal(0, 2, n),
    'swing_path_tilt_dev':   rng.normal(0, 2, n),
    'plate_x':  rng.uniform(-1.5, 1.5, n),
    'plate_z':  rng.uniform(1.0, 4.0, n),
    'balls':    rng.integers(0, 4, n),
    'strikes':  rng.integers(0, 3, n),
    'is_bip':   rng.integers(0, 2, n),
    'is_contact': rng.integers(0, 2, n),
    'is_whiff':  rng.integers(0, 2, n),
    'is_swing':  np.ones(n, dtype=int),
    'launch_speed': rng.uniform(60, 110, n),
    'launch_angle': rng.uniform(-30, 50, n),
    'delta_run_exp': rng.uniform(-0.5, 0.5, n),
    'pc150_emp_dev_x': rng.normal(0, 0.05, n),
    'pc150_emp_dev_z': rng.normal(0, 0.05, n),
    'pc150_emp_x_proj': rng.uniform(-1.5, 1.5, n),
    'pc150_emp_z_proj': rng.uniform(1.0, 4.0, n),
})
# BIP rows need non-null EV/LA
df.loc[df['is_bip'] == 0, ['launch_speed', 'launch_angle']] = np.nan

result = fit_outcome_models(df)
assert len(result) == 6, f'Expected 6 returns, got {len(result)}'
bip_m, foul_m, ev_m, la_m, ev_la_rv_m, whiff_rv = result

# Smoke-test prediction shapes
feat = df[OUTCOME_FEATURES].fillna(0)
ev_pred = ev_m.predict(feat)
la_pred = la_m.predict(feat)
assert len(ev_pred) == n
assert len(la_pred) == n

print('fit_outcome_models: PASS (6 returns, correct prediction shapes)')
"
```

Expected: `fit_outcome_models: PASS (6 returns, correct prediction shapes)`

- [ ] **Step 6: Commit**

```bash
git add src/models/causal_models.py
git commit -m "feat: replace xwoba_model with exit_velo_model + launch_angle_model + ev_la_rv_model in Phase B"
```

---

### Task 4: Orchestrator, leaderboards, notebook, CLAUDE.md

**Files:**
- Modify: `src/models/run_pipeline.py`
- Modify: `src/viz/generate_results.ipynb`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: 6-return `fit_outcome_models()` from Task 3; `ev_suppression` column in `results` DataFrame
- Produces:
  - `results/xrv_causal.parquet` with additive columns: `ev_realized`, `la_realized`, `ev_intended`, `la_intended`, `ev_suppression`
  - `results/distortion_pitcher.csv` and `results/distortion_batter.csv` with additive columns: `mean_ev_suppression`, `mean_la_change`
  - New notebook cell `ev-suppression-scatter`

- [ ] **Step 1: Update `run_pipeline.py` — outcome model call site**

Find line 219:
```python
bip_model, foul_model, xwoba_model, _ = _B.fit_outcome_models(swings, commit_ms=commit_ms)
```

Replace with:
```python
bip_model, foul_model, exit_velo_model, launch_angle_model, ev_la_rv_model, _ = \
    _B.fit_outcome_models(swings, commit_ms=commit_ms)
```

- [ ] **Step 2: Update `run_pipeline.py` — `disruption_tax_split()` call site**

Find the `disruption_tax_split` call (around line 232). Replace:
```python
results = _B.disruption_tax_split(
    swings,
    bip_model,
    foul_model,
    xwoba_model,
    whiff_rv,
    foul_rv,
    mediator_models,
    commit_ms=commit_ms,
)
```

With:
```python
results = _B.disruption_tax_split(
    swings,
    bip_model,
    foul_model,
    exit_velo_model,
    launch_angle_model,
    ev_la_rv_model,
    whiff_rv,
    foul_rv,
    mediator_models,
    commit_ms=commit_ms,
)
```

- [ ] **Step 3: Update `run_pipeline.py` — `indirect_effect()` call site**

Find the `indirect_effect` call (around line 265). Replace:
```python
ie = _B.indirect_effect(
    mediator_models, bip_model, foul_model, xwoba_model,
    whiff_rv, foul_rv, swings, commit_ms=commit_ms,
)
```

With:
```python
ie = _B.indirect_effect(
    mediator_models, bip_model, foul_model,
    exit_velo_model, launch_angle_model, ev_la_rv_model,
    whiff_rv, foul_rv, swings, commit_ms=commit_ms,
)
```

- [ ] **Step 4: Update `run_pipeline.py` — save EV columns in `save_cols`**

Find (around line 278):
```python
tax_cols = ["disruption_tax", "adjusted_disruption_tax", "spatial_distortion_tax",
            "distortion_tax", "selection_tax", "distortion_share",
            "miss_distortion_tax", "decision_cost"]
```

Replace with:
```python
tax_cols = ["disruption_tax", "adjusted_disruption_tax", "spatial_distortion_tax",
            "distortion_tax", "selection_tax", "distortion_share",
            "miss_distortion_tax", "decision_cost",
            "ev_realized", "la_realized", "ev_intended", "la_intended", "ev_suppression"]
```

- [ ] **Step 5: Update `run_pipeline.py` — leaderboard aggregations**

Find the leaderboard aggregation block (around line 291). Replace the `.agg(...)` call:

```python
agg = (
    results[results["distortion_tax"].notna()]
    .groupby(group_col)
    .agg(
        n_swings=("disruption_tax", "size"),
        mean_disruption_tax=("disruption_tax", "mean"),
        mean_spatial_distortion_tax=("spatial_distortion_tax", "mean"),
        mean_distortion_tax=("distortion_tax", "mean"),
        mean_selection_tax=("selection_tax", "mean"),
        mean_distortion_share=("distortion_share", "mean"),
        mean_adjusted_disruption_tax=("adjusted_disruption_tax", "mean"),
        mean_miss_distortion_tax=("miss_distortion_tax", "mean"),
        mean_decision_cost=("decision_cost", "mean"),
        mean_ev_suppression=("ev_suppression", "mean"),
        mean_la_change=("ev_suppression", lambda x: (
            results.loc[x.index, "la_realized"] - results.loc[x.index, "la_intended"]
        ).mean()),
    )
    .query("n_swings >= 50")
    .sort_values("mean_distortion_tax")
)
```

- [ ] **Step 6: Update `run_pipeline.py` — joblib save block**

Find the `joblib.dump` call (around line 309). Replace `xwoba_model` key with new models:

```python
joblib.dump(
    {
        "mediator_models":    mediator_models,
        "bip_model":          bip_model,
        "foul_model":         foul_model,
        "exit_velo_model":    exit_velo_model,
        "launch_angle_model": launch_angle_model,
        "ev_la_rv_model":     ev_la_rv_model,
        "whiff_rv":           whiff_rv,
        "foul_rv":            foul_rv,
        "whiff_miss_model":   whiff_miss_model,
        "contact_miss_model": contact_miss_model,
        "miss_rv_slope":      miss_rv_slope,
        "indirect_effects":   ie,
    },
    "models/causal_models.joblib",
)
```

Also update the `return` dict at the bottom of `run()` to match.

- [ ] **Step 7: Update `run_pipeline.py` — remove `add_xwoba()` call**

Find (around line 167):
```python
swings = add_xwoba(swings, lw_path)
```

Remove this line. The `xwoba` column is no longer used as a training target — `launch_speed` and `launch_angle` are used instead. `delta_run_exp` (already in the data) is used by `ev_la_rv_model`.

Also remove the `add_xwoba` function definition from `run_pipeline.py` (lines 71–99). It is no longer needed.

- [ ] **Step 8: Add EV suppression scatter cell to `generate_results.ipynb`**

Append a new cell with id `"ev-suppression-scatter"` after the existing `"economic-translation"` cell:

```python
MIN_SWINGS = 200  # already defined in cell 0

pitchers = pd.read_csv("results/distortion_pitcher.csv").query(f"n_swings >= {MIN_SWINGS}")
pitcher_names = (
    sxrv_extended()[["pitcher_id", "pitcher_full_name"]]
    .drop_duplicates("pitcher_id")
)
pitchers = pitchers.merge(pitcher_names, on="pitcher_id", how="left")

# Drop rows where either metric is missing (pitchers with no BIP in sample)
plot_df = pitchers.dropna(subset=["mean_ev_suppression", "mean_distortion_tax"])

fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
ax.set_facecolor(BG)
ax.scatter(
    plot_df["mean_ev_suppression"],
    plot_df["mean_distortion_tax"],
    alpha=0.35, s=18, color=BLUE,
)

# Annotate top-10 by distortion_tax (most negative = best)
top10 = plot_df.nsmallest(10, "mean_distortion_tax")
for _, row in top10.iterrows():
    last = str(row["pitcher_full_name"]).split()[-1] if pd.notna(row["pitcher_full_name"]) else ""
    ax.annotate(
        last,
        xy=(row["mean_ev_suppression"], row["mean_distortion_tax"]),
        xytext=(4, 2), textcoords="offset points",
        fontsize=7, color=DARK, alpha=0.8,
    )

ax.axvline(0, color=MID, lw=0.6, linestyle="--")
ax.axhline(0, color=MID, lw=0.6, linestyle="--")
ax.set_xlabel("Mean EV Suppression (mph, realized − intended)", color=DARK)
ax.set_ylabel("Mean Distortion Tax (xRV/swing)", color=DARK)
ax.set_title(
    "EV Suppression vs. Distortion Tax by Pitcher\n"
    f"Each dot = one pitcher  |  min {MIN_SWINGS} swings",
    color=DARK, fontweight="bold",
)
_spine_clean(ax)
ax.tick_params(colors=DARK)
fig.tight_layout()

out = "results/plots/applications/ev_suppression_scatter.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out}")

assert Path(out).exists()
assert len(plot_df) > 100, f"Expected >100 pitchers, got {len(plot_df)}"
# EV suppression and distortion tax should be negatively correlated
# (more EV suppression → more distortion tax)
from scipy.stats import pearsonr
r_ev, _ = pearsonr(plot_df["mean_ev_suppression"], plot_df["mean_distortion_tax"])
assert r_ev < 0, f"Expected negative correlation, got r={r_ev:.3f}"
print(f"EV suppression vs distortion tax r={r_ev:.3f}  PASS")
```

- [ ] **Step 9: Update `CLAUDE.md`**

Replace the "Running the pipeline" section:

```markdown
## Running the pipeline

```bash
# Step 1 — R data pull (requires R ≥ 4.0, sabRmetrics, tidyverse)
# Pulls 2023-2025 Statcast, writes data/savant_raw.csv + results/linear_weights.csv + results/count_values.csv
Rscript src/data/pull_savant.R

# Step 2 — Python swing processing
.venv/bin/python src/data/process_savant.py     # data/swings_2023_2025.csv

# Step 3 — Pre/post-commit split (unchanged)
.venv/bin/python src/data/precommit_split.py    # data/swings_precommit.parquet

# Step 4 — Full causal pipeline
.venv/bin/python src/models/run_pipeline.py     # results/xrv_causal.parquet + leaderboards
```

`pull_data.py` and `run_values.py` have been deleted — do not use them.
```

Also update the Outputs table to add:
```
| `ev_realized`, `la_realized`  | Actual EV/LA from raw data (NaN on non-BIP) |
| `ev_intended`, `la_intended`  | Model-predicted EV/LA under intended swing   |
| `ev_suppression`              | ev_realized − ev_intended (negative = pitcher suppressed EV) |
| `mean_ev_suppression`         | Pitcher/batter-level mean EV suppression (in leaderboard CSVs) |
```

- [ ] **Step 10: Smoke-test the full orchestrator (dry run with --skip-phase-a)**

```bash
.venv/bin/python src/models/run_pipeline.py --skip-phase-a --method vi
```

Expected: pipeline completes without AttributeError or KeyError; `results/xrv_causal.parquet` written; verify new columns:

```bash
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('results/xrv_causal.parquet')
for col in ['ev_realized', 'la_realized', 'ev_intended', 'la_intended', 'ev_suppression']:
    assert col in df.columns, f'Missing: {col}'
    bip_rows = df[df['ev_realized'].notna()]
    print(f'{col}: {bip_rows[col].describe().to_dict()}')
print('All EV/LA columns present PASS')

pitchers = pd.read_csv('results/distortion_pitcher.csv')
assert 'mean_ev_suppression' in pitchers.columns
assert 'mean_la_change' in pitchers.columns
print('Leaderboard columns PASS')
"
```

- [ ] **Step 11: Commit**

```bash
git add src/models/run_pipeline.py src/viz/generate_results.ipynb CLAUDE.md
git commit -m "feat: wire EV/LA models through orchestrator; add ev_suppression to outputs and leaderboards; add EV scatter notebook cell"
```
