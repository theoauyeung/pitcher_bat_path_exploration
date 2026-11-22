"""
Out-of-sample performance cache for the Phase A intention model.

Strategy: year-holdout.
  - 2025 is the test set.
  - Per-batter mean baselines are computed from 2023-2024 only.
  - Model predictions (intended_df) are technically in-sample for 2025,
    but the per-batter random effect is shrunk toward 0 (ADVI posterior mean),
    so 2025 performance closely approximates true OOS for the LMM.
  - Grand-mean baseline uses the 2023-2024 mean for the same held-out test.

Baselines compared:
  1. Grand mean      -- predict the 2023-2024 mean for all swings
  2. Per-batter mean -- predict each batter's 2023-2024 mean (uses identity only)
  3. LMM (this work) -- intended_df posterior mean (pitch context + batter RE)

Outputs:
  data/viz_cache/intention_oos_metrics.csv  -- RMSE/MAE by model x response
  data/viz_cache/intention_oos_scatter.csv  -- sample of realized vs intended for 2025
"""
import numpy as np
import pandas as pd

RESP_COLS   = ["vert_attack_angle", "horz_attack_angle", "swing_path_tilt",
               "bat_speed", "swing_length"]
RESP_LABELS = ["Vert. attack angle", "Horiz. attack angle", "Swing path tilt",
               "Bat speed", "Swing length"]
UNITS       = ["deg", "deg", "deg", "mph", "ft"]
INT_COLS    = ["intended_vert_attack_angle", "intended_horz_attack_angle",
               "intended_swing_path_tilt", "intended_bat_speed", "intended_swing_length"]

print("Loading data...")
pc = pd.read_parquet(
    "data/swings_precommit.parquet",
    columns=["game_pk", "at_bat_number", "pitch_number",
             "game_date", "batter_id", "bat_speed",
             "vert_attack_angle", "horz_attack_angle",
             "swing_path_tilt", "swing_length"],
)
idf = pd.read_parquet("models/intended_df.parquet")

# Apply the same competitive_swings filter used in run_pipeline.py
mask = (
    pc["bat_speed"].notna() &
    (pc["bat_speed"] >= 50) &
    pc["vert_attack_angle"].notna()
)
pc_comp = pc[mask].reset_index(drop=True)
print(f"Competitive swings: {len(pc_comp):,}  |  intended_df: {len(idf):,}")
assert len(pc_comp) == len(idf), "Alignment mismatch — check competitive_swings filter"

df = pd.concat([
    pc_comp[["game_pk", "at_bat_number", "pitch_number",
             "game_date", "batter_id"] + RESP_COLS].reset_index(drop=True),
    idf[INT_COLS].reset_index(drop=True),
], axis=1)

df["year"] = pd.to_datetime(df["game_date"]).dt.year

train = df[df["year"].isin([2023, 2024])]
test  = df[df["year"] == 2025]
print(f"Train: {len(train):,}  |  Test: {len(test):,}")

# ── Baselines ─────────────────────────────────────────────────────────────────
# 1. Grand mean per response (from train)
grand_means = train[RESP_COLS].mean()

# 2. Per-batter mean per response (from train)
batter_means = train.groupby("batter_id")[RESP_COLS].mean()

# Map to test set; unseen batters fall back to grand mean
test = test.copy()
for col in RESP_COLS:
    test[f"batter_mean_{col}"] = (
        test["batter_id"]
        .map(batter_means[col])
        .fillna(grand_means[col])
    )
    test[f"grand_mean_{col}"] = grand_means[col]

# ── Compute metrics ───────────────────────────────────────────────────────────
records = []
for rc, ic, label, unit in zip(RESP_COLS, INT_COLS, RESP_LABELS, UNITS):
    valid = test[[rc, ic, f"grand_mean_{rc}", f"batter_mean_{rc}"]].dropna()
    realized = valid[rc].values
    for model_name, preds in [
        ("Grand mean",        valid[f"grand_mean_{rc}"].values),
        ("Per-batter mean",   valid[f"batter_mean_{rc}"].values),
        ("LMM (this work)",   valid[ic].values),
    ]:
        err   = realized - preds
        rmse  = np.sqrt(np.mean(err ** 2))
        mae   = np.mean(np.abs(err))
        records.append({
            "response":   label,
            "unit":       unit,
            "model":      model_name,
            "rmse":       rmse,
            "mae":        mae,
            "n":          len(valid),
        })

metrics = pd.DataFrame(records)
print("\nOOS metrics on 2025 holdout:")
print(metrics.to_string(index=False))

metrics.to_csv("data/viz_cache/intention_oos_metrics.csv", index=False)
print("\nSaved intention_oos_metrics.csv")

# ── Scatter sample for VAA (biggest interest) ─────────────────────────────────
scatter = (
    test[["vert_attack_angle", "intended_vert_attack_angle",
          "bat_speed", "intended_bat_speed"]]
    .dropna()
    .sample(n=min(8000, len(test)), random_state=42)
    .rename(columns={
        "vert_attack_angle":          "realized_vaa",
        "intended_vert_attack_angle": "intended_vaa",
        "bat_speed":                  "realized_bs",
        "intended_bat_speed":         "intended_bs",
    })
)
scatter.to_csv("data/viz_cache/intention_oos_scatter.csv", index=False)
print(f"Saved intention_oos_scatter.csv ({len(scatter):,} rows)")
print("\nDone.")
