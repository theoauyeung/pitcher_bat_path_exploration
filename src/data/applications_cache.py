"""
Build application cache CSVs for results.ipynb.

Outputs (data/viz_cache/):
  app_sequencing.csv        -- distortion_share by (prev_pitch_type, pitch_type) heatmap
  app_batter_vulnerability.csv  -- top batters by mean distortion_tax overall + by pitch type
  knn_k_pitch_spread.csv    -- mean |emp_dev_z| by pitch type + K=50 spread vs linear
"""
import numpy as np
import pandas as pd

KEY = ["game_pk", "at_bat_number", "pitch_number"]
PITCH_NAMES = {
    "FF": "4-Seam",   "SI": "Sinker",  "FC": "Cutter",
    "CH": "Changeup", "FS": "Splitter",
    "SL": "Slider",   "CU": "Curveball", "ST": "Sweeper", "KC": "Knuckle-curve",
}
MIN_SWINGS_BATTER = 300
MIN_SWINGS_SEQ    = 50

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")

xrv = pd.read_parquet(
    "results/xrv_causal.parquet",
    columns=KEY + [
        "batter_id", "pitcher_id", "pitch_type",
        "distortion_tax", "selection_tax", "disruption_tax",
        "distortion_share", "ev_suppression",
    ],
).dropna(subset=["distortion_tax", "selection_tax"])

sw = pd.read_csv(
    "data/swings_2023_2025.csv",
    usecols=KEY + ["batter_full_name", "pitcher_full_name", "prev_pitch_type"],
)

df = xrv.merge(sw, on=KEY, how="inner")
print(f"Merged df: {len(df):,} rows")

# ── 1. Pitch sequencing: distortion_share by prev -> current type ─────────────
print("\nBuilding sequencing cache...")

TYPES = list(PITCH_NAMES.keys())

seq = (
    df[
        df["pitch_type"].isin(TYPES) &
        df["prev_pitch_type"].isin(TYPES)
    ]
    .groupby(["prev_pitch_type", "pitch_type"], observed=True)
    .agg(
        mean_distortion_share=("distortion_share", "mean"),
        mean_distortion_tax=("distortion_tax", "mean"),
        n=("distortion_share", "count"),
    )
    .reset_index()
    .query("n >= @MIN_SWINGS_SEQ")
)
seq["prev_pitch_name"] = seq["prev_pitch_type"].map(PITCH_NAMES)
seq["pitch_name"]      = seq["pitch_type"].map(PITCH_NAMES)

seq.to_csv("data/viz_cache/app_sequencing.csv", index=False)
print(f"Saved app_sequencing.csv ({len(seq)} pairs)")

# Top sequencing boosts: same-type vs cross-type
same = seq[seq["prev_pitch_type"] == seq["pitch_type"]]["mean_distortion_share"].mean()
cross = seq[seq["prev_pitch_type"] != seq["pitch_type"]]["mean_distortion_share"].mean()
print(f"  Same-type mean distortion_share = {same:.3f}")
print(f"  Cross-type mean distortion_share = {cross:.3f}")

# ── 2. Batter vulnerability overall + by pitch type ───────────────────────────
print("\nBuilding batter vulnerability cache...")

# Overall batter leaderboard
batter_overall = (
    df.groupby(["batter_id", "batter_full_name"], observed=True)
    .agg(
        n_swings=("distortion_share", "count"),
        mean_distortion_share=("distortion_share", "mean"),
        mean_distortion_tax=("distortion_tax", "mean"),
        mean_ev_suppression=("ev_suppression", "mean"),
    )
    .reset_index()
    .query("n_swings >= @MIN_SWINGS_BATTER")
    .sort_values("mean_distortion_share", ascending=False)
)

# Batter vulnerability by pitch type (top + bottom 15 batters, by pitch type)
batter_pt = (
    df[df["pitch_type"].isin(TYPES)]
    .groupby(["batter_id", "batter_full_name", "pitch_type"], observed=True)
    .agg(
        n_swings=("distortion_share", "count"),
        mean_distortion_share=("distortion_share", "mean"),
        mean_distortion_tax=("distortion_tax", "mean"),
    )
    .reset_index()
    .query("n_swings >= 50")
)
batter_pt["pitch_name"] = batter_pt["pitch_type"].map(PITCH_NAMES)

# Keep top/bottom 10 batters per pitch type for visualization
top_bottom = (
    batter_pt.groupby("pitch_type", group_keys=False, observed=True)
    .apply(lambda g: g.nlargest(10, "mean_distortion_share"))
)

batter_overall.to_csv("data/viz_cache/app_batter_vulnerability.csv", index=False)
top_bottom.to_csv("data/viz_cache/app_batter_by_pitch_type.csv", index=False)

print(f"Saved app_batter_vulnerability.csv ({len(batter_overall)} batters)")
print(f"Saved app_batter_by_pitch_type.csv ({len(top_bottom)} rows)")
print("\nTop 5 most vulnerable batters overall:")
print(batter_overall[["batter_full_name","n_swings","mean_distortion_share"]].head(5).to_string(index=False))

# ── 3. K sensitivity: pitch-type spread metric ────────────────────────────────
print("\nBuilding K spread cache...")

# From pc_sample we have emp_dev at K=50 (production)
# Compute mean |emp_dev_z| per pitch type at K=50 and linear
pc_sample = pd.read_csv("data/viz_cache/pc_sample.csv", show_col_types=False) if False else pd.read_csv("data/viz_cache/pc_sample.csv")
pt_spread = (
    pc_sample[pc_sample["pitch_type"].isin(TYPES)]
    .groupby("pitch_type")
    .agg(
        mean_abs_emp_dev_z=("pc150_emp_dev_z", lambda x: x.abs().mean()),
        std_emp_dev_z=("pc150_emp_dev_z", "std"),
        mean_abs_lin_dev_z=("pc150_dev_z", lambda x: x.abs().mean()),
        std_lin_dev_z=("pc150_dev_z", "std"),
        n=("pc150_emp_dev_z", "count"),
    )
    .reset_index()
    .sort_values("mean_abs_emp_dev_z", ascending=False)
)
pt_spread["pitch_name"] = pt_spread["pitch_type"].map(PITCH_NAMES)
pt_spread.to_csv("data/viz_cache/knn_pitch_type_spread.csv", index=False)
print(f"Saved knn_pitch_type_spread.csv")
print(pt_spread[["pitch_name","mean_abs_emp_dev_z","mean_abs_lin_dev_z"]].to_string(index=False))

print("\nDone.")
