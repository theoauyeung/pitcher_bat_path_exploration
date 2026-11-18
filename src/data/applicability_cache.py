"""
Build applicability/validity cache CSVs for validation.ipynb.

Outputs (data/viz_cache/):
  val_year_stability.csv    -- pitcher distortion_share 2023 vs 2024 (year-over-year scatter)
  val_selection_zone.csv    -- mean selection_tax by in/out of projected strike zone
  val_ev_suppression.csv    -- EV suppression and LA change by distortion_share quartile
"""
import numpy as np
import pandas as pd

COMMIT_MS = 150
KEY = ["game_pk", "at_bat_number", "pitch_number"]

PLATE_HALF_WIDTH = 0.8333   # ft (17 in / 2)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")

# xrv results (all years)
xrv = pd.read_parquet(
    "results/xrv_causal.parquet",
    columns=KEY + [
        "pitcher_id", "distortion_tax", "selection_tax",
        "spatial_distortion_tax", "disruption_tax",
        "distortion_share", "ev_suppression", "la_realized", "la_intended",
    ],
).dropna(subset=["distortion_tax", "selection_tax"])

# swings csv (game_year, projected zone, pitcher name)
sw = pd.read_csv(
    "data/swings_2023_2025.csv",
    usecols=KEY + [
        "pitcher_id", "pitcher_full_name", "game_year",
        "sz_top", "sz_bot",
    ],
)

# precommit parquet (projected plate location at commit time)
pc = pd.read_parquet(
    "data/swings_precommit.parquet",
    columns=KEY + [f"pc{COMMIT_MS}_x_proj", f"pc{COMMIT_MS}_z_proj"],
)

# Merge everything (drop duplicate pitcher_id from sw since xrv has it)
df = (
    xrv
    .merge(sw.drop(columns=["pitcher_id"]), on=KEY, how="inner")
    .merge(pc, on=KEY, how="inner")
)
df["la_change"] = df["la_realized"] - df["la_intended"]

print(f"Merged df: {len(df):,} rows | years: {sorted(df['game_year'].unique())}")

# ── 1. Year-over-year pitcher stability: 2023 vs 2024 ─────────────────────────
print("\nBuilding year-over-year stability cache...")

yr_agg = (
    df.groupby(["pitcher_id", "pitcher_full_name", "game_year"])
    .agg(
        n_swings=("distortion_share", "count"),
        mean_distortion_share=("distortion_share", "mean"),
        mean_distortion_tax=("distortion_tax", "mean"),
        mean_selection_tax=("selection_tax", "mean"),
        mean_ev_suppression=("ev_suppression", "mean"),
    )
    .reset_index()
)

# Pivot to get 2023 and 2024 side by side, require ≥ 200 swings in each year
y2023 = yr_agg[yr_agg["game_year"] == 2023].set_index("pitcher_id")
y2024 = yr_agg[yr_agg["game_year"] == 2024].set_index("pitcher_id")

stable = y2023.join(y2024, lsuffix="_2023", rsuffix="_2024", how="inner")
stable = stable[
    (stable["n_swings_2023"] >= 200) & (stable["n_swings_2024"] >= 200)
].reset_index()

stable["pitcher_full_name"] = stable["pitcher_full_name_2023"].fillna(stable["pitcher_full_name_2024"])

out_cols = [
    "pitcher_id", "pitcher_full_name",
    "n_swings_2023", "n_swings_2024",
    "mean_distortion_share_2023", "mean_distortion_share_2024",
    "mean_distortion_tax_2023", "mean_distortion_tax_2024",
    "mean_ev_suppression_2023", "mean_ev_suppression_2024",
]
stable[out_cols].to_csv("data/viz_cache/val_year_stability.csv", index=False)

r = np.corrcoef(stable["mean_distortion_share_2023"], stable["mean_distortion_share_2024"])[0, 1]
print(f"Saved val_year_stability.csv | n={len(stable)} pitchers | r={r:.3f}")

# ── 2. Selection tax by projected-zone decision quality ───────────────────────
print("\nBuilding selection zone cache...")

# Classify projected plate location vs. batter's strike zone
df["proj_x"]    = df[f"pc{COMMIT_MS}_x_proj"]
df["proj_z"]    = df[f"pc{COMMIT_MS}_z_proj"]
df["in_zone"]   = (
    (df["proj_x"].abs() <= PLATE_HALF_WIDTH) &
    (df["proj_z"] >= df["sz_bot"]) &
    (df["proj_z"] <= df["sz_top"])
)

# Also classify chase distance (how far outside projected zone)
df["zone_dist_x"] = (df["proj_x"].abs() - PLATE_HALF_WIDTH).clip(lower=0)
df["zone_dist_z"] = np.maximum(
    (df["sz_bot"] - df["proj_z"]).clip(lower=0),
    (df["proj_z"] - df["sz_top"]).clip(lower=0),
)
df["zone_dist"] = np.sqrt(df["zone_dist_x"]**2 + df["zone_dist_z"]**2)

# Bin by distance from zone (0 = inside zone)
df["zone_bin"] = pd.cut(
    df["zone_dist"],
    bins=[-0.001, 0.001, 0.15, 0.30, 0.50, 2.0],
    labels=["In zone", "0-2 in. out", "2-4 in. out", "4-6 in. out", "6+ in. out"],
)

zone = (
    df.dropna(subset=["zone_bin"])
    .groupby("zone_bin", observed=True)
    .agg(
        mean_selection_tax=("selection_tax", "mean"),
        mean_disruption_tax=("disruption_tax", "mean"),
        mean_distortion_share=("distortion_share", "mean"),
        pct_in_zone=("in_zone", "mean"),
        n=("selection_tax", "count"),
    )
    .reset_index()
)
zone.to_csv("data/viz_cache/val_selection_zone.csv", index=False)
print(f"Saved val_selection_zone.csv")
print(zone[["zone_bin", "mean_selection_tax", "n"]].to_string(index=False))

# ── 3. EV suppression and LA change by distortion share quartile ──────────────
print("\nBuilding EV suppression cache...")

df_bip = df.dropna(subset=["ev_suppression", "la_change", "distortion_share"]).copy()
df_bip["dist_quartile"] = pd.qcut(
    df_bip["distortion_share"], q=4,
    labels=["Q1\n(least distortion)", "Q2", "Q3", "Q4\n(most distortion)"]
)

ev_supp = (
    df_bip.groupby("dist_quartile", observed=True)
    .agg(
        mean_ev_suppression=("ev_suppression", "mean"),
        se_ev_suppression=("ev_suppression", lambda x: x.std() / np.sqrt(len(x))),
        mean_la_change=("la_change", "mean"),
        se_la_change=("la_change", lambda x: x.std() / np.sqrt(len(x))),
        mean_distortion_share=("distortion_share", "mean"),
        n=("ev_suppression", "count"),
    )
    .reset_index()
)
ev_supp.to_csv("data/viz_cache/val_ev_suppression.csv", index=False)
print(f"Saved val_ev_suppression.csv")
print(ev_supp[["dist_quartile", "mean_ev_suppression", "mean_la_change", "n"]].to_string(index=False))

print("\nDone.")
