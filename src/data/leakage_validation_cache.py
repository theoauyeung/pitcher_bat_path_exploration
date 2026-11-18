"""
Build leakage-validation cache CSVs for results.ipynb.

Outputs (data/viz_cache/):
  lv_monotonicity.csv   -- mean distortion_tax by decile of post-commit deviation magnitude
  lv_pitch_type.csv     -- mean tax metrics by pitch type
  lv_phase_a.csv        -- intended_vaa & vaa_dev vs binned pc150_emp_dev_z (Phase A leakage test)
"""
import numpy as np
import pandas as pd

COMMIT_MS = 150
KEY = ["game_pk", "at_bat_number", "pitch_number"]

PITCH_NAMES = {
    "FF": "4-Seam",   "SI": "Sinker",  "FC": "Cutter",
    "CH": "Changeup", "FS": "Splitter",
    "SL": "Slider",   "CU": "Curveball", "ST": "Sweeper", "KC": "Knuckle-curve",
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")

pc = pd.read_parquet(
    "data/swings_precommit.parquet",
    columns=KEY + [
        f"pc{COMMIT_MS}_emp_dev_x", f"pc{COMMIT_MS}_emp_dev_z",
        "is_swing", "bat_speed", "vert_attack_angle",
    ]
)

# Apply same competitive_swings filter as pipeline
swings = pc[
    (pc["is_swing"] == 1) &
    pc["bat_speed"].notna() &
    (pc["bat_speed"] >= 50) &
    pc["vert_attack_angle"].notna()
].copy()
print(f"Competitive swings: {len(swings):,}")

# Load intended_df (positionally aligned with swings above)
intended = pd.read_parquet("models/intended_df.parquet")
assert len(intended) == len(swings), (
    f"Row mismatch: intended={len(intended)}, swings={len(swings)}"
)
swings = swings.reset_index(drop=True)
swings["intended_vaa"] = intended["intended_vert_attack_angle"].values
swings["intended_haa"] = intended["intended_horz_attack_angle"].values

# Load realized VAA and pfx from swings csv
print("Loading swings csv...")
sw_csv = pd.read_csv(
    "data/swings_2023_2025.csv",
    usecols=KEY + ["vaa", "haa", "pfx_x", "pfx_z", "pitch_type"],
)
swings = swings.merge(sw_csv, on=KEY, how="inner")
swings["vaa_dev"] = swings["vaa"] - swings["intended_vaa"]
swings["haa_dev"] = swings["haa"] - swings["intended_haa"]
swings["emp_dev_mag"] = np.sqrt(
    swings[f"pc{COMMIT_MS}_emp_dev_x"] ** 2 +
    swings[f"pc{COMMIT_MS}_emp_dev_z"] ** 2
)

# Load xrv results
xrv = pd.read_parquet(
    "results/xrv_causal.parquet",
    columns=KEY + ["distortion_tax", "selection_tax", "spatial_distortion_tax",
                   "disruption_tax", "distortion_share", "pitch_type"],
)
xrv = xrv.dropna(subset=["distortion_tax", "selection_tax"])

df = swings.merge(
    xrv[KEY + ["distortion_tax", "selection_tax", "spatial_distortion_tax",
               "disruption_tax", "distortion_share"]],
    on=KEY, how="inner"
)
print(f"Joined df: {len(df):,} rows")

# ── 1. Monotonicity: distortion_tax vs. movement magnitude ───────────────────
print("\nBuilding monotonicity cache...")
df["mag_decile"] = pd.qcut(df["emp_dev_mag"], q=10, labels=False, duplicates="drop")
mono = (
    df.groupby("mag_decile", observed=True)
    .agg(
        mag_lo=("emp_dev_mag", "min"),
        mag_hi=("emp_dev_mag", "max"),
        mag_mid=("emp_dev_mag", "median"),
        mean_distortion_tax=("distortion_tax", "mean"),
        mean_selection_tax=("selection_tax", "mean"),
        mean_spatial_tax=("spatial_distortion_tax", "mean"),
        mean_disruption_tax=("disruption_tax", "mean"),
        mean_distortion_share=("distortion_share", "mean"),
        n=("distortion_tax", "count"),
    )
    .reset_index()
)
mono.to_csv("data/viz_cache/lv_monotonicity.csv", index=False)
print(f"Saved lv_monotonicity.csv ({len(mono)} rows)")
print(mono[["mag_decile", "mag_mid", "mean_distortion_tax", "mean_distortion_share"]].to_string(index=False))

# ── 2. By pitch type ──────────────────────────────────────────────────────────
print("\nBuilding pitch type cache...")
pt = (
    df[df["pitch_type"].isin(PITCH_NAMES)]
    .groupby("pitch_type", observed=True)
    .agg(
        pitch_name=("pitch_type", lambda x: PITCH_NAMES[x.iloc[0]]),
        mean_distortion_tax=("distortion_tax", "mean"),
        mean_selection_tax=("selection_tax", "mean"),
        mean_spatial_tax=("spatial_distortion_tax", "mean"),
        mean_disruption_tax=("disruption_tax", "mean"),
        mean_distortion_share=("distortion_share", "mean"),
        n=("distortion_tax", "count"),
    )
    .reset_index()
    .sort_values("mean_distortion_share", ascending=False)
)
pt.to_csv("data/viz_cache/lv_pitch_type.csv", index=False)
print(f"Saved lv_pitch_type.csv")
print(pt[["pitch_name", "mean_distortion_share", "mean_distortion_tax", "n"]].to_string(index=False))

# ── 3. Phase A leakage test: intended_vaa & vaa_dev vs pc150_emp_dev_z ───────
print("\nBuilding Phase A leakage cache...")
df_clean = df.dropna(subset=["vaa_dev", f"pc{COMMIT_MS}_emp_dev_z", "intended_vaa"])

df_clean = df_clean.copy()
df_clean["dev_z_decile"] = pd.qcut(
    df_clean[f"pc{COMMIT_MS}_emp_dev_z"], q=10, labels=False, duplicates="drop"
)
phase_a = (
    df_clean.groupby("dev_z_decile", observed=True)
    .agg(
        dev_z_lo=(f"pc{COMMIT_MS}_emp_dev_z", "min"),
        dev_z_hi=(f"pc{COMMIT_MS}_emp_dev_z", "max"),
        dev_z_mid=(f"pc{COMMIT_MS}_emp_dev_z", "median"),
        mean_intended_vaa=("intended_vaa", "mean"),
        mean_realized_vaa=("vaa", "mean"),
        mean_vaa_dev=("vaa_dev", "mean"),
        n=("vaa_dev", "count"),
    )
    .reset_index()
)
phase_a.to_csv("data/viz_cache/lv_phase_a.csv", index=False)
print(f"Saved lv_phase_a.csv")
print(phase_a[["dev_z_decile", "dev_z_mid", "mean_intended_vaa", "mean_vaa_dev"]].to_string(index=False))

print("\nDone.")
