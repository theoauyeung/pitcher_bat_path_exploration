"""
Pitcher mechanics analysis cache.

Outputs (data/viz_cache/):
  app_pitcher_adjusted.csv    -- pitch-mix-adjusted disruption leaderboard
  app_pitcher_ols.csv         -- pitcher-level mechanics + disruption for OLS
  app_tyler_rogers_waterfall.csv -- OLS decomposition for Tyler Rogers
"""
import numpy as np
import pandas as pd
from scipy import stats

KEY = ["game_pk", "at_bat_number", "pitch_number"]
PITCH_NAMES = {
    "FF": "4-Seam", "SI": "Sinker", "FC": "Cutter",
    "CH": "Changeup", "FS": "Splitter",
    "SL": "Slider", "CU": "Curveball", "ST": "Sweeper", "KC": "Knuckle-curve",
}
KNOWN_TYPES = list(PITCH_NAMES.keys())
MIN_SWINGS = 500

print("Loading data...")
xrv = pd.read_parquet(
    "results/xrv_causal.parquet",
    columns=KEY + ["pitcher_id", "pitch_type", "disruption_tax",
                   "distortion_tax", "selection_tax", "distortion_share"],
).dropna(subset=["distortion_tax", "selection_tax"])

sw = pd.read_csv(
    "data/swings_2023_2025.csv",
    usecols=KEY + ["pitcher_id", "pitcher_full_name", "pitch_type",
                   "arm_angle", "release_pos_x", "release_pos_z",
                   "release_extension", "release_speed", "pfx_x", "pfx_z"],
)

df = xrv.merge(
    sw.drop(columns=["pitcher_id", "pitch_type"]),  # avoid duplicates
    on=KEY, how="inner",
)
print(f"Merged: {len(df):,} rows")

# ── Pitcher-level aggregates ──────────────────────────────────────────────────
print("\nBuilding pitcher-level table...")

mech_cols = ["arm_angle", "release_pos_x", "release_pos_z",
             "release_extension", "release_speed", "pfx_x", "pfx_z"]

pitcher_df = (
    df.groupby(["pitcher_id", "pitcher_full_name"], observed=True)
    .agg(
        n_swings        = ("disruption_tax", "count"),
        mean_disruption = ("disruption_tax", "mean"),
        mean_distortion_share = ("distortion_share", "mean"),
        **{f"mean_{c}": (c, "mean") for c in mech_cols},
    )
    .reset_index()
    .query("n_swings >= @MIN_SWINGS")
)

# Pitch-type mix per pitcher (fraction of swings per type)
pt_counts = (
    df[df["pitch_type"].isin(KNOWN_TYPES)]
    .groupby(["pitcher_id", "pitch_type"], observed=True)
    .size()
    .reset_index(name="n")
)
pt_totals = pt_counts.groupby("pitcher_id")["n"].transform("sum")
pt_counts["frac"] = pt_counts["n"] / pt_totals

# League-mean distortion_share per pitch type
league_pt = (
    df[df["pitch_type"].isin(KNOWN_TYPES)]
    .groupby("pitch_type", observed=True)["distortion_share"]
    .mean()
    .reset_index()
    .rename(columns={"distortion_share": "league_mean_ds"})
)

# Expected distortion share given pitch mix
pt_joined = pt_counts.merge(league_pt, on="pitch_type", how="left")
expected = (
    pt_joined.assign(contrib=lambda d: d["frac"] * d["league_mean_ds"])
    .groupby("pitcher_id")["contrib"]
    .sum()
    .reset_index()
    .rename(columns={"contrib": "expected_distortion_share"})
)

pitcher_df = pitcher_df.merge(expected, on="pitcher_id", how="left")
pitcher_df["adjusted_distortion_share"] = (
    pitcher_df["mean_distortion_share"] - pitcher_df["expected_distortion_share"]
)

# Dominant pitch type per pitcher
dominant = (
    pt_counts.sort_values("frac", ascending=False)
    .groupby("pitcher_id")
    .first()
    .reset_index()[["pitcher_id", "pitch_type", "frac"]]
    .rename(columns={"pitch_type": "dominant_pitch", "frac": "dominant_pct"})
)
dominant["dominant_name"] = dominant["dominant_pitch"].map(PITCH_NAMES)

pitcher_df = pitcher_df.merge(dominant, on="pitcher_id", how="left")

print(f"Pitcher table: {len(pitcher_df)} pitchers")
print("\nTop 10 by adjusted distortion share:")
print(
    pitcher_df.sort_values("adjusted_distortion_share", ascending=False)
    .head(10)[["pitcher_full_name", "n_swings", "mean_distortion_share",
               "expected_distortion_share", "adjusted_distortion_share"]]
    .to_string(index=False)
)

pitcher_df.to_csv("data/viz_cache/app_pitcher_adjusted.csv", index=False)
print("\nSaved app_pitcher_adjusted.csv")

# ── Pitcher OLS data ──────────────────────────────────────────────────────────
print("\nBuilding OLS dataset...")

# Keep only complete mechanics rows
ols_cols = ["pitcher_id", "pitcher_full_name", "n_swings",
            "mean_disruption", "mean_distortion_share", "adjusted_distortion_share",
            "mean_arm_angle", "mean_release_pos_z", "mean_release_pos_x",
            "mean_release_extension", "mean_release_speed",
            "mean_pfx_x", "mean_pfx_z",
            "expected_distortion_share", "dominant_name", "dominant_pct"]
ols_df = pitcher_df[ols_cols].dropna(subset=["mean_arm_angle", "mean_release_pos_z"])

ols_df.to_csv("data/viz_cache/app_pitcher_ols.csv", index=False)
print(f"Saved app_pitcher_ols.csv ({len(ols_df)} pitchers)")

# ── Tyler Rogers OLS decomposition ───────────────────────────────────────────
print("\nBuilding Tyler Rogers waterfall decomposition...")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ["mean_arm_angle", "mean_release_pos_z", "mean_release_extension",
            "mean_release_speed", "mean_pfx_z"]
FEATURE_LABELS = ["Arm angle", "Release height", "Extension",
                  "Velocity", "Vertical movement"]

X = ols_df[FEATURES].values
y = ols_df["mean_disruption"].values

# z-score features so coefficients are comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

result = stats.linregress  # not using this, doing manual OLS
# OLS via numpy for decomposition
from numpy.linalg import lstsq
coefs, _, _, _ = lstsq(
    np.column_stack([np.ones(len(X_scaled)), X_scaled]), y, rcond=None
)
intercept = coefs[0]
betas = coefs[1:]  # coefficients on standardized features

# Tyler Rogers row
tr_row = ols_df[ols_df["pitcher_full_name"].str.lower() == "tyler rogers"].iloc[0]
tr_X = scaler.transform([tr_row[FEATURES].values])
league_mean_X = np.zeros((1, len(FEATURES)))  # zero in standardized space = league mean

# Contribution of each feature: beta_i * (tr_z_i - 0)
contributions = betas * tr_X[0]  # deviations from league mean, weighted by beta

# League mean disruption tax (intercept in mean-centered model)
league_mean_disruption = ols_df["mean_disruption"].mean()
predicted_tr = intercept + (betas * tr_X[0]).sum()
residual = tr_row["mean_disruption"] - predicted_tr

waterfall = pd.DataFrame({
    "component": ["League average"] + FEATURE_LABELS + ["Unexplained"],
    "value": [league_mean_disruption] + list(contributions) + [residual],
    "type": ["baseline"] + ["feature"] * len(FEATURE_LABELS) + ["residual"],
})

# Running total for waterfall chart positioning
waterfall["cumulative_end"] = (
    waterfall["value"].cumsum()
)
waterfall["cumulative_start"] = waterfall["cumulative_end"] - waterfall["value"]

print(waterfall.to_string(index=False))
print(f"\nActual Tyler Rogers disruption: {tr_row['mean_disruption']:.4f}")
print(f"OLS predicted:                  {predicted_tr:.4f}")

waterfall.to_csv("data/viz_cache/app_tyler_rogers_waterfall.csv", index=False)
print("Saved app_tyler_rogers_waterfall.csv")

# OLS R^2 for reference
y_pred = intercept + (X_scaled * betas).sum(axis=1)
ss_res = ((y - y_pred) ** 2).sum()
ss_tot = ((y - y.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot
print(f"\nOLS R^2 = {r2:.3f}  (n={len(ols_df)} pitchers)")
print("Coefficients (standardized):")
for label, beta in zip(FEATURE_LABELS, betas):
    print(f"  {label:25s} {beta:.5f}")

print("\nDone.")
