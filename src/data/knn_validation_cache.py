"""
Compute KNN validation cache files for post_commit_analysis.ipynb.

Outputs (data/viz_cache/):
  knn_k_sensitivity.csv  -- RMSE vs K + linear baseline RMSE (bias-variance curve)
  knn_purity.csv         -- neighborhood pitch-type purity at K=50 by pitch type
  knn_pca.csv            -- PCA of commit-time features (PC1, PC2, pitch_type)

Uses scipy.spatial.cKDTree -- avoids sklearn/xgboost import chain.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.linalg import svd

COMMIT_MS  = 150
K_VALUES   = [1, 3, 5, 10, 20, 30, 50, 75, 100]
K_PROD     = 50
PCA_SAMPLE = 30_000

FEAT_COLS = [
    f"pc{COMMIT_MS}_x_commit",  f"pc{COMMIT_MS}_z_commit",
    f"pc{COMMIT_MS}_vx_commit", f"pc{COMMIT_MS}_vz_commit",
]

PITCH_NAMES = {
    "FF": "4-Seam",   "SI": "Sinker",  "FC": "Cutter",
    "CH": "Changeup", "FS": "Splitter",
    "SL": "Slider",   "CU": "Curveball", "ST": "Sweeper", "KC": "Knuckle-curve",
}

# ── Load & merge ──────────────────────────────────────────────────────────────
print("Loading data...")
pc = pd.read_parquet(
    "data/swings_precommit.parquet",
    columns=["game_pk", "at_bat_number", "pitch_number"]
           + FEAT_COLS
           + ["plate_x", "plate_z", "pc150_dev_x", "pc150_dev_z"],
)
sw = pd.read_csv(
    "data/swings_2023_2025.csv",
    usecols=["game_pk", "at_bat_number", "pitch_number", "pitcher_id", "pitch_type"],
)

KEY = ["game_pk", "at_bat_number", "pitch_number"]
df = (pc.merge(sw, on=KEY, how="inner")
        .query("pitch_type in @PITCH_NAMES")
        .dropna(subset=FEAT_COLS + ["plate_x", "plate_z", "pitcher_id"]))

print(f"Dataset: {len(df):,} pitches  |  {df['pitcher_id'].nunique():,} pitchers")

# Global normalization (same as pipeline)
mu    = df[FEAT_COLS].mean()
sigma = df[FEAT_COLS].std().replace(0, 1.0)
X_norm_all = ((df[FEAT_COLS] - mu) / sigma).values
df["_px"] = X_norm_all[:, 0]
df["_pz"] = X_norm_all[:, 1]
df["_vx"] = X_norm_all[:, 2]
df["_vz"] = X_norm_all[:, 3]
NORM = ["_px", "_pz", "_vx", "_vz"]

# Keep pitchers with enough pitches for K=100 LOO
MIN_N = max(K_VALUES) + 1
pitcher_n = df["pitcher_id"].value_counts()
eligible  = pitcher_n[pitcher_n >= MIN_N].index
df_e = df[df["pitcher_id"].isin(eligible)].copy()
print(f"Eligible for K sensitivity: {len(df_e):,} pitches  |  {len(eligible):,} pitchers")

print("K sensitivity already saved — skipping.")

# ── 2. Neighborhood purity at K=50 ───────────────────────────────────────────
print("\nNeighborhood purity (K=50)...")
purity_rows = []

for pitcher_id, grp in df_e.groupby("pitcher_id"):
    if len(grp) < K_PROD + 1:
        continue
    X     = grp[NORM].values
    types = grp["pitch_type"].to_numpy(dtype=str)   # numpy str array

    tree = cKDTree(X)
    _, idxs = tree.query(X, k=K_PROD + 1)
    nb_types = types[idxs[:, 1:]]           # (n_pitches, K_PROD)
    same = (nb_types == types[:, None])     # bool mask
    purity_rows.append(pd.DataFrame({
        "pitch_type": types,
        "purity":     same.mean(axis=1),
    }))

purity_all = pd.concat(purity_rows, ignore_index=True)
purity_agg = (purity_all
    .groupby("pitch_type")["purity"]
    .agg(mean_purity="mean", n="count")
    .reset_index()
    .query("pitch_type in @PITCH_NAMES")
    .assign(pitch_name=lambda d: d["pitch_type"].map(PITCH_NAMES)))

purity_agg.to_csv("data/viz_cache/knn_purity.csv", index=False)
print(purity_agg[["pitch_name", "mean_purity", "n"]].to_string(index=False))
print("Saved knn_purity.csv")

# ── 3. PCA of commit-time feature space ──────────────────────────────────────
print("\nPCA...")
np.random.seed(42)
pca_df = df.sample(n=min(PCA_SAMPLE, len(df))).copy()
X_pca  = ((pca_df[FEAT_COLS] - mu) / sigma).values      # normalized, (n, 4)

# Center and SVD
X_c   = X_pca - X_pca.mean(axis=0)
_, s, Vt = svd(X_c, full_matrices=False)
var_explained = (s ** 2) / (s ** 2).sum()

PC = X_c @ Vt.T                                         # (n, 4) projections
pca_out = pd.DataFrame({
    "PC1":       PC[:, 0],
    "PC2":       PC[:, 1],
    "pitch_type": pca_df["pitch_type"].values,
    "pitch_name": pca_df["pitch_type"].map(PITCH_NAMES).values,
})
pca_out.to_csv("data/viz_cache/knn_pca.csv", index=False)
print(f"Variance explained: PC1={var_explained[0]:.1%}  PC2={var_explained[1]:.1%}")

# Save variance explained separately for subtitle
pd.DataFrame({
    "pc": [1, 2, 3, 4],
    "var_explained": var_explained,
}).to_csv("data/viz_cache/knn_pca_var.csv", index=False)
print("Saved knn_pca.csv + knn_pca_var.csv")

print("\nDone.")
