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

# ── 1. K sensitivity: RMSE + std(deviation) per K ────────────────────────────
import os as _os
_ksens_path = "data/viz_cache/knn_k_sensitivity.csv"
_ksens_existing = pd.read_csv(_ksens_path) if _os.path.exists(_ksens_path) else None

if _ksens_existing is not None and "std_dev_z" in _ksens_existing.columns:
    print("K sensitivity already saved — skipping.")
else:
    print("Computing K sensitivity (RMSE + deviation std)...")
    lin_rmse = None
    rows = []
    for k in K_VALUES:
        sq_errs, devs = [], []
        for pitcher_id, grp in df_e.groupby("pitcher_id"):
            if len(grp) < k + 1:
                continue
            X      = grp[NORM].values
            plate_z = grp["plate_z"].values
            tree = cKDTree(X)
            _, idxs = tree.query(X, k=k + 1)  # first col is self
            nb_z  = plate_z[idxs[:, 1:k+1]]
            pred  = nb_z.mean(axis=1)
            dev   = plate_z - pred
            sq_errs.extend((dev ** 2).tolist())
            devs.extend(dev.tolist())
        import numpy as _np
        rmse = _np.sqrt(_np.mean(sq_errs))
        std_dev = _np.std(devs)
        if lin_rmse is None:
            lin_errs = []
            for pitcher_id, grp in df_e.groupby("pitcher_id"):
                lin_errs.extend(((grp["plate_z"] - grp["pc150_dev_z"].add(grp["plate_z"] - grp["pc150_dev_z"])) ** 2).tolist())
            # linear baseline: predict plate_z from linear trajectory
            lin_dev = df_e["pc150_dev_z"].dropna()  # dev from linear = already the linear prediction error
            lin_rmse_val = _np.sqrt((df_e["pc150_dev_z"].dropna() ** 2).mean())
        rows.append({"k": k, "rmse": rmse, "std_dev_z": std_dev})
        print(f"  K={k:3d}  rmse={rmse:.4f}  std_dev={std_dev:.4f}")

    ksens = pd.DataFrame(rows)
    # linear baseline RMSE from existing file
    if _ksens_existing is not None:
        ksens["linear_rmse"] = _ksens_existing["linear_rmse"].iloc[0]
    ksens.to_csv(_ksens_path, index=False)
    print("Saved knn_k_sensitivity.csv with std_dev_z")

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
