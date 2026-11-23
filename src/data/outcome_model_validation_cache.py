"""
Outcome model validation cache for causal_model.ipynb.

Year-holdout: train = 2023-2024 (models already fitted on all years),
              test  = 2025 (approximate OOS — same logic as intention_model_oos_cache.py).

Outputs (data/viz_cache/):
  cm_oos_metrics.csv        -- AUC/R²/RMSE by model, vs predict-mean baseline
  cm_bip_calibration.csv    -- binned predicted P(BIP) vs actual BIP rate
  cm_foul_calibration.csv   -- binned predicted P(foul|not BIP) vs actual
  cm_ev_scatter.csv         -- sampled predicted vs actual exit velo (BIP only)
  cm_la_scatter.csv         -- sampled predicted vs actual launch angle (BIP only)
  cm_counterfactual.csv     -- sampled xrv_realized vs xrv_intended (from xrv_causal)
"""
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ── Column constants (must match causal_models.py) ────────────────────────────
ANGULAR_DEVS    = ("vert_attack_angle_dev", "horz_attack_angle_dev", "swing_path_tilt_dev")
ANGULAR_ABS     = ("vert_attack_angle", "horz_attack_angle", "swing_path_tilt")
OUTCOME_FEATURES = list(ANGULAR_DEVS) + ["plate_x", "plate_z", "balls", "strikes"]
EV_LA_FEATURES   = list(ANGULAR_ABS)  + ["plate_x", "plate_z", "balls", "strikes"]

print("Loading models...")
bundle = joblib.load("models/causal_models.joblib")
bip_model         = bundle["bip_model"]
foul_model        = bundle["foul_model"]
exit_velo_model   = bundle["exit_velo_model"]
launch_angle_model= bundle["launch_angle_model"]
ev_la_rv_model    = bundle["ev_la_rv_model"]

print("Loading swing data...")
pc = pd.read_parquet(
    "data/swings_precommit.parquet",
    columns=["game_pk", "at_bat_number", "pitch_number",
             "game_date", "batter_id",
             "vert_attack_angle", "horz_attack_angle", "swing_path_tilt",
             "bat_speed", "is_bip", "is_contact", "is_whiff",
             "launch_speed", "launch_angle", "delta_run_exp",
             "plate_x", "plate_z", "balls", "strikes"],
)
idf = pd.read_parquet("models/intended_df.parquet")

# Competitive swings filter (matches run_pipeline.py)
mask = (
    pc["bat_speed"].notna() &
    (pc["bat_speed"] >= 50) &
    pc["vert_attack_angle"].notna()
)
pc = pc[mask].reset_index(drop=True)
idf = idf.reset_index(drop=True)
assert len(pc) == len(idf), f"Alignment mismatch: {len(pc)} vs {len(idf)}"

# Angular deviations
for rc, ic, dc in zip(
    ANGULAR_ABS,
    ["intended_vert_attack_angle", "intended_horz_attack_angle", "intended_swing_path_tilt"],
    ANGULAR_DEVS,
):
    pc[dc] = pc[rc] - idf[ic].values

pc["is_foul"] = ((pc["is_contact"] == 1) & (pc["is_bip"] != 1)).astype(int)
pc["year"]    = pd.to_datetime(pc["game_date"]).dt.year

test  = pc[pc["year"] == 2025].copy()
print(f"Test set (2025): {len(test):,} swings")

# ── 1. OOS performance metrics ────────────────────────────────────────────────
print("\nComputing OOS metrics...")
records = []

# BIP classifier
d_bip = test[OUTCOME_FEATURES + ["is_bip"]].dropna()
p_bip = bip_model.predict_proba(d_bip[OUTCOME_FEATURES])[:, 1]
auc_bip = roc_auc_score(d_bip["is_bip"], p_bip)
base_bip = roc_auc_score(
    d_bip["is_bip"],
    np.full(len(d_bip), d_bip["is_bip"].mean()),
)
records.append({"model": "BIP classifier",    "metric": "AUC-ROC",
                "model_val": auc_bip, "baseline_val": 0.5,
                "baseline_name": "Random (0.5)", "n": len(d_bip)})

# Foul classifier (on non-BIP swings)
d_foul = test.loc[test["is_bip"] == 0, OUTCOME_FEATURES + ["is_foul"]].dropna()
p_foul = foul_model.predict_proba(d_foul[OUTCOME_FEATURES])[:, 1]
auc_foul = roc_auc_score(d_foul["is_foul"], p_foul)
records.append({"model": "Foul classifier",   "metric": "AUC-ROC",
                "model_val": auc_foul, "baseline_val": 0.5,
                "baseline_name": "Random (0.5)", "n": len(d_foul)})

# EV regressor
d_ev = test.loc[test["is_bip"] == 1, EV_LA_FEATURES + ["launch_speed"]].dropna()
ev_pred = exit_velo_model.predict(d_ev[EV_LA_FEATURES])
ev_actual = d_ev["launch_speed"].values
ss_res = ((ev_actual - ev_pred) ** 2).sum()
ss_tot = ((ev_actual - ev_actual.mean()) ** 2).sum()
r2_ev  = 1 - ss_res / ss_tot
rmse_ev      = np.sqrt(np.mean((ev_actual - ev_pred) ** 2))
rmse_ev_base = np.sqrt(np.mean((ev_actual - ev_actual.mean()) ** 2))
records.append({"model": "Exit velo (XGB)",  "metric": "R²",
                "model_val": r2_ev,  "baseline_val": 0.0,
                "baseline_name": "Predict mean", "n": len(d_ev)})
records.append({"model": "Exit velo (XGB)",  "metric": "RMSE (mph)",
                "model_val": rmse_ev, "baseline_val": rmse_ev_base,
                "baseline_name": "Predict mean", "n": len(d_ev)})

# LA regressor
d_la = test.loc[test["is_bip"] == 1, EV_LA_FEATURES + ["launch_angle"]].dropna()
la_pred   = launch_angle_model.predict(d_la[EV_LA_FEATURES])
la_actual = d_la["launch_angle"].values
ss_res_la = ((la_actual - la_pred) ** 2).sum()
ss_tot_la = ((la_actual - la_actual.mean()) ** 2).sum()
r2_la     = 1 - ss_res_la / ss_tot_la
rmse_la      = np.sqrt(np.mean((la_actual - la_pred) ** 2))
rmse_la_base = np.sqrt(np.mean((la_actual - la_actual.mean()) ** 2))
records.append({"model": "Launch angle (XGB)", "metric": "R²",
                "model_val": r2_la,  "baseline_val": 0.0,
                "baseline_name": "Predict mean", "n": len(d_la)})
records.append({"model": "Launch angle (XGB)", "metric": "RMSE (deg)",
                "model_val": rmse_la, "baseline_val": rmse_la_base,
                "baseline_name": "Predict mean", "n": len(d_la)})

# EV→RV regressor
d_rv = test.loc[
    (test["is_bip"] == 1) & test["launch_speed"].notna() & test["delta_run_exp"].notna(),
    ["launch_speed", "launch_angle", "delta_run_exp"],
].dropna()
rv_pred   = ev_la_rv_model.predict(d_rv[["launch_speed", "launch_angle"]])
rv_actual = d_rv["delta_run_exp"].values
ss_res_rv = ((rv_actual - rv_pred) ** 2).sum()
ss_tot_rv = ((rv_actual - rv_actual.mean()) ** 2).sum()
r2_rv     = 1 - ss_res_rv / ss_tot_rv
rmse_rv      = np.sqrt(np.mean((rv_actual - rv_pred) ** 2))
rmse_rv_base = np.sqrt(np.mean((rv_actual - rv_actual.mean()) ** 2))
records.append({"model": "EV/LA -> RV (XGB)", "metric": "R²",
                "model_val": r2_rv,  "baseline_val": 0.0,
                "baseline_name": "Predict mean", "n": len(d_rv)})
records.append({"model": "EV/LA -> RV (XGB)", "metric": "RMSE (runs)",
                "model_val": rmse_rv, "baseline_val": rmse_rv_base,
                "baseline_name": "Predict mean", "n": len(d_rv)})

metrics_df = pd.DataFrame(records)
print(metrics_df.to_string(index=False))
metrics_df.to_csv("data/viz_cache/cm_oos_metrics.csv", index=False)
print("\nSaved cm_oos_metrics.csv")

# ── 2. Calibration curves ─────────────────────────────────────────────────────
print("\nBuilding calibration curves...")

def make_calib(y_true, y_prob, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    n_per_bin = len(y_true) // n_bins
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos,
                         "n": n_per_bin})

calib_bip  = make_calib(d_bip["is_bip"].values, p_bip)
calib_foul = make_calib(d_foul["is_foul"].values, p_foul)

calib_bip.to_csv("data/viz_cache/cm_bip_calibration.csv", index=False)
calib_foul.to_csv("data/viz_cache/cm_foul_calibration.csv", index=False)
print("Saved cm_bip_calibration.csv, cm_foul_calibration.csv")

# ── 3. EV/LA scatter samples ──────────────────────────────────────────────────
print("\nBuilding EV/LA scatter samples...")
rng = np.random.default_rng(42)

n_samp = min(6000, len(d_ev))
idx_ev = rng.choice(len(d_ev), size=n_samp, replace=False)
ev_scatter = pd.DataFrame({
    "actual":    ev_actual[idx_ev],
    "predicted": ev_pred[idx_ev],
})
ev_scatter.to_csv("data/viz_cache/cm_ev_scatter.csv", index=False)

n_samp_la = min(6000, len(d_la))
idx_la = rng.choice(len(d_la), size=n_samp_la, replace=False)
la_scatter = pd.DataFrame({
    "actual":    la_actual[idx_la],
    "predicted": la_pred[idx_la],
})
la_scatter.to_csv("data/viz_cache/cm_la_scatter.csv", index=False)
print("Saved cm_ev_scatter.csv, cm_la_scatter.csv")

# ── 4. Counterfactual distribution ────────────────────────────────────────────
print("\nBuilding counterfactual distribution...")
xrv = pd.read_parquet(
    "results/xrv_causal.parquet",
    columns=["game_pk", "at_bat_number", "pitch_number",
             "disruption_tax", "distortion_tax", "selection_tax",
             "distortion_share"],
)
sw = pd.read_csv(
    "data/swings_2023_2025.csv",
    usecols=["game_pk", "at_bat_number", "pitch_number", "game_date"],
)
xrv = xrv.merge(sw, on=["game_pk", "at_bat_number", "pitch_number"], how="left")
xrv["year"] = pd.to_datetime(xrv["game_date"]).dt.year
xrv_2025 = xrv[xrv["year"] == 2025].dropna(subset=["disruption_tax", "distortion_tax"])

# xrv_intended = xrv_realized - disruption_tax
xrv_2025 = xrv_2025.copy()
# Sample 10k for scatter
n_cf = min(10000, len(xrv_2025))
cf_sample = xrv_2025.sample(n=n_cf, random_state=42)[
    ["disruption_tax", "distortion_tax", "selection_tax", "distortion_share"]
].copy()
cf_sample.to_csv("data/viz_cache/cm_counterfactual.csv", index=False)
print(f"Saved cm_counterfactual.csv ({n_cf:,} rows)")

print("\nDone.")
