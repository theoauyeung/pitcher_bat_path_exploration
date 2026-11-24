"""
Outcome model story cache: two plots that show the model doing something meaningful.

1. Deviation gradient — actual outcomes (P(BIP), mean EV, mean delta_run_exp)
   binned by absolute VAA deviation size. Shows the pipeline learns the right signal.

2. EV/LA run-value surface — heat map of ev_la_rv_model predictions over the
   EV × LA grid. Recovers the well-known baseball sweet spot.

Outputs:
  data/viz_cache/cm_dev_gradient.csv    -- binned P(BIP), EV, RV by |VAA_dev| decile
  data/viz_cache/cm_ev_la_surface.csv   -- EV/LA grid predictions
"""
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ANGULAR_DEVS = ("vert_attack_angle_dev", "horz_attack_angle_dev", "swing_path_tilt_dev")
ANGULAR_ABS  = ("vert_attack_angle", "horz_attack_angle", "swing_path_tilt")

print("Loading models...")
bundle = joblib.load("models/causal_models.joblib")
bip_model          = bundle["bip_model"]
ev_la_rv_model     = bundle["ev_la_rv_model"]
_ev_features       = bundle["exit_velo_model"].get_booster().feature_names

print("Loading swing data...")
pc = pd.read_parquet(
    "data/swings_precommit.parquet",
    columns=["game_pk", "at_bat_number", "pitch_number",
             "bat_speed", "vert_attack_angle", "horz_attack_angle", "swing_path_tilt",
             "is_bip", "is_contact", "launch_speed", "delta_run_exp",
             "plate_x", "plate_z", "balls", "strikes"],
)
idf = pd.read_parquet("models/intended_df.parquet")

mask = (
    pc["bat_speed"].notna() &
    (pc["bat_speed"] >= 50) &
    pc["vert_attack_angle"].notna()
)
pc  = pc[mask].reset_index(drop=True)
idf = idf.reset_index(drop=True)
assert len(pc) == len(idf)

# Angular deviations
for rc, ic, dc in zip(
    ANGULAR_ABS,
    ["intended_vert_attack_angle", "intended_horz_attack_angle", "intended_swing_path_tilt"],
    ANGULAR_DEVS,
):
    pc[dc] = pc[rc] - idf[ic].values

pc["is_foul"] = ((pc["is_contact"] == 1) & (pc["is_bip"] != 1)).astype(int)
pc["abs_vaa_dev"] = pc["vert_attack_angle_dev"].abs()

print(f"Working dataset: {len(pc):,} swings")

# ── 1. Deviation gradient ─────────────────────────────────────────────────────
print("\nBuilding deviation gradient...")

# Bin by |VAA deviation| into 10 deciles
pc["dev_decile"] = pd.qcut(pc["abs_vaa_dev"], q=10, labels=False, duplicates="drop")

grad = (
    pc.groupby("dev_decile")
    .agg(
        n               = ("abs_vaa_dev", "count"),
        dev_mid         = ("abs_vaa_dev", "median"),
        dev_lo          = ("abs_vaa_dev", lambda x: x.quantile(0.1)),
        dev_hi          = ("abs_vaa_dev", lambda x: x.quantile(0.9)),
        p_bip           = ("is_bip", "mean"),
        mean_ev         = ("launch_speed", lambda x: x[pc.loc[x.index, "is_bip"] == 1].mean()),
        mean_delta_rv   = ("delta_run_exp", "mean"),
    )
    .reset_index()
)

# Model-predicted P(BIP) per decile
feat_df = pc[bip_model.get_booster().feature_names].copy()
pc["pred_bip"] = bip_model.predict_proba(feat_df)[:, 1]
grad_pred = pc.groupby("dev_decile")["pred_bip"].mean().reset_index()
grad = grad.merge(grad_pred, on="dev_decile")

print(grad[["dev_decile", "dev_mid", "p_bip", "pred_bip", "mean_ev", "mean_delta_rv"]].to_string(index=False))
grad.to_csv("data/viz_cache/cm_dev_gradient.csv", index=False)
print("Saved cm_dev_gradient.csv")

# ── 2. EV/LA run-value surface ────────────────────────────────────────────────
print("\nBuilding EV/LA surface...")

ev_grid = np.arange(55, 116, 1.0)
la_grid = np.arange(-40, 51, 1.0)
EV, LA  = np.meshgrid(ev_grid, la_grid)
grid_df = pd.DataFrame({"launch_speed": EV.ravel(), "launch_angle": LA.ravel()})
grid_df["pred_rv"] = ev_la_rv_model.predict(grid_df[["launch_speed", "launch_angle"]])

grid_df.to_csv("data/viz_cache/cm_ev_la_surface.csv", index=False)
print(f"Saved cm_ev_la_surface.csv ({len(grid_df):,} grid points)")
print(f"  RV range: {grid_df['pred_rv'].min():.3f} to {grid_df['pred_rv'].max():.3f} runs")

print("\nDone.")
