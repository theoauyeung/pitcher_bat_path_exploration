"""
Phase B: run-value mediation models.

DAG (two disruption channels):
  post-commit movement  →  angular deviation  →  run value   (angular / mediated path)
  post-commit movement  →  spatial displacement →  run value  (spatial / direct path)
      (treatment)                                  (outcome)

Breaking balls expose a limitation of the angular-only model: a batter can execute
their intended swing plane perfectly yet miss because the ball crossed the plate 8"
from where they projected it.  The redesign addresses this with two changes:

  Option B — changed counterfactual
    Old: xRV(intended) = predict(zero angular devs, ACTUAL plate location)
    New: xRV(intended) = predict(zero angular devs, zero post-commit movement)
         → ball stays at the projected location (pc{ms}_x_proj, pc{ms}_z_proj),
           not the actual plate_x / plate_z.
    This directly prices the spatial shift caused by late movement.


Three components
────────────────
1. Mediator models (one per angular deviation axis) — unchanged
     dev ~ pc{ms}_dev_x + pc{ms}_dev_z + x_proj + z_proj
           + release_speed + balls + strikes + offset_y_ms
           + (1 | batter_id)

2. Outcome models — three channels, actual plate location
     P(BIP):            logistic on angular_devs + plate_x + plate_z + count
     P(foul | not BIP): logistic on same, non-BIP swings only
     xwOBAcon:          linear on same, BIP only

3. Disruption tax — three-scenario predict-twice
     xrv_realized    = predict(actual_devs,  plate_x=actual,  plate_z=actual)
     xrv_spatial     = predict(zero_devs,    plate_x=actual,  plate_z=actual)
     xrv_intended    = predict(zero_devs,    plate_x=x_proj,  plate_z=z_proj)

     disruption_tax       = xrv_realized − xrv_intended
     spatial_distortion   = xrv_spatial  − xrv_intended  (location shift, perfect swing)
     angular_disruption   = xrv_realized − xrv_spatial   (deviation on top of shift)

   Angular distortion share (from mediator models):
     distortion_dev_m = a_x_m × pc_dev_x + a_z_m × pc_dev_z
     angular_distortion_share = ||distortion_dev||² / ||total_dev||²  (clipped to [0,1])

   Final:
     distortion_tax = spatial_distortion + angular_disruption × angular_distortion_share
     selection_tax  = angular_disruption × (1 − angular_distortion_share)
     distortion_share = distortion_tax / disruption_tax  (clipped to [0,1])

Negative-control check built in: filter to FF with dev_total < threshold
and verify that disruption_tax ≈ 0.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


# ── Column conventions ──────────────────────────────────────────────────────────

ANGULAR_DEVS = ("vert_attack_angle_dev", "horz_attack_angle_dev", "swing_path_tilt_dev")
ANGULAR_RESP = ("vert_attack_angle", "horz_attack_angle", "swing_path_tilt")

def _pc(commit_ms, suffix):
    return f"pc{commit_ms}_{suffix}"


# ── 1. Mediator models ──────────────────────────────────────────────────────────

def fit_mediator_models(df, commit_ms=150):
    """Linear mixed-effects mediator models for each angular deviation axis.

    Response: {metric}_dev  (realized − intended, from Phase A)
    Treatment: pc{ms}_dev_x, pc{ms}_dev_z  (post-commit movement)
    Controls: pre-commit projected plate location, pitch speed, count, timing

    Returns dict mapping deviation column name → fitted MixedLMResults.
    """
    prefix = f"pc{commit_ms}"
    dev_x  = f"{prefix}_dev_x"
    dev_z  = f"{prefix}_dev_z"
    x_proj = f"{prefix}_x_proj"
    z_proj = f"{prefix}_z_proj"

    needed = [dev_x, dev_z, x_proj, z_proj,
              "release_speed", "balls", "strikes", "offset_y_ms", "batter_id", "pitcher_id"]

    models = {}
    for dev_col in ANGULAR_DEVS:
        d = df[needed + [dev_col]].dropna()
        d = d[d["batter_id"].map(d["batter_id"].value_counts()) >= 20].copy()
        d = d[d["pitcher_id"].map(d["pitcher_id"].value_counts()) >= 10].copy()
        d["batter_id"]  = d["batter_id"].astype("category")
        d["pitcher_id"] = d["pitcher_id"].astype("category")

        formula = (
            f"{dev_col} ~ {dev_x} + {dev_z}"
            f" + {x_proj} + {z_proj}"
            f" + release_speed + balls + strikes + offset_y_ms"
        )
        # To add crossed pitcher RE (slow on large datasets): add
        #   vc_formula={"pitcher_id": "0 + C(pitcher_id)"}
        # to the mixedlm call below.
        model = smf.mixedlm(formula, d, groups=d["batter_id"]).fit(reml=False)
        models[dev_col] = model

    return models


# ── 2. Outcome models ───────────────────────────────────────────────────────────

def fit_outcome_models(df, commit_ms=150):
    """Fit all outcome components. Returns (bip_model, foul_model, xwoba_model, whiff_rv_table).

    Outcome models use actual plate_x / plate_z (correctly estimated location effect).
    Spatial disruption is priced in the counterfactual (_xrv_from_shape with zero_spatial=True
    substitutes x_proj/z_proj for plate_x/plate_z), not through a decomposed-location formula.

    Decomposed location (Option C) was reverted: pc150_dev_z absorbs gravity and is always
    negative, so the regression coefficient is backward relative to causal direction.
    """
    deviation_terms = " + ".join(ANGULAR_DEVS)
    controls = (
        f"{deviation_terms}"
        f" + plate_x + plate_z"
        f" + balls + strikes"
    )
    needed = list(ANGULAR_DEVS) + ["plate_x", "plate_z", "balls", "strikes"]

    d = df.copy()
    if "is_foul" not in d.columns:
        d["is_foul"] = ((d["is_contact"] == 1) & (d["is_bip"] != 1)).astype(int)

    d_bip = d[needed + ["is_bip"]].dropna()
    bip_model = smf.logit(f"is_bip ~ {controls}", d_bip).fit(cov_type="HC1", disp=False)

    d_foul = d.loc[d["is_bip"] == 0, needed + ["is_foul"]].dropna()
    foul_model = smf.logit(f"is_foul ~ {controls}", d_foul).fit(cov_type="HC1", disp=False)

    d_xwoba = df.loc[df["is_bip"] == 1, needed + ["xwoba"]].dropna()
    xwoba_model = smf.ols(f"xwoba ~ {controls}", d_xwoba).fit(cov_type="HC1")

    whiffs = df[(df["is_swing"] == 1) & (df["is_whiff"] == 1) &
                df["delta_run_exp"].notna()]
    whiff_rv = whiffs.groupby(["balls", "strikes"])["delta_run_exp"].mean().to_dict()
    whiff_rv["default"] = whiffs["delta_run_exp"].mean()

    return bip_model, foul_model, xwoba_model, whiff_rv


# ── 3. Disruption tax ───────────────────────────────────────────────────────────

def _xrv_from_shape(df, bip_model, foul_model, xwoba_model, whiff_rv, foul_rv,
                    commit_ms=150, zero_angular=False, zero_spatial=False):
    """Predict per-swing xRV for one of three counterfactual scenarios.

    zero_angular=False, zero_spatial=False → realized xRV (actual everything)
    zero_angular=True,  zero_spatial=True  → intended xRV:
        ball at projected location (pc_dev=0), perfect swing (angular_dev=0).
        This is the 'what if the ball stayed where the batter expected it to?' baseline.
    zero_angular=True,  zero_spatial=False → spatial-only xRV:
        actual post-commit movement, perfect swing angles.
        Isolates the cost of spatial displacement alone.

    Three-channel formula:
      P(foul) = (1 − P(BIP)) × P(foul | not BIP)
      P(whiff) = (1 − P(BIP)) × (1 − P(foul | not BIP))
      xRV = P(BIP)×E[xwOBA|BIP] + P(foul)×foul_rv[count] + P(whiff)×whiff_rv[count]
    """
    prefix = f"pc{commit_ms}"
    x_proj = f"{prefix}_x_proj"
    z_proj = f"{prefix}_z_proj"

    # Outcome models use plate_x / plate_z. When zero_spatial=True, substitute the
    # pre-commit projected location — Option B counterfactual ("ball stayed where
    # the batter expected it"). This correctly prices spatial disruption through the
    # same plate-location regression used at training time.
    score = df[["balls", "strikes"]].copy()
    score["plate_x"] = df[x_proj] if zero_spatial else df["plate_x"]
    score["plate_z"] = df[z_proj] if zero_spatial else df["plate_z"]
    for dev in ANGULAR_DEVS:
        score[dev] = 0.0 if zero_angular else df[dev]

    p_bip            = bip_model.predict(score)
    p_foul_given_not = foul_model.predict(score)
    e_xwoba          = xwoba_model.predict(score)

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

    return p_bip * e_xwoba + p_foul * foul_vals + p_whiff * whiff_vals


def disruption_tax_split(df, bip_model, foul_model, xwoba_model, whiff_rv, foul_rv,
                         mediator_models, commit_ms=150):
    """Full disruption tax with two-channel distortion/selection decomposition.

    Three xRV scenarios:
      xrv_realized  — actual angular deviations + actual post-commit movement
      xrv_spatial   — zero angular deviations + actual post-commit movement
                       (perfect swing, ball at actual plate location)
      xrv_intended  — zero angular deviations + zero post-commit movement
                       (perfect swing, ball at projected location = batter's information)

    Decomposition:
      disruption_tax     = xrv_realized − xrv_intended  [negative = pitcher advantage]
      spatial_distortion = xrv_spatial  − xrv_intended  spatial displacement alone
      angular_disruption = xrv_realized − xrv_spatial   angular deviation on top of shift

    Angular distortion share (from mediator models, clipped to [0, 1]):
      distortion_dev_m = a_x_m × pc_dev_x + a_z_m × pc_dev_z
      angular_distortion_share = ||distortion_dev||² / ||total_dev||²

    Final outputs:
      distortion_tax = spatial_distortion + angular_disruption × angular_distortion_share
      selection_tax  = angular_disruption × (1 − angular_distortion_share)
      distortion_share = distortion_tax / disruption_tax  (clipped to [0, 1])

    Note: disruption_tax = distortion_tax + selection_tax is preserved exactly.
    """
    prefix    = f"pc{commit_ms}"
    dev_x_col = f"{prefix}_dev_x"
    dev_z_col = f"{prefix}_dev_z"

    args = (df, bip_model, foul_model, xwoba_model, whiff_rv, foul_rv)

    xrv_realized = _xrv_from_shape(*args, commit_ms=commit_ms,
                                    zero_angular=False, zero_spatial=False)
    xrv_spatial  = _xrv_from_shape(*args, commit_ms=commit_ms,
                                    zero_angular=True,  zero_spatial=False)
    xrv_intended = _xrv_from_shape(*args, commit_ms=commit_ms,
                                    zero_angular=True,  zero_spatial=True)

    disruption_tax     = xrv_realized - xrv_intended
    spatial_distortion = xrv_spatial  - xrv_intended
    angular_disruption = xrv_realized - xrv_spatial

    # ── angular distortion share (mediator attribution) ───────────────────────
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

    # distortion_share: fraction of total disruption attributable to movement.
    # Defined post-hoc so it always satisfies distortion_tax + selection_tax = disruption_tax.
    distortion_share = np.where(
        disruption_tax.abs() > 1e-8,
        np.clip(distortion_tax / disruption_tax, 0.0, 1.0),
        np.nan,
    )

    out = df.copy()
    out["disruption_tax"]        = disruption_tax
    out["spatial_distortion_tax"] = spatial_distortion
    out["distortion_tax"]        = distortion_tax
    out["selection_tax"]         = selection_tax
    out["distortion_share"]      = distortion_share
    return out


# ── 4. Indirect effect (product-of-coefficients) ───────────────────────────────

def indirect_effect(mediator_models, bip_model, foul_model, xwoba_model,
                    whiff_rv, foul_rv, df, commit_ms=150):
    """Analytical indirect effect of post-commit movement on run value.

    For each angular deviation m:
      indirect_{m} = a_{m} × ∂xRV/∂m

    ∂xRV/∂m accounts for all three channels (BIP, foul, whiff):
      ∂xRV/∂m = ∂P(BIP)/∂m × (E[xwOBA] − foul_rv + (foul_rv − whiff_rv)×P(foul|not BIP))
               + P(BIP) × ∂E[xwOBA]/∂m
               + (1 − P(BIP)) × ∂P(foul|not BIP)/∂m × (foul_rv − whiff_rv)

    Returns a DataFrame with rows = deviation axes, columns:
      [a_x, a_z, grad_xrv, indirect_x, indirect_z]
    """
    prefix = f"pc{commit_ms}"
    dev_x  = f"{prefix}_dev_x"
    dev_z  = f"{prefix}_dev_z"

    score_cols = list(ANGULAR_DEVS) + ["plate_x", "plate_z", "balls", "strikes"]
    d = df[score_cols].dropna()

    p_bip   = bip_model.predict(d)
    p_foul_cond = foul_model.predict(d)
    e_xw    = xwoba_model.predict(d)

    beta_bip  = bip_model.params
    beta_foul = foul_model.params
    beta_xw   = xwoba_model.params

    whiff_val = d.apply(
        lambda r: whiff_rv.get((int(r["balls"]), int(r["strikes"])), whiff_rv["default"]),
        axis=1,
    ).mean()
    foul_val = d.apply(
        lambda r: foul_rv.get((int(r["balls"]), int(r["strikes"])), foul_rv["default"]),
        axis=1,
    ).mean()

    rows = []
    for dev_col, med_model in mediator_models.items():
        a_x = med_model.params.get(dev_x, np.nan)
        a_z = med_model.params.get(dev_z, np.nan)

        dp_bip_dm        = (beta_bip.get(dev_col, 0.0) * p_bip * (1 - p_bip)).mean()
        dp_foul_cond_dm  = (beta_foul.get(dev_col, 0.0) * p_foul_cond * (1 - p_foul_cond)).mean()
        de_xw_dm         = beta_xw.get(dev_col, 0.0)

        grad_xrv = (
            dp_bip_dm * (e_xw.mean() - foul_val + (foul_val - whiff_val) * p_foul_cond.mean())
            + p_bip.mean() * de_xw_dm
            + (1 - p_bip.mean()) * dp_foul_cond_dm * (foul_val - whiff_val)
        )

        rows.append({
            "dev_col":    dev_col,
            "a_x":        a_x,
            "a_z":        a_z,
            "grad_xrv":   grad_xrv,
            "indirect_x": a_x * grad_xrv,
            "indirect_z": a_z * grad_xrv,
        })

    return pd.DataFrame(rows).set_index("dev_col")


# ── 5. Validation ───────────────────────────────────────────────────────────────

def negative_control_check(df_with_tax, dev_total_col="pc150_dev_total",
                            pitch_type_col="pitch_type", threshold_in=1.0):
    """Negative control: near-straight four-seamers should show ~zero disruption tax.

    Filters to FF with post-commit deviation < threshold_in inches (converted to ft).
    A nonzero mean disruption_tax on this subset means the pre/post split is leaking
    selection into the distortion regressor.

    Prints summary and returns the filtered subset.
    """
    threshold_ft = threshold_in / 12.0
    ff_straight = df_with_tax[
        (df_with_tax[pitch_type_col] == "FF") &
        (df_with_tax[dev_total_col] < threshold_ft) &
        df_with_tax["disruption_tax"].notna()
    ]
    n = len(ff_straight)
    mean_tax = ff_straight["disruption_tax"].mean()
    print(f"Negative control (FF, dev_total < {threshold_in}\"):")
    print(f"  n={n:,}  mean disruption_tax={mean_tax:.4f} xRV/swing")
    print(f"  {'PASS' if abs(mean_tax) < 0.005 else 'WARN — possible selection leak'}")
    return ff_straight


def positive_control_check(df_with_tax, dev_total_col="pc150_dev_total",
                            pitch_type_col="pitch_type"):
    """Positive control: high-movement pitches should show the largest distortion tax.

    Compares mean distortion_tax across pitch types, sorted by mean post-commit deviation.
    Prints summary table.
    """
    agg = (
        df_with_tax[df_with_tax["distortion_tax"].notna()]
        .groupby(pitch_type_col)
        .agg(
            n=(dev_total_col, "size"),
            mean_dev_total=(dev_total_col, "mean"),
            mean_distortion_tax=("distortion_tax", "mean"),
            mean_disruption_tax=("disruption_tax", "mean"),
        )
        .query("n >= 200")
        .sort_values("mean_dev_total", ascending=False)
    )
    print("\nPositive control — distortion tax by pitch type (sorted by post-commit deviation):")
    print(agg.to_string(float_format="{:.4f}".format))
    return agg
