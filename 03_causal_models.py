"""
Phase B: run-value mediation models.

DAG:
  post-commit movement  →  angular deviation  →  run value
      (treatment)            (mediator, §A)        (outcome)

The total effect of post-commit movement on run value decomposes into:
  - indirect (distortion): the path mediated by measurable swing shape change
  - direct (residual, including selection): everything else

Three components
────────────────
1. Mediator models (one per angular deviation axis)
     dev ~ pc{ms}_dev_x + pc{ms}_dev_z          ← post-commit treatment
           + pc{ms}_x_proj + pc{ms}_z_proj       ← pre-commit projected plate location
           + release_speed + balls + strikes      ← pitch and count controls
           + offset_y_ms                          ← timing (removes arc-sampling artifact)
           + (1 | batter_id)                      ← batter grouping
   LinearMixedLM (statsmodels). Treatment coefficients (a_x, a_z) per deviation axis
   are the causal leverage: how much post-commit movement shifts each swing dimension.

2. Outcome models — three channels (foul balls separated from whiffs)
     P(BIP):              logistic on deviation + plate location + count
     P(foul | not BIP):   logistic on same, fit on non-BIP swings only
     xwOBAcon:            linear on same, BIP only
   Composite: xRV = P(BIP)×E[xwOBA|BIP] + P(foul)×foul_rv[count] + P(whiff)×whiff_rv[count]
   where P(foul) = (1−P(BIP))×P(foul|not BIP), P(whiff) = (1−P(BIP))×(1−P(foul|not BIP)).
   Foul run value differs from whiff: fouls are count-neutral at 2 strikes, advance the
   count at 0-1 strikes. Conflating them biases xRV at moderate disruption levels.

3. Disruption tax and distortion attribution
     xrv_realized  = predict(realized swing shape)
     xrv_intended  = predict(intended_{metric} from Phase A, same controls)
     disruption_tax = xrv_realized − xrv_intended  [negative = pitcher cost batter runs]

   Distortion share per swing:
     distortion_dev_m  = a_x_m × pc_dev_x + a_z_m × pc_dev_z   (mediator prediction)
     total_dev_m       = {metric}_dev
     distortion_share  = ||distortion_dev|| / (||distortion_dev|| + ||selection_dev||)
     where selection_dev = total_dev − distortion_dev

   Final:
     distortion_tax = disruption_tax × distortion_share
     selection_tax  = disruption_tax × (1 − distortion_share)

Indirect effect (product-of-coefficients):
  For each deviation axis m and each outcome channel c:
     indirect_{m,c} = a_{x→m} × ∂c/∂m  (gradient from outcome model, evaluated at realized shape)
  Summed across axes and channels gives the analytical indirect-effect estimate.
  Consistent with the counterfactual tax under linearity; differ only through nonlinearity in
  the contact channel.

Negative-control check built in: filter to FF (four-seam fastballs) with dev_total < threshold
and verify that disruption_tax ≈ 0.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


# ── Column conventions ──────────────────────────────────────────────────────────

ANGULAR_DEVS = ("vert_attack_angle_dev", "horz_attack_angle_dev", "swing_path_tilt_dev")
ANGULAR_RESP = ("vert_attack_angle", "horz_attack_angle", "swing_path_tilt")

_DEVIATION_TERMS  = " + ".join(ANGULAR_DEVS)
_LOCATION_TERMS   = "plate_x + plate_z + balls + strikes"
_OUTCOME_CONTROLS = f"{_DEVIATION_TERMS} + {_LOCATION_TERMS}"


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

def fit_outcome_models(df):
    """Fit all outcome components. Returns (bip_model, foul_model, xwoba_model, whiff_rv_table).

    bip_model      — logistic P(BIP | swing) ~ angular deviations + pitch controls (HC1 SEs)
    foul_model     — logistic P(foul | not BIP) ~ same predictors, fit on non-BIP swings
    xwoba_model    — OLS xwOBAcon ~ same predictors, BIP only (fouls/whiffs dropped via NaN)
    whiff_rv_table — empirical mean delta_run_exp per (balls, strikes) cell on whiffs
    """
    d = df.copy()
    if "is_foul" not in d.columns:
        # foul: made contact but not a ball in play
        d["is_foul"] = ((d["is_contact"] == 1) & (d["is_bip"] != 1)).astype(int)

    d_bip = d[list(ANGULAR_DEVS) + ["is_bip", "plate_x", "plate_z",
                                     "balls", "strikes"]].dropna()
    bip_model = smf.logit(f"is_bip ~ {_OUTCOME_CONTROLS}", d_bip).fit(
        cov_type="HC1", disp=False
    )

    # P(foul | not BIP): fit on non-BIP swings so the model is conditional on not going in play
    d_foul = d.loc[d["is_bip"] == 0,
                   list(ANGULAR_DEVS) + ["is_foul", "plate_x", "plate_z",
                                          "balls", "strikes"]].dropna()
    foul_model = smf.logit(f"is_foul ~ {_OUTCOME_CONTROLS}", d_foul).fit(
        cov_type="HC1", disp=False
    )

    d_xwoba = df.loc[df["is_bip"] == 1,
                     list(ANGULAR_DEVS) + ["xwoba", "plate_x", "plate_z",
                                            "balls", "strikes"]].dropna()
    xwoba_model = smf.ols(f"xwoba ~ {_OUTCOME_CONTROLS}", d_xwoba).fit(cov_type="HC1")

    whiffs = df[(df["is_swing"] == 1) & (df["is_whiff"] == 1) &
                df["delta_run_exp"].notna()]
    whiff_rv = whiffs.groupby(["balls", "strikes"])["delta_run_exp"].mean().to_dict()
    whiff_rv["default"] = whiffs["delta_run_exp"].mean()

    return bip_model, foul_model, xwoba_model, whiff_rv


# ── 3. Disruption tax ───────────────────────────────────────────────────────────

def _xrv_from_shape(df, bip_model, foul_model, xwoba_model, whiff_rv, foul_rv,
                    use_zero_devs=False):
    """Predict per-swing xRV given a swing shape.

    use_zero_devs=True:  evaluate at the intended swing (all angular deviations = 0).
    use_zero_devs=False: evaluate at the realized swing (actual angular deviations).

    Three-channel formula:
      P(foul) = (1 − P(BIP)) × P(foul | not BIP)
      P(whiff) = (1 − P(BIP)) × (1 − P(foul | not BIP))
      xRV = P(BIP)×E[xwOBA|BIP] + P(foul)×foul_rv[count] + P(whiff)×whiff_rv[count]
    """
    score = df[["plate_x", "plate_z", "balls", "strikes"]].copy()
    for dev in ANGULAR_DEVS:
        score[dev] = 0.0 if use_zero_devs else df[dev]

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
    """Full disruption tax with distortion/selection decomposition.

    disruption_tax:   xRV(realized) − xRV(intended)  [negative = pitcher cost batter runs]
    distortion_share: fraction of angular deviation variance explained by post-commit movement
    distortion_tax:   disruption_tax × distortion_share
    selection_tax:    disruption_tax × (1 − distortion_share)

    Returns df with the four columns above added.
    """
    prefix    = f"pc{commit_ms}"
    dev_x_col = f"{prefix}_dev_x"
    dev_z_col = f"{prefix}_dev_z"

    # ── disruption tax (predict-twice) ────────────────────────────────────────
    xrv_realized = _xrv_from_shape(df, bip_model, foul_model, xwoba_model, whiff_rv, foul_rv,
                                    use_zero_devs=False)
    xrv_intended = _xrv_from_shape(df, bip_model, foul_model, xwoba_model, whiff_rv, foul_rv,
                                    use_zero_devs=True)
    tax = (xrv_realized - xrv_intended).rename("disruption_tax")

    # ── distortion attribution ────────────────────────────────────────────────
    attr_cols = {}
    for dev_col, model in mediator_models.items():
        a_x = model.params.get(dev_x_col, 0.0)
        a_z = model.params.get(dev_z_col, 0.0)
        distortion = a_x * df[dev_x_col] + a_z * df[dev_z_col]
        attr_cols[f"distortion_dev_{dev_col}"] = distortion
        attr_cols[f"selection_dev_{dev_col}"]  = df[dev_col] - distortion

    attr = pd.DataFrame(attr_cols, index=df.index)
    # Squared-norm share: proportion of total deviation variance attributable to
    # post-commit movement. Avoids the triangle-inequality problem (dist_norm +
    # sel_norm ≠ total_norm when the two components point in opposite directions).
    distortion_sq = sum(attr[f"distortion_dev_{d}"] ** 2 for d in ANGULAR_DEVS)
    total_sq      = sum(df[d] ** 2 for d in ANGULAR_DEVS)
    distortion_share = np.where(total_sq > 1e-8, distortion_sq / total_sq, np.nan)

    out = df.copy()
    out["disruption_tax"]   = tax
    out["distortion_share"] = distortion_share
    out["distortion_tax"]   = out["disruption_tax"] * out["distortion_share"]
    out["selection_tax"]    = out["disruption_tax"] * (1 - out["distortion_share"])
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
