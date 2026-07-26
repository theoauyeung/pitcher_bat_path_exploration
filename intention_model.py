"""
Batter intended-swing model.

Estimates what each batter intended to do given count, pitch location, contact
timing, and platoon handedness. Per-batter random effects capture baseline swing
tendencies and how each batter adjusts under count pressure.

The residual (realized − intended) is the swing deviation used as the Phase B
mediator. A larger residual means the batter's swing drifted further from their
plan — either due to late pitch movement or a poor decision.

Five responses: vert_attack_angle, horz_attack_angle, swing_path_tilt (angular),
bat_speed, swing_length (effort). Each gets a separate Bambi/PyMC Gaussian LMM.

Default inference: method="vi" (ADVI, ~2 min). Only posterior means are used
downstream so VI is equivalent to MCMC for this pipeline.

Outputs per swing:
  intended_{metric}   — posterior-mean predicted swing shape
  {metric}_dev        — realized − intended  (the Phase B mediator)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import bambi as bmb


# ── Response sets ──────────────────────────────────────────────────────────────

ANGULAR  = ("vert_attack_angle", "horz_attack_angle", "swing_path_tilt")
EFFORT   = ("bat_speed", "swing_length")
ALL_RESPONSES = ANGULAR + EFFORT


# ── Model formulas ─────────────────────────────────────────────────────────────

def _angular_formula(resp):
    """Quadratic smooth on plate_z + timing for the three angular mediators.

    Batter RE is intercept + strikes only. Including plate_x_bat/plate_z in the
    random slopes creates a 4×4 LKJ prior that makes NUTS degenerate (max_treedepth
    warnings, R-hat > 1.01 for tail batters regardless of tuning). Location effects
    are captured by the fixed effects; the random slope captures per-batter count
    adjustment, which is the intention signal.
    """
    return (
        f"{resp} ~ scale(balls) + scale(strikes)"
        " + scale(plate_x_bat) + scale(plate_z) + scale(plate_z_sq)"
        " + scale(offset_y_ms) + offset_y_ms_missing"
        " + pitcher_throws_L + pitcher_throws_L:scale(plate_x_bat)"
        " + (1 + scale(strikes) | batter_id)"
        " + (1 | pitcher_id)"
    )


def _effort_formula(resp):
    """Simpler formula for bat speed and swing length; no quadratic or timing."""
    return (
        f"{resp} ~ scale(balls) + scale(strikes)"
        " + scale(plate_x_bat) + scale(plate_z)"
        " + pitcher_throws_L"
        " + (1 + scale(strikes) | batter_id)"
        " + (1 | pitcher_id)"
    )


# ── Data preparation ───────────────────────────────────────────────────────────

def _prep(df, n_subsample=None, seed=345):
    """Filter to valid tracked swings and build derived columns."""
    need = list(ALL_RESPONSES) + [
        "balls", "strikes", "plate_x", "plate_z",
        "batter_stand", "batter_id", "pitcher_id",
        "offset_y_ms", "pitcher_throws",
    ]
    # Drop rows with NaN in any formula variable except offset_y_ms (imputed below).
    drop_on = list(ALL_RESPONSES) + ["balls", "strikes", "plate_x", "plate_z",
                                      "batter_stand", "batter_id", "pitcher_id",
                                      "pitcher_throws"]
    d = df.loc[df["is_swing"].astype(bool), need].dropna(subset=drop_on).copy()
    d = d[d["bat_speed"] > 0]

    # location in batter's frame: inside is positive for both hands
    d["plate_x_bat"] = d["plate_x"] * d["batter_stand"].map({"R": -1.0, "L": 1.0})
    d["plate_z_sq"] = d["plate_z"] ** 2
    # pitcher handedness: absorbs spin-direction reversal across platoon matchups
    d["pitcher_throws_L"] = (d["pitcher_throws"] == "L").astype(float)

    d["batter_id"]  = d["batter_id"].astype("category")
    d["pitcher_id"] = d["pitcher_id"].astype("category")

    # timing: missing indicator + mean imputation.
    # Imputing 0 without indicator biases the timing coefficient because missingness
    # is systematic (likely correlated with contact quality / whiff rate).
    # The indicator absorbs the average shift for missing-timing rows; the timing
    # coefficient is then identified only from rows where it was observed.
    d["offset_y_ms_missing"] = d["offset_y_ms"].isna().astype(float)
    timing_mean = float(d["offset_y_ms"].mean())  # mean of observed
    d["offset_y_ms"] = d["offset_y_ms"].fillna(timing_mean)

    if n_subsample is not None and n_subsample < len(d):
        d = d.sample(n=n_subsample, random_state=seed)
    return d.reset_index(drop=True)


# ── Fitting ────────────────────────────────────────────────────────────────────

def fit(df, n_subsample=None, draws=1000, tune=1000, chains=4,
        target_accept=0.9, max_treedepth=15, cores=4, seed=345, responses=None,
        method="mcmc", n_vi_iter=50_000, n_vi_draws=1000):
    """Fit the Phase A intention models.

    method:       "mcmc" (default) or "vi" (ADVI, ~2 min, equivalent posterior means).
                  Use "vi" when only point estimates are needed — all downstream code
                  uses posterior means only so VI output is functionally identical.
    n_vi_iter:    ADVI iterations (default 50k). Ignored for method="mcmc".
    n_vi_draws:   draws sampled from the VI approximation to build InferenceData.
                  Ignored for method="mcmc".
    n_subsample:  rows for fitting. Default None = full dataset.
    responses:    subset of ALL_RESPONSES to fit; default is all five.
    Returns {"models": {resp: bmb.Model}, "idata": {resp: InferenceData}, "data": d}.
    """
    d = _prep(df, n_subsample=n_subsample, seed=seed)

    if responses is None:
        responses = ALL_RESPONSES

    models, idata = {}, {}
    for resp in responses:
        formula = _angular_formula(resp) if resp in ANGULAR else _effort_formula(resp)

        # Bambi 0.18 lacks skewnormal; gaussian used for all responses.
        model = bmb.Model(formula, d, family="gaussian")

        if method == "vi":
            approx = model.fit(
                inference_method="vi",
                n=n_vi_iter,
                random_seed=seed,
            )
            # approx.sample() requires the PyMC model on the context stack
            with model.backend.model:
                idata[resp] = approx.sample(n_vi_draws, return_inferencedata=True)
        else:
            idata[resp] = model.fit(
                draws=draws, tune=tune, chains=chains, cores=cores,
                target_accept=target_accept, random_seed=seed,
                nuts={"max_treedepth": max_treedepth},
            )
        models[resp] = model

    return {"models": models, "idata": idata, "data": d}


# ── Prediction ─────────────────────────────────────────────────────────────────

def _posterior_mean_predict(idata, scoring, d_train):
    """Posterior-mean linear predictor. Bypasses Bambi's model.predict() which
    materializes (n_obs × n_groups × n_draws) arrays and OOMs at 763k observations.

    Computes E[y|X,θ̄] = X @ β̄ + Z_batter @ ū_batter where β̄/ū are posterior means.
    Pitcher RE is excluded (always 0 — the intention baseline strips mound quality).
    Unseen batters get RE = 0 (population mean).

    Valid for Gaussian LMM because E[Xβ + Zu] = X E[β] + Z E[u].
    """
    post = idata.posterior

    _RE_suffixes = ("_sigma", "_offset")
    _skip_vars = {"sigma", "mu"}

    # --- Posterior mean of scalar fixed effects ---
    beta = {
        v: float(post[v].mean(dim=("chain", "draw")).values)
        for v in post.data_vars
        if (set(post[v].dims) == {"chain", "draw"}
            and v not in _skip_vars
            and not any(v.endswith(s) for s in _RE_suffixes))
    }

    # --- Posterior mean of batter RE vectors ---
    batter_cats = d_train["batter_id"].cat.categories
    batter_re = {}
    for v in post.data_vars:
        if ("batter_id__factor_dim" in post[v].dims
                and not any(v.endswith(s) for s in _RE_suffixes)):
            batter_re[v] = dict(zip(batter_cats, post[v].mean(dim=("chain", "draw")).values))

    # --- Scaling stats from training data (mirrors Bambi's scale() transform) ---
    scale_stats = {}
    all_vars = list(beta) + list(batter_re)
    for v in all_vars:
        for part in v.split(":"):
            if part.startswith("scale("):
                col = part[6:part.index(")")]
                if col not in scale_stats:
                    scale_stats[col] = (float(d_train[col].mean()), float(d_train[col].std()))

    def _z(col):
        m, s = scale_stats[col]
        return (scoring[col].values - m) / s

    # --- Fixed effects linear predictor ---
    yhat = np.zeros(len(scoring), dtype=np.float64)
    for v, coef in beta.items():
        if v == "Intercept":
            yhat += coef
        elif v.startswith("scale(") and ":" not in v:
            yhat += coef * _z(v[6:-1])
        elif ":" in v:
            feat = np.ones(len(scoring), dtype=np.float64)
            for part in v.split(":"):
                if part.startswith("scale("):
                    feat *= _z(part[6:-1])
                else:
                    feat *= scoring[part].values.astype(float)
            yhat += coef * feat
        else:
            yhat += coef * scoring[v].values.astype(float)

    # --- Batter RE ---
    for re_var, re_dict in batter_re.items():
        re_vals = scoring["batter_id"].map(re_dict).fillna(0.0).values
        if re_var == "1|batter_id":
            yhat += re_vals
        elif re_var.startswith("scale(") and "|" in re_var:
            col = re_var[6:re_var.index(")")]
            yhat += re_vals * _z(col)

    return yhat


def predict_intended(result, swings):
    """Per-swing intended swing shape via posterior-mean linear predictor.

    Pitcher RE is excluded (intention baseline strips mound quality).
    Batter RE is included — this is the intention signal.
    Unseen batters (not in training subsample) get RE = 0.

    Returns a DataFrame indexed like swings with intended_{resp} per response.
    """
    d_train = result["data"]
    _timing_mean = float(d_train["offset_y_ms"].mean())

    scoring = swings.copy()

    if "plate_x_bat" not in scoring.columns:
        stand = scoring.get("batter_stand", pd.Series("R", index=scoring.index))
        scoring["plate_x_bat"] = scoring["plate_x"] * stand.map({"R": -1.0, "L": 1.0})

    if "plate_z_sq" not in scoring.columns:
        scoring["plate_z_sq"] = scoring["plate_z"] ** 2

    if "pitcher_throws_L" not in scoring.columns:
        if "pitcher_throws" in scoring.columns:
            scoring["pitcher_throws_L"] = (scoring["pitcher_throws"] == "L").astype(float)
        else:
            scoring["pitcher_throws_L"] = 0.0

    scoring["offset_y_ms_missing"] = (
        scoring["offset_y_ms"].isna().astype(float)
        if "offset_y_ms" in scoring.columns else 1.0
    )
    scoring["offset_y_ms"] = (
        scoring.get("offset_y_ms", pd.Series(_timing_mean, index=scoring.index))
               .fillna(_timing_mean)
    )

    out = {}
    for resp, idata_fitted in result["idata"].items():
        out[f"intended_{resp}"] = _posterior_mean_predict(idata_fitted, scoring, d_train)

    return pd.DataFrame(out, index=swings.index)


# ── Deviation residuals ────────────────────────────────────────────────────────

def swing_deviations(swings, intended_df):
    """Add {metric}_dev = realized − intended columns to swings.

    Positive values mean the batter's realized shape exceeded their intention
    (pitched higher than expected → attack angle higher than intended, etc.).
    The Phase B mediator model regresses these deviations on post-commit movement.
    """
    out = swings.copy()
    for resp in result_responses(intended_df):
        out[f"{resp}_dev"] = out[resp] - intended_df[f"intended_{resp}"]
    return out


def result_responses(intended_df):
    """Extract fitted response names from an intended_df.

    Filters to columns in ALL_RESPONSES only, so the _sd uncertainty columns
    (intended_{resp}_sd) added by predict_intended are not mistaken for responses.
    """
    return [r for r in ALL_RESPONSES if f"intended_{r}" in intended_df.columns]


# ── Within-batter calibration ──────────────────────────────────────────────────

def calibrate(swings_with_devs, responses=None, min_swings=30):
    """Within-batter calibration check for the Phase A intention model.

    For each batter with >= min_swings, computes:
      - mean_dev: mean signed deviation (bias; should be near 0 due to batter intercept)
      - corr: Pearson correlation between intended and realized
      - rmse: root-mean-square deviation from intention

    Returns a DataFrame with one row per batter per response, plus a summary dict.
    Prints a human-readable summary to stdout.
    """
    if responses is None:
        responses = [r for r in ALL_RESPONSES if f"{r}_dev" in swings_with_devs.columns]

    rows = []
    for resp in responses:
        dev_col = f"{resp}_dev"
        int_col = f"intended_{resp}"
        if dev_col not in swings_with_devs.columns or int_col not in swings_with_devs.columns:
            continue

        for batter, g in swings_with_devs.groupby("batter_id"):
            g = g[[resp, int_col, dev_col]].dropna()
            if len(g) < min_swings:
                continue
            rows.append({
                "batter_id": batter,
                "response":  resp,
                "n":         len(g),
                "mean_dev":  g[dev_col].mean(),
                "rmse":      np.sqrt((g[dev_col] ** 2).mean()),
                "corr":      g[resp].corr(g[int_col]),
            })

    cal = pd.DataFrame(rows)
    if cal.empty:
        print("Calibration: no batters met the minimum swing threshold.")
        return cal

    print("── Phase A calibration summary ─────────────────────────────────")
    for resp in responses:
        sub = cal[cal["response"] == resp]
        if sub.empty:
            continue
        print(f"\n  {resp}")
        print(f"    batters: {len(sub)}")
        print(f"    mean_dev  mean={sub['mean_dev'].mean():.3f}  "
              f"|mean|<0.5: {(sub['mean_dev'].abs() < 0.5).mean():.0%}")
        print(f"    rmse      mean={sub['rmse'].mean():.3f}  median={sub['rmse'].median():.3f}")
        print(f"    corr      mean={sub['corr'].mean():.3f}  "
              f"corr>0.3: {(sub['corr'] > 0.3).mean():.0%}")
    print()

    return cal



def _axis_label(col):
    labels = {
        "vert_attack_angle_dev":  "Attack angle deviation (°)",
        "horz_attack_angle_dev":  "Attack direction deviation (°)",
        "swing_path_tilt_dev":    "Swing path tilt deviation (°)",
    }
    return labels.get(col, col)
