"""
Phase A: Batter intended-swing model.

Estimates each batter's intended swing shape conditional on count, pitch location,
and timing. The deviation residual is the mediator that Phase B prices in run value.

Powers-Yurko skeleton extended per proj_desc.md §8:

  Responses — primary (angular, Gaussian/Student-t):
    vert_attack_angle   — attack angle at contact
    horz_attack_angle   — attack direction at contact
    swing_path_tilt     — tilt of the bat head over ~40ms before contact

  Responses — secondary (effort, skew-normal):
    bat_speed, swing_length

  Population predictors (angular):
    count (balls, strikes)
    plate_x_bat  — location in the batter's frame (inside is positive for both hands)
    plate_z, plate_z²  — quadratic smooth on height; batters tilt to match pitch plane,
                          under-modeling height mislabels appropriate adaptation as error
    offset_y_ms  — timing (early/on-time/late); removes arc-sampling artifact from deviation

  Population predictors (effort):
    count, plate_x_bat, plate_z (no quadratic, no timing — location is a minor modifier)

  Batter RE: intercept + slopes on strikes + plate_x_bat + plate_z  (one per response)
  Pitcher RE: intercept only (partials mound quality out of the intention baseline)

True joint mvbind fit (correlated batter RE across responses) requires brms. Python
approximation: separate Bambi fits per response + empirical residual covariance matrix
for Mahalanobis deviation. The joint covariance structure is recovered post-hoc rather
than jointly identified, which is the only material concession.

Outputs per swing:
  intended_{metric}   — E[metric | batter, count, location, timing]
  {metric}_dev        — realized - intended  (the Phase B mediator)
  angular_mahal       — Mahalanobis distance in joint angular space
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


# ── Mahalanobis distance in joint angular space ────────────────────────────────

def angular_mahalanobis(swings_with_devs, cov=None):
    """Mahalanobis distance of each swing's angular deviation from intention.

    cov: 3×3 covariance matrix of angular deviations. If None, estimated from the
         swings_with_devs data itself (empirical joint residual covariance). Pass
         the training-set covariance when scoring new data to keep a fixed metric space.

    Returns a Series indexed like swings_with_devs.
    """
    dev_cols = [f"{r}_dev" for r in ANGULAR]
    D = swings_with_devs[dev_cols].dropna()

    if cov is None:
        cov = np.cov(D.values.T)

    cov_inv = np.linalg.inv(cov)
    d_vals = D.values
    # Mahalanobis: sqrt(d^T Σ^{-1} d) row-wise
    mahal = np.sqrt(np.einsum("ij,jk,ik->i", d_vals, cov_inv, d_vals))

    result = pd.Series(np.nan, index=swings_with_devs.index)
    result.loc[D.index] = mahal
    return result


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


# ── Batter Mahalanobis plot ────────────────────────────────────────────────────

def _mahal_ellipse(cov2, center=(0, 0), n_std=2.0, **kwargs):
    """Return a Matplotlib Ellipse patch for the n_std Mahalanobis contour of a 2×2 cov."""
    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    return Ellipse(xy=center, width=width, height=height, angle=angle, **kwargs)


def plot_batter_mahalanobis(swings_with_devs, batter_id, cov=None,
                            color_by="pitch_type", n_label=5,
                            x_axis="vert_attack_angle_dev",
                            y_axis="swing_path_tilt_dev"):
    """Three-panel diagnostic for one batter's angular swing deviations.

    Panel 1 — 2D scatter in angular deviation space (x_axis vs y_axis), points
               colored by Mahalanobis distance on a warm colormap. The 1σ and 2σ
               Mahalanobis ellipses (from the 2D marginal of the joint covariance)
               are overlaid. The n_label most disrupted swings are annotated with
               their pitch type.

    Panel 2 — Same scatter, points colored by pitch_type (or whatever color_by
               column is passed), so you can see which pitch types drive outliers.

    Panel 3 — Histogram of Mahalanobis distances with vertical lines at the 50th,
               90th, and 95th percentiles.

    Parameters
    ----------
    swings_with_devs : DataFrame
        Output of swing_deviations() with angular_mahal column added.
    batter_id : int or str
        The batter to plot.
    cov : 3×3 ndarray, optional
        Joint angular covariance matrix (from angular_mahalanobis). If None,
        estimated from this batter's own swings.
    color_by : str
        Column used to color Panel 2 (default "pitch_type").
    n_label : int
        Number of highest-Mahalanobis swings to annotate.
    x_axis, y_axis : str
        Angular deviation columns for the scatter axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    b = swings_with_devs[swings_with_devs["batter_id"] == batter_id].copy()
    if b.empty:
        raise ValueError(f"No rows found for batter_id={batter_id!r}")

    dev_cols = [f"{r}_dev" for r in ANGULAR]
    b_clean = b[dev_cols + ["angular_mahal"]].dropna()
    if b_clean.empty:
        raise ValueError(f"Batter {batter_id} has no rows with complete angular deviations")

    # ── marginal 2D covariance for ellipses ───────────────────────────────────
    x_col, y_col = x_axis, y_axis
    xy_data = b_clean[[x_col, y_col]].values
    if cov is not None:
        xi = dev_cols.index(x_col)
        yi = dev_cols.index(y_col)
        cov2 = cov[np.ix_([xi, yi], [xi, yi])]
    else:
        cov2 = np.cov(xy_data.T)

    mahal = b_clean["angular_mahal"]
    name  = b["batter_full_name"].iloc[0] if "batter_full_name" in b.columns else str(batter_id)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{name}  —  Angular swing deviations from intention baseline",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Mahalanobis distance as color ────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(
        b_clean[x_col], b_clean[y_col],
        c=mahal, cmap="YlOrRd", s=18, alpha=0.7,
        vmin=mahal.quantile(0.05), vmax=mahal.quantile(0.95),
    )
    plt.colorbar(sc, ax=ax, label="Mahalanobis distance")

    for n_std, alpha, lw in [(1.0, 0.6, 1.5), (2.0, 0.4, 1.0)]:
        ell = _mahal_ellipse(cov2, n_std=n_std,
                             edgecolor="steelblue", facecolor="none",
                             linestyle="--", linewidth=lw, alpha=alpha)
        ax.add_patch(ell)

    # annotate the n_label most disrupted swings
    top_idx = b_clean["angular_mahal"].nlargest(n_label).index
    for idx in top_idx:
        row = b.loc[idx]
        pt  = str(row.get("pitch_type", "?"))
        ax.annotate(pt,
                    xy=(b_clean.loc[idx, x_col], b_clean.loc[idx, y_col]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="darkred")

    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel(_axis_label(x_col))
    ax.set_ylabel(_axis_label(y_col))
    ax.set_title("Deviation colored by Mahalanobis distance")

    # ── Panel 2: pitch-type color ─────────────────────────────────────────────
    ax = axes[1]
    if color_by in b.columns:
        categories = b.loc[b_clean.index, color_by].fillna("?")
        unique_cats = sorted(categories.unique())
        palette = plt.cm.tab10.colors
        cat_colors = {c: palette[i % len(palette)] for i, c in enumerate(unique_cats)}
        colors = categories.map(cat_colors)
        ax.scatter(b_clean[x_col], b_clean[y_col],
                   c=colors, s=18, alpha=0.7)
        legend_handles = [
            mpatches.Patch(color=cat_colors[c], label=c) for c in unique_cats
        ]
        ax.legend(handles=legend_handles, fontsize=7, title=color_by,
                  loc="upper right", framealpha=0.7)
    else:
        ax.scatter(b_clean[x_col], b_clean[y_col], s=18, alpha=0.5, color="steelblue")

    for n_std, alpha, lw in [(1.0, 0.6, 1.5), (2.0, 0.4, 1.0)]:
        ell = _mahal_ellipse(cov2, n_std=n_std,
                             edgecolor="steelblue", facecolor="none",
                             linestyle="--", linewidth=lw, alpha=alpha)
        ax.add_patch(ell)

    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel(_axis_label(x_col))
    ax.set_ylabel(_axis_label(y_col))
    ax.set_title(f"Deviation colored by {color_by}")

    # ── Panel 3: histogram of Mahalanobis distances ───────────────────────────
    ax = axes[2]
    ax.hist(mahal, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    for pct, ls, label in [(50, "--", "p50"), (90, "-.", "p90"), (95, ":", "p95")]:
        v = mahal.quantile(pct / 100)
        ax.axvline(v, color="darkred", linestyle=ls, linewidth=1.2,
                   label=f"{label}={v:.2f}")
    ax.set_xlabel("Mahalanobis distance")
    ax.set_ylabel("Swings")
    ax.set_title("Distribution of disruption distance")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def _axis_label(col):
    labels = {
        "vert_attack_angle_dev":  "Attack angle deviation (°)",
        "horz_attack_angle_dev":  "Attack direction deviation (°)",
        "swing_path_tilt_dev":    "Swing path tilt deviation (°)",
    }
    return labels.get(col, col)
