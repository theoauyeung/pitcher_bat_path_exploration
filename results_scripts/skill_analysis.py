"""
09_skill_analysis.py
Four-part pitcher distortion skill analysis:

1. Leaderboard  — pitcher-season distortion_tax with empirical-Bayes shrinkage
2. Axis fingerprint — which angular axis (VAA / HAA / tilt) each pitcher distorts most
3. Physical drivers — movement properties that predict distortion via OLS regression
4. Incremental validity — does current-season distortion_tax predict next-season xrv_vs_count?

Run:
    .venv\\Scripts\\python.exe results_scripts/09_skill_analysis.py

Outputs:
    results/09_leaderboard.csv
    results/09_axis_fingerprint.csv
    results/09_physical_drivers.csv
    results/figures/09_leaderboard.png
    results/figures/09_physical_drivers.png
    results/figures/09_incremental_validity.png
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

Path("results").mkdir(exist_ok=True)
Path("results/figures").mkdir(exist_ok=True)

MIN_SWINGS  = 200
COMMIT_MS   = 150
AXIS_COLS   = ("vert_attack_angle_dev", "horz_attack_angle_dev", "swing_path_tilt_dev")
AXIS_LABELS = {"vert_attack_angle_dev": "VAA", "horz_attack_angle_dev": "HAA",
               "swing_path_tilt_dev": "Tilt"}


# ── 1. Load and merge ─────────────────────────────────────────────────────────

print("Loading data...")
causal   = pd.read_parquet("results/xrv_causal.parquet")
precommit_cols = [
    "game_pk", "at_bat_number", "pitch_number", "game_year",
    f"pc{COMMIT_MS}_dev_x", f"pc{COMMIT_MS}_dev_z",
    "ball_bat_miss", "is_whiff",
]
precommit = pd.read_parquet("data/swings_precommit.parquet", columns=precommit_cols)

swing_xrv_cols = [
    "game_pk", "at_bat_number", "pitch_number",
    "pitcher_full_name",
    "pfx_x", "pfx_z", "release_speed", "release_extension", "arm_angle",
    "ivb_transverse", "hb_transverse", "release_spin_rate", "spin_axis",
    "pitcher_throws", "stuff_grade", "xrv_vs_count", "xrv",
]
sxrv = pd.read_parquet("results/swing_xrv.parquet", columns=swing_xrv_cols)

JOIN_KEYS = ["game_pk", "at_bat_number", "pitch_number"]
df = (
    causal
    .merge(precommit, on=JOIN_KEYS, how="left")
    .merge(sxrv,      on=JOIN_KEYS, how="left")
)
print(f"  merged: {len(df):,} rows | {df['game_year'].nunique()} years | "
      f"{df['pitcher_id'].nunique():,} pitchers")

causal_models = joblib.load("models/causal_models.joblib")
mediator_models = causal_models["mediator_models"]

dev_x_col = f"pc{COMMIT_MS}_dev_x"
dev_z_col = f"pc{COMMIT_MS}_dev_z"


# ── 2. Leaderboard ────────────────────────────────────────────────────────────

print("\nBuilding leaderboard...")

def _eb_shrink(group_means: pd.Series, group_ns: pd.Series,
               within_var: float, between_var: float, grand_mean: float) -> pd.Series:
    """Empirical-Bayes (James-Stein) shrinkage toward the grand mean.

    Reliability weight per pitcher: n / (n + within_var / between_var)
    shrunk = weight * raw_mean + (1 - weight) * grand_mean
    """
    w = group_ns / (group_ns + within_var / between_var)
    return w * group_means + (1 - w) * grand_mean


def build_leaderboard(df, metric="distortion_tax", min_swings=MIN_SWINGS):
    agg = (
        df[df[metric].notna()]
        .groupby(["pitcher_id", "pitcher_full_name", "game_year"])[metric]
        .agg(n="size", mean=np.mean, sd=np.std)
        .reset_index()
        .rename(columns={"mean": f"{metric}_mean", "sd": f"{metric}_sd"})
        .query(f"n >= {min_swings}")
    )
    # Empirical-Bayes shrinkage across all pitcher-seasons
    within_var  = agg[f"{metric}_sd"].pow(2).mean()
    between_var = agg[f"{metric}_mean"].var()
    grand_mean  = agg[f"{metric}_mean"].mean()
    agg[f"{metric}_shrunk"] = _eb_shrink(
        agg[f"{metric}_mean"], agg["n"], within_var, between_var, grand_mean
    )
    return agg.sort_values(f"{metric}_shrunk")


lb = build_leaderboard(df)
lb.to_csv("results/leaderboard.csv", index=False)
print(f"  {len(lb):,} pitcher-seasons  |  top distorter: "
      f"{lb.iloc[-1]['pitcher_full_name']} ({lb.iloc[-1]['distortion_tax_shrunk']:.4f} xRV/swing)")


# ── 3. Axis fingerprint ───────────────────────────────────────────────────────

print("\nBuilding axis fingerprint...")

# Movement-caused deviation per axis: a_x * dev_x + a_z * dev_z
# We use absolute mean so direction doesn't cancel within a pitcher's season.
axis_rows = []
for axis, model in mediator_models.items():
    a_x = model.params.get(dev_x_col, 0.0)
    a_z = model.params.get(dev_z_col, 0.0)
    df[f"_dist_dev_{axis}"] = (a_x * df[dev_x_col] + a_z * df[dev_z_col]).abs()

fingerprint_cols = [f"_dist_dev_{a}" for a in AXIS_COLS]
fp = (
    df[df[dev_x_col].notna()]
    .groupby(["pitcher_id", "pitcher_full_name", "game_year"])[fingerprint_cols + ["distortion_tax"]]
    .agg(n=("distortion_tax", "size"), **{c: (c, "mean") for c in fingerprint_cols})
    .reset_index()
    .query(f"n >= {MIN_SWINGS}")
)

total = fp[fingerprint_cols].sum(axis=1).replace(0, np.nan)
for col, axis in zip(fingerprint_cols, AXIS_COLS):
    fp[f"{AXIS_LABELS[axis]}_share"] = fp[col] / total

# dominant axis
fp["dominant_axis"] = fp[["VAA_share", "HAA_share", "Tilt_share"]].idxmax(axis=1)

out_cols = ["pitcher_id", "pitcher_full_name", "game_year", "n",
            "VAA_share", "HAA_share", "Tilt_share", "dominant_axis"]
fp[out_cols].to_csv("results/axis_fingerprint.csv", index=False)
print(f"  axis distribution across pitcher-seasons:\n"
      f"    {fp['dominant_axis'].value_counts().to_dict()}")

# clean up temp cols
df.drop(columns=fingerprint_cols, inplace=True)


# ── 4. Physical drivers ───────────────────────────────────────────────────────

print("\nFitting physical driver models...")

driver_features = [
    "pfx_x", "pfx_z", "release_speed", "release_extension",
    "arm_angle", "ivb_transverse", "hb_transverse",
]

# Aggregate to pitcher-pitch-type; normalize pfx_x sign to arm side for
# interpretable arm-side / glove-side coefficients
df["pfx_arm"]   = np.where(df["pitcher_throws"] == "R",  df["pfx_x"], -df["pfx_x"])
df["pfx_glove"] = np.where(df["pitcher_throws"] == "R", -df["pfx_x"],  df["pfx_x"])

pt_agg = (
    df[df["distortion_tax"].notna() & df["pfx_x"].notna()]
    .groupby(["pitcher_id", "pitch_type"])
    .agg(
        n          =("distortion_tax", "size"),
        distortion =("distortion_tax", "mean"),
        pfx_arm    =("pfx_arm",        "mean"),
        pfx_glove  =("pfx_glove",      "mean"),
        pfx_z      =("pfx_z",          "mean"),
        release_speed=("release_speed", "mean"),
        release_extension=("release_extension", "mean"),
        arm_angle  =("arm_angle",      "mean"),
        ivb_transverse=("ivb_transverse", "mean"),
        hb_transverse =("hb_transverse",  "mean"),
    )
    .reset_index()
    .query(f"n >= {MIN_SWINGS // 2}")   # lower threshold at pitch-type grain
)

driver_formula = (
    "distortion ~ pfx_arm + pfx_z + release_speed + "
    "release_extension + arm_angle + ivb_transverse + hb_transverse"
)
driver_model = smf.ols(driver_formula, pt_agg.dropna(subset=driver_formula.split(" ~ ")[1].split(" + "))).fit(cov_type="HC1")

driver_out = driver_model.summary2().tables[1].reset_index()
driver_out.columns = ["feature", "coef", "std_err", "t", "p_value", "ci_lower", "ci_upper"]
driver_out.to_csv("results/physical_drivers.csv", index=False)
print(f"  driver model R²={driver_model.rsquared:.3f} | n={int(driver_model.nobs)} pitcher-pitch-types")
print(driver_out[["feature", "coef", "p_value"]].to_string(index=False))


# ── 5. Incremental validity ───────────────────────────────────────────────────

print("\nIncremental validity analysis...")

MIN_WHIFFS = 50  # minimum whiff rows with measured ball_bat_miss per pitcher-season

# Distortion aggregation — all qualifying swings
distortion_agg = (
    df[df["distortion_tax"].notna()]
    .groupby(["pitcher_id", "pitcher_full_name", "game_year"])
    .agg(
        n              =("distortion_tax",          "size"),
        distortion_tax =("distortion_tax",          "mean"),
        adj_distortion =("adjusted_disruption_tax", "mean"),
        stuff_grade    =("stuff_grade",             "mean"),
        xrv            =("xrv",                     "mean"),
    )
    .reset_index()
    .query(f"n >= {MIN_SWINGS}")
)

# ball_bat_miss aggregation — whiff rows only where Hawk-Eye measured miss
bbm_agg = (
    df[(df["is_whiff"] == 1) & df["ball_bat_miss"].notna()]
    .groupby(["pitcher_id", "game_year"])
    .agg(n_whiffs=("ball_bat_miss", "size"), ball_bat_miss=("ball_bat_miss", "mean"))
    .reset_index()
    .query(f"n_whiffs >= {MIN_WHIFFS}")
)

season_agg = distortion_agg.merge(bbm_agg, on=["pitcher_id", "game_year"], how="left")

validity_rows = []
for year_a, year_b in [(2023, 2024), (2024, 2025)]:
    ya = season_agg[season_agg["game_year"] == year_a].copy()

    # Primary outcome: ball_bat_miss (all 3 years, most direct link to distortion)
    yb_bbm = (
        season_agg[season_agg["game_year"] == year_b][["pitcher_id", "ball_bat_miss"]]
        .rename(columns={"ball_bat_miss": "next_ball_bat_miss"})
    )
    pair_bbm = ya.merge(yb_bbm, on="pitcher_id", how="inner").dropna(
        subset=["ball_bat_miss", "next_ball_bat_miss", "distortion_tax"]
    )

    # Secondary outcome: model xRV per swing (available 2024-2025 only)
    yb_xrv = (
        season_agg[season_agg["game_year"] == year_b][["pitcher_id", "xrv"]]
        .rename(columns={"xrv": "next_xrv"})
    )
    pair_xrv = ya.merge(yb_xrv, on="pitcher_id", how="inner").dropna(
        subset=["xrv", "next_xrv", "distortion_tax"]
    )

    row = {"transition": f"{year_a}->{year_b}"}

    # -- ball_bat_miss validity --
    if len(pair_bbm) >= 20:
        m0 = smf.ols("next_ball_bat_miss ~ ball_bat_miss", pair_bbm).fit()
        m1 = smf.ols("next_ball_bat_miss ~ ball_bat_miss + distortion_tax", pair_bbm).fit()
        row.update({
            "bbm_n":           len(pair_bbm),
            "bbm_baseline_r2": m0.rsquared,
            "bbm_full_r2":     m1.rsquared,
            "bbm_delta_r2":    m1.rsquared - m0.rsquared,
            "bbm_coef":        m1.params.get("distortion_tax", np.nan),
            "bbm_pvalue":      m1.pvalues.get("distortion_tax", np.nan),
        })
        print(f"  {year_a}->{year_b} [ball_bat_miss]  n={len(pair_bbm)}  "
              f"baseline R2={m0.rsquared:.3f}  full R2={m1.rsquared:.3f}  "
              f"dR2={m1.rsquared - m0.rsquared:.3f}  "
              f"coef={m1.params['distortion_tax']:.3f} p={m1.pvalues['distortion_tax']:.3f}")
    else:
        print(f"  {year_a}->{year_b} [ball_bat_miss]: only {len(pair_bbm)} pitchers, skipping")

    # -- xRV validity --
    if len(pair_xrv) >= 20:
        m0x = smf.ols("next_xrv ~ xrv", pair_xrv).fit()
        m1x = smf.ols("next_xrv ~ xrv + distortion_tax", pair_xrv).fit()
        row.update({
            "xrv_n":           len(pair_xrv),
            "xrv_baseline_r2": m0x.rsquared,
            "xrv_full_r2":     m1x.rsquared,
            "xrv_delta_r2":    m1x.rsquared - m0x.rsquared,
            "xrv_coef":        m1x.params.get("distortion_tax", np.nan),
            "xrv_pvalue":      m1x.pvalues.get("distortion_tax", np.nan),
        })
        print(f"  {year_a}->{year_b} [xrv]           n={len(pair_xrv)}  "
              f"baseline R2={m0x.rsquared:.3f}  full R2={m1x.rsquared:.3f}  "
              f"dR2={m1x.rsquared - m0x.rsquared:.3f}  "
              f"coef={m1x.params['distortion_tax']:.3f} p={m1x.pvalues['distortion_tax']:.3f}")
    else:
        print(f"  {year_a}->{year_b} [xrv]: only {len(pair_xrv)} pitchers, skipping")

    if len(row) > 1:
        validity_rows.append(row)

validity_df = pd.DataFrame(validity_rows)
validity_df.to_csv("results/incremental_validity.csv", index=False)


# ── 6. Figures ────────────────────────────────────────────────────────────────

DARK  = "#2c3e50"
MID   = "#7f8c8d"
GOLD  = "#e8b84b"
RED   = "#e74c3c"
BLUE  = "#3498db"
GREEN = "#27ae60"
BG    = "#f8f8f8"

def _spine_clean(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MID)
    ax.spines["bottom"].set_color(MID)


# Figure 1: Top & bottom 20 distorters (2024 season)
print("\nRendering figures...")
lb24 = lb[lb["game_year"] == 2024].copy()
top20 = lb24.nlargest(20, "distortion_tax_shrunk")
bot20 = lb24.nsmallest(20, "distortion_tax_shrunk")
# Sort ascending so most negative (best distorters) appear at the top of the chart
show = pd.concat([top20, bot20]).drop_duplicates().sort_values("distortion_tax_shrunk", ascending=True)

fig, ax = plt.subplots(figsize=(10, 9), facecolor=BG)
ax.set_facecolor(BG)
# Blue = movement favors pitcher (negative), Gold = movement favors batter (positive)
colors = [GOLD if x > 0 else BLUE for x in show["distortion_tax_shrunk"]]
bars = ax.barh(show["pitcher_full_name"], show["distortion_tax_shrunk"],
               color=colors, height=0.7)
ax.axvline(0, color=DARK, lw=0.8)
ax.set_xlabel("Distortion Tax (xRV/swing, EB-shrunk)  |  Negative = pitcher advantage", color=DARK)
ax.set_title("Pitcher Distortion Tax — 2024 Season\n"
             "Most Disruptive (blue) vs. Batter-Favorable (gold)  |  min 200 swings, EB shrinkage",
             color=DARK, fontweight="bold")
_spine_clean(ax)
ax.tick_params(colors=DARK)
fig.tight_layout()
fig.savefig("results/figures/distortion_tax_leaderboard_2024.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Figure 2: Physical driver coefficients (forest plot)
plot_drivers = driver_out[driver_out["feature"] != "Intercept"].copy()
plot_drivers = plot_drivers.sort_values("coef")
sig = plot_drivers["p_value"] < 0.05

fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
ax.set_facecolor(BG)
dot_colors = [RED if s else MID for s in sig]
ax.barh(plot_drivers["feature"], plot_drivers["coef"],
        xerr=[plot_drivers["coef"] - plot_drivers["ci_lower"],
              plot_drivers["ci_upper"] - plot_drivers["coef"]],
        color=dot_colors, height=0.5, capsize=3, error_kw={"ecolor": DARK, "lw": 1})
ax.axvline(0, color=DARK, lw=0.8, ls="--")
ax.set_xlabel("Coefficient (xRV/swing per unit)", color=DARK)
ax.set_title(f"Physical Drivers of Distortion Tax\n"
             f"OLS at pitcher-pitch-type level  R²={driver_model.rsquared:.2f}  "
             f"(red = p<0.05, HC1 SE)", color=DARK, fontweight="bold")
_spine_clean(ax)
ax.tick_params(colors=DARK)
fig.tight_layout()
fig.savefig("results/figures/physical_drivers_ols.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Figure 3: Incremental validity scatter — ball_bat_miss (best available transition)
best_transition = "2024->2025" if "2024->2025" in validity_df["transition"].values else "2023->2024"
yr_a, yr_b = [int(x) for x in best_transition.split("->")]
ya = season_agg[season_agg["game_year"] == yr_a]
yb = season_agg[season_agg["game_year"] == yr_b][["pitcher_id", "ball_bat_miss"]]
pair = ya.merge(yb.rename(columns={"ball_bat_miss": "next_ball_bat_miss"}),
                on="pitcher_id", how="inner").dropna(
                    subset=["distortion_tax", "ball_bat_miss", "next_ball_bat_miss"])

fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
ax.set_facecolor(BG)
ax.scatter(pair["distortion_tax"], pair["next_ball_bat_miss"],
           alpha=0.55, color=BLUE, edgecolors="white", lw=0.4, s=45)
m_plot = smf.ols("next_ball_bat_miss ~ distortion_tax", pair).fit()
xs = np.linspace(pair["distortion_tax"].min(), pair["distortion_tax"].max(), 200)
ax.plot(xs, m_plot.params["Intercept"] + m_plot.params["distortion_tax"] * xs,
        color=RED, lw=2)
row = validity_df[validity_df["transition"] == best_transition].iloc[0]
ax.set_xlabel(f"{yr_a} Distortion Tax (xRV/swing)", color=DARK)
ax.set_ylabel(f"{yr_b} Mean Ball-Bat Miss (inches, whiffs)", color=DARK)
ax.set_title(
    f"Incremental Validity: {yr_a} Distortion Tax -> {yr_b} Ball-Bat Miss\n"
    f"n={row.get('bbm_n', '?')}  b={row.get('bbm_coef', float('nan')):.3f}  "
    f"p={row.get('bbm_pvalue', float('nan')):.3f}  dR2={row.get('bbm_delta_r2', float('nan')):.3f}",
    color=DARK, fontweight="bold",
)
_spine_clean(ax)
ax.tick_params(colors=DARK)
fig.tight_layout()
fig.savefig("results/figures/distortion_tax_incremental_validity.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nDone.")
print("  results/leaderboard.csv")
print("  results/axis_fingerprint.csv")
print("  results/physical_drivers.csv")
print("  results/incremental_validity.csv")
print("  results/figures/distortion_tax_leaderboard_2024.png")
print("  results/figures/physical_drivers_ols.png")
print("  results/figures/distortion_tax_incremental_validity.png")
