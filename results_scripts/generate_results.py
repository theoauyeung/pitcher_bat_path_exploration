"""
generate_results.py
All paper figures from one script.

Usage
-----
    .venv\\Scripts\\python.exe results_scripts/generate_results.py              # all
    .venv\\Scripts\\python.exe results_scripts/generate_results.py axis         # one key
    .venv\\Scripts\\python.exe results_scripts/generate_results.py reliability leaderboard

Figure keys
-----------
    leaderboard   — EB-shrunk distortion tax bar chart (2024)
    axis          — Angular axis fingerprint: VAA/HAA/Tilt dominance by handedness  [NEW]
    drivers       — Physical driver OLS forest plot
    incremental   — Incremental validity scatter
    outcomes      — Outcome rates by tax quintile
    xwoba         — xwOBA vs distortion/selection tax
    reliability   — Split-half and YoY reliability table
    count_effects — Intended swing shape by count (intention model)
    fixed_effects — Intention model fixed-effects table
    trajectory    — 3D pitch trajectory (requires plotly + kaleido)
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
from scipy.stats import sem

Path("results").mkdir(exist_ok=True)
Path("results/figures").mkdir(exist_ok=True)

# ── Palette and constants ─────────────────────────────────────────────────────

DARK  = "#2c3e50"; MID  = "#7f8c8d"; GOLD  = "#e8b84b"
RED   = "#e74c3c"; BLUE = "#3498db"; GREEN = "#27ae60"; BG = "#f8f8f8"

MIN_SWINGS = 200
COMMIT_MS  = 150
DEV_X      = f"pc{COMMIT_MS}_dev_x"
DEV_Z      = f"pc{COMMIT_MS}_dev_z"
AXIS_COLS  = ("vert_attack_angle_dev", "horz_attack_angle_dev", "swing_path_tilt_dev")
AXIS_LBL   = {"vert_attack_angle_dev": "VAA",
              "horz_attack_angle_dev": "HAA",
              "swing_path_tilt_dev":   "Tilt"}


def _spine_clean(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MID)
    ax.spines["bottom"].set_color(MID)


# ── Lazy data loading ─────────────────────────────────────────────────────────

_cache = {}


def _load(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def xrv():  return _load("xrv",  lambda: pd.read_parquet("results/xrv_causal.parquet"))
def pc():   return _load("pc",   lambda: pd.read_parquet("data/swings_precommit.parquet"))
def sxrv(): return _load("sxrv", lambda: pd.read_parquet("results/swing_xrv.parquet"))


def merged():
    """Causal + precommit + swing_xrv joined on pitch keys (cached)."""
    if "merged" in _cache:
        return _cache["merged"]
    pc_cols = ["game_pk", "at_bat_number", "pitch_number", "game_year",
               DEV_X, DEV_Z, "ball_bat_miss", "is_whiff"]
    sx_cols = ["game_pk", "at_bat_number", "pitch_number", "pitcher_full_name",
               "pfx_x", "pfx_z", "release_speed", "release_extension", "arm_angle",
               "ivb_transverse", "hb_transverse", "pitcher_throws",
               "stuff_grade", "xrv_vs_count", "xrv"]
    K = ["game_pk", "at_bat_number", "pitch_number"]
    df = (xrv()
          .merge(pc()[pc_cols], on=K, how="left")
          .merge(sxrv()[sx_cols], on=K, how="left"))
    _cache["merged"] = df
    return df


def _eb_shrink(means, ns, within_var, between_var, grand):
    w = ns / (ns + within_var / between_var)
    return w * means + (1 - w) * grand


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_leaderboard():
    """EB-shrunk distortion tax bar chart — top/bottom 20 pitchers, 2024."""
    csv = Path("results/leaderboard.csv")
    if not csv.exists():
        df = merged()
        agg = (df[df["distortion_tax"].notna()]
               .groupby(["pitcher_id", "pitcher_full_name", "game_year"])["distortion_tax"]
               .agg(n="size", mean=np.mean, sd=np.std)
               .reset_index()
               .rename(columns={"mean": "distortion_tax_mean", "sd": "distortion_tax_sd"})
               .query(f"n >= {MIN_SWINGS}"))
        wv = agg["distortion_tax_sd"].pow(2).mean()
        bv = agg["distortion_tax_mean"].var()
        gm = agg["distortion_tax_mean"].mean()
        agg["distortion_tax_shrunk"] = _eb_shrink(agg["distortion_tax_mean"], agg["n"], wv, bv, gm)
        agg.sort_values("distortion_tax_shrunk").to_csv(csv, index=False)

    lb   = pd.read_csv(csv)
    lb24 = lb[lb["game_year"] == 2024]
    show = (pd.concat([lb24.nsmallest(20, "distortion_tax_shrunk"),
                       lb24.nlargest(20,  "distortion_tax_shrunk")])
              .drop_duplicates()
              .sort_values("distortion_tax_shrunk", ascending=True))

    fig, ax = plt.subplots(figsize=(10, 9), facecolor=BG)
    ax.set_facecolor(BG)
    colors = [GOLD if x > 0 else BLUE for x in show["distortion_tax_shrunk"]]
    ax.barh(show["pitcher_full_name"], show["distortion_tax_shrunk"], color=colors, height=0.7)
    ax.axvline(0, color=DARK, lw=0.8)
    ax.set_xlabel("Distortion Tax (xRV/swing, EB-shrunk)  |  Negative = pitcher advantage",
                  color=DARK)
    ax.set_title("Pitcher Distortion Tax — 2024 Season\n"
                 "Most Disruptive (blue) vs. Batter-Favorable (gold)  |  min 200 swings, EB shrinkage",
                 color=DARK, fontweight="bold")
    _spine_clean(ax); ax.tick_params(colors=DARK)
    fig.tight_layout()
    out = "results/figures/distortion_tax_leaderboard_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_axis_fingerprint():
    """Angular axis fingerprint: which axis dominates distortion, by pitcher handedness."""
    csv = Path("results/axis_fingerprint.csv")
    if not csv.exists():
        df = merged()
        models = joblib.load("models/causal_models.joblib")["mediator_models"]
        fp_cols = []
        for axis, m in models.items():
            col = f"_dd_{axis}"
            a_x = m.params.get(DEV_X, 0.0); a_z = m.params.get(DEV_Z, 0.0)
            df[col] = (a_x * df[DEV_X] + a_z * df[DEV_Z]).abs()
            fp_cols.append(col)
        fp = (df[df[DEV_X].notna()]
              .groupby(["pitcher_id", "pitcher_full_name", "game_year"])
              [fp_cols + ["distortion_tax"]]
              .agg(n=("distortion_tax", "size"), **{c: (c, "mean") for c in fp_cols})
              .reset_index()
              .query(f"n >= {MIN_SWINGS}"))
        total = fp[fp_cols].sum(axis=1).replace(0, np.nan)
        for col, axis in zip(fp_cols, AXIS_COLS):
            fp[f"{AXIS_LBL[axis]}_share"] = fp[col] / total
        fp["dominant_axis"] = fp[["VAA_share", "HAA_share", "Tilt_share"]].idxmax(axis=1)
        fp[["pitcher_id", "pitcher_full_name", "game_year", "n",
            "VAA_share", "HAA_share", "Tilt_share", "dominant_axis"]].to_csv(csv, index=False)
        df.drop(columns=fp_cols, inplace=True)

    af = pd.read_csv(csv)
    throws = sxrv()[["pitcher_id", "pitcher_throws"]].drop_duplicates("pitcher_id")
    af = af.merge(throws, on="pitcher_id", how="left")
    af["tilt_margin"] = af["Tilt_share"] - af["HAA_share"]

    rhp = af[af["pitcher_throws"] == "R"]
    lhp = af[af["pitcher_throws"] == "L"]
    dom = af["dominant_axis"].value_counts()
    n_tot = len(af)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG,
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor(BG)

    # Left: scatter VAA_share vs tilt margin, colored by handedness
    ax = axes[0]; ax.set_facecolor(BG)
    ax.scatter(rhp["VAA_share"], rhp["tilt_margin"],
               color=BLUE, alpha=0.35, s=18, linewidths=0, label="RHP")
    ax.scatter(lhp["VAA_share"], lhp["tilt_margin"],
               color=RED,  alpha=0.60, s=22, linewidths=0, label="LHP")
    ax.axhline(0, color=DARK, lw=1.0, ls="--")
    xlim = ax.get_xlim()
    ax.text(xlim[1], 0.0005, "← Tilt dominant", color=DARK, fontsize=8.5,
            ha="right", va="bottom")
    ax.text(xlim[1], -0.0005, "← HAA dominant", color=DARK, fontsize=8.5,
            ha="right", va="top")
    ax.set_xlabel("VAA Angular Share", color=DARK)
    ax.set_ylabel("Tilt Share − HAA Share  (positive = Tilt dominant)", color=DARK)
    ax.set_title("Axis Dominance by Pitcher Handedness\n"
                 "Each point = one pitcher-season  ·  min 200 swings",
                 color=DARK, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    _spine_clean(ax); ax.tick_params(colors=DARK)

    # Right: dominant axis percentage bar
    ax2 = axes[1]; ax2.set_facecolor(BG)
    axis_names = ["VAA", "HAA", "Tilt"]
    axis_pcts  = [dom.get("VAA_share", 0) / n_tot * 100,
                  dom.get("HAA_share",  0) / n_tot * 100,
                  dom.get("Tilt_share", 0) / n_tot * 100]
    axis_colors = [RED, BLUE, GREEN]
    bars = ax2.barh(axis_names, axis_pcts, color=axis_colors, height=0.5, edgecolor="white")
    for bar, pct in zip(bars, axis_pcts):
        if pct > 0.5:
            ax2.text(pct + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{pct:.1f}%", va="center", fontsize=10, fontweight="bold", color=DARK)
    n_lhp_haa = int((lhp["dominant_axis"] == "HAA_share").sum())
    ax2.text(1.5, 1, f"All {n_lhp_haa} are LHP",
             va="center", fontsize=8, color=MID, style="italic")
    ax2.set_xlim(0, 110)
    ax2.set_xlabel("% of Pitcher-Seasons", color=DARK)
    ax2.set_title("Dominant Axis\nDistribution", color=DARK, fontweight="bold")
    _spine_clean(ax2); ax2.tick_params(colors=DARK)

    fig.suptitle("Angular Axis Fingerprint — Pitcher Distortion Tax\n"
                 "Which axis (VAA / HAA / Tilt) drives each pitcher's swing disruption",
                 fontsize=12, fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    out = "results/figures/axis_fingerprint.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_drivers():
    """OLS forest plot of physical drivers of distortion tax."""
    df = merged()
    df = df.copy()
    df["pfx_arm"]   = np.where(df["pitcher_throws"] == "R",  df["pfx_x"], -df["pfx_x"])
    df["pfx_glove"] = np.where(df["pitcher_throws"] == "R", -df["pfx_x"],  df["pfx_x"])

    pt_agg = (df[df["distortion_tax"].notna() & df["pfx_x"].notna()]
              .groupby(["pitcher_id", "pitch_type"])
              .agg(n=("distortion_tax",    "size"),
                   distortion=("distortion_tax", "mean"),
                   pfx_arm=("pfx_arm",     "mean"),
                   pfx_z=("pfx_z",         "mean"),
                   release_speed=("release_speed",     "mean"),
                   release_extension=("release_extension", "mean"),
                   arm_angle=("arm_angle", "mean"),
                   ivb_transverse=("ivb_transverse", "mean"),
                   hb_transverse=("hb_transverse",   "mean"))
              .reset_index()
              .query(f"n >= {MIN_SWINGS // 2}"))

    formula = ("distortion ~ pfx_arm + pfx_z + release_speed + "
                "release_extension + arm_angle + ivb_transverse + hb_transverse")
    model = smf.ols(formula, pt_agg.dropna()).fit(cov_type="HC1")

    driver_out = model.summary2().tables[1].reset_index()
    driver_out.columns = ["feature", "coef", "std_err", "t", "p_value", "ci_lower", "ci_upper"]
    csv = Path("results/physical_drivers.csv")
    if not csv.exists():
        driver_out.to_csv(csv, index=False)

    plot_d = driver_out[driver_out["feature"] != "Intercept"].sort_values("coef")
    sig    = plot_d["p_value"] < 0.05

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.barh(plot_d["feature"], plot_d["coef"],
            xerr=[plot_d["coef"] - plot_d["ci_lower"],
                  plot_d["ci_upper"] - plot_d["coef"]],
            color=[RED if s else MID for s in sig],
            height=0.5, capsize=3, error_kw={"ecolor": DARK, "lw": 1})
    ax.axvline(0, color=DARK, lw=0.8, ls="--")
    ax.set_xlabel("Coefficient (xRV/swing per unit)", color=DARK)
    ax.set_title(f"Physical Drivers of Distortion Tax\n"
                 f"OLS at pitcher-pitch-type level  R²={model.rsquared:.3f}  "
                 f"(red = p<0.05, HC1 SE)", color=DARK, fontweight="bold")
    _spine_clean(ax); ax.tick_params(colors=DARK)
    fig.tight_layout()
    out = "results/figures/physical_drivers_ols.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_incremental():
    """Scatter: current-season distortion tax → next-season ball-bat miss."""
    df = merged()
    MIN_WHIFFS = 50

    dist_agg = (df[df["distortion_tax"].notna()]
                .groupby(["pitcher_id", "pitcher_full_name", "game_year"])
                .agg(n=("distortion_tax", "size"),
                     distortion_tax=("distortion_tax", "mean"),
                     xrv=("xrv", "mean"))
                .reset_index()
                .query(f"n >= {MIN_SWINGS}"))

    bbm_agg = (df[(df["is_whiff"] == 1) & df["ball_bat_miss"].notna()]
               .groupby(["pitcher_id", "game_year"])
               .agg(n_whiffs=("ball_bat_miss", "size"), ball_bat_miss=("ball_bat_miss", "mean"))
               .reset_index()
               .query(f"n_whiffs >= {MIN_WHIFFS}"))

    season_agg = dist_agg.merge(bbm_agg, on=["pitcher_id", "game_year"], how="left")

    validity_rows = []
    for yr_a, yr_b in [(2023, 2024), (2024, 2025)]:
        ya  = season_agg[season_agg["game_year"] == yr_a]
        yb  = (season_agg[season_agg["game_year"] == yr_b][["pitcher_id", "ball_bat_miss"]]
               .rename(columns={"ball_bat_miss": "next_bbm"}))
        pair = (ya.merge(yb, on="pitcher_id", how="inner")
                  .dropna(subset=["ball_bat_miss", "next_bbm", "distortion_tax"]))
        if len(pair) < 20:
            continue
        m0 = smf.ols("next_bbm ~ ball_bat_miss", pair).fit()
        m1 = smf.ols("next_bbm ~ ball_bat_miss + distortion_tax", pair).fit()
        validity_rows.append({
            "transition": f"{yr_a}->{yr_b}", "n": len(pair),
            "baseline_r2": m0.rsquared, "full_r2": m1.rsquared,
            "delta_r2": m1.rsquared - m0.rsquared,
            "coef":   m1.params.get("distortion_tax", np.nan),
            "pvalue": m1.pvalues.get("distortion_tax", np.nan),
        })

    csv = Path("results/incremental_validity.csv")
    if not csv.exists() and validity_rows:
        pd.DataFrame(validity_rows).to_csv(csv, index=False)

    if not validity_rows:
        print("fig_incremental: insufficient data, skipping.")
        return

    best = max(validity_rows, key=lambda r: r["n"])
    yr_a, yr_b = [int(x) for x in best["transition"].split("->")]
    ya   = season_agg[season_agg["game_year"] == yr_a]
    yb   = (season_agg[season_agg["game_year"] == yr_b][["pitcher_id", "ball_bat_miss"]]
            .rename(columns={"ball_bat_miss": "next_bbm"}))
    pair = (ya.merge(yb, on="pitcher_id", how="inner")
              .dropna(subset=["distortion_tax", "ball_bat_miss", "next_bbm"]))

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.scatter(pair["distortion_tax"], pair["next_bbm"],
               alpha=0.55, color=BLUE, edgecolors="white", lw=0.4, s=45)
    m_plot = smf.ols("next_bbm ~ distortion_tax", pair).fit()
    xs = np.linspace(pair["distortion_tax"].min(), pair["distortion_tax"].max(), 200)
    ax.plot(xs, m_plot.params["Intercept"] + m_plot.params["distortion_tax"] * xs,
            color=RED, lw=2)
    ax.set_xlabel(f"{yr_a} Distortion Tax (xRV/swing)", color=DARK)
    ax.set_ylabel(f"{yr_b} Mean Ball-Bat Miss (inches, whiffs only)", color=DARK)
    ax.set_title(
        f"Incremental Validity: {yr_a} Distortion Tax → {yr_b} Ball-Bat Miss\n"
        f"n={best['n']}  β={best['coef']:.3f}  p={best['pvalue']:.3f}  "
        f"ΔR²={best['delta_r2']:.3f}",
        color=DARK, fontweight="bold")
    _spine_clean(ax); ax.tick_params(colors=DARK)
    fig.tight_layout()
    out = "results/figures/distortion_tax_incremental_validity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_outcomes():
    """Stacked bar of swing outcome rates by distortion/selection tax quintile."""
    sw = (pc()[["game_pk", "at_bat_number", "pitch_number",
                "is_whiff", "is_contact", "is_bip",
                "is_single", "is_double", "is_triple", "is_home_run"]]
          .drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"]))
    df = (xrv()
          .merge(sw, on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
          .dropna(subset=["distortion_tax", "selection_tax"]))

    df["is_foul"]        = (df["is_contact"] == 1) & (df["is_bip"] == 0)
    df["is_out_in_play"] = ((df["is_bip"] == 1) & (df["is_home_run"] == 0) &
                            (df["is_triple"] == 0) & (df["is_double"] == 0) &
                            (df["is_single"] == 0))
    df["is_xbh"]  = ((df["is_bip"] == 1) & ((df["is_double"] == 1) |
                     (df["is_triple"] == 1) | (df["is_home_run"] == 1)))

    OUTCOMES = [("Whiff",       "is_whiff",       "#c0392b"),
                ("Foul",        "is_foul",        "#e67e22"),
                ("Out in Play", "is_out_in_play", "#95a5a6"),
                ("Single",      "is_single",      "#27ae60"),
                ("XBH / HR",    "is_xbh",         "#2980b9")]
    N_BINS = 5
    BIN_LABS = ["Q1\n(most\ndisruptive)", "Q2", "Q3", "Q4", "Q5\n(least\ndisruptive)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    for ax, metric, xlabel, title in [
        (axes[0], "distortion_tax", "Distortion Tax Quintile",
         "Outcome Rates by Distortion Tax Level"),
        (axes[1], "selection_tax",  "Selection Tax Quintile",
         "Outcome Rates by Selection Tax Level"),
    ]:
        df["_bin"] = pd.qcut(df[metric], q=N_BINS, labels=False, duplicates="drop")
        bottoms = np.zeros(N_BINS)
        for lbl, col, color in OUTCOMES:
            rates = df.groupby("_bin")[col].mean().reindex(range(N_BINS)).fillna(0).values
            ax.bar(range(N_BINS), rates, 0.65, bottom=bottoms,
                   color=color, label=lbl, edgecolor="white", linewidth=0.4)
            for i, (r, b) in enumerate(zip(rates, bottoms)):
                if r > 0.04:
                    ax.text(i, b + r / 2, f"{r:.1%}", ha="center", va="center",
                            fontsize=7.5, color="white", fontweight="bold")
            bottoms += rates
        for i in range(N_BINS):
            n = (df["_bin"] == i).sum()
            ax.text(i, bottoms[i] + 0.005, f"n={n:,}", ha="center", va="bottom",
                    fontsize=7, color="#555")
        ax.set_xticks(range(N_BINS)); ax.set_xticklabels(BIN_LABS, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Share of Swings", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    handles = [mpatches.Patch(color=c, label=l) for l, _, c in OUTCOMES]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Swing Outcome Rates by Distortion / Selection Tax Level  ·  2023–2025",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = "results/figures/outcome_rates.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def fig_xwoba():
    """xwOBA (all swings, whiff/foul = 0) vs distortion/selection tax."""
    sw = (pc()[["game_pk", "at_bat_number", "pitch_number",
                "is_contact", "is_bip", "is_single", "is_double",
                "is_triple", "is_home_run"]]
          .drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"]))
    df = (xrv()
          .merge(sw, on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
          .dropna(subset=["distortion_tax", "selection_tax"]))

    df["is_out_in_play"] = ((df["is_bip"] == 1) & (df["is_home_run"] == 0) &
                            (df["is_triple"] == 0) & (df["is_double"] == 0) &
                            (df["is_single"] == 0))
    lw = pd.read_csv("results/linear_weights.csv").set_index("outcome_type")["lw"]
    bip = df["is_bip"] == 1
    xw  = pd.Series(0.0, index=df.index)
    xw.loc[bip & (df["is_home_run"] == 1)] = lw["home_run"]
    xw.loc[bip & (df["is_triple"] == 1) & ~(df["is_home_run"] == 1)] = lw["triple"]
    xw.loc[bip & (df["is_double"] == 1) & ~(df["is_triple"] == 1) & ~(df["is_home_run"] == 1)] = lw["double"]
    xw.loc[bip & (df["is_single"] == 1) & ~(df["is_double"] == 1) & ~(df["is_triple"] == 1) & ~(df["is_home_run"] == 1)] = lw["single"]
    xw.loc[df["is_out_in_play"]] = lw["out_in_play"]
    df["xwoba_all"] = xw

    fig, axes = plt.subplots(1, 2, figsize=(13, 5)); fig.patch.set_facecolor("white")
    for ax, metric, label, color in [
        (axes[0], "distortion_tax", "Distortion Tax", "#d73027"),
        (axes[1], "selection_tax",  "Selection Tax",  "#2166ac"),
    ]:
        df["_bin"] = pd.qcut(df[metric], q=10, labels=False, duplicates="drop")
        s = (df.groupby("_bin")
             .agg(mid=(metric, "mean"),
                  mean_xw=("xwoba_all", "mean"),
                  se_xw=("xwoba_all", lambda x: sem(x, nan_policy="omit")),
                  contact=("is_contact", "mean"),
                  n=("xwoba_all", "count"))
             .reset_index())
        ax.fill_between(s["mid"], s["mean_xw"] - 1.96*s["se_xw"],
                        s["mean_xw"] + 1.96*s["se_xw"], alpha=0.15, color=color)
        ax.plot(s["mid"], s["mean_xw"], "o-", color=color, lw=2, ms=6,
                mfc="white", mew=2, label="xwOBA (all swings)")
        ax2r = ax.twinx()
        ax2r.plot(s["mid"], s["contact"], "s--", color=color, lw=1.2, ms=4,
                  alpha=0.5, label="Contact rate")
        ax2r.set_ylabel("Contact rate", fontsize=9, color=color, alpha=0.7)
        ax2r.tick_params(axis="y", labelcolor=color, labelsize=8)
        ax2r.set_ylim(0, 1)
        ax2r.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        for _, r in s.iterrows():
            ax.text(r["mid"], r["mean_xw"] + 0.003, f"n={int(r['n']):,}",
                    ha="center", va="bottom", fontsize=6.5, color="#666")
        ax.set_xlabel(f"{label}  (most negative = most disrupted)", fontsize=10)
        ax.set_ylabel("Mean xwOBA (all swings, whiff/foul = 0)", fontsize=10)
        ax.set_title(f"Swing Quality vs {label}", fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.25, linestyle="--")
        l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2r.get_legend_handles_labels()
        ax.legend(l1+l2, lb1+lb2, fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Swing Quality vs Distortion / Selection Tax  ·  All Swings  ·  2023–2025\n"
                 "(xwOBA = 0 for whiffs/fouls — no BIP conditioning)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = "results/figures/xwoba_relationship.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def fig_reliability():
    """Split-half (Spearman-Brown) and YoY reliability table."""
    METRICS = ["disruption_tax", "adjusted_disruption_tax", "distortion_tax", "selection_tax"]
    METRIC_LABELS = {
        "disruption_tax":          "xRV Residual (Actual - Intended)",
        "adjusted_disruption_tax": "Adjusted Disruption Tax",
        "distortion_tax":          "Distortion Tax",
        "selection_tax":           "Selection Tax",
    }
    N_SPLITS = 100; SEED = 42; MIN_SH = 50

    sw = (pc()[["game_pk", "at_bat_number", "pitch_number", "game_year"]]
          .drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"]))
    df = (xrv()
          .merge(sw, on=["game_pk", "at_bat_number", "pitch_number"], how="left")
          .dropna(subset=METRICS + ["game_year", "pitcher_id"]))

    def _icc21(y):
        y = np.asarray(y, float); n, k = y.shape
        if n < 3 or k < 2: return np.nan
        gm = y.mean(); rm = y.mean(1); cm = y.mean(0)
        ssb = k * ((rm-gm)**2).sum(); ssr = n * ((cm-gm)**2).sum()
        sse = ((y-gm)**2).sum() - ssb - ssr
        msb = ssb/(n-1); msr = ssr/(k-1); mse = sse/((n-1)*(k-1))
        d = msb + (k-1)*mse + k/n*(msr-mse)
        return np.nan if d <= 0 else float((msb-mse)/d)

    def split_half(metric):
        rng   = np.random.default_rng(SEED)
        valid = df.groupby("pitcher_id")[metric].transform("size") >= MIN_SH * 2
        pool  = df.loc[valid, ["pitcher_id", metric]].copy()
        pids  = pool["pitcher_id"].unique()
        if len(pids) < 10: return np.nan, np.nan
        rs, iccs = [], []
        for _ in range(N_SPLITS):
            h1, h2 = [], []
            for pid in pids:
                g = pool.loc[pool["pitcher_id"] == pid, metric].values
                idx = rng.permutation(len(g)); half = len(g) // 2
                h1.append(g[idx[:half]].mean()); h2.append(g[idx[half:]].mean())
            r = np.corrcoef(h1, h2)[0, 1]
            rs.append(2*r/(1+r))
            iccs.append(_icc21(np.column_stack([h1, h2])))
        return float(np.mean(rs)), float(np.nanmean(iccs))

    def yoy(yr_a, yr_b, metric):
        def means(yr):
            g = df[df["game_year"] == yr].groupby("pitcher_id")[metric]
            a = g.agg(mean="mean", n="size"); return a[a["n"] >= MIN_SH]["mean"]
        a = means(yr_a); b = means(yr_b); both = a.index.intersection(b.index)
        if len(both) < 10: return np.nan, np.nan
        return (float(a.loc[both].corr(b.loc[both])),
                float(_icc21(np.column_stack([a.loc[both].values, b.loc[both].values]))))

    rows = []
    for m in METRICS:
        sh_r, sh_icc     = split_half(m)
        r2324, icc2324   = yoy(2023, 2024, m)
        r2425, icc2425   = yoy(2024, 2025, m)
        rows.append({"metric": m, "sh_r": sh_r, "sh_icc": sh_icc,
                     "yoy_23_24_r": r2324, "yoy_23_24_icc": icc2324,
                     "yoy_24_25_r": r2425, "yoy_24_25_icc": icc2425})

    tbl = pd.DataFrame(rows).set_index("metric")
    tbl.to_csv("results/figures/08_reliability.csv")

    def _fmt(r): return "—" if np.isnan(r) else f"{r:.3f}"
    def _col(r):
        if np.isnan(r): return "#f0f0f0"
        if r >= 0.7:    return "#c8e6c9"
        if r >= 0.5:    return "#fff9c4"
        if r >= 0.3:    return "#ffe0b2"
        return "#ffcdd2"

    COL_HDR = ["Split-half r\n(SB corrected, 100 splits)", "Split-half\nICC(2,1)",
               "YoY 2023→2024\nPearson r",                 "YoY 2023→2024\nICC(2,1)",
               "YoY 2024→2025\nPearson r",                 "YoY 2024→2025\nICC(2,1)"]
    cell_text, cell_color = [], []
    for _, row in tbl.iterrows():
        vals = [row["sh_r"], row["sh_icc"], row["yoy_23_24_r"],
                row["yoy_23_24_icc"], row["yoy_24_25_r"], row["yoy_24_25_icc"]]
        cell_text.append([_fmt(v) for v in vals])
        cell_color.append([_col(v) for v in vals])

    fig, ax = plt.subplots(figsize=(16, 3.2)); ax.axis("off")
    fig.suptitle(f"Pitcher Distortion / Disruption Tax — Reliability Analysis\n"
                 f"(min {MIN_SH} swings per pitcher per half/season)",
                 fontsize=11, fontweight="bold", y=1.03)
    t = ax.table(cellText=cell_text, cellColours=cell_color,
                 rowLabels=[METRIC_LABELS[m] for m in tbl.index],
                 colLabels=COL_HDR, cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 2.6)
    for j in range(len(COL_HDR)):
        t[(0, j)].set_facecolor("#2c3e50"); t[(0, j)].get_text().set_color("white")
        t[(0, j)].get_text().set_fontweight("bold")
    fig.tight_layout()
    out = "results/figures/08_reliability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_count_effects():
    """Intended swing shape by count — VAA and bat speed only."""
    LABELS = {"vert_attack_angle": "Vert. Attack Angle (°)", "bat_speed": "Bat Speed (mph)"}
    COUNT_ORDER  = ["Hitter", "Early", "Full", "Pitcher"]
    COUNT_COLORS = {"Hitter": "#2ca02c", "Early": "#4878d0",
                    "Full": "#ff7f0e",   "Pitcher": "#d62728"}

    sw   = pc()
    intd = pd.read_parquet("models/intended_df.parquet")
    df   = sw.join(intd)
    df   = df[(df["is_swing"] == 1) & (df["bat_speed"] >= 50) &
              df["intended_vert_attack_angle"].notna()].copy()

    metrics = ["vert_attack_angle", "bat_speed"]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(14, 4.2 * len(metrics)))
    fig.suptitle("Intended Swing Shape by Count", fontsize=12, fontweight="bold")

    df_cg = df.dropna(subset=["count_group"])
    df_cm = df[(df["balls"].between(0, 3)) & (df["strikes"].between(0, 2))].copy()

    for r, resp in enumerate(metrics):
        lbl  = LABELS[resp]; icol = f"intended_{resp}"

        ax = axes[r, 0]
        grp   = df_cg.groupby("count_group")[icol]
        means = grp.mean().reindex(COUNT_ORDER); sds = grp.std().reindex(COUNT_ORDER)
        bars  = ax.bar(COUNT_ORDER, means.values,
                       color=[COUNT_COLORS[g] for g in COUNT_ORDER],
                       alpha=0.85, edgecolor="white", width=0.6)
        ax.errorbar(COUNT_ORDER, means.values, yerr=sds.values,
                    fmt="none", color="black", capsize=4, lw=1.2)
        for bar, m, sd in zip(bars, means.values, sds.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sd*0.02,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel(f"Mean {lbl}", fontsize=9)
        ax.set_title(f"{lbl} — Count Group", fontsize=10)
        ns = df_cg.groupby("count_group").size().reindex(COUNT_ORDER)
        for i, n in enumerate(ns):
            ax.text(i, ax.get_ylim()[0], f"n={n:,}", ha="center", va="top",
                    fontsize=7, color="grey")

        ax = axes[r, 1]
        grand = df_cm[icol].mean()
        pivot = (df_cm.groupby(["balls", "strikes"])[icol].mean().subtract(grand)
                 .unstack("strikes").reindex(index=[0,1,2,3], columns=[0,1,2]))
        raw   = (df_cm.groupby(["balls", "strikes"])[icol].mean()
                 .unstack("strikes").reindex(index=[0,1,2,3], columns=[0,1,2]))
        vabs  = np.nanmax(np.abs(pivot.values))
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.85, label=f"Δ vs. grand mean ({lbl})")
        for bi in range(4):
            for si in range(3):
                val = raw.iloc[bi, si]
                if not np.isnan(val):
                    tc = "white" if abs(pivot.iloc[bi, si]) > vabs*0.4 else "black"
                    ax.text(si, bi, f"{val:.1f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=tc)
        ax.set_xticks([0,1,2]); ax.set_xticklabels(["0 str", "1 str", "2 str"], fontsize=9)
        ax.set_yticks([0,1,2,3]); ax.set_yticklabels(["0 b","1 b","2 b","3 b"], fontsize=9)
        ax.set_title(f"{lbl} — Δ from mean by Count", fontsize=10)

    fig.tight_layout()
    out = "results/figures/count_effects.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_fixed_effects():
    """Intention model fixed-effects table (academic format)."""
    RESPONSES = ["vert_attack_angle", "horz_attack_angle", "swing_path_tilt",
                 "bat_speed", "swing_length"]
    ANGULAR    = {"vert_attack_angle", "horz_attack_angle", "swing_path_tilt"}
    PARAM_ORDER = ["Intercept", "scale(balls)", "scale(strikes)",
                   "scale(plate_x_bat)", "scale(plate_z)", "scale(plate_z_sq)",
                   "scale(offset_y_ms)", "pitcher_throws_L",
                   "pitcher_throws_L:scale(plate_x_bat)"]
    PARAM_GREEK = {
        "Intercept":                            r"$\mu_0$",
        "scale(balls)":                         r"$\beta^B$",
        "scale(strikes)":                       r"$\beta^S$",
        "scale(plate_x_bat)":                   r"$\beta^X$",
        "scale(plate_z)":                       r"$\beta^Z$",
        "scale(plate_z_sq)":                    r"$\beta^{Z^2}$",
        "scale(offset_y_ms)":                   r"$\beta^T$",
        "pitcher_throws_L":                     r"$\beta^L$",
        "pitcher_throws_L:scale(plate_x_bat)":  r"$\beta^{LX}$",
    }
    COL_GROUP = {"vert_attack_angle": "Vert. Attack Angle",
                 "horz_attack_angle": "Horz. Attack Angle",
                 "swing_path_tilt":   "Swing Path Tilt",
                 "bat_speed":         "Bat Speed",
                 "swing_length":      "Swing Length"}
    RE_SFX    = ("_sigma", "_offset"); SKIP = {"sigma", "mu"}
    ANG_ONLY  = {"scale(plate_z_sq)", "scale(offset_y_ms)",
                 "pitcher_throws_L:scale(plate_x_bat)"}

    result = joblib.load("models/intention_result.joblib")

    def _fe(idata):
        post = idata.posterior; out = {}
        for v in post.data_vars:
            if (set(post[v].dims) == {"chain", "draw"} and v not in SKIP
                    and not any(v.endswith(s) for s in RE_SFX)):
                draws = post[v].values.ravel()
                out[v] = {"mean": float(draws.mean()),
                          "lo":   float(np.percentile(draws, 2.5)),
                          "hi":   float(np.percentile(draws, 97.5))}
        return out

    coefs = {r: _fe(result["idata"][r]) for r in RESPONSES}

    LW=1.5; SW=0.82; RH=0.30; HH=0.30
    ML=0.08; MR=0.08; MT=0.12; MB=0.12
    NR=len(RESPONSES); NP=len(PARAM_ORDER)
    fw = ML + LW + NR*3*SW + MR; fh = MT + 2*HH + NP*RH + MB
    fig = plt.figure(figsize=(fw, fh)); ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0, fw); ax.set_ylim(0, fh); ax.axis("off")

    x0=ML; x1=fw-MR; yt=fh-MT; yg=yt-HH; ys=yg-HH; yb=MB
    gx = lambda j: x0+LW+j*3*SW
    cx = lambda j,k: gx(j)+(k+0.5)*SW

    for y, lw in [(yt,0.9),(yg,0.4),(ys,0.9),(yb,0.9)]:
        ax.plot([x0,x1],[y,y],"k-",lw=lw)
    for xv in [x0+LW]+[gx(j) for j in range(1,NR)]+[x1]:
        ax.plot([xv,xv],[yb,yt],"k-",lw=0.5)

    for j,resp in enumerate(RESPONSES):
        ax.text(gx(j)+1.5*SW,(yt+yg)/2,COL_GROUP[resp],
                ha="center",va="center",fontsize=8.5,fontfamily="serif")
    ax.text(x0+LW/2,(yg+ys)/2,"Parameter",ha="center",va="center",
            fontsize=8,fontfamily="serif")
    for j in range(NR):
        for k,lbl in enumerate(["Mean","Lower","Upper"]):
            ax.text(cx(j,k),(yg+ys)/2,lbl,ha="center",va="center",
                    fontsize=8,fontfamily="serif")
    for i,param in enumerate(PARAM_ORDER):
        cy = ys-(i+0.5)*RH
        ax.text(x0+LW/2,cy,PARAM_GREEK[param],ha="center",va="center",
                fontsize=9,fontfamily="serif")
        for j,resp in enumerate(RESPONSES):
            if param in ANG_ONLY and resp not in ANGULAR:
                for k in range(3):
                    ax.text(cx(j,k),cy,"—",ha="center",va="center",
                            fontsize=8.5,color="#999",fontfamily="serif")
            else:
                s = coefs[resp].get(param)
                if s:
                    for k,val in enumerate([s["mean"],s["lo"],s["hi"]]):
                        ax.text(cx(j,k),cy,f"{val:.3f}",ha="center",va="center",
                                fontsize=8.5,fontfamily="serif")
    out = "results/figures/fixed_effects.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def fig_trajectory():
    """3D catcher's-eye pitch trajectory for Shohei Ohtani (requires plotly + kaleido)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("fig_trajectory: plotly not installed, skipping.")
        return

    PITCH_COLORS = {"FF":"#d62728","SI":"#ff7f0e","FC":"#e377c2","CH":"#2ca02c",
                    "FS":"#17becf","SL":"#bcbd22","CU":"#1f77b4","ST":"#9467bd","KC":"#1f77b4"}
    PITCH_NAMES  = {"FF":"Four-seam fastball","SI":"Sinker","FC":"Cutter","CH":"Changeup",
                    "FS":"Splitter","SL":"Slider","CU":"Curveball","ST":"Sweeper","KC":"Knuckle-curve"}
    PITCHER_NAME = "Shohei Ohtani"
    COMMIT_S = COMMIT_MS / 1000.0
    PFX = f"pc{COMMIT_MS}_"
    TRAJ = [f"{PFX}{c}" for c in ["R_x","R_y","R_z","V_x","V_y","V_z","A_x","A_y","A_z","t_plate"]]

    df  = pc()
    sub = df[(df["pitcher_full_name"] == PITCHER_NAME) &
             df[TRAJ].notna().all(axis=1) &
             df["pitch_type"].isin(PITCH_COLORS)].copy()
    means = sub.groupby("pitch_type")[TRAJ].mean()

    def _snap(Rx,Ry,Rz,Vx,Vy,Vz,Ax,Ay,Az,t_max):
        ts = np.arange(0, t_max+0.005/2, 0.005)
        return (Rx+Vx*ts+0.5*Ax*ts**2, Ry+Vy*ts+0.5*Ay*ts**2,
                Rz+Vz*ts+0.5*Az*ts**2, ts)

    fig = go.Figure()
    commit_ys = []
    for pt in means.index:
        m = means.loc[pt]
        Rx,Ry,Rz = m[f"{PFX}R_x"],m[f"{PFX}R_y"],m[f"{PFX}R_z"]
        Vx,Vy,Vz = m[f"{PFX}V_x"],m[f"{PFX}V_y"],m[f"{PFX}V_z"]
        Ax,Ay,Az = m[f"{PFX}A_x"],m[f"{PFX}A_y"],m[f"{PFX}A_z"]
        t_plate  = m[f"{PFX}t_plate"]; t_c = t_plate - COMMIT_S
        x,y,z,ts = _snap(Rx,Ry,Rz,Vx,Vy,Vz,Ax,Ay,Az,t_plate)
        ci = int(np.searchsorted(ts, t_c)); color = PITCH_COLORS[pt]
        fig.add_trace(go.Scatter3d(x=x[:ci+1],y=y[:ci+1],z=z[:ci+1],mode="lines",
                                   line=dict(color=color,width=5),opacity=0.5,
                                   name=PITCH_NAMES.get(pt,pt),legendgroup=pt,showlegend=True))
        fig.add_trace(go.Scatter3d(x=x[ci:],y=y[ci:],z=z[ci:],mode="lines",
                                   line=dict(color=color,width=9),legendgroup=pt,showlegend=False))
        fig.add_trace(go.Scatter3d(x=[float(x[-1])],y=[float(y[-1])],z=[float(z[-1])],
                                   mode="markers",marker=dict(size=11,color=color),
                                   legendgroup=pt,showlegend=False))
        xc=Rx+Vx*t_c+0.5*Ax*t_c**2; yc=Ry+Vy*t_c+0.5*Ay*t_c**2; zc=Rz+Vz*t_c+0.5*Az*t_c**2
        fig.add_trace(go.Scatter3d(x=[float(xc)],y=[float(yc)],z=[float(zc)],mode="markers",
                                   marker=dict(size=14,color=color,line=dict(color="white",width=3)),
                                   legendgroup=pt,showlegend=False))
        commit_ys.append(float(yc))

    mcy = float(np.mean(commit_ys))
    fig.add_trace(go.Scatter3d(x=[-1.,1.],y=[mcy,mcy],z=[2.5,2.5],mode="lines",
                               line=dict(color="rgba(255,255,255,0.55)",width=2,dash="dot"),
                               showlegend=False))
    fig.add_trace(go.Scatter3d(x=[1.1],y=[mcy],z=[3.0],mode="text",
                               text=[f"  Decision point<br>  ({COMMIT_MS} ms)"],
                               textfont=dict(size=11,color="white"),showlegend=False))

    # Field geometry
    _xg,_yg = np.meshgrid(np.linspace(-3,3,30), np.linspace(1.4,62,30))
    fig.add_trace(go.Surface(x=_xg,y=_yg,z=np.zeros_like(_xg),
                             colorscale=[[0,"#4a7a34"],[1,"#4a7a34"]],
                             opacity=1,showscale=False,showlegend=False,hoverinfo="skip"))
    _th,_r = np.meshgrid(np.linspace(0,2*np.pi,50), np.linspace(0,3,50))
    fig.add_trace(go.Surface(x=_r*np.cos(_th),y=_r*np.sin(_th)+2,
                             z=np.full((50,50),0.005),
                             colorscale=[[0,"#c4a47c"],[1,"#c4a47c"]],
                             opacity=1,showscale=False,showlegend=False,hoverinfo="skip"))
    _hp_x=[-8.5/12,8.5/12,10/12,0,-10/12,-8.5/12]
    _hp_y=[2,2,1,0,1,2]; _hp_z=[0.01]*6
    fig.add_trace(go.Mesh3d(x=_hp_x,y=_hp_y,z=_hp_z,
                            i=[0,1,2,2,3,4],j=[1,2,3,3,4,5],k=[5,5,5,4,5,0],
                            color="white",opacity=1,flatshading=True,
                            showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[-10/12,10/12,10/12,-10/12,-10/12],y=[2,2,2,2,2],
                               z=[1.5,1.5,3.5,3.5,1.5],mode="lines",
                               line=dict(color="#dddddd",width=5),showlegend=False))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-3,3],showticklabels=False,showgrid=False,zeroline=False,
                       title="",backgroundcolor="rgb(55,55,55)"),
            yaxis=dict(range=[0,62],showticklabels=False,showgrid=False,zeroline=False,
                       title="",backgroundcolor="rgb(55,55,55)"),
            zaxis=dict(range=[0,6],showticklabels=False,showgrid=False,zeroline=False,
                       title="",backgroundcolor="rgb(55,55,55)"),
            aspectmode="manual",aspectratio=dict(x=1,y=10,z=1),bgcolor="rgb(55,55,55)"),
        scene_camera=dict(eye=dict(x=0.7,y=-5.75,z=0.1),
                          center=dict(x=0,y=5,z=-0.25),up=dict(x=0,y=0,z=1)),
        paper_bgcolor="rgb(45,45,45)",width=1400,height=800,
        margin=dict(l=0,r=0,t=55,b=0),
        title=dict(text=f"{PITCHER_NAME} — Mean Pitch Trajectories by Type",
                   font=dict(size=17,color="white"),x=0.5,xanchor="center"),
        legend=dict(font=dict(size=13,color="white"),bgcolor="rgba(255,255,255,0.12)",
                    bordercolor="rgba(255,255,255,0.3)",borderwidth=1,
                    x=0.01,y=0.97,itemsizing="constant"))
    out = "results/figures/batter_view_3d.png"
    try:
        fig.write_image(str(out), scale=2)
        print(f"Saved {out}")
    except Exception as e:
        print(f"fig_trajectory: write_image failed ({e}). Is kaleido installed?")


# ── Registry and CLI dispatch ─────────────────────────────────────────────────

FIGURES = {
    "leaderboard":   fig_leaderboard,
    "axis":          fig_axis_fingerprint,
    "drivers":       fig_drivers,
    "incremental":   fig_incremental,
    "outcomes":      fig_outcomes,
    "xwoba":         fig_xwoba,
    "reliability":   fig_reliability,
    "count_effects": fig_count_effects,
    "fixed_effects": fig_fixed_effects,
    "trajectory":    fig_trajectory,
}

if __name__ == "__main__":
    keys = sys.argv[1:] if sys.argv[1:] else list(FIGURES)
    invalid = [k for k in keys if k not in FIGURES]
    if invalid:
        print(f"Unknown figure keys: {invalid}")
        print(f"Valid keys: {list(FIGURES)}")
        sys.exit(1)
    for key in keys:
        print(f"\n=== {key} ===")
        FIGURES[key]()
    print("\nAll done.")
