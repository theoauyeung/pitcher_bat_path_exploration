"""
outcome_analysis.py
How swing outcomes shift as distortion / selection tax increases.

The disruption tax = xRV(realized) - xRV(intended). To interpret correctly,
we bin swings BY tax level and ask: what outcomes does high distortion produce?

Run from project root:
    .venv\\Scripts\\python.exe results_scripts\\outcome_analysis.py

Outputs:
    results/figures/outcome_rates.png      -- outcome rates by tax quintile
    results/figures/xwoba_relationship.png -- xwOBA on contact vs tax metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import sem

Path("results/figures").mkdir(parents=True, exist_ok=True)

# ── Load & merge ──────────────────────────────────────────────────────────────

xrv = pd.read_parquet("results/xrv_causal.parquet")

sw = pd.read_parquet("data/swings_precommit.parquet", columns=[
    "game_pk", "at_bat_number", "pitch_number",
    "is_whiff", "is_contact", "is_bip",
    "is_single", "is_double", "is_triple", "is_home_run",
    "balls", "strikes",
]).drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])

df = (
    xrv.merge(sw, on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
    .dropna(subset=["distortion_tax", "selection_tax", "disruption_tax"])
)

df["is_foul"]       = (df["is_contact"] == 1) & (df["is_bip"] == 0)
df["is_out_in_play"] = (
    (df["is_bip"] == 1)
    & (df["is_home_run"] == 0) & (df["is_triple"] == 0)
    & (df["is_double"] == 0)  & (df["is_single"] == 0)
)
df["is_xbh"] = (
    (df["is_bip"] == 1)
    & ((df["is_double"] == 1) | (df["is_triple"] == 1) | (df["is_home_run"] == 1))
)

# xwOBA for BIP
lw = pd.read_csv("results/linear_weights.csv").set_index("outcome_type")["lw"]
bip = df["is_bip"] == 1
xwoba = pd.Series(np.nan, index=df.index)
xwoba.loc[bip & (df["is_home_run"] == 1)]                                              = lw["home_run"]
xwoba.loc[bip & (df["is_triple"]   == 1) & ~(df["is_home_run"] == 1)]                 = lw["triple"]
xwoba.loc[bip & (df["is_double"]   == 1) & ~(df["is_triple"] == 1)
                                         & ~(df["is_home_run"] == 1)]                  = lw["double"]
xwoba.loc[bip & (df["is_single"]   == 1) & ~(df["is_double"] == 1)
                                         & ~(df["is_triple"] == 1)
                                         & ~(df["is_home_run"] == 1)]                  = lw["single"]
xwoba.loc[df["is_out_in_play"]]                                                         = lw["out_in_play"]
df["xwoba"] = xwoba

# ── Figure 1: outcome rates by tax quintile ───────────────────────────────────
#
# The right question: given a distortion/selection level, what outcomes follow?
# Bin swings by tax quintile; show whiff / foul / out-in-play / single / XBH rates.

OUTCOMES_STACKED = [
    ("Whiff",       "is_whiff",       "#c0392b"),
    ("Foul",        "is_foul",        "#e67e22"),
    ("Out in Play", "is_out_in_play", "#95a5a6"),
    ("Single",      "is_single",      "#27ae60"),
    ("XBH / HR",    "is_xbh",         "#2980b9"),
]

N_BINS   = 5
BIN_LABS = ["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"]

fig1, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
fig1.patch.set_facecolor("white")

for ax, metric, col_label, title in [
    (axes[0], "distortion_tax", "Distortion Tax Quintile",
     "Outcome Rates by Distortion Tax Level"),
    (axes[1], "selection_tax",  "Selection Tax Quintile",
     "Outcome Rates by Selection Tax Level"),
]:
    df["_bin"] = pd.qcut(df[metric], q=N_BINS, labels=False, duplicates="drop")

    bottoms = np.zeros(N_BINS)
    bar_w   = 0.65

    for outcome_label, col, color in OUTCOMES_STACKED:
        rates = df.groupby("_bin")[col].mean().reindex(range(N_BINS)).fillna(0).values
        bars  = ax.bar(range(N_BINS), rates, bar_w, bottom=bottoms,
                       color=color, label=outcome_label, edgecolor="white", linewidth=0.4)
        # Label if segment is wide enough to read
        for b_idx, (rate, bot) in enumerate(zip(rates, bottoms)):
            if rate > 0.04:
                ax.text(b_idx, bot + rate / 2, f"{rate:.1%}",
                        ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="bold")
        bottoms += rates

    # Bin-level N annotation above bars
    for b_idx in range(N_BINS):
        n = (df["_bin"] == b_idx).sum()
        ax.text(b_idx, bottoms[b_idx] + 0.005, f"n={n:,}",
                ha="center", va="bottom", fontsize=7, color="#555")

    ax.set_xticks(range(N_BINS))
    ax.set_xticklabels(BIN_LABS, fontsize=9)
    ax.set_xlabel(col_label, fontsize=10)
    ax.set_ylabel("Share of Swings", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

# Shared legend
handles = [mpatches.Patch(color=c, label=l) for l, _, c in OUTCOMES_STACKED]
fig1.legend(handles=handles, loc="lower center", ncol=5,
            fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

fig1.suptitle(
    "Swing Outcome Rates by Distortion / Selection Tax Level  ·  2023–2025",
    fontsize=12, fontweight="bold", y=1.01,
)
fig1.tight_layout()
out1 = "results/figures/outcome_rates.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"Saved {out1}")

# ── Figure 2: xwOBA on contact vs tax metrics ─────────────────────────────────

bip_df = df[bip].copy()
N_BINS2 = 10

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.patch.set_facecolor("white")

for ax, metric, label, color in [
    (axes2[0], "distortion_tax", "Distortion Tax", "#d73027"),
    (axes2[1], "selection_tax",  "Selection Tax",  "#2166ac"),
]:
    bip_df["_bin"] = pd.qcut(bip_df[metric], q=N_BINS2, labels=False, duplicates="drop")
    summary = (
        bip_df.groupby("_bin")
        .agg(
            mid    =(metric,  "mean"),
            mean_xw=("xwoba", "mean"),
            se_xw  =("xwoba", lambda x: sem(x, nan_policy="omit")),
            n      =("xwoba", "count"),
        )
        .reset_index()
    )

    ax.fill_between(
        summary["mid"],
        summary["mean_xw"] - 1.96 * summary["se_xw"],
        summary["mean_xw"] + 1.96 * summary["se_xw"],
        alpha=0.15, color=color,
    )
    ax.plot(summary["mid"], summary["mean_xw"],
            "o-", color=color, linewidth=2, markersize=6,
            markerfacecolor="white", markeredgewidth=2)

    for _, r in summary.iterrows():
        ax.text(r["mid"], r["mean_xw"] + 0.003, f"n={int(r['n']):,}",
                ha="center", va="bottom", fontsize=6.5, color="#666")

    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Mean xwOBA (ball in play)", fontsize=11)
    ax.set_title(f"Contact Quality vs {label}", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.axhline(0, color="#aaa", lw=0.8, linestyle=":")

fig2.suptitle(
    "xwOBA on Contact vs Distortion / Selection Tax  ·  BIP only  ·  2023–2025",
    fontsize=12, fontweight="bold", y=1.01,
)
fig2.tight_layout()
out2 = "results/figures/xwoba_relationship.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved {out2}")
