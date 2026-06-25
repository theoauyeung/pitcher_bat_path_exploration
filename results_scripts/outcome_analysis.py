"""
outcome_analysis.py
Distortion / selection tax broken down by swing outcome, and their
relationship with contact quality (xwOBA).

Run from project root:
    .venv\\Scripts\\python.exe results_scripts\\outcome_analysis.py

Outputs:
    results/figures/outcome_table.png      -- mean taxes by outcome category
    results/figures/xwoba_relationship.png -- xwOBA vs tax metrics (BIP only)
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

df["is_foul"] = (df["is_contact"] == 1) & (df["is_bip"] == 0)

# xwOBA from linear weights (BIP only)
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
is_out_in_play = bip & ~(df["is_home_run"] == 1) & ~(df["is_triple"] == 1) \
               & ~(df["is_double"] == 1) & ~(df["is_single"] == 1)
xwoba.loc[is_out_in_play] = lw["out_in_play"]
df["xwoba"] = xwoba

# ── Outcome categories ────────────────────────────────────────────────────────

OUTCOMES = [
    ("All Swings",   pd.Series(True, index=df.index)),
    ("Whiff",        df["is_whiff"] == 1),
    ("Foul",         df["is_foul"]),
    ("Ball in Play", bip),
    ("Out in Play",  is_out_in_play),
    ("Single",       bip & (df["is_single"] == 1) & ~(df["is_double"] == 1)
                         & ~(df["is_triple"] == 1) & ~(df["is_home_run"] == 1)),
    ("Double",       bip & (df["is_double"] == 1) & ~(df["is_triple"] == 1)
                         & ~(df["is_home_run"] == 1)),
    ("Triple",       bip & (df["is_triple"] == 1) & ~(df["is_home_run"] == 1)),
    ("Home Run",     bip & (df["is_home_run"] == 1)),
]

rows = []
for label, mask in OUTCOMES:
    sub = df[mask]
    if len(sub) < 30:
        continue
    rows.append({
        "Outcome":       label,
        "N":             len(sub),
        "Disruption":    sub["disruption_tax"].mean(),
        "Distortion":    sub["distortion_tax"].mean(),
        "Selection":     sub["selection_tax"].mean(),
        "Dist Share %":  sub["distortion_share"].mean() * 100,
    })

tbl = pd.DataFrame(rows)

# ── Figure 1: outcome table ───────────────────────────────────────────────────

COLS  = ["Outcome", "N", "Disruption", "Distortion", "Selection", "Dist Share %"]
HDRS  = ["Outcome", "N", "Disruption Tax", "Distortion Tax", "Selection Tax", "Distortion %"]
ALIGNS = ["left", "right", "right", "right", "right", "right"]
COL_W  = [2.10, 0.80, 1.40, 1.40, 1.40, 1.30]

N_ROWS  = len(tbl)
ROW_H   = 0.34
HDR_H   = 0.42
MARG_T  = 0.55
MARG_B  = 0.20
MARG_L  = 0.18
FIG_W   = sum(COL_W) + 2 * MARG_L
FIG_H   = MARG_T + HDR_H + N_ROWS * ROW_H + MARG_B

fig1, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig1.patch.set_facecolor("white")

# Column x positions
xs = [MARG_L]
for w in COL_W[:-1]:
    xs.append(xs[-1] + w)

y_top = FIG_H - MARG_T
y_hdr = y_top - HDR_H
y_bot = MARG_B

# Top & bottom rules
for y, lw_px in [(y_top, 1.4), (y_hdr, 0.6), (y_bot, 1.4)]:
    ax.plot([MARG_L, FIG_W - MARG_L], [y, y], color="#444", lw=lw_px)

# Header background
ax.add_patch(mpatches.Rectangle(
    (MARG_L, y_hdr), FIG_W - 2 * MARG_L, HDR_H,
    facecolor="#f0f0f0", edgecolor="none", zorder=1,
))

# Column headers
for i, (hdr, xi, align) in enumerate(zip(HDRS, xs, ALIGNS)):
    pad = 0.06 if align == "left" else COL_W[i] - 0.06
    ha  = "left" if align == "left" else "right"
    ax.text(xi + pad, y_hdr + HDR_H / 2, hdr,
            ha=ha, va="center", fontsize=9, fontweight="bold",
            color="#111", fontfamily="serif")

# Data rows
DISRUPTION_RANGE = (tbl["Disruption"].min(), tbl["Disruption"].max())
CMAP = plt.cm.RdYlGn_r  # red = high disruption, green = low

for i, (_, row) in enumerate(tbl.iterrows()):
    y_row_top = y_hdr - i * ROW_H
    y_row_bot = y_row_top - ROW_H
    cy = (y_row_top + y_row_bot) / 2

    # Alternating shade
    if i % 2 == 1:
        ax.add_patch(mpatches.Rectangle(
            (MARG_L, y_row_bot), FIG_W - 2 * MARG_L, ROW_H,
            facecolor="#fafafa", edgecolor="none", zorder=0,
        ))

    # Row separator
    ax.plot([MARG_L, FIG_W - MARG_L], [y_row_bot, y_row_bot],
            color="#e0e0e0", lw=0.4)

    # Cell values
    vals = [
        row["Outcome"],
        f"{int(row['N']):,}",
        f"{row['Disruption']:+.4f}",
        f"{row['Distortion']:+.4f}",
        f"{row['Selection']:+.4f}",
        f"{row['Dist Share %']:.1f}%",
    ]
    for j, (val, xi, align) in enumerate(zip(vals, xs, ALIGNS)):
        pad = 0.06 if align == "left" else COL_W[j] - 0.06
        ha  = "left" if align == "left" else "right"
        fw  = "bold" if j == 0 and i == 0 else "normal"
        ax.text(xi + pad, cy, val,
                ha=ha, va="center", fontsize=8.5, color="#111",
                fontfamily="serif", fontweight=fw)

# Title
ax.text(MARG_L, FIG_H - MARG_T * 0.38,
        "Mean Disruption / Distortion / Selection Tax by Swing Outcome",
        ha="left", va="center", fontsize=11, fontweight="bold",
        color="#111", fontfamily="serif")
ax.text(MARG_L, FIG_H - MARG_T * 0.75,
        "2023–2025 · min. 30 swings per cell · negative = pitcher advantage",
        ha="left", va="center", fontsize=8.5, color="#555", fontfamily="serif")

out1 = "results/figures/outcome_table.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"Saved {out1}")

# ── Figure 2: xwOBA relationship ──────────────────────────────────────────────

bip_df = df[bip].copy()
N_BINS = 10

fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
fig2.patch.set_facecolor("white")

METRIC_META = [
    ("distortion_tax", "Distortion Tax", "#d73027"),
    ("selection_tax",  "Selection Tax",  "#2166ac"),
]

for ax, (metric, label, color) in zip(axes, METRIC_META):
    bip_df["_bin"] = pd.qcut(bip_df[metric], q=N_BINS, labels=False, duplicates="drop")
    summary = (
        bip_df.groupby("_bin")
        .agg(
            mid      =(metric,  "mean"),
            mean_xw  =("xwoba", "mean"),
            se_xw    =("xwoba", lambda x: sem(x, nan_policy="omit")),
            n        =("xwoba", "count"),
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

    # Annotate n per bin
    for _, r in summary.iterrows():
        ax.text(r["mid"], r["mean_xw"] + 0.003, f"n={int(r['n']):,}",
                ha="center", va="bottom", fontsize=6.5, color="#666")

    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Mean xwOBA (ball in play)", fontsize=11)
    ax.set_title(f"Contact Quality vs {label}", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--")

    # Zero line
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
