"""
causal_chain.py
Four charts that address the core interpretability question:
"Are these metrics doing anything, and what are they actually measuring?"

1. Mechanism check: does more post-commit movement → more disruption_tax?
   If yes, the causal chain is working. If no, the mediator models are broken.

2. Pitcher type quadrant: distortion (movement) vs selection (deception/sequence).
   Shows two distinct modes of disruption — not all pitchers work the same way.

3. Contact & whiff rates by disruption quintile (not BIP-conditional).
   The right outcome validation for the total disruption_tax.

4. Pitch type distribution: which pitch types generate the most distortion?
   Direct pitch design signal.

Run from project root:
    .venv\\Scripts\\python.exe results_scripts\\causal_chain.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

Path("results/figures").mkdir(parents=True, exist_ok=True)

# ── Load & merge ──────────────────────────────────────────────────────────────

xrv = pd.read_parquet("results/xrv_causal.parquet")

sw_cols = [
    "game_pk", "at_bat_number", "pitch_number",
    "pitcher_full_name",
    "pc150_dev_total",
    "is_whiff", "is_contact",
]
sw = (
    pd.read_parquet("data/swings_precommit.parquet", columns=sw_cols)
    .drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
)

df = (
    xrv.merge(sw, on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
    .dropna(subset=["disruption_tax", "distortion_tax", "selection_tax",
                    "pc150_dev_total", "pitch_type"])
)

# Pitch type display labels
PITCH_LABELS = {
    "FF": "4-Seam FB", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider",    "SW": "Sweeper","CU": "Curveball",
    "CH": "Changeup",  "FS": "Splitter","KC": "Knuckle-CB",
    "ST": "Sweeper",   "CS": "Curveball",
}
df["pitch_label"] = df["pitch_type"].map(PITCH_LABELS).fillna(df["pitch_type"])

PITCH_FOCUS  = ["4-Seam FB", "Sinker", "Cutter", "Slider", "Sweeper",
                "Curveball", "Changeup", "Splitter"]
PT_COLORS    = ["#e41a1c", "#ff7f00", "#984ea3", "#377eb8", "#4daf4a",
                "#a65628", "#f781bf", "#999999"]
PT_COLOR_MAP = dict(zip(PITCH_FOCUS, PT_COLORS))

# ── Figure 1: Mechanism check ─────────────────────────────────────────────────
# Does more post-commit deviation → more disruption?
# If the causal chain works: more movement → more negative disruption_tax.
# If flat: the outcome models aren't picking up on angular deviations.

fig1, ax1 = plt.subplots(figsize=(10, 5.5))
fig1.patch.set_facecolor("white")

for pt in PITCH_FOCUS:
    sub = df[df["pitch_label"] == pt].copy()
    if len(sub) < 200:
        continue
    # Convert feet to inches for readability
    sub["dev_in"] = sub["pc150_dev_total"] * 12
    sub["_bin"]   = pd.qcut(sub["dev_in"], q=10, duplicates="drop")
    summary = (
        sub.groupby("_bin", observed=True)
        .agg(
            mid=("dev_in",         "median"),
            tax=("disruption_tax", "mean"),
            n  =("disruption_tax", "count"),
        )
        .reset_index()
        .query("n >= 20")
    )
    color = PT_COLOR_MAP.get(pt, "#888888")
    ax1.plot(summary["mid"], summary["tax"] * 100,
             "o-", color=color, linewidth=1.8, markersize=5,
             label=f"{pt} (n={len(sub):,})", alpha=0.85)

ax1.axhline(0, color="#aaa", lw=0.8, linestyle=":")
ax1.set_xlabel("Post-commit deviation at plate (inches, pc150)", fontsize=11)
ax1.set_ylabel("Mean disruption tax (runs per 100 swings)", fontsize=11)
ax1.set_title(
    "Mechanism Check: Post-commit Movement → Swing Disruption?\n"
    "(downward slope = more movement causes more disruption)",
    fontsize=11, fontweight="bold",
)
ax1.legend(fontsize=8, frameon=False, loc="lower left")
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(True, alpha=0.2, linestyle="--")
# More disruption = more negative → invert so "worse" is up on the chart
ax1.invert_yaxis()

fig1.tight_layout()
p1 = "results/figures/09a_mechanism_check.png"
fig1.savefig(p1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"Saved {p1}")

# ── Figure 2: Pitcher type quadrant ──────────────────────────────────────────
# Each pitcher = a point. X = mean distortion_tax (movement-caused disruption).
# Y = mean selection_tax (deception / sequencing / batter decision error).
# Quadrant: who generates disruption from movement vs. other means?

pitcher_agg = (
    df.groupby("pitcher_id")
    .agg(
        name          =("pitcher_full_name", "first"),
        n             =("disruption_tax",    "count"),
        mean_dist_tax =("distortion_tax",    "mean"),
        mean_sel_tax  =("selection_tax",     "mean"),
        mean_total    =("disruption_tax",    "mean"),
    )
    .reset_index()
    .query("n >= 200")   # ≥200 competitive swings faced (season-level stability)
)

if len(pitcher_agg) < 5:
    print("Not enough pitchers with ≥200 swings for quadrant chart — skipping Figure 2.")
else:
    dist_mu, dist_sd = pitcher_agg["mean_dist_tax"].mean(), pitcher_agg["mean_dist_tax"].std()
    sel_mu,  sel_sd  = pitcher_agg["mean_sel_tax"].mean(),  pitcher_agg["mean_sel_tax"].std()
    pitcher_agg["dist_z"] = (pitcher_agg["mean_dist_tax"] - dist_mu) / dist_sd
    pitcher_agg["sel_z"]  = (pitcher_agg["mean_sel_tax"]  - sel_mu)  / sel_sd

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    fig2.patch.set_facecolor("white")

    sc = ax2.scatter(
        pitcher_agg["dist_z"], pitcher_agg["sel_z"],
        s=np.clip(pitcher_agg["n"] / 8, 20, 180),
        c=pitcher_agg["mean_total"] * 100,
        cmap="RdBu_r", vmin=-3, vmax=0,
        alpha=0.65, edgecolors="#ccc", linewidth=0.4,
    )
    plt.colorbar(sc, ax=ax2, label="Mean disruption tax (runs per 100 swings)")

    # Label the top disruptors by total
    top = pitcher_agg.nsmallest(15, "mean_total")
    for _, row in top.iterrows():
        ax2.annotate(
            row["name"].split()[-1],   # last name only
            (row["dist_z"], row["sel_z"]),
            fontsize=6.5, ha="center", va="bottom",
            xytext=(0, 4), textcoords="offset points", color="#333",
        )

    ax2.axhline(0, color="#bbb", lw=0.8, linestyle="--")
    ax2.axvline(0, color="#bbb", lw=0.8, linestyle="--")

    # Both axes: more negative = more disruption = plotted at top/left
    ax2.invert_xaxis()
    ax2.invert_yaxis()

    ax2.set_xlabel("← more movement-caused disruption  |  Distortion Tax (z-score)", fontsize=10)
    ax2.set_ylabel("← more decision-error disruption  |  Selection Tax (z-score)", fontsize=10)

    # Quadrant semantics with BOTH axes inverted:
    #   invert_xaxis → right edge = most negative dist_z = most distortion
    #   invert_yaxis → top  edge = most negative sel_z  = most selection
    # transform (0,0) = bottom-left = least distortion + least selection
    # transform (1,0) = bottom-right = most distortion  + least selection
    # transform (0,1) = top-left    = least distortion  + most selection
    # transform (1,1) = top-right   = most distortion   + most selection
    fs = 7.5
    kw = dict(transform=ax2.transAxes, fontsize=fs, color="#666", alpha=0.55, ha="left")
    ax2.text(0.02, 0.02, "Low disruption\n(neither channel)",       va="bottom", **kw)
    ax2.text(0.70, 0.02, "Movement specialists\n(late break)",      va="bottom", **kw)
    ax2.text(0.02, 0.92, "Deception specialists\n(tunneling/sequence)", va="top", **kw)
    ax2.text(0.70, 0.92, "Elite disruptors\n(both channels)",       va="top",    **kw)

    ax2.set_title(
        "Pitcher Disruption Profile: Movement vs. Deception  ·  2023–2025\n"
        "(≥200 competitive swings faced; size = swing count; color = total disruption)",
        fontsize=10, fontweight="bold",
    )
    ax2.spines[["top", "right"]].set_visible(False)

    fig2.tight_layout()
    p2 = "results/figures/09b_pitcher_quadrant.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved {p2}")

# ── Figure 3: Contact & whiff rates by disruption quintile ───────────────────
# Not BIP-conditional. Shows how disruption manifests across ALL swings.
# Q1 = most disruptive (lowest = most negative tax). Q5 = least.

N_BINS = 5
BIN_LABS = ["Q1\n(most\ndisruptive)", "Q2", "Q3", "Q4", "Q5\n(least\ndisruptive)"]

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.patch.set_facecolor("white")

for ax, metric, label, color in [
    (axes3[0], "distortion_tax", "Distortion Tax",  "#d73027"),
    (axes3[1], "selection_tax",  "Selection Tax",   "#2166ac"),
]:
    df["_bin"] = pd.qcut(df[metric], q=N_BINS, labels=False, duplicates="drop")
    summary = (
        df.groupby("_bin")
        .agg(
            contact=("is_contact", "mean"),
            whiff  =("is_whiff",   "mean"),
            n      =("is_whiff",   "count"),
        )
        .reset_index()
    )

    x    = np.arange(N_BINS)
    w    = 0.35
    ax.bar(x - w/2, summary["contact"], w, label="Contact rate",
           color=color, alpha=0.75)
    ax.bar(x + w/2, summary["whiff"],   w, label="Whiff rate",
           color=color, alpha=0.35, hatch="//", edgecolor=color, linewidth=0.6)

    for i, (cr, wr) in enumerate(zip(summary["contact"], summary["whiff"])):
        ax.text(i - w/2, cr + 0.008, f"{cr:.1%}", ha="center", va="bottom", fontsize=7.5)
        ax.text(i + w/2, wr + 0.008, f"{wr:.1%}", ha="center", va="bottom", fontsize=7.5)
    for i, n in enumerate(summary["n"]):
        ax.text(i, -0.03, f"n={n:,}", ha="center", va="top", fontsize=6.5, color="#666")

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABS, fontsize=8.5)
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_title(f"Contact & Whiff Rate by {label}", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

fig3.suptitle(
    "Contact & Whiff Rate by Disruption Level  ·  All Swings  ·  2023–2025\n"
    "Q1 = most disruptive (most negative tax)",
    fontsize=11, fontweight="bold", y=1.02,
)
fig3.tight_layout()
p3 = "results/figures/09c_contact_whiff_rates.png"
fig3.savefig(p3, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig3)
print(f"Saved {p3}")

# ── Figure 4: Pitch type disruption distributions ────────────────────────────
# Violin plot of disruption_tax and distortion_tax by pitch type.
# Which pitch types generate the most disruption, and through which channel?

pt_counts = df["pitch_label"].value_counts()
qualified  = pt_counts[pt_counts >= 200].index.tolist()
qualified_focus = [p for p in PITCH_FOCUS if p in qualified]

if len(qualified_focus) < 2:
    print("Not enough pitch types with ≥200 swings — skipping Figure 4.")
else:
    # Sort by mean distortion_tax (most disruptive first)
    type_means = (
        df[df["pitch_label"].isin(qualified_focus)]
        .groupby("pitch_label")["distortion_tax"].mean()
        .sort_values()
    )
    sorted_types = type_means.index.tolist()

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5.5))
    fig4.patch.set_facecolor("white")

    for ax, metric, label in [
        (axes4[0], "distortion_tax", "Distortion Tax (runs/swing)"),
        (axes4[1], "selection_tax",  "Selection Tax (runs/swing)"),
    ]:
        data_series = [
            df.loc[df["pitch_label"] == pt, metric].dropna().values
            for pt in sorted_types
        ]
        means = [type_means[pt] if metric == "distortion_tax"
                 else df.loc[df["pitch_label"] == pt, metric].mean()
                 for pt in sorted_types]
        colors = [PT_COLOR_MAP.get(pt, "#888") for pt in sorted_types]

        vp = ax.violinplot(data_series, positions=range(len(sorted_types)),
                           showmedians=False, showextrema=False, widths=0.7)
        for body, c in zip(vp["bodies"], colors):
            body.set_alpha(0.4)
            body.set_facecolor(c)
            body.set_edgecolor(c)

        ax.scatter(range(len(sorted_types)), means,
                   s=55, zorder=5, edgecolors="white", linewidth=0.8,
                   c=colors)
        for i, (m, pt) in enumerate(zip(means, sorted_types)):
            ax.text(i, m - 0.0015, f"{m*100:.1f}", ha="center", va="top",
                    fontsize=6.5, color="#444")

        ax.set_xticks(range(len(sorted_types)))
        ax.set_xticklabels(sorted_types, rotation=25, ha="right", fontsize=9)
        ax.axhline(0, color="#aaa", lw=0.8, linestyle=":")
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label} by Pitch Type", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{y*100:.2f}")
        )
        ax.set_ylabel(f"{label.split('(')[0].strip()} (runs per 100 swings)", fontsize=10)

    fig4.suptitle(
        "Disruption Profile by Pitch Type  ·  2023–2025\n"
        "(violin = distribution; dot = mean; number = mean × 100 runs/swing)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig4.tight_layout()
    p4 = "results/figures/09d_pitch_type_disruption.png"
    fig4.savefig(p4, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig4)
    print(f"Saved {p4}")
