"""
07_intention_diagnostics.py
Intention model diagnostic visualizations.

Three figures:
  07a — Distributions: intended vs realized per response + deviation histograms
  07b — Count effects: mean intended swing shape by count group and (balls × strikes) matrix
  07c — Zone heatmaps: mean intended shape and deviation across the strike zone

Run:
    .venv\\Scripts\\python.exe 07_intention_diagnostics.py

Output:
    results/figures/07a_intention_distributions.png
    results/figures/07b_count_effects.png
    results/figures/07c_zone_heatmaps.png
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

Path("results/figures").mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ── Config ────────────────────────────────────────────────────────────────────

RESPONSES = [
    "vert_attack_angle",
    "horz_attack_angle",
    "swing_path_tilt",
    "bat_speed",
    "swing_length",
]
LABELS = {
    "vert_attack_angle": "Vert. Attack Angle (°)",
    "horz_attack_angle": "Horz. Attack Angle (°)",
    "swing_path_tilt":   "Swing Path Tilt (°)",
    "bat_speed":         "Bat Speed (mph)",
    "swing_length":      "Swing Length (ft)",
}
ANGULAR = ["vert_attack_angle", "horz_attack_angle", "swing_path_tilt"]

# Paper-friendly subsets: strongest count and zone signals
KEY_COUNT_METRICS = ["vert_attack_angle", "bat_speed"]
KEY_ZONE_METRICS  = ["vert_attack_angle", "horz_attack_angle"]

COUNT_ORDER  = ["Hitter", "Early", "Full", "Pitcher"]
COUNT_COLORS = {"Hitter": "#2ca02c", "Early": "#4878d0", "Full": "#ff7f0e", "Pitcher": "#d62728"}

C_INTENT  = "#4878d0"
C_REAL    = "#ef553b"
C_DEV     = "#7b4f8e"

# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(min_bat_speed=50):
    sw     = pd.read_parquet("data/swings_precommit.parquet")
    intent = pd.read_parquet("models/intended_df.parquet")
    df = sw.join(intent)
    df = df[
        (df["is_swing"] == 1) &
        (df["bat_speed"] >= min_bat_speed) &
        df["intended_vert_attack_angle"].notna()
    ].copy()
    for resp in RESPONSES:
        df[f"{resp}_dev"] = df[resp] - df[f"intended_{resp}"]
    return df




# ── Figure 2: Count effects ────────────────────────────────────────────────────

def plot_count_effects(df, out="results/figures/07b_count_effects.png"):
    """
    Paper-friendly: VAA and bat speed only, wide/short layout.
    Left col: bar chart of mean intended shape by count_group.
    Right col: (balls × strikes) heatmap of Δ from grand mean.
    """
    metrics = KEY_COUNT_METRICS
    fig, axes = plt.subplots(len(metrics), 2, figsize=(14, 4.2 * len(metrics)))
    fig.suptitle(
        "Intended Swing Shape by Count",
        fontsize=12, fontweight="bold",
    )

    df_cg = df.dropna(subset=["count_group"])
    df_cm = df[(df["balls"].between(0, 3)) & (df["strikes"].between(0, 2))].copy()

    for row, resp in enumerate(metrics):
        lbl  = LABELS[resp]
        icol = f"intended_{resp}"

        # ── Left: bar by count group ───────────────────────────────────────
        ax = axes[row, 0]
        grp   = df_cg.groupby("count_group")[icol]
        means = grp.mean().reindex(COUNT_ORDER)
        sds   = grp.std().reindex(COUNT_ORDER)
        colors = [COUNT_COLORS[g] for g in COUNT_ORDER]

        bars = ax.bar(COUNT_ORDER, means.values, color=colors, alpha=0.85,
                      edgecolor="white", width=0.6)
        ax.errorbar(COUNT_ORDER, means.values, yerr=sds.values, fmt="none",
                    color="black", capsize=4, lw=1.2)
        for bar, m, sd in zip(bars, means.values, sds.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sd * 0.02,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel(f"Mean {lbl}", fontsize=9)
        ax.set_title(f"{lbl} — Count Group", fontsize=10)
        ax.tick_params(axis="x", labelsize=9)
        _add_n_labels(ax, df_cg, "count_group", COUNT_ORDER)

        # ── Right: (balls × strikes) heatmap of deviation from grand mean ─
        ax = axes[row, 1]
        grand = df_cm[icol].mean()
        pivot = (
            df_cm.groupby(["balls", "strikes"])[icol]
            .mean()
            .subtract(grand)
            .unstack("strikes")
            .reindex(index=[0, 1, 2, 3], columns=[0, 1, 2])
        )
        raw_pivot = (
            df_cm.groupby(["balls", "strikes"])[icol]
            .mean()
            .unstack("strikes")
            .reindex(index=[0, 1, 2, 3], columns=[0, 1, 2])
        )
        vabs = np.nanmax(np.abs(pivot.values))
        im = ax.imshow(pivot.values, cmap="RdBu_r",
                       vmin=-vabs, vmax=vabs, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.85, label=f"Δ vs. grand mean ({lbl})")

        for bi in range(4):
            for si in range(3):
                val = raw_pivot.iloc[bi, si]
                if not np.isnan(val):
                    txt_col = "white" if abs(pivot.iloc[bi, si]) > vabs * 0.4 else "black"
                    ax.text(si, bi, f"{val:.1f}",
                            ha="center", va="center", fontsize=9, fontweight="bold",
                            color=txt_col)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["0 strikes", "1 strike", "2 strikes"], fontsize=9)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["0 balls", "1 ball", "2 balls", "3 balls"], fontsize=9)
        ax.set_title(f"{lbl} — Δ from mean by Count", fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _add_n_labels(ax, df, group_col, order):
    """Add n= annotations below each bar."""
    ns = df.groupby(group_col).size().reindex(order)
    ymin = ax.get_ylim()[0]
    for i, n in enumerate(ns):
        ax.text(i, ymin, f"n={n:,}", ha="center", va="top", fontsize=7, color="grey")




# ── Figure 4: Fixed effects summary table ─────────────────────────────────────

# Canonical row order and display labels
_PARAM_ORDER = [
    "Intercept",
    "scale(balls)",
    "scale(strikes)",
    "scale(plate_x_bat)",
    "scale(plate_z)",
    "scale(plate_z_sq)",
    "scale(offset_y_ms)",
    "pitcher_throws_L",
    "pitcher_throws_L:scale(plate_x_bat)",
]

# Greek-letter labels for academic table
_PARAM_GREEK = {
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

# Column group headers (no units — kept in sub-header context)
_COL_GROUP = {
    "vert_attack_angle":  "Vert. Attack Angle",
    "horz_attack_angle":  "Horz. Attack Angle",
    "swing_path_tilt":    "Swing Path Tilt",
    "bat_speed":          "Bat Speed",
    "swing_length":       "Swing Length",
}

_RE_SUFFIXES  = ("_sigma", "_offset")
_SKIP_VARS    = {"sigma", "mu"}
_ANGULAR_ONLY = {
    "scale(plate_z_sq)", "scale(offset_y_ms)",
    "pitcher_throws_L:scale(plate_x_bat)",
}


def _extract_fixed_effects(idata):
    """Pull posterior mean and 95% CI for every scalar fixed effect."""
    post = idata.posterior
    out  = {}
    for v in post.data_vars:
        if (set(post[v].dims) == {"chain", "draw"}
                and v not in _SKIP_VARS
                and not any(v.endswith(s) for s in _RE_SUFFIXES)):
            draws = post[v].values.ravel()
            out[v] = {
                "mean": float(draws.mean()),
                "lo":   float(np.percentile(draws, 2.5)),
                "hi":   float(np.percentile(draws, 97.5)),
            }
    return out


def plot_fixed_effects_table(
    model_path="models/intention_result.joblib",
    out="results/figures/07d_fixed_effects.png",
):
    """Academic-style fixed-effects table: grouped column headers, Greek row labels,
    separate Mean / Lower / Upper columns, horizontal rules only."""
    print(f"  Loading {model_path}...")
    result    = joblib.load(model_path)
    coefs     = {r: _extract_fixed_effects(result["idata"][r]) for r in RESPONSES}

    # ── Layout (all in inches) ────────────────────────────────────────────────
    LABEL_W  = 1.5    # parameter label column
    SUB_W    = 0.82   # each of Mean / Lower / Upper
    ROW_H    = 0.30   # data row height
    HDR_H    = 0.30   # each header tier height
    MARG_L   = 0.08
    MARG_R   = 0.08
    MARG_T   = 0.12
    MARG_B   = 0.12
    N_SUB    = 3      # Mean, Lower, Upper
    N_R      = len(RESPONSES)
    N_P      = len(_PARAM_ORDER)

    fig_w = MARG_L + LABEL_W + N_R * N_SUB * SUB_W + MARG_R
    fig_h = MARG_T + 2 * HDR_H + N_P * ROW_H + MARG_B

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    x0    = MARG_L
    x1    = fig_w - MARG_R
    y_top = fig_h - MARG_T
    y_g   = y_top - HDR_H        # bottom of group-header row
    y_s   = y_g   - HDR_H        # bottom of sub-header row = top of data
    y_bot = MARG_B

    def grp_x(j):   return x0 + LABEL_W + j * N_SUB * SUB_W
    def sub_cx(j, k): return grp_x(j) + (k + 0.5) * SUB_W

    lw_h = 0.9   # horizontal rule weight
    lw_v = 0.5   # vertical divider weight

    # Horizontal rules
    ax.plot([x0, x1], [y_top, y_top], "k-", lw=lw_h)          # top
    ax.plot([x0 + LABEL_W, x1], [y_g, y_g], "k-", lw=0.4)     # between header tiers
    ax.plot([x0, x1], [y_s,   y_s],   "k-", lw=lw_h)          # below headers
    ax.plot([x0, x1], [y_bot, y_bot], "k-", lw=lw_h)          # bottom

    # Vertical dividers: after label col + between response groups + right edge
    for xv in [x0 + LABEL_W] + [grp_x(j) for j in range(1, N_R)] + [x1]:
        ax.plot([xv, xv], [y_bot, y_top], "k-", lw=lw_v)

    # Group headers (response names, centered over 3 sub-cols)
    for j, resp in enumerate(RESPONSES):
        cx = grp_x(j) + 1.5 * SUB_W
        cy = (y_top + y_g) / 2
        ax.text(cx, cy, _COL_GROUP[resp],
                ha="center", va="center", fontsize=8.5, fontfamily="serif")

    # Sub-headers
    ax.text(x0 + LABEL_W / 2, (y_g + y_s) / 2, "Parameter",
            ha="center", va="center", fontsize=8, fontfamily="serif")
    for j in range(N_R):
        for k, lbl in enumerate(["Mean", "Lower", "Upper"]):
            ax.text(sub_cx(j, k), (y_g + y_s) / 2, lbl,
                    ha="center", va="center", fontsize=8, fontfamily="serif")

    # Data rows
    for i, param in enumerate(_PARAM_ORDER):
        cy = y_s - (i + 0.5) * ROW_H
        ax.text(x0 + LABEL_W / 2, cy, _PARAM_GREEK[param],
                ha="center", va="center", fontsize=9, fontfamily="serif")
        for j, resp in enumerate(RESPONSES):
            is_ang = resp in ANGULAR
            if param in _ANGULAR_ONLY and not is_ang:
                for k in range(N_SUB):
                    ax.text(sub_cx(j, k), cy, "—",
                            ha="center", va="center", fontsize=8.5,
                            color="#999999", fontfamily="serif")
            else:
                stats = coefs[resp].get(param)
                if stats:
                    for k, val in enumerate([stats["mean"], stats["lo"], stats["hi"]]):
                        ax.text(sub_cx(j, k), cy, f"{val:.3f}",
                                ha="center", va="center", fontsize=8.5,
                                fontfamily="serif")

    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # CSV
    csv_path = out.replace(".png", ".csv")
    rows_csv = []
    for param in _PARAM_ORDER:
        row = {"Parameter": _PARAM_GREEK[param]}
        for resp in RESPONSES:
            is_ang = resp in ANGULAR
            pfx = _COL_GROUP[resp]
            if param in _ANGULAR_ONLY and not is_ang:
                row[f"{pfx}_Mean"] = "—"; row[f"{pfx}_Lower"] = "—"; row[f"{pfx}_Upper"] = "—"
            else:
                s = coefs[resp].get(param, {})
                row[f"{pfx}_Mean"]  = f"{s.get('mean', float('nan')):.3f}"
                row[f"{pfx}_Lower"] = f"{s.get('lo',   float('nan')):.3f}"
                row[f"{pfx}_Upper"] = f"{s.get('hi',   float('nan')):.3f}"
        rows_csv.append(row)
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False)

    print(f"Saved {out}")
    print(f"Saved {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} competitive swings")

   

    print("Figure 2: count effects...")
    plot_count_effects(df)



    print("Figure 4: fixed effects table...")
    plot_fixed_effects_table()

    print("Done.")
