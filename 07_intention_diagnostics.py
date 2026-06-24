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


# ── Figure 1: Distributions ───────────────────────────────────────────────────

def plot_distributions(df, out="results/figures/intention_distributions.png"):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(
        "Posterior Predictions: Intended vs. Realized Swing Shape",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for col, resp in enumerate(RESPONSES):
        lbl    = LABELS[resp]
        v_int  = df[f"intended_{resp}"].dropna()
        v_real = df[resp].dropna()
        v_dev  = df[f"{resp}_dev"].dropna()

        # Row 0: intended vs realized
        ax = axes[0, col]
        lo   = min(v_int.quantile(0.005), v_real.quantile(0.005))
        hi   = max(v_int.quantile(0.995), v_real.quantile(0.995))
        bins = np.linspace(lo, hi, 55)
        ax.hist(v_real, bins=bins, color=C_REAL,   alpha=0.45, density=True, label="Realized")
        ax.hist(v_int,  bins=bins, color=C_INTENT, alpha=0.65, density=True, label="Intended")
        ax.axvline(v_int.mean(),  color=C_INTENT, lw=1.5, ls="--")
        ax.axvline(v_real.mean(), color=C_REAL,   lw=1.5, ls="--")
        ax.set_xlabel(lbl, fontsize=9)
        ax.set_ylabel("Density" if col == 0 else "", fontsize=9)
        ax.set_title(resp.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.annotate(
            f"μ_int={v_int.mean():.1f}\nμ_real={v_real.mean():.1f}",
            xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )
        if col == 0:
            ax.legend(fontsize=8, framealpha=0.7)

        # Row 1: deviation distribution
        ax2 = axes[1, col]
        bins_dev = np.linspace(v_dev.quantile(0.005), v_dev.quantile(0.995), 55)
        ax2.hist(v_dev, bins=bins_dev, color=C_DEV, alpha=0.7, density=True)
        ax2.axvline(0,          color="black",  lw=1.2, ls="-",  label="zero")
        ax2.axvline(v_dev.mean(), color=C_DEV,  lw=1.5, ls="--", label=f"μ={v_dev.mean():.2f}")
        p25, p75 = v_dev.quantile(0.25), v_dev.quantile(0.75)
        ax2.axvspan(p25, p75, alpha=0.12, color=C_DEV)
        ax2.set_xlabel(f"Deviation ({lbl})", fontsize=9)
        ax2.set_ylabel("Density" if col == 0 else "", fontsize=9)
        ax2.set_title(f"Deviation  IQR={p75 - p25:.1f}  σ={v_dev.std():.1f}", fontsize=9)
        ax2.legend(fontsize=7.5, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


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


# ── Figure 3: Zone heatmaps ────────────────────────────────────────────────────

def plot_zone_heatmaps(
    df,
    out="results/figures/zone_heatmaps.png",
    n_x=5, n_z=6,
):
    """
    Paper-friendly: VAA and HAA intended only, single-row layout.
    Shows batter's mechanical adaptation across the strike zone.
    """
    metrics = KEY_ZONE_METRICS
    cmaps   = {"vert_attack_angle": "RdYlGn", "horz_attack_angle": "RdBu_r"}

    x_edges = np.linspace(-1.5, 1.5, n_x + 1)
    z_edges = np.linspace( 0.5, 4.5, n_z + 1)
    x_ctrs  = (x_edges[:-1] + x_edges[1:]) / 2
    z_ctrs  = (z_edges[:-1] + z_edges[1:]) / 2

    df = df.copy()
    df["x_bin"] = pd.cut(df["plate_x"], bins=x_edges, labels=False)
    df["z_bin"]  = pd.cut(df["plate_z"], bins=z_edges, labels=False)

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    fig.suptitle(
        "Intended Swing Shape Across the Strike Zone",
        fontsize=12, fontweight="bold",
    )

    sz_rect_kwargs = dict(linewidth=1.8, edgecolor="white", facecolor="none", zorder=5)

    for col, resp in enumerate(metrics):
        ax   = axes[col]
        lbl  = LABELS[resp]
        icol = f"intended_{resp}"
        cmap = cmaps.get(resp, "RdYlGn")

        grid = (
            df.groupby(["z_bin", "x_bin"])[icol]
            .mean()
            .unstack("x_bin")
            .reindex(index=range(n_z), columns=range(n_x))
            .values
        )
        vmin, vmax = np.nanmin(grid), np.nanmax(grid)
        im = ax.imshow(
            grid, cmap=cmap, vmin=vmin, vmax=vmax,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
            aspect="auto",
        )
        plt.colorbar(im, ax=ax, shrink=0.8, label=lbl)

        denom = max(abs(vmin), abs(vmax))
        for zi, zc in enumerate(z_ctrs):
            for xi, xc in enumerate(x_ctrs):
                v = grid[zi, xi]
                if not np.isnan(v):
                    txt_color = "white" if abs(v) / denom > 0.55 else "black"
                    ax.text(xc, zc, f"{v:.1f}", ha="center", va="center",
                            fontsize=8, color=txt_color)

        ax.add_patch(patches.Rectangle((-0.83, 1.5), 1.66, 2.0, **sz_rect_kwargs))
        ax.set_xlabel("plate_x (ft)  ←arm  |  glove→", fontsize=9)
        ax.set_ylabel("plate_z (ft)" if col == 0 else "", fontsize=9)
        ax.set_title(f"{lbl} — Intended", fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} competitive swings")

    print("Figure 1: distributions...")
    plot_distributions(df)

    print("Figure 2: count effects...")
    plot_count_effects(df)

    print("Figure 3: zone heatmaps...")
    plot_zone_heatmaps(df)

    print("Done.")
