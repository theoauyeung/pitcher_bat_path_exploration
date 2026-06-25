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

# Display names for Bambi's posterior variable names → readable labels
_PARAM_LABELS = {
    "Intercept":                        "Intercept",
    "scale(balls)":                     "Balls (z)",
    "scale(strikes)":                   "Strikes (z)",
    "scale(plate_x_bat)":               "Plate X batter (z)",
    "scale(plate_z)":                   "Plate Z (z)",
    "scale(plate_z_sq)":                "Plate Z² (z)",
    "scale(offset_y_ms)":               "Timing (z)",
    "pitcher_throws_L":                 "Pitcher LHP",
    "pitcher_throws_L:scale(plate_x_bat)": "LHP × Plate X",
}

# Canonical row order
_PARAM_ORDER = list(_PARAM_LABELS.keys())

_COL_LABELS = {
    "vert_attack_angle":  "VAA (°)",
    "horz_attack_angle":  "HAA (°)",
    "swing_path_tilt":    "Tilt (°)",
    "bat_speed":          "Bat Spd (mph)",
    "swing_length":       "Swing Len (ft)",
}

_RE_SUFFIXES  = ("_sigma", "_offset")
_SKIP_VARS    = {"sigma", "mu"}
_ANGULAR_ONLY = {
    "scale(plate_z_sq)", "scale(offset_y_ms)",
    "pitcher_throws_L:scale(plate_x_bat)",
}


def _extract_fixed_effects(idata):
    """Pull posterior mean, SD, and 95% CI for every scalar fixed effect."""
    post = idata.posterior
    out  = {}
    for v in post.data_vars:
        if (set(post[v].dims) == {"chain", "draw"}
                and v not in _SKIP_VARS
                and not any(v.endswith(s) for s in _RE_SUFFIXES)):
            draws = post[v].values.ravel()
            out[v] = {
                "mean": float(draws.mean()),
                "sd":   float(draws.std()),
                "lo":   float(np.percentile(draws, 2.5)),
                "hi":   float(np.percentile(draws, 97.5)),
            }
    return out


def plot_fixed_effects_table(
    model_path="models/intention_result.joblib",
    out="results/figures/07d_fixed_effects.png",
):
    """Load Bambi idata from joblib, extract fixed-effect posteriors, render table."""
    print(f"  Loading {model_path}...")
    result = joblib.load(model_path)
    idata_dict = result["idata"]   # {resp: InferenceData}

    # Build per-response coefficient dicts
    coefs = {resp: _extract_fixed_effects(idata_dict[resp]) for resp in RESPONSES}

    # Assemble cell text and colour
    cell_text, cell_color = [], []
    for param_key in _PARAM_ORDER:
        row_text, row_color = [], []
        for resp in RESPONSES:
            is_ang = resp in ANGULAR
            if param_key in _ANGULAR_ONLY and not is_ang:
                row_text.append("—")
                row_color.append("#f0f0f0")
                continue
            stats = coefs[resp].get(param_key)
            if stats is None:
                row_text.append("—")
                row_color.append("#f0f0f0")
                continue
            # Significance: p≈0 when CI excludes 0
            sig = "" if (stats["lo"] < 0 < stats["hi"]) else "*"
            row_text.append(
                f"{stats['mean']:+.3f}{sig}\n[{stats['lo']:.3f}, {stats['hi']:.3f}]"
            )
            if sig == "":
                row_color.append("#ffffff")
            elif stats["mean"] > 0:
                row_color.append("#dff0d8")
            else:
                row_color.append("#fde8e8")
        cell_text.append(row_text)
        cell_color.append(row_color)

    row_labels = [_PARAM_LABELS[k] for k in _PARAM_ORDER]
    col_labels  = [_COL_LABELS[r] for r in RESPONSES]
    n_rows, n_cols = len(_PARAM_ORDER), len(RESPONSES)

    fig, ax = plt.subplots(figsize=(13, 0.55 * n_rows + 1.8))
    ax.axis("off")
    fig.suptitle(
        "Fixed Effects (Bambi ADVI posterior, 95% CI)\n"
        "* 95% CI excludes 0   green = positive   red = negative   grey = n/a",
        fontsize=10, fontweight="bold", y=1.01,
    )

    tbl = ax.table(
        cellText=cell_text,
        cellColours=cell_color,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.2)

    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].get_text().set_color("white")
        tbl[(0, j)].get_text().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CSV dump for easy inspection
    csv_path = out.replace(".png", ".csv")
    rows_for_csv = []
    for label, row in zip(row_labels, cell_text):
        rows_for_csv.append([label] + row)
    pd.DataFrame(rows_for_csv, columns=["Predictor"] + col_labels).to_csv(csv_path, index=False)

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
