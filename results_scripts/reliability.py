"""
08_reliability.py
Reliability analysis for the pitcher distortion / disruption tax metrics.

Two approaches:
  1. Split-half  — 100 random even splits per pitcher, Spearman-Brown corrected r
  2. Year-over-year — mean tax per pitcher per season, Pearson r across seasons

Run:
    .venv\\Scripts\\python.exe 08_reliability.py

Outputs:
    results/figures/08_reliability.png
    results/figures/08_reliability.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Path("results/figures").mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

METRICS = ["disruption_tax", "adjusted_disruption_tax", "distortion_tax", "selection_tax"]
METRIC_LABELS = {
    "disruption_tax":          "xRV Residual (Actual - Intended)",
    "adjusted_disruption_tax": "Adjusted Disruption Tax",
    "distortion_tax":          "Distortion Tax",
    "selection_tax":           "Selection Tax",
    "distortion_share":        "Distortion Share",
}
MIN_SWINGS     = 50   # per half (split-half) or per season (YoY)
N_SPLITS       = 100
SEED           = 42


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    xrv = pd.read_parquet("results/xrv_causal.parquet")
    sw  = (
        pd.read_parquet(
            "data/swings_precommit.parquet",
            columns=["game_pk", "at_bat_number", "pitch_number", "game_year"],
        )
        .drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    )
    df = xrv.merge(sw, on=["game_pk", "at_bat_number", "pitch_number"], how="left")
    return df.dropna(subset=METRICS + ["game_year", "pitcher_id"])


# ── Split-half reliability ─────────────────────────────────────────────────────

def _icc21(y):
    """ICC(2,1) — two-way random effects, absolute agreement, single measures.

    y : (n_subjects, k_raters) array
    Returns the ICC point estimate. Returns nan if computation is degenerate.

    Formula (Shrout & Fleiss 1979, Case 2):
      ICC(2,1) = (MSB - MSE) / (MSB + (k-1)*MSE + k/n * (MSR - MSE))
    """
    y = np.asarray(y, dtype=float)
    n, k = y.shape
    if n < 3 or k < 2:
        return np.nan

    gm       = y.mean()
    row_mean = y.mean(axis=1)
    col_mean = y.mean(axis=0)

    ssb = k * ((row_mean - gm) ** 2).sum()
    ssr = n * ((col_mean - gm) ** 2).sum()
    sst = ((y - gm) ** 2).sum()
    sse = sst - ssb - ssr

    msb = ssb / (n - 1)
    msr = ssr / (k - 1)
    mse = sse / ((n - 1) * (k - 1))

    denom = msb + (k - 1) * mse + k / n * (msr - mse)
    if denom <= 0:
        return np.nan
    return float((msb - mse) / denom)


def split_half(df, metric, min_swings=MIN_SWINGS, n_splits=N_SPLITS, seed=SEED):
    """Spearman-Brown corrected split-half r and ICC(2,1), averaged over n_splits random splits.

    Only pitchers with >= 2*min_swings observations are included so each half
    meets the minimum threshold independently.
    """
    rng    = np.random.default_rng(seed)
    valid  = df.groupby("pitcher_id")[metric].transform("size") >= min_swings * 2
    pool   = df.loc[valid, ["pitcher_id", metric]].copy()
    pids   = pool["pitcher_id"].unique()

    if len(pids) < 10:
        return np.nan, np.nan, np.nan, 0

    rs, iccs = [], []
    for _ in range(n_splits):
        h1_means, h2_means = [], []
        for pid in pids:
            g   = pool.loc[pool["pitcher_id"] == pid, metric].values
            idx = rng.permutation(len(g))
            half = len(g) // 2
            h1_means.append(g[idx[:half]].mean())
            h2_means.append(g[idx[half:]].mean())

        r_raw = np.corrcoef(h1_means, h2_means)[0, 1]
        rs.append(2 * r_raw / (1 + r_raw))   # Spearman-Brown
        iccs.append(_icc21(np.column_stack([h1_means, h2_means])))

    return float(np.mean(rs)), float(np.std(rs)), float(np.nanmean(iccs)), len(pids)


# ── Year-over-year reliability ─────────────────────────────────────────────────

def yoy_corr(df, year_a, year_b, metric, min_swings=MIN_SWINGS):
    """Pearson r and ICC(2,1) of per-pitcher season means between year_a and year_b."""
    def season_means(yr):
        g = df[df["game_year"] == yr].groupby("pitcher_id")[metric]
        agg = g.agg(mean="mean", n="size")
        return agg[agg["n"] >= min_swings]["mean"]

    a = season_means(year_a)
    b = season_means(year_b)
    both = a.index.intersection(b.index)
    n = len(both)
    if n < 10:
        return np.nan, np.nan, 0
    r   = a.loc[both].corr(b.loc[both])
    icc = _icc21(np.column_stack([a.loc[both].values, b.loc[both].values]))
    return float(r), float(icc), n


# ── Build results table ────────────────────────────────────────────────────────

def build_table(df):
    rows = []
    for metric in METRICS:
        sh_r, sh_sd, sh_icc, sh_n          = split_half(df, metric)
        yoy_23_24_r, yoy_23_24_icc, n_2324 = yoy_corr(df, 2023, 2024, metric)
        yoy_24_25_r, yoy_24_25_icc, n_2425 = yoy_corr(df, 2024, 2025, metric)
        rows.append({
            "metric":         metric,
            "sh_r":           sh_r,
            "sh_icc":         sh_icc,
            "yoy_23_24_r":    yoy_23_24_r,
            "yoy_23_24_icc":  yoy_23_24_icc,
            "yoy_24_25_r":    yoy_24_25_r,
            "yoy_24_25_icc":  yoy_24_25_icc,
        })
    return pd.DataFrame(rows).set_index("metric")


# ── Render table ───────────────────────────────────────────────────────────────

def _fmt_r(r):
    return "—" if np.isnan(r) else f"{r:.3f}"


def _r_color(r):
    if np.isnan(r):
        return "#f0f0f0"
    if r >= 0.7:
        return "#c8e6c9"   # strong  → green
    if r >= 0.5:
        return "#fff9c4"   # moderate → yellow
    if r >= 0.3:
        return "#ffe0b2"   # weak    → orange
    return "#ffcdd2"       # poor    → red


COL_HEADERS = [
    "Split-half r\n(SB corrected, 100 splits)",
    "Split-half\nICC(2,1)",
    "YoY 2023→2024\nPearson r",
    "YoY 2023→2024\nICC(2,1)",
    "YoY 2024→2025\nPearson r",
    "YoY 2024→2025\nICC(2,1)",
]

HEADER_BG = "#2c3e50"
HEADER_FG = "white"


def render_table(tbl, out="results/figures/08_reliability.png"):
    row_labels = [METRIC_LABELS[m] for m in tbl.index]
    cell_text, cell_color = [], []

    for metric, row in tbl.iterrows():
        cell_text.append([
            _fmt_r(row["sh_r"]),
            _fmt_r(row["sh_icc"]),
            _fmt_r(row["yoy_23_24_r"]),
            _fmt_r(row["yoy_23_24_icc"]),
            _fmt_r(row["yoy_24_25_r"]),
            _fmt_r(row["yoy_24_25_icc"]),
        ])
        cell_color.append([
            _r_color(row["sh_r"]),
            _r_color(row["sh_icc"]),
            _r_color(row["yoy_23_24_r"]),
            _r_color(row["yoy_23_24_icc"]),
            _r_color(row["yoy_24_25_r"]),
            _r_color(row["yoy_24_25_icc"]),
        ])

    fig, ax = plt.subplots(figsize=(16, 3.2))
    ax.axis("off")
    fig.suptitle(
        f"Pitcher Distortion / Disruption Tax — Reliability Analysis\n"
        f"(min {MIN_SWINGS} swings per pitcher per half/season)",
        fontsize=11, fontweight="bold", y=1.03,
    )

    t = ax.table(
        cellText=cell_text,
        cellColours=cell_color,
        rowLabels=row_labels,
        colLabels=COL_HEADERS,
        cellLoc="center",
        loc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 2.6)

    for j in range(len(COL_HEADERS)):
        t[(0, j)].set_facecolor(HEADER_BG)
        t[(0, j)].get_text().set_color(HEADER_FG)
        t[(0, j)].get_text().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} swings  |  {df['pitcher_id'].nunique():,} pitchers  "
          f"|  years {sorted(df['game_year'].unique())}")

    print("Computing reliability...")
    tbl = build_table(df)

    csv_out = "results/figures/08_reliability.csv"
    tbl.to_csv(csv_out)
    print(f"Saved {csv_out}")
    print(tbl.to_string(float_format="{:.3f}".format))

    render_table(tbl)
    print("Done.")
