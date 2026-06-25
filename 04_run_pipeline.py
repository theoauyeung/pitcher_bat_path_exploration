"""
Full pipeline: Phase A (batter intention) -> Phase B (causal mediation) -> disruption tax.

Expects data/swings_precommit.parquet (run 01_pull_data.py then 02_precommit_split.py first).

Outputs:
  results/xrv_causal.parquet     — per-swing disruption, distortion, selection tax
  results/xrv_causal.csv         — same, CSV for quick inspection
  results/distortion_pitcher.csv  — distortion tax aggregated by pitcher (≥50 swings)
  results/distortion_batter.csv   — distortion tax aggregated by batter (≥50 swings)
  models/intended_df.parquet      — Phase A intended swing shape per swing (cache)
  models/causal_models.joblib     — Phase B fitted models

Run:
    python 04_run_pipeline.py
"""

import argparse
import os
import re
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from importlib import import_module as _im
_A  = _im("02_intention_model")
_B  = _im("03_causal_models")


def get_secret(name):
    val = os.environ.get(name)
    if val:
        return val
    env_file = Path.home() / ".claude" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(rf"^\s*{re.escape(name)}\s*=\s*(.+)$", line)
            if m:
                return m.group(1).strip()
    return None


# ── Data loading ────────────────────────────────────────────────────────────────

def load_swings(path="data/swings_precommit.parquet"):
    """Load precomputed swings with trajectory and pre/post-commit columns.

    Raises FileNotFoundError with instructions if the parquet is missing —
    run 01_pull_data.py then 02_precommit_split.py to build it.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found. "
            "Run 01_pull_data.py then 02_precommit_split.py to build the parquet first."
        )
    return pd.read_parquet(p)


def competitive_swings(df):
    """Tracked, full-effort swings with valid angular metrics."""
    return df[
        (df["is_swing"] == 1) &
        df["bat_speed"].notna() &
        (df["bat_speed"] >= 50) &
        df["vert_attack_angle"].notna()
    ].copy()


# ── Run value inputs ────────────────────────────────────────────────────────────

def add_xwoba(df, lw_path="results/linear_weights.csv"):
    """Add xwoba column: realized linear-weight run value per batted ball.

    Defined only for balls in play (is_bip == 1); fouls and whiffs stay NaN and
    are naturally excluded from fit_outcome_models' xwoba regression via dropna().

    Hit outcomes are mutually exclusive and tested in descending order of value
    to handle any edge-case double-coding in the source data.
    """
    lw = pd.read_csv(lw_path).set_index("outcome_type")["lw"]

    out = df.copy()
    xwoba = pd.Series(np.nan, index=df.index)

    bip = df["is_bip"] == 1
    xwoba.loc[bip & (df["is_home_run"] == 1)] = lw["home_run"]
    xwoba.loc[bip & (df["is_triple"]   == 1) & (df["is_home_run"] == 0)] = lw["triple"]
    xwoba.loc[bip & (df["is_double"]   == 1) & (df["is_triple"]   == 0)
                                             & (df["is_home_run"] == 0)] = lw["double"]
    xwoba.loc[bip & (df["is_single"]   == 1) & (df["is_double"]   == 0)
                                             & (df["is_triple"]   == 0)
                                             & (df["is_home_run"] == 0)] = lw["single"]
    # remaining BIP that aren't hits: outs in play
    out_mask = bip & (df["is_home_run"] == 0) & (df["is_triple"] == 0) & \
               (df["is_double"] == 0) & (df["is_single"] == 0)
    xwoba.loc[out_mask] = lw["out_in_play"]

    out["xwoba"] = xwoba
    return out


def whiff_rv_from_count_values(path="results/count_values.csv"):
    """Build whiff run-value table from count-state expected run values.

    Whiff at (b, s < 2): count advances to (b, s+1).
      delta = ERV(b, s+1) − ERV(b, s)

    Whiff at (b, 2): strikeout — PA ends with no further run value.
      delta = 0 − ERV(b, 2)

    Returns dict {(balls, strikes): delta_rv, "default": mean_delta_rv}.
    """
    cv = pd.read_csv(path).set_index(["balls", "strikes"])["expected_run_value"].to_dict()

    table = {}
    for (b, s), rv in cv.items():
        table[(b, s)] = (cv.get((b, s + 1), 0.0) - rv) if s < 2 else (0.0 - rv)

    table["default"] = np.mean(list(table.values()))
    return table


def foul_rv_from_count_values(path="results/count_values.csv"):
    """Build foul-ball run-value table from count-state expected run values.

    Foul at (b, s < 2): count advances to (b, s+1), same effect as a whiff.
      delta = ERV(b, s+1) − ERV(b, s)

    Foul at (b, 2): count stays at (b, 2), PA continues — no run-value change.
      delta = 0.0

    Returns dict {(balls, strikes): delta_rv, "default": mean_delta_rv}.
    """
    cv = pd.read_csv(path).set_index(["balls", "strikes"])["expected_run_value"].to_dict()

    table = {}
    for (b, s), rv in cv.items():
        table[(b, s)] = (cv.get((b, s + 1), 0.0) - rv) if s < 2 else 0.0

    table["default"] = np.mean(list(table.values()))
    return table


# ── Main pipeline ───────────────────────────────────────────────────────────────

def run(
    data_path="data/swings_precommit.parquet",
    lw_path="results/linear_weights.csv",
    count_values_path="results/count_values.csv",
    commit_ms=150,
    n_subsample=75_000,
    draws=1000,
    tune=1000,
    chains=4,
    max_treedepth=15,
    skip_phase_a=False,
    intention_cache="models/intended_df.parquet",
    method="mcmc",
):
    """Run the full Phase A -> Phase B pipeline.

    commit_ms:       commit time for pre/post-commit split (default 150ms; robustness
                     grid runs 125/150/175/200 separately)
    n_subsample:     rows for Phase A MCMC fitting. Default 75k — keeps the pitcher_id
                     contrast matrix (n_rows × n_pitchers) under ~500 MB. Use None for
                     production, but ensure ≥16 GB RAM free.
    draws/tune:      MCMC settings. tune=1000 minimum for this model; 500 causes
                     divergences and R-hat > 1.01.
    max_treedepth:   NUTS tree depth cap (default 15). The simplified batter RE formula
                     (intercept + strikes only) should rarely hit depth 10, but 15 gives
                     headroom without exploding runtime.
    """
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # ── 1. Load and prep ──────────────────────────────────────────────────────────
    df = load_swings(data_path)
    swings = competitive_swings(df)
    swings = add_xwoba(swings, lw_path)
    print(f"Competitive swings: {len(swings):,}")

    # ── 2. Phase A: intended swing model ─────────────────────────────────────────
    # Cache is the intended_df parquet (plain DataFrame), not the Bambi model objects.
    # Bambi Model objects cannot be pickled on Python 3.14 (FrameLocalsProxy in formulae.Environment).
    if skip_phase_a:
        cache = Path(intention_cache)
        if not cache.exists():
            raise FileNotFoundError(
                f"--skip-phase-a requires {intention_cache} but it was not found. "
                "Run without --skip-phase-a first to build it."
            )
        print(f"\nLoading Phase A intended_df from cache: {intention_cache}")
        intended_df = pd.read_parquet(intention_cache)
    else:
        print(f"\nFitting Phase A intention model (method={method})...")
        intention = _A.fit(
            swings,
            n_subsample=n_subsample,
            draws=draws,
            tune=tune,
            chains=chains,
            max_treedepth=max_treedepth,
            method=method,
        )
        intended_df = _A.predict_intended(intention, swings)
        intended_df.to_parquet(intention_cache, index=True)
        # Save idata + training data for downstream coefficient extraction.
        # Bambi Model objects can't be pickled (Python 3.14 FrameLocalsProxy), so only
        # idata and data are written — sufficient for _extract_fixed_effects in 07.py.
        joblib.dump(
            {"idata": intention["idata"], "data": intention["data"]},
            "models/intention_result.joblib",
        )
        print(f"Phase A done -> {intention_cache}, models/intention_result.joblib")

    # swing_deviations: adds {resp}_dev = realized − intended columns
    swings = _A.swing_deviations(swings, intended_df)
    for col in intended_df.columns:
        swings[col] = intended_df[col].values

    # ── 3. Phase B: mediator models ───────────────────────────────────────────────
    # Response:  {angular_metric}_dev  (realized − intended)
    # Treatment: pc{commit_ms}_dev_x, pc{commit_ms}_dev_z  (post-commit movement)
    # Controls:  pre-commit projection, pitch speed, count, timing
    print("\nFitting mediator models...")
    mediator_models = _B.fit_mediator_models(swings, commit_ms=commit_ms)
    print("Mediator models done.")

    # ── 4. Phase B: outcome models ────────────────────────────────────────────────
    # bip_model:   logistic P(BIP) ~ angular deviations + location + count
    # foul_model:  logistic P(foul | not BIP) ~ same predictors, non-BIP swings only
    # xwoba_model: OLS xwOBAcon ~ same, BIP only
    # whiff_rv/foul_rv: count-transition values from count_values.csv
    print("\nFitting outcome models...")
    bip_model, foul_model, xwoba_model, _ = _B.fit_outcome_models(swings)
    whiff_rv = whiff_rv_from_count_values(count_values_path)
    foul_rv  = foul_rv_from_count_values(count_values_path)
    print("Outcome models done.")

    # ── 5. Disruption tax + distortion/selection split ────────────────────────────
    print("\nComputing disruption tax...")
    results = _B.disruption_tax_split(
        swings,
        bip_model,
        foul_model,
        xwoba_model,
        whiff_rv,
        foul_rv,
        mediator_models,
        commit_ms=commit_ms,
    )

    # ── 6. Analytical indirect effect ────────────────────────────────────────────
    ie = _B.indirect_effect(
        mediator_models, bip_model, foul_model, xwoba_model,
        whiff_rv, foul_rv, swings, commit_ms=commit_ms,
    )
    print("\nIndirect effects (post-commit movement -> xRV via swing distortion):")
    print(ie.to_string(float_format="{:.5f}".format))

    # ── 7. Validation controls ────────────────────────────────────────────────────
    dev_total_col = f"pc{commit_ms}_dev_total"
    _B.negative_control_check(results, dev_total_col=dev_total_col)
    _B.positive_control_check(results, dev_total_col=dev_total_col)

    # ── 8. Save ───────────────────────────────────────────────────────────────────
    id_cols = ["batter_id", "pitcher_id", "game_pk", "at_bat_number", "pitch_number",
               "balls", "strikes", "pitch_type"]
    tax_cols = ["disruption_tax", "distortion_tax", "selection_tax", "distortion_share"]
    save_cols = [c for c in id_cols + tax_cols if c in results.columns]

    results[save_cols].to_parquet("results/xrv_causal.parquet", index=False)
    results[save_cols].to_csv("results/xrv_causal.csv", index=False)

    for group_col, out_path in [("pitcher_id", "results/distortion_pitcher.csv"),
                                 ("batter_id",  "results/distortion_batter.csv")]:
        agg = (
            results[results["distortion_tax"].notna()]
            .groupby(group_col)
            .agg(
                n_swings=("disruption_tax", "size"),
                mean_disruption_tax=("disruption_tax", "mean"),
                mean_distortion_tax=("distortion_tax", "mean"),
                mean_selection_tax=("selection_tax", "mean"),
                mean_distortion_share=("distortion_share", "mean"),
            )
            .query("n_swings >= 50")
            .sort_values("mean_distortion_tax")
        )
        agg.to_csv(out_path)

    joblib.dump(
        {
            "mediator_models": mediator_models,
            "bip_model":       bip_model,
            "foul_model":      foul_model,
            "xwoba_model":     xwoba_model,
            "whiff_rv":        whiff_rv,
            "foul_rv":         foul_rv,
            "indirect_effects": ie,
        },
        "models/causal_models.joblib",
    )

    print("\nSaved:")
    print("  results/xrv_causal.parquet")
    print("  results/distortion_pitcher.csv")
    print("  results/distortion_batter.csv")
    print("  models/intention_result.joblib")
    print("  models/causal_models.joblib")

    return results, {
        "mediator_models": mediator_models,
        "bip_model":       bip_model,
        "foul_model":      foul_model,
        "xwoba_model":     xwoba_model,
        "whiff_rv":        whiff_rv,
        "foul_rv":         foul_rv,
        "indirect_effects": ie,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-phase-a", action="store_true",
                   help="Load Phase A from models/intended_df.parquet instead of refitting.")
    p.add_argument("--intention-cache", default="models/intended_df.parquet",
                   help="Path to intended_df parquet cache (default: models/intended_df.parquet).")
    p.add_argument("--method", default="mcmc", choices=["mcmc", "vi"],
                   help="Phase A inference method: 'mcmc' (default, ~hours) or 'vi' (ADVI, ~2 min).")
    p.add_argument("--n-subsample", type=int, default=75_000)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--commit-ms", type=int, default=150)
    args = p.parse_args()
    run(
        skip_phase_a=args.skip_phase_a,
        intention_cache=args.intention_cache,
        method=args.method,
        n_subsample=args.n_subsample,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        commit_ms=args.commit_ms,
    )
  