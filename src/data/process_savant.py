"""
Read data/savant_raw.csv from the R pull, rename columns to pipeline convention,
derive outcome flags, compute within-PA lag features, filter to tracked swings.

Output: data/swings_2023_2025.csv
  Same schema as old pull_data.py output plus launch_speed and launch_angle.
"""
from pathlib import Path
import pandas as pd
import numpy as np

RAW_PATH = "data/savant_raw.csv"
OUT_PATH = "data/swings_2023_2025.csv"

# ---------------------------------------------------------------------------
# Transformations (importable for tests)
# ---------------------------------------------------------------------------

SWING_DESC = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip",
    "missed_bunt", "bunt_foul_tip",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "foul", "foul_bunt",
}
WHIFF_DESC = {
    "swinging_strike", "swinging_strike_blocked", "missed_bunt", "bunt_foul_tip",
}
BIP_DESC = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}

RENAME_MAP = {
    "batter":        "batter_id",
    "pitcher":       "pitcher_id",
    "player_name":   "pitcher_full_name",   # Statcast player_name = pitcher
    "stand":         "batter_stand",
    "p_throws":      "pitcher_throws",
    "type":          "pitch_outcome",        # B=ball, S=strike, X=in-play
}


def _count_group(balls: int, strikes: int) -> str:
    if strikes == 2:
        return "two_strike"
    if balls == 3:
        return "three_ball"
    return "other"


def transform(df: pd.DataFrame, resolve_batter_names: bool = True) -> pd.DataFrame:
    """
    Apply all column renames, flag derivations, lag computations, and swing filter
    to a raw Statcast DataFrame.  Returns the filtered swings DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with Statcast column names (pre-rename).
    resolve_batter_names : bool
        If True (default), attempt pybaseball lookup for batter_full_name.
        Set False in tests to avoid network calls.

    Returns
    -------
    pd.DataFrame
        Filtered swing rows with pipeline column names.
    """
    # ── 1. Column renames ─────────────────────────────────────────────────────
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    # batter_full_name: Statcast doesn't include batter name in bulk pull.
    if "batter_full_name" not in df.columns:
        if resolve_batter_names:
            try:
                from pybaseball import playerid_reverse_lookup
                batter_ids = df["batter_id"].dropna().unique().tolist()
                lookup = playerid_reverse_lookup(batter_ids, key_type="mlbam")
                lookup["batter_full_name"] = lookup["name_first"] + " " + lookup["name_last"]
                lookup = lookup.rename(columns={"key_mlbam": "batter_id"})[
                    ["batter_id", "batter_full_name"]
                ]
                df = df.merge(lookup, on="batter_id", how="left")
            except Exception as e:
                print(
                    f"WARNING: could not resolve batter names ({e})."
                    " Setting batter_full_name = batter_id."
                )
                df["batter_full_name"] = df["batter_id"].astype(str)
        else:
            df["batter_full_name"] = df["batter_id"].astype(str)

    # ── 2. Derive outcome flags ───────────────────────────────────────────────
    df["is_swing"]   = df["description"].isin(SWING_DESC).astype(int)
    df["is_whiff"]   = df["description"].isin(WHIFF_DESC).astype(int)
    # is_contact: swung and made contact (includes fouls, BIP) — not a whiff
    df["is_contact"] = ((df["is_swing"] == 1) & (df["is_whiff"] == 0)).astype(int)
    df["is_bip"]     = df["description"].isin(BIP_DESC).astype(int)

    df["is_home_run"] = (df["events"] == "home_run").astype(int)
    df["is_triple"]   = (df["events"] == "triple").astype(int)
    df["is_double"]   = (df["events"] == "double").astype(int)
    df["is_single"]   = (df["events"] == "single").astype(int)

    df["count_group"] = [
        _count_group(b, s)
        for b, s in zip(
            df["balls"].fillna(0).astype(int),
            df["strikes"].fillna(0).astype(int),
        )
    ]

    df["is_same_side_matchup"] = (df["pitcher_throws"] == df["batter_stand"]).astype(int)

    # ── 3. Within-PA sequence lag features ───────────────────────────────────
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    g = df.groupby(["game_pk", "at_bat_number"])

    df["prev_pitch_type"]    = g["pitch_type"].shift(1)
    df["prev_release_speed"] = g["release_speed"].shift(1)
    df["prev_plate_x"]       = g["plate_x"].shift(1)
    df["prev_plate_z"]       = g["plate_z"].shift(1)
    df["velo_delta"]         = df["release_speed"] - df["prev_release_speed"]
    df["plate_z_delta"]      = df["plate_z"] - df["prev_plate_z"]
    df["plate_x_delta"]      = df["plate_x"] - df["prev_plate_x"]

    cur_outcome = pd.Series("other", index=df.index)
    cur_outcome[df["pitch_outcome"] == "B"] = "ball"
    cur_outcome[
        (df["pitch_outcome"] == "S") & (df["is_swing"] == 1) & (df["is_whiff"] == 1)
    ] = "whiff"
    cur_outcome[
        (df["pitch_outcome"] == "S") & (df["is_swing"] == 1) & (df["is_whiff"] == 0)
    ] = "foul"
    cur_outcome[
        (df["pitch_outcome"] == "S") & (df["is_swing"] == 0)
    ] = "called_strike"
    df["cur_outcome"]  = cur_outcome
    df["prev_outcome"] = g["cur_outcome"].shift(1)

    # ── 4. haa / vaa aliases ─────────────────────────────────────────────────
    # Both full column name and short alias must be present.
    if "horz_attack_angle" in df.columns and "haa" not in df.columns:
        df["haa"] = df["horz_attack_angle"]
    if "vert_attack_angle" in df.columns and "vaa" not in df.columns:
        df["vaa"] = df["vert_attack_angle"]
    if "haa" in df.columns and "horz_attack_angle" not in df.columns:
        df["horz_attack_angle"] = df["haa"]
    if "vaa" in df.columns and "vert_attack_angle" not in df.columns:
        df["vert_attack_angle"] = df["vaa"]

    # ── 5. Filter to tracked swings ───────────────────────────────────────────
    # Requires bat tracking data: bat_speed > 40 mph, VAA in plausible range.
    # NaN bat_speed / vert_attack_angle rows are dropped by this filter.
    df = df[
        (df["is_swing"] == 1)
        & df["bat_speed"].notna()
        & (df["bat_speed"] > 40)
        & df["vert_attack_angle"].notna()
        & df["vert_attack_angle"].between(-45, 45)
    ].copy()

    return df


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Reading {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"Raw shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")  # inspect on first run to verify rename map

    df = transform(df, resolve_batter_names=True)

    print(f"After swing + tracking filter: {len(df):,} rows")

    # ── Verify EV/LA columns are present ─────────────────────────────────────
    assert "launch_speed" in df.columns, \
        "launch_speed missing — check savant_raw.csv columns"
    assert "launch_angle" in df.columns, \
        "launch_angle missing — check savant_raw.csv columns"
    bip_rows = df[df["is_bip"] == 1]
    if len(bip_rows) > 0:
        ev_coverage = bip_rows["launch_speed"].notna().mean()
        print(f"EV coverage on BIP rows: {ev_coverage:.1%}")
        assert ev_coverage > 0.80, \
            f"Expected >80% EV coverage on BIP, got {ev_coverage:.1%}"

    # ── Save ──────────────────────────────────────────────────────────────────
    Path("data").mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Years: {sorted(df['game_year'].unique())}")
    print(f"Batters: {df['batter_id'].nunique():,}")
    print(f"Pitchers: {df['pitcher_id'].nunique():,}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
