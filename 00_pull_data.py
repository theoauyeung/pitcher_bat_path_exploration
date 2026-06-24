"""
Pull 2023-2025 MLB pitches, compute within-PA sequence lags, then filter to
tracked swings and save as a single CSV.

Why pull ALL pitches (not just swings):
  Sequence features (prev pitch type, velo delta, location shift, prior outcome)
  require knowing what the PREVIOUS pitch was — which may have been a take, ball,
  or called strike that never shows up in a swing-only pull. Pulling everything
  first lets us compute the correct lag for each swing event.

Output: data/swings_2023_2025.csv
  Bat-tracking coverage starts 2H 2023; 2023 rows will have NaN for bat-tracking
  columns on pitches before that rollout. Filter downstream as needed.

Columns added vs earlier versions:
  x0/y0/z0, vx0/vy0/vz0, ax/ay/az  — 9-param Statcast trajectory at y=50 ft;
                                       required for the pre/post-commit split
  sz_top, sz_bot                     — batter-specific strike zone bounds
  delta_run_exp                      — run expectancy change per pitch (xRV)
  outs_when_up, inning               — game-state context

Swing quality filters applied:
  bat_speed > 40 mph         — removes sub-human / tracker garbage
  vert_attack_angle ±45 deg  — removes 0.1% extreme outliers

Run:
    python 01_pull_data.py
"""

import os
import re
from pathlib import Path

import mysql.connector
import pandas as pd


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


# ── 1. Connect ─────────────────────────────────────────────────────────────────

print("Connecting to mlb_db…")
conn = mysql.connector.connect(
    host=get_secret("BIOMECH_DB_HOST") or "10.200.200.107",
    port=int(get_secret("BIOMECH_DB_PORT") or 3306),
    user=get_secret("BIOMECH_DB_USER") or "analystreadonly",
    password=get_secret("BIOMECH_DB_PASS") or "IZTcommandRead99",
    database="mlb_db",
    connection_timeout=120,
)

# ── 2. Pull ALL pitches (swings + takes + balls + called strikes) ──────────────
# We need every pitch in the sequence so we can correctly identify what came
# before each swing. If we pulled swings only, a previous-pitch take would be
# invisible and the lag would jump across PAs.

print("Pulling all 2023-2025 MLB pitches (~2.1M rows, takes 3-4 min)…")
df = pd.read_sql("""
    SELECT
        r.game_pk,
        r.game_date,
        r.game_year,
        r.at_bat_number,
        r.pitch_number,
        r.batter_id,
        r.batter_full_name,
        r.batter_stand,
        r.pitcher_id,
        r.pitcher_full_name,
        r.pitcher_throws,
        r.pitch_type,
        r.release_speed,
        r.pfx_x,
        r.pfx_z,
        r.plate_x,
        r.plate_z,
        r.sz_top,
        r.sz_bot,
        r.release_pos_x,
        r.release_pos_y,
        r.release_pos_z,
        r.release_extension,
        r.arm_angle,
        r.x0,
        r.y0,
        r.z0,
        r.vx0,
        r.vy0,
        r.vz0,
        r.ax,
        r.ay,
        r.az,
        r.balls,
        r.strikes,
        r.outs_when_up,
        r.inning,
        r.pitch_outcome,
        r.vert_attack_angle,
        r.horz_attack_angle,
        r.bat_speed,
        r.swing_length,
        r.swing_path_tilt,
        r.ball_bat_intercept_y,
        r.ball_bat_miss,
        r.offset_y_ms,
        r.offset_z_in,
        r.offset_x_in,
        d.is_swing,
        d.is_whiff,
        d.is_contact,
        d.is_bip,
        d.count_group,
        d.is_same_side_matchup,
        d.is_single,
        d.is_double,
        d.is_triple,
        d.is_home_run,
        d.delta_run_exp,
        c.haa,
        c.vaa
    FROM pbp_raw r
    JOIN pbp_descriptions d ON r.play_id = d.play_id
    LEFT JOIN pbp_calculations c ON r.play_id = c.play_id
    WHERE r.level_id = 1
      AND r.game_type = 'R'
      AND r.game_year IN (2023, 2024, 2025)
    ORDER BY r.game_pk, r.at_bat_number, r.pitch_number
""", conn)
conn.close()
print(f"Pulled {len(df):,} total pitches (2023 bat-tracking coverage starts 2H; NaNs expected in 2023 bat-tracking cols)")

# ── 3. Compute within-PA sequence lag features ────────────────────────────────
# For each pitch, record what the previous pitch in the same PA was.
# Pitches that are first in their PA will get NaN — that is correct and expected.

print("Computing sequence lags…")

# group by plate appearance (each game × at-bat is one PA)
g = df.groupby(["game_pk", "at_bat_number"])

# previous pitch type and location
df["prev_pitch_type"]    = g["pitch_type"].shift(1)
df["prev_release_speed"] = g["release_speed"].shift(1)
df["prev_plate_x"]       = g["plate_x"].shift(1)
df["prev_plate_z"]       = g["plate_z"].shift(1)

# how much faster / higher / more inside the current pitch is vs the previous one
df["velo_delta"]    = df["release_speed"]  - df["prev_release_speed"]
df["plate_z_delta"] = df["plate_z"]        - df["prev_plate_z"]
df["plate_x_delta"] = df["plate_x"]        - df["prev_plate_x"]

# what happened on the previous pitch (this helps encode what the batter just saw)
# encode the CURRENT pitch outcome first, then shift it to get prev_outcome
cur_outcome = pd.Series("other", index=df.index)
cur_outcome[df["pitch_outcome"] == "B"] = "ball"
cur_outcome[(df["pitch_outcome"] == "S") & (df["is_swing"] == 1) & (df["is_whiff"] == 1)] = "whiff"
cur_outcome[(df["pitch_outcome"] == "S") & (df["is_swing"] == 1) & (df["is_whiff"] == 0)] = "foul"
cur_outcome[(df["pitch_outcome"] == "S") & (df["is_swing"] == 0)]                          = "called_strike"
df["cur_outcome"] = cur_outcome
df["prev_outcome"] = g["cur_outcome"].shift(1)

# ── 4. Filter to tracked swing events ─────────────────────────────────────────
# Now that lag features are computed, drop everything except swings with
# bat-tracking data. Non-swing rows served their purpose for the lags above.

df = df[
    (df["is_swing"] == 1) &           # must be a swing
    df["bat_speed"].notna() &          # must have bat tracking
    (df["bat_speed"] > 40) &           # remove sub-human / garbage readings
    (df["vert_attack_angle"].between(-45, 45))  # remove 0.1% extreme outliers
].copy()

print(f"After swing + tracking filter: {len(df):,} rows")

# ── 4. Save ───────────────────────────────────────────────────────────────────

Path("data").mkdir(exist_ok=True)
out = Path("data/swings_2023_2025.csv")
df.to_csv(out, index=False)

print(f"\nSaved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
print(f"Shape: {df.shape}")
print(f"Years: {sorted(df['game_year'].unique())}")
print(f"Batters: {df['batter_id'].nunique():,}")
print(f"Pitchers: {df['pitcher_id'].nunique():,}")
print(f"Whiffs: {df['is_whiff'].sum():,} ({100*df['is_whiff'].mean():.1f}%)")
print(f"Has prior pitch context (prev_pitch_type not NaN): "
      f"{df['prev_pitch_type'].notna().sum():,} ({100*df['prev_pitch_type'].notna().mean():.1f}%)")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
