"""
Build the run-value foundation from 2024-2025 Statcast pitch-by-pitch data.

Three outputs:

  1. RE24 matrix — expected runs from each of the 24 base × out states
     Built from scratch: for each plate appearance start, compute runs scored
     by the batting team for the remainder of that half-inning.
     RE24[base_state][outs] = mean(runs_scored_in_rest_of_inning)

  2. Count-state run values — how valuable is each ball-strike count to the batter?
     For every pitch, we know the count (balls, strikes).
     Count value = expected run value of the PA from this count forward.
     Computed as the mean of (sum of remaining delta_run_exp within the PA)
     for all pitches in that count.

  3. Linear weights — run value of each PA outcome type
     For each PA-ending event (1B, 2B, 3B, HR, BB, HBP, K, out):
     LW = RE_after + runs_scored_on_play - RE_before
     where RE_before and RE_after come from the RE24 matrix.
     Also computes count-conditional linear weights:
     how much is a walk worth from a 3-0 count vs a 0-2 count?

Works entirely in memory from DB — no parquets written.

Run:
    python run_values.py
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import mysql.connector
import pathlib

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

# ── 1. Pull all 2024-2025 MLB regular season pitches ──────────────────────────
# Not just swings — every pitch, including takes and balls, because we need
# full PA sequences for RE24 and count transitions.

QUERY = """
    SELECT
        r.game_pk,
        r.game_year,
        r.inning,
        r.inning_topbot,
        r.at_bat_number,
        r.pitch_number,
        r.batter_id,
        r.pitcher_id,
        r.balls,
        r.strikes,
        r.outs_when_up,
        r.pitch_outcome,
        r.pa_outcome,
        r.pa_outcome_explanation,
        d.bat_score,
        d.post_bat_score,
        (d.on_1b_id IS NOT NULL) AS on_1b,
        (d.on_2b_id IS NOT NULL) AS on_2b,
        (d.on_3b_id IS NOT NULL) AS on_3b,
        d.is_first_pitch_of_pa,
        d.is_last_pitch_of_pa,
        d.is_swing,
        d.is_whiff,
        d.is_called_strike,
        d.is_contact,
        d.is_bip,
        d.is_hbp,
        d.is_strikeout,
        d.is_hit,
        d.is_single,
        d.is_double,
        d.is_triple,
        d.is_home_run,
        d.delta_run_exp,
        d.count_group,
        d.pitch_group
    FROM pbp_raw r
    JOIN pbp_descriptions d ON r.play_id = d.play_id
    WHERE r.level_id = 1
      AND r.game_type = 'R'
      AND r.game_year IN (2024, 2025)
"""

# Fetch in 100k-row chunks using cursor.fetchmany() to avoid connection timeout
# on large result sets (pd.read_sql calls fetchall() which drops the connection
# before all rows arrive on queries this size).
host = get_secret("BIOMECH_DB_HOST")
user = get_secret("BIOMECH_DB_USER")
password = get_secret("BIOMECH_DB_PASS")
if not all([host, user, password]):
    missing = [k for k, v in {"BIOMECH_DB_HOST": host, "BIOMECH_DB_USER": user,
                               "BIOMECH_DB_PASS": password}.items() if not v]
    raise RuntimeError(f"Missing DB credentials in ~/.claude/.env: {', '.join(missing)}")

print("Connecting to mlb_db…")
conn = mysql.connector.connect(
    host=host,
    port=int(get_secret("BIOMECH_DB_PORT") or 3306),
    user=user,
    password=password,
    database="mlb_db",
    connection_timeout=300,
)
cursor = conn.cursor()
cursor.execute("SET SESSION net_read_timeout=600")
cursor.execute("SET SESSION net_write_timeout=600")
cursor.execute("SET SESSION wait_timeout=600")

print("Pulling 2024-2025 pitch-by-pitch data (chunked fetch)…")
cursor.execute(QUERY)
cols = [d[0] for d in cursor.description]

chunks = []
while True:
    rows = cursor.fetchmany(100_000)
    if not rows:
        break
    chunks.append(pd.DataFrame(rows, columns=cols))
    print(f"  fetched {sum(len(c) for c in chunks):,} rows…", end="\r")

cursor.close()
conn.close()
df = pd.concat(chunks, ignore_index=True)
# sort by game, at_bat, pitch order
df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

print(f"Pulled {len(df):,} pitches  ({df['game_pk'].nunique():,} games)")

# cast base occupation columns to bool/int (MySQL returns them as integers 0/1)
for col in ["on_1b", "on_2b", "on_3b",
            "is_first_pitch_of_pa", "is_last_pitch_of_pa",
            "is_swing", "is_whiff", "is_called_strike", "is_contact",
            "is_bip", "is_hbp", "is_strikeout", "is_hit",
            "is_single", "is_double", "is_triple", "is_home_run"]:
    df[col] = df[col].astype(float).fillna(0).astype(int)

# build base-state string: "000" = bases empty, "100" = runner on 1st only, etc.
df["base_state"] = (df["on_1b"].astype(str) +
                    df["on_2b"].astype(str) +
                    df["on_3b"].astype(str))

BASE_STATE_LABELS = {
    "000": "---", "100": "1--", "010": "-2-", "001": "--3",
    "110": "12-", "101": "1-3", "011": "-23", "111": "123",
}
df["base_label"] = df["base_state"].map(BASE_STATE_LABELS)

# ── 2. RE24 matrix ─────────────────────────────────────────────────────────────
#
# For each plate appearance START (first pitch of PA), compute:
#   runs_in_remainder = max(post_bat_score in this half-inning) - bat_score now
#
# Then: RE24[base_state][outs] = mean(runs_in_remainder)

# maximum batting-team score achieved in each half-inning
# (= final score for that team at end of that inning)
half_inning_final = (
    df.groupby(["game_pk", "inning", "inning_topbot"])["post_bat_score"]
    .max()
    .reset_index()
    .rename(columns={"post_bat_score": "half_inn_final_score"})
)

# join to the first pitch of each PA
pa_starts = df[df["is_first_pitch_of_pa"] == 1].copy()
pa_starts = pa_starts.merge(half_inning_final,
                             on=["game_pk", "inning", "inning_topbot"],
                             how="left")

# runs scored by the batting team from this PA start to end of half-inning
pa_starts["runs_in_remainder"] = (
    pa_starts["half_inn_final_score"] - pa_starts["bat_score"]
).clip(lower=0)   # clip at 0 to handle any score-tracking edge cases

# RE24: mean runs in remainder by (base_state, outs_when_up)
re24 = (pa_starts
        .groupby(["base_state", "outs_when_up"])["runs_in_remainder"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "re24", "count": "n_pa"})
        .reset_index())

re24["base_label"] = re24["base_state"].map(BASE_STATE_LABELS)
re24 = re24.sort_values(["outs_when_up", "base_state"])

print(f"\n{'='*55}")
print("RE24 Matrix — Expected Runs from Each Base-Out State")
print(f"{'='*55}")
re24_pivot = re24.pivot(index="base_label", columns="outs_when_up", values="re24")
re24_pivot.columns = ["0 outs", "1 out", "2 outs"]
re24_pivot = re24_pivot.reindex(["---","1--","-2-","--3","12-","1-3","-23","123"])
print(re24_pivot.round(3).to_string())

# also store as a lookup dict for linear weights
re24_lookup = re24.set_index(["base_state", "outs_when_up"])["re24"].to_dict()

# ── 3. Count-state run values ─────────────────────────────────────────────────
#
# For each pitch at count (balls, strikes), what is the expected run value
# remaining in the PA?
#
# Method: for each pitch, sum all delta_run_exp from THIS PITCH to the end of
# the PA. Call this "runs_from_here." Average over all pitches at each count.
#
# This answers: "if the PA is currently in count (b, s), what run value
# is the batting team expected to accrue from this point forward?"

# ── Count-state run values (fully empirical) ──────────────────────────────────
# For every pitch in a PA, pa_total_dre is the total run value that PA produced
# (sum of all delta_run_exp within the PA = RE24 outcome value).
# Count-state expected run value = mean(pa_total_dre) for all pitches at (b,s).
# No hardcoded weights — the outcome distribution is read directly from the data.

df_valid = df[df["delta_run_exp"].notna()].copy()
df_valid = df_valid.sort_values(["game_pk", "at_bat_number", "pitch_number"])

# sum delta_run_exp across each PA to get its total empirical run value
pa_total_dre = (df_valid.groupby(["game_pk", "at_bat_number"])["delta_run_exp"]
                .sum()
                .reset_index()
                .rename(columns={"delta_run_exp": "pa_total_dre"}))
df_valid = df_valid.merge(pa_total_dre, on=["game_pk", "at_bat_number"], how="left")

# classify outcome type on the last pitch of each PA
def classify_pa_outcome(row):
    if row["is_home_run"]:                                   return "home_run"
    if row["is_triple"]:                                     return "triple"
    if row["is_double"]:                                     return "double"
    if row["is_single"]:                                     return "single"
    if row["is_hbp"]:                                        return "hbp"
    if row["pitch_outcome"] == "B" and not row["is_hbp"]:   return "walk"
    if row["is_strikeout"]:                                  return "strikeout"
    if row["is_bip"] and not row["is_hit"]:                  return "out_in_play"
    return "other"

pa_outcomes = df_valid[df_valid["is_last_pitch_of_pa"] == 1].copy()
pa_outcomes["outcome_type"] = pa_outcomes.apply(classify_pa_outcome, axis=1)

# empirical linear weights: mean pa_total_dre by outcome type
# (derived from data, not hardcoded)
empirical_lw = pa_outcomes.groupby("outcome_type")["pa_total_dre"].mean()
print(f"\nEmpirical linear weights (mean PA run value by outcome):")
print(empirical_lw.sort_values(ascending=False).round(4).to_string())

# join pa run value back to all pitches in those PAs
df_valid = df_valid.join(
    pa_outcomes.set_index(["game_pk", "at_bat_number"])[["outcome_type", "pa_total_dre"]]
    .rename(columns={"pa_total_dre": "pa_run_value", "outcome_type": "pa_outcome_type"}),
    on=["game_pk", "at_bat_number"]
)

# count-state expected run value: mean pa run value for pitches at each count
count_values = (df_valid
                .groupby(["balls", "strikes"])
                .agg(
                    expected_run_value=("pa_run_value", "mean"),
                    n_pitches=("pa_run_value", "count"),
                )
                .reset_index())

league_avg_rv = df_valid["pa_run_value"].mean()

print(f"\n{'='*55}")
print("Count-State Expected Run Value (empirical)")
print(f"(Mean PA run value when pitch is thrown at each count)")
print(f"League average PA run value: {league_avg_rv:.4f} runs")
print(f"{'='*55}")
cv_pivot = count_values.pivot(index="balls", columns="strikes", values="expected_run_value")
cv_pivot.index.name = "Balls \\ Strikes"
print(cv_pivot.round(4).to_string())

count_values["rv_vs_avg"] = count_values["expected_run_value"] - league_avg_rv
print(f"\nCount-state advantage vs league average ({league_avg_rv:.4f}):")
adv_pivot = count_values.pivot(index="balls", columns="strikes", values="rv_vs_avg")
adv_pivot.index.name = "Balls \\ Strikes"
print(adv_pivot.round(4).to_string())

# ── 4. Count transition probabilities ─────────────────────────────────────────
#
# For each count (balls, strikes), what fraction of pitches result in:
#   ball, called_strike, swinging_strike, foul, in_play, hbp
# And therefore what count follows?
#
# PA-ending events (strikeout, walk, HBP, in-play) are separate from
# count-changing events (ball, strike, foul).

def classify_pitch_result(row):
    """Map pitch outcome and flags to a descriptive event label."""
    if row["is_hbp"]:
        return "hbp"
    if row["pitch_outcome"] == "B":
        return "ball"
    if row["pitch_outcome"] == "X":
        return "in_play"
    # pitch_outcome == "S": called strike, swinging strike, or foul
    if row["is_whiff"]:
        return "swinging_strike"
    if row["is_called_strike"]:
        return "called_strike"
    # not a whiff, not called strike → foul
    return "foul"

print("\nClassifying pitch results… (takes ~30s)")
df["pitch_result"] = df.apply(classify_pitch_result, axis=1)

transitions = (df
               .groupby(["balls", "strikes", "pitch_result"])
               .size()
               .reset_index(name="n"))

totals = transitions.groupby(["balls", "strikes"])["n"].sum().reset_index(name="total")
transitions = transitions.merge(totals, on=["balls", "strikes"])
transitions["pct"] = (transitions["n"] / transitions["total"] * 100).round(1)

print(f"\n{'='*55}")
print("Count Transition Probabilities")
print(f"{'='*55}")
trans_pivot = transitions.pivot_table(
    index=["balls", "strikes"], columns="pitch_result", values="pct", fill_value=0.0
)
# reorder columns for readability
col_order = [c for c in ["ball","called_strike","swinging_strike","foul","in_play","hbp"]
             if c in trans_pivot.columns]
print(trans_pivot[col_order].round(1).to_string())

# ── 5. Linear weights ─────────────────────────────────────────────────────────
#
# For each PA-ending event, compute run value using RE24:
#   LW = RE_state_after + runs_scored_on_play - RE_state_before
#
# RE_state_before = RE24[base_state_at_PA_start, outs_at_PA_start]
# RE_state_after  = RE24[base_state_after_event, outs_after_event]
#
# For outs (K, GIDP, flyout, etc.) after the event we need the next PA's
# base-out state. We get this by taking the FIRST pitch of the next PA.
#
# Simple approach: use delta_run_exp already in data.
# Sum delta_run_exp across the full PA → PA-level run value.
# Average by outcome type → linear weights.

# pa_outcomes already has outcome_type and pa_total_dre from the section above
pa_level = pa_outcomes.assign(
    balls_at_end   = lambda d: d["balls"],
    strikes_at_end = lambda d: d["strikes"],
).copy()

lw = (pa_level
      .groupby("outcome_type")
      .agg(
          lw_raw=("pa_total_dre", "mean"),
          n=("pa_total_dre", "count"),
      )
      .reset_index()
      .sort_values("lw_raw", ascending=False))

# center around out value (standard linear weights are relative to average out)
avg_out = lw.loc[lw["outcome_type"] == "out_in_play", "lw_raw"].values[0]
lw["lw"] = lw["lw_raw"] - avg_out

print(f"\n{'='*55}")
print("Linear Weights (run value above average out)")
print(f"{'='*55}")
print(f"  {'Outcome':<15}  {'LW (runs)':<12}  {'n':>8}")
for _, row in lw.iterrows():
    print(f"  {row['outcome_type']:<15}  {row['lw']:>+.4f}       {int(row['n']):>8,}")

# ── 6. Count-conditional linear weights ───────────────────────────────────────
# How much is a walk worth from 3-1 vs 0-2? A strikeout from 0-2 vs 3-2?
# Compute linear weights for key outcomes broken out by the count they occurred in.

key_outcomes = ["walk", "strikeout", "home_run", "single", "out_in_play"]
lw_by_count = (pa_level[pa_level["outcome_type"].isin(key_outcomes)]
               .groupby(["balls_at_end", "strikes_at_end", "outcome_type"])
               ["pa_total_dre"].mean()
               .reset_index()
               .rename(columns={"pa_total_dre": "lw_raw"}))

lw_by_count["lw"] = lw_by_count["lw_raw"] - avg_out

print(f"\n{'='*55}")
print("Count-conditional linear weights (selected outcomes)")
print(f"{'='*55}")
for outcome in key_outcomes:
    sub = lw_by_count[lw_by_count["outcome_type"] == outcome].copy()
    if sub.empty:
        continue
    pivot = sub.pivot(index="balls_at_end", columns="strikes_at_end", values="lw")
    pivot.index.name = f"{outcome}  B \\ S"
    print(f"\n{outcome}:")
    print(pivot.round(4).to_string())

# ── 7. Save results ───────────────────────────────────────────────────────────

pathlib.Path("results").mkdir(exist_ok=True)

re24_pivot.round(3).to_csv("results/re24.csv")
count_values.to_csv("results/count_values.csv", index=False)
transitions.to_csv("results/count_transitions.csv", index=False)
lw.to_csv("results/linear_weights.csv", index=False)
lw_by_count.to_csv("results/linear_weights_by_count.csv", index=False)

print(f"\nSaved to results/:")
print(f"  re24.csv, count_values.csv, count_transitions.csv,")
print(f"  linear_weights.csv, linear_weights_by_count.csv")
