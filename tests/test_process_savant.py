"""
Tests for src/data/process_savant.py.

Uses a synthetic savant_raw.csv fixture — the real data/savant_raw.csv
does not exist yet (the R pull hasn't been run). Imports `transform` directly
from process_savant to avoid exec overhead and network calls.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src/data to path so process_savant can be imported as a module
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data"))
from process_savant import transform, SWING_DESC, WHIFF_DESC, BIP_DESC  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

SWING_DESCS_LIST = [
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "hit_into_play",
    "foul_tip",
    "missed_bunt",
]
NON_SWING_DESCS_LIST = ["ball", "called_strike", "blocked_ball"]


def _make_fixture(n_swings: int = 20, n_non_swings: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Build a minimal synthetic Statcast DataFrame with pre-rename column names.
    Includes batter_full_name so the pybaseball network lookup is bypassed.
    """
    rng = np.random.default_rng(seed)
    n = n_swings + n_non_swings

    swing_descriptions = rng.choice(SWING_DESCS_LIST, size=n_swings).tolist()
    non_swing_descriptions = rng.choice(NON_SWING_DESCS_LIST, size=n_non_swings).tolist()
    all_descriptions = swing_descriptions + non_swing_descriptions

    launch_speeds = [None] * n
    launch_angles = [None] * n
    events_col = [""] * n
    for i, d in enumerate(all_descriptions):
        if d == "hit_into_play":
            launch_speeds[i] = float(rng.uniform(60, 110))
            launch_angles[i] = float(rng.uniform(-10, 40))
            events_col[i] = rng.choice(
                ["single", "double", "triple", "home_run", "field_out"]
            )

    df = pd.DataFrame(
        {
            # Pre-rename Statcast column names
            "game_pk":              [1] * n,
            "game_date":            ["2024-04-01"] * n,
            "game_year":            [2024] * n,
            "at_bat_number":        rng.integers(1, 4, size=n).tolist(),
            "pitch_number":         rng.integers(1, 6, size=n).tolist(),
            "batter":               rng.integers(500000, 700000, size=n).tolist(),
            "pitcher":              rng.integers(400000, 600000, size=n).tolist(),
            "player_name":          ["Yamamoto, Yoshinobu"] * n,
            "stand":                rng.choice(["R", "L"], size=n).tolist(),
            "p_throws":             rng.choice(["R", "L"], size=n).tolist(),
            "type":                 rng.choice(["B", "S", "X"], size=n).tolist(),
            "pitch_type":           rng.choice(["FF", "SL", "CH"], size=n).tolist(),
            "release_speed":        rng.uniform(70, 100, size=n).tolist(),
            "pfx_x":                rng.uniform(-1, 1, size=n).tolist(),
            "pfx_z":                rng.uniform(-1, 1, size=n).tolist(),
            "plate_x":              rng.uniform(-1.5, 1.5, size=n).tolist(),
            "plate_z":              rng.uniform(1.0, 4.0, size=n).tolist(),
            "sz_top":               [3.5] * n,
            "sz_bot":               [1.5] * n,
            "release_pos_x":        rng.uniform(-3, 3, size=n).tolist(),
            "release_pos_y":        [55.0] * n,
            "release_pos_z":        rng.uniform(4, 7, size=n).tolist(),
            "release_extension":    rng.uniform(5, 7, size=n).tolist(),
            "arm_angle":            rng.uniform(0, 90, size=n).tolist(),
            "x0":                   rng.uniform(-3, 3, size=n).tolist(),
            "y0":                   [50.0] * n,
            "z0":                   rng.uniform(4, 7, size=n).tolist(),
            "vx0":                  rng.uniform(-10, 10, size=n).tolist(),
            "vy0":                  rng.uniform(-140, -120, size=n).tolist(),
            "vz0":                  rng.uniform(-10, 5, size=n).tolist(),
            "ax":                   rng.uniform(-5, 5, size=n).tolist(),
            "ay":                   rng.uniform(20, 30, size=n).tolist(),
            "az":                   rng.uniform(-20, -10, size=n).tolist(),
            "balls":                rng.integers(0, 4, size=n).tolist(),
            "strikes":              rng.integers(0, 3, size=n).tolist(),
            "outs_when_up":         rng.integers(0, 3, size=n).tolist(),
            "inning":               rng.integers(1, 10, size=n).tolist(),
            "description":          all_descriptions,
            "events":               events_col,
            # Bat-tracking columns (swing rows need valid values to pass filter)
            "vert_attack_angle":    (
                [float(rng.uniform(-20, 20)) for _ in range(n_swings)]
                + [None] * n_non_swings
            ),
            "horz_attack_angle":    (
                [float(rng.uniform(-30, 30)) for _ in range(n_swings)]
                + [None] * n_non_swings
            ),
            "bat_speed":            (
                [float(rng.uniform(50, 80)) for _ in range(n_swings)]
                + [None] * n_non_swings
            ),
            "swing_length":         (
                [float(rng.uniform(5, 10)) for _ in range(n_swings)]
                + [None] * n_non_swings
            ),
            "swing_path_tilt":      (
                [float(rng.uniform(-10, 10)) for _ in range(n_swings)]
                + [None] * n_non_swings
            ),
            "ball_bat_intercept_y": (
                [float(rng.uniform(0, 2)) for _ in range(n_swings)]
                + [None] * n_non_swings
            ),
            "ball_bat_miss":        (
                [float(rng.uniform(0, 5)) for _ in range(n_swings)]
                + [None] * n_non_swings
            ),
            "offset_y_ms":          (
                [float(rng.uniform(-10, 10)) for _ in range(n_swings)]
                + [None] * n_non_swings
            ),
            "offset_z_in":          (
                [float(rng.uniform(-3, 3)) for _ in range(n_swings)]
                + [None] * n_non_swings
            ),
            "offset_x_in":          (
                [float(rng.uniform(-3, 3)) for _ in range(n_swings)]
                + [None] * n_non_swings
            ),
            # EV/LA — present on hit_into_play rows; NaN on others
            "launch_speed":         launch_speeds,
            "launch_angle":         launch_angles,
            # run expectancy (passthrough)
            "delta_run_exp":        rng.uniform(-0.5, 0.5, size=n).tolist(),
            # Include batter_full_name so the pybaseball lookup is skipped
            "batter_full_name":     [f"Batter {i}" for i in range(n)],
        }
    )
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_has_launch_speed_and_launch_angle():
    """Output must have launch_speed and launch_angle columns (passthrough from savant raw)."""
    raw = _make_fixture(n_swings=30, n_non_swings=10)
    out = transform(raw, resolve_batter_names=False)
    assert "launch_speed" in out.columns, "launch_speed missing from output"
    assert "launch_angle" in out.columns, "launch_angle missing from output"


def test_only_swing_rows_remain():
    """After filter, every row must have is_swing == 1."""
    raw = _make_fixture(n_swings=30, n_non_swings=15)
    out = transform(raw, resolve_batter_names=False)
    assert (out["is_swing"] == 1).all(), "Non-swing rows found in output"
    # Non-swing descriptions must not appear
    non_swing_set = set(NON_SWING_DESCS_LIST)
    assert out["description"].isin(non_swing_set).sum() == 0


def test_lag_features_present():
    """Required lag columns must exist in output."""
    raw = _make_fixture(n_swings=30, n_non_swings=10)
    out = transform(raw, resolve_batter_names=False)
    lag_cols = [
        "prev_pitch_type",
        "prev_release_speed",
        "prev_plate_x",
        "prev_plate_z",
        "velo_delta",
        "plate_x_delta",
        "plate_z_delta",
        "cur_outcome",
        "prev_outcome",
    ]
    for col in lag_cols:
        assert col in out.columns, f"Lag column {col!r} missing from output"


def test_column_renames():
    """Statcast column names must be renamed to pipeline convention."""
    raw = _make_fixture(n_swings=20, n_non_swings=5)
    out = transform(raw, resolve_batter_names=False)
    # Renamed columns must be present
    assert "batter_id" in out.columns
    assert "pitcher_id" in out.columns
    assert "pitcher_full_name" in out.columns
    assert "batter_stand" in out.columns
    assert "pitcher_throws" in out.columns
    assert "pitch_outcome" in out.columns
    # Raw names must NOT appear
    assert "batter" not in out.columns
    assert "pitcher" not in out.columns
    assert "player_name" not in out.columns
    assert "stand" not in out.columns
    assert "p_throws" not in out.columns


def test_haa_vaa_aliases_present():
    """Both full column name and short alias must exist: haa/horz_attack_angle, vaa/vert_attack_angle."""
    raw = _make_fixture(n_swings=20, n_non_swings=5)
    out = transform(raw, resolve_batter_names=False)
    assert "haa" in out.columns
    assert "vaa" in out.columns
    assert "horz_attack_angle" in out.columns
    assert "vert_attack_angle" in out.columns


def test_outcome_flags_correct():
    """is_whiff, is_contact, is_bip flags must be mutually consistent."""
    raw = _make_fixture(n_swings=50, n_non_swings=10)
    out = transform(raw, resolve_batter_names=False)
    # All rows are swings
    assert (out["is_swing"] == 1).all()
    # whiff and contact are mutually exclusive
    assert ((out["is_whiff"] == 1) & (out["is_contact"] == 1)).sum() == 0
    # Every swing is either a whiff or contact
    assert ((out["is_whiff"] == 1) | (out["is_contact"] == 1)).all()
    # BIP is a subset of contact
    bip_rows = out[out["is_bip"] == 1]
    assert (bip_rows["is_contact"] == 1).all()
    assert (bip_rows["is_swing"] == 1).all()


def test_batter_full_name_present():
    """batter_full_name column must exist."""
    raw = _make_fixture(n_swings=20, n_non_swings=5)
    out = transform(raw, resolve_batter_names=False)
    assert "batter_full_name" in out.columns
    assert out["batter_full_name"].notna().all()


def test_count_group_values():
    """count_group must be one of the expected categories."""
    raw = _make_fixture(n_swings=30, n_non_swings=5)
    out = transform(raw, resolve_batter_names=False)
    assert "count_group" in out.columns
    valid_groups = {"two_strike", "three_ball", "other"}
    unexpected = set(out["count_group"].unique()) - valid_groups
    assert not unexpected, f"Unexpected count_group values: {unexpected}"


def test_bat_speed_filter():
    """Rows with bat_speed <= 40 must be filtered out."""
    raw = _make_fixture(n_swings=0, n_non_swings=0)
    # Create one swing row with low bat speed (should be filtered) and one with high
    rows = pd.DataFrame(
        [
            {
                "game_pk": 99, "game_date": "2024-04-01", "game_year": 2024,
                "at_bat_number": 1, "pitch_number": 1,
                "batter": 123456, "pitcher": 234567,
                "player_name": "Test, Pitcher",
                "stand": "R", "p_throws": "R", "type": "S",
                "pitch_type": "FF", "release_speed": 95.0,
                "pfx_x": 0.5, "pfx_z": 1.0,
                "plate_x": 0.0, "plate_z": 2.5,
                "sz_top": 3.5, "sz_bot": 1.5,
                "release_pos_x": 1.0, "release_pos_y": 55.0, "release_pos_z": 6.0,
                "release_extension": 6.0, "arm_angle": 45.0,
                "x0": 1.0, "y0": 50.0, "z0": 6.0,
                "vx0": 0.0, "vy0": -130.0, "vz0": -5.0,
                "ax": 0.0, "ay": 25.0, "az": -15.0,
                "balls": 0, "strikes": 0, "outs_when_up": 0, "inning": 1,
                "description": "swinging_strike", "events": "",
                "vert_attack_angle": 10.0, "horz_attack_angle": 5.0,
                "bat_speed": 35.0,  # below threshold → must be filtered
                "swing_length": 7.0, "swing_path_tilt": 0.0,
                "ball_bat_intercept_y": 1.0, "ball_bat_miss": 2.0,
                "offset_y_ms": None, "offset_z_in": None, "offset_x_in": None,
                "launch_speed": None, "launch_angle": None,
                "delta_run_exp": -0.1, "batter_full_name": "Test Batter",
            },
            {
                "game_pk": 99, "game_date": "2024-04-01", "game_year": 2024,
                "at_bat_number": 1, "pitch_number": 2,
                "batter": 123456, "pitcher": 234567,
                "player_name": "Test, Pitcher",
                "stand": "R", "p_throws": "R", "type": "S",
                "pitch_type": "FF", "release_speed": 95.0,
                "pfx_x": 0.5, "pfx_z": 1.0,
                "plate_x": 0.0, "plate_z": 2.5,
                "sz_top": 3.5, "sz_bot": 1.5,
                "release_pos_x": 1.0, "release_pos_y": 55.0, "release_pos_z": 6.0,
                "release_extension": 6.0, "arm_angle": 45.0,
                "x0": 1.0, "y0": 50.0, "z0": 6.0,
                "vx0": 0.0, "vy0": -130.0, "vz0": -5.0,
                "ax": 0.0, "ay": 25.0, "az": -15.0,
                "balls": 0, "strikes": 0, "outs_when_up": 0, "inning": 1,
                "description": "swinging_strike", "events": "",
                "vert_attack_angle": 10.0, "horz_attack_angle": 5.0,
                "bat_speed": 65.0,  # above threshold → must be kept
                "swing_length": 7.0, "swing_path_tilt": 0.0,
                "ball_bat_intercept_y": 1.0, "ball_bat_miss": 2.0,
                "offset_y_ms": None, "offset_z_in": None, "offset_x_in": None,
                "launch_speed": None, "launch_angle": None,
                "delta_run_exp": -0.1, "batter_full_name": "Test Batter",
            },
        ]
    )
    out = transform(rows, resolve_batter_names=False)
    assert len(out) == 1, f"Expected 1 row (bat_speed=65), got {len(out)}"
    assert (out["bat_speed"] > 40).all(), "Row with bat_speed <= 40 should have been filtered"
