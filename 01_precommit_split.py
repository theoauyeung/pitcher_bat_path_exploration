"""
Pre/post-commit trajectory split via Zobrist-style 9-parameter reconstruction.

Identification strategy (proj_desc.md §6):
  A batter cannot select on information they do not have. The swing is committed
  early; post-commit movement is the gap between where the ball actually crossed
  the plate and where it would have crossed had no further forces acted after
  commit. Conditioning on the full pre-commit trajectory makes post-commit
  deviation exogenous to the swing decision (conditional ignorability).

Trajectory model (origin at the actual release point):
  x(t) = R_x + V_x·t + ½·A_x·t²
  y(t) = R_y + V_y·t + ½·A_y·t²   (y=0 at front of plate)
  z(t) = R_z + V_z·t + ½·A_z·t²

  R_y = 60.5 − extension (release y-position, ft from front of plate).

  Parameters are reconstructed from release position, pfx_x/pfx_z (horizontal
  and induced-vertical break), and release speed — following the same approach
  as Zobrist's hitter_advance_report.create_9p_fit. This is equivalent to the
  previous ax/ay/az approach but anchored at release rather than the Statcast
  50 ft reference, and uses pfx for better stability under missing/noisy ax.

  A_y: aerodynamic drag in the y direction (decelerates the ball).
  A_x, A_z: solved analytically to reproduce the known plate crossing (plate_x,
  plate_z) given the spinless-trajectory release angles.

  Key identity: A_x ≈ 2·pfx_x / t_plate², so dev_x = pfx_x·(commit_s/t_plate)²
  and similarly dev_z. Post-commit deviation is an exact fractional rescaling of
  the total break, with the fraction determined by how early the batter commits.

Commit time is set early deliberately (default 150 ms). Conservative choice gives
a lower bound on distortion; robustness grid demotes it to a sensitivity check.

Columns added (prefix pc{commit_ms}_):
  dev_x, dev_z, dev_total  — post-commit deviation at plate (ft)
  x_proj, z_proj           — pre-commit projected plate crossing (ft)
  x_commit, y_commit, z_commit  — ball position at commit time (ft)
  vx_commit, vy_commit, vz_commit — velocity at commit time (ft/s)
  t_plate                  — flight time from release to plate (s)
  R_x, R_y, R_z            — release position (ft)
  V_x, V_y, V_z            — release velocity (ft/s)
  A_x, A_y, A_z            — trajectory accelerations (ft/s²)

Output: data/swings_precommit.parquet

Run:
    .venv/Scripts/python.exe 01_precommit_split.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

COMMIT_MS_DEFAULT = 150
COMMIT_MS_GRID = [125, 150, 175, 200]

# Physics constants matching Zobrist's hitter_advance_report.py
GRAVITY_DROP_CONSTANT  = 523    # Tango gravity estimate (in·mph² units)
AVG_AIR_DENSITY        = 0.00537
PLATE_SPEED_ESTIMATE   = 0.92
DRAG_COEFFICIENT       = 0.327
Y_PLATE                = 17 / 12   # ft — front edge of home plate

_RELEASE_COLS = [
    "release_pos_x", "release_extension", "release_pos_z",
    "pfx_x", "pfx_z", "release_speed", "plate_x", "plate_z",
]


def _reconstruct_9p(release_x, extension, release_z,
                    pfx_x_ft, pfx_z_ft, speed_mph,
                    plate_x, plate_z):
    """
    Vectorised 9-parameter reconstruction from release-point inputs.

    Returns R_x, R_y, R_z, V_x, V_y, V_z, A_x, A_y, A_z, t_plate
    as numpy arrays matching the input shapes.
    """
    R_x = release_x
    R_y = 60.5 - extension          # y-distance from front of plate to release
    R_z = release_z

    # Convert IVB → total VB by adding back the gravity contribution.
    # (GRAVITY_DROP_CONSTANT / speed)² is the gravity drop in inches; / 12 → ft.
    gravity_vb_ft = (GRAVITY_DROP_CONSTANT / speed_mph) ** 2 / 12
    vb = pfx_z_ft - gravity_vb_ft   # total vertical break (ft)
    hb = pfx_x_ft                   # horizontal break = pfx_x (ft)

    # Spinless plate location — where the ball crosses without Magnus forces
    X1 = plate_x - hb
    Z1 = plate_z - vb
    dist_y = R_y - Y_PLATE

    # Release angles toward the spinless plate location
    x_angle = np.arctan((X1 - R_x) / dist_y)
    z_angle = np.arctan((Z1 - R_z) / dist_y)

    # Release velocity components (ft/s; V_y < 0 — toward plate)
    # Use unsigned speed for V_x/V_z so downward-aimed angles give V_z < 0.
    spd = speed_mph * 1.466667
    V_x = spd * np.cos(z_angle) * np.sin(x_angle)
    V_y = -spd * np.cos(z_angle) * np.cos(x_angle)
    V_z = spd * np.sin(z_angle)

    # Aerodynamic drag decelerates the ball in y
    A_y = AVG_AIR_DENSITY * DRAG_COEFFICIENT * PLATE_SPEED_ESTIMATE * V_y ** 2

    # Flight time: solve 0.5·A_y·t² + V_y·t + (R_y − Y_plate) = 0
    a   = 0.5 * A_y
    b   = V_y
    c   = R_y - Y_PLATE
    disc = b ** 2 - 4.0 * a * c
    t_plate = (-b - np.sqrt(disc)) / (2.0 * a)

    # A_x, A_z: solved analytically to land at (plate_x, plate_z)
    A_x = 2.0 * ((plate_x - R_x) - V_x * t_plate) / t_plate ** 2
    A_z = 2.0 * ((plate_z - R_z) - V_z * t_plate) / t_plate ** 2

    return R_x, R_y, R_z, V_x, V_y, V_z, A_x, A_y, A_z, t_plate


def add_precommit_features(df, commit_ms):
    """
    Return df with pre/post-commit columns for one commit time.
    Rows missing any release column get NaN in all new columns.
    """
    commit_s = commit_ms / 1000.0
    p = f"pc{commit_ms}_"

    ok = df[_RELEASE_COLS].notna().all(axis=1)
    d  = df.loc[ok]

    R_x, R_y, R_z, V_x, V_y, V_z, A_x, A_y, A_z, t_plate = _reconstruct_9p(
        d["release_pos_x"].values,
        d["release_extension"].values,
        d["release_pos_z"].values,
        d["pfx_x"].values,
        d["pfx_z"].values,
        d["release_speed"].values,
        d["plate_x"].values,
        d["plate_z"].values,
    )

    t_commit = t_plate - commit_s

    # Ball position and velocity at commit time
    x_c  = R_x + V_x * t_commit + 0.5 * A_x * t_commit ** 2
    y_c  = R_y + V_y * t_commit + 0.5 * A_y * t_commit ** 2
    z_c  = R_z + V_z * t_commit + 0.5 * A_z * t_commit ** 2
    vx_c = V_x + A_x * t_commit
    vy_c = V_y + A_y * t_commit
    vz_c = V_z + A_z * t_commit

    # Post-commit deviation: ½·A·commit_s²  (algebraically = pfx * (commit_s/t_plate)²)
    dev_x = 0.5 * A_x * commit_s ** 2
    dev_z = 0.5 * A_z * commit_s ** 2

    cols = {
        f"{p}dev_x":      dev_x,
        f"{p}dev_z":      dev_z,
        f"{p}dev_total":  np.sqrt(dev_x ** 2 + dev_z ** 2),
        f"{p}x_proj":     d["plate_x"].values - dev_x,   # = x_c + vx_c · commit_s
        f"{p}z_proj":     d["plate_z"].values - dev_z,
        f"{p}x_commit":   x_c,
        f"{p}y_commit":   y_c,
        f"{p}z_commit":   z_c,
        f"{p}vx_commit":  vx_c,
        f"{p}vy_commit":  vy_c,
        f"{p}vz_commit":  vz_c,
        f"{p}t_plate":    t_plate,
        # 9p params — used by snapshot-based visualisations
        f"{p}R_x": R_x,  f"{p}R_y": R_y,  f"{p}R_z": R_z,
        f"{p}V_x": V_x,  f"{p}V_y": V_y,  f"{p}V_z": V_z,
        f"{p}A_x": A_x,  f"{p}A_y": A_y,  f"{p}A_z": A_z,
    }

    result = pd.DataFrame(np.nan, index=df.index, columns=list(cols))
    result.loc[ok] = pd.DataFrame(cols, index=d.index)
    return pd.concat([df, result], axis=1)


def build_precommit(df, commit_ms_grid=COMMIT_MS_GRID):
    """Add pre/post-commit columns for every commit time in the grid."""
    for ms in commit_ms_grid:
        df = add_precommit_features(df, ms)
    return df


def _validate(df, commit_ms=150, tol=1e-6):
    """
    Sanity checks:
    1. x_proj + dev_x = plate_x  (algebraic identity, should hold to FP precision)
    2. z_proj + dev_z = plate_z
    3. t_plate in plausible range [0.35, 0.60] s
    """
    p = f"pc{commit_ms}_"

    ok = df[[f"{p}dev_x", f"{p}x_proj", "plate_x"]].dropna()
    err_x = (ok[f"{p}x_proj"] + ok[f"{p}dev_x"] - ok["plate_x"]).abs().max()
    ok2 = df[[f"{p}dev_z", f"{p}z_proj", "plate_z"]].dropna()
    err_z = (ok2[f"{p}z_proj"] + ok2[f"{p}dev_z"] - ok2["plate_z"]).abs().max()
    assert err_x < tol, f"x reconstruction error {err_x:.2e} > tolerance {tol}"
    assert err_z < tol, f"z reconstruction error {err_z:.2e} > tolerance {tol}"

    t = df[f"{p}t_plate"].dropna()
    assert t.between(0.30, 0.65).mean() > 0.98, (
        f"Flight time out of range: {t.describe()}"
    )

    print(
        f"Validation passed (commit={commit_ms}ms): "
        f"max x err={err_x:.2e}, z err={err_z:.2e}, "
        f"t_plate mean={t.mean():.3f}s"
    )


def snapshot(R_x, R_y, R_z, V_x, V_y, V_z, A_x, A_y, A_z, t):
    """
    Position along a 9p trajectory at time t (scalar or array).
    Returns (x, y, z) arrays — used by visualisations to draw
    time-indexed trajectory snapshots.
    """
    x = R_x + V_x * t + 0.5 * A_x * t ** 2
    y = R_y + V_y * t + 0.5 * A_y * t ** 2
    z = R_z + V_z * t + 0.5 * A_z * t ** 2
    return x, y, z


if __name__ == "__main__":
    src = Path("data/swings_2023_2025.csv")
    out = Path("data/swings_precommit.parquet")

    print(f"Loading {src}…")
    df = pd.read_csv(src, low_memory=False)
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    cov = df[_RELEASE_COLS].notna().all(axis=1).mean()
    print(f"  Release params present: {cov:.1%} of rows")

    print(f"Computing pre/post-commit split for grid {COMMIT_MS_GRID} ms…")
    df = build_precommit(df)

    _validate(df, commit_ms=150)

    Path("data").mkdir(exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nSaved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"Shape: {df.shape}")

    p = f"pc{COMMIT_MS_DEFAULT}_"
    dev = df[f"{p}dev_total"].dropna()
    print(f"\nPost-commit deviation ({COMMIT_MS_DEFAULT}ms), ft:")
    print(f"  mean={dev.mean():.3f}  median={dev.median():.3f}  "
          f"p95={dev.quantile(0.95):.3f}  max={dev.max():.3f}")

    ff = df[df["pitch_type"] == "FF"][[f"{p}dev_z"]].dropna()
    print(
        f"\nFF dev_z mean (should be negative — ball drops below linear path): "
        f"{ff[f'{p}dev_z'].mean():.4f} ft"
    )
