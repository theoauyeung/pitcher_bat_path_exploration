"""
3D pitch trajectory viewer — catcher's-eye PNG.

Mean trajectory per pitch type, with the batter decision/commit point
marked on each curve. Camera sits low behind home plate, looking toward
the mound (catcher's perspective).

Run:
    .venv/Scripts/python.exe 05_results_explorer.py

Output:
    results/figures/03_batter_view_3d.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

Path("results/figures").mkdir(parents=True, exist_ok=True)

PITCH_COLORS = {
    "FF": "#d62728",
    "SI": "#ff7f0e",
    "FC": "#e377c2",
    "CH": "#2ca02c",
    "FS": "#17becf",
    "SL": "#bcbd22",
    "CU": "#1f77b4",
    "SW": "#8c564b",
    "ST": "#9467bd",
    "KC": "#1f77b4",
    "FO": "#2ca02c",
}

PITCH_NAMES = {
    "FF": "Four-seam fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "CH": "Changeup",
    "FS": "Splitter",
    "SL": "Slider",
    "CU": "Curveball",
    "SW": "Sweeper",
    "ST": "Sweeper",
    "KC": "Knuckle-curve",
    "FO": "Forkball",
}

GRASS_COLOR = "#4a7a34"
DIRT_COLOR  = "#c4a47c"

PITCHER_NAME = "Shohei Ohtani"
COMMIT_MS    = 150
COMMIT_S     = COMMIT_MS / 1000.0
SNAP_DT      = 0.005

PREFIX = f"pc{COMMIT_MS}_"
TRAJ_COLS = [f"{PREFIX}{c}" for c in
             ["R_x", "R_y", "R_z", "V_x", "V_y", "V_z",
              "A_x", "A_y", "A_z", "t_plate"]]

# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading swings_precommit.parquet …")
df = pd.read_parquet("data/swings_precommit.parquet")

sub = df[
    (df["pitcher_full_name"] == PITCHER_NAME) &
    df[TRAJ_COLS].notna().all(axis=1) &
    df["pitch_type"].isin(PITCH_COLORS)
].copy()
print(f"  {len(sub):,} pitches for {PITCHER_NAME} with valid trajectory params")

# Mean 9p params per pitch type — one representative trajectory per type
means = sub.groupby("pitch_type")[TRAJ_COLS].mean()
print(f"  Pitch types: {list(means.index)}")


# ── Trajectory helper ──────────────────────────────────────────────────────────

def _snap(R_x, R_y, R_z, V_x, V_y, V_z, A_x, A_y, A_z, t_max):
    ts = np.arange(0, t_max + SNAP_DT / 2, SNAP_DT)
    x = R_x + V_x * ts + 0.5 * A_x * ts ** 2
    y = R_y + V_y * ts + 0.5 * A_y * ts ** 2
    z = R_z + V_z * ts + 0.5 * A_z * ts ** 2
    return x, y, z, ts


# ── Build figure ───────────────────────────────────────────────────────────────

fig = go.Figure()

# -- Mean trajectory per pitch type --------------------------------------------

commit_positions = {}  # pt -> (x_c, y_c, z_c)

print("Drawing mean trajectory traces …")
for pt in means.index:
    m = means.loc[pt]
    R_x, R_y, R_z = m[f"{PREFIX}R_x"], m[f"{PREFIX}R_y"], m[f"{PREFIX}R_z"]
    V_x, V_y, V_z = m[f"{PREFIX}V_x"], m[f"{PREFIX}V_y"], m[f"{PREFIX}V_z"]
    A_x, A_y, A_z = m[f"{PREFIX}A_x"], m[f"{PREFIX}A_y"], m[f"{PREFIX}A_z"]
    t_plate = m[f"{PREFIX}t_plate"]
    t_commit = t_plate - COMMIT_S

    x, y, z, ts = _snap(R_x, R_y, R_z, V_x, V_y, V_z, A_x, A_y, A_z, t_plate)
    ci = int(np.searchsorted(ts, t_commit))

    color = PITCH_COLORS[pt]
    name  = PITCH_NAMES.get(pt, pt)

    # Pre-commit segment — tunnel phase, thinner/transparent
    fig.add_trace(go.Scatter3d(
        x=x[:ci + 1], y=y[:ci + 1], z=z[:ci + 1],
        mode="lines",
        line=dict(color=color, width=5),
        opacity=0.5,
        name=name,
        legendgroup=pt,
        showlegend=True,
    ))

    # Post-commit segment — where the pitch diverges, thicker/opaque
    fig.add_trace(go.Scatter3d(
        x=x[ci:], y=y[ci:], z=z[ci:],
        mode="lines",
        line=dict(color=color, width=9),
        opacity=1.0,
        legendgroup=pt,
        showlegend=False,
    ))

    # Plate-crossing dot
    fig.add_trace(go.Scatter3d(
        x=[float(x[-1])], y=[float(y[-1])], z=[float(z[-1])],
        mode="markers",
        marker=dict(size=11, color=color, opacity=1.0),
        legendgroup=pt,
        showlegend=False,
    ))

    # Commit point — white-ringed marker on each trajectory
    x_c = R_x + V_x * t_commit + 0.5 * A_x * t_commit ** 2
    y_c = R_y + V_y * t_commit + 0.5 * A_y * t_commit ** 2
    z_c = R_z + V_z * t_commit + 0.5 * A_z * t_commit ** 2
    commit_positions[pt] = (float(x_c), float(y_c), float(z_c))

    fig.add_trace(go.Scatter3d(
        x=[float(x_c)], y=[float(y_c)], z=[float(z_c)],
        mode="markers",
        marker=dict(
            size=14,
            color=color,
            line=dict(color="white", width=3),
        ),
        legendgroup=pt,
        showlegend=False,
    ))

# -- Decision point indicator: dashed horizontal line + label -----------------

commit_ys = [v[1] for v in commit_positions.values()]
mean_commit_y = float(np.mean(commit_ys))

# Horizontal dashed line across the width of the strike zone at commit depth
fig.add_trace(go.Scatter3d(
    x=[-1.0, 1.0],
    y=[mean_commit_y, mean_commit_y],
    z=[2.5, 2.5],
    mode="lines",
    line=dict(color="rgba(255,255,255,0.55)", width=2, dash="dot"),
    showlegend=False,
))

# Text label (placed slightly above and to the right)
fig.add_trace(go.Scatter3d(
    x=[1.1],
    y=[mean_commit_y],
    z=[3.0],
    mode="text",
    text=[f"  Decision point<br>  ({COMMIT_MS} ms)"],
    textfont=dict(size=11, color="white", family="Arial"),
    showlegend=False,
))

# -- Field geometry ------------------------------------------------------------

# Grass ground plane
_xg = np.linspace(-3, 3, 30)
_yg = np.linspace(1.4, 62, 30)
_xg, _yg = np.meshgrid(_xg, _yg)
fig.add_trace(go.Surface(
    x=_xg, y=_yg, z=np.zeros_like(_xg),
    colorscale=[[0, GRASS_COLOR], [1, GRASS_COLOR]],
    opacity=1, showscale=False, showlegend=False, hoverinfo="skip",
))

# Dirt infield circle
_np_c  = 50
_th_c  = np.linspace(0, 2 * np.pi, _np_c)
_r_arr = np.linspace(0, 3, _np_c)
_th_c, _r_arr = np.meshgrid(_th_c, _r_arr)
fig.add_trace(go.Surface(
    x=_r_arr * np.cos(_th_c),
    y=_r_arr * np.sin(_th_c) + 2,
    z=np.full((_np_c, _np_c), 0.005),
    colorscale=[[0, DIRT_COLOR], [1, DIRT_COLOR]],
    opacity=1, showscale=False, showlegend=False, hoverinfo="skip",
))

# Pitcher's mound (hemisphere)
_th_m, _ph_m = np.meshgrid(
    np.linspace(0, 2 * np.pi, 30),
    np.linspace(0, np.pi, 30),
)
_r_m = 9
fig.add_trace(go.Surface(
    x=_r_m * np.cos(_th_m) * np.sin(_ph_m),
    y=_r_m * np.sin(_th_m) * np.sin(_ph_m) + 60.5,
    z=_r_m * np.cos(_ph_m) - 9 + (10 / 12),
    colorscale=[[0, DIRT_COLOR], [1, DIRT_COLOR]],
    opacity=1, showscale=False, showlegend=False, hoverinfo="skip",
))

# Home plate (white pentagon)
_x_hp = [-8.5 / 12, 8.5 / 12, 10 / 12,  0, -10 / 12, -8.5 / 12]
_y_hp = [2,          2,         1,         0,  1,         2        ]
_z_hp = [0.01] * 6
fig.add_trace(go.Mesh3d(
    x=_x_hp, y=_y_hp, z=_z_hp,
    i=[0, 1, 2, 2, 3, 4],
    j=[1, 2, 3, 3, 4, 5],
    k=[5, 5, 5, 4, 5, 0],
    color="white", opacity=1, flatshading=True,
    showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter3d(
    x=_x_hp + [_x_hp[0]],
    y=_y_hp + [_y_hp[0]],
    z=_z_hp + [_z_hp[0]],
    mode="lines",
    line=dict(color="black", width=5),
    showlegend=False,
))

# Strike zone
fig.add_trace(go.Scatter3d(
    x=[-10 / 12, 10 / 12, 10 / 12, -10 / 12, -10 / 12],
    y=[2, 2, 2, 2, 2],
    z=[1.5, 1.5, 3.5, 3.5, 1.5],
    mode="lines",
    line=dict(color="#dddddd", width=5),
    showlegend=False,
))

# -- Camera and layout ---------------------------------------------------------

fig.update_layout(
    scene=dict(
        xaxis=dict(
            range=[-3, 3],
            showticklabels=False, showgrid=False, zeroline=False, title="",
            backgroundcolor="rgb(55, 55, 55)",
        ),
        yaxis=dict(
            range=[0, 62],
            showticklabels=False, showgrid=False, zeroline=False, title="",
            backgroundcolor="rgb(55, 55, 55)",
        ),
        zaxis=dict(
            range=[0, 6],
            showticklabels=False, showgrid=False, zeroline=False, title="",
            backgroundcolor="rgb(55, 55, 55)",
        ),
        aspectmode="manual",
        aspectratio=dict(x=1, y=10, z=1),
        bgcolor="rgb(55, 55, 55)",
    ),
    scene_camera=dict(
        eye=dict(x=0.7, y=-5.75, z=0.1),
        center=dict(x=0, y=5, z=-0.25),
        up=dict(x=0, y=0, z=1),
    ),
    paper_bgcolor="rgb(45, 45, 45)",
    plot_bgcolor="rgb(45, 45, 45)",
    width=1400,
    height=800,
    margin=dict(l=0, r=0, t=55, b=0),
    title=dict(
        text=f"{PITCHER_NAME} — Mean Pitch Trajectories by Type",
        font=dict(size=17, color="white", family="Arial"),
        x=0.5, xanchor="center",
    ),
    legend=dict(
        font=dict(size=13, color="white", family="Arial"),
        bgcolor="rgba(255,255,255,0.12)",
        bordercolor="rgba(255,255,255,0.3)",
        borderwidth=1,
        x=0.01, y=0.97,
        xanchor="left", yanchor="top",
        itemsizing="constant",
    ),
)

# ── Save ───────────────────────────────────────────────────────────────────────

out = Path("results/figures/03_batter_view_3d.png")
fig.write_image(str(out), scale=2)
print(f"\nSaved {out}  ({out.stat().st_size / 1e3:.0f} KB)")
