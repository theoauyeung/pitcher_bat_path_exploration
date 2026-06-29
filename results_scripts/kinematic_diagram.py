"""
Kinematic diagram: post-commit trajectory distortion.


Run:
    .venv/Scripts/python.exe 06_kinematic_diagram.py

Saves: results/figures/yamamoto_bernabel_annotation.png
       results/figures/bradley_alonso_annotation.png
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Colour palette ────────────────────────────────────────────────────────────
BG     = "#0d1117"
FG     = "#e6edf3"
BLUE   = "#58a6ff"
AMBER  = "#e3b341"
RED    = "#f85149"
GREEN  = "#3fb950"
GRAY   = "#8b949e"
BORDER = "#30363d"

_PITCH_TYPE_NAMES = {
    "FF": "Four-seam Fastball (FF)", "SI": "Sinker (SI)",
    "CH": "Changeup (CH)",           "SL": "Slider (SL)",
    "CU": "Curveball (CU)",          "FC": "Cutter (FC)",
    "FS": "Splitter (FS)",           "KC": "Knuckle-curve (KC)",
    "ST": "Sweeper (ST)",            "SV": "Slurve (SV)",
    "FO": "Forkball (FO)",           "KN": "Knuckleball (KN)",
}

_PITCH_SHORT = {
    "FF": "Four-seam FB", "SI": "Sinker",  "CH": "Changeup",
    "SL": "Slider",       "CU": "Curveball", "FC": "Cutter",
    "FS": "Splitter",     "KC": "Knuckle-curve", "ST": "Sweeper",
    "SV": "Slurve",       "FO": "Forkball", "KN": "Knuckleball",
}

_DESC_TO_OUTCOME = {
    "ball": "Ball", "blocked_ball": "Ball", "pitchout": "Ball",
    "foul_pitchout": "Ball", "hit_by_pitch": "Ball",
    "called_strike": "Called Strike",
    "swinging_strike": "Swinging Strike",
    "swinging_strike_blocked": "Swinging Strike",
    "missed_bunt": "Swinging Strike",
    "foul": "Foul", "foul_tip": "Foul", "foul_bunt": "Foul",
    "bunt_foul_tip": "Foul",
    "hit_into_play": "In Play",
}

_OUTCOME_COLOR = {
    "Ball": GREEN,
    "Called Strike": AMBER,
    "Swinging Strike": RED,
    "Foul": FG,
    "In Play": FG,
}


# ── Data loader ───────────────────────────────────────────────────────────────

def _get_secret(name):
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


def load_pitch_metrics(
    game_pk, at_bat_number, pitch_number,
    commit_ms=150,
    precommit_path="data/swings_precommit.parquet",
    causal_path="results/xrv_causal.parquet",
):
    """Load all metrics for a single pitch from the project data files.

    Reads swings_precommit.parquet (trajectory + swing shape) and
    xrv_causal.parquet (disruption / distortion / selection tax), then
    makes a single DB query for release_spin_rate and the previous pitch
    in the at-bat (not stored in the parquet).

    Returns a flat dict suitable for passing to make_broadcast_annotation().
    prev_pitch is None when this is the first pitch of the at-bat.
    """
    import mysql.connector

    pc = pd.read_parquet(precommit_path)
    cx = pd.read_parquet(causal_path)

    rp = pc.loc[(pc["game_pk"] == game_pk) &
                (pc["at_bat_number"] == at_bat_number) &
                (pc["pitch_number"] == pitch_number)].iloc[0]
    rc = cx.loc[(cx["game_pk"] == game_pk) &
                (cx["at_bat_number"] == at_bat_number) &
                (cx["pitch_number"] == pitch_number)].iloc[0]

    conn = mysql.connector.connect(
        host=_get_secret("BIOMECH_DB_HOST") or "10.200.200.107",
        port=3306, user="readonlyuser",
        password=_get_secret("BIOMECH_DB_PASS"), database="mlb_db",
    )
    cur = conn.cursor(dictionary=True)

    # spin rate for the current pitch
    cur.execute(
        "SELECT release_spin_rate FROM pbp_raw "
        "WHERE game_pk=%s AND at_bat_number=%s AND pitch_number=%s LIMIT 1",
        (game_pk, at_bat_number, pitch_number),
    )
    spin_row = cur.fetchone() or {}

    # previous pitch in this at-bat (None if first pitch)
    prev_pitch = None
    if pitch_number > 1:
        cur.execute(
            "SELECT pitch_type, pfx_x, pfx_z, pitch_outcome_explanation FROM pbp_raw "
            "WHERE game_pk=%s AND at_bat_number=%s AND pitch_number=%s LIMIT 1",
            (game_pk, at_bat_number, pitch_number - 1),
        )
        prev_row = cur.fetchone()
        if prev_row:
            pt_prev   = str(prev_row.get("pitch_type") or "")
            raw_outcome = prev_row.get("pitch_outcome_explanation", "") or ""
            outcome     = _DESC_TO_OUTCOME.get(raw_outcome, raw_outcome or "—")
            pfx_x_raw = prev_row.get("pfx_x")
            pfx_z_raw = prev_row.get("pfx_z")
            prev_pitch = dict(
                pitch_type = _PITCH_SHORT.get(pt_prev, pt_prev) or "—",
                ivb_in     = float(pfx_z_raw) * 12 if pfx_z_raw is not None else None,
                hb_in      = float(pfx_x_raw) * 12 if pfx_x_raw is not None else None,
                outcome    = outcome,
            )

    conn.close()

    pt_code = str(rp.get("pitch_type", ""))
    balls   = int(rp["balls"])
    strikes = int(rp["strikes"])
    timing  = rp.get("offset_y_ms")

    pc_z_proj = float(rp[f"pc{commit_ms}_z_proj"])
    pc_z_dev  = float(rp[f"pc{commit_ms}_dev_z"])
    plate_z   = float(rp["plate_z"])

    return dict(
        # identity
        pitcher          = str(rp["pitcher_full_name"]),
        batter           = str(rp["batter_full_name"]),
        game_date        = str(rp["game_date"])[:10],
        count            = f"{balls}–{strikes}",
        pitch_type       = _PITCH_TYPE_NAMES.get(pt_code, pt_code),
        # pitch profile
        release_speed    = float(rp["release_speed"]),
        spin_rate        = float(spin_row.get("release_spin_rate") or 0),
        pfx_h_in         = float(rp["pfx_x"]) * 12,
        pfx_v_in         = float(rp["pfx_z"]) * 12,
        vaa              = float(rp["vaa"]),
        # swing shape
        bat_speed        = float(rp["bat_speed"]),
        vert_attack_angle= float(rp["vert_attack_angle"]),
        timing_ms        = float(timing) if pd.notna(timing) else None,
        miss_in          = float(rp["ball_bat_miss"]),
        # post-commit deviation
        pc_dev_z_in      = pc_z_dev * 12,
        pc_z_proj_ft     = pc_z_proj,
        pc_z_actual_ft   = plate_z,
        sz_top           = float(rp["sz_top"]),
        sz_bot           = float(rp["sz_bot"]),
        # disruption model
        disruption_tax          = float(rc["disruption_tax"]),
        adjusted_disruption_tax = float(rc["adjusted_disruption_tax"]),
        decision_cost           = float(rc["decision_cost"]),
        distortion_share        = float(rc["distortion_share"]) * 100,
        # previous pitch context
        prev_pitch       = prev_pitch,
    )


# ── Broadcast annotation ──────────────────────────────────────────────────────

def make_broadcast_annotation(
    screenshot_path, data,
    callout_xy, callout_xytext, callout_label,
    callout_color=RED,
):
    """Two-panel broadcast card: game screenshot (left) + dark metrics panel (right).

    screenshot_path : path to the game image
    data            : dict from load_pitch_metrics()
    callout_xy      : (x, y) pixel coords of the arrow tip on the image
    callout_xytext  : (x, y) pixel coords of the text box
    callout_label   : string for the callout annotation
    callout_color   : RED for distortion-dominant, AMBER for selection-dominant
    """
    import matplotlib.image as mpimg

    img = mpimg.imread(screenshot_path)
    ih, iw = img.shape[:2]

    img_frac   = 0.68
    panel_frac = 0.32
    fig_h      = 9.0
    fig_w      = fig_h * (iw / ih) / img_frac

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    # ── Image ─────────────────────────────────────────────────────────────────
    ax_img = fig.add_axes([0, 0, img_frac, 1.0])
    ax_img.imshow(img, aspect="auto", extent=[0, iw, ih, 0])
    ax_img.set_xlim(0, iw); ax_img.set_ylim(ih, 0); ax_img.axis("off")

    ax_img.text(
        callout_xytext[0], callout_xytext[1],
        callout_label,
        color=callout_color, fontsize=9, fontweight="bold",
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG,
                  edgecolor=callout_color, alpha=0.92, linewidth=1.8),
        zorder=10,
    )

    # ── Metrics panel ─────────────────────────────────────────────────────────
    ax_p = fig.add_axes([img_frac, 0, panel_frac, 1.0])
    ax_p.set_facecolor(BG); ax_p.set_xlim(0, 1); ax_p.set_ylim(0, 1); ax_p.axis("off")
    ax_p.axvline(0.0, color=BORDER, lw=1.0, zorder=1)

    xs, xe = 0.08, 0.95

    def hline(y, color=BORDER, lw=0.7):
        ax_p.plot([xs - 0.04, xe + 0.02], [y, y], color=color, lw=lw,
                  transform=ax_p.transAxes, zorder=2)

    def row(y, label, val, vc=FG, ls=7.5, vs=8.5):
        ax_p.text(xs, y, label, color=GRAY, fontsize=ls, va="top",
                  transform=ax_p.transAxes)
        ax_p.text(xe, y, val, color=vc, fontsize=vs, va="top", ha="right",
                  fontweight="bold", transform=ax_p.transAxes)

    def section_title(y, title, tc=BLUE):
        ax_p.text(xs, y, title, color=tc, fontsize=8.5, fontweight="bold",
                  va="top", transform=ax_p.transAxes)
        hline(y - 0.020, color=tc, lw=1.2)
        return y - 0.038

    rs  = 0.037   # row spacing
    pad = 0.010   # inter-section gap

    # header
    y = 0.96
    ax_p.text(xs, y, data["pitcher"], color=FG, fontsize=11, fontweight="bold",
              va="top", transform=ax_p.transAxes); y -= 0.044
    ax_p.text(xs, y, f"vs.  {data['batter']}", color=BLUE, fontsize=9.5,
              va="top", transform=ax_p.transAxes); y -= 0.036
    ax_p.text(xs, y, f"{data['game_date']}   ·   {data['count']} count",
              color=GRAY, fontsize=7.5, va="top", transform=ax_p.transAxes); y -= 0.032
    ax_p.text(xs, y, data["pitch_type"], color=AMBER, fontsize=8.5, fontweight="bold",
              va="top", transform=ax_p.transAxes); y -= 0.034
    hline(y, BORDER); y -= 0.022

    # pitch profile
    y = section_title(y, "PITCH PROFILE", BLUE)
    row(y, "Velocity",       f"{data['release_speed']:.1f} mph",    AMBER); y -= rs
    row(y, "Spin rate",      f"{data['spin_rate']:,.0f} rpm",        FG);   y -= rs
    row(y, "V-movement",     f"{data['pfx_v_in']:+.1f} in",         RED);  y -= rs
    row(y, "H-movement",     f"{data['pfx_h_in']:+.1f} in",         FG);   y -= rs
    row(y, "Vert. approach", f"{data['vaa']:.1f}°",                 FG);   y -= rs
    y -= pad

    # swing shape
    y = section_title(y, "SWING SHAPE", GREEN)
    row(y, "Bat speed",    f"{data['bat_speed']:.1f} mph",           AMBER); y -= rs
    row(y, "Attack angle", f"{data['vert_attack_angle']:+.1f}°",     FG);   y -= rs
    if data["timing_ms"] is not None:
        timing_str = f"{abs(data['timing_ms']):.0f} ms {'late' if data['timing_ms'] > 0 else 'early'}"
        row(y, "Timing", timing_str, RED if abs(data["timing_ms"]) > 5 else FG)
    else:
        row(y, "Timing", "n/a", GRAY)
    y -= rs
    row(y, "Miss distance", f"{data['miss_in']:.1f} in", RED); y -= rs
    y -= pad

    # previous pitch context
    y = section_title(y, "PREVIOUS PITCH", GRAY)
    pp = data.get("prev_pitch")
    if pp is None:
        ax_p.text(
            (xs + xe) / 2, y, "First pitch of at-bat",
            color=GRAY, fontsize=7.5, va="top", ha="center",
            fontstyle="italic", transform=ax_p.transAxes,
        )
        y -= 0.033
    else:
        rs_pp = 0.033
        row(y, "Pitch type", pp["pitch_type"],                           FG);    y -= rs_pp
        ivb_str = f"{pp['ivb_in']:+.1f} in" if pp["ivb_in"] is not None else "—"
        hb_str  = f"{pp['hb_in']:+.1f} in"  if pp["hb_in"]  is not None else "—"
        row(y, "IVB",        ivb_str,                                    FG);    y -= rs_pp
        row(y, "HB",         hb_str,                                     FG);    y -= rs_pp
        oc      = pp["outcome"]
        row(y, "Outcome",    oc, _OUTCOME_COLOR.get(oc, FG));                    y -= rs_pp
    y -= pad

    # disruption analysis
    y = section_title(y, "DISRUPTION ANALYSIS", RED)
    dev_in    = abs(data["pc_dev_z_in"])
    proj_in   = data["pc_z_proj_ft"] * 12
    act_in    = data["pc_z_actual_ft"] * 12
    sz_top_in = data["sz_top"] * 12
    above_in  = proj_in - sz_top_in

    rs_d = 0.034  # tighter spacing for this section to fit 5 rows
    loc_note = f'+{above_in:.1f}" above zone' if above_in > 0 else "in zone"
    row(y, "Post-commit drop", f"−{dev_in:.1f} in", RED); y -= rs_d
    row(y, "Proj. → actual",
        f'{proj_in:.1f}" ({loc_note}) → {act_in:.1f}"', RED); y -= rs_d

    dt = data["disruption_tax"]
    row(y, "Swing disruption",
        f"{dt:+.3f} runs", RED if dt < 0 else AMBER); y -= rs_d

    dc = data["decision_cost"]
    dc_color = RED if dc > 0 else GREEN
    dc_label = "Chase cost" if dc > 0 else "Decision"
    dc_str   = f"{dc:+.3f} runs" if dc > 0 else f"correct ({dc:+.3f})"
    row(y, dc_label, dc_str, dc_color); y -= rs_d

    adj = data["adjusted_disruption_tax"]
    row(y, "Total burden",
        f"{adj:+.3f} runs", RED if adj < 0 else AMBER); y -= rs_d
    y -= 0.005

    # distortion / selection bar
    hline(y + 0.006, BORDER); y -= 0.010
    bx = xs - 0.04; bw = xe - xs + 0.06; bh = 0.028; by = y - 0.012
    dfrac    = max(data["distortion_share"] / 100, 0.0)
    sel_pct  = 100 - data["distortion_share"]
    dominant = dfrac >= 0.5

    ax_p.add_patch(mpatches.Rectangle(
        (bx, by), bw, bh, facecolor=BORDER, edgecolor="none",
        transform=ax_p.transAxes, zorder=3, clip_on=False))

    if dfrac > 0.005:
        ax_p.add_patch(mpatches.Rectangle(
            (bx, by), bw * dfrac, bh, facecolor=RED, edgecolor="none",
            transform=ax_p.transAxes, zorder=4, clip_on=False))

    if (1 - dfrac) > 0.005:
        ax_p.add_patch(mpatches.Rectangle(
            (bx + bw * dfrac, by), bw * (1 - dfrac), bh,
            facecolor=AMBER if not dominant else BORDER, edgecolor="none",
            transform=ax_p.transAxes, zorder=4, clip_on=False))

    if dominant:
        lx = bx + bw * dfrac / 2
        ax_p.text(lx, by + bh / 2, f"DISTORTION  {data['distortion_share']:.0f}%",
                  color="white", fontsize=7.5, ha="center", va="center",
                  fontweight="bold", transform=ax_p.transAxes, zorder=5)
        rx = bx + bw * dfrac + bw * (1 - dfrac) / 2
        ax_p.text(rx, by + bh / 2, f"SEL.  {sel_pct:.0f}%",
                  color=FG, fontsize=7, ha="center", va="center",
                  transform=ax_p.transAxes, zorder=5)
    else:
        rx = bx + bw * dfrac + bw * (1 - dfrac) / 2
        ax_p.text(rx, by + bh / 2, f"SELECTION  {sel_pct:.0f}%",
                  color="#0d1117", fontsize=7.5, ha="center", va="center",
                  fontweight="bold", transform=ax_p.transAxes, zorder=5)
        if dfrac > 0.03:
            lx = bx + bw * dfrac / 2
            ax_p.text(lx, by + bh / 2, f"{data['distortion_share']:.0f}%",
                      color=FG, fontsize=6.5, ha="center", va="center",
                      transform=ax_p.transAxes, zorder=5)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


# ── Screenshot paths ──────────────────────────────────────────────────────────

_YB_SCREENSHOT = r"screenshots for kin diagrams\Screenshot 2026-06-22 100819.png"
_LR_SCREENSHOT = r"screenshots for kin diagrams\Screenshot 2026-06-23 102000.png"
_HM_SCREENSHOT = r"screenshots for kin diagrams\Screenshot 2026-06-23 104858.png"
_SH_SCREENSHOT = r"screenshots for kin diagrams\Screenshot 2026-06-23 105031.png"


# ── Load metrics from data ────────────────────────────────────────────────────

print("Loading Yamamoto/Bernabel metrics...")
_YB = load_pitch_metrics(776693, 33, 3)   # CU whiff, 2–0 count



print("Loading Leiter/Ramirez metrics...")
_LR = load_pitch_metrics(777664, 58, 2)   # CU swinging strike, 1–0 count, 2 outs

print("Loading Helsley/Mullins metrics...")
_HM = load_pitch_metrics(716604, 66, 4)   # FF 100 mph above zone, 99.4% selection

print("Loading Sale/Harper metrics...")
_SH = load_pitch_metrics(778406, 2, 2)    # SL 77.7 mph, 40.4" miss, 99.6% selection


# ── Render ────────────────────────────────────────────────────────────────────

Path("results/figures").mkdir(parents=True, exist_ok=True)

# Yamamoto/Bernabel — distortion case (91% distortion)
_yb_dev  = abs(_YB["pc_dev_z_in"])
fig_yb = make_broadcast_annotation(
    _YB_SCREENSHOT, _YB,
    callout_xy     = (360, 345),
    callout_xytext = (520, 180),
    callout_label  = f"−{_yb_dev:.1f}\" below\nprojected path",
    callout_color  = RED,
)
fig_yb.savefig("results/figures/yamamoto_bernabel_annotation.png",
               dpi=180, bbox_inches="tight")
plt.close()
print("Saved: results/figures/yamamoto_bernabel_annotation.png")



# Leiter/Ramirez — 50/50 distortion case (CU -6.0" post-commit drop)
_lr_dev = abs(_LR["pc_dev_z_in"])
fig_lr = make_broadcast_annotation(
    _LR_SCREENSHOT, _LR,
    callout_xy     = (510, 345),
    callout_xytext = (320, 170),
    callout_label  = f"−{_lr_dev:.1f}\" below\nprojected path",
    callout_color  = RED,
)
fig_lr.savefig("results/figures/leiter_ramirez_annotation.png",
               dpi=180, bbox_inches="tight")
plt.close()
print("Saved: results/figures/leiter_ramirez_annotation.png")

# Helsley/Mullins — selection case (99.4% selection, FF 100 mph above zone)
_hm_dev = abs(_HM["pc_dev_z_in"])
fig_hm = make_broadcast_annotation(
    _HM_SCREENSHOT, _HM,
    callout_xy     = (490, 125),
    callout_xytext = (260, 330),
    callout_label  = f"−{_hm_dev:.1f}\" off projected\npure swing decision",
    callout_color  = AMBER,
)
fig_hm.savefig("results/figures/helsley_mullins_annotation.png",
               dpi=180, bbox_inches="tight")
plt.close()
print("Saved: results/figures/helsley_mullins_annotation.png")

# Sale/Harper — selection case (99.6% selection, SL 77.7 mph, 40.4" miss)
_sh_dev = abs(_SH["pc_dev_z_in"])
fig_sh = make_broadcast_annotation(
    _SH_SCREENSHOT, _SH,
    callout_xy     = (700, 375),
    callout_xytext = (430, 160),
    callout_label  = f"−{_sh_dev:.1f}\" off projected\n40.4\" miss — batter decision",
    callout_color  = AMBER,
)
fig_sh.savefig("results/figures/sale_harper_annotation.png",
               dpi=180, bbox_inches="tight")
plt.close()
print("Saved: results/figures/sale_harper_annotation.png")
