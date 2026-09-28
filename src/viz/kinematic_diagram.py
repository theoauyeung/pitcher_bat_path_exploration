"""
Kinematic diagram: post-commit trajectory distortion.


Run:
    .venv/Scripts/python.exe 06_kinematic_diagram.py

Saves: results/plots/case_studies/yamamoto_bernabel_annotation.png
       results/plots/case_studies/bradley_alonso_annotation.png
"""

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
    pc = pd.read_parquet(precommit_path)
    cx = pd.read_parquet(causal_path)

    rp = pc.loc[(pc["game_pk"] == game_pk) &
                (pc["at_bat_number"] == at_bat_number) &
                (pc["pitch_number"] == pitch_number)].iloc[0]
    rc = cx.loc[(cx["game_pk"] == game_pk) &
                (cx["at_bat_number"] == at_bat_number) &
                (cx["pitch_number"] == pitch_number)].iloc[0]

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
    )


# ── Broadcast annotation ──────────────────────────────────────────────────────

def make_broadcast_annotation(
    screenshot_path, data,
    callout_xy=None, callout_xytext=None, callout_label=None,
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

    img_frac   = 0.55
    panel_frac = 0.45
    fig_h      = 9.0
    fig_w      = fig_h * (iw / ih) / img_frac

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    # ── Image ─────────────────────────────────────────────────────────────────
    ax_img = fig.add_axes([0, 0, img_frac, 1.0])
    ax_img.imshow(img, aspect="auto", extent=[0, iw, ih, 0])
    ax_img.set_xlim(0, iw); ax_img.set_ylim(ih, 0); ax_img.axis("off")

    # ── Metrics panel ─────────────────────────────────────────────────────────
    ax_p = fig.add_axes([img_frac, 0, panel_frac, 1.0])
    ax_p.set_facecolor(BG); ax_p.set_xlim(0, 1); ax_p.set_ylim(0, 1); ax_p.axis("off")
    ax_p.axvline(0.0, color=BORDER, lw=1.0, zorder=1)

    xs, xe = 0.08, 0.95

    def hline(y, color=BORDER, lw=0.7):
        ax_p.plot([xs - 0.04, xe + 0.02], [y, y], color=color, lw=lw,
                  transform=ax_p.transAxes, zorder=2)

    def row(y, label, val, vc=FG, ls=12, vs=13):
        ax_p.text(xs, y, label, color=GRAY, fontsize=ls, va="top",
                  transform=ax_p.transAxes)
        ax_p.text(xe, y, val, color=vc, fontsize=vs, va="top", ha="right",
                  fontweight="bold", transform=ax_p.transAxes)

    def section_title(y, title, tc=BLUE):
        ax_p.text(xs, y, title, color=tc, fontsize=13, fontweight="bold",
                  va="top", transform=ax_p.transAxes)
        hline(y - 0.022, color=tc, lw=1.2)
        return y - 0.040

    rs  = 0.044   # row spacing
    pad = 0.010   # inter-section gap

    # header
    y = 0.96
    ax_p.text(xs, y, data["pitcher"], color=FG, fontsize=16, fontweight="bold",
              va="top", transform=ax_p.transAxes); y -= 0.052
    ax_p.text(xs, y, f"vs.  {data['batter']}", color=BLUE, fontsize=13,
              va="top", transform=ax_p.transAxes); y -= 0.042
    ax_p.text(xs, y, f"{data['game_date']}   ·   {data['count']} count",
              color=GRAY, fontsize=11, va="top", transform=ax_p.transAxes); y -= 0.036
    ax_p.text(xs, y, data["pitch_type"], color=AMBER, fontsize=12, fontweight="bold",
              va="top", transform=ax_p.transAxes); y -= 0.038
    hline(y, BORDER); y -= 0.022

    # pitch profile
    y = section_title(y, "PITCH PROFILE", BLUE)
    row(y, "Velocity",       f"{data['release_speed']:.1f} mph",    AMBER); y -= rs
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

    # disruption analysis
    y = section_title(y, "DISRUPTION ANALYSIS", RED)
    dev_in    = abs(data["pc_dev_z_in"])
    proj_in   = data["pc_z_proj_ft"] * 12
    act_in    = data["pc_z_actual_ft"] * 12
    sz_top_in = data["sz_top"] * 12
    above_in  = proj_in - sz_top_in

    rs_d = 0.040  # tighter spacing for this section to fit 5 rows
    loc_note = f'+{above_in:.1f}" above zone' if above_in > 0 else "in zone"
    row(y, "Post-commit drop", f"−{dev_in:.1f} in", RED); y -= rs_d
    row(y, "Proj. → actual",
        f'{proj_in:.1f}" ({loc_note}) → {act_in:.1f}"', RED); y -= rs_d

    dt = data["disruption_tax"]
    row(y, "Disruption Tax",
        f"{dt:+.3f} runs", RED if dt < 0 else AMBER); y -= rs_d

    dc = data["decision_cost"]
    dc_color = RED if dc > 0 else GREEN
    dc_str   = f"{dc:+.3f} runs"
    row(y, "Decision Cost", dc_str, dc_color); y -= rs_d

    adj = data["adjusted_disruption_tax"]
    row(y, "Adj Disruption Tax",
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
                  color="white", fontsize=11, ha="center", va="center",
                  fontweight="bold", transform=ax_p.transAxes, zorder=5)
        rx = bx + bw * dfrac + bw * (1 - dfrac) / 2
        ax_p.text(rx, by + bh / 2, f"SEL.  {sel_pct:.0f}%",
                  color=FG, fontsize=10.5, ha="center", va="center",
                  transform=ax_p.transAxes, zorder=5)
    else:
        rx = bx + bw * dfrac + bw * (1 - dfrac) / 2
        ax_p.text(rx, by + bh / 2, f"SELECTION  {sel_pct:.0f}%",
                  color="#0d1117", fontsize=11, ha="center", va="center",
                  fontweight="bold", transform=ax_p.transAxes, zorder=5)
        if dfrac > 0.03:
            lx = bx + bw * dfrac / 2
            ax_p.text(lx, by + bh / 2, f"{data['distortion_share']:.0f}%",
                      color=FG, fontsize=10, ha="center", va="center",
                      transform=ax_p.transAxes, zorder=5)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


# ── Screenshot paths ──────────────────────────────────────────────────────────

_YB_SCREENSHOT = "docs/screenshots/Screenshot 2026-09-27 at 10.00.02 PM.png"
_LR_SCREENSHOT = "docs/screenshots/Screenshot 2026-06-23 102000.png"
_HM_SCREENSHOT = "docs/screenshots/Screenshot 2026-06-23 104858.png"
_SH_SCREENSHOT = "docs/screenshots/Screenshot 2026-06-23 105031.png"


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

Path("results/plots/case_studies").mkdir(parents=True, exist_ok=True)

# Yamamoto/Bernabel — distortion case (91% distortion)
_yb_dev  = abs(_YB["pc_dev_z_in"])
fig_yb = make_broadcast_annotation(
    _YB_SCREENSHOT, _YB,
    callout_xy     = (360, 345),
    callout_xytext = (520, 180),
    callout_label  = f"−{_yb_dev:.1f}\" below\nprojected path",
    callout_color  = RED,
)
fig_yb.savefig("results/plots/case_studies/yamamoto_bernabel_annotation.png",
               dpi=180, bbox_inches="tight")
plt.close()
print("Saved: results/plots/case_studies/yamamoto_bernabel_annotation.png")



# Leiter/Ramirez — 50/50 distortion case (CU -6.0" post-commit drop)
_lr_dev = abs(_LR["pc_dev_z_in"])
fig_lr = make_broadcast_annotation(
    _LR_SCREENSHOT, _LR,
    callout_xy     = (510, 345),
    callout_xytext = (320, 170),
    callout_label  = f"−{_lr_dev:.1f}\" below\nprojected path",
    callout_color  = RED,
)
fig_lr.savefig("results/plots/case_studies/leiter_ramirez_annotation.png",
               dpi=180, bbox_inches="tight")
plt.close()
print("Saved: results/plots/case_studies/leiter_ramirez_annotation.png")

# Helsley/Mullins — selection case (99.4% selection, FF 100 mph above zone)
_hm_dev = abs(_HM["pc_dev_z_in"])
fig_hm = make_broadcast_annotation(
    _HM_SCREENSHOT, _HM,
    callout_xy     = (490, 125),
    callout_xytext = (260, 330),
    callout_label  = f"−{_hm_dev:.1f}\" off projected\npure swing decision",
    callout_color  = AMBER,
)
fig_hm.savefig("results/plots/case_studies/helsley_mullins_annotation.png",
               dpi=180, bbox_inches="tight")
plt.close()
print("Saved: results/plots/case_studies/helsley_mullins_annotation.png")

# Sale/Harper — selection case (99.6% selection, SL 77.7 mph, 40.4" miss)
_sh_dev = abs(_SH["pc_dev_z_in"])
fig_sh = make_broadcast_annotation(
    _SH_SCREENSHOT, _SH,
    callout_xy     = (700, 375),
    callout_xytext = (430, 160),
    callout_label  = f"−{_sh_dev:.1f}\" off projected\n40.4\" miss — batter decision",
    callout_color  = AMBER,
)
fig_sh.savefig("results/plots/case_studies/sale_harper_annotation.png",
               dpi=180, bbox_inches="tight")
plt.close()
print("Saved: results/plots/case_studies/sale_harper_annotation.png")
