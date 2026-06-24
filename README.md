# Pitcher Bat-Path Exploration

Causal mediation analysis of how post-commit pitch movement disrupts batter swing shape and costs run value. Developed at Driveline Baseball.

---

## What this does

A batter commits to their swing before the ball reaches them. Any pitch movement that happens *after* that commitment point is invisible to the batter's decision — they can't select against it. This project exploits that timing asymmetry to decompose per-swing run-value loss into two causal channels:

- **Distortion** — swing deviation mechanically caused by post-commit pitch movement
- **Selection** — swing deviation attributable to the batter's own decision (e.g., misjudging pitch type, late recognition)

The causal chain is:

```
post-commit movement  →  swing-shape deviation  →  run value
    (treatment)              (mediator)             (outcome)
```

---

## Pipeline

Run scripts in order:

```bash
python 00_pull_data.py          # pull MLB pitch-by-pitch data from mlb_db → data/
python 01_precommit_split.py    # compute pre/post-commit trajectory split → data/swings_precommit.parquet
python 02_intention_model.py    # Phase A: fit batter intended-swing LMMs → models/
python 03_causal_models.py      # Phase B: fit mediation + outcome models → models/
python 04_run_pipeline.py       # orchestrate Phase A → B → results/xrv_causal.parquet
```

Visualization (optional, after pipeline):

```bash
python 05_trajectory_plot.py       # pitch trajectory plots with pre/post-commit split
python 06_kinematic_diagram.py     # annotated batter-view kinematic diagrams
python 07_intention_diagnostics.py # Phase A model diagnostics (distributions, count/zone effects)
```

---

## Key outputs

| File | Contents |
|------|----------|
| `results/xrv_causal.parquet` | Per-swing disruption / distortion / selection tax |
| `results/distortion_pitcher.csv` | Pitcher-level distortion leaderboard (≥50 swings) |
| `results/distortion_batter.csv` | Batter-level disruption leaderboard (≥50 swings) |
| `results/figures/` | Kinematic diagrams, trajectory plots, and intention model diagnostics |

---

## Architecture

### Phase A — Batter intended swing (`02_intention_model.py`)

Fits a Bayesian linear mixed-effects model (Bambi/PyMC, ADVI) per swing-shape response:

- **Responses:** `vert_attack_angle`, `horz_attack_angle`, `swing_path_tilt`, `bat_speed`, `swing_length`
- **Fixed effects:** pitch location (`plate_x`, `plate_z`), count, timing (`offset_y_ms`), platoon handedness
- **Random effects:** per-batter intercept + count-pressure slope; per-pitcher intercept (excluded from counterfactual predictions)

The residual `realized − predicted` is the swing-shape deviation used as the Phase B mediator.

### Phase B — Run-value mediation (`03_causal_models.py`)

Three outcome channels:
- **P(BIP)** — logistic contact model
- **P(foul | not BIP)** — logistic foul model (separate from whiff at 2 strikes)
- **E[xwOBAcon | BIP]** — OLS on balls in play

Composite expected run value:
```
xRV = P(BIP)·E[xwOBA|BIP] + P(foul)·foul_rv[count] + P(whiff)·whiff_rv[count]
```

Disruption tax = `xRV(realized) − xRV(intended at zero deviation)`.

Distortion share uses squared-norm decomposition:
```
distortion_share = ‖distortion_dev‖² / ‖total_dev‖²
```

---

## Data

Source: `mlb_db` (internal Driveline MySQL), MLB regular-season 2023–2025. Requires internal network access.

Large data files (`data/`, `models/*.joblib`, `results/*.parquet` > 25 MB) are not tracked in this repo. Re-generate by running the pipeline.

---

## Dependencies

Managed via `.venv` (project-local):

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
```

Core packages: `bambi`, `pymc`, `statsmodels`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `sqlalchemy`, `pymysql`.

---

## Figures

**Intention model diagnostics** (`07_intention_diagnostics.py`):

| Figure | Contents |
|--------|----------|
| `07a_intention_distributions.png` | Intended vs. realized distributions + deviation histograms per response |
| `07b_count_effects.png` | Mean intended swing shape by count group and (balls × strikes) matrix |
| `07c_zone_heatmaps.png` | Mean intended shape and deviation across the strike zone |

**Annotated kinematic diagrams** (`06_kinematic_diagram.py`):

Each figure is a two-panel broadcast card: game screenshot with arrow callout (left) + dark metrics panel (right). The **DISRUPTION ANALYSIS** section shows:

- **Post-commit drop** — vertical inches the ball moves after the batter commits (~150 ms pre-contact). This is movement the batter cannot react to; a splitter dropping 6" post-commit means the swing plane was set 6" too high through no fault of their read.
- **Proj. → actual** — the ball's projected plate-crossing height (what the batter's brain used to set swing plane) vs. its actual height after late movement. The parenthetical shows whether the *projected* location was already above or below the zone before any late break. The gap between the two numbers equals the post-commit drop.
- **Disruption tax** — run-value cost of the swing deviation, in runs (negative = pitcher advantage). Computed by predicting the batter's xRV twice — once with the actual swing deviations, once with all deviations zeroed — and taking the difference.
- **Distortion / Selection bar** — what fraction of the disruption was mechanically caused by post-commit pitch movement (distortion, red) vs. the batter's own decision (selection, amber).

| Pitcher / Batter | Pitch | Dominant cause |
|-----------------|-------|----------------|
| Yamamoto / Bernabel | Splitter | Distortion |
| Leiter / Ramirez | Curveball | Mixed |
| Helsley / Mullins | Sweeper | Selection |
| Sale / Harper | Slider | Selection |

See `results/figures/` for full-resolution PNGs.
