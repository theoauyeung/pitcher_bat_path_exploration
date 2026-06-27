# Pitcher Bat-Path Exploration

Causal mediation analysis of how post-commit pitch movement disrupts batter swing shape and costs run value. 

---

## What this does

The purpose of this research is to explore to what extent pitchers distort batter swing shapes - how much run value does a pitcher generate by forcing a batter into a worse swing shape than the pitch that was delivered warranted. How much of the run-value cost of pitch-induced swing deviation can be attributed to that late, unactionable movement - and how much would have occurred regardless, simply because the batter chose to swing.

 This project exploits that timing asymmetry to decompose per-swing run-value loss into two causal channels:

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

Visualization and results scripts (optional, after pipeline — run from project root):

```bash
python results_scripts/05_trajectory_plot.py       # pitch trajectory plots with pre/post-commit split
python results_scripts/06_kinematic_diagram.py     # annotated batter-view kinematic diagrams
python results_scripts/07_intention_diagnostics.py # Phase A model diagnostics (distributions, count/zone effects)
python results_scripts/08_reliability.py           # split-half and year-over-year reliability of distortion tax
python results_scripts/outcome_analysis.py         # outcome rates and swing quality by distortion/selection tax quintile
python results_scripts/causal_chain.py             # mechanism validation, pitcher type quadrant, pitch type profiles
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

###  Batter intended swing (`02_intention_model.py`)

Fits a Bayesian linear mixed-effects model (Bambi/PyMC, ADVI) per swing-shape response:

- **Responses:** `vert_attack_angle`, `horz_attack_angle`, `swing_path_tilt`, `bat_speed`, `swing_length`
- **Fixed effects:** pitch location (`plate_x`, `plate_z`), count, timing (`offset_y_ms`), platoon handedness
- **Random effects:** per-batter intercept + count-pressure slope; per-pitcher intercept (excluded from counterfactual predictions)

The residual `realized − predicted` is the swing-shape deviation used as the Phase B mediator.

### Run-value mediation (`03_causal_models.py`)

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

**Intention model diagnostics** (`results_scripts/07_intention_diagnostics.py`):

| Figure | Contents |
|--------|----------|
| `07a_intention_distributions.png` | Intended vs. realized distributions + deviation histograms per response |
| `07b_count_effects.png` | Mean intended swing shape by count group and (balls × strikes) matrix |
| `07c_zone_heatmaps.png` | Mean intended shape and deviation across the strike zone |
| `07d_fixed_effects.png` | Phase A fixed-effect coefficient table (posterior means + 95% CI) |

**Reliability analysis** (`results_scripts/08_reliability.py`):

| Figure | Contents |
|--------|----------|
| `08_reliability.png` | Split-half (Spearman-Brown) and year-over-year r for each tax metric |

**Annotated kinematic diagrams** (`results_scripts/06_kinematic_diagram.py`):

Pulled from the following videos: (https://baseballsavant.mlb.com/sporty-videos?playId=fe30b4fe-120e-4f6c-a258-a624bc52452f, 
https://baseballsavant.mlb.com/sporty-videos?playId=84f68d2c-ea0d-351c-b752-2d4aec739924, https://baseballsavant.mlb.com/sporty-videos?playId=cf50242d-6c5b-30f4-a051-300f33655ef9, https://baseballsavant.mlb.com/sporty-videos?playId=39e968ef-398c-3112-8856-75799afc21df, https://baseballsavant.mlb.com/sporty-videos?playId=0c3f33b2-cac0-3bdc-b0bb-c4771a8ebe66, https://baseballsavant.mlb.com/sporty-videos?playId=4a531f99-1294-308c-9ff7-0619334db309)

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
