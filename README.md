# Pitcher Bat-Path Exploration

Causal mediation analysis estimating how post-commit pitch movement disrupts batter swing shape and costs run value.

---

## Research question

When a pitcher throws a breaking ball, some of the swing deviation it induces is physically unavoidable — the ball moved after the batter's swing was already committed. Some is attributable to the batter's own decision. This project separates the two.

We exploit the neuromuscular timing asymmetry: a batter cannot react to ball movement that occurs within ~150 ms of contact. Any deviation between the ball's projected and actual plate-crossing location after that window is exogenous to the swing decision. We use this to decompose per-swing run-value loss into:

- **Distortion** — run-value cost caused by post-commit pitch movement (spatial displacement + induced swing-plane deviation)
- **Selection** — run-value cost attributable to the batter's own swing decision

---

## Pipeline

Run scripts in order from project root:

```bash
python 00_pull_data.py          # pull MLB pitch-by-pitch data from mlb_db → data/
python 01_precommit_split.py    # compute pre/post-commit trajectory split → data/swings_precommit.parquet
python 02_intention_model.py    # Phase A: fit batter intended-swing LMMs → models/
python 03_causal_models.py      # Phase B: fit mediation + outcome models → models/
python 04_run_pipeline.py       # orchestrate Phase A → B → results/xrv_causal.parquet
```

Visualization scripts (run from project root after pipeline):

```bash
python results_scripts/06_kinematic_diagram.py     # annotated broadcast cards per pitch
python results_scripts/07_intention_diagnostics.py # Phase A model diagnostics
```

---

## Key outputs

| File | Contents |
|------|----------|
| `results/xrv_causal.parquet` | Per-swing disruption / distortion / selection / spatial distortion tax |
| `results/distortion_pitcher.csv` | Pitcher-level distortion leaderboard (≥50 swings) |
| `results/distortion_batter.csv` | Batter-level disruption leaderboard (≥50 swings) |
| `results/figures/` | Kinematic diagrams and intention model diagnostics |

---

## Methodology

### Step 1 — Pre/post-commit trajectory split (`01_precommit_split.py`)

Each pitch is modeled as a nine-parameter constant-acceleration trajectory anchored at release. For each swing, we compute where the ball would have crossed the plate if it had continued on that trajectory from commit time onward (`x_proj`, `z_proj`). The gap between this projected location and the actual plate crossing (`dev_x`, `dev_z`) is the post-commit deviation — movement the batter had no time to respond to.

Commit time is set conservatively at 150 ms pre-contact. This understates post-commit movement rather than overstating it, ensuring any measured distortion effect is a lower bound.

### Step 2 — Batter intended swing (`02_intention_model.py`)

For each of five swing-shape responses (`vert_attack_angle`, `horz_attack_angle`, `swing_path_tilt`, `bat_speed`, `swing_length`), we fit a Bayesian linear mixed-effects model (ADVI) using:

- Pitch location and height
- Ball-strike count
- Contact timing
- Platoon handedness

Random effects capture each batter's baseline swing tendency and how they adjust under count pressure. The residual — realized minus predicted — is the swing deviation used as the mediator in Step 3.

### Step 3 — Run-value mediation (`03_causal_models.py`)

**Mediator models** estimate how much of each swing deviation is mechanically caused by post-commit movement, using the treatment (post-commit deviation) and pre-commit projected location as predictors.

**Outcome models** price swing deviation in run value via three contact channels:

- P(ball in play) — logistic regression
- P(foul | not in play) — logistic regression on non-BIP swings (kept separate from whiff because at two strikes, a foul keeps the at-bat alive while a whiff ends it)
- E[xwOBA | ball in play] — OLS on balls in play

**Composite expected run value:**
```
xRV = P(BIP) · E[xwOBA|BIP] + P(foul) · foul_rv[count] + P(whiff) · whiff_rv[count]
```

**Disruption tax** uses a three-scenario counterfactual. We evaluate xRV three times per swing:

| Scenario | Swing angles | Plate location |
|----------|-------------|----------------|
| Realized | actual deviations | actual (post-movement) |
| Spatial only | zero deviations | actual (post-movement) |
| Intended | zero deviations | projected (pre-movement) |

```
disruption_tax       = xRV(realized)  − xRV(intended)
spatial_distortion   = xRV(spatial)   − xRV(intended)   # cost of location shift alone
angular_disruption   = xRV(realized)  − xRV(spatial)    # cost of swing-plane deviation on top
```

The angular component is further split by how much of the deviation was mechanically caused by movement vs. the batter's own decision, using squared-norm decomposition across the three angular axes. Spatial disruption is fully attributed to distortion by construction.

```
distortion_tax = spatial_distortion + angular_disruption × angular_distortion_share
selection_tax  = angular_disruption × (1 − angular_distortion_share)
```

---

## Data

Source: `mlb_db` (internal Driveline MySQL), MLB regular-season 2023–2025. Requires internal network access. ~763k competitive swings after filtering (bat speed ≥ 50 mph).

Large files (`data/`, `models/*.joblib`, `results/*.parquet` > 25 MB) are not tracked. Re-generate by running the pipeline.

---

## Dependencies

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
```

Core packages: `bambi`, `pymc`, `statsmodels`, `pandas`, `numpy`, `matplotlib`, `sqlalchemy`, `pymysql`.

---

## Kinematic diagrams (`results_scripts/06_kinematic_diagram.py`)

Each figure is a two-panel broadcast card: game screenshot with arrow callout (left) + dark metrics panel (right). The **DISRUPTION ANALYSIS** section shows:

- **Post-commit drop** — vertical inches the ball moves after the batter commits (~150 ms pre-contact). Movement the batter cannot react to; a splitter dropping 6" post-commit means the swing plane was set 6" too high through no fault of their read.
- **Proj. → actual** — the ball's projected plate-crossing height (what the batter's brain used to set swing plane) vs. its actual height after late movement. The parenthetical notes whether the projected location was already above or below the strike zone before any late break.
- **Disruption tax** — run-value cost of the swing deviation, in runs (negative = pitcher advantage). Computed by predicting xRV twice — once with actual swing deviations, once with all deviations zeroed — and taking the difference.
- **Distortion / Selection bar** — fraction of disruption caused by post-commit movement (distortion, red) vs. the batter's own decision (selection, amber).

| Pitcher / Batter | Pitch | Dominant cause |
|-----------------|-------|----------------|
| Yamamoto / Bernabel | Splitter | Distortion |
| Leiter / Ramirez | Curveball | Mixed |
| Helsley / Mullins | Sweeper | Selection |
| Sale / Harper | Slider | Selection |
