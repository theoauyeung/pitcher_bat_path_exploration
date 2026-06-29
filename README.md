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

```bash
python 00_pull_data.py          # pull MLB pitch-by-pitch from mlb_db → data/
python 01_precommit_split.py    # compute pre/post-commit trajectory split
python run_values.py            # build RE24, linear weights
python 04_run_pipeline.py       # Phase A + Phase B → results/xrv_causal.parquet
```

`04_run_pipeline.py` also accepts `--skip-phase-a` to reload cached Phase A output and `--method vi` for fast ADVI inference (~2 min vs. hours for MCMC).

Visualization scripts (run after pipeline):

```bash
python results_scripts/06_kinematic_diagram.py     # annotated broadcast cards per pitch
python results_scripts/07_intention_diagnostics.py # Phase A model diagnostics
```

---

## Key outputs

| File | Contents |
|------|----------|
| `results/xrv_causal.parquet` | Per-swing disruption / adjusted disruption / distortion / selection / spatial distortion / miss / decision cost |
| `results/distortion_pitcher.csv` | Pitcher-level distortion leaderboard (≥50 swings) |
| `results/distortion_batter.csv` | Batter-level disruption leaderboard (≥50 swings) |
| `results/figures/` | Kinematic diagrams and intention model diagnostics |

---

## Methodology

### Step 1 — Pre/post-commit trajectory split

Each pitch's full flight path is reconstructed from release parameters. For each swing, we compute where the ball *would have* crossed the plate had it continued on a constant-acceleration trajectory from commit time forward. The gap between this projected location and the actual plate crossing is the post-commit deviation — movement the batter had no time to respond to.

Commit time is set conservatively at 150 ms pre-contact to understate rather than overstate late movement. The robustness grid over 125–200 ms treats this as a sensitivity check.

### Step 2 — Batter intended swing

For each of five swing-shape responses (vertical and horizontal attack angle, swing path tilt, bat speed, swing length), we fit a Bayesian linear mixed-effects model using pitch location, count, contact timing, and platoon handedness as predictors. Per-batter random effects capture each batter's baseline tendencies and how they adjust under count pressure.

The residual — realized minus predicted — is the swing deviation used as the mediator in Step 3.

### Step 3 — Run-value mediation

**Mediator models** estimate how much of each angular swing deviation is mechanically caused by post-commit movement. The treatment coefficients give the causal leverage — how many degrees of swing deviation does one foot of late movement produce.

**Outcome models** price swing deviation in run value via three channels: P(ball in play), P(foul | not in play), and E[xwOBA | ball in play]. Foul and whiff are modeled separately because at two strikes a foul keeps the at-bat alive while a whiff ends it.

**Disruption tax** uses three counterfactual scenarios:

| Scenario | Swing angles | Plate location |
|----------|-------------|----------------|
| Realized | actual deviations | actual (post-movement) |
| Spatial only | zero deviations | actual (post-movement) |
| Intended | zero deviations | projected (pre-movement) |

This lets us decompose the total disruption tax into spatial distortion (the ball ended up somewhere different than the batter expected) and angular disruption (the batter's swing plane was knocked off-target). The angular component is further split by how much was mechanically caused by movement vs. the batter's own decision.

The pipeline also computes **physical miss** (bat-to-ball contact quality degradation from late movement), **decision cost** (opportunity cost of swinging vs. taking at the projected location), and **adjusted disruption tax** (total batter burden vs. the optimal action — `disruption_tax − max(0, decision_cost)`).

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

Core packages: `bambi`, `pymc`, `statsmodels`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `sqlalchemy`, `pymysql`.

---

## Kinematic diagrams

Each figure is a two-panel broadcast card: game screenshot with arrow callout (left) + dark metrics panel (right). The **DISRUPTION ANALYSIS** section shows:

- **Post-commit drop** — vertical inches the ball moves after the batter commits. Movement the batter cannot react to.
- **Proj. → actual** — the ball's projected vs. actual plate-crossing height, with zone context.
- **Swing disruption** — run-value cost conditional on the decision to swing (`disruption_tax`; negative = pitcher advantage).
- **Decision / Chase cost** — opportunity cost of swinging vs. taking at the projected location. Green when swinging was correct; red when taking was better.
- **Total burden** — `adjusted_disruption_tax = disruption_tax − max(0, decision_cost)`. The headline metric: total batter cost vs. the optimal available action.
- **Distortion / Selection bar** — fraction of swing disruption caused by post-commit movement (red) vs. the batter's own decision (amber).

| Pitcher / Batter | Pitch | Dominant cause |
|-----------------|-------|----------------|
| Yamamoto / Bernabel | Curveball | Distortion (99.7%) |
| Leiter / Ramirez | Curveball | Mixed |
| Helsley / Mullins | Four-seam FB | Selection (95%) |
| Sale / Harper | Slider | Mixed + chase penalty |
