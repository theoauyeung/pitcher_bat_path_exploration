# Pitcher Bat-Path Exploration

Causal mediation analysis estimating how post-commit pitch movement disrupts batter swing shape and costs run value.

---

## What this is

When a pitcher throws a breaking ball, some of the swing disruption it causes is physically unavoidable — the ball moved after the batter's swing was already committed. Some is the batter's own decision. This project separates the two.

A batter cannot react to ball movement within ~150 ms of contact. Any deviation between the ball's projected and actual plate-crossing location after that window is exogenous to the swing decision. The pipeline uses this to decompose per-swing run-value loss into:

- **Distortion** — run-value cost caused by post-commit pitch movement (spatial displacement + induced swing-plane deviation)
- **Selection** — run-value cost attributable to the batter's own swing decision

---

## Pipeline

```bash
# 1. Pull Statcast data (R ≥ 4.0, sabRmetrics, tidyverse)
Rscript src/data/pull_savant.R          # data/savant_raw.csv + results/linear_weights.csv

# 2. Process swings
.venv/bin/python src/data/process_savant.py     # data/swings_2023_2025.csv

# 3. Pre/post-commit trajectory split
.venv/bin/python src/data/precommit_split.py    # data/swings_precommit.parquet

# 4. Phase A + Phase B causal pipeline
.venv/bin/python src/models/run_pipeline.py     # results/xrv_causal.parquet + leaderboards
```

`run_pipeline.py` accepts `--skip-phase-a` to reload cached Phase A output without refitting, and `--method vi` for fast ADVI inference (~2 min vs. hours for MCMC).

---

## Methodology

### Pre/post-commit trajectory split

Each pitch's full flight path is reconstructed from release parameters. For each swing, we find where the ball *would have* crossed the plate on a constant-acceleration trajectory from commit time forward. The gap between this projected location and the actual plate crossing is the post-commit deviation — movement the batter had no time to respond to.

Commit time is set at 150 ms pre-contact. This is intentionally conservative (understates late movement) so the effect is not overstated. The pipeline runs a sensitivity grid over 125–200 ms; the relative ordering between pitchers is stable across the grid.

### Batter intended swing (Phase A)

For each of five swing-shape responses (vertical and horizontal attack angle, swing path tilt, bat speed, swing length), a Bayesian linear mixed-effects model is fit using pitch location, count, contact timing, and platoon handedness. Per-batter random effects capture each batter's baseline tendencies and their count-pressure adjustments.

The residual — realized minus predicted — is the swing deviation used as the Phase B mediator.

Default inference is ADVI (`method="vi"`, ~2 min). Only posterior means are used downstream so VI is equivalent to MCMC here.

### Run-value mediation (Phase B)

**Mediator models** (OLS) estimate how much of each angular swing deviation was mechanically forced by post-commit movement. The treatment coefficients give causal leverage: degrees of swing deviation per foot of late movement.

**Outcome models** (XGBoost) price swing deviation in run value across three channels:
- P(ball in play)
- P(foul | not in play) — separated from whiff because a two-strike foul extends the PA
- E[exit velocity | BIP] → E[launch angle | BIP] → E[run value | BIP] via linear weights

The BIP channel runs through explicit EV/LA predictions rather than a direct xwOBA model, so the distortion story is interpretable: "late movement cost X mph of exit velocity and Y° of launch angle, worth Z runs."

**Disruption tax** is computed via three counterfactual scenarios:

| Scenario | Swing angles | Plate location |
|----------|-------------|----------------|
| Realized | actual deviations | actual (post-movement) |
| Spatial only | zero deviations | actual (post-movement) |
| Intended | zero deviations | projected (pre-movement) |

```
disruption_tax          = xrv_realized − xrv_intended
spatial_distortion      = xrv_spatial  − xrv_intended
angular_disruption      = xrv_realized − xrv_spatial
distortion_tax          = spatial_distortion + angular_disruption × angular_distortion_share
selection_tax           = angular_disruption × (1 − angular_distortion_share)
adjusted_disruption_tax = disruption_tax − max(0, decision_cost)
```

`angular_distortion_share` uses squared-norm decomposition across VAA, HAA, and swing-path tilt. `adjusted_disruption_tax` shifts the baseline to `take_xrv` when the batter should have taken the pitch.

---

## Repository structure

```
src/
  data/     pull_savant.R, process_savant.py, precommit_split.py
  models/   intention_model.py, causal_models.py, run_pipeline.py
  viz/      eda.ipynb, intention_model.ipynb, causal_model.ipynb,
            validation.ipynb, results.ipynb,
            kinematic_diagram.py, leaderboard_table.R
  utils/    db.py
data/       savant_raw.csv, swings_2023_2025.csv, swings_precommit.parquet  (not tracked)
models/     *.joblib, *.parquet  (not tracked)
results/
  *.csv                    leaderboards, linear weights, RE24, validity checks
  plots/
    leaderboards/          distortion tax leaderboard tables
    validation/            reliability, outcome rates, xwOBA relationship
    diagnostics/           axis fingerprint, count effects, fixed effects
    case_studies/          annotated broadcast cards
docs/       design notes for the mediation model, intention model, and output metrics
tests/      unit tests for process_savant.py and the EV/LA model chain
```

---

## Key outputs

| File | Contents |
|------|----------|
| `results/xrv_causal.parquet` | Per-swing: disruption tax, distortion tax, selection tax, spatial distortion, EV/LA realized and intended, EV suppression, decision cost |
| `results/distortion_pitcher.csv` | Pitcher-level leaderboard (≥500 swings) |
| `results/distortion_batter.csv` | Batter-level leaderboard (≥500 swings) |
| `results/plots/leaderboards/` | Leaderboard tables with MLB headshots (R/gt) |
| `results/plots/validation/` | Reliability, outcome rates, xwOBA relationship, convergent/incremental validity |
| `results/plots/diagnostics/` | Axis fingerprint, count effects, fixed effects, physical drivers |
| `results/plots/case_studies/` | Annotated broadcast cards — Yamamoto, Leiter, Helsley, Sale |

---

## Data

Source: Baseball Savant (MLB Statcast), 2023–2025 regular season, pulled via the [`sabRmetrics`](https://github.com/drivelineresearch/sabRmetrics) R package. ~763k competitive swings after filtering (bat speed ≥ 50 mph).

Large files (`data/`, `models/*.joblib`, `results/*.parquet`) are not tracked. Run the pipeline to regenerate.

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv python install 3.14
uv venv --python 3.14
source .venv/bin/activate
uv pip install -r requirements.txt
```

R packages (for data pull and leaderboard tables):

```r
install.packages(c("sabRmetrics", "tidyverse", "arrow", "dplyr",
                   "gt", "gtExtras", "mlbplotR", "scales", "webshot2"))
```

---

## Kinematic diagrams

`src/viz/kinematic_diagram.py` generates two-panel broadcast cards: game screenshot with callout arrow (left) + metrics panel (right). The metrics panel shows post-commit drop, projected vs. actual plate crossing, disruption/decision/total burden, and the distortion/selection split.

Requires a DB connection for the game video frames. The four case studies in `results/plots/case_studies/` are pre-generated.

| Pitcher / Batter | Pitch | Dominant cause |
|-----------------|-------|----------------|
| Yamamoto / Bernabel | Curveball | Distortion (99.7%) |
| Leiter / Ramirez | Curveball | Mixed |
| Helsley / Mullins | Four-seam FB | Selection (95%) |
| Sale / Harper | Slider | Mixed + chase penalty |
