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
python pull_data.py          # pull MLB pitch-by-pitch from mlb_db → data/
python precommit_split.py    # compute pre/post-commit trajectory split
python run_values.py            # build RE24, linear weights
python run_pipeline.py       # Phase A + Phase B → results/xrv_causal.parquet
```

`run_pipeline.py` also accepts `--skip-phase-a` to reload cached Phase A output and `--method vi` for fast ADVI inference (~2 min vs. hours for MCMC).

Visualization scripts (run after pipeline):

```bash
# All paper figures (or pass individual keys: axis, reliability, drivers, etc.)
.venv/bin/python results_scripts/generate_results.py

# Annotated broadcast cards per pitch (requires DB connection)
.venv/bin/python results_scripts/kinematic_diagram.py

# Leaderboard tables with MLB headshots (requires R)
Rscript results_scripts/leaderboard_table.R
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

This project uses [uv](https://docs.astral.sh/uv/) and Python **3.14**. All direct dependencies are pinned in `requirements.txt`.

```bash
# Install uv (once per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh     # Mac/Linux
# or: brew install uv

# Create venv and install
uv venv --python 3.14
source .venv/bin/activate                           # Mac/Linux
# Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

---

## Resuming on a new Mac

### 1 — Get the data bundle

Download `pitcher_bat_path_bundle.zip` from OneDrive (Driveline email account → OneDrive → top level).

### 2 — Clone and unzip

```bash
git clone https://github.com/theoauyeung/pitcher_bat_path_exploration.git
cd pitcher_bat_path_exploration
unzip /path/to/pitcher_bat_path_bundle.zip   # extracts into data/, models/, results/
```

The zip uses repo-relative paths so it unpacks directly into the right directories.

### 3 — Set up Python

```bash
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env   # or open a new terminal

uv venv --python 3.14
source .venv/bin/activate
uv pip install -r requirements.txt
```

If Python 3.14 is not available yet via uv, install it first:
```bash
uv python install 3.14
```

### 4 — Set up R (for leaderboard tables)

```bash
brew install r
# R packages (run once inside R or Rscript):
Rscript -e 'install.packages(c("arrow","dplyr","gt","gtExtras","mlbplotR","scales","webshot2"))'
```

On Mac, `Rscript` will be on PATH after the brew install — no hardcoded path needed. The R script (`results_scripts/leaderboard_table.R`) uses only `Rscript` by name and is cross-platform.

### 5 — Verify

```bash
.venv/bin/python results_scripts/generate_results.py reliability
# → should write results/figures/reliability.png without errors
```

### Machine-specific things that change

| Item | Old (Windows) | New (Mac) |
|------|--------------|-----------|
| Python venv activate | `.venv\Scripts\activate` | `source .venv/bin/activate` |
| Python binary | `.venv\Scripts\python.exe` | `.venv/bin/python` |
| Rscript path | `C:\Users\theo.an-yeung\AppData\Local\Programs\R\R-4.6.0\bin\Rscript.exe` | `Rscript` (on PATH via brew) |
| DB access (`pull_data.py`) | Requires internal network / VPN | Same — needs VPN or on-site |

### What you don't need to rerun

With the bundle in place, you can immediately run `generate_results.py` and `leaderboard_table.R` to regenerate all figures. You only need to rerun the pipeline (`00`–`04`) if you want fresh data from the DB.

### Personal checklist (do this yourself on the new Mac)

- [ ] Sign into OneDrive with your Driveline account and wait for sync to show "Up to date" on `pitcher_bat_path_bundle.zip` before downloading (3.3 GB — allow 10–20 min upload time on the source machine)
- [ ] Download the bundle; confirm the zip is 3.29 GB before unzipping
- [ ] `git clone` and `unzip` as above
- [ ] `uv venv --python 3.14` — if uv can't find 3.14, run `uv python install 3.14` first
- [ ] Run the verify command and confirm it writes `results/figures/reliability.png` without errors
- [ ] Install R and R packages if you want to regenerate leaderboard tables
- [ ] For DB access (`pull_data.py`): connect to VPN or be on-site at Driveline

> **Access risk:** The bundle is in your Driveline work OneDrive. If you lose access to the Driveline account, the bundle is inaccessible. Consider downloading a personal backup copy to a USB or personal cloud before returning the machine.

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
