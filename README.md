# Pitcher Bat-Path Exploration

Causal mediation analysis estimating how post-commit pitch movement disrupts batter swing shape and costs run value.

---

## Research question

When a pitcher throws a breaking ball, some of the swing deviation it induces is physically unavoidable — the ball moved after the batter's swing was already committed. Some is attributable to the batter's own decision. This project separates the two.

A batter cannot react to ball movement that occurs within ~150 ms of contact. Any deviation between the ball's projected and actual plate-crossing location after that window is exogenous to the swing decision. We use this to decompose per-swing run-value loss into:

- **Distortion** — run-value cost caused by post-commit pitch movement (spatial displacement + induced swing-plane deviation)
- **Selection** — run-value cost attributable to the batter's own swing decision

---

## Pipeline

```bash
.venv/bin/python pull_data.py          # pull MLB pitch-by-pitch from mlb_db → data/
.venv/bin/python precommit_split.py    # compute pre/post-commit trajectory split
.venv/bin/python run_values.py         # build RE24, linear weights
.venv/bin/python run_pipeline.py       # Phase A + Phase B → results/xrv_causal.parquet
```

`run_pipeline.py` accepts `--skip-phase-a` to reload cached Phase A output without refitting, and `--method vi` for fast ADVI inference (~2 min vs. hours for MCMC).

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

**Mediator models** estimate how much of each angular swing deviation is mechanically caused by post-commit movement. The treatment coefficients give the causal leverage: degrees of swing deviation per foot of late movement.

**Outcome models** price swing deviation in run value via three channels: P(ball in play), P(foul | not in play), and E[xwOBA | ball in play]. Foul and whiff are modeled separately because at two strikes a foul keeps the at-bat alive while a whiff ends it.

**Disruption tax** uses three counterfactual scenarios:

| Scenario | Swing angles | Plate location |
|----------|-------------|----------------|
| Realized | actual deviations | actual (post-movement) |
| Spatial only | zero deviations | actual (post-movement) |
| Intended | zero deviations | projected (pre-movement) |

Total disruption decomposes into spatial distortion (the ball arrived somewhere different than the batter expected) and angular disruption (the batter's swing plane was knocked off-target). The angular component splits further based on how much was mechanically forced by movement vs. the batter's own decision.

The pipeline also computes **physical miss** (bat-to-ball separation on contact and whiffs), **decision cost** (opportunity cost of swinging vs. taking at the projected location), and **adjusted disruption tax**: `disruption_tax − max(0, decision_cost)`. The adjusted metric captures total batter cost vs. the optimal available action.

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
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Resuming on a new Mac

> **No Driveline account or DB access needed** — everything required for day-to-day analysis (regenerating all figures, running the R leaderboard tables, editing the results paper) is self-contained once you have the data bundle. Only `pull_data.py` (fresh data pull) and `kinematic_diagram.py` (broadcast cards) require Driveline network/VPN.

### What requires Driveline access vs. what doesn't

| Task | Needs Driveline? |
|------|-----------------|
| Regenerate all paper figures | No |
| Regenerate R leaderboard tables | No |
| Edit results draft | No |
| Rerun Phase A + Phase B models | No (cached in bundle) |
| Pull fresh MLB data (`pull_data.py`) | Yes — internal DB/VPN |
| Generate kinematic broadcast cards | Yes — internal DB |

When you do run `pull_data.py` at Driveline or via VPN, set `DATA_EXPORT_DIR` once in `~/.claude/.env` and the output CSV copies to personal cloud automatically on every pull:

```
# ~/.claude/.env
DATA_EXPORT_DIR=/Users/theo/Library/CloudStorage/iCloudDrive/pitcher_data
# or Google Drive: DATA_EXPORT_DIR=/Users/theo/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive/pitcher_data
```

The directory must already exist. The script prints confirmation after copying.

---

### Get the data bundle (browser — no desktop app needed)

The data bundle (`pitcher_bat_path_bundle.zip`, 3.29 GB) is in **Driveline OneDrive**. Download from any browser — no OneDrive desktop app or email client required:

1. Go to **onedrive.com** (or **office.com → OneDrive**) and sign in with `theo.an-yeung@drivelinebaseball.com`
2. In *My files* (root level), find **`pitcher_bat_path_bundle.zip`** and download it
3. Confirm the downloaded file is 3.29 GB before continuing

---

### 1 — Clone the repo

```bash
git clone https://github.com/theoauyeung/pitcher_bat_path_exploration.git
cd pitcher_bat_path_exploration
```

No credentials needed — the repo is public.

### 2 — Unzip the bundle into the repo

```bash
unzip /path/to/pitcher_bat_path_bundle.zip
# extracts data/, models/, results/ directly into the repo root
```

Confirm these exist after unzipping:
- `data/swings_precommit.parquet` (617 MB)
- `models/causal_models.joblib` (1.5 GB)
- `results/xrv_causal.parquet` (52 MB)

### 3 — Set up Python

```bash
# Install uv (package + Python version manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env    # or open a new terminal

# Install Python 3.14 and create the venv
uv python install 3.14
uv venv --python 3.14
source .venv/bin/activate

# Install all dependencies
uv pip install -r requirements.txt
```

This takes 3–5 minutes on first install. No Driveline access needed — all packages are from PyPI.

### 4 — Set up R (for leaderboard tables)

```bash
# Install Homebrew first if not present
/bin/bash -c "$(curl -fsSF https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install r

# Install R packages (run once — downloads from CRAN, no Driveline needed)
Rscript -e 'install.packages(c("arrow","dplyr","gt","gtExtras","mlbplotR","scales","webshot2"), repos="https://cloud.r-project.org")'
```

### 5 — Verify everything works

```bash
# Should write results/figures/reliability.png — no network or DB needed
.venv/bin/python results_scripts/generate_results.py reliability

# Should write results/figures/*.png for all leaderboard tables — no network or DB needed
Rscript results_scripts/leaderboard_table.R
```

Both commands run fully offline from the bundle files.

### Path changes from Windows to Mac

| Item | Windows | Mac |
|------|---------|-----|
| Activate venv | `.venv\Scripts\activate` | `source .venv/bin/activate` |
| Run Python | `.venv\Scripts\python.exe` | `.venv/bin/python` |
| Run Rscript | Full path to `Rscript.exe` | `Rscript` (on PATH after brew) |
| Path separator | `\` | `/` |

No hardcoded Windows paths exist in the Python or R scripts — all paths use forward slashes or `pathlib.Path`.

### Personal checklist

**On the old machine before returning it:**
- [ ] Confirm `pitcher_bat_path_bundle.zip` on Driveline OneDrive shows "Up to date" (green checkmark, not sync arrows) — allow 10–20 min if recently written

**On the new Mac:**
- [ ] Go to **onedrive.com** in a browser, sign in with `theo.an-yeung@drivelinebaseball.com`, download `pitcher_bat_path_bundle.zip` — no desktop app needed
- [ ] Confirm downloaded file is 3.29 GB
- [ ] `git clone https://github.com/theoauyeung/pitcher_bat_path_exploration.git && cd pitcher_bat_path_exploration`
- [ ] `unzip ~/Downloads/pitcher_bat_path_bundle.zip -d .` — confirms `data/`, `models/`, `results/` land in repo root
- [ ] `uv python install 3.14 && uv venv --python 3.14 && source .venv/bin/activate && uv pip install -r requirements.txt`
- [ ] `.venv/bin/python results_scripts/generate_results.py reliability` — confirm it writes `results/figures/reliability.png` with no errors
- [ ] `brew install r` then `Rscript -e 'install.packages(c("arrow","dplyr","gt","gtExtras","mlbplotR","scales","webshot2"), repos="https://cloud.r-project.org")'`
- [ ] `Rscript results_scripts/leaderboard_table.R` — confirm three PNGs saved to `results/figures/`

---

## Kinematic diagrams

Each figure is a two-panel broadcast card: game screenshot with arrow callout (left) + dark metrics panel (right). The **DISRUPTION ANALYSIS** section shows:

- **Post-commit drop** — vertical inches the ball moves after the batter commits. Movement the batter cannot react to.
- **Proj. → actual** — the ball's projected vs. actual plate-crossing height, with zone context.
- **Swing disruption** — run-value cost conditional on the decision to swing (`disruption_tax`; negative = pitcher advantage).
- **Decision / Chase cost** — opportunity cost of swinging vs. taking at the projected location. Green when swinging was correct; red when taking was better.
- **Total burden** — `adjusted_disruption_tax = disruption_tax − max(0, decision_cost)`. Total batter cost vs. the optimal available action.
- **Distortion / Selection bar** — fraction of swing disruption caused by post-commit movement (red) vs. the batter's own decision (amber).

| Pitcher / Batter | Pitch | Dominant cause |
|-----------------|-------|----------------|
| Yamamoto / Bernabel | Curveball | Distortion (99.7%) |
| Leiter / Ramirez | Curveball | Mixed |
| Helsley / Mullins | Four-seam FB | Selection (95%) |
| Sale / Harper | Slider | Mixed + chase penalty |
