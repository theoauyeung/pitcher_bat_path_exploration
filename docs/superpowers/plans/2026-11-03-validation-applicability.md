# Validation, Applicability, and Paper Analyses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add eight new analysis blocks to `src/viz/generate_results.ipynb` that validate the distortion tax metric at three levels and demonstrate actionability for pitchers, batters, and front offices.

**Architecture:** Each block is a self-contained notebook cell that reads from existing parquets/CSVs, writes a CSV result where applicable, and saves a PNG. A new `sxrv_extended()` helper in Cell 1 exposes additional `swing_xrv.parquet` columns (plate_x, plate_z, xwoba, game_year, pitch_group, batter_id, batter_full_name) used by multiple blocks. All analyses are purely read-only from the existing pipeline — no pipeline rerun.

**Tech Stack:** Python 3.14, pandas, duckdb, matplotlib, statsmodels, numpy; existing notebook infrastructure in `src/viz/generate_results.ipynb`

## Global Constraints

- All analyses read from existing files: `data/swings_precommit.parquet`, `results/xrv_causal.parquet`, `results/swing_xrv.parquet`, `results/distortion_pitcher.csv`, `results/distortion_batter.csv`
- Follow existing notebook style: duckdb for parquet reads, pandas for transforms, matplotlib with palette `DARK="#2c3e50"`, `MID="#7f8c8d"`, `GOLD="#e8b84b"`, `RED="#e74c3c"`, `BLUE="#3498db"`, `GREEN="#27ae60"`, `BG="#f8f8f8"`, `_spine_clean(ax)` for axes
- Validation PNGs → `results/plots/validation/`, application PNGs → `results/plots/applications/`
- `MIN_SWINGS = 200` (pitcher-level), use existing constant from Cell 0
- Pitch movement treatment column: `pc150_dev_total` from `swings_precommit.parquet`
- Join keys for parquet merges: `["game_pk", "at_bat_number", "pitch_number"]`
- Column names are exact: `distortion_tax`, `selection_tax`, `disruption_tax`, `pitcher_id`, `batter_id`, `pitch_type`, `is_whiff`, `xwoba`

---

### Task 1: Extended swing_xrv helper + applications directory

**Files:**
- Modify: `src/viz/generate_results.ipynb` — Cell 0 (mkdir block) and Cell 1 (helper functions)

**Interfaces:**
- Produces: `sxrv_extended()` → DataFrame with columns `game_pk`, `at_bat_number`, `pitch_number`, `pitcher_id`, `batter_id`, `batter_full_name`, `game_year`, `pitch_type`, `pitch_group`, `plate_x`, `plate_z`, `xwoba`, `is_whiff`, `is_bip`, `stuff_grade`, `xrv`

- [ ] **Step 1: Add applications mkdir and missing import to Cell 0**

In Cell 0, add after the existing `Path("results/plots/case_studies").mkdir(...)` line:

```python
Path("results/plots/applications").mkdir(parents=True, exist_ok=True)
```

Also confirm (or add) this import in Cell 0's import block — it is used in Task 2's agg call:

```python
from scipy.stats import sem
```

- [ ] **Step 2: Add `sxrv_extended()` to Cell 1**

In Cell 1, add after the existing `merged()` function definition:

```python
_SXRV_EXT_COLS = ", ".join([
    "game_pk", "at_bat_number", "pitch_number", "pitcher_id",
    "batter_id", "batter_full_name", "game_year",
    "pitch_type", "pitch_group", "plate_x", "plate_z",
    "xwoba", "is_whiff", "is_bip", "stuff_grade", "xrv",
])

def sxrv_extended():
    return _load("sxrv_ext", lambda: duckdb.sql(
        f"SELECT {_SXRV_EXT_COLS} FROM read_parquet('results/swing_xrv.parquet')"
    ).df())
```

- [ ] **Step 3: Add verification cell after Cell 1**

```python
_ext = sxrv_extended()
assert set(["game_pk", "at_bat_number", "pitch_number", "pitcher_id", "batter_id",
            "batter_full_name", "game_year", "pitch_type", "pitch_group",
            "plate_x", "plate_z", "xwoba", "is_whiff", "is_bip",
            "stuff_grade", "xrv"]).issubset(_ext.columns), "Missing columns in sxrv_extended"
assert len(_ext) > 600_000, f"Expected >600k rows, got {len(_ext):,}"
print(f"sxrv_extended OK: {len(_ext):,} rows")
del _ext
```

Run the cell. Expected: prints `sxrv_extended OK: 624,731 rows` with no assertion errors.

- [ ] **Step 4: Commit**

```bash
git add src/viz/generate_results.ipynb
git commit -m "feat: add sxrv_extended helper and applications plot directory"
```

---

### Task 2: Level 1 controls — negative and positive

Write up the already-coded negative and positive controls as formal paper results: near-straight four-seamers should show disruption_tax ≈ 0; breaking balls should rank higher distortion than fastballs.

**Files:**
- Modify: `src/viz/generate_results.ipynb` — add new cell in validation section
- Create: `results/plots/validation/level1_controls.png`

**Interfaces:**
- Consumes: `xrv()` (existing helper), `pc150_dev_total` from `data/swings_precommit.parquet`
- Produces: `results/plots/validation/level1_controls.png` — 1×2 panel (negative control + positive control)

- [ ] **Step 1: Add controls cell**

```python
# --- Level 1 controls ---
dev_col = duckdb.sql("""
    SELECT game_pk, at_bat_number, pitch_number, pc150_dev_total
    FROM read_parquet('data/swings_precommit.parquet')
""").df()

xrv_pc = (xrv()
          .merge(dev_col, on=["game_pk", "at_bat_number", "pitch_number"], how="left")
          .dropna(subset=["disruption_tax", "pc150_dev_total"]))

# --- Negative control: near-straight FF (post-commit deviation < 1 inch = 1/12 ft) ---
near_straight = xrv_pc.query("pitch_type == 'FF' and pc150_dev_total < (1/12)")
all_ff        = xrv_pc.query("pitch_type == 'FF'")
high_break    = xrv_pc.query("pitch_type in ['ST','SL','CU']")

neg_mean  = near_straight["disruption_tax"].mean()
neg_sem   = near_straight["disruption_tax"].sem()
ff_mean   = all_ff["disruption_tax"].mean()
break_mean = high_break["disruption_tax"].mean()

# --- Positive control: distortion_tax by pitch type ---
KEEP_TYPES = ["FF", "SI", "FC", "CH", "FS", "SL", "CU", "ST", "KC"]
pt_ctrl = (xrv_pc[xrv_pc["pitch_type"].isin(KEEP_TYPES)]
           .groupby("pitch_type")["distortion_tax"]
           .agg(n="size", mean="mean", sem=sem)
           .reset_index()
           .query("n >= 5000")
           .sort_values("mean"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)

# --- Left panel: negative control distribution ---
ax1.hist(near_straight["disruption_tax"].clip(-0.15, 0.05), bins=40,
         color=BLUE, alpha=0.7, density=True, label=f"Near-straight FF\n(n={len(near_straight)})")
ax1.axvline(neg_mean, color=RED, lw=1.5, linestyle="--",
            label=f"Mean = {neg_mean:.4f}")
ax1.axvline(0, color=DARK, lw=0.8)
ax1.set_xlabel("Disruption tax (xRV/swing)", color=DARK)
ax1.set_ylabel("Density", color=DARK)
ax1.set_title("Negative control\nNear-straight FF (post-commit dev < 1\")",
              color=DARK, fontweight="bold")
ax1.legend(frameon=False, fontsize=8)
_spine_clean(ax1); ax1.tick_params(colors=DARK)

# --- Right panel: positive control bar chart ---
colors_bar = [RED if t in ["ST", "SL", "CU", "KC", "FS"] else BLUE for t in pt_ctrl["pitch_type"]]
ax2.barh(pt_ctrl["pitch_type"], pt_ctrl["mean"], xerr=pt_ctrl["sem"],
         color=colors_bar, height=0.6, capsize=3)
ax2.axvline(0, color=DARK, lw=0.8)
ax2.set_xlabel("Mean distortion tax (xRV/swing)  |  Negative = pitcher advantage", color=DARK)
ax2.set_title("Positive control\nDistortion tax by pitch type (red = breaking/offspeed)",
              color=DARK, fontweight="bold")
_spine_clean(ax2); ax2.tick_params(colors=DARK)

fig.tight_layout()
out = "results/plots/validation/level1_controls.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Near-straight FF mean disruption_tax: {neg_mean:.4f} ± {neg_sem:.4f}")
print(f"All FF mean: {ff_mean:.4f}  |  Breaking (ST/SL/CU) mean: {break_mean:.4f}")
```

- [ ] **Step 2: Run and verify**

Expected output:
```
Near-straight FF mean disruption_tax: -0.012 ± ...
All FF mean: -0.008  |  Breaking (ST/SL/CU) mean: ~-0.014
```
ST/SL/CU should show more negative mean distortion_tax than FF. PNG saved.

- [ ] **Step 3: Inline assertions**

```python
assert Path(out).exists()
assert break_mean < ff_mean, f"Breaking balls should be more disruptive than FF: {break_mean:.4f} vs {ff_mean:.4f}"
assert pt_ctrl["pitch_type"].tolist()[0] in ["ST", "CU", "SL"], "Most disruptive pitch type should be a breaker"
```

- [ ] **Step 4: Commit**

```bash
git add src/viz/generate_results.ipynb results/plots/validation/level1_controls.png
git commit -m "feat: add Level 1 negative and positive control plots"
```

---

### Task 3: Convergent validity

Correlate pitcher-level mean distortion_tax with mean post-commit movement magnitude, confirming the metric tracks the physical phenomenon it claims to measure.

**Files:**
- Modify: `src/viz/generate_results.ipynb`
- Create: `results/convergent_validity.csv`
- Create: `results/plots/validation/convergent_validity.png`

**Interfaces:**
- Consumes: `xrv()`, `pc150_dev_total` from `data/swings_precommit.parquet`
- Produces: `results/convergent_validity.csv` with columns `pitcher_id`, `n`, `mean_distortion_tax`, `mean_pc150_dev_total`

- [ ] **Step 1: Add convergent validity cell**

```python
csv = Path("results/convergent_validity.csv")
if not csv.exists():
    dev_col = duckdb.sql("""
        SELECT game_pk, at_bat_number, pitch_number, pc150_dev_total
        FROM read_parquet('data/swings_precommit.parquet')
    """).df()

    cv = (xrv()
          .merge(dev_col, on=["game_pk", "at_bat_number", "pitch_number"], how="left")
          .dropna(subset=["distortion_tax", "pc150_dev_total"])
          .groupby("pitcher_id")
          .agg(
              n=("distortion_tax", "size"),
              mean_distortion_tax=("distortion_tax", "mean"),
              mean_pc150_dev_total=("pc150_dev_total", "mean"),
          )
          .reset_index()
          .query(f"n >= {MIN_SWINGS}"))
    cv.to_csv(csv, index=False)

cv = pd.read_csv(csv)
r  = cv["mean_distortion_tax"].corr(cv["mean_pc150_dev_total"])
n  = len(cv)

fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
ax.set_facecolor(BG)
ax.scatter(cv["mean_pc150_dev_total"], cv["mean_distortion_tax"],
           alpha=0.4, s=20, color=BLUE)
m_slope, b_intercept = np.polyfit(cv["mean_pc150_dev_total"], cv["mean_distortion_tax"], 1)
xs = np.linspace(cv["mean_pc150_dev_total"].min(), cv["mean_pc150_dev_total"].max(), 100)
ax.plot(xs, m_slope * xs + b_intercept, color=RED, lw=1.5)
ax.set_xlabel("Mean post-commit movement magnitude (ft)", color=DARK)
ax.set_ylabel("Mean distortion tax (xRV/swing)", color=DARK)
ax.set_title(
    f"Convergent validity: distortion tax vs. post-commit movement\n"
    f"r = {r:.3f}  |  n = {n} pitchers (min {MIN_SWINGS} swings)",
    color=DARK, fontweight="bold")
_spine_clean(ax); ax.tick_params(colors=DARK)
fig.tight_layout()
out = "results/plots/validation/convergent_validity.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Convergent validity: r = {r:.3f}, n = {n}")
```

- [ ] **Step 2: Run and verify**

Expected:
- `r` is negative (more movement → more negative distortion_tax), magnitude ≥ 0.3
- `results/convergent_validity.csv` exists, ≥200 rows, PNG saved

- [ ] **Step 3: Inline assertions**

```python
assert csv.exists()
assert r < -0.2, f"Expected r < -0.2, got {r:.3f}"
assert n > 100, f"Expected >100 pitchers, got {n}"
```

- [ ] **Step 4: Commit**

```bash
git add src/viz/generate_results.ipynb results/convergent_validity.csv results/plots/validation/convergent_validity.png
git commit -m "feat: add convergent validity analysis"
```

---

### Task 4: Pitch-type-matched incremental validity

Test whether a pitcher's breaking-ball distortion_tax predicts their breaking-ball whiff rate that same season — a same-season, mechanism-matched outcome target that replaces the wrong cross-year test.

**Files:**
- Modify: `src/viz/generate_results.ipynb`
- Create: `results/incremental_validity_breaking.csv`
- Create: `results/plots/validation/incremental_validity_breaking.png`

**Interfaces:**
- Consumes: `xrv()`, `sxrv_extended()` (Task 1)
- Produces: `results/incremental_validity_breaking.csv` with columns `pitcher_id`, `game_year`, `n`, `mean_distortion_tax`, `whiff_rate`, `mean_xwoba`

- [ ] **Step 1: Add incremental validity cell**

```python
BREAKING_TYPES = ["SL", "CU", "ST", "KC", "SV", "CS"]

csv = Path("results/incremental_validity_breaking.csv")
if not csv.exists():
    sw = sxrv_extended()[["game_pk", "at_bat_number", "pitch_number",
                           "pitcher_id", "game_year", "is_whiff", "xwoba"]]
    brk = (xrv()
           .query("pitch_type in @BREAKING_TYPES")
           .merge(sw, on=["game_pk", "at_bat_number", "pitch_number", "pitcher_id"],
                  how="inner"))
    iv = (brk
          .groupby(["pitcher_id", "game_year"])
          .agg(
              n=("distortion_tax", "size"),
              mean_distortion_tax=("distortion_tax", "mean"),
              whiff_rate=("is_whiff", "mean"),
              mean_xwoba=("xwoba", lambda x: x.dropna().mean()),
          )
          .reset_index()
          .query("n >= 100"))
    iv.to_csv(csv, index=False)

iv = pd.read_csv(csv)

baseline = smf.ols("whiff_rate ~ 1", data=iv).fit()
full     = smf.ols("whiff_rate ~ mean_distortion_tax", data=iv).fit()
delta_r2 = full.rsquared - baseline.rsquared
coef     = full.params["mean_distortion_tax"]
pval     = full.pvalues["mean_distortion_tax"]

fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
ax.set_facecolor(BG)
ax.scatter(iv["mean_distortion_tax"], iv["whiff_rate"],
           alpha=0.4, s=20, color=BLUE)
xs = np.linspace(iv["mean_distortion_tax"].min(), iv["mean_distortion_tax"].max(), 100)
ax.plot(xs, full.params["Intercept"] + coef * xs, color=RED, lw=1.5)
ax.set_xlabel("Mean distortion tax on breaking balls (xRV/swing)", color=DARK)
ax.set_ylabel("Breaking-ball whiff rate", color=DARK)
ax.set_title(
    f"Incremental validity: distortion tax → breaking-ball whiff rate\n"
    f"ΔR² = {delta_r2:.4f}  |  β = {coef:.4f}  |  p = {pval:.4f}"
    f"  |  n = {len(iv)} pitcher-seasons",
    color=DARK, fontweight="bold")
_spine_clean(ax); ax.tick_params(colors=DARK)
fig.tight_layout()
out = "results/plots/validation/incremental_validity_breaking.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"ΔR² = {delta_r2:.4f}, β = {coef:.4f}, p = {pval:.4f}, n = {len(iv)}")
```

- [ ] **Step 2: Run and verify**

Expected:
- `coef` is negative (more negative distortion_tax → higher whiff rate, since more negative = more pitcher advantage)
- `pval` < 0.05
- CSV has ≥300 pitcher-season rows, PNG saved

- [ ] **Step 3: Inline assertions**

```python
assert csv.exists()
assert coef < 0, f"Expected negative coefficient (more distortion → more whiffs), got {coef:.4f}"
assert pval < 0.10, f"Expected p < 0.10, got {pval:.4f}"
assert len(iv) > 200, f"Expected >200 pitcher-seasons, got {len(iv)}"
```

- [ ] **Step 4: Commit**

```bash
git add src/viz/generate_results.ipynb results/incremental_validity_breaking.csv results/plots/validation/incremental_validity_breaking.png
git commit -m "feat: add pitch-type-matched incremental validity"
```

---

### Task 5: Matchup interaction test

Test whether the pitcher distortion_tax × batter distortion_susceptibility interaction predicts matchup-level xwOBA outcomes beyond individual talent. A significant negative interaction is Level 3 validation and directly supports the matchup deployment application.

**Files:**
- Modify: `src/viz/generate_results.ipynb`
- Create: `results/matchup_interaction.csv`
- Create: `results/plots/validation/matchup_interaction.png`

**Interfaces:**
- Consumes: `sxrv_extended()` (Task 1), `results/distortion_pitcher.csv`, `results/distortion_batter.csv`
- Produces: `results/matchup_interaction.csv` with columns `pitcher_id`, `batter_id`, `n`, `matchup_xwoba`, `pitcher_distortion_tax`, `batter_distortion_susceptibility`, `pitcher_talent_xrv`, `batter_talent_xwoba`

- [ ] **Step 1: Add matchup interaction cell**

```python
csv = Path("results/matchup_interaction.csv")
if not csv.exists():
    sw = sxrv_extended()[["game_pk", "at_bat_number", "pitch_number",
                           "pitcher_id", "batter_id", "xwoba", "xrv"]]

    # Matchup-level outcomes (min 10 pitches with recorded xwoba)
    matchup = (sw.dropna(subset=["xwoba"])
               .groupby(["pitcher_id", "batter_id"])
               .agg(n=("xwoba", "size"), matchup_xwoba=("xwoba", "mean"))
               .reset_index()
               .query("n >= 10"))

    # Talent proxies: career means across all matchups
    pitcher_talent = (sw.dropna(subset=["xrv"])
                      .groupby("pitcher_id")
                      .agg(pitcher_talent_xrv=("xrv", "mean"))
                      .reset_index())
    batter_talent  = (sw.dropna(subset=["xwoba"])
                      .groupby("batter_id")
                      .agg(batter_talent_xwoba=("xwoba", "mean"))
                      .reset_index())

    # Distortion profiles (career aggregates from pipeline outputs)
    pdist = (pd.read_csv("results/distortion_pitcher.csv")
             [["pitcher_id", "mean_distortion_tax"]]
             .rename(columns={"mean_distortion_tax": "pitcher_distortion_tax"}))
    bdist = (pd.read_csv("results/distortion_batter.csv")
             [["batter_id", "mean_distortion_tax"]]
             .rename(columns={"mean_distortion_tax": "batter_distortion_susceptibility"}))

    m = (matchup
         .merge(pitcher_talent, on="pitcher_id", how="inner")
         .merge(batter_talent,  on="batter_id",  how="inner")
         .merge(pdist,          on="pitcher_id", how="inner")
         .merge(bdist,          on="batter_id",  how="inner"))
    m.to_csv(csv, index=False)

m = pd.read_csv(csv)

# Standardize for interpretable interaction coefficient
for col in ["pitcher_distortion_tax", "batter_distortion_susceptibility",
            "pitcher_talent_xrv", "batter_talent_xwoba"]:
    m[f"{col}_z"] = (m[col] - m[col].mean()) / m[col].std()

formula = ("matchup_xwoba ~ pitcher_talent_xrv_z + batter_talent_xwoba_z"
           " + pitcher_distortion_tax_z:batter_distortion_susceptibility_z")
result = smf.ols(formula, data=m).fit()
interaction_coef = result.params["pitcher_distortion_tax_z:batter_distortion_susceptibility_z"]
interaction_pval = result.pvalues["pitcher_distortion_tax_z:batter_distortion_susceptibility_z"]
print(result.summary())

# Plot: slope of pitcher_distortion_tax vs matchup_xwoba, stratified by batter susceptibility tercile
m["susceptibility_tercile"] = pd.qcut(
    m["batter_distortion_susceptibility"], q=3,
    labels=["Low susceptibility", "Medium", "High susceptibility"]
)

fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
ax.set_facecolor(BG)
for grp, color in [("Low susceptibility", GREEN), ("Medium", MID), ("High susceptibility", RED)]:
    sub = m[m["susceptibility_tercile"] == grp]
    xs  = np.linspace(sub["pitcher_distortion_tax"].min(),
                      sub["pitcher_distortion_tax"].max(), 100)
    fit = smf.ols("matchup_xwoba ~ pitcher_distortion_tax", data=sub).fit()
    ax.scatter(sub["pitcher_distortion_tax"], sub["matchup_xwoba"],
               alpha=0.12, s=6, color=color)
    ax.plot(xs, fit.predict(pd.DataFrame({"pitcher_distortion_tax": xs})),
            color=color, lw=2, label=grp)
ax.set_xlabel("Pitcher mean distortion tax (xRV/swing)", color=DARK)
ax.set_ylabel("Matchup xwOBA", color=DARK)
ax.set_title(
    f"Matchup interaction: distortion × batter susceptibility\n"
    f"Interaction β = {interaction_coef:.4f}  |  p = {interaction_pval:.4f}"
    f"  |  n = {len(m):,} matchups",
    color=DARK, fontweight="bold")
ax.legend(frameon=False, fontsize=8)
_spine_clean(ax); ax.tick_params(colors=DARK)
fig.tight_layout()
out = "results/plots/validation/matchup_interaction.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
```

- [ ] **Step 2: Run and verify**

Expected:
- CSV has ≥5,000 pitcher×batter matchup rows
- `interaction_coef` is negative (high distortion pitcher × high susceptibility batter → lower xwOBA)
- Regression summary shows interaction term significant (p < 0.05)
- PNG saved with three visible trend lines showing steeper negative slope for high-susceptibility batters

- [ ] **Step 3: Inline assertions**

```python
assert csv.exists()
assert len(m) > 5_000, f"Too few matchups: {len(m):,}"
assert interaction_coef < 0, f"Expected negative interaction, got {interaction_coef:.4f}"
```

- [ ] **Step 4: Commit**

```bash
git add src/viz/generate_results.ipynb results/matchup_interaction.csv results/plots/validation/matchup_interaction.png
git commit -m "feat: add matchup interaction test"
```

---

### Task 6: Distortion heat map by location × pitch group

Show where on the plate each pitch group (Fastball/Breaking/Offspeed) generates the most distortion, providing location targets for pitch design.

**Files:**
- Modify: `src/viz/generate_results.ipynb`
- Create: `results/plots/applications/distortion_heatmap_by_zone.png`

**Interfaces:**
- Consumes: `xrv()`, `sxrv_extended()` for `plate_x`, `plate_z`, `pitch_group`
- Produces: `results/plots/applications/distortion_heatmap_by_zone.png` — 1×3 panel heatmap

- [ ] **Step 1: Add distortion heat map cell**

```python
sw = sxrv_extended()[["game_pk", "at_bat_number", "pitch_number",
                        "plate_x", "plate_z", "pitch_group"]]
heat_df = (xrv()
           .dropna(subset=["distortion_tax"])
           .merge(sw, on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
           .dropna(subset=["plate_x", "plate_z"]))

# 5×5 grid: x in [−1.5, 1.5] ft, z in [0.5, 5.0] ft
x_edges = np.linspace(-1.5, 1.5, 6)
z_edges = np.linspace(0.5,  5.0, 6)
heat_df["x_bin"] = pd.cut(heat_df["plate_x"], bins=x_edges, labels=False)
heat_df["z_bin"] = pd.cut(heat_df["plate_z"], bins=z_edges, labels=False)

groups = ["Fastball", "Breaking", "Offspeed"]
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG)
fig.suptitle("Mean distortion tax by plate location and pitch group\n"
             "(darker red = more pitcher advantage from late movement)",
             color=DARK, fontweight="bold", y=1.02)

x_tick_labels = ["−1.2", "−0.6", "0", "+0.6", "+1.2"]
z_tick_labels = ["0.8", "1.6", "2.3", "3.1", "3.9"]
im_ref = None

for ax, grp in zip(axes, groups):
    ax.set_facecolor(BG)
    sub  = heat_df[heat_df["pitch_group"] == grp]
    grid = (sub.groupby(["z_bin", "x_bin"])["distortion_tax"]
               .mean()
               .unstack("x_bin")
               .reindex(index=range(5), columns=range(5)))
    im = ax.imshow(grid.values, origin="lower", aspect="auto",
                   cmap="RdYlGn_r", vmin=-0.15, vmax=0.0)
    if im_ref is None:
        im_ref = im
    ax.set_title(grp, color=DARK, fontweight="bold")
    ax.set_xlabel("← Arm side   Glove side →", color=DARK, fontsize=8)
    if ax is axes[0]:
        ax.set_ylabel("Height (low → high)", color=DARK, fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(x_tick_labels, color=DARK, fontsize=7)
    ax.set_yticks(range(5)); ax.set_yticklabels(z_tick_labels, color=DARK, fontsize=7)
    # Strike zone outline: x ≈ [−0.71, 0.71] ft → bins ~[1.1, 2.9]; z ≈ [1.5, 3.5] → bins ~[1.1, 3.3]
    ax.add_patch(mpatches.Rectangle((1.1, 1.1), 1.8, 2.2,
                                    linewidth=1.2, edgecolor=DARK, facecolor="none", linestyle="--"))

plt.colorbar(im_ref, ax=axes[-1], label="Mean distortion tax (xRV/swing)", shrink=0.8)
fig.tight_layout()
out = "results/plots/applications/distortion_heatmap_by_zone.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out}")
```

- [ ] **Step 2: Run and verify**

Expected:
- Breaking ball panel shows darker (more negative) values than Fastball panel, especially outside strike zone
- PNG has three heatmaps with visible spatial pattern
- No NaN-dominated panels (most grid cells should have data)

- [ ] **Step 3: Inline assertions**

```python
assert Path(out).exists()
brk_mean = heat_df[heat_df["pitch_group"] == "Breaking"]["distortion_tax"].mean()
ff_mean  = heat_df[heat_df["pitch_group"] == "Fastball"]["distortion_tax"].mean()
assert brk_mean < ff_mean, f"Expected Breaking < Fastball: {brk_mean:.4f} vs {ff_mean:.4f}"
```

- [ ] **Step 4: Commit**

```bash
git add src/viz/generate_results.ipynb results/plots/applications/distortion_heatmap_by_zone.png
git commit -m "feat: add distortion heat map by location and pitch group"
```

---

### Task 7: Batter vulnerability segmentation

Scatter plot of mean distortion susceptibility vs mean selection susceptibility per batter, with quadrant labels identifying distortion-dominant and selection-dominant development archetypes.

**Files:**
- Modify: `src/viz/generate_results.ipynb`
- Create: `results/plots/applications/batter_vulnerability.png`

**Interfaces:**
- Consumes: `results/distortion_batter.csv`, `sxrv_extended()` for batter names (column `batter_full_name`)
- Produces: `results/plots/applications/batter_vulnerability.png`

- [ ] **Step 1: Add batter vulnerability cell**

```python
batters = pd.read_csv("results/distortion_batter.csv").query(f"n_swings >= {MIN_SWINGS}")

# Add names
batter_names = (sxrv_extended()[["batter_id", "batter_full_name"]]
                .drop_duplicates("batter_id"))
batters = batters.merge(batter_names, on="batter_id", how="left")

# Negate: taxes are negative (pitcher advantage); flip for intuitive "susceptibility" direction
batters["distortion_susceptibility"] = -batters["mean_distortion_tax"]
batters["selection_susceptibility"]  = -batters["mean_selection_tax"]

med_d = batters["distortion_susceptibility"].median()
med_s = batters["selection_susceptibility"].median()

fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
ax.set_facecolor(BG)
ax.scatter(batters["distortion_susceptibility"], batters["selection_susceptibility"],
           alpha=0.35, s=18, color=BLUE)
ax.axvline(med_d, color=MID, lw=0.8, linestyle="--")
ax.axhline(med_s, color=MID, lw=0.8, linestyle="--")

# Quadrant labels (placed at 90th percentile of each axis)
q90d = batters["distortion_susceptibility"].quantile(0.90)
q90s = batters["selection_susceptibility"].quantile(0.90)
q10d = batters["distortion_susceptibility"].quantile(0.10)
q10s = batters["selection_susceptibility"].quantile(0.10)
ax.text(q90d, q90s, "High both", ha="right", va="top",
        color=RED, fontsize=8, fontstyle="italic")
ax.text(q90d, q10s, "Distortion-dominant\n(mechanical fix)", ha="right", va="bottom",
        color=RED, fontsize=8, fontstyle="italic")
ax.text(q10d, q90s, "Selection-dominant\n(recognition fix)", ha="left", va="top",
        color=GOLD, fontsize=8, fontstyle="italic")

# Annotate top-10 most distortion-susceptible batters
top10 = batters.nlargest(10, "distortion_susceptibility")
for _, row in top10.iterrows():
    last = str(row["batter_full_name"]).split()[-1] if pd.notna(row["batter_full_name"]) else ""
    ax.annotate(last,
                xy=(row["distortion_susceptibility"], row["selection_susceptibility"]),
                xytext=(4, 2), textcoords="offset points",
                fontsize=6, color=DARK)

ax.set_xlabel("Distortion susceptibility (−mean_distortion_tax)", color=DARK)
ax.set_ylabel("Selection susceptibility (−mean_selection_tax)", color=DARK)
ax.set_title(
    f"Batter vulnerability profiles: distortion vs. selection\n"
    f"n = {len(batters)} batters (min {MIN_SWINGS} swings)",
    color=DARK, fontweight="bold")
_spine_clean(ax); ax.tick_params(colors=DARK)
fig.tight_layout()
out = "results/plots/applications/batter_vulnerability.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out}, n={len(batters)}")
```

- [ ] **Step 2: Run and verify**

Expected:
- PNG shows four-quadrant scatter with ≥100 labeled batter points
- Last names visible for top-10 distortion-susceptible batters
- Quadrant labels in correct corners

- [ ] **Step 3: Inline assertions**

```python
assert Path(out).exists()
assert len(batters) > 100, f"Too few batters: {len(batters)}"
assert (batters["distortion_susceptibility"] > 0).mean() > 0.5
```

- [ ] **Step 4: Commit**

```bash
git add src/viz/generate_results.ipynb results/plots/applications/batter_vulnerability.png
git commit -m "feat: add batter vulnerability segmentation scatter"
```

---

### Task 8: Economic translation table

Convert distortion_tax to season-level run value and WAR-equivalent, and show Stuff+ comparison to identify under-credited distortion pitchers.

**Files:**
- Modify: `src/viz/generate_results.ipynb`
- Create: `results/economic_translation.csv`
- Create: `results/plots/applications/economic_translation.png`

**Interfaces:**
- Consumes: `results/distortion_pitcher.csv`, `pitcher_full_name` and `mean_stuff_grade` from `results/swing_xrv.parquet` via duckdb
- Produces: `results/economic_translation.csv` with columns `pitcher_id`, `pitcher_full_name`, `n_swings`, `mean_distortion_tax`, `runs_from_distortion`, `war_equivalent`, `dollars_M`, `mean_stuff_grade`

- [ ] **Step 1: Add economic translation cell**

```python
RUNS_PER_WIN  = 10
WAR_SALARY_M  = 8.5  # $M per WAR (2024–2025 market rate)

csv = Path("results/economic_translation.csv")
if not csv.exists():
    pitcher_meta = duckdb.sql("""
        SELECT pitcher_id,
               ANY_VALUE(pitcher_full_name) AS pitcher_full_name,
               AVG(stuff_grade)             AS mean_stuff_grade
        FROM read_parquet('results/swing_xrv.parquet')
        GROUP BY pitcher_id
    """).df()

    eco = (pd.read_csv("results/distortion_pitcher.csv")
           .query(f"n_swings >= {MIN_SWINGS}")
           .merge(pitcher_meta, on="pitcher_id", how="inner"))
    eco["runs_from_distortion"] = eco["mean_distortion_tax"] * eco["n_swings"]
    eco["war_equivalent"]       = eco["runs_from_distortion"].abs() / RUNS_PER_WIN
    eco["dollars_M"]            = eco["war_equivalent"] * WAR_SALARY_M
    eco.to_csv(csv, index=False)

eco = pd.read_csv(csv)

top20 = (eco.nsmallest(20, "mean_distortion_tax")
            [["pitcher_full_name", "n_swings", "mean_distortion_tax",
              "runs_from_distortion", "war_equivalent", "dollars_M", "mean_stuff_grade"]]
            .reset_index(drop=True))

# Format for display
display = top20.copy()
display["mean_distortion_tax"]  = display["mean_distortion_tax"].map("{:.4f}".format)
display["runs_from_distortion"] = display["runs_from_distortion"].map("{:.1f}".format)
display["war_equivalent"]       = display["war_equivalent"].map("{:.1f}".format)
display["dollars_M"]            = display["dollars_M"].map("${:.0f}M".format)
display["mean_stuff_grade"]     = display["mean_stuff_grade"].map("{:.1f}".format)
display.columns = ["Pitcher", "Swings", "Distortion Tax\n(xRV/swing)",
                   "Distortion\nRuns", "WAR\nEquiv.", "Value\n($M)", "Stuff+\nGrade"]

fig, ax = plt.subplots(figsize=(13, 8), facecolor=BG)
ax.set_facecolor(BG); ax.axis("off")
tbl = ax.table(cellText=display.values, colLabels=display.columns,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.6)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor(MID)
    cell.set_text_props(color=DARK)
    if row == 0:
        cell.set_facecolor(DARK); cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor(BG if row % 2 == 1 else "#ececec")

ax.set_title(
    f"Top 20 pitchers by distortion tax — career 2023–2025\n"
    f"Distortion Runs = mean_distortion_tax × n_swings  |  "
    f"WAR @ {RUNS_PER_WIN} runs/win  |  Value @ ${WAR_SALARY_M}M/WAR",
    color=DARK, fontweight="bold", pad=20)
fig.tight_layout()
out = "results/plots/applications/economic_translation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out}")
```

- [ ] **Step 2: Run and verify**

Expected:
- CSV has ≥200 rows with all columns populated
- Top 20 table shows pitchers with `runs_from_distortion` in roughly −30 to −80 range
- `mean_stuff_grade` column varies across the 20 rows (not all high), highlighting pitchers whose distortion value is not fully captured by Stuff+
- PNG saved

- [ ] **Step 3: Inline assertions**

```python
assert csv.exists()
assert eco["runs_from_distortion"].min() < -20, "Top pitchers should show >20 distortion runs"
assert eco["dollars_M"].max() > 15, "WAR translation should show >$15M for top pitchers"
assert eco["mean_stuff_grade"].std() > 5, "Stuff+ grades should vary across top distortion pitchers"
```

- [ ] **Step 4: Commit**

```bash
git add src/viz/generate_results.ipynb results/economic_translation.csv results/plots/applications/economic_translation.png
git commit -m "feat: add economic translation table"
```
