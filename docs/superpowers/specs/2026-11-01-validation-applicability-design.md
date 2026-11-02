# Validation, Applicability, and Paper Positioning Design

## Overview

This spec covers what needs to be added to make the distortion tax research publishable and actionable for a dual audience: academic/sabermetrics community and MLB front offices. Three areas: (1) validation architecture, (2) actionability framework, (3) paper structure and economic translation.

---

## Positioning against existing literature

### What already exists

| Work | What it measures | Gap it leaves |
|------|-----------------|---------------|
| Stuff+ / Location+ / StuffPro | Pitch physical quality as expected run value — pitcher-side, predictive | Doesn't measure what late movement *causes* to the batter's swing |
| Pitch Tunneling (BP, 2017+) | Pre-commit deception window — how similar two pitches look at decision point | No measurement of post-commit consequence; doesn't price disruption in run value |
| Powers & Yurko, *The American Statistician* (2026) | Bat tracking metrics as outcomes; Bayesian LMM to estimate batter intent | Treats swing metrics as the thing to explain, not as mediators; no pitch-movement causation |
| Sutton-Brown, BP (2025) | Infers batter pitch expectation from bat tracking | No run-value decomposition; not a causal attribution framework |

### The three-part contribution

1. **Post-commit as causal identification.** The 150ms threshold creates a defensible exogenous treatment — movement after commit is physically impossible to react to. No existing pitch quality metric makes this identification argument.

2. **Bat tracking as mediator, not outcome.** VAA deviation, HAA deviation, and swing path tilt are the causal pathway through which pitch movement costs run value. This is a different model structure from Powers & Yurko, who treat the same variables as outcomes.

3. **Distortion vs. Selection decomposition.** No existing metric distinguishes unavoidable disruption (post-commit movement) from batter decision error (swing choice given available information). Stuff+ credits both to the pitcher. This paper attributes each to its source.

### Differentiating claim (for paper Introduction)

> Existing pitch quality metrics measure what a pitch *is* (physical properties) or what it *does* (expected outcomes). This paper measures what late pitch movement *causes* — specifically, the causal share of run-value loss attributable to physically unavoidable post-commit disruption versus the batter's own decision. We use bat tracking as the observed mediator and the 150ms commit threshold as the identification instrument.

---

## Section 1 — Validation architecture

Three levels, each addressing a different skeptic.

### Level 1 — Pitch-level controls

**Negative control:** four-seamers with post-commit deviation < 1" should show disruption_tax ≈ 0. Report mean and 95% CI for this subset. Already coded in `causal_models.py` as `negative_control_check` — needs to be written up as a formal result, not just a diagnostic.

**Positive control:** pitch types ranked by mean distortion_tax should match movement ranking (sweepers > sinkers > four-seamers). Present as a pitch-type comparison table or bar chart. Already coded as `positive_control_check`.

Both controls are currently internal checks. In the paper they are evidence that the identification strategy is working.

### Level 2 — Pitcher-season reliability and matched incremental validity

**Reliability (already strong):** distortion_tax YoY r = 0.92 (23→24), 0.91 (24→25). Report in paper as is.

**Incremental validity fix:** current test (does distortion_tax predict next-year overall performance?) is the wrong target — distortion is an in-season physical property, not a cross-year talent predictor. New target: does a pitcher's distortion_tax on breaking balls predict their breaking-ball whiff rate or xwOBA-against on breaking balls that same season?

This is a same-season, pitch-type-matched test. It asks whether the metric tracks the mechanism it claims to measure.

**Convergent validity (new):** correlate pitcher distortion_tax with their average post-commit vertical and horizontal deviation magnitudes. Confirms the metric tracks the physical phenomenon.

### Level 3 — Matchup-level validity (novel academic contribution)

Using `xrv_causal.parquet`, group by pitcher × batter matchups. Test: do matchups where the pitcher has high distortion_tax AND the batter has high distortion susceptibility (mean distortion_tax as a victim) produce worse xwOBA outcomes than talent alone predicts?

This test survives individual talent confounds because it uses the interaction. It is also the validation layer that directly supports the actionability claim (matchup deployment).

Analysis: OLS or mixed model regressing actual xwOBA on pitcher talent (ERA-/FIP-), batter talent (wRC+), and the pitcher distortion_tax × batter distortion_susceptibility interaction term. A significant negative interaction coefficient confirms the metric captures something real beyond talent.

---

## Section 2 — Actionability framework

### Pitcher side — pitch design and location optimization

**New analysis required:** break distortion_tax down by pitch type × location zone per pitcher. Identify where each pitcher's movement profile generates the most unavoidable disruption.

**Output:** per-pitcher "distortion heat map" — location zones (e.g., 3×3 grid or shadow / heart / chase) by pitch type, showing mean distortion_tax. Highlights the structural location targets for pitch design.

**Prescription:** a pitcher whose sweeper generates 3× the distortion tax in the arm-side shadow zone vs. the heart has a data-driven location target. Direct input to pitch design sessions.

**Paper framing:** contrast with Stuff+ — Stuff+ says the sweeper is good; distortion_tax says *where* it's structurally good and *why* (post-commit movement at that location is maximally disruptive).

### Batter side — vulnerability profiles and development prescriptions

**New analysis required:** for each batter, compute two measures: (1) mean distortion_tax in absolute terms — their *distortion susceptibility* used in the matchup test; (2) the distortion_tax vs. selection_tax ratio — their *archetype classification* for development prescriptions. These are related but distinct: a batter can have high absolute distortion_tax but still be more selection-dominant in their ratio. Segment into two archetypes:

- **Distortion-dominant:** high distortion_tax, low selection_tax. Batter makes good decisions but gets physically beaten by late movement. Prescription: mechanical — earlier commit time, conservative zone entry on breaking balls, pitch-type-specific approach adjustments.
- **Selection-dominant:** high selection_tax regardless of distortion. Batter is beaten by decisions, not physics. Prescription: cognitive — pitch recognition training, count discipline, zone management.

**Paper framing:** this is a new batter development framework. The distortion-susceptible vs. selection-susceptible distinction does not exist in existing literature.

### Front office — economic translation and matchup deployment

**Economic translation:**

- Per-swing means from `distortion_pitcher.csv` × realistic season swing volumes = runs/season attributable to post-commit movement
- Example: distortion_tax mean −0.083 × 650 swings (starter) ≈ −54 runs/season from movement alone
- At 10 runs/win and $8.5M/WAR: ~5.4 WAR / $46M of distortion value from a starter's movement profile
- Include a comparison table: distortion_tax contribution vs. total xRV vs. Stuff+ for the same pitchers. Shows that some modest-Stuff+ pitchers generate outsized distortion tax because their movement is structurally late.

**Matchup deployment:**

- Cross pitcher distortion profile (high distortion_tax pitcher) against batter vulnerability profile (distortion-susceptible batter)
- High-distortion pitcher vs. distortion-susceptible batter = structurally favorable matchup beyond talent prediction
- Actionable for bullpen deployment, roster construction, platoon decisions
- This is also the Level 3 validity test — the matchup interaction analysis validates the metric while simultaneously demonstrating the application

---

## Section 3 — Paper structure

1. **Introduction** — causal attribution gap in the literature; the post-commit identification argument; paper contribution statement (three-part, above)
2. **Related work** — Stuff+/Location+, tunneling, Powers & Yurko, Sutton-Brown; position this paper against each
3. **Data and identification strategy** — commit-time threshold, pre/post trajectory split, why 150ms is conservative, robustness grid as sensitivity check
4. **Methods** — Phase A (Bayesian LMM, batter intent), Phase B (mediator models, outcome models, three-scenario counterfactual)
5. **Validation**
   - Level 1: negative/positive controls (pitch-level)
   - Level 2: reliability + matched incremental validity (breaking-ball whiff/xwOBA targets) + convergent validity
   - Level 3: matchup interaction test
6. **Results** — leaderboards, distortion share by pitch type, axis fingerprint, case studies (Yamamoto, Leiter, Helsley, Sale)
7. **Applications**
   - Pitch design: location × movement interaction heat maps
   - Batter vulnerability profiles: distortion-susceptible vs. selection-susceptible archetypes
   - Economic translation + matchup deployment
8. **Robustness** — commit-time sensitivity grid (125–200ms), XGBoost vs. linear outcome models
9. **Discussion / Limitations** — commit-time conservatism understates distortion; bat tracking coverage gaps (2023 partial season); selection tax conflates poor pitch recognition with true decision error
10. **Conclusion**

---

## New analyses required

| Analysis | Input | Output | Section |
|----------|-------|--------|---------|
| Pitch-type-matched incremental validity | `xrv_causal.parquet` + pitch outcome data | delta-r² of distortion_tax over baseline for breaking-ball whiff/xwOBA | Validation Level 2 |
| Convergent validity | `distortion_pitcher.csv` + movement magnitude cols | Correlation: distortion_tax vs. mean post-commit dev | Validation Level 2 |
| Matchup interaction test | `xrv_causal.parquet` grouped pitcher×batter | OLS: xwOBA ~ pitcher talent + batter talent + distortion interaction | Validation Level 3 |
| Distortion heat map by location × pitch type | `xrv_causal.parquet` | Per-pitcher PNG: mean distortion_tax by zone and pitch type | Applications — pitcher side |
| Batter vulnerability segmentation | `distortion_batter.csv` | Scatter: distortion_tax vs. selection_tax per batter; archetype labels | Applications — batter side |
| Economic translation table | `distortion_pitcher.csv` + season swing volumes | Runs/season and $/WAR equivalent per pitcher | Applications — front office |

---

## Open questions / limitations to address

- Selection_tax conflates poor pitch recognition with true decision error — acknowledge in Discussion
- 2023 bat tracking partial rollout affects Phase A sample size — mention in Data section
- Commit-time conservatism (150ms) deliberately understates distortion — frame this as making the causal claim *conservative*, not weak
- Incremental validity (next-year overall outcomes) remains non-significant — reframe explicitly: distortion is not a talent predictor, it is a per-pitch physical attribution
