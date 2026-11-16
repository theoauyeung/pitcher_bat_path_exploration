suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(arrow)
})

.root <- getwd()
while (!file.exists(file.path(.root, "README.md")) && dirname(.root) != .root)
  .root <- dirname(.root)
setwd(.root)
rm(.root)

dir.create("data/viz_cache", showWarnings = FALSE)

PITCH_NAMES <- c(
  FF = "4-Seam",    SI = "Sinker",    FC = "Cutter",
  CH = "Changeup",  FS = "Splitter",
  SL = "Slider",    CU = "Curveball", ST = "Sweeper",   KC = "Knuckle-curve"
)

cat("Reading swings CSV...\n")
sw <- as_tibble(fread("data/swings_2023_2025.csv")) |>
  filter(pitch_type %in% names(PITCH_NAMES)) |>
  mutate(
    pitch_name = PITCH_NAMES[pitch_type],
    outcome    = case_when(
      is_bip     == 1 ~ "BIP",
      is_contact == 1 ~ "Foul",
      TRUE            ~ "Whiff"
    )
  )

# ── 1. coverage.csv  (3 rows) ─────────────────────────────────────────────────
# Used by: season bar chart
cat("coverage.csv\n")
sw |>
  group_by(game_year) |>
  summarise(n_swings = n(), pct_tracked = mean(!is.na(bat_speed)), .groups = "drop") |>
  write_csv("data/viz_cache/coverage.csv")

# ── 2. pitch_outcomes.csv  (9 rows) ───────────────────────────────────────────
# Used by: pitch type distribution bar + outcome rates by pitch type stacked bar
# Overall outcome distribution: derive in notebook via summarise(across()) on this file
cat("pitch_outcomes.csv\n")
sw |>
  group_by(pitch_type, pitch_name) |>
  summarise(
    n          = n(),
    whiff_rate = mean(outcome == "Whiff"),
    foul_rate  = mean(outcome == "Foul"),
    bip_rate   = mean(outcome == "BIP"),
    .groups    = "drop"
  ) |>
  write_csv("data/viz_cache/pitch_outcomes.csv")

# ── 3. count_stats.csv  (12 rows) ─────────────────────────────────────────────
# Merges count profile (heatmap + whiff bar) and VAA by count (bar chart)
# Used by: count heatmap, whiff-rate-by-count bar, mean-VAA-by-count bar
cat("count_stats.csv\n")
count_base <- sw |>
  filter(balls <= 3, strikes <= 2) |>
  group_by(balls, strikes) |>
  summarise(
    n          = n(),
    whiff_rate = mean(outcome == "Whiff"),
    foul_rate  = mean(outcome == "Foul"),
    bip_rate   = mean(outcome == "BIP"),
    .groups    = "drop"
  )

count_vaa <- sw |>
  filter(!is.na(vert_attack_angle), balls <= 3, strikes <= 2) |>
  group_by(balls, strikes) |>
  summarise(
    mean_vaa = mean(vert_attack_angle),
    se_vaa   = sd(vert_attack_angle) / sqrt(n()),
    .groups  = "drop"
  )

count_base |>
  left_join(count_vaa, by = c("balls", "strikes")) |>
  mutate(
    count_label = sprintf("%d-%d", balls, strikes),
    count_type  = case_when(
      strikes == 2 & balls < 3    ~ "Two strikes",
      balls   == 3 & strikes < 2  ~ "Three balls",
      balls   == 3 & strikes == 2 ~ "Full",
      balls   == 0 & strikes == 0 ~ "First pitch",
      TRUE                        ~ "Neutral"
    )
  ) |>
  write_csv("data/viz_cache/count_stats.csv")

# ── 4. batter_summary.csv  (~500 rows) ────────────────────────────────────────
# Used by: batter mean VAA hist, bat speed hist, two-strike VAA adjustment hist
cat("batter_summary.csv\n")
batter_base <- sw |>
  filter(!is.na(vert_attack_angle), !is.na(bat_speed)) |>
  group_by(batter_id) |>
  filter(n() >= 100) |>
  summarise(
    n              = n(),
    mean_vaa       = mean(vert_attack_angle),
    mean_bat_speed = mean(bat_speed),
    .groups        = "drop"
  )

batter_two_strike <- sw |>
  filter(!is.na(vert_attack_angle), balls <= 3, strikes <= 2) |>
  group_by(batter_id) |>
  filter(n() >= 200) |>
  summarise(
    vaa_0strike = mean(vert_attack_angle[strikes == 0], na.rm = TRUE),
    vaa_2strike = mean(vert_attack_angle[strikes == 2], na.rm = TRUE),
    .groups     = "drop"
  ) |>
  filter(!is.na(vaa_0strike), !is.na(vaa_2strike)) |>
  mutate(two_strike_adj = vaa_2strike - vaa_0strike)

batter_base |>
  left_join(batter_two_strike, by = "batter_id") |>
  write_csv("data/viz_cache/batter_summary.csv")

# ── 5. ev_la_surface.csv  (~200 rows) ─────────────────────────────────────────
# Used by: EV x LA run value surface tile, EV/LA joint distribution
# (joint dist uses sample_swings.csv; surface uses this pre-binned file)
cat("ev_la_surface.csv\n")
sw |>
  filter(is_bip == 1, !is.na(launch_speed), !is.na(launch_angle), !is.na(delta_run_exp)) |>
  mutate(
    ev_bin = cut(launch_speed, breaks = seq(40, 120, by = 5),
                 include.lowest = TRUE, right = FALSE),
    la_bin = cut(launch_angle, breaks = seq(-90, 90, by = 10),
                 include.lowest = TRUE, right = FALSE)
  ) |>
  group_by(ev_bin, la_bin) |>
  summarise(mean_rv = mean(delta_run_exp), n = n(), .groups = "drop") |>
  filter(n >= 20) |>
  write_csv("data/viz_cache/ev_la_surface.csv")

# ── 6. sample_swings.csv  (80k rows) ──────────────────────────────────────────
# Single sample covering density plots, scatter plots, plate location maps,
# and EV/LA joint distribution. bat_speed/swing_length are NA for non-tracked rows.
# Used by: plate location density (all 3 outcomes), bat speed density,
#          swing length density, bat speed vs length scatter,
#          plate height vs VAA scatter, EV/LA joint dist
cat("sample_swings.csv\n")
set.seed(42)
sw |>
  filter(!is.na(plate_x), !is.na(plate_z),
         between(plate_x, -2.5, 2.5), between(plate_z, 0.5, 5)) |>
  select(
    pitch_type, pitch_name, outcome, game_year,
    plate_x, plate_z,
    bat_speed, swing_length,
    vert_attack_angle, horz_attack_angle,
    is_bip, launch_speed, launch_angle, delta_run_exp
  ) |>
  slice_sample(n = 80000) |>
  write_csv("data/viz_cache/sample_swings.csv")

# ── 7 & 8. Post-commit outcomes (requires precommit parquet) ──────────────────
cat("Reading precommit parquet...\n")
pc_raw <- read_parquet("data/swings_precommit.parquet") |>
  select(game_pk, at_bat_number, pitch_number,
         pc125_dev_total, pc150_dev_total, pc175_dev_total, pc200_dev_total,
         pc150_dev_x, pc150_dev_z,
         pc150_emp_dev_x, pc150_emp_dev_z)

sw_pc <- sw |>
  left_join(pc_raw, by = c("game_pk", "at_bat_number", "pitch_number")) |>
  mutate(emp_dev_total = sqrt(pc150_emp_dev_x^2 + pc150_emp_dev_z^2)) |>
  filter(!is.na(emp_dev_total))

# postcommit_quartile.csv  (4 rows)
# Used by: swing outcomes by deviation quartile line plot
cat("postcommit_quartile.csv\n")
sw_pc |>
  mutate(
    dev_q     = ntile(emp_dev_total, 4),
    dev_label = factor(dev_q, labels = c("Q1\n(least)", "Q2", "Q3", "Q4\n(most)"))
  ) |>
  group_by(dev_label) |>
  summarise(
    whiff_rate = mean(outcome == "Whiff"),
    bip_rate   = mean(outcome == "BIP"),
    n          = n(),
    .groups    = "drop"
  ) |>
  write_csv("data/viz_cache/postcommit_quartile.csv")

# postcommit_by_pitch.csv  (~27 rows)
# Used by: mean deviation by pitch type x outcome grouped bar
cat("postcommit_by_pitch.csv\n")
sw_pc |>
  group_by(pitch_name, outcome) |>
  summarise(mean_dev = mean(emp_dev_total), .groups = "drop") |>
  filter(!is.na(pitch_name)) |>
  write_csv("data/viz_cache/postcommit_by_pitch.csv")

# ── post_commit_analysis.ipynb cache ──────────────────────────────────────────
sw_pc_full <- sw |>
  left_join(pc_raw, by = c("game_pk", "at_bat_number", "pitch_number")) |>
  filter(pitch_type %in% names(PITCH_NAMES)) |>
  mutate(
    emp_dev_total = sqrt(pc150_emp_dev_x^2 + pc150_emp_dev_z^2),
    lin_dev_total = sqrt(pc150_dev_x^2    + pc150_dev_z^2)
  )

# pc_sample.csv  (30k rows)
# Used by: density (emp vs lin dev), pfx_z scatter plots
cat("pc_sample.csv\n")
set.seed(42)
sw_pc_full |>
  filter(!is.na(emp_dev_total), !is.na(lin_dev_total)) |>
  select(pitch_type, pitch_name, pfx_z,
         emp_dev_total, lin_dev_total,
         pc150_dev_z, pc150_emp_dev_z) |>
  slice_sample(n = 30000) |>
  write_csv("data/viz_cache/pc_sample.csv")

# pc_deception.csv  (~9 rows)
# Used by: deception ratio bar chart
cat("pc_deception.csv\n")
sw_pc_full |>
  filter(!is.na(emp_dev_total), !is.na(lin_dev_total), lin_dev_total > 0.01) |>
  group_by(pitch_type) |>
  summarise(
    n             = n(),
    mean_emp_dev  = mean(emp_dev_total),
    mean_lin_dev  = mean(lin_dev_total),
    emp_lin_ratio = mean_emp_dev / mean_lin_dev,
    .groups       = "drop"
  ) |>
  filter(n >= 10000) |>
  mutate(pitch_name = PITCH_NAMES[pitch_type]) |>
  write_csv("data/viz_cache/pc_deception.csv")

# pc_sensitivity.csv  (~36 rows)
# Used by: commit-time sensitivity line plot
cat("pc_sensitivity.csv\n")
sw_pc_full |>
  filter(!is.na(pc125_dev_total), !is.na(pc150_dev_total),
         !is.na(pc175_dev_total), !is.na(pc200_dev_total)) |>
  select(pitch_type, pc125_dev_total, pc150_dev_total, pc175_dev_total, pc200_dev_total) |>
  pivot_longer(
    cols      = starts_with("pc") & ends_with("dev_total"),
    names_to  = "col",
    values_to = "dev_total"
  ) |>
  mutate(commit_ms = as.integer(str_extract(col, "(?<=pc)\\d+"))) |>
  filter(commit_ms %in% c(125, 150, 175, 200)) |>
  group_by(pitch_type, commit_ms) |>
  summarise(mean_dev = mean(dev_total, na.rm = TRUE), .groups = "drop") |>
  mutate(pitch_name = PITCH_NAMES[pitch_type]) |>
  write_csv("data/viz_cache/pc_sensitivity.csv")

# pc_count_dev.csv  (12 rows)
# Used by: late movement by count heatmap
cat("pc_count_dev.csv\n")
sw_pc_full |>
  filter(!is.na(emp_dev_total), balls <= 3, strikes <= 2) |>
  group_by(balls, strikes) |>
  summarise(
    n            = n(),
    mean_emp_dev = mean(emp_dev_total),
    .groups      = "drop"
  ) |>
  write_csv("data/viz_cache/pc_count_dev.csv")

cat("\nDone. 12 files written to data/viz_cache/\n")
cat("Run with: Rscript src/data/prep_viz_cache.R\n")
