# leaderboard_table.R
# Pitcher leaderboard tables using gt + mlbplotR.
# Produces two PNGs: sorted by xRV Residual and by Distortion Tax.
#
# Run from project root:
#   Rscript results_scripts/leaderboard_table.R
#
# Required packages:
#   install.packages(c("arrow", "dplyr", "gt", "gtExtras", "mlbplotR", "scales", "webshot2"))

library(arrow)
library(dplyr)
library(gt)
library(mlbplotR)
library(scales)

dir.create("results/figures", showWarnings = FALSE, recursive = TRUE)

# ── Load data ─────────────────────────────────────────────────────────────────

xrv <- read_parquet("results/xrv_causal.parquet")

names_df <- read_parquet(
  "data/swings_precommit.parquet",
  col_select = c("game_pk", "at_bat_number", "pitch_number", "pitcher_full_name")
) |>
  distinct(game_pk, at_bat_number, pitch_number, .keep_all = TRUE)

df <- xrv |>
  left_join(names_df, by = c("game_pk", "at_bat_number", "pitch_number"))

# ── Aggregate by pitcher × pitch type ─────────────────────────────────────────

PITCH_LABELS <- c(
  FF = "Four-Seam FB", SI = "Sinker",    CH = "Changeup",
  SL = "Slider",       CU = "Curveball", FC = "Cutter",
  FS = "Splitter",     KC = "Knuckle-Curve", ST = "Sweeper",
  SV = "Slurve",       FO = "Forkball",  KN = "Knuckleball"
)

by_pitch <- df |>
  filter(!is.na(disruption_tax), !is.na(pitcher_full_name)) |>
  group_by(pitcher_id, pitcher_full_name, pitch_type) |>
  summarise(
    Pitches    = n(),
    xRV        = round(mean(disruption_tax,          na.rm = TRUE), 3),
    AdjXRV     = round(mean(adjusted_disruption_tax, na.rm = TRUE), 3),
    Distortion = round(mean(distortion_tax,           na.rm = TRUE), 3),
    Selection  = round(mean(selection_tax,            na.rm = TRUE), 3),
    MissTax    = round(mean(miss_distortion_tax,      na.rm = TRUE), 3),
    # decision_cost: positive = taking was better (batter chased). Reported as
    # a positive "Chase" number so higher = more batter mistakes on these pitches.
    Chase      = round(mean(decision_cost,            na.rm = TRUE), 3),
    .groups = "drop"
  ) |>
  filter(Pitches >= 100) |>
  mutate(
    pitch_label    = coalesce(PITCH_LABELS[pitch_type], pitch_type),
    # Inverted percentile ranks on full dataset: 100 = best (lowest raw value)
    xRV_pct        = as.integer(round((n() + 1 - rank(xRV))        / n() * 100)),
    adjxrv_pct     = as.integer(round((n() + 1 - rank(AdjXRV))     / n() * 100)),
    distortion_pct = as.integer(round((n() + 1 - rank(Distortion)) / n() * 100))
  )

# Color palette: 0 = blue (worst), 100 = red (best) — used for Table 3 percentiles
PCT_PAL <- col_numeric(c("#2166ac", "#f7f7f7", "#d73027"), domain = c(0, 100))
# Raw-value palettes anchored to full-dataset range so leaderboard rows are
# colored relative to all pitchers, not just the 18 displayed.
ADJXRV_PAL <- col_numeric(c("#d73027", "#f7f7f7", "#2166ac"),
                           domain = range(by_pitch$AdjXRV,  na.rm = TRUE))
DIST_PAL   <- col_numeric(c("#d73027", "#f7f7f7", "#2166ac"),
                           domain = range(by_pitch$Distortion, na.rm = TRUE))

# ── Shared table styling ───────────────────────────────────────────────────────

apply_style <- function(tbl) {
  tbl |>
    tab_style(
      style     = list(cell_text(weight = "bold")),
      locations = cells_column_labels(everything())
    ) |>
    tab_options(
      table.font.size                 = 13,
      data_row.padding                = px(4),
      table.width                     = px(900),
      heading.background.color        = "#f7f7f7",
      column_labels.background.color  = "#f0f0f0",
      table.border.top.color          = "#cccccc",
      table.border.bottom.color       = "#cccccc"
    )
}

save_png <- function(tbl, path) {
  tryCatch(
    { gtsave(tbl, path, vwidth = 1400, vheight = 900); cat("Saved", path, "\n") },
    error = function(e) cat("PNG export skipped:", conditionMessage(e), "\n")
  )
}

# ── Table 1: top 18 by Adjusted xRV percentile ────────────────────────────────
# AdjXRV = disruption_tax − max(0, decision_cost): total burden vs. optimal action.
# Chase > 0 means batters were better off taking the pitch at the projected location.

top18_xrv <- by_pitch |>
  arrange(AdjXRV) |>
  slice_head(n = 18)

tbl_xrv <- top18_xrv |>
  select(pitcher_id, pitcher_full_name, pitch_label, Pitches,
         xRV, AdjXRV, Chase) |>
  gt() |>
  gt_fmt_mlb_headshot(columns = pitcher_id, height = 35) |>
  cols_label(
    pitcher_id        = "",
    pitcher_full_name = "Pitcher",
    pitch_label       = "Pitch",
    Pitches           = "Pitches",
    xRV               = "Swing Disruption",
    AdjXRV            = "Total Burden",
    Chase             = "Chase Cost"
  ) |>
  data_color(
    columns = AdjXRV,
    fn      = ADJXRV_PAL
  ) |>
  tab_header(
    title    = md("**Pitcher Total Burden Leaderboard**"),
    subtitle = md("Sorted by Adjusted xRV (swing disruption + chase penalty)  ·  ≥100 pitches  ·  2023–2025")
  ) |>
  tab_footnote(
    footnote  = "Total Burden (Adj. xRV): disruption_tax − max(0, decision_cost). When taking was better, baseline shifts to take value. Negative = pitcher advantage.",
    locations = cells_column_labels(AdjXRV)
  ) |>
  tab_footnote(
    footnote  = "Chase Cost: mean decision_cost per swing. Positive = batters were better off taking the pitch at its projected location.",
    locations = cells_column_labels(Chase)
  ) |>
  apply_style()

save_png(tbl_xrv, "results/figures/leaderboard_total_burden.png")

# ── Table 2: top 18 by Distortion Tax percentile ──────────────────────────────
# MissTax: physical bat-to-ball miss channel — independent corroboration of Distortion.

top18_dist <- by_pitch |>
  arrange(Distortion) |>
  slice_head(n = 18)

tbl_dist <- top18_dist |>
  select(pitcher_id, pitcher_full_name, pitch_label, Pitches,
         Distortion, MissTax, AdjXRV) |>
  gt() |>
  gt_fmt_mlb_headshot(columns = pitcher_id, height = 35) |>
  cols_label(
    pitcher_id        = "",
    pitcher_full_name = "Pitcher",
    pitch_label       = "Pitch",
    Pitches           = "Pitches",
    Distortion        = "Distortion Tax",
    MissTax           = "Physical Miss Tax",
    AdjXRV            = "Total Burden"
  ) |>
  data_color(
    columns = Distortion,
    fn      = DIST_PAL
  ) |>
  tab_header(
    title    = md("**Pitcher Distortion Tax Leaderboard**"),
    subtitle = md("Movement-caused swing deviation cost  ·  ≥100 pitches  ·  2023–2025")
  ) |>
  tab_footnote(
    footnote  = "Physical Miss Tax: run-value cost from movement-caused increase in bat-to-ball distance. Negative = pitcher advantage. Independent of angular-deviation channel.",
    locations = cells_column_labels(MissTax)
  ) |>
  apply_style()

save_png(tbl_dist, "results/figures/leaderboard_distortion.png")

# ── Table 3: combined bottom / top distortion tax (pitcher-season, 2024, EB-shrunk) ─────
# Reads EB-shrunk values from results/leaderboard.csv (written by skill_analysis.py).
# Shows 10 most disruptive (most negative) + 10 most batter-favorable (most positive)
# in one table with row groups, headshots, and the same palette as Tables 1-2.

lb_raw <- read.csv("results/leaderboard.csv") |>
  filter(game_year == 2024)

n_tot <- nrow(lb_raw)

lb24 <- lb_raw |>
  mutate(
    dist_pct = as.integer(round((n_tot + 1 - rank(distortion_tax_shrunk)) / n_tot * 100))
  )

bot10 <- lb24 |>
  arrange(distortion_tax_shrunk) |>
  slice_head(n = 10) |>
  mutate(group = "Most Disruptive  (Pitcher Advantage)")

top10 <- lb24 |>
  arrange(desc(distortion_tax_shrunk)) |>
  slice_head(n = 10) |>
  mutate(group = "Most Batter-Favorable  (Movement Helps Batter)")

combined <- bind_rows(bot10, top10) |>
  mutate(
    distortion_tax_mean   = round(distortion_tax_mean,   4),
    distortion_tax_shrunk = round(distortion_tax_shrunk, 4)
  )

tbl_combined <- combined |>
  select(pitcher_id, pitcher_full_name, group, n,
         dist_pct, distortion_tax_mean, distortion_tax_shrunk) |>
  gt(groupname_col = "group") |>
  gt_fmt_mlb_headshot(columns = pitcher_id, height = 35) |>
  cols_label(
    pitcher_id            = "",
    pitcher_full_name     = "Pitcher",
    n                     = "Swings",
    dist_pct              = "Distortion Pctile",
    distortion_tax_mean   = "Raw Mean",
    distortion_tax_shrunk = "EB-Shrunk"
  ) |>
  data_color(
    columns = dist_pct,
    fn      = PCT_PAL
  ) |>
  fmt_number(
    columns = c(distortion_tax_mean, distortion_tax_shrunk),
    decimals = 4
  ) |>
  tab_header(
    title    = md("**Pitcher Distortion Tax Leaderboard — 2024 Season**"),
    subtitle = md("Most disruptive (negative) and most batter-favorable (positive)  ·  EB shrinkage  ·  min 200 swings")
  ) |>
  tab_footnote(
    footnote  = "Distortion Tax: run-value cost of post-commit movement. Negative = pitcher advantage (movement displaced ball to worse position for batter).",
    locations = cells_column_labels(distortion_tax_shrunk)
  ) |>
  tab_footnote(
    footnote  = "EB-Shrunk: empirical-Bayes estimate shrunk toward grand mean by reliability weight n / (n + within-var / between-var).",
    locations = cells_column_labels(distortion_tax_shrunk)
  ) |>
  tab_style(
    style     = cell_fill(color = "#fde8e8"),
    locations = cells_row_groups(groups = "Most Disruptive  (Pitcher Advantage)")
  ) |>
  tab_style(
    style     = cell_fill(color = "#e8f4fd"),
    locations = cells_row_groups(groups = "Most Batter-Favorable  (Movement Helps Batter)")
  ) |>
  apply_style()

save_png(tbl_combined, "results/figures/distortion_tax_leaderboard_2024.png")
