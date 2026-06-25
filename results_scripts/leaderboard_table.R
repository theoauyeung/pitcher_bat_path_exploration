# 09_leaderboard_table.R
# Pitcher distortion / xRV-residual leaderboard table using gt + mlbplotR.
#
# Run from project root:
#   Rscript results_scripts/09_leaderboard_table.R
#
# Required packages:
#   install.packages(c("arrow", "dplyr", "gt", "gtExtras", "mlbplotR", "scales"))
#
# Outputs:
#   results/figures/09_leaderboard_by_pitch.html
#   results/figures/09_leaderboard_by_pitch.png   (if webshot2 installed)

library(arrow)
library(dplyr)
library(gt)
library(mlbplotR)
library(scales)

dir.create("results/figures", showWarnings = FALSE, recursive = TRUE)

# ── Load data ─────────────────────────────────────────────────────────────────

xrv <- read_parquet("results/xrv_causal.parquet")

# Pull pitcher names and pitch type from swings parquet (avoid loading all cols)
names_df <- read_parquet(
  "data/swings_precommit.parquet",
  col_select = c("game_pk", "at_bat_number", "pitch_number",
                 "pitcher_full_name")
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
    Pitches       = n(),
    xRV_Residual  = round(mean(disruption_tax, na.rm = TRUE), 3),
    Distortion    = round(mean(distortion_tax,  na.rm = TRUE), 3),
    Selection     = round(mean(selection_tax,   na.rm = TRUE), 3),
    .groups = "drop"
  ) |>
  filter(Pitches >= 100) |>
  mutate(
    pitch_label = coalesce(PITCH_LABELS[pitch_type], pitch_type)
  ) |>
  arrange(xRV_Residual)   # most negative = best pitcher advantage

# ── Build gt table (top 18 by xRV Residual) ───────────────────────────────────

top18 <- by_pitch |>
  slice_head(n = 18)

tbl <- top18 |>
  select(pitcher_id, pitcher_full_name, pitch_label,
         Pitches, xRV_Residual, Distortion, Selection) |>
  gt() |>

  # Headshot via MLBAM ID
  gt_fmt_mlb_headshot(columns = pitcher_id, height = 35) |>

  # Column labels
  cols_label(
    pitcher_id        = "",
    pitcher_full_name = "Pitcher",
    pitch_label       = "Pitch",
    Pitches           = "Pitches",
    xRV_Residual      = "xRV Residual",
    Distortion        = "Distortion Tax",
    Selection         = "Selection Tax"
  ) |>

  # Colour scale by percentile rank: red = lowest (best pitcher), blue = highest
  data_color(
    columns = xRV_Residual,
    fn = function(x) {
      pct <- rank(x) / length(x)
      col_numeric(c("#d73027", "#f7f7f7", "#2166ac"), domain = c(0, 1))(pct)
    }
  ) |>

  # Header
  tab_header(
    title    = md("**Pitcher xRV Residual Leaderboard**"),
    subtitle = md("Mean run-value cost per swing  ·  min. 100 pitches  ·  2023–2025")
  ) |>

  # Footnote explaining the metrics
  tab_footnote(
    footnote = "xRV Residual: total swing-disruption cost (realized − intended xRV). Distortion: movement-caused share. Selection: batter-decision share.",
    locations = cells_column_labels(xRV_Residual)
  ) |>

  # Bold column headers
  tab_style(
    style     = list(cell_text(weight = "bold")),
    locations = cells_column_labels(everything())
  ) |>

  # Table styling
  tab_options(
    table.font.size            = 13,
    data_row.padding           = px(4),
    table.width                = px(900),
    heading.background.color   = "#f7f7f7",
    column_labels.background.color = "#f0f0f0",
    table.border.top.color     = "#cccccc",
    table.border.bottom.color  = "#cccccc"
  )

# ── Save ──────────────────────────────────────────────────────────────────────

png_out <- "results/figures/leaderboard_by_pitch.png"
tryCatch(
  { gtsave(tbl, png_out, vwidth = 1400, vheight = 900); cat("Saved", png_out, "\n") },
  error = function(e) cat("PNG export skipped (install webshot2 for PNG):", conditionMessage(e), "\n")
)
