library(sabRmetrics)
library(tidyverse)

# ── 1. Pull 2023–2025 Statcast ───────────────────────────────────────────────
cluster <- parallel::makeCluster(parallel::detectCores())

data_2023 <- sabRmetrics::download_baseballsavant(
  start_date = "2023-01-01",
  end_date   = "2023-12-31",
  cl         = cluster
)
data_2024 <- sabRmetrics::download_baseballsavant(
  start_date = "2024-01-01",
  end_date   = "2024-12-31",
  cl         = cluster
)
data_2025 <- sabRmetrics::download_baseballsavant(
  start_date = "2025-01-01",
  end_date   = "2025-12-31",
  cl         = cluster
)

parallel::stopCluster(cluster)

raw <- bind_rows(data_2023, data_2024, data_2025)
cat("Total rows:", nrow(raw), "\n")
cat("Columns:", paste(names(raw), collapse = ", "), "\n")

# ── 2. Write raw data ─────────────────────────────────────────────────────────
dir.create("data", showWarnings = FALSE)
write.csv(raw, "data/savant_raw.csv", row.names = FALSE)
cat("Saved data/savant_raw.csv\n")

# ── 3. Linear weights from delta_run_exp ─────────────────────────────────────
# For PA-ending pitches, delta_run_exp equals the linear weight of the event.
# We map Statcast event names to the pipeline's outcome_type keys.
outcome_map <- c(
  "single"                        = "single",
  "double"                        = "double",
  "triple"                        = "triple",
  "home_run"                      = "home_run",
  "strikeout"                     = "strikeout",
  "strikeout_double_play"         = "strikeout",
  "field_out"                     = "out_in_play",
  "grounded_into_double_play"     = "out_in_play",
  "force_out"                     = "out_in_play",
  "double_play"                   = "out_in_play",
  "fielders_choice"               = "out_in_play",
  "fielders_choice_out"           = "out_in_play",
  "sac_fly"                       = "out_in_play",
  "sac_fly_double_play"           = "out_in_play",
  "sac_bunt"                      = "out_in_play",
  "walk"                          = "walk",
  "hit_by_pitch"                  = "hit_by_pitch",
  "intent_walk"                   = "walk"
)

lw_raw <- raw %>%
  filter(!is.na(events), events != "") %>%
  mutate(outcome_type = outcome_map[events]) %>%
  filter(!is.na(outcome_type)) %>%
  group_by(outcome_type) %>%
  summarise(lw = mean(delta_run_exp, na.rm = TRUE), n = n()) %>%
  filter(n >= 50) %>%
  select(outcome_type, lw)

dir.create("results", showWarnings = FALSE)
write.csv(lw_raw, "results/linear_weights.csv", row.names = FALSE)
cat("Saved results/linear_weights.csv\n")
print(lw_raw)

# ── 4. Count values from delta_run_exp ───────────────────────────────────────
# Count value = mean remaining run expectancy from this count forward within a PA.
count_vals <- raw %>%
  filter(!is.na(delta_run_exp), !is.na(balls), !is.na(strikes)) %>%
  arrange(game_pk, at_bat_number, pitch_number) %>%
  group_by(game_pk, at_bat_number) %>%
  mutate(remaining_rv = rev(cumsum(rev(delta_run_exp)))) %>%
  ungroup() %>%
  group_by(balls, strikes) %>%
  summarise(expected_run_value = mean(remaining_rv, na.rm = TRUE), n = n()) %>%
  filter(n >= 100) %>%
  select(balls, strikes, expected_run_value)

write.csv(count_vals, "results/count_values.csv", row.names = FALSE)
cat("Saved results/count_values.csv\n")
print(count_vals)
