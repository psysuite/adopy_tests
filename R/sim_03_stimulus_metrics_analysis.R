# ============================================================================== =
# 08_stimulus_metrics_analysis.R
# Analysis of stimulus metrics: asymmetry_index, stimulus_center, 
# stimulus_spread
# ============================================================================== =

library(tidyverse)
library(readxl)
library(here)
library(janitor)
library(lme4)
library(lmerTest)
library(permuco)

# Load effect size utilities
source("effect_size_utils.R")


cat("================================================================================\n")
cat("STIMULUS METRICS ANALYSIS: Simulation Fidelity Validation\n")
cat("================================================================================\n\n")

# ============================================================================== =
# SETUP
# ============================================================================== =

# Define paths
if(!exists('root_dir')){
  root_dir = "/data/CODE/python/adopy_tests/"
}

if(!exists('project_name')){
  project_name      <- "R"
}


project_dir <- paste0(root_dir, project_name, "/")
results_filepath <- paste0(project_dir, "results_simulations")

setwd(project_dir)

# ============================================================================== =
# DATA PREPARATION
# ============================================================================== =

# cat("\n=== Data Preparation ===\n")

data_clean <- data_raw %>%
  mutate(
    model = factor(model, levels = c("ABS1", "REL1", "REL2")),
    pse_true = as.numeric(pse_true),
    jnd_true = as.numeric(jnd_true),
    subject_id = factor(subject_id),
    group = factor(group),  # Preserve group column
    trial_block = as.numeric(trial_block),
    
    # Standardize for modeling
    pse_true_z = scale(pse_true)[,1],
    jnd_true_z = scale(jnd_true)[,1]
  ) %>%
  arrange(model, pse_true, jnd_true, subject_id, trial_block)

cat("Data summary:", "  Rows:", nrow(data_clean)," Models:", paste(levels(data_clean$model), collapse = ", "), "  Subjects:", n_distinct(data_clean$subject_id), "\n")

# ============================================================================== =
# EXTRACT FINAL TRIAL BLOCK METRICS
# ============================================================================== =

# cat("\n=== Extracting Final Trial Block Metrics ===\n")

# Get only the final trial block (N=200) for each subject
stimulus_metrics_final <- data_clean %>%
  dplyr::filter(trial_block == 200) %>%
  dplyr::select(
    model, group, pse_true, jnd_true, subject_id,
    asymmetry_index, stimulus_center, stimulus_spread
  ) %>%
  dplyr::mutate(
    pse_true_z = scale(pse_true)[,1],
    jnd_true_z = scale(jnd_true)[,1]
  )

# cat("Final metrics extracted for", n_distinct(stimulus_metrics_final$subject_id), "subjects\n")
# cat("Sample:\n")
# print(head(stimulus_metrics_final, 10))

# ============================================================================== =
# DESCRIPTIVE STATISTICS ====
# ============================================================================== =

cat("\n=== Descriptive Statistics ===\n")

# ...Stimulus Center ====
# Group by PSE level
center_desc <- stimulus_metrics_final %>%
  mutate(
    pse_group = case_when(
      group %in% c("G1", "G2", "G3") ~ "PSE_480",
      group %in% c("G4", "G5", "G6") ~ "PSE_500",
      group %in% c("G7", "G8", "G9") ~ "PSE_520",
      TRUE ~ NA_character_
    )
  ) %>%
  group_by(model, pse_group) %>%
  summarise(
    n = n(),
    mean = mean(stimulus_center, na.rm = TRUE),
    sd = sd(stimulus_center, na.rm = TRUE),
    median = median(stimulus_center, na.rm = TRUE),
    min = min(stimulus_center, na.rm = TRUE),
    max = max(stimulus_center, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nStimulus Center:\n")
print(center_desc, n=9)

# ...Stimulus Spread ====
spread_desc <- stimulus_metrics_final %>%
  group_by(model) %>%
  summarise(
    n = n(),
    mean = mean(stimulus_spread, na.rm = TRUE),
    sd = sd(stimulus_spread, na.rm = TRUE),
    median = median(stimulus_spread, na.rm = TRUE),
    min = min(stimulus_spread, na.rm = TRUE),
    max = max(stimulus_spread, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nStimulus Spread:\n")
print(spread_desc, n=3)

# ============================================================================== =
# STATISTICAL TESTS: ANOVA (Stimulus Center and Spread only) ====
# ============================================================================== =

cat("\n=== Statistical Tests: ANOVA ===\n")

# ...Stimulus Center ====
cat("\n--- Stimulus Center ---\n")
anova_center <- aovperm(stimulus_center ~ model + pse_true_z + jnd_true_z,
                        data = stimulus_metrics_final,
                        np = 5000)
cat("\nANOVA for model effect:\n")
print(anova_center)
# Resampling test using freedman_lane to handle nuisance variables and 5000 permutations.
# SS  df        F parametric P(>F) resampled P(>F)
# model        143.34   2   0.7702           0.4634          0.4706
# pse_true_z 53952.23   1 579.7821           0.0000          0.0002
# jnd_true_z    41.97   1   0.4510           0.5021          0.5034

# Calculate effect sizes
effect_sizes_center <- extract_eta_squared(anova_center)
print_effect_sizes(effect_sizes_center, "Effect Sizes for Stimulus Center (η²)")

cat("\nPost-hoc pairwise comparisons:\n")
res <- do_npar_anova_phpw(stimulus_metrics_final, "model", "stimulus_center", "pse_true_z")
# 
# [1] "in ABS1 (H=151.713118329045, p=9.793335929831e-25)"
# 1] "NOT SIGNIFICANT in REL1 (H=20.2639913813823, p=0.122035661572199)"
# [1] "in REL2 (H=146.124216190015, p=6.41350520967471e-24)"

# ...Stimulus Spread ====
cat("\n--- Stimulus Spread ---\n")
anova_spread <- aovperm(stimulus_spread ~ model + jnd_true_z + pse_true_z,
                        data = stimulus_metrics_final,
                        np = 5000)
cat("\nANOVA for model effect:\n")
print(anova_spread)
# Resampling test using freedman_lane to handle nuisance variables and 5000 permutations.
# SS  df         F parametric P(>F) resampled P(>F)
# model       18408.31   2  209.6687           0.0000          0.0002
# jnd_true_z 164111.92   1 3738.4357           0.0000          0.0002
# pse_true_z     27.84   1    0.6342           0.4262          0.4322

# Calculate effect sizes
effect_sizes_spread <- extract_eta_squared(anova_spread)
print_effect_sizes(effect_sizes_spread, "Effect Sizes for Stimulus Spread (η²)")

cat("\nPost-hoc pairwise comparisons:\n")
do_npar_anova_main(stimulus_metrics_final, "stimulus_spread", "model")

# Main effect: stimulus_spread ~ model, H = 44.4456, p = 0"
# [1] "SIGNIFICANT - Running pairwise comparisons..."
#        Comparison    Stat   p.value  p.adjust
# 1 ABS1 - REL1 = 0  -5.418 6.042e-08 9.063e-08
# 2 ABS1 - REL2 = 0  -6.491 8.547e-11 2.564e-10
# 3 REL1 - REL2 = 0 -0.9134     0.361 3.610e-01

res <- do_npar_anova_phpw(stimulus_metrics_final, "model", "stimulus_spread", "jnd_true_z")

# ============================================================================== =
# CORRELATIONS: Stimulus Metrics vs True Parameters (by Model) ====
# ============================================================================== =

cat("\n=== Correlations: Stimulus Metrics vs True Parameters (by Model) ===\n")

# Initialize list to store correlation results
correlation_results_by_model <- list()

# Test correlations for each model separately
for (m in c("ABS1", "REL1", "REL2")) {
  # cat("\n", strrep("=", 70), "\n", sep = "")
  # cat("MODEL: ", m, "\n", sep = "")[1] "in ABS1 (H=157.624446560685, p=3.19535102223007e-26)"

  # cat(strrep("=", 70), "\n", sep = "")
  
  # Filter data for current model
  data_model <- stimulus_metrics_final %>%
    filter(model == m)
  
  # ...Stimulus Center vs PSE ====
  # cat("\n--- Stimulus Center vs PSE ---\n")
  corr_center_pse <- cor.test(data_model$stimulus_center, 
                              data_model$pse_true,
                              method = "pearson")
  # print(corr_center_pse)
  
  # ...Stimulus Spread vs JND ====
  # cat("\n--- Stimulus Spread vs JND ---\n")
  corr_spread_jnd <- cor.test(data_model$stimulus_spread, 
                              data_model$jnd_true,
                              method = "pearson")
  # print(corr_spread_jnd)
  
  # Store results
  correlation_results_by_model[[m]] <- data.frame(
    model = m,
    comparison = c("Stimulus Center vs PSE", "Stimulus Spread vs JND"),
    r = c(corr_center_pse$estimate, corr_spread_jnd$estimate),
    p_value = c(corr_center_pse$p.value, corr_spread_jnd$p.value),
    n = c(nrow(data_model), nrow(data_model)),
    row.names = NULL
  )
}

# Combine all correlation results
correlation_results_combined <- do.call(rbind, correlation_results_by_model)
rownames(correlation_results_combined) <- NULL

cat("\n", strrep("=", 70), "\n", sep = "")
cat("SUMMARY: Correlations by Model\n")
cat(strrep("=", 70), "\n", sep = "")
print(correlation_results_combined)



# cat("\n=== Stimulus Metrics by PSE/JND Condition ===\n")

# Get the mapping of group to (pse_true, jnd_true) - take the first value per group
group_mapping <- data_clean %>%
  filter(trial_block == 200) %>%
  distinct(group, pse_true, jnd_true) %>%
  group_by(group) %>%
  slice(1) %>%
  ungroup()

# Create summary table: 9 rows per model (one for each group G1-G9)
# Use ONLY the final trial block (200) for each subject
# Aggregate across all subjects within each group
# Columns: model, group, pse_true, jnd_true, and 3 metrics (each with mean and SD)
stimulus_by_condition <- stimulus_metrics_final %>%
  group_by(model, group) %>%
  summarise(
    stimulus_center_mean = mean(stimulus_center, na.rm = TRUE),
    stimulus_center_sd = sd(stimulus_center, na.rm = TRUE),
    stimulus_spread_mean = mean(stimulus_spread, na.rm = TRUE),
    stimulus_spread_sd = sd(stimulus_spread, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(group_mapping, by = "group") %>%
  arrange(model, group)

cat("\nStimulus Metrics by Group (9 groups per model):\n")
print(stimulus_by_condition, n=27)

# Print by model for clarity
# for (m in c("ABS1", "REL1", "REL2")) {
#   cat("\n--- ", m, " ---\n", sep = "")
#   print(filter(stimulus_by_condition, model == m))
# }

# ============================================================================== =
# SAVE RESULTS
# ============================================================================== =

cat("\n=== Saving Results ===\n")

# Save descriptive statistics
write_csv(center_desc, file.path(results_filepath, "tables", "stimulus_center_desc.csv"))
write_csv(spread_desc, file.path(results_filepath, "tables", "stimulus_spread_desc.csv"))
# cat("✓ Saved: descriptive statistics tables\n")

# Save stimulus metrics by condition
write_csv(stimulus_by_condition, file.path(results_filepath, "tables", "stimulus_metrics_by_group.csv"))
# cat("✓ Saved: stimulus metrics by PSE/JND group\n")

# Combine all effect sizes
all_effect_sizes_stim <- bind_rows(
  effect_sizes_center %>% mutate(analysis = "Stimulus Center"),
  effect_sizes_spread %>% mutate(analysis = "Stimulus Spread")
)

# Save effect sizes
write_csv(all_effect_sizes_stim, file.path(results_filepath, "tables", "effect_sizes_stimulus_eta_squared.csv"))
cat("✓ Saved: effect sizes (η²) for stimulus metrics\n")

# Save ANOVA results
anova_stimulus_results <- list(
  stimulus_center = anova_center,
  stimulus_spread = anova_spread
)
saveRDS(anova_stimulus_results, file.path(results_filepath, "models", "anova_stimulus_results.rds"))
cat("✓ Saved: ANOVA results\n")

# Save pairwise comparisons
pairwise_stimulus_results <- list(
  stimulus_center = anova_center,
  stimulus_spread = anova_spread
)
saveRDS(pairwise_stimulus_results, file.path(results_filepath, "models", "pairwise_stimulus_results.rds"))
cat("✓ Saved: pairwise comparison results\n")

# Save correlation results
write_csv(correlation_results_combined, file.path(results_filepath, "tables", "stimulus_correlations_by_model.csv"))
# cat("✓ Saved: correlation results by model\n")

# Save clean data for plotting
saveRDS(stimulus_metrics_final, file.path(results_filepath, "models", "stimulus_metrics_final.rds"))
# cat("✓ Saved: clean stimulus metrics data\n")

cat("\n================================================================================\n")
cat("STIMULUS METRICS ANALYSIS COMPLETE\n")
cat("================================================================================\n")
cat("\nResults saved to:", results_filepath, "\n\n")
