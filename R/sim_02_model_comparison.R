# ============================================================================== =
# 06_model_comparison.R
# Model comparison: 1model_abs vs 2model_rel
# Convergence metrics and statistical analysis
# ============================================================================== =

library(tidyverse)
library(readxl)
library(here)
library(janitor)
library(permuco)

# Load effect size utilities
source("effect_size_utils.R")
source("npar_posthoc.R")

cat("================================================================================\n")
cat("MODEL COMPARISON ANALYSIS: ABS1 vs REL1 vs REL2\n")
cat("================================================================================\n")

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

cat("\n=== Data Preparation ===\n")

data_clean <- data_raw %>%
  # Ensure proper data types
  mutate(
    model = factor(model, levels = c("ABS1", "REL1", "REL2")),
    pse_true = as.numeric(pse_true),
    jnd_true = as.numeric(jnd_true),
    subject_id = factor(subject_id),
    group = factor(group),  # Preserve group column
    trial_block = as.numeric(trial_block),
    trial_block_f = factor(trial_block),
    
    # Calculate errors (note: pse_error and jnd_error are now pre-computed in Excel)
    pse_error     = pse - pse_true,
    pse_error_pct = abs(pse_error) / pse_true * 100,
    jnd_error     = jnd - jnd_true,
    jnd_error_pct = abs(jnd_error) / jnd_true * 100,
    
    # Standardize for modeling
    pse_est_z = scale(pse)[,1],
    jnd_est_z = scale(jnd)[,1],
    
    pse_true_z = scale(pse_true)[,1],
    jnd_true_z = scale(jnd_true)[,1],
    
    trial_block_z = scale(trial_block)[,1]
  ) %>%
  arrange(model, pse_true, jnd_true, subject_id, trial_block)

cat("\nCleaned data summary:\n")
cat("  Rows:", nrow(data_clean), "\n")
cat("  Models:", paste(levels(data_clean$model), collapse = ", "), "\n")
cat("  PSE values:", paste(sort(unique(data_clean$pse_true)), collapse = ", "), "\n")
cat("  JND values:", paste(sort(unique(data_clean$jnd_true)), collapse = ", "), "\n")
cat("  Trial blocks:", paste(sort(unique(data_clean$trial_block)), collapse = ", "), "\n")
cat("  Subjects:", n_distinct(data_clean$subject_id), "\n")

# ============================================================================== =
# CONVERGENCE METRICS - READ FROM EXCEL (NO RECALCULATION)
# ============================================================================== =

cat("\n=== Loading Pre-calculated Convergence Metrics from Excel ===\n")

# NOTE: All convergence metrics are PRE-CALCULATED in the Excel file:
# - pse_stability_block, jnd_stability_block (first trial block within 10% of ground truth)
# - pse_auc, jnd_auc (cumulative distance from ground truth across all blocks)
# - pse_error, pse_error_pct (final error vs ground truth)
# - jnd_error, jnd_error_pct (final error vs ground truth)
#
# We extract unique values per subject (they are constant across trial_blocks).
# NO recalculation in R - Excel is authoritative.

convergence_metrics <- data_clean %>%
  dplyr::select(
    model, group, subject_id, pse_true, jnd_true,
    pse_stability_block, jnd_stability_block,
    pse_auc, jnd_auc,
    pse_error, pse_error_pct,
    jnd_error, jnd_error_pct
  ) %>%
  distinct() %>%
  arrange(model, group, subject_id)

cat(sprintf("  Loaded %d convergence metrics\n", nrow(convergence_metrics)))

# ============================================================================== =
# DESCRIPTIVE STATISTICS ====
# ============================================================================== =

cat("\n=== Descriptive Statistics ===\n")

# ..... PSE Final Accuracy ====

cat("\nPSE Final Accuracy (N=200):\n")

# Final Accuracy
pse_acc_desc <- convergence_metrics %>%
  group_by(model) %>%
  summarise(
    n = n(),
    mean_error = mean(pse_error, na.rm = TRUE),
    sd_error = sd(pse_error, na.rm = TRUE),
    mean_error_pct = mean(pse_error_pct, na.rm = TRUE),
    sd_error_pct = sd(pse_error_pct, na.rm = TRUE),
    .groups = "drop"
  )

print(pse_acc_desc)

jnd_acc_desc <- convergence_metrics %>%
  group_by(model) %>%
  summarise(
    n = n(),
    mean_error = mean(jnd_error, na.rm = TRUE),
    sd_error = sd(jnd_error, na.rm = TRUE),
    mean_error_pct = mean(jnd_error_pct, na.rm = TRUE),
    sd_error_pct = sd(jnd_error_pct, na.rm = TRUE),
    .groups = "drop"
  )

# ..... JND Final Accuracy ====

cat("\nJND Final Accuracy (N=200):\n")
print(jnd_acc_desc)

# ..... PSE Stability Point ====

# PSE Stability Point
pse_stab_desc <- convergence_metrics %>%
  group_by(model) %>%
  summarise(
    n = n(),
    mean = mean(pse_stability_block, na.rm = TRUE),
    sd = sd(pse_stability_block, na.rm = TRUE),
    median = median(pse_stability_block, na.rm = TRUE),
    min = min(pse_stability_block, na.rm = TRUE),
    max = max(pse_stability_block, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nPSE Stability Point (trials to within 10% of final):\n")
print(pse_stab_desc)

# ..... JND Stability Point ====
jnd_stab_desc <- convergence_metrics %>%
  group_by(model) %>%
  summarise(
    n = n(),
    mean = mean(jnd_stability_block, na.rm = TRUE),
    sd = sd(jnd_stability_block, na.rm = TRUE),
    median = median(jnd_stability_block, na.rm = TRUE),
    min = min(jnd_stability_block, na.rm = TRUE),
    max = max(jnd_stability_block, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nJND Stability Point (trials to within 10% of final):\n")
print(jnd_stab_desc)

# ..... AUC ====
auc_desc <- convergence_metrics %>%
  group_by(model) %>%
  summarise(
    n = n(),
    mean_pse_auc = mean(pse_auc, na.rm = TRUE),
    sd_pse_auc = sd(pse_auc, na.rm = TRUE),
    mean_jnd_auc = mean(jnd_auc, na.rm = TRUE),
    sd_jnd_auc = sd(jnd_auc, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nArea Under Curve (lower = faster convergence):\n")
print(auc_desc)

# ============================================================================== =
# STATISTICAL TESTS ====
# ============================================================================== =

cat("\n=== Statistical Tests ===\n")

# Add standardized variables to convergence_metrics
convergence_metrics <- convergence_metrics %>%
  mutate(
    pse_true_z = scale(pse_true)[,1],
    jnd_true_z = scale(jnd_true)[,1]
  )

# ..... PSE Final Error ====
cat("\n--- PSE Final Error ---\n")

anova_pse_err <- aovperm(pse_error ~ model + pse_true_z + jnd_true_z,
                         data = convergence_metrics,
                         np = 5000)
print(anova_pse_err)

cat("\nANOVA for model effect:\n")
print(anova_pse_err)

# Calculate effect sizes
effect_sizes_pse_err <- extract_eta_squared(anova_pse_err)
print_effect_sizes(effect_sizes_pse_err, "Effect Sizes for PSE Final Error (η²)")

cat("\nPost-hoc pairwise comparisons:\n")

res <- do_npar_anova_phpw(convergence_metrics, "model", "pse_error", "pse_true_z")

# ..... JND Final Error ====

cat("\n--- JND Final Error ---\n")
anova_jnd_err <- aovperm(jnd_error ~ model + pse_true_z + jnd_true_z,
                         data = convergence_metrics,
                         np = 5000)

cat("\nANOVA for model effect:\n")
print(anova_jnd_err)
# Resampling test using freedman_lane to handle nuisance variables and 5000 permutations.
#                   SS  df         F parametric P(>F) resampled P(>F)
# model      1.173e+01   2   0.19372           0.8239          0.8360
# pse_true_z 5.328e-01   1   0.01759           0.8945          0.8842
# jnd_true_z 6.145e+03   1 202.89798           0.0000          0.0002

# Calculate effect sizes
effect_sizes_jnd_err <- extract_eta_squared(anova_jnd_err)
print_effect_sizes(effect_sizes_jnd_err, "Effect Sizes for JND Final Error (η²)")


# ..... PSE Stability Point ====
cat("\n--- PSE Stability Point ---\n")
anova_pse_stab <- aovperm(pse_stability_block ~ model + pse_true_z + jnd_true_z,
                          data = convergence_metrics,
                          np = 5000)
cat("\nANOVA for model effect:\n")
print(anova_pse_stab)
# Resampling test using freedman_lane to handle nuisance variables and 5000 permutations.
# SS  df     F parametric P(>F) resampled P(>F)
# model        10.40   2 1.431          0.23994          0.2426
# pse_true_z   10.41   1 2.865          0.09113          0.0902
# jnd_true_z   17.00   1 4.678          0.03099          0.0306   *

# Calculate effect sizes
effect_sizes_pse_stab <- extract_eta_squared(anova_pse_stab)
print_effect_sizes(effect_sizes_pse_stab, "Effect Sizes for PSE Stability Point (η²)")

# ..... JND Stability Point ====
cat("\n--- JND Stability Point ---\n")
anova_jnd_stab <- aovperm(jnd_stability_block ~ model + pse_true_z + jnd_true_z,
                          data = convergence_metrics,
                          np = 5000)
cat("\nANOVA for model effect:\n")
print(anova_jnd_stab)
# Resampling test using freedman_lane to handle nuisance variables and 5000 permutations.
# SS  df       F parametric P(>F) resampled P(>F)
# model       13695.43   2 3.93402          0.02013          0.0200
# pse_true_z     31.84   1 0.01829          0.89246          0.8984
# jnd_true_z     51.12   1 0.02937          0.86399          0.8718

# Calculate effect sizes
effect_sizes_jnd_stab <- extract_eta_squared(anova_jnd_stab)
print_effect_sizes(effect_sizes_jnd_stab, "Effect Sizes for JND Stability Point (η²)")

do_npar_anova_main(convergence_metrics, "jnd_stability_block", "model")
# [1] "Main effect: jnd_stability_block ~ model, H = 11.4252, p = 0.0033"
# Comparison   Stat   p.value  p.adjust
# Comparison  Stat  p.value p.adjust
# 1 ABS1 - REL1 = 0 1.708  0.08772  0.13160
# 2 ABS1 - REL2 = 0 2.785 0.005351  0.01605
# 3 REL1 - REL2 = 0 1.058     0.29  0.29000

# ..... PSE AUC ====
cat("\n--- PSE AUC (Convergence Speed) ---\n")
anova_pse_auc <- aovperm(pse_auc ~ model + pse_true_z + jnd_true_z,
                         data = convergence_metrics,
                         np = 5000)
cat("\nANOVA for model effect:\n")
print(anova_pse_auc)
# Resampling test using freedman_lane to handle nuisance variables and 5000 permutations.
# SS  df       F parametric P(>F) resampled P(>F)
# model        2936025   2   7.652        0.0005288          0.0012
# pse_true_z   1042927   1   5.436        0.0200906          0.0200
# jnd_true_z  51977697   1 270.945        0.0000000          0.0002

# Calculate effect sizes
effect_sizes_pse_auc <- extract_eta_squared(anova_pse_auc)
print_effect_sizes(effect_sizes_pse_auc, "Effect Sizes for PSE AUC (η²)")

cat("\nPost-hoc pairwise comparisons (Tukey):\n")

do_npar_anova_main(convergence_metrics, "pse_auc", "model")
# [1] "Main effect: pse_auc ~ model, H = 11.8676, p = 0.0026"
# [1] "SIGNIFICANT - Running pairwise comparisons..."
# Comparison   Stat  p.value p.adjust
# 1 ABS1 - REL1 = 0 -1.329   0.1837 0.183700
# 2 ABS1 - REL2 = 0 -3.103 0.001916 0.005748
# 3 REL1 - REL2 = 0 -1.887  0.05916 0.088740

# ..... JND AUC ====
cat("\n--- JND AUC (Convergence Speed) ---\n")
anova_jnd_auc <- aovperm(jnd_auc ~ model + pse_true_z + jnd_true_z,
                         data = convergence_metrics,
                         np = 5000)
cat("\nANOVA for model effect:\n")
print(anova_jnd_auc)
# Resampling test using freedman_lane to handle nuisance variables and 5000 permutations.
# SS  df        F parametric P(>F) resampled P(>F)
# model        8540103   2  13.4689        1.964e-06          0.0002
# pse_true_z    173903   1   0.5485        4.592e-01          0.4746
# jnd_true_z  71683158   1 226.1073        0.000e+00          0.0002

# Calculate effect sizes
effect_sizes_jnd_auc <- extract_eta_squared(anova_jnd_auc)
print_effect_sizes(effect_sizes_jnd_auc, "Effect Sizes for JND AUC (η²)")

do_npar_anova_main(convergence_metrics, "jnd_auc", "model")

# [1] "Main effect: jnd_auc ~ model"
# [1] "H = 11.2766, p = 0.0036"
# [1] "SIGNIFICANT - Running pairwise comparisons..."
# Comparison   Stat   p.value  p.adjust
# 1 ABS1 - REL1 = 0  3.123  0.001792 0.0026880
# 2 ABS1 - REL2 = 0  3.679 0.0002342 0.0007026
# 3 REL1 - REL2 = 0 0.7462    0.4556 0.4556000

# ============================================================================== =
# SAVE RESULTS ====
# ============================================================================== =

cat("\n=== Saving Results ===\n")

# Save convergence metrics
write_csv(convergence_metrics, file.path(results_filepath, "tables", "convergence_metrics.csv"))

# Save descriptive statistics
write_csv(pse_stab_desc, file.path(results_filepath, "tables", "pse_stability_desc.csv"))
write_csv(jnd_stab_desc, file.path(results_filepath, "tables", "jnd_stability_desc.csv"))
write_csv(pse_acc_desc, file.path(results_filepath, "tables", "pse_accuracy_desc.csv"))
write_csv(jnd_acc_desc, file.path(results_filepath, "tables", "jnd_accuracy_desc.csv"))
write_csv(auc_desc, file.path(results_filepath, "tables", "auc_desc.csv"))
cat("✓ Saved: descriptive statistics tables\n")

# Combine all effect sizes
all_effect_sizes <- bind_rows(
  effect_sizes_pse_err %>% mutate(analysis = "PSE Final Error"),
  effect_sizes_jnd_err %>% mutate(analysis = "JND Final Error"),
  effect_sizes_pse_stab %>% mutate(analysis = "PSE Stability Point"),
  effect_sizes_jnd_stab %>% mutate(analysis = "JND Stability Point"),
  effect_sizes_pse_auc %>% mutate(analysis = "PSE AUC"),
  effect_sizes_jnd_auc %>% mutate(analysis = "JND AUC")
)

# Save effect sizes
write_csv(all_effect_sizes, file.path(results_filepath, "tables", "effect_sizes_eta_squared.csv"))
cat("✓ Saved: effect sizes (η²) table\n")

# Save ANOVA results
anova_results <- list(
  pse_stability = anova_pse_stab,
  jnd_stability = anova_jnd_stab,
  pse_error = anova_pse_err,
  jnd_error = anova_jnd_err,
  pse_auc = anova_pse_auc,
  jnd_auc = anova_jnd_auc
)
saveRDS(anova_results, file.path(results_filepath, "models", "anova_results.rds"))
cat("✓ Saved: ANOVA results\n")

# Save clean data for plotting
saveRDS(data_clean, file.path(results_filepath, "data_clean.rds"))
saveRDS(convergence_metrics, file.path(results_filepath, "convergence_metrics.rds"))
cat("✓ Saved: clean data for plotting\n")

cat("\n================================================================================\n")
cat("MODEL COMPARISON ANALYSIS COMPLETE\n")
cat("================================================================================\n")
cat("\nResults saved to:", results_filepath, "\n")
