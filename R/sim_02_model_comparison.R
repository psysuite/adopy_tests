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
    
    # Calculate errors
    pse_error     = pse_est - pse_true,
    pse_error_pct = abs(pse_error) / pse_true * 100,
    jnd_error     = jnd_est - jnd_true,
    jnd_error_pct = abs(jnd_error) / jnd_true * 100,
    
    # Standardize for modeling
    pse_est_z = scale(pse_est)[,1],
    jnd_est_z = scale(jnd_est)[,1],
    
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
# CONVERGENCE METRICS CALCULATION
# ============================================================================== =

cat("\n=== Calculating Convergence Metrics ===\n")

# 1. Stability Point: First trial count where estimate within 10% of final
stability_data <- data_clean %>%
  group_by(model, pse_true, jnd_true, subject_id) %>%
  arrange(trial_block) %>%
  mutate(
    pse_final = last(pse_est),
    jnd_final = last(jnd_est),
    pse_error_pct_from_final = abs(pse_est - pse_final) / abs(pse_final) * 100,
    jnd_error_pct_from_final = abs(jnd_est - jnd_final) / abs(jnd_final) * 100,
    pse_stable = pse_error_pct_from_final < 10,
    jnd_stable = jnd_error_pct_from_final < 10
  ) %>%
  summarise(
    pse_stability_point = ifelse(any(pse_stable), min(trial_block[pse_stable]), 200),
    jnd_stability_point = ifelse(any(jnd_stable), min(trial_block[jnd_stable]), 200),
    pse_final_error = abs(last(pse_error)),
    jnd_final_error = abs(last(jnd_error)),
    pse_final_error_pct = abs(last(pse_error_pct)),
    jnd_final_error_pct = abs(last(jnd_error_pct)),
    .groups = "drop"
  )

# 2. Area Under Curve (AUC): Total distance from final estimate
auc_data <- data_clean %>%
  group_by(model, pse_true, jnd_true, subject_id) %>%
  arrange(trial_block) %>%
  mutate(
    pse_final = last(pse_est),
    jnd_final = last(jnd_est),
    pse_diff = abs(pse_est - pse_final),
    jnd_diff = abs(jnd_est - jnd_final)
  ) %>%
  summarise(
    auc_pse = sum(pse_diff * 20),  # 20 = interval between trial blocks
    auc_jnd = sum(jnd_diff * 20),
    .groups = "drop"
  )

# 3. Convergence trajectory: Slope of convergence
trajectory_data <- data_clean %>%
  group_by(model, pse_true, jnd_true, subject_id) %>%
  arrange(trial_block) %>%
  mutate(
    pse_final = last(pse_est),
    jnd_final = last(jnd_est),
    pse_error_from_final = abs(pse_est - pse_final),
    jnd_error_from_final = abs(jnd_est - jnd_final)
  ) %>%
  summarise(
    # Fit linear model to error vs trial_block
    pse_slope = ifelse(n() > 1, 
                       coef(lm(pse_error_from_final ~ trial_block))[2], 
                       NA),
    jnd_slope = ifelse(n() > 1, 
                       coef(lm(jnd_error_from_final ~ trial_block))[2], 
                       NA),
    .groups = "drop"
  )

# Combine all metrics
convergence_metrics <- stability_data %>%
  left_join(auc_data, by = c("model", "pse_true", "jnd_true", "subject_id")) %>%
  left_join(trajectory_data, by = c("model", "pse_true", "jnd_true", "subject_id"))

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
    mean_error = mean(pse_final_error, na.rm = TRUE),
    sd_error = sd(pse_final_error, na.rm = TRUE),
    mean_error_pct = mean(pse_final_error_pct, na.rm = TRUE),
    sd_error_pct = sd(pse_final_error_pct, na.rm = TRUE),
    .groups = "drop"
  )

print(pse_acc_desc)

jnd_acc_desc <- convergence_metrics %>%
  group_by(model) %>%
  summarise(
    n = n(),
    mean_error = mean(jnd_final_error, na.rm = TRUE),
    sd_error = sd(jnd_final_error, na.rm = TRUE),
    mean_error_pct = mean(jnd_final_error_pct, na.rm = TRUE),
    sd_error_pct = sd(jnd_final_error_pct, na.rm = TRUE),
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
    mean = mean(pse_stability_point, na.rm = TRUE),
    sd = sd(pse_stability_point, na.rm = TRUE),
    median = median(pse_stability_point, na.rm = TRUE),
    min = min(pse_stability_point, na.rm = TRUE),
    max = max(pse_stability_point, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nPSE Stability Point (trials to within 10% of final):\n")
print(pse_stab_desc)

# ..... JND Stability Point ====
jnd_stab_desc <- convergence_metrics %>%
  group_by(model) %>%
  summarise(
    n = n(),
    mean = mean(jnd_stability_point, na.rm = TRUE),
    sd = sd(jnd_stability_point, na.rm = TRUE),
    median = median(jnd_stability_point, na.rm = TRUE),
    min = min(jnd_stability_point, na.rm = TRUE),
    max = max(jnd_stability_point, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nJND Stability Point (trials to within 10% of final):\n")
print(jnd_stab_desc)

# ..... AUC ====
auc_desc <- convergence_metrics %>%
  group_by(model) %>%
  summarise(
    n = n(),
    mean_auc_pse = mean(auc_pse, na.rm = TRUE),
    sd_auc_pse = sd(auc_pse, na.rm = TRUE),
    mean_auc_jnd = mean(auc_jnd, na.rm = TRUE),
    sd_auc_jnd = sd(auc_jnd, na.rm = TRUE),
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

anova_pse_err <- aovperm(pse_final_error ~ model + pse_true_z + jnd_true_z,
                         data = convergence_metrics,
                         np = 5000)
print(anova_pse_err)
#                   SS  df    F             parametric P(>F) resampled P(>F)
# model      2.511e+01   2 8.772e-01           0.4165          0.4276
# pse_true_z 1.732e-02   1 1.211e-03           0.9723          0.9692
# jnd_true_z 1.976e+03   1 1.381e+02           0.0000          0.0002




cat("\nANOVA for model effect:\n")
print(anova_pse_err)

# Calculate effect sizes
effect_sizes_pse_err <- extract_eta_squared(anova_pse_err)
print_effect_sizes(effect_sizes_pse_err, "Effect Sizes for PSE Final Error (η²)")

cat("\nPost-hoc pairwise comparisons:\n")

res <- do_npar_anova_phpw(convergence_metrics, "model", "pse_final_error", "pse_true_z")
# [1] "pse_final_error x pse_true_z splitted by model"
# [1] "NOT SIGNIFICANT in ABS1 (H=21.788013376292, p=0.249228995912993)"
# [1] "NOT SIGNIFICANT in REL1 (H=18.3911669045896, p=0.28431199770755)"
# [1] "NOT SIGNIFICANT in REL2 (H=14.7851494951301, p=0.392997513832787)"

# ..... JND Final Error ====

cat("\n--- JND Final Error ---\n")
anova_jnd_err <- aovperm(jnd_final_error ~ model + pse_true_z + jnd_true_z,
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
anova_pse_stab <- aovperm(pse_stability_point ~ model + pse_true_z + jnd_true_z,
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
anova_jnd_stab <- aovperm(jnd_stability_point ~ model + pse_true_z + jnd_true_z,
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

do_npar_anova_main(convergence_metrics, "jnd_stability_point", "model")
# [1] "Main effect: jnd_stability_point ~ model, H = 11.4252, p = 0.0033"
# Comparison   Stat   p.value  p.adjust
# Comparison  Stat  p.value p.adjust
# 1 ABS1 - REL1 = 0 1.708  0.08772  0.13160
# 2 ABS1 - REL2 = 0 2.785 0.005351  0.01605
# 3 REL1 - REL2 = 0 1.058     0.29  0.29000

# ..... PSE AUC ====
cat("\n--- PSE AUC (Convergence Speed) ---\n")
anova_pse_auc <- aovperm(auc_pse ~ model + pse_true_z + jnd_true_z,
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

do_npar_anova_main(convergence_metrics, "auc_pse", "model")
# [1] "Main effect: auc_pse ~ model, H = 11.8676, p = 0.0026"
# [1] "SIGNIFICANT - Running pairwise comparisons..."
# Comparison   Stat  p.value p.adjust
# 1 ABS1 - REL1 = 0 -1.329   0.1837 0.183700
# 2 ABS1 - REL2 = 0 -3.103 0.001916 0.005748
# 3 REL1 - REL2 = 0 -1.887  0.05916 0.088740

# ..... JND AUC ====
cat("\n--- JND AUC (Convergence Speed) ---\n")
anova_jnd_auc <- aovperm(auc_jnd ~ model + pse_true_z + jnd_true_z,
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

do_npar_anova_main(convergence_metrics, "auc_jnd", "model")

# [1] "Main effect: auc_jnd ~ model"
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
