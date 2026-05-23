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
library(emmeans)

# Load effect size utilities
source("effect_size_utils.R")

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

cat("\nANOVA for model effect:\n")
print(anova_pse_err)

# Calculate effect sizes
effect_sizes_pse_err <- extract_eta_squared(anova_pse_err)
print_effect_sizes(effect_sizes_pse_err, "Effect Sizes for PSE Final Error (η²)")

cat("\nPost-hoc pairwise comparisons (Tukey):\n")
emm_pse_err <- emmeans(lm(pse_final_error ~ model + pse_true_z + jnd_true_z, data = convergence_metrics), ~ model)
pairs_pse_err <- pairs(emm_pse_err, adjust = "tukey")
print(pairs_pse_err)

# ..... JND Final Error ====

cat("\n--- JND Final Error ---\n")
anova_jnd_err <- aovperm(jnd_final_error ~ model + pse_true_z + jnd_true_z,
                         data = convergence_metrics,
                         np = 5000)

cat("\nANOVA for model effect:\n")
print(anova_jnd_err)

# Calculate effect sizes
effect_sizes_jnd_err <- extract_eta_squared(anova_jnd_err)
print_effect_sizes(effect_sizes_jnd_err, "Effect Sizes for JND Final Error (η²)")

cat("\nPost-hoc pairwise comparisons (Tukey):\n")
emm_jnd_err <- emmeans(lm(jnd_final_error ~ model + pse_true_z + jnd_true_z, data = convergence_metrics), ~ model)
pairs_jnd_err <- pairs(emm_jnd_err, adjust = "tukey")
print(pairs_jnd_err)

# ..... PSE Stability Point ====
cat("\n--- PSE Stability Point ---\n")
anova_pse_stab <- aovperm(pse_stability_point ~ model + pse_true_z + jnd_true_z,
                          data = convergence_metrics,
                          np = 5000)

cat("\nANOVA for model effect:\n")
print(anova_pse_stab)

# Calculate effect sizes
effect_sizes_pse_stab <- extract_eta_squared(anova_pse_stab)
print_effect_sizes(effect_sizes_pse_stab, "Effect Sizes for PSE Stability Point (η²)")

cat("\nPost-hoc pairwise comparisons (Tukey):\n")
emm_pse_stab <- emmeans(lm(pse_stability_point ~ model + pse_true_z + jnd_true_z, data = convergence_metrics), ~ model)
pairs_pse_stab <- pairs(emm_pse_stab, adjust = "tukey")
print(pairs_pse_stab)

# ..... JND Stability Point ====
cat("\n--- JND Stability Point ---\n")
anova_jnd_stab <- aovperm(jnd_stability_point ~ model + pse_true_z + jnd_true_z,
                          data = convergence_metrics,
                          np = 5000)

cat("\nANOVA for model effect:\n")
print(anova_jnd_stab)

# Calculate effect sizes
effect_sizes_jnd_stab <- extract_eta_squared(anova_jnd_stab)
print_effect_sizes(effect_sizes_jnd_stab, "Effect Sizes for JND Stability Point (η²)")

cat("\nPost-hoc pairwise comparisons (Tukey):\n")
emm_jnd_stab <- emmeans(lm(jnd_stability_point ~ model + pse_true_z + jnd_true_z, data = convergence_metrics), ~ model)
pairs_jnd_stab <- pairs(emm_jnd_stab, adjust = "tukey")
print(pairs_jnd_stab)

# ..... PSE AUC ====
cat("\n--- PSE AUC (Convergence Speed) ---\n")
anova_pse_auc <- aovperm(auc_pse ~ model + pse_true_z + jnd_true_z,
                         data = convergence_metrics,
                         np = 5000)

cat("\nANOVA for model effect:\n")
print(anova_pse_auc)

# Calculate effect sizes
effect_sizes_pse_auc <- extract_eta_squared(anova_pse_auc)
print_effect_sizes(effect_sizes_pse_auc, "Effect Sizes for PSE AUC (η²)")

cat("\nPost-hoc pairwise comparisons (Tukey):\n")
emm_pse_auc <- emmeans(lm(auc_pse ~ model + pse_true_z + jnd_true_z, data = convergence_metrics), ~ model)
pairs_pse_auc <- pairs(emm_pse_auc, adjust = "tukey")
print(pairs_pse_auc)

# ..... JND AUC ====
cat("\n--- JND AUC (Convergence Speed) ---\n")
anova_jnd_auc <- aovperm(auc_jnd ~ model + pse_true_z + jnd_true_z,
                         data = convergence_metrics,
                         np = 5000)

cat("\nANOVA for model effect:\n")
print(anova_jnd_auc)

# Calculate effect sizes
effect_sizes_jnd_auc <- extract_eta_squared(anova_jnd_auc)
print_effect_sizes(effect_sizes_jnd_auc, "Effect Sizes for JND AUC (η²)")

cat("\nPost-hoc pairwise comparisons (Tukey):\n")
emm_jnd_auc <- emmeans(lm(auc_jnd ~ model + pse_true_z + jnd_true_z, data = convergence_metrics), ~ model)
pairs_jnd_auc <- pairs(emm_jnd_auc, adjust = "tukey")
print(pairs_jnd_auc)

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
