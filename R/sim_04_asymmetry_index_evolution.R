# ============================================================================== =
# 10_asymmetry_index_evolution.R
# Analysis of Asymmetry Index evolution across trial blocks
# PRIMARY: AI and |AI| evolution as ordered categorical
# SECONDARY: Early vs Late comparison
# ============================================================================== =

library(tidyverse)
library(readxl)
library(here)
library(janitor)
library(lme4)
library(lmerTest)
library(permuco)
library(emmeans)

# Load effect size utilities
source("effect_size_utils.R")


cat("================================================================================\n")
cat("ASYMMETRY INDEX EVOLUTION ANALYSIS\n")
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
    trial_block_f = factor(trial_block),
    asymmetry_index_abs = abs(asymmetry_index),
    pse_true_z = scale(pse_true)[,1],
    jnd_true_z = scale(jnd_true)[,1]
  ) %>%
  arrange(model, pse_true, jnd_true, subject_id, trial_block)

cat("Data summary:  Rows:", nrow(data_clean), "  Models:", paste(levels(data_clean$model), collapse = ", "), "  Trial blocks:", paste(sort(unique(data_clean$trial_block)), collapse = ", "), "\n")
# cat("  Subjects:", n_distinct(data_clean$subject_id), "\n")

# ============================================================================== =
# DESCRIPTIVE STATISTICS ====
# ...Evolution of AI and |AI|  ====
# ============================================================================== =

cat("\n=== PRIMARY ANALYSIS: Asymmetry Index Evolution (Ordered Categorical) ===\n")

# ...... AI ====
cat("\n--- AI (Real Values) by Model and Trial Block ---\n")
ai_trajectory <- data_clean %>%
  group_by(model, trial_block) %>%
  summarise(
    n = n(),
    mean_ai = mean(asymmetry_index, na.rm = TRUE),
    sd_ai = sd(asymmetry_index, na.rm = TRUE),
    se_ai = sd_ai / sqrt(n),
    median_ai = median(asymmetry_index, na.rm = TRUE),
    min_ai = min(asymmetry_index, na.rm = TRUE),
    max_ai = max(asymmetry_index, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nMean AI (real) by model and trial block:\n")
print(ai_trajectory, n=9)

# ......|AI| ====

cat("\n--- |AI| (Absolute Values) by Model and Trial Block ---\n") 
aiabs_trajectory <- data_clean %>%
  group_by(model, trial_block) %>%
  summarise(
    n = n(),
    mean_ai_abs = mean(asymmetry_index_abs, na.rm = TRUE),
    sd_ai_abs = sd(asymmetry_index_abs, na.rm = TRUE),
    se_ai_abs = sd_ai_abs / sqrt(n),
    median_ai_abs = median(asymmetry_index_abs, na.rm = TRUE),
    min_ai_abs = min(asymmetry_index_abs, na.rm = TRUE),
    max_ai_abs = max(asymmetry_index_abs, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nMean |AI| (absolute) by model and trial block:\n")
print(aiabs_trajectory, n=9)

# ============================================================================== =
# ...Final Trial Block (N=200) by PSE Group ====
# ============================================================================== =

cat("\n=== Descriptive Statistics: Final Trial Block (N=200) by PSE Group ===\n")

# Extract final trial block data
stimulus_metrics_final <- data_clean %>%
  dplyr::filter(trial_block == 200) %>%
  
  mutate(
    pse_true_z = scale(pse_true)[,1],
    jnd_true_z = scale(jnd_true)[,1]
  ) %>%
  dplyr::select(
    model, group, pse_true, pse_true_z, jnd_true, jnd_true_z, subject_id,
    asymmetry_index, asymmetry_index_abs
  )



# ============================================================================== =
# STATISTICAL TESTS: ABS1-ONLY ANALYSIS ====
# ============================================================================== =

cat("\n=== Statistical Tests: ABS1-Only Analysis ===\n")

# Filter data for ABS1 only
data_abs1 <- data_clean %>% filter(model == "ABS1")
stimulus_metrics_final_abs1 <- stimulus_metrics_final %>% filter(model == "ABS1")

# ============================================================================== =
# 1. FINAL VALUE TEST (N=200): One-sample test against μ=0 (with covariates)
# ============================================================================== =

cat("\n--- 1. Final Value Test (Trial Block 200, N=200) ---\n")

# AI final values - test against 0 with covariates
cat("\nOne-sample test: AI ~ 0 (controlling for pse_true_z, jnd_true_z)\n")

anova_ai_final <- aovperm(asymmetry_index ~ pse_true_z + jnd_true_z,
                          data = stimulus_metrics_final_abs1,
                          np = 5000)

cat("ANOVA results:\n")
print(anova_ai_final)

# Extract effect sizes
effect_sizes_ai_final <- extract_eta_squared(anova_ai_final)
print_effect_sizes(effect_sizes_ai_final, "Effect Sizes for AI Final (η²)")

# Descriptive stats
ai_final <- stimulus_metrics_final_abs1$asymmetry_index
cat("\n  Mean AI:", round(mean(ai_final, na.rm = TRUE), 4), "\n")
cat("  SD AI:", round(sd(ai_final, na.rm = TRUE), 4), "\n")
cohens_d_ai <- mean(ai_final, na.rm = TRUE) / sd(ai_final, na.rm = TRUE)
cat("  Cohen's d (vs 0):", round(cohens_d_ai, 4), "\n")

# |AI| final values - test against 0 with covariates
cat("\nOne-sample test: |AI| ~ 0 (controlling for pse_true_z, jnd_true_z)\n")

anova_aiabs_final <- aovperm(asymmetry_index_abs ~ pse_true_z + jnd_true_z,
                             data = stimulus_metrics_final_abs1,
                             np = 5000)

cat("ANOVA results:\n")
print(anova_aiabs_final)

# Extract effect sizes
effect_sizes_aiabs_final <- extract_eta_squared(anova_aiabs_final)
print_effect_sizes(effect_sizes_aiabs_final, "Effect Sizes for |AI| Final (η²)")

# Descriptive stats
aiabs_final <- stimulus_metrics_final_abs1$asymmetry_index_abs
cat("\n  Mean |AI|:", round(mean(aiabs_final, na.rm = TRUE), 4), "\n")
cat("  SD |AI|:", round(sd(aiabs_final, na.rm = TRUE), 4), "\n")
cohens_d_aiabs <- mean(aiabs_final, na.rm = TRUE) / sd(aiabs_final, na.rm = TRUE)
cat("  Cohen's d (vs 0):", round(cohens_d_aiabs, 4), "\n")

# Store results
final_value_results <- tibble(
  measure = c("AI", "|AI|"),
  n = c(length(ai_final), length(aiabs_final)),
  mean = c(mean(ai_final, na.rm = TRUE), mean(aiabs_final, na.rm = TRUE)),
  sd = c(sd(ai_final, na.rm = TRUE), sd(aiabs_final, na.rm = TRUE)),
  cohens_d = c(cohens_d_ai, cohens_d_aiabs)
)

# ============================================================================== =
# 2. EVOLUTION TEST: ANOVA with hierarchical structure (trial_block evolution)
# ============================================================================== =

cat("\n--- 2. Evolution Test: ANOVA AI ~ trial_block_f (controlling for subject variability) ---\n")

# Define cache file paths
cache_ai_evo <- file.path(results_filepath, "models", "model_ai_evo_abs1.rds")
cache_aiabs_evo <- file.path(results_filepath, "models", "model_aiabs_evo_abs1.rds")

# AI evolution with hierarchical structure
if (file.exists(cache_ai_evo)) {
  cat("Loading cached AI evolution ANOVA results...\n")
  anova_ai_evo <- readRDS(cache_ai_evo)
} else {
  cat("Computing AI evolution ANOVA (this may take a few minutes)...\n")
  anova_ai_evo <- aovperm(asymmetry_index ~ trial_block_f + pse_true_z + jnd_true_z + Error(subject_id/(trial_block_f)),
                          data = data_abs1,
                          np = 5000,
                          method = "Rd_kheradPajouh_renaud")
  saveRDS(anova_ai_evo, cache_ai_evo)
  cat("Cached AI evolution ANOVA results\n")
}

cat("\nANOVA for AI evolution:\n")
print(anova_ai_evo)

# Extract effect sizes
effect_sizes_ai_evo <- extract_eta_squared(anova_ai_evo)
print_effect_sizes(effect_sizes_ai_evo, "Effect Sizes for AI Evolution (η²)")

# |AI| evolution with hierarchical structure
if (file.exists(cache_aiabs_evo)) {
  cat("Loading cached |AI| evolution ANOVA results...\n")
  anova_aiabs_evo <- readRDS(cache_aiabs_evo)
} else {
  cat("Computing |AI| evolution ANOVA (this may take a few minutes)...\n")
  anova_aiabs_evo <- aovperm(asymmetry_index_abs ~ trial_block_f + pse_true_z + jnd_true_z + Error(subject_id/(trial_block_f)),
                             data = data_abs1,
                             np = 5000,
                             method = "Rd_kheradPajouh_renaud")
  saveRDS(anova_aiabs_evo, cache_aiabs_evo)
  cat("Cached |AI| evolution ANOVA results\n")
}

cat("\nANOVA for |AI| evolution:\n")
print(anova_aiabs_evo)

# Extract effect sizes
effect_sizes_aiabs_evo <- extract_eta_squared(anova_aiabs_evo)
print_effect_sizes(effect_sizes_aiabs_evo, "Effect Sizes for |AI| Evolution (η²)")

# Store results (simplified for ANOVA)
evolution_results <- tibble(
  measure = c("AI", "|AI|"),
  analysis = c("ANOVA trial_block_f", "ANOVA trial_block_f"),
  note = c("See ANOVA output above for details", "See ANOVA output above for details")
)


# ============================================================================== =
# SAVE RESULTS
# ============================================================================== =

cat("\n=== Saving Results ===\n")

# Save descriptive statistics
write_csv(ai_trajectory, file.path(results_filepath, "tables", "asymmetry_index_ai_trajectory.csv"))
write_csv(aiabs_trajectory, file.path(results_filepath, "tables", "asymmetry_index_aiabs_trajectory.csv"))

# Save ABS1-only analysis results
write_csv(final_value_results, file.path(results_filepath, "tables", "asymmetry_abs1_final_value_tests.csv"))
write_csv(evolution_results, file.path(results_filepath, "tables", "asymmetry_abs1_evolution_tests.csv"))

cat("✓ Saved: ABS1-only asymmetry index analysis tables\n")

# Save statistical models
saveRDS(list(
  final_value_ai = anova_ai_final,
  final_value_aiabs = anova_aiabs_final,
  evolution_ai = anova_ai_evo,
  evolution_aiabs = anova_aiabs_evo
), file.path(results_filepath, "models", "anova_asymmetry_abs1_analysis.rds"))

cat("✓ Saved: statistical models\n")

saveRDS(data_abs1, file.path(results_filepath, "models", "asymmetry_abs1_data.rds"))
cat("✓ Saved: ABS1 clean data for plotting\n")

cat("\n================================================================================\n")
cat("ASYMMETRY INDEX EVOLUTION ANALYSIS COMPLETE\n")
cat("================================================================================\n")
cat("\nResults saved to:", results_filepath, "\n\n")
