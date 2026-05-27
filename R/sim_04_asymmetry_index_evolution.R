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

# Load effect size utilities
source("effect_size_utils.R")
source("npar_posthoc.R")


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
# Resampling test using freedman_lane to handle nuisance variables and 5000 permutations.
# SS  df       F parametric P(>F) resampled P(>F)
# pse_true_z 5.772335   1 180.209           0.0000          0.0002
# jnd_true_z 0.008135   1   0.254           0.6149          0.6206

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

anova_aiabs_final <- aovperm(asymmetry_index_abs ~ pse_true_z * jnd_true_z,
                             data = stimulus_metrics_final_abs1,
                             np = 5000)

cat("ANOVA results:\n")
print(anova_aiabs_final)
# Resampling test using freedman_lane to handle nuisance variables and 5000 permutations.
# SS  df       F parametric P(>F) resampled P(>F)
# pse_true_z            0.006592   1  0.2545           0.6146          0.6222
# jnd_true_z            2.324772   1 89.7576           0.0000          0.0002
# pse_true_z:jnd_true_z 0.013620   1  0.5259           0.4693          0.4628

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
# SSn dfn   SSd  dfd     MSEn   MSEd        F parametric P(>F) resampled P(>F)
# pse_true_z    79.0569   1 57.99  177 79.05687 0.3276 241.3191           0.0000          0.0002
# jnd_true_z     0.0495   1 57.99  177  0.04950 0.3276   0.1511           0.6979          0.7048
# trial_block_f  0.2021   8 17.61 1432  0.02526 0.0123   2.0535           0.0374          0.0338

# Extract effect sizes
effect_sizes_ai_evo <- extract_eta_squared(anova_ai_evo)
print_effect_sizes(effect_sizes_ai_evo, "Effect Sizes for AI Evolution (η²)")

# Post-hoc pairwise comparisons for AI evolution
cat("\nPost-hoc pairwise comparisons for AI evolution (Wilcoxon signed-rank tests with FDR correction):\n")
npar_ph_pairwise_within(data_abs1, "asymmetry_index", "trial_block_f", "subject_id", corr="fdr")

# Comparison p.value p.adjust Significant
# <chr>        <dbl>    <dbl> <lgl>      
# 1 40 vs 60    0.338     0.518 FALSE      
# 2 40 vs 80    0.174     0.481 FALSE      
# 3 40 vs 100   0.162     0.481 FALSE      
# 4 40 vs 120   0.216     0.481 FALSE      
# 5 40 vs 140   0.193     0.481 FALSE      
# 6 40 vs 160   0.145     0.481 FALSE      
# 7 40 vs 180   0.124     0.481 FALSE      
# 8 40 vs 200   0.124     0.481 FALSE      
# 9 60 vs 80    0.0672    0.481 FALSE      
# 10 60 vs 100   0.167     0.481 FALSE      
# 11 60 vs 120   0.281     0.517 FALSE      
# 12 60 vs 140   0.168     0.481 FALSE      
# 13 60 vs 160   0.141     0.481 FALSE      
# 14 60 vs 180   0.123     0.481 FALSE      
# 15 60 vs 200   0.102     0.481 FALSE      
# 16 80 vs 100   0.796     0.821 FALSE      
# 17 80 vs 120   0.788     0.821 FALSE      
# 18 80 vs 140   0.679     0.782 FALSE      
# 19 80 vs 160   0.587     0.705 FALSE      
# 20 80 vs 180   0.490     0.630 FALSE      
# 21 80 vs 200   0.451     0.601 FALSE      
# 22 100 vs 120  0.295     0.517 FALSE      
# 23 100 vs 140  0.821     0.821 FALSE      
# 24 100 vs 160  0.695     0.782 FALSE      
# 25 100 vs 180  0.583     0.705 FALSE      
# 26 100 vs 200  0.427     0.591 FALSE      
# 27 120 vs 140  0.173     0.481 FALSE      
# 28 120 vs 160  0.346     0.518 FALSE      
# 29 120 vs 180  0.271     0.517 FALSE      
# 30 120 vs 200  0.220     0.481 FALSE      
# 31 140 vs 160  0.422     0.591 FALSE      
# 32 140 vs 180  0.309     0.517 FALSE      
# 33 140 vs 200  0.316     0.517 FALSE      
# 34 160 vs 180  0.222     0.481 FALSE      
# 35 160 vs 200  0.227     0.481 FALSE      
# 36 180 vs 200  0.801     0.821 FALSE   


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
# Resampling test using Rd_kheradPajouh_renaud to handle nuisance variables and 5000 permutations.
# SSn dfn   SSd  dfd    MSEn     MSEd       F parametric P(>F) resampled P(>F)
# pse_true_z     0.1692   1 49.28  177  0.1692 0.278431  0.6075        4.368e-01          0.4304
# jnd_true_z    16.1436   1 49.28  177 16.1436 0.278431 57.9805        1.525e-12          0.0002
# trial_block_f  5.6744   8 10.84 1432  0.7093 0.007567 93.7338        0.000e+00          0.0002

# Extract effect sizes
effect_sizes_aiabs_evo <- extract_eta_squared(anova_aiabs_evo)
print_effect_sizes(effect_sizes_aiabs_evo, "Effect Sizes for |AI| Evolution (η²)")

# Post-hoc pairwise comparisons for |AI| evolution
cat("\nPost-hoc pairwise comparisons for |AI| evolution (Wilcoxon signed-rank tests with FDR correction):\n")
npar_ph_pairwise_within(data_abs1, "asymmetry_index_abs", "trial_block_f", "subject_id", corr="fdr")



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

# Save data for plotting in paper figures (format expected by 00_create_paper_figures.R)
asymmetry_evolution_data <- data_clean %>%
  dplyr::select(model, trial_block, asymmetry_index, asymmetry_index_abs) %>%
  filter(model == "ABS1")

saveRDS(asymmetry_evolution_data, file.path(results_filepath, "models", "asymmetry_evolution_data.rds"))
cat("✓ Saved: asymmetry_evolution_data.rds for paper figures\n")

cat("\n================================================================================\n")
cat("ASYMMETRY INDEX EVOLUTION ANALYSIS COMPLETE\n")
cat("================================================================================\n")
cat("\nResults saved to:", results_filepath, "\n\n")
