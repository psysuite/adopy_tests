#!/usr/bin/env Rscript
# Latency Entropy Analysis - Progressive evolution across trial blocks
# Analyzes lat_entropy across models (ABS1, REL1, REL2) independent of PSE/JND

library(tidyverse)
library(permuco)

# Load effect size utilities
source("effect_size_utils.R")


df <- data_raw

# Ensure proper data types
df$model <- factor(df$model, levels = c("ABS1", "REL1", "REL2"))
df$trial_block <- as.integer(df$trial_block)
df$subject_id <- as.factor(df$subject_id)
df$group <- as.factor(df$group)  # Preserve group column

# Remove rows with missing lat_entropy
df <- df %>% filter(!is.na(lat_entropy))

cat("Data loaded:\n")
cat(sprintf("  Total rows: %d\n", nrow(df)))
cat(sprintf("  Models: %s\n", paste(levels(df$model), collapse = ", ")))
cat(sprintf("  Trial blocks: %s\n", paste(sort(unique(df$trial_block)), collapse = ", ")))
cat(sprintf("  Subjects per model: %d\n", length(unique(df$subject_id)) / 3))
cat("\n")

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

cat("=== DESCRIPTIVE STATISTICS ===\n")

# Overall statistics by model
desc_by_model <- df %>%
  group_by(model) %>%
  summarise(
    n = n(),
    mean = mean(lat_entropy, na.rm = TRUE),
    sd = sd(lat_entropy, na.rm = TRUE),
    min = min(lat_entropy, na.rm = TRUE),
    max = max(lat_entropy, na.rm = TRUE),
    .groups = 'drop'
  )

cat("Latency Entropy by Model (all trial blocks combined):\n")
print(desc_by_model)
cat("\n")

# Statistics by model and trial block
desc_by_model_block <- df %>%
  group_by(model, trial_block) %>%
  summarise(
    n = n(),
    mean = mean(lat_entropy, na.rm = TRUE),
    sd = sd(lat_entropy, na.rm = TRUE),
    .groups = 'drop'
  )

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

cat("=== STATISTICAL ANALYSIS ===\n\n")


# ============================================================================
# FINAL TRIAL BLOCK ANALYSIS (N=200)
# ============================================================================
# Define results directory (use global if available, otherwise set it)
if(!exists('results_filepath')){
  if(!exists('root_dir')){
    root_dir = "/data/CODE/python/adopy_tests/"
  }
  if(!exists('project_name')){
    project_name = "R"
  }
  results_filepath <- paste0(root_dir, project_name, "/results_simulations")
}

dir.create(results_filepath, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(results_filepath, "tables"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(results_filepath, "models"), showWarnings = FALSE, recursive = TRUE)


cat("\n=== FINAL TRIAL BLOCK ANALYSIS (N=200) ===\n\n")

data_final <- df %>% filter(trial_block == 200) %>%
  dplyr::mutate(
    pse_true_z = scale(pse_true)[,1],
    jnd_true_z = scale(jnd_true)[,1]
  )

cat("Descriptive statistics at N=200:\n")
desc_final <- data_final %>%
  group_by(model) %>%
  summarise(
    n = n(),
    mean = mean(lat_entropy, na.rm = TRUE),
    sd = sd(lat_entropy, na.rm = TRUE),
    se = sd / sqrt(n),
    .groups = "drop"
  )
print(desc_final)

cat("\n\nANOVA for model effect at N=200:\n")

# Define cache file path for final ANOVA
cache_final_aov <- file.path(results_filepath, "models", "lat_entropy_final_aov.rds")

if (file.exists(cache_final_aov)) {
  cat("Loading cached final ANOVA results...\n")
  anova_final <- readRDS(cache_final_aov)
} else {
  cat("Computing final ANOVA...\n")
  anova_final <- aovperm(lat_entropy ~ model + pse_true_z + jnd_true_z , data = data_final, np = 5000)
  saveRDS(anova_final, cache_final_aov)
  cat("Cached final ANOVA results\n")
}

print(anova_final)

# Resampling test using freedman_lane to handle nuisance variables and 5000 permutations.
#                 SS    df     F         parametric P(>F) resampled P(>F)
# model      8.427e+00   2 7.849e+01           0.0000          0.0002
# pse_true_z 2.391e-09   1 4.454e-08           0.9998          0.9998
# jnd_true_z 2.884e+01   1 5.373e+02           0.0000          0.0002

# Calculate effect sizes
effect_sizes_entropy_final <- extract_eta_squared(anova_final)
print_effect_sizes(effect_sizes_entropy_final, "Effect Sizes for Latency Entropy at N=200 (η²)")

cat("\n\nPost-hoc pairwise comparisons at N=200 (Friedman with multiple comparison adjustment):\n")
do_npar_anova_main(data_final, "lat_entropy", "model")
# [1] "Main effect: lat_entropy ~ model, H = 68.7847, p = 0"
# [1] "SIGNIFICANT - Running pairwise comparisons..."
# Comparison   Stat   p.value  p.adjust
# 1 ABS1 - REL1 = 0 -2.146   0.03189 3.189e-02
# 2 ABS1 - REL2 = 0  6.568 5.085e-11 7.628e-11
# 3 REL1 - REL2 = 0  7.217 5.311e-13 1.593e-12

# ANOVA with permutation test: lat_entropy ~ model + trial_block + model:trial_block + Error(subject_id/(trial_block))
df$trial_block_f <- factor(df$trial_block)

cat("ANOVA Results:\n")
cat("Formula: lat_entropy ~ model * trial_block + Error(subject_id/(trial_block))\n\n")



cache_model_aov <- file.path(results_filepath, "models", "model_lat_entropy_aov.rds")

# Check if cached results exist BEFORE computing
if (file.exists(cache_model_aov)) {
  cat("Loading cached latency entropy ANOVA results...\n")
  model_aov <- readRDS(cache_model_aov)
} else {
  cat("Computing latency entropy ANOVA (this may take a few minutes)...\n")
  model_aov <- aovperm(
    lat_entropy ~ model * trial_block_f + Error(subject_id/(trial_block_f)),
    data = df,
    np = 5000,
    method = "Rd_kheradPajouh_renaud"
  )
  saveRDS(model_aov, cache_model_aov)
  cat("Cached latency entropy ANOVA results\n")
}

print(model_aov)
cat("\n")

# Calculate effect sizes
effect_sizes_entropy_main <- extract_eta_squared(model_aov)
print_effect_sizes(effect_sizes_entropy_main, "Effect Sizes for Latency Entropy (η²)")

# ============================================================================
# POST-HOC ANALYSIS: Pairwise comparisons by trial block
# ============================================================================

cat("=== POST-HOC ANALYSIS: Pairwise Comparisons by Trial Block ===\n\n")

# Define cache file path for posthoc results
cache_posthoc <- file.path(results_filepath, "models", "lat_entropy_posthoc_by_block.rds")

# Check if cached posthoc results exist
if (file.exists(cache_posthoc)) {
  cat("Loading cached posthoc results by trial block...\n")
  posthoc_results <- readRDS(cache_posthoc)
} else {
  cat("Computing posthoc comparisons by trial block (this may take a few minutes)...\n")
  
  posthoc_results <- list()
  
  # For each trial block, test which models differ
  for (tb in sort(unique(df$trial_block))) {
    data_tb <- df %>% filter(trial_block == tb)
    
    # Get pairwise comparisons using non-parametric test
    posthoc_results[[as.character(tb)]] <- do_npar_anova_main(data_tb, "lat_entropy", "model")
  }
  
  saveRDS(posthoc_results, cache_posthoc)
  cat("Cached posthoc results by trial block\n")
}

# Print posthoc results
for (tb in sort(unique(df$trial_block))) {
  cat(sprintf("\n--- Trial Block %d ---\n", tb))
  print(posthoc_results[[as.character(tb)]])
}



# Identify when REL2 becomes significantly lower than other models
cat("\n\n=== TRIAL BLOCK WHERE REL2 BECOMES SIGNIFICANTLY LOWER ===\n\n")

results_by_block <- data.frame()

for (tb in sort(unique(df$trial_block))) {
  data_tb <- df %>% filter(trial_block == tb)
  
  # Use non-parametric Kruskal-Wallis test for pairwise comparisons
  kw_test <- kruskal.test(lat_entropy ~ model, data = data_tb)
  
  if (kw_test$p.value < 0.05) {
    # Perform pairwise Mann-Whitney U tests with Bonferroni correction
    models <- unique(data_tb$model)
    n_comparisons <- choose(length(models), 2)
    bonferroni_alpha <- 0.05 / n_comparisons
    
    comparisons <- combn(models, 2, simplify = FALSE)
    
    for (comp in comparisons) {
      data_comp <- data_tb %>% filter(model %in% comp)
      mw_test <- wilcox.test(lat_entropy ~ model, data = data_comp)
      
      if (comp[1] == "ABS1" && comp[2] == "REL2") {
        results_by_block <- rbind(results_by_block, data.frame(
          trial_block = tb,
          rel2_vs_abs1_p = mw_test$p.value,
          rel2_vs_abs1_sig = mw_test$p.value < bonferroni_alpha,
          rel2_vs_rel1_p = NA,
          rel2_vs_rel1_sig = NA
        ))
      } else if (comp[1] == "REL1" && comp[2] == "REL2") {
        if (nrow(results_by_block) > 0 && results_by_block$trial_block[nrow(results_by_block)] == tb) {
          results_by_block$rel2_vs_rel1_p[nrow(results_by_block)] <- mw_test$p.value
          results_by_block$rel2_vs_rel1_sig[nrow(results_by_block)] <- mw_test$p.value < bonferroni_alpha
        }
      }
    }
  }
}

cat("P-values for REL2 vs other models by trial block:\n")
print(results_by_block)

# Find first trial block where REL2 is significantly lower than BOTH others
first_sig_block <- results_by_block %>%
  filter(rel2_vs_abs1_sig & rel2_vs_rel1_sig) %>%
  slice(1)

if (nrow(first_sig_block) > 0) {
  cat(sprintf("\nREL2 becomes significantly lower than both ABS1 and REL1 at trial block: %d\n", 
              first_sig_block$trial_block[1]))
} else {
  cat("\nREL2 does not become significantly lower than both other models at any trial block\n")
}

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

cat("=== CORRELATION ANALYSIS ===\n\n")

# Correlation between lat_entropy and trial_block by model
for (m in levels(df$model)) {
  model_data <- df %>% filter(model == m)
  corr <- cor.test(model_data$trial_block, model_data$lat_entropy, method = "pearson")
  cat(sprintf("%s: r = %.3f, p = %.4f\n", m, corr$estimate, corr$p.value))
}
cat("\n")

# ============================================================================
# PREPARE DATA FOR PLOTTING
# ============================================================================

# Calculate mean and SE by model and trial block
plot_data <- df %>%
  group_by(model, trial_block) %>%
  summarise(
    mean_entropy = mean(lat_entropy, na.rm = TRUE),
    se_entropy = sd(lat_entropy, na.rm = TRUE) / sqrt(n()),
    n = n(),
    .groups = 'drop'
  )

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save descriptive statistics
write.csv(desc_by_model, 
          file.path(results_filepath, "tables", "lat_entropy_by_model.csv"), 
          row.names = FALSE)

write.csv(desc_by_model_block, 
          file.path(results_filepath, "tables", "lat_entropy_by_model_block.csv"), 
          row.names = FALSE)

# Save plot data
write.csv(plot_data, 
          file.path(results_filepath, "tables", "lat_entropy_plot_data.csv"), 
          row.names = FALSE)

# Save final analysis results
write.csv(desc_final,
          file.path(results_filepath, "tables", "lat_entropy_final_desc.csv"),
          row.names = FALSE)

write.csv(results_by_block,
          file.path(results_filepath, "tables", "lat_entropy_rel2_significance_by_block.csv"),
          row.names = FALSE)

# Combine all effect sizes
all_effect_sizes_entropy <- bind_rows(
  effect_sizes_entropy_main %>% mutate(analysis = "Latency Entropy Main"),
  effect_sizes_entropy_final %>% mutate(analysis = "Latency Entropy Final (N=200)")
)

# Save effect sizes
write.csv(all_effect_sizes_entropy,
          file.path(results_filepath, "tables", "effect_sizes_entropy_eta_squared.csv"),
          row.names = FALSE)

cat(sprintf("Results saved to %s\n", results_filepath))
cat("\n")

# ============================================================================
# RETURN DATA FOR PLOTTING
# ============================================================================

# Make plot_data available for plotting script
saveRDS(plot_data, file.path(results_filepath, "models", "lat_entropy_plot_data.rds"))
saveRDS(model_aov, file.path(results_filepath, "models", "model_lat_entropy_aov.rds"))
saveRDS(desc_final, file.path(results_filepath, "models", "lat_entropy_final_desc.rds"))
saveRDS(results_by_block, file.path(results_filepath, "models", "lat_entropy_rel2_significance_by_block.rds"))
