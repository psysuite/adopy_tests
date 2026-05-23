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
# Define cache file path
results_dir <- "/data/Dropbox/RDATA/R_bis_ad_fx/results_simulations"
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(results_dir, "tables"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(results_dir, "models"), showWarnings = FALSE, recursive = TRUE)


cat("\n=== FINAL TRIAL BLOCK ANALYSIS (N=200) ===\n\n")

data_final <- df %>% filter(trial_block == 200)

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
cache_final_aov <- file.path(results_dir, "models", "lat_entropy_final_aov.rds")

if (file.exists(cache_final_aov)) {
  cat("Loading cached final ANOVA results...\n")
  anova_final <- readRDS(cache_final_aov)
} else {
  cat("Computing final ANOVA...\n")
  anova_final <- aovperm(lat_entropy ~ model , data = data_final, np = 5000)
  saveRDS(anova_final, cache_final_aov)
  cat("Cached final ANOVA results\n")
}

print(anova_final)


# Calculate effect sizes
effect_sizes_entropy_final <- extract_eta_squared(anova_final)
print_effect_sizes(effect_sizes_entropy_final, "Effect Sizes for Latency Entropy at N=200 (η²)")

cat("\n\nPost-hoc pairwise comparisons at N=200 (Tukey):\n")
mod_final <- lm(lat_entropy ~ model, data = data_final)
emm_final <- emmeans(mod_final, ~ model)
pairs_final <- pairs(emm_final, adjust = "tukey")
print(pairs_final)


# ANOVA with permutation test: lat_entropy ~ model + trial_block + model:trial_block + Error(subject_id/(trial_block))
df$trial_block_f <- factor(df$trial_block)

cat("ANOVA Results:\n")
cat("Formula: lat_entropy ~ model * trial_block + Error(subject_id/(trial_block))\n\n")



cache_model_aov <- file.path(results_dir, "models", "model_lat_entropy_aov.rds")

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
cache_posthoc <- file.path(results_dir, "models", "lat_entropy_posthoc_by_block.rds")

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
    
    # Fit model for this trial block
    mod_tb <- lm(lat_entropy ~ model, data = data_tb)
    
    # Get pairwise comparisons
    emm_tb <- emmeans(mod_tb, ~ model)
    pairs_tb <- pairs(emm_tb, adjust = "tukey")
    
    posthoc_results[[as.character(tb)]] <- pairs_tb
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
  
  mod_tb <- lm(lat_entropy ~ model, data = data_tb)
  emm_tb <- emmeans(mod_tb, ~ model)
  pairs_tb <- pairs(emm_tb, adjust = "tukey")
  
  # Extract p-values for REL2 comparisons
  pairs_df <- as.data.frame(pairs_tb)
  
  rel2_vs_abs1 <- pairs_df %>% filter(contrast == "ABS1 - REL2")
  rel2_vs_rel1 <- pairs_df %>% filter(contrast == "REL1 - REL2")
  
  if (nrow(rel2_vs_abs1) > 0 && nrow(rel2_vs_rel1) > 0) {
    results_by_block <- rbind(results_by_block, data.frame(
      trial_block = tb,
      rel2_vs_abs1_p = rel2_vs_abs1$p.value[1],
      rel2_vs_rel1_p = rel2_vs_rel1$p.value[1],
      rel2_vs_abs1_sig = rel2_vs_abs1$p.value[1] < 0.05,
      rel2_vs_rel1_sig = rel2_vs_rel1$p.value[1] < 0.05
    ))
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
          file.path(results_dir, "tables", "lat_entropy_by_model.csv"), 
          row.names = FALSE)

write.csv(desc_by_model_block, 
          file.path(results_dir, "tables", "lat_entropy_by_model_block.csv"), 
          row.names = FALSE)

# Save plot data
write.csv(plot_data, 
          file.path(results_dir, "tables", "lat_entropy_plot_data.csv"), 
          row.names = FALSE)

# Save final analysis results
write.csv(desc_final,
          file.path(results_dir, "tables", "lat_entropy_final_desc.csv"),
          row.names = FALSE)

write.csv(results_by_block,
          file.path(results_dir, "tables", "lat_entropy_rel2_significance_by_block.csv"),
          row.names = FALSE)

# Combine all effect sizes
all_effect_sizes_entropy <- bind_rows(
  effect_sizes_entropy_main %>% mutate(analysis = "Latency Entropy Main"),
  effect_sizes_entropy_final %>% mutate(analysis = "Latency Entropy Final (N=200)")
)

# Save effect sizes
write.csv(all_effect_sizes_entropy,
          file.path(results_dir, "tables", "effect_sizes_entropy_eta_squared.csv"),
          row.names = FALSE)

cat(sprintf("Results saved to %s\n", results_dir))
cat("\n")

# ============================================================================
# RETURN DATA FOR PLOTTING
# ============================================================================

# Make plot_data available for plotting script
saveRDS(plot_data, file.path(results_dir, "models", "lat_entropy_plot_data.rds"))
saveRDS(model_aov, file.path(results_dir, "models", "model_lat_entropy_aov.rds"))
saveRDS(desc_final, file.path(results_dir, "models", "lat_entropy_final_desc.rds"))
saveRDS(results_by_block, file.path(results_dir, "models", "lat_entropy_rel2_significance_by_block.rds"))
