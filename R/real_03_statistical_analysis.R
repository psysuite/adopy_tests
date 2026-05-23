# ============================================================================== =
# 03_statistical_analysis.R
# Mixed-effects models and statistical tests
# ============================================================================== =

library(tidyverse)
library(dplyr)
library(here)
library(lme4)
library(lmerTest)
library(robustlmm)
library(permuco)

library(emmeans)
library(car)

# Define color palette if not already defined
if(!exists('global_fill_colors')) {
  global_fill_colors <- c("Adaptive" = "#E8C3A4", "Fixed" = "#A5C1B6", "SZ" = "#C4A4D4")
}

#======================================================================================================= = 
# LOAD CUSTOM SCRIPT = 
#======================================================================================================= = 
# setwd(root_dir)
source("npar_posthoc.R")
source("effect_size_utils.R")
# setwd(project_dir)

# Load clean data
data_clean <- readRDS(here(results_filepath, "data_clean.rds"))


cat("=== Statistical Analysis ===\n")


cat("\n=== Latency Statistics ===\n")
# Test entropy difference between modes


model_entropy_perm <- aovperm(
  lat_entropy ~ modality * algorithm + age_z + gender + Error(subj/(algorithm)),
  data = filter(data_clean, n_trials == 200),
  np = 5000
)
res <- summary(model_entropy_perm)
print(res)

# Calculate effect sizes using helper function
effect_sizes_entropy <- extract_eta_squared(model_entropy_perm)
print_effect_sizes(effect_sizes_entropy, "Effect Sizes for Latency Entropy (η²)")

cat("\nEntropy comparison (AD vs FX):\n")
cat(sprintf("F = %.2f, p = %.4f\n",
            res$F[4],  res$`resampled P(>F)`[4]))

# ============================================================================== =
# Stimulus Spread Analysis ====
# ============================================================================== =

cat("\n=== Stimulus Spread (SS) ===\n")
cat("Testing if Adaptive mode reduces within-subject latency variability compared to Fixed mode\n")
cat("Rationale: ADA mode presents stimuli closer to PSE (narrower range),\n")
cat("while FIX mode spans a larger predetermined interval, leading to\n")
cat("more variable response latencies in FIX condition.\n")

# Calculate descriptive statistics for within-subject latency variability
latency_desc <- data_clean %>%
  filter(n_trials == 200) %>%
  group_by(modality, algorithm) %>%
  summarise(
    n = n(),
    ss_avg = mean(ss, na.rm = TRUE),
    ss_sd = sd(ss, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    ss_formatted = sprintf("%.0f ± %.0f", ss_avg, ss_sd)
  )

cat("\nWithin-subject latency variability (Mean ± SD of individual SDs):\n")
print(dplyr::select(latency_desc, modality, algorithm, ss_formatted, n))

# Statistical analysis using aovperm (consistent with other analyses)
cat("\n=== Statistical Test: Latency Variability ===\n")

model_latstd_perm <- aovperm(
  ss ~ modality * algorithm + age_z + gender + Error(subj/algorithm),
  data = filter(data_clean, n_trials == 200),
  np = 5000
)
saveRDS(model_latstd_perm, file.path(results_filepath, "models", "model_latstd_perm.rds"))

res_latstd <- summary(model_latstd_perm)
print(res_latstd)

# Calculate effect sizes using helper function
effect_sizes_latstd <- extract_eta_squared(model_latstd_perm)
print_effect_sizes(effect_sizes_latstd, "Effect Sizes for Latency Variability (η²)")

cat("\nKey results for latency variability:\n")
cat(sprintf("Task effect: F = %.2f, p = %.4f\n", 
            res_latstd$F[1], res_latstd$`resampled P(>F)`[1]))
cat(sprintf("Mode effect: F = %.2f, p = %.4f %s\n", 
            res_latstd$F[4], res_latstd$`resampled P(>F)`[4],
            ifelse(res_latstd$`resampled P(>F)`[4] < 0.05, "***", "")))
cat(sprintf("Task × Mode interaction: F = %.2f, p = %.4f %s\n", 
            res_latstd$F[5], res_latstd$`resampled P(>F)`[5],
            ifelse(res_latstd$`resampled P(>F)`[5] < 0.05, "***", "")))

# ============================================================================== =
# 2: Task × Mode @ (n_trials = 200)  ====
# ============================================================================== =
cat("\n\n=== PRIMARY ANALYSIS: Task × Mode @ (n_trials == 200) ===\n")
cat("PSE")
model_finalpse_perm <- aovperm(
  pse ~ modality * algorithm + age_z + gender + Error(subj/algorithm),
  data = dplyr::filter(data_clean, n_trials == 200),
  np = 5000
)
saveRDS(model_finalpse_perm, file.path(results_filepath, "models", "model_finalpse_perm.rds"))

res_finalpse <- summary(model_finalpse_perm)
print(res_finalpse)

# Resampling test using Rd_kheradPajouh_renaud to handle nuisance variables and 5000 permutations.
# SSn dfn  SSd dfd    MSEn  MSEd        F parametric P(>F) resampled P(>F)
# modality             21.95   1 6667  16   21.95 416.7  0.05268        0.8213717          0.8282
# age_z               275.87   1 6667  16  275.87 416.7  0.66208        0.4277709          0.4238
# gender               33.00   1 6667  16   33.00 416.7  0.07921        0.7819831          0.7842
# algorithm          1242.19   1 1186  18 1242.19  65.9 18.84908        0.0003932          0.0004
# modality:algorithm  475.36   1 1186  18  475.36  65.9  7.21320        0.0151019          0.0164

# Calculate effect sizes using helper function
effect_sizes_finalpse <- extract_eta_squared(model_finalpse_perm)
print_effect_sizes(effect_sizes_finalpse, "Effect Sizes for PSE Final (η²)")

# Post-hoc tests for PSE
cat("\n=== Post-hoc Tests: PSE Final ===\n")

# main algorithm effect 
# cat("\nAlgorithm effect (Adaptive vs Fixed):\n")
# res_pse_algo <- do_npar_anova_main_repeated(
#   data_clean %>% dplyr::filter(n_trials == 200),
#   "pse", "algorithm"
# )

# Posthoc for modality × algorithm interaction
cat("\nModality × Algorithm interaction:\n")
res_pse_by_modality <- do_npar_anova_phpw(
  data_clean %>% dplyr::filter(n_trials == 200),
  "algorithm", "pse", "modality"
)
# [1] "pse x modality splitted by algorithm"
# [1] "NOT SIGNIFICANT in Adaptive (H=1.65142857142858, p=0.397529212746468)"
# [1] "NOT SIGNIFICANT in Fixed (H=0.36571428571429, p=0.545349668011121)"

res_pse_by_algorithm <- do_npar_anova_repeated_phpw(
  data_clean %>% dplyr::filter(n_trials == 200),
  "modality", "pse", "algorithm"
)

# [1] "pse x algorithm (within) splitted by modality (between)"
# [1] "in Auditory (chi-sq=10, p=0.0031)"
# [1] "NOT SIGNIFICANT in Visual (chi-sq=1.6, p=0.2059)"

if(FALSE){
# algorithm_effect <- data_clean %>%
#   filter(n_trials == 200) %>%
#   select(subj, modality, algorithm, pse) %>%
#   pivot_wider(names_from = algorithm, values_from = pse) %>%
#   mutate(diff = Adaptive - Fixed) %>%
#   group_by(modality) %>%
#   summarise(
#     mean_diff = mean(diff, na.rm = TRUE),
#     sd_diff = sd(diff, na.rm = TRUE),
#     n_positive = sum(diff > 0, na.rm = TRUE),
#     n_total = n(),
#     .groups = "drop"
#   )
# print(algorithm_effect)

# Also check individual differences
# cat("\n=== Individual differences (Adaptive - Fixed) ===\n")
# individual_diff <- data_clean %>%
#   filter(n_trials == 200) %>%
#   select(subj, modality, algorithm, pse) %>%
#   pivot_wider(names_from = algorithm, values_from = pse) %>%
#   mutate(diff = Adaptive - Fixed)
# print(individual_diff)
}


cat("\nJND")

model_finaljnd_perm <- aovperm(
  jnd ~ modality * algorithm + age_z + gender + Error(subj/algorithm),
  data = dplyr::filter(data_clean, n_trials == 200),
  np = 5000
)
saveRDS(model_finaljnd_perm, file.path(results_filepath, "models", "model_finaljnd_perm.rds"))

res_finaljnd <- summary(model_finaljnd_perm)
print(res_finaljnd)

# Resampling test using Rd_kheradPajouh_renaud to handle nuisance variables and 5000 permutations.
# SSn dfn  SSd dfd    MSEn  MSEd       F parametric P(>F) resampled P(>F)
# modality           1180.66   1 4136  16 1180.66 258.5 4.56720          0.04838          0.0396
# age_z               181.63   1 4136  16  181.63 258.5 0.70260          0.41426          0.3924
# gender               17.43   1 4136  16   17.43 258.5 0.06741          0.79846          0.8012
# algorithm           525.96   1 3116  18  525.96 173.1 3.03859          0.09836          0.0952
# modality:algorithm   17.39   1 3116  18   17.39 173.1 0.10049          0.75489          0.7562

# Calculate effect sizes using helper function
effect_sizes_finaljnd <- extract_eta_squared(model_finaljnd_perm)
print_effect_sizes(effect_sizes_finaljnd, "Effect Sizes for JND Final (η²)")

# Post-hoc tests for JND
cat("\n=== Post-hoc Tests: JND Final ===\n")

# Posthoc for modality effect (Auditory vs Visual)
cat("\nModality effect (Auditory vs Visual):\n")
res_jnd_mod <- do_npar_anova_main_with_summary(
  data_clean %>% dplyr::filter(n_trials == 200),
  "jnd", "modality"
)

# [1] "\nSummary statistics:"
# modality jnd.mean   jnd.sd
# 1 Auditory 35.03219  8.22432
# 2   Visual 46.19584 18.79550

# ============================================================================== =
# 2: Task × Mode x n_trials  ====
# ============================================================================== =
# cat("\n\n=== CONVERGENCE ANALYSIS: Task × Mode x n_trials ===\n")
# cat("PSE")
# model_pse_perm <- aovperm(
#   pse ~ modality * algorithm * n_trials_f + age_z + gender + Error(subj/(algorithm * n_trials_f)),
#   data = data_clean,
#   np = 5000
# )
# saveRDS(model_pse_perm, file.path(results_filepath, "models", "model_pse_full_perm.rds"))
# 
# res_pse <- summary(model_pse_perm)
# print(res_pse)
# 
# # Resampling test using Rd_kheradPajouh_renaud to handle nuisance variables and 5000 permutations.
# # SSn dfn   SSd dfd     MSEn    MSEd       F parametric P(>F) resampled P(>F)
# # modality                       2204.8   1 81759  16  2204.81 5109.94 0.43148        5.206e-01          0.5272
# # age_z                          3755.3   1 81759  16  3755.30 5109.94 0.73490        4.040e-01          0.4004
# # gender                          167.7   1 81759  16   167.73 5109.94 0.03283        8.585e-01          0.8604
# # algorithm                      8421.9   1 25027  18  8421.88 1390.39 6.05721        2.418e-02          0.0268
# # modality:algorithm            12916.1   1 25027  18 12916.10 1390.39 9.28956        6.924e-03          0.0082
# # n_trials_f                     2761.2   8 11126 144   345.15   77.26 4.46730        7.353e-05          0.0004
# # modality:n_trials_f            1262.3   8 11126 144   157.78   77.26 2.04223        4.543e-02          0.0438
# # algorithm:n_trials_f            103.7   8  8087 144    12.96   56.16 0.23077        9.847e-01          0.9832
# # modality:algorithm:n_trials_f  1049.0   8  8087 144   131.12   56.16 2.33469        2.182e-02          0.0186
# 
# 
# # Calculate effect sizes using helper function
# effect_sizes_pse_conv <- extract_eta_squared(model_pse_perm)
# print_effect_sizes(effect_sizes_pse_conv, "Effect Sizes for PSE Convergence (η²)")
# 
# res <- do_npar_anova_phpw(data_clean %>% dplyr::filter(modality == "Auditory"), "n_trials_f", "pse", "algorithm")
# res <- do_npar_anova_phpw(data_clean %>% dplyr::filter(modality == "Visual"), "n_trials_f", "pse", "algorithm")
# 
# cat("JND")
# 
# model_jnd_perm <- aovperm(
#   jnd ~ modality * algorithm * n_trials_f + age_z + gender + Error(subj/(algorithm * n_trials_f)),
#   data = data_clean,
#   np = 5000
# )
# 
# res_jnd <- summary(model_jnd_perm)
# print(res_jnd)
# 
# # Calculate effect sizes using helper function
# effect_sizes_jnd_conv <- extract_eta_squared(model_jnd_perm)
# print_effect_sizes(effect_sizes_jnd_conv, "Effect Sizes for JND Convergence (η²)")
# 
# saveRDS(model_jnd_perm, file.path(results_filepath, "models", "model_jnd_full_perm.rds"))
# 
# aovperm_desc_plots_2f(data_clean %>% filter(modality == "Auditory"), c("jnd"), "algorithm", "n_trials_f", plot_raw = TRUE)
# aovperm_desc_plots_2f(data_clean %>% filter(modality == "Visual"), c("jnd"), "algorithm", "n_trials_f", plot_raw = TRUE)
# 
# res <- do_npar_anova_phpw(data_clean %>% filter(modality == "Auditory"), "n_trials_f", "jnd", "algorithm")
# res <- do_npar_anova_phpw(data_clean %>% filter(modality == "Visual"), "n_trials_f", "jnd", "algorithm")
# 



# ============================================================================== =
# Save ANOVA Results ====
# ============================================================================== =

cat("\n\n=== Saving ANOVA Results ===\n")

# Extract and save ANOVA results from permutation tests
save_anova_results <- function(model, filename) {
  res <- summary(model)
  # Convert to data frame
  anova_df <- as.data.frame(res)
  anova_df$term <- rownames(anova_df)
  rownames(anova_df) <- NULL
  # Reorder columns
  anova_df <- anova_df[, c(ncol(anova_df), 1:(ncol(anova_df)-1))]
  write_csv(anova_df, file.path(results_filepath, "tables", filename))
  cat("Saved:", filename, "\n")
  return(anova_df)
}

# Save entropy model
anova_entropy <- save_anova_results(model_entropy_perm, "anova_entropy.csv")

# Save latency variability model
anova_latstd <- save_anova_results(model_latstd_perm, "anova_latstd.csv")

# Save final PSE model
anova_finalpse <- save_anova_results(model_finalpse_perm, "anova_pse_final.csv")

# Save final JND model
anova_finaljnd <- save_anova_results(model_finaljnd_perm, "anova_jnd_final.csv")

# Save PSE convergence model
# anova_pse <- save_anova_results(model_pse_perm, "anova_pse_convergence.csv")

# Save JND convergence model
# anova_jnd <- save_anova_results(model_jnd_perm, "anova_jnd_convergence.csv")

cat("\n=== ANOVA Results Saved ===\n")
