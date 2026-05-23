# ============================================================================== =
# 04_convergence_analysis.R
# Convergence metrics and stability analysis
# ============================================================================== =

library(tidyverse)
library(here)
library(permuco)

source("effect_size_utils.R")

# Load clean data
data_clean <- readRDS(here(results_filepath, "data_clean.rds"))

cat("=== Convergence Analysis ===\n")

# ============================================================================== =
# Stability Point Analysis ====
# ============================================================================== =

cat("\n=== Stability Point Analysis ===\n")
cat("Definition: First n_trials where estimate is within 10% of final (N=200)\n\n")

# Calculate stability point for each subject
stability_data <- data_clean %>%
  group_by(subj, modality, algorithm) %>%
  arrange(n_trials) %>%
  mutate(
    pse_final = first(pse[n_trials == 200]),
    jnd_final = first(jnd[n_trials == 200]),
    pse_diff_pct = abs(pse - pse_final) / pse_final * 100,
    jnd_diff_pct = abs(jnd - jnd_final) / jnd_final * 100,
    pse_stable = pse_diff_pct < 10,
    jnd_stable = jnd_diff_pct < 10
  ) %>%
  summarise(
    # PSE stability point
    pse_stability_point = ifelse(
      any(pse_stable),
      min(n_trials[pse_stable]),
      200
    ),
    # JND stability point
    jnd_stability_point = ifelse(
      any(jnd_stable),
      min(n_trials[jnd_stable]),
      200
    ),
    .groups = "drop"
  )

# Summary by Modality and Algorithm
stability_summary <- stability_data %>%
  group_by(modality, algorithm) %>%
  summarise(
    n = n(),
    pse_stab_mean = mean(pse_stability_point),
    pse_stab_sd = sd(pse_stability_point),
    pse_stab_median = median(pse_stability_point),
    jnd_stab_mean = mean(jnd_stability_point),
    jnd_stab_sd = sd(jnd_stability_point),
    jnd_stab_median = median(jnd_stability_point),
    .groups = "drop"
  )

cat("Stability point summary:\n")
print(stability_summary)

# Statistical test: Mode effect on stability
cat("\n=== Statistical Tests: Stability Point ===\n")

# PSE stability

cat("\nPSE stability point (AD vs FX):\n")


cat(sprintf("  Adaptive: M = %.1f, Fixed: M = %.1f\n",
            mean(stability_data$pse_stability_point[stability_data$algorithm == "Adaptive"]),
            mean(stability_data$pse_stability_point[stability_data$algorithm == "Fixed"])))

model_pse_stab_perm <- aovperm(pse_stability_point ~ modality * algorithm + Error(subj/(algorithm)), data = stability_data, np = 5000)
resPSE <- summary(model_pse_stab_perm)
print(resPSE)

# Calculate effect sizes using helper function
effect_sizes_pse_stab <- extract_eta_squared(model_pse_stab_perm)
print_effect_sizes(effect_sizes_pse_stab, "Effect Sizes for PSE Stability Point (η²)")

# JND stability
cat("\nJND stability point (Adaptive vs Fixed):\n")
cat(sprintf("  Adaptive: M = %.1f, Fixed: M = %.1f\n", 
            mean(stability_data$jnd_stability_point[stability_data$algorithm == "Adaptive"]),
            mean(stability_data$jnd_stability_point[stability_data$algorithm == "Fixed"])
            ))

model_jnd_stab_perm <- aovperm(jnd_stability_point ~ modality * algorithm + Error(subj/(algorithm)), data = stability_data, np = 5000)
resJND <- summary(model_jnd_stab_perm)
print(resJND)

# Calculate effect sizes using helper function
effect_sizes_jnd_stab <- extract_eta_squared(model_jnd_stab_perm)
print_effect_sizes(effect_sizes_jnd_stab, "Effect Sizes for JND Stability Point (η²)")




# ============================================================================== =
# Coefficient of Variation (CV) (no mode differences) 
# ============================================================================== =

# cat("\n=== Coefficient of Variation ===\n")
# 
# cv_data <- data_clean %>%
#   group_by(subj, modality, algorithm, age, gender) %>%
#   summarise(
#     cv_pse = 100 * sd(pse, na.rm = TRUE) / mean(pse, na.rm = TRUE),
#     cv_jnd = 100 * sd(jnd, na.rm = TRUE) / mean(jnd, na.rm = TRUE),
#     .groups = "drop"
#   )
# 
# # Summary
# cv_summary <- cv_data %>%
#   group_by(modality, algorithm) %>%
#   summarise(
#     n = n(),
#     cv_pse_mean = mean(cv_pse, na.rm = TRUE),
#     cv_pse_sd = sd(cv_pse, na.rm = TRUE),
#     cv_jnd_mean = mean(cv_jnd, na.rm = TRUE),
#     cv_jnd_sd = sd(cv_jnd, na.rm = TRUE),
#     .groups = "drop"
#   )
# 
# cat("\nCV summary:\n")
# print(cv_summary)
# 
# # Statistical test
# 
# cat("\nCV PSE (AD vs FX):\n")
# 
# cv_pse_test <- t.test(cv_pse ~ mode, data = cv_data)
# cat(sprintf(" t(%.2f) = %.2f, p = %.4f\n",
#             cv_pse_test$parameter, cv_pse_test$statistic, cv_pse_test$p.value))
# 
# res_cv_pse <- summary(aovperm(cv_pse ~ mode*task, data=cv_data))
# #               SS  df F          parametric P(>F) resampled P(>F)
# # mode       2.7645  1 2.6190           0.1161          0.1118
# # task       0.2903  1 0.2751           0.6038          0.6032
# # mode:task  2.7729  1 2.6270           0.1155          0.1126
# 
# 
# cat("\nCV JND (AD vs FX):\n")
# 
# cv_jnd_test <- t.test(cv_jnd ~ mode, data = cv_data)
# cat(sprintf("  t(%.2f) = %.2f, p = %.4f\n",
#             cv_jnd_test$parameter, cv_jnd_test$statistic, cv_jnd_test$p.value))
# 
# res_cv_jnd <- summary(aovperm(cv_jnd ~ mode*task, data=cv_data))
#               SS df  F          parametric P(>F) resampled P(>F)
# mode       261.20  1 2.4195           0.1303          0.1358
# task        29.38  1 0.2721           0.6057          0.6154
# mode:task  132.55  1 1.2278           0.2766          0.2810

# ============================================================================== =
# Area Under the Curve (AUC) ====
# ============================================================================== =

cat("\n=== Area Under the Curve ===\n")
cat("Measuring total 'distance' from final estimate across all n_trials\n\n")

auc_data <- data_clean %>%
  group_by(subj, modality, algorithm) %>%
  arrange(n_trials) %>%
  mutate(
    pse_final = first(pse[n_trials == 200]),
    jnd_final = first(jnd[n_trials == 200]),
    pse_diff = abs(pse - pse_final),
    jnd_diff = abs(jnd - jnd_final)
  ) %>%
  summarise(
    auc_pse = sum(pse_diff * 20),  # 20 = interval between n_trials
    auc_jnd = sum(jnd_diff * 20),
    .groups = "drop"
  )

# Summary
auc_summary <- auc_data %>%
  group_by(modality, algorithm) %>%
  summarise(
    n = n(),
    auc_pse_mean = mean(auc_pse),
    auc_pse_sd = sd(auc_pse),
    auc_jnd_mean = mean(auc_jnd),
    auc_jnd_sd = sd(auc_jnd),
    .groups = "drop"
  )

cat("AUC summary:\n")
cat("  Lower AUC = faster convergence\n")
print(auc_summary)

# Statistical test

model_pse_auc_perm <- aovperm(auc_pse ~ modality * algorithm + Error(subj/(algorithm)), data = auc_data, np = 5000)
resPSE_AUC <- summary(model_pse_auc_perm)
print(resPSE_AUC)

# Resampling test using Rd_kheradPajouh_renaud to handle nuisance variables and 5000 permutations.
# SSn dfn      SSd dfd    MSEn    MSEd      F parametric P(>F) resampled P(>F)
# modality           2171656   1 20255805  18 2171656 1125322 1.9298          0.18173          0.1870
# algorithm          7171739   1 18353695  18 7171739 1019650 7.0335          0.01622          0.0164
# modality:algorithm  131481   1 18353695  18  131481 1019650 0.1289          0.72370          0.7234

# Calculate effect sizes using helper function
effect_sizes_pse_auc <- extract_eta_squared(model_pse_auc_perm)
print_effect_sizes(effect_sizes_pse_auc, "Effect Sizes for PSE AUC (η²)")


model_jnd_auc_perm <- aovperm(auc_jnd ~ modality * algorithm + Error(subj/(algorithm)), data = auc_data, np = 5000)
resJND_AUC <- summary(model_jnd_auc_perm)
print(resJND_AUC)

# Resampling test using Rd_kheradPajouh_renaud to handle nuisance variables and 5000 permutations.
# SSn dfn     SSd dfd    MSEn   MSEd      F parametric P(>F) resampled P(>F)
# modality            499792   1 3833468  18  499792 212970  2.347        0.1429299          0.1438
# algorithm          4488084   1 3543463  18 4488084 196859 22.798        0.0001515          0.0002
# modality:algorithm  570718   1 3543463  18  570718 196859  2.899        0.1058334          0.1052

# Calculate effect sizes using helper function
effect_sizes_jnd_auc <- extract_eta_squared(model_jnd_auc_perm)
print_effect_sizes(effect_sizes_jnd_auc, "Effect Sizes for JND AUC (η²)")


# ============================================================================== =
# Correlation: Latency Entropy and Convergence 
# ============================================================================== =
# 
# cat("\n=== Entropy and Convergence ===\n")
# 
# # Merge entropy with convergence metrics
# entropy_conv <- data_clean %>%
#   filter(n_trials == 200) %>%
#   select(subj, modality, algorithm, lat_entropy) %>%
#   left_join(cv_data, by = c("subj", "modality", "algorithm"))
# 
# # Correlations
# cor_entropy_cv_pse <- cor.test(entropy_conv$lat_entropy, entropy_conv$cv_pse)
# cat("\nCorrelation: Entropy vs CV(PSE)\n")
# cat(sprintf("  r = %.3f, p = %.4f\n", 
#             cor_entropy_cv_pse$estimate, cor_entropy_cv_pse$p.value))
# 
# cor_entropy_cv_jnd <- cor.test(entropy_conv$lat_entropy, entropy_conv$cv_jnd)
# cat("\nCorrelation: Entropy vs CV(JND)\n")
# cat(sprintf("  r = %.3f, p = %.4f\n",
#             cor_entropy_cv_jnd$estimate, cor_entropy_cv_jnd$p.value))


# ============================================================================== =
# Save Results
# ============================================================================== = 

write_csv(stability_data, file.path(results_filepath, "tables", "stability_points.csv"))
write_csv(stability_summary, file.path(results_filepath, "tables", "stability_summary.csv"))
# write_csv(cv_data, file.path(results_filepath, "tables", "cv_individual.csv"))
# write_csv(cv_summary, file.path(results_filepath, "tables", "cv_summary.csv"))
write_csv(auc_data, file.path(results_filepath, "tables", "auc_individual.csv"))
write_csv(auc_summary, file.path(results_filepath, "tables", "auc_summary.csv"))
# write_csv(entropy_conv, file.path(results_filepath, "tables", "entropy_convergence.csv"))

cat("\n=== Convergence Analysis Complete ===\n")
cat("Results saved to:", file.path(results_filepath, "tables"), "\n")

