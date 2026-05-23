# ============================================================================== =
# 02_descriptive_analysis.R
# Descriptive statistics and distribution checks
# ============================================================================== =

library(tidyverse)
library(here)
library(knitr)
library(kableExtra)

# Load clean data
data_clean  <- readRDS(here(results_filepath, "data_clean.rds"))

cat("=== Descriptive Analysis ===\n")

# ============================================================================== =
# Demographics ====
# ============================================================================== =

cat("\n=== Demographics ===\n")

# Age and gender by Modality
demo_task <- data_clean %>%
  filter(n_trials == 200) %>%
  group_by(modality) %>%
  summarise(
    n = n(),
    age_mean = mean(age),
    age_sd = sd(age),
    age_min = min(age),
    age_max = max(age),
    n_female = sum(gender == "Female"),
    n_male = sum(gender == "Male"),
    pct_female = round(100 * n_female / n, 1),
    .groups = "drop"
  )

cat("\nDemographics by Modality:\n")
print(demo_task)

# Test age difference between modalities
age_test <- t.test(age ~ modality, data = filter(data_clean, n_trials == 200))
cat("\nAge comparison (Auditory vs Visual):\n")
cat(sprintf("  t(%.1f) = %.2f, p = %.4f\n", 
            age_test$parameter, age_test$statistic, age_test$p.value))

# Test gender distribution
gender_table <- table(
  filter(data_clean, n_trials == 200)$modality,
  filter(data_clean, n_trials == 200)$gender
)
gender_test <- chisq.test(gender_table)
cat("\nGender distribution (Auditory vs Visual):\n")
cat(sprintf("  χ²(%d) = %.2f, p = %.4f\n",
            gender_test$parameter, gender_test$statistic, gender_test$p.value))

# ============================================================================== =
# Descriptive Statistics for PSE and JND ====
# ============================================================================== =

cat("\n=== Descriptive Statistics ===\n")

# PSE by Modality, Algorithm, and n_trials
pse_desc <- data_clean %>%
  group_by(modality, algorithm, n_trials) %>%
  summarise(
    n = n(),
    mean = mean(pse, na.rm = TRUE),
    sd = sd(pse, na.rm = TRUE),
    median = median(pse, na.rm = TRUE),
    q25 = quantile(pse, 0.25, na.rm = TRUE),
    q75 = quantile(pse, 0.75, na.rm = TRUE),
    min = min(pse, na.rm = TRUE),
    max = max(pse, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nPSE descriptive statistics :\n")
print(filter(pse_desc, n_trials == 200))

# JND by Modality, Algorithm, and n_trials
jnd_desc <- data_clean %>%
  group_by(modality, algorithm, n_trials) %>%
  summarise(
    n = n(),
    mean = mean(jnd, na.rm = TRUE),
    sd = sd(jnd, na.rm = TRUE),
    median = median(jnd, na.rm = TRUE),
    q25 = quantile(jnd, 0.25, na.rm = TRUE),
    q75 = quantile(jnd, 0.75, na.rm = TRUE),
    min = min(jnd, na.rm = TRUE),
    max = max(jnd, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nJND descriptive statistics:\n")
print(filter(jnd_desc, n_trials == 200))

# ============================================================================== =
# Normality Tests ====
# ============================================================================== =

cat("\n=== Normality Tests ===\n")

# Shapiro-Wilk test for PSE at each n_trials and condition
normality_pse <- data_clean %>%
  group_by(modality, algorithm, n_trials) %>%
  summarise(
    n = n(),
    shapiro_w = shapiro.test(pse)$statistic,
    shapiro_p = shapiro.test(pse)$p.value,
    is_normal = shapiro_p > 0.05,
    .groups = "drop"
  )

cat("\nPSE normality tests (Shapiro-Wilk):\n")
cat("Non-normal distributions (p < 0.05):\n")
print(filter(normality_pse, !is_normal))

# Shapiro-Wilk test for JND
normality_jnd <- data_clean %>%
  group_by(modality, algorithm, n_trials) %>%
  summarise(
    n = n(),
    shapiro_w = shapiro.test(jnd)$statistic,
    shapiro_p = shapiro.test(jnd)$p.value,
    is_normal = shapiro_p > 0.05,
    .groups = "drop"
  )

cat("\nJND normality tests (Shapiro-Wilk):\n")
cat("Non-normal distributions (p < 0.05):\n")
print(filter(normality_jnd, !is_normal))

# ============================================================================== =
# Coefficient of Variation (Convergence Metric)
# ============================================================================== =
# 
# cat("\n=== Coefficient of Variation ===\n")
# 
# cv_summary <- data_clean %>%
#   group_by(subj, modality, algorithm) %>%
#   summarise(
#     cv_pse = 100 * sd(pse, na.rm = TRUE) / mean(pse, na.rm = TRUE),
#     cv_jnd = 100 * sd(jnd, na.rm = TRUE) / mean(jnd, na.rm = TRUE),
#     .groups = "drop"
#   ) %>%
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
# cat("\nCoefficient of Variation by Task and Mode:\n")
# print(cv_summary)

# ============================================================================== =
# Latency Statistics ====
# ============================================================================== =

cat("\n=== Latency Statistics (SC & SS) ===\n")

latency_summary <- data_clean %>%
  filter(n_trials == 200) %>%
  group_by(modality, algorithm) %>%
  summarise(
    sc_avg = mean(sc, na.rm = TRUE),
    sc_sd  = sd(sc, na.rm = TRUE),
    ss_avg = mean(ss, na.rm = TRUE),
    ss_sd  = sd(ss, na.rm = TRUE),
    
    lat_entropy_avg = mean(lat_entropy, na.rm = TRUE),
    lat_entropy_sd  = sd(lat_entropy, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nLatency characteristics at N=200:\n")
print(latency_summary)


# ============================================================================== =
# Save Results ====
# ============================================================================== =

# Save descriptive tables
write_csv(demo_task, file.path(results_filepath, "tables", "demographics.csv"))
write_csv(pse_desc, file.path(results_filepath, "tables", "pse_descriptives.csv"))
write_csv(jnd_desc, file.path(results_filepath, "tables", "jnd_descriptives.csv"))
write_csv(normality_pse, file.path(results_filepath, "tables", "normality_pse.csv"))
write_csv(normality_jnd, file.path(results_filepath, "tables", "normality_jnd.csv"))
# write_csv(cv_summary, file.path(results_filepath, "tables", "cv_summary.csv"))
write_csv(latency_summary, file.path(results_filepath, "tables", "latency_summary.csv"))

cat("\n=== Descriptive Analysis Complete ===\n")
cat("Tables saved to:", file.path(results_filepath, "tables"), "\n")
