# ==============================================================================
# 01_import_data.R
# Import and prepare progressive psychophysics data
# ==============================================================================

# Load required packages
library(tidyverse)
library(readxl)
library(here)
library(janitor)



# Create output directories if they don't exist
dir.create(file.path(results_filepath, "tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(results_filepath, "plots"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(results_filepath, "models"), recursive = TRUE, showWarnings = FALSE)

cat("=== Data Import ===\n")
cat("Reading data from:", data_file, "\n")

# Import data
data_raw <- read_excel(data_file)

# Display structure
# cat("\nRaw data structure:\n")
# print(str(data_raw))
# cat("\nFirst few rows:\n")
# print(head(data_raw))

# ==============================================================================
# Data Cleaning and Preparation
# ==============================================================================

data_clean2 <- data_raw %>%
  # Clean column names
  clean_names() %>%
  
  # Convert categorical variables to factors
  mutate(
    subj = factor(subj),
    gender = factor(gender, levels = c("f", "m"), labels = c("Female", "Male")),
    modality = factor(modality, levels = c("BISA", "BISV"), labels = c("Auditory", "Visual")),
    algorithm = factor(algorithm, levels = c("AD", "FX"), labels = c("Adaptive", "Fixed")),
    n_trials = as.integer(n_trials)
  ) %>%
  
  # Remove group variable (all TD)
  dplyr::select(-group) %>%
  
  # Create derived variables
  mutate(
    # Log-transform JND for normality (if needed)
    jnd_log = log(jnd + 1),  # +1 to handle zeros
    
    # Standardize age for modeling
    age_z = scale(age)[,1],
    
    # Create modality-algorithm combination
    modality_algorithm = interaction(modality, algorithm, sep = "_")
  ) %>%
  
  # Arrange data
  arrange(subj, n_trials)

# data_clean <- data_clean2 %>% filter(subj != "dibiase_gaia") #| subj != "pietronave_erica")
# data_clean <- data_clean  %>% filter(subj != "pietronave_erica")

data_clean <- data_clean2

data_clean$n_trials_f <- factor(data_clean$n_trials) # good to run aovperm

# ==============================================================================
# Data Quality Checks
# ==============================================================================

cat("\n=== Data Quality Checks ===\n")

# Check for missing values
missing_summary <- data_clean %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing") %>%
  filter(n_missing > 0)

if(nrow(missing_summary) > 0) {
  cat("\nMissing values detected:\n")
  print(missing_summary)
} else {
  cat("\nNo missing values detected.\n")
}

# Check for negative JND values (should not exist after correction)
neg_jnd <- data_clean %>% filter(jnd < 0)
if(nrow(neg_jnd) > 0) {
  cat("\nWARNING: Negative JND values found:\n")
  print(neg_jnd)
} else {
  cat("No negative JND values (good!).\n")
}

# Check for extreme outliers (> 3 SD from mean)
outliers_pse <- data_clean %>%
  group_by(modality, algorithm, n_trials) %>%
  mutate(
    pse_z = scale(pse)[,1],
    is_outlier = abs(pse_z) > 3
  ) %>%
  filter(is_outlier) %>%
  dplyr::select(subj, modality, algorithm, n_trials, pse, pse_z)

outliers_jnd <- data_clean %>%
  group_by(modality, algorithm, n_trials) %>%
  mutate(
    jnd_z = scale(jnd)[,1],
    is_outlier = abs(jnd_z) > 3
  ) %>%
  filter(is_outlier) %>%
  dplyr::select(subj, modality, algorithm, n_trials, jnd, jnd_z)

cat("\nPSE outliers (|z| > 3):", nrow(outliers_pse), "\n")
if(nrow(outliers_pse) > 0) print(outliers_pse)

cat("\nJND outliers (|z| > 3):", nrow(outliers_jnd), "\n")
if(nrow(outliers_jnd) > 0) print(outliers_jnd)

# ==============================================================================
# Sample Summary
# ==============================================================================

cat("\n=== Sample Summary ===\n")

# Overall sample size
cat("\nTotal observations:", nrow(data_clean), "\n")
cat("Unique subjects:", n_distinct(data_clean$subj), "\n")
cat("Progressive points:", n_distinct(data_clean$n_trials), "\n")

# Sample by Modality and Algorithm
sample_summary <- data_clean %>%
  filter(n_trials == 200) %>%  # One row per subject
  group_by(modality, algorithm) %>%
  summarise(
    n = n(),
    age_mean = mean(age, na.rm = TRUE),
    age_sd = sd(age, na.rm = TRUE),
    n_female = sum(gender == "Female"),
    n_male = sum(gender == "Male"),
    .groups = "drop"
  )

cat("\nSample by Modality and Algorithm:\n")
print(sample_summary)

# ==============================================================================
# Save Clean Data
# ==============================================================================

# Save as RDS for R
saveRDS(data_clean, file.path(results_filepath, "data_clean.rds"))
cat("\nClean data saved to:", file.path(results_filepath, "data_clean.rds"), "\n")

# Save as CSV for inspection
write_csv(data_clean, file.path(results_filepath, "tables", "data_clean.csv"))
cat("Clean data saved to:", file.path(results_filepath, "tables", "data_clean.csv"), "\n")

# Save sample summary
write_csv(sample_summary, file.path(results_filepath, "tables", "sample_summary.csv"))
cat("Sample summary saved to:", file.path(results_filepath, "tables", "sample_summary.csv"), "\n")

# cat("\n=== Data Import Complete ===\n")
