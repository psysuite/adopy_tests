# ==============================================================================
# 01_import_data.R
# Import and prepare simulation data
# ==============================================================================

# Load required packages
library(tidyverse)
library(readxl)
library(here)
library(janitor)

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
# Updated: use new Excel format instead of CSV
sim_file_path <- paste0(project_dir, "indata/synthetic_data_long.xlsx")

# Create output directories
dir.create(file.path(results_filepath, "tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(results_filepath, "plots"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(results_filepath, "models"), recursive = TRUE, showWarnings = FALSE)

setwd(project_dir)

# ============================================================================== =
# IMPORT DATA
# ============================================================================== =

cat("=== Data Import ===\n")

if (! file.exists(sim_file_path)) {
  stop("Could not find stimulus_metrics_all_models.csv in any expected location")
}

cat("Reading data from:", sim_file_path, "\n")

# Import data
data_raw <- read_excel(sim_file_path)

# Note: Data now includes progressive columns:
# - pct_correct_40, pct_correct_60, ..., pct_correct_200: % correct responses at each trial block
# - posterior_sd_pse_*, posterior_sd_jnd_*: Posterior uncertainty at each block
# - pse_auc, jnd_auc: Convergence speed (time-weighted error accumulation)
