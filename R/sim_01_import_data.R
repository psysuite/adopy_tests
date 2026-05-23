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
sim_file_path <- paste0(project_dir, "indata/", sim_file_name)

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
data_raw <- read_csv(sim_file_path, show_col_types = FALSE) #, stringsAsFactors = FALSE)
