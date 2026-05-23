# ==============================================================================
# 00_run_model_comparison.R
# Master script to run complete model comparison analysis
# ==============================================================================

cat("================================================================================\n")
cat("MODEL COMPARISON ANALYSIS PIPELINE\n")
cat("ABS1 vs REL1 vs REL2\n")
cat("================================================================================\n")

# Check and install required packages
required_packages <- c(
  "tidyverse", "readxl", "here", "janitor",
  "lme4", "lmerTest", "emmeans", "car",
  "permuco", "patchwork", "ggplot2"
)

missing_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]

if(length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n\n")
  install.packages(missing_packages)
}

# Load libraries
library(tidyverse)
library(here)

# ==============================================================================
# SETUP
# ==============================================================================
root_dir        <- "/data/CODE/python/adopy_tests/"
project_name    <- "R"
sim_file_name   <- "stimulus_metrics_all_models.csv"

project_dir <- paste0(root_dir, project_name, "/")

setwd(project_dir)

# ==============================================================================
# RUN ANALYSIS SCRIPTS
# ==============================================================================

cat("\n\n[1/5] Import data...\n")
# cat("================================================================================\n")
source("sim_01_import_data.R")

cat("\n\n[2/5] Running model comparison analysis...\n")
# cat("================================================================================\n")
source("sim_02_model_comparison.R")

cat("\n\n[3/5] Analyzing stimulus metrics...\n")
# cat("================================================================================\n")
source("sim_03_stimulus_metrics_analysis.R")

cat("\n\n[4/5] Analyzing asymmetry index evolution...\n")
# cat("================================================================================\n")
source("sim_04_asymmetry_index_evolution.R")

cat("\n\n[5/5] Analyzing latency entropy...\n")
# cat("================================================================================\n")
source("sim_05_lat_entropy_analysis.R")

cat("\n\n[5/5] Creating publication figures...\n")
# cat("================================================================================\n")
# source("00_create_paper_figures.R")

# ==============================================================================
# SUMMARY
# ==============================================================================

cat("================================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("================================================================================\n")

# cat("Results saved to:\n")
# cat("  - Tables: ./results_simulations/tables/\n")
# cat("  - Plots:  ./results_simulations/plots/\n")
# cat("  - Models: ./results_simulations/models/\n\n")

cat("Key output files:\n")
cat("  - convergence_metrics.csv: All convergence metrics by subject\n")
cat("  - pse_stability_desc.csv: PSE stability point summary\n")
cat("  - jnd_stability_desc.csv: JND stability point summary\n")
cat("  - pse_accuracy_desc.csv: PSE final accuracy summary\n")
cat("  - jnd_accuracy_desc.csv: JND final accuracy summary\n")
cat("  - auc_desc.csv: AUC (convergence speed) summary\n")
cat("  - stimulus_center_desc.csv: Stimulus center summary\n")
cat("  - stimulus_spread_desc.csv: Stimulus spread summary\n")
cat("  - stimulus_correlations_by_model.csv: Correlations with true parameters\n")
cat("  - asymmetry_index_ai_trajectory.csv: AI evolution by model and trial block\n")
cat("  - asymmetry_index_aiabs_trajectory.csv: |AI| evolution by model and trial block\n")
cat("  - lat_entropy_by_model.csv: Latency entropy summary by model\n")
cat("  - lat_entropy_by_model_block.csv: Latency entropy by model and trial block\n\n")

cat("Publication figures:\n")
cat("  - Figure4.png: Models Performance (Accuracy + Convergence)\n")
cat("  - Figure6.png: Analysis Metrics (Stimuli + Asymmetry + Entropy)\n\n")

# cat("Next steps:\n")
# cat("  1. Review tables in ./results_simulations/tables/\n")
# cat("  2. Examine plots in ./results_simulations/plots/\n")
# cat("  3. Update MODEL_COMPARISON_REPORT.md with findings\n")
# cat("  4. Share results with team\n\n")

cat("================================================================================\n")

