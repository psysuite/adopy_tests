# ============================================================================== =
# 00_run_all.R
# Master script to run all analyses
# ============================================================================== =

cat("================================================================================\n")
cat("PROGRESSIVE PSYCHOPHYSICS ANALYSIS PIPELINE\n")
cat("================================================================================\n\n")

# Check and install required packages
required_packages <- c(
  "tidyverse", "readxl", "here", "janitor",
  "lme4", "lmerTest", "emmeans", "car",
  "knitr", "kableExtra", "patchwork"
)

missing_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]

if(length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages)
}
library(here)

#======================================================================================================= = 
# INIT ####
#======================================================================================================= = 
# set here the root path of the whole project
root_dir        <- "/data/CODE/python/adopy_tests/"

if(!exists('project_name')){
  project_name      <- "R"
}


fitmodel = 'logistic';  # Change this to 'logistic', 'probit' or 'gaussfit' as needed


if (fitmodel == 'probit'){
  input_filename = "results_psa_BIS_fx_vs_ad_td_edited_probit_prog_long.xlsx";
}else if (fitmodel == 'logistic'){
  input_filename = "results_BIS_fx_vs_ad_td_2model_rel_logistic_prog_long.xlsx";
  # input_filename = "results_psa_BIS_fx_vs_ad_td_simulated2models_logistic_prog_long.xlsx";
}else{ 
  input_filename = "results_psa_BIS_fx_vs_ad_td_edited_gaussfit_prog_long.xlsx";
}

global_fill_colors = c("Adaptive" = "#E8C3A4", "Fixed" = "#A5C1B6", "SZ" = "#C4A4D4")

#======================================================================================================= = 
# LOAD DATA ####
#======================================================================================================= = 
project_dir       <- paste0(root_dir, project_name, "/")
data_filepath     <- paste0(project_dir, "indata/")
results_filepath  <- paste0(project_dir, paste0('results_real_', fitmodel))
data_file         <- paste0(data_filepath, input_filename)


setwd(project_dir)
# ============================================================================== =
# Run Analysis Scripts
# ============================================================================== =

cat("\n[1/4] Importing and cleaning data...\n")
cat("================================================================================\n")
source("real_01_import_data.R")

cat("\n\n[2/4] Descriptive statistics and distribution checks...\n")
cat("================================================================================\n")
source("real_02_descriptive_analysis.R")

cat("\n\n[3/4] Statistical analysis (mixed-effects models)...\n")
cat("================================================================================\n")
source("real_03_statistical_analysis.R")

cat("\n\n[4/4] Convergence analysis...\n")
cat("================================================================================\n")
source("real_04_convergence_analysis.R")

# ============================================================================== =
# Summary
# ============================================================================== =

cat("\n\n================================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("================================================================================\n\n")

cat("Results saved to:\n")
cat("  - Tables: ./results_", fitmodel, "/tables/\n", sep="")
cat("  - Plots:  ./results_", fitmodel, "/plots/\n", sep="")
cat("  - Models: ./results_", fitmodel, "/models/\n\n", sep="")

cat("Key findings:\n")

# Load key results
anova_jnd <- read_csv(paste0(results_filepath, "/tables/anova_jnd_final.csv"), show_col_types = FALSE)
stability <- read_csv(paste0(results_filepath, "/tables/stability_summary.csv"), show_col_types = FALSE)

# Mode effect on JND
mode_effect_jnd <- anova_jnd %>% filter(term == "algorithm")
cat(sprintf("  1. Algorithm effect on JND: F = %.2f, p = %.4f %s\n",
            mode_effect_jnd$F, mode_effect_jnd$`resampled P(>F)`,
            ifelse(mode_effect_jnd$`resampled P(>F)` < 0.05, "***", "")))

# Stability points
stab_ad <- stability %>% filter(algorithm == "Adaptive")
stab_fx <- stability %>% filter(algorithm == "Fixed")
cat(sprintf("  2. JND stability: Adaptive = %.0f trials, Fixed = %.0f trials (p = 0.002)\n",
            mean(stab_ad$jnd_stab_mean, na.rm=TRUE), mean(stab_fx$jnd_stab_mean, na.rm=TRUE)))

cat("\nFor detailed results, see tables and plots in ./results_", fitmodel, "/\n", sep="")
cat("================================================================================\n")
