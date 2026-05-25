#!/usr/bin/env Rscript
# ============================================================================== =
# 00_create_paper_figures.R
# Master script to create all publication figures
# Combines simulation and real data analysis plots
# ============================================================================== =

library(tidyverse)
library(patchwork)
library(ggplot2)

cat("================================================================================\n")
cat("CREATING PUBLICATION FIGURES\n")
cat("================================================================================\n\n")

# ============================================================================== =
# SETUP
# ============================================================================== =
root_dir <- "/data/CODE/python/adopy_tests/"
project_name <- "R"
project_dir <- paste0(root_dir, project_name, "/")
indata_dir  <- paste0(root_dir, project_name, "/indata/")
sim_results_filepath <- paste0(project_dir, "results_simulations")
real_results_filepath <- paste0(project_dir, "results_real_logistic")
paper_plots_dir <- paste0(root_dir, "data/paper_plots/")

setwd(project_dir)

# Create output directory
dir.create(file.path(sim_results_filepath, "plots"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(real_results_filepath, "plots"), recursive = TRUE, showWarnings = FALSE)


sim_results_data_input <- paste0(indata_dir, "stimulus_metrics_all_models.csv")

# ============================================================================== =
# SIMULATION DATA FIGURES
# ============================================================================== =

cat("=== SIMULATION DATA FIGURES ===\n\n")

# Define color palette for 3 models
model_colors <- c("ABS1" = "#E41A1C", "REL1" = "#377EB8", "REL2" = "#4DAF4A")

# Define colors for real data
colors_algorithm <- c("Adaptive" = "#CC3333", "Fixed" = "#3333CC")
colors_modality <- c("Auditory" = "#E69F00", "Visual" = "#56B4E9")

# ============================================================================== =
# PLOT THEME FOR PAPER
# ============================================================================== =

theme_paper <- function() {
  theme_minimal() + theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA),
    strip.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 10, face = "bold", hjust = 0, margin = margin(b = 4)),
    plot.subtitle = element_text(size = 9, color = "gray40", margin = margin(b = 3)),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 8),
    legend.position = "top",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    legend.margin = margin(t = 0, b = 0),
    plot.margin = margin(t = 4, r = 4, b = 4, l = 4),
    strip.text = element_text(size = 8, face = "bold")
  )
}

theme_set(theme_paper())

# ============================================================================== =
# LOAD DATA
# ============================================================================== =

cat("=== Loading Data ===\n")

# Simulation data
data_sim_clean            <- readRDS(file.path(sim_results_filepath, "data_clean.rds"))
convergence_metrics       <- readRDS(file.path(sim_results_filepath, "convergence_metrics.rds"))
stimulus_metrics_final    <- readRDS(file.path(sim_results_filepath, "models", "stimulus_metrics_final.rds"))
asymmetry_evolution_data  <- readRDS(file.path(sim_results_filepath, "models", "asymmetry_evolution_data.rds"))
lat_entropy_plot_data     <- readRDS(file.path(sim_results_filepath, "models", "lat_entropy_plot_data.rds"))

# Load raw simulation data for lat_entropy boxplot
df_sim_raw                <- read.csv(sim_results_data_input, stringsAsFactors = FALSE)
df_sim_raw$model          <- factor(df_sim_raw$model, levels = c("ABS1", "REL1", "REL2"))
df_sim_raw$trial_block    <- as.integer(df_sim_raw$trial_block)

# Real data
data_real_clean           <- readRDS(file.path(real_results_filepath, "data_clean.rds"))

# ============================================================================== =
# FIGURE 4: Models Performance (Accuracy + Convergence) ====
# ============================================================================== =

cat("=== Creating FIGURE 4: Models Performance ===\n")

# --- PSE Convergence ---
pse_convergence <- data_sim_clean %>%
  group_by(model, trial_block) %>%
  summarise(
    mean_error = mean(abs(pse_error), na.rm = TRUE),
    se_error = sd(abs(pse_error), na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

p1_pse <- ggplot(pse_convergence, aes(x = trial_block, y = mean_error, 
                                       color = model, fill = model)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  geom_ribbon(aes(ymin = mean_error - se_error, ymax = mean_error + se_error),
              alpha = 0.2, color = NA) +
  scale_color_manual(values = model_colors) +
  scale_fill_manual(values = model_colors) +
  labs(
    title = "PSE Convergence",
    x = "Trial Block",
    y = "Absolute Error (ms)",
    color = "Model",
    fill = "Model"
  ) +
  theme_paper() +
  theme(legend.position = "bottom")

# --- JND Convergence ---
jnd_convergence <- data_sim_clean %>%
  group_by(model, trial_block) %>%
  summarise(
    mean_error = mean(abs(jnd_error), na.rm = TRUE),
    se_error = sd(abs(jnd_error), na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

p2_jnd <- ggplot(jnd_convergence, aes(x = trial_block, y = mean_error, 
                                       color = model, fill = model)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  geom_ribbon(aes(ymin = mean_error - se_error, ymax = mean_error + se_error),
              alpha = 0.2, color = NA) +
  scale_color_manual(values = model_colors) +
  scale_fill_manual(values = model_colors) +
  labs(
    title = "JND Convergence",
    x = "Trial Block",
    y = "Absolute Error",
    color = "Model",
    fill = "Model"
  ) +
  theme_paper() +
  theme(legend.position = "bottom")

# --- PSE Stability ---
p3_pse_stab <- ggplot(convergence_metrics, aes(x = model, y = pse_stability_point, 
                                                fill = model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.size = 1.5, color = "black", linewidth = 0.4) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1.5) +
  scale_fill_manual(values = model_colors) +
  labs(
    title = "PSE Stability",
    x = "Model",
    y = "Stability Point",
    fill = "Model"
  ) +
  theme_paper() +
  theme(legend.position = "none")

# --- JND Stability ---
p4_jnd_stab <- ggplot(convergence_metrics, aes(x = model, y = jnd_stability_point, 
                                                fill = model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.size = 1.5, color = "black", linewidth = 0.4) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1.5) +
  scale_fill_manual(values = model_colors) +
  labs(
    title = "JND Stability",
    x = "Model",
    y = "Stability Point",
    fill = "Model"
  ) +
  theme_paper() +
  theme(legend.position = "none")

# --- PSE Accuracy ---
p5_pse_acc <- ggplot(convergence_metrics, aes(x = model, y = pse_final_error_pct, 
                                               fill = model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.size = 1.5, linewidth = 0.4) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1.5) +
  scale_fill_manual(values = model_colors) +
  labs(
    title = "PSE Accuracy",
    x = "Model",
    y = "Percent Error (%)",
    fill = "Model"
  ) +
  theme_paper() +
  theme(legend.position = "none")

# --- JND Accuracy ---
p6_jnd_acc <- ggplot(convergence_metrics, aes(x = model, y = jnd_final_error_pct, 
                                               fill = model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.size = 1.5, linewidth = 0.4) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1.5) +
  scale_fill_manual(values = model_colors) +
  labs(
    title = "JND Accuracy",
    x = "Model",
    y = "Percent Error (%)",
    fill = "Model"
  ) +
  theme_paper() +
  theme(legend.position = "none")

# --- PSE AUC ---
p7_pse_auc <- ggplot(convergence_metrics, aes(x = model, y = auc_pse, 
                                               fill = model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.size = 1.5, linewidth = 0.4) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1.5) +
  scale_fill_manual(values = model_colors) +
  labs(
    title = "PSE Convergence Speed",
    x = "Model",
    y = "Area Under Curve",
    fill = "Model"
  ) +
  theme_paper() +
  theme(legend.position = "none")

# --- JND AUC ---
p8_jnd_auc <- ggplot(convergence_metrics, aes(x = model, y = auc_jnd, 
                                               fill = model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.size = 1.5, linewidth = 0.4) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1.5) +
  scale_fill_manual(values = model_colors) +
  labs(
    title = "JND Convergence Speed",
    x = "Model",
    y = "Area Under Curve",
    fill = "Model"
  ) +
  theme_paper() +
  theme(legend.position = "none")

# Remove titles for cleaner grid
p1_pse_notitle <- p1_pse + labs(title = NULL)
p2_jnd_notitle <- p2_jnd + labs(title = NULL)
p3_pse_stab_notitle <- p3_pse_stab + labs(title = NULL)
p4_jnd_stab_notitle <- p4_jnd_stab + labs(title = NULL)
p5_pse_acc_notitle <- p5_pse_acc + labs(title = NULL)
p6_jnd_acc_notitle <- p6_jnd_acc + labs(title = NULL)
p7_pse_auc_notitle <- p7_pse_auc + labs(title = NULL)
p8_jnd_auc_notitle <- p8_jnd_auc + labs(title = NULL)

# Add subtitles for PSE/JND reference
p1_pse_sub <- p1_pse_notitle + labs(subtitle = "PSE")
p2_jnd_sub <- p2_jnd_notitle + labs(subtitle = "JND")
p3_pse_stab_sub <- p3_pse_stab_notitle + labs(subtitle = "PSE")
p4_jnd_stab_sub <- p4_jnd_stab_notitle + labs(subtitle = "JND")

# Create section labels
title_accuracy <- ggplot() + 
  annotate("text", x = 0.5, y = 0.5, label = "Accuracy", 
           size = 5, fontface = "bold", color = "gray40") +
  theme_void()

title_convergence <- ggplot() + 
  annotate("text", x = 0.5, y = 0.5, label = "Convergence", 
           size = 5, fontface = "bold", color = "gray40") +
  theme_void()

# Create Figure 4 (Accuracy) and Figure 5 (Convergence) separately
p_figure4_accuracy <- (p1_pse_sub | p5_pse_acc_notitle) / 
                      (p2_jnd_sub | p6_jnd_acc_notitle)

p_figure5_convergence <- (p3_pse_stab_sub | p7_pse_auc_notitle) / 
                         (p4_jnd_stab_sub | p8_jnd_auc_notitle)

# Remove titles for cleaner combination
p_figure4_accuracy_notitle <- p_figure4_accuracy + labs(title = NULL)
p_figure5_convergence_notitle <- p_figure5_convergence + labs(title = NULL)

# Combine Figure 4 with section labels
p_figure4 <- (title_accuracy | title_convergence) / 
             (p_figure4_accuracy_notitle | p_figure5_convergence_notitle) +
  plot_layout(heights = c(0.05, 1)) +
  plot_annotation(
    title = "Models Performance",
    theme = theme(plot.title = element_text(face = "bold", size = 13, hjust = 0.5))
  )

ggsave(file.path(paper_plots_dir, "Figure4.tif"),
       p_figure4, width = 32, height = 14.5, dpi = 300, units = "cm",
       compression = "lzw")
cat("✓ Saved: Figure4.tif\n")

# ============================================================================== =
# FIGURE 6: Combined Analysis Metrics ====
# ============================================================================== =

cat("\n=== Creating FIGURE 6: Combined Analysis Metrics ===\n")

# --- Stimulus Center vs PSE ---
p_center_scatter <- stimulus_metrics_final %>%
  ggplot(aes(x = pse_true, y = stimulus_center, color = model)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", se = TRUE, alpha = 0.2, linewidth = 0.8) +
  scale_color_manual(values = model_colors) +
  labs(
    title = "Stimulus Center vs PSE",
    x = "PSE (ms)",
    y = "Stimulus Center"
  ) +
  theme_paper()

# --- Stimulus Center Boxplot ---
p_center_box <- stimulus_metrics_final %>%
  ggplot(aes(x = model, y = stimulus_center, fill = model)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1.5, linewidth = 0.4) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1.5) +
  scale_fill_manual(values = model_colors) +
  labs(
    x = "Model",
    y = "Stimulus Center"
  ) +
  theme_paper() +
  theme(legend.position = "none")

# --- Stimulus Spread vs JND ---
p_spread_scatter <- stimulus_metrics_final %>%
  ggplot(aes(x = jnd_true, y = stimulus_spread, color = model)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", se = TRUE, alpha = 0.2, linewidth = 0.8) +
  scale_color_manual(values = model_colors) +
  labs(
    title = "Stimulus Spread vs JND",
    x = "JND (ms)",
    y = "Stimulus Spread"
  ) +
  theme_paper()

# --- Stimulus Spread Boxplot ---
p_spread_box <- stimulus_metrics_final %>%
  ggplot(aes(x = model, y = stimulus_spread, fill = model)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1.5, linewidth = 0.4) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1.5) +
  scale_fill_manual(values = model_colors) +
  labs(
    x = "Model",
    y = "Stimulus Spread"
  ) +
  theme_paper() +
  theme(legend.position = "none")

# --- Asymmetry Index Evolution (ABS1 only) ---
data_abs1 <- asymmetry_evolution_data %>%
  filter(model == "ABS1")

data_abs1_combined <- bind_rows(
  data_abs1 %>% 
    dplyr::select(trial_block, asymmetry_index) %>%
    rename(value = asymmetry_index) %>%
    mutate(type = "AI (Real)"),
  data_abs1 %>% 
    dplyr::select(trial_block, asymmetry_index_abs) %>%
    rename(value = asymmetry_index_abs) %>%
    mutate(type = "|AI| (Absolute)")
)

medians_by_block <- data_abs1_combined %>%
  group_by(trial_block, type) %>%
  summarise(median_value = median(value, na.rm = TRUE), .groups = "drop")

p_ai_evolution <- data_abs1_combined %>%
  ggplot(aes(x = factor(trial_block), y = value, fill = type)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1.2, linewidth = 0.4) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1.2, aes(color = type)) +
  geom_line(data = medians_by_block, aes(x = as.numeric(factor(trial_block)), y = median_value, 
                                          color = type, group = type), 
            linewidth = 0.8, alpha = 0.8, linetype = "solid") +
  geom_point(data = medians_by_block, aes(x = as.numeric(factor(trial_block)), y = median_value, 
                                           color = type), size = 2.5, shape = 16) +
  scale_fill_manual(values = c("AI (Real)" = "#377EB8", "|AI| (Absolute)" = "#E41A1C")) +
  scale_color_manual(values = c("AI (Real)" = "#377EB8", "|AI| (Absolute)" = "#E41A1C")) +
  facet_wrap(~type, scales = "free_y", ncol = 2) +
  labs(
    title = "ABS1 - AI Evolution",
    x = "Trial Block",
    y = "Value"
  ) +
  theme_paper() +
  theme(legend.position = "none")

# --- Latency Entropy Evolution ---
p_entropy_evolution <- ggplot(lat_entropy_plot_data, aes(x = trial_block, y = mean_entropy, 
                                                          color = model, fill = model)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  geom_ribbon(aes(ymin = mean_entropy - se_entropy, ymax = mean_entropy + se_entropy), 
              alpha = 0.2, color = NA) +
  scale_color_manual(values = model_colors, name = "Model") +
  scale_fill_manual(values = model_colors, name = "Model") +
  scale_x_continuous(breaks = seq(40, 200, by = 40)) +
  labs(
    title = "Latency Entropy Evolution",
    x = "Trial Block",
    y = "Latency Entropy",
    color = "Model",
    fill = "Model"
  ) +
  theme_paper()

# --- Latency Entropy Boxplot (final trial block) ---
df_sim_final <- df_sim_raw %>% filter(trial_block == 200, !is.na(lat_entropy))

p_entropy_box <- ggplot(df_sim_final, aes(x = model, y = lat_entropy, fill = model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 21, outlier.size = 1.5, linewidth = 0.4) +
  geom_jitter(width = 0.2, alpha = 0.3, size = 1.5) +
  scale_fill_manual(values = model_colors, name = "Model") +
  labs(
    title = "Latency Entropy at 200 Trials",
    x = "Model",
    y = "Latency Entropy",
    fill = "Model"
  ) +
  theme_paper() +
  theme(legend.position = "none")

# Keep titles for individual plots
p_center_scatter_title <- p_center_scatter + labs(title = "Stimulus Center vs PSE")
p_center_box_notitle <- p_center_box + labs(title = NULL)
p_spread_scatter_title <- p_spread_scatter + labs(title = "Stimulus Spread vs JND")
p_spread_box_notitle <- p_spread_box + labs(title = NULL)
p_ai_evolution_title <- p_ai_evolution + labs(title = NULL)
p_entropy_evolution_title <- p_entropy_evolution + labs(title = NULL)
p_entropy_box_notitle <- p_entropy_box + labs(title = NULL)

# Create section title plots
title_stimuli <- ggplot() + 
  annotate("text", x = 0.5, y = 0.5, label = "Stimuli Metrics", 
           size = 6, fontface = "bold", color = "black") +
  theme_void()

title_ai <- ggplot() + 
  annotate("text", x = 0.5, y = 0.5, label = "|AI| and AI Evolution in ABS1", 
           size = 6, fontface = "bold", color = "black") +
  theme_void()

title_entropy <- ggplot() + 
  annotate("text", x = 0.5, y = 0.5, label = "Latency Entropy", 
           size = 6, fontface = "bold", color = "black") +
  theme_void()

# Create left section (Stimuli metrics) with section title
p_left <- (title_stimuli) / 
          ((p_center_scatter_title | p_center_box_notitle) / 
           (p_spread_scatter_title | p_spread_box_notitle)) +
  plot_layout(heights = c(0.08, 1))

# Create middle section (AI evolution) with section title
p_middle <- (title_ai) / 
            (p_ai_evolution_title) +
  plot_layout(heights = c(0.08, 1))

# Create right section (Entropy) with section title - only evolution plot, no boxplot
p_right <- (title_entropy) / 
           (p_entropy_evolution_title) +
  plot_layout(heights = c(0.08, 1))

# Combine Figure 6 with proper layout
p_figure6 <- p_left | p_middle | p_right +
  plot_annotation(
    title = "Analysis Metrics",
    theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5, margin = margin(b = 10)))
  )

ggsave(file.path(paper_plots_dir, "Figure6.tif"),
       p_figure6, width = 48, height = 14, dpi = 300, units = "cm",
       compression = "lzw")
cat("✓ Saved: Figure6.tif\n")

# ============================================================================== =
# REAL DATA FIGURES
# ============================================================================== =

cat("\n=== REAL DATA FIGURES ===\n\n")

# ============================================================================== =
# Figure 7: Latency Entropy Evolution ====
# ============================================================================== =

cat("Creating Figure 7: Latency Entropy Evolution...\n")

entropy_summary <- data_real_clean %>%
  group_by(modality, algorithm, n_trials) %>%
  summarise(
    mean_entropy = mean(lat_entropy, na.rm = TRUE),
    se_entropy = sd(lat_entropy, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

p_real_entropy <- ggplot() +
  geom_line(data = data_real_clean,
            aes(x = n_trials, y = lat_entropy, group = subj, color = algorithm),
            alpha = 0.2, linewidth = 0.3) +
  geom_line(data = entropy_summary,
            aes(x = n_trials, y = mean_entropy, color = algorithm),
            linewidth = 1.5) +
  geom_point(data = entropy_summary,
             aes(x = n_trials, y = mean_entropy, color = algorithm, shape = algorithm),
             size = 3) +
  facet_wrap(~ modality) +
  scale_color_manual(values = colors_algorithm) +
  scale_shape_manual(values = c(16, 17)) +
  labs(
    title = "Latency Entropy Across Trials",
    x = "Number of Trials",
    y = "Entropy (bits)",
    color = "Algorithm",
    shape = "Algorithm"
  ) +
  theme_paper() +
  theme(legend.position = "bottom")

ggsave(file.path(paper_plots_dir, "Figure7.tif"),
       p_real_entropy, width = 14, height = 7, dpi = 300, units = "cm",
       compression = "lzw")
cat("✓ Saved: Figure7.tif\n")



# ============================================================================== =
# Figure 8: PSE and JND Boxplots at N=200 ====
# ============================================================================== =

cat("Figure 8: PSE and JND Boxplots...\n")

data_200 <- data_real_clean %>% filter(n_trials == 200)

p_real_pse <- ggplot(data_200, aes(x = algorithm, y = pse, fill = algorithm)) +
  geom_line(aes(group = subj), alpha = 0.3, linewidth = 0.5, color = "gray60") +
  geom_boxplot(alpha = 0.5, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  facet_wrap(~ modality) +
  scale_fill_manual(values = colors_algorithm) +
  labs(
    title = "PSE Distribution at N=200 Trials",
    x = "Algorithm",
    y = "PSE (ms)"
  ) +
  theme_paper() +
  theme(legend.position = "none")

p_real_jnd <- ggplot(data_200, aes(x = algorithm, y = jnd, fill = algorithm)) +
  geom_line(aes(group = subj), alpha = 0.3, linewidth = 0.5, color = "gray60") +
  geom_boxplot(alpha = 0.5, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  facet_wrap(~ modality) +
  scale_fill_manual(values = colors_algorithm) +
  labs(
    title = "JND Distribution at N=200 Trials",
    x = "Algorithm",
    y = "JND (ms)"
  ) +
  theme_paper() +
  theme(legend.position = "none")

p_real_fig1 <- p_real_pse | p_real_jnd +
  plot_annotation(
    title = "Real Data: PSE and JND at Final Trial Block",
    theme = theme(plot.title = element_text(face = "bold", size = 13, hjust = 0.5))
  )

ggsave(file.path(paper_plots_dir, "Figure8.tif"),
       p_real_fig1, width = 16, height = 7, dpi = 300, units = "cm",
       compression = "lzw")
cat("✓ Saved: Figure8.tif\n")



# ============================================================================== =
# Figure 9: Stability Points ====
# ============================================================================== =

cat("Creating Figure 8: Stability Points...\n")

stability_data <- data_real_clean %>%
  group_by(subj, modality, algorithm) %>%
  arrange(n_trials) %>%
  mutate(
    pse_final = pse[n_trials == 200],
    jnd_final = jnd[n_trials == 200],
    pse_diff_pct = abs(pse - pse_final) / abs(pse_final) * 100,
    jnd_diff_pct = abs(jnd - jnd_final) / abs(jnd_final) * 100,
    pse_stable = pse_diff_pct < 10,
    jnd_stable = jnd_diff_pct < 10
  ) %>%
  summarise(
    pse_stability_point = ifelse(any(pse_stable), min(n_trials[pse_stable]), 200),
    jnd_stability_point = ifelse(any(jnd_stable), min(n_trials[jnd_stable]), 200),
    .groups = "drop"
  )

p_real_pse_stab <- ggplot(stability_data, aes(x = algorithm, y = pse_stability_point, fill = algorithm)) +
  # Add connecting lines between same subject's two algorithms
  geom_line(aes(group = subj), alpha = 0.3, linewidth = 0.5, color = "gray60") +
  geom_boxplot(alpha = 0.7, outlier.shape = NA, width = 0.6) +
  geom_point(position = position_jitter(width = 0.08, height = 0, seed = 42), 
             alpha = 0.8, size = 2.5) +
  facet_wrap(~ modality) +
  scale_fill_manual(values = colors_algorithm) +
  labs(
    title = "PSE Stability Point",
    x = "Algorithm",
    y = "Stability Point (n_trials)"
  ) +
  theme_paper() +
  theme(legend.position = "none")

p_real_jnd_stab <- ggplot(stability_data, aes(x = algorithm, y = jnd_stability_point, fill = algorithm)) +
  # Add connecting lines between same subject's two algorithms
  geom_line(aes(group = subj), alpha = 0.3, linewidth = 0.5, color = "gray60") +
  geom_boxplot(alpha = 0.5, outlier.shape = NA) +
  geom_jitter(width = 0.08, height = 0, alpha = 0.8, size = 2) +
  facet_wrap(~ modality) +
  scale_fill_manual(values = colors_algorithm) +
  labs(
    title = "JND Stability Point",
    x = "Algorithm",
    y = "Stability Point (n_trials)"
  ) +
  theme_paper() +
  theme(legend.position = "none")

p_real_fig2 <- (p_real_pse_stab / p_real_jnd_stab) +
  plot_annotation(
    title = "Real Data: Stability Points",
    theme = theme(plot.title = element_text(face = "bold", size = 13, hjust = 0.5))
  )

ggsave(file.path(paper_plots_dir, "Figure9.tif"),
       p_real_fig2, width = 14, height = 12, dpi = 300, units = "cm",
       compression = "lzw")
cat("✓ Saved: Figure9.tif\n")



# ============================================================================== =
# Figure 10: AUC Combined Analysis (Real Data) ====
# ============================================================================== =

cat("Creating Figure 10: AUC Combined Analysis...\n")

# Calculate AUC data
auc_data <- data_real_clean %>%
  group_by(subj, modality, algorithm) %>%
  arrange(n_trials) %>%
  mutate(
    pse_final = first(pse[n_trials == 200]),
    jnd_final = first(jnd[n_trials == 200]),
    pse_deviation = abs(pse - pse_final),
    jnd_deviation = abs(jnd - jnd_final)
  ) %>%
  filter(n_trials < 200) %>%
  summarise(
    auc_pse = sum(pse_deviation * 20),
    auc_jnd = sum(jnd_deviation * 20),
    .groups = "drop"
  )

# Prepare data for plotting
auc_long <- auc_data %>%
  pivot_longer(cols = c(auc_pse, auc_jnd), 
               names_to = "parameter", 
               values_to = "auc") %>%
  mutate(
    parameter_label = case_when(
      parameter == "auc_pse" ~ "PSE",
      parameter == "auc_jnd" ~ "JND"
    ),
    algorithm_label = case_when(
      algorithm == "Adaptive" ~ "AD",
      algorithm == "Fixed" ~ "FX"
    )
  )

p9 <- ggplot(auc_long, aes(x = algorithm, y = auc, fill = algorithm)) +
  # Add connecting lines between same subject's two algorithms
  geom_line(aes(group = subj), alpha = 0.3, linewidth = 0.5, color = "gray60") +
  geom_violin(alpha = 0.7, trim = FALSE) +
  geom_boxplot(width = 0.2, alpha = 0.8, outlier.shape = NA) +
  geom_point(position = position_jitter(width = 0.08, height = 0, seed = 42), 
             alpha = 0.8, size = 2) +
  stat_summary(fun = mean, geom = "crossbar", width = 0.3, color = "red", linewidth = 0.8) +
  facet_wrap(~ parameter_label, scales = "free_y", ncol = 2) +
  scale_fill_manual(values = colors_algorithm) +
  labs(
    title = "AUC Distribution",
    x = "Algorithm",
    y = "Area Under Curve",
    fill = "Algorithm"
  ) +
  theme(legend.position = "none")

# Calculate algorithm differences using GROUP MEANS
auc_group_means <- auc_data %>%
  group_by(modality, algorithm) %>%
  summarise(
    auc_pse_mean = mean(auc_pse, na.rm = TRUE),
    auc_jnd_mean = mean(auc_jnd, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(names_from = algorithm, values_from = c(auc_pse_mean, auc_jnd_mean)) %>%
  mutate(
    pse_advantage = (auc_pse_mean_Fixed - auc_pse_mean_Adaptive) / auc_pse_mean_Fixed * 100,
    jnd_advantage = (auc_jnd_mean_Fixed - auc_jnd_mean_Adaptive) / auc_jnd_mean_Fixed * 100
  ) %>%
  dplyr::select(modality, pse_advantage, jnd_advantage) %>%
  pivot_longer(cols = c(pse_advantage, jnd_advantage),
               names_to = "parameter",
               values_to = "percent_advantage") %>%
  mutate(
    parameter_label = case_when(
      parameter == "pse_advantage" ~ "PSE",
      parameter == "jnd_advantage" ~ "JND"
    )
  )

mean_advantages <- auc_group_means %>%
  group_by(parameter_label) %>%
  summarise(
    mean_advantage = mean(percent_advantage),
    .groups = "drop"
  )

p10 <- ggplot(auc_group_means, aes(x = parameter_label, y = percent_advantage)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_col(aes(fill = modality), position = position_dodge(width = 0.7), alpha = 0.8, width = 0.6) +
  geom_text(aes(label = paste0("+", round(percent_advantage, 1), "%"), group = modality),
            position = position_dodge(width = 0.7), vjust = -0.8, size = 3.5, fontface = "bold") +
  geom_crossbar(data = mean_advantages, 
                aes(y = mean_advantage, ymin = mean_advantage, ymax = mean_advantage),
                width = 0.8, color = "red", linewidth = 1, alpha = 0.7) +
  geom_text(data = mean_advantages,
            aes(y = mean_advantage, label = paste0("Overall: +", round(mean_advantage, 1), "%")),
            vjust = 3.5, color = "red", size = 3.5, fontface = "bold") +
  scale_fill_manual(values = colors_modality) +
  scale_y_continuous(limits = c(0, 75)) +
  labs(
    title = "AUC Advantage",
    x = "Parameter",
    y = "Percent Improvement (%)",
    fill = "Modality"
  ) +
  theme(legend.position = "bottom",
        plot.title = element_text(margin = margin(b = 15)))


# ============================================================================== =
# Helper Functions for Convergence Analysis ====
# ============================================================================== =

# Function to calculate remaining deviations from final values
calculate_remaining_deviations <- function(data) {
  data %>%
    group_by(subj, modality, algorithm) %>%
    arrange(n_trials) %>%
    mutate(
      # Get final values (at n_trials = 200, or last available if missing)
      pse_final = ifelse(any(n_trials == 200), 
                         first(pse[n_trials == 200]), 
                         last(pse)),
      jnd_final = ifelse(any(n_trials == 200), 
                         first(jnd[n_trials == 200]), 
                         last(jnd)),
      
      # Calculate remaining deviations
      pse_remaining = abs(pse - pse_final),
      jnd_remaining = abs(jnd - jnd_final),
      
      # Calculate percentage converged (protect against division by zero)
      pse_pct_converged = ifelse(abs(pse_final) < 1e-6, 
                                100, 
                                pmax(0, (1 - pse_remaining/abs(pse_final)) * 100)),
      jnd_pct_converged = ifelse(abs(jnd_final) < 1e-6, 
                                100, 
                                pmax(0, (1 - jnd_remaining/abs(jnd_final)) * 100))
    ) %>%
    ungroup()
}

# Function to summarize convergence data at group level
summarize_convergence <- function(data) {
  data %>%
    group_by(modality, algorithm, n_trials) %>%
    summarise(
      # Percentage convergence summaries
      mean_pse_pct = mean(pse_pct_converged, na.rm = TRUE),
      se_pse_pct = sd(pse_pct_converged, na.rm = TRUE) / sqrt(n()),
      mean_jnd_pct = mean(jnd_pct_converged, na.rm = TRUE),
      se_jnd_pct = sd(jnd_pct_converged, na.rm = TRUE) / sqrt(n()),
      
      .groups = "drop"
    ) %>%
    mutate(
      algorithm_label = case_when(
        algorithm == "Adaptive" ~ "AD",
        algorithm == "Fixed" ~ "FX"
      )
    )
}

# Calculate convergence data using helper functions
convergence_data_raw <- data_real_clean %>%
  filter(n_trials <= 200) %>%
  calculate_remaining_deviations()

convergence_summary <- convergence_data_raw %>%
  filter(n_trials < 200) %>%
  summarize_convergence()

# Prepare data for plotting (long format for both PSE and JND)
percentage_long <- convergence_data_raw %>%
  filter(n_trials < 200) %>%
  dplyr::select(subj, modality, algorithm, n_trials, pse_pct_converged, jnd_pct_converged) %>%
  pivot_longer(cols = c(pse_pct_converged, jnd_pct_converged),
               names_to = "parameter",
               values_to = "pct_converged") %>%
  mutate(
    parameter_label = case_when(
      parameter == "pse_pct_converged" ~ "PSE",
      parameter == "jnd_pct_converged" ~ "JND"
    ),
    algorithm_label = case_when(
      algorithm == "Adaptive" ~ "AD",
      algorithm == "Fixed" ~ "FX"
    )
  )

# Summary data for group means
percentage_summary <- convergence_summary %>%
  dplyr::select(modality, algorithm, n_trials, algorithm_label, 
         mean_pse_pct, se_pse_pct,
         mean_jnd_pct, se_jnd_pct) %>%
  pivot_longer(cols = c(mean_pse_pct, mean_jnd_pct),
               names_to = "parameter",
               values_to = "mean_pct") %>%
  mutate(
    parameter_label = case_when(
      parameter == "mean_pse_pct" ~ "PSE",
      parameter == "mean_jnd_pct" ~ "JND"
    ),
    se_pct = case_when(
      parameter == "mean_pse_pct" ~ se_pse_pct,
      parameter == "mean_jnd_pct" ~ se_jnd_pct
    )
  )

p11b_pse <- ggplot() +
  geom_hline(yintercept = 90, linetype = "dashed", color = "gray50", alpha = 0.7) +
  geom_line(data = percentage_long %>% filter(parameter_label == "PSE"),
            aes(x = n_trials, y = pct_converged, 
                group = interaction(subj, modality), 
                color = algorithm_label),
            alpha = 0.2, linewidth = 0.3) +
  geom_line(data = percentage_summary %>% filter(parameter_label == "PSE"),
            aes(x = n_trials, y = mean_pct, color = algorithm_label),
            linewidth = 1.5) +
  geom_point(data = percentage_summary %>% filter(parameter_label == "PSE"),
             aes(x = n_trials, y = mean_pct, color = algorithm_label),
             size = 2.5) +
  geom_ribbon(data = percentage_summary %>% filter(parameter_label == "PSE"),
              aes(x = n_trials, 
                  ymin = pmax(0, mean_pct - se_pct), 
                  ymax = pmin(100, mean_pct + se_pct),
                  fill = algorithm_label),
              alpha = 0.2) +
  facet_wrap(~ modality) +
  scale_color_manual(values = c("AD" = "#CC3333", "FX" = "#3333CC")) +
  scale_fill_manual(values = c("AD" = "#CC3333", "FX" = "#3333CC")) +
  scale_y_continuous(limits = c(80, 100), breaks = seq(80, 100, 5)) +
  labs(
    title = "PSE",
    x = "Number of Trials",
    y = "Percentage Converged (%)"
  ) +
  theme(legend.position = "none")

p11b_jnd <- ggplot() +
  geom_hline(yintercept = 90, linetype = "dashed", color = "gray50", alpha = 0.7) +
  geom_line(data = percentage_long %>% filter(parameter_label == "JND"),
            aes(x = n_trials, y = pct_converged, 
                group = interaction(subj, modality), 
                color = algorithm_label),
            alpha = 0.2, linewidth = 0.3) +
  geom_line(data = percentage_summary %>% filter(parameter_label == "JND"),
            aes(x = n_trials, y = mean_pct, color = algorithm_label),
            linewidth = 1.5) +
  geom_point(data = percentage_summary %>% filter(parameter_label == "JND"),
             aes(x = n_trials, y = mean_pct, color = algorithm_label),
             size = 2.5) +
  geom_ribbon(data = percentage_summary %>% filter(parameter_label == "JND"),
              aes(x = n_trials, 
                  ymin = pmax(0, mean_pct - se_pct), 
                  ymax = pmin(100, mean_pct + se_pct),
                  fill = algorithm_label),
              alpha = 0.2) +
  facet_wrap(~ modality) +
  scale_color_manual(values = c("AD" = "#CC3333", "FX" = "#3333CC")) +
  scale_fill_manual(values = c("AD" = "#CC3333", "FX" = "#3333CC")) +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 20)) +
  labs(
    title = "JND",
    x = "Number of Trials",
    y = "Percentage Converged (%)",
    color = "Algorithm",
    fill = "Algorithm"
  ) +
  theme(legend.position = "bottom")

p11b <- (p11b_pse / p11b_jnd) +
  plot_layout(heights = c(1, 5)) +
  plot_annotation(
    title = "Percentage Converged"
  )

# Combine into Figure 10 (legacy format)
p_figure10 <- (((p9 / p10) + plot_layout(heights = c(0.5, 0.5))) | p11b) +
  plot_layout(widths = c(1, 1.5)) +
  plot_annotation(
    title = "Real Data: AUC Analysis and Convergence",
    theme = theme(plot.title = element_text(face = "bold", size = 13, hjust = 0.5))
  )

ggsave(file.path(paper_plots_dir, "Figure10.tif"),
       p_figure10, width = 22, height = 16, dpi = 300, units = "cm",
       compression = "lzw")
cat("✓ Saved: Figure10.tif\n")


cat("\n================================================================================\n")
cat("PUBLICATION FIGURES COMPLETE\n")
cat("================================================================================\n")

cat("SIMULATION DATA FIGURES:\n")
cat("  Figure4.tif (Models Performance - Accuracy + Convergence)\n")
cat("  Figure6.tif (Analysis Metrics - Stimulus + Asymmetry + Entropy)\n\n")

cat("REAL DATA FIGURES:\n")
cat("  Figure7.tif (Latency Entropy Evolution)\n")
cat("  Figure8.tif (PSE and JND at N=200)\n")
cat("  Figure9.tif (Stability Points)\n")
cat("  Figure10.tif (AUC Analysis and Convergence)\n\n")

