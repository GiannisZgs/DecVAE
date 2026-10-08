#' Subspace analysis figure (datasets other than SimVowels): prediction and swap matrices for TIMIT,
#' IEMOCAP, VOC-ALS (and the VOC-ALS speaker-disjoint run) and SimCoupled (and its independent-factors
#' run), and the selectivity of each against random partitions. Std over seeds where a configuration has
#' several (SimCoupled FD), over CV folds otherwise (std_over column of the CSVs).
#' Plots data/subspace/subspace_matrices.csv and config_summary.csv, written by
#' scripts/post-training/subspace_analysis.py. One PNG per panel.

library(vscDebugger)
library(ggplot2)
library(dplyr)
library(readr)
library(scales)
library(viridis)

plot_font_family <- "Arial"
plot_title_size <- 28
title_font_face <- "plain"
axis_title_size <- 28
axis_text_size <- 22
legend_title_size <- 20
legend_text_size <- 18
legend_font_face <- "plain"
cell_text_size <- 7
yellow_block_threshold <- 1.0

# Fixed colour limits per matrix type, as in subspace_simvowels.R
matrix_limits <- list(P = c(0, 1), dT = c(0, 0.5))
matrix_legend <- c(P = "Prediction", dT = "Swap\nT − Tc")
matrix_file <- c(P = "prediction", dT = "swap")
display_factor_names <- c(phoneme = "Phoneme", phoneme_frame = "Phoneme", speaker_frame = "Speaker",
                          cat_emotion_frame = "Emotion", king_stage_frame = "King's stage", lag = "Lag", gain = "Gain")
display_dataset_names <- c(timit = "TIMIT", iemocap = "IEMOCAP", VOC_ALS = "VOC-ALS", sim_coupled = "SimCoupled")
display_run_names <- c(main = "", speaker_disjoint = " (speaker-disjoint)", indep = " (independent factors)")
display_partition_names <- c(own = "DecVAE subspaces", null = "Random partitions")

# Load data from
load_dir <- file.path('..', 'data', 'subspace')

# Save data at
save_dir <- file.path('..', 'supplementary_figures', 'SI_subspace', 'subspace_real')
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

family_a_theme <- theme_minimal(base_size = 14, base_family = plot_font_family) +
  theme(
    plot.title = element_text(size = plot_title_size, face = title_font_face),
    axis.title = element_text(size = axis_title_size),
    axis.text = element_text(size = axis_text_size),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.title = element_text(size = legend_title_size, face = legend_font_face),
    legend.text = element_text(size = legend_text_size),
    panel.grid.minor = element_blank(),
    legend.position = "right"
  )

plot_matrix <- function(df, y_levels, limits, legend_title) {
  df <- df %>%
    mutate(
      scaled = (pmin(pmax(mean, limits[1]), limits[2]) - limits[1]) / (limits[2] - limits[1]),
      text_color = ifelse(!is.na(scaled) & scaled > 0.6, "black", "white"),
      label = case_when(
        is.na(mean) ~ "n/a",
        !is.na(std) ~ sprintf("%.2f\n± %.2f", mean, std),
        TRUE ~ sprintf("%.2f", mean)
      )
    )
  ggplot(df, aes(x = factor_display, y = subspace, fill = mean)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = label, color = text_color), size = cell_text_size, family = plot_font_family, lineheight = 0.9) +
    scale_color_identity() +
    scale_fill_viridis(option = "plasma", limits = limits, oob = scales::squish, na.value = "grey85",
                       name = legend_title) +
    scale_y_discrete(limits = rev(y_levels)) +
    labs(title = "", x = "", y = "Subspace") +
    family_a_theme +
    theme(panel.grid = element_blank())
}

save_plot <- function(p, name, width, height) {
  save_path <- file.path(save_dir, paste0(name, ".png"))
  ggsave(filename = save_path, plot = p, width = width, height = height, dpi = 600, bg = "white")
  cat("Saved plot to:", save_path, "\n")
}

run_display <- function(dataset, run) {
  paste0(display_dataset_names[dataset], display_run_names[run])
}

matrices <- read_csv(file.path(load_dir, "subspace_matrices.csv"), show_col_types = FALSE) %>%
  filter(dataset != "sim_vowels") %>%
  mutate(factor_display = display_factor_names[factor])
if (nrow(matrices) == 0) stop("No real-dataset results in subspace_matrices.csv")

for (key in unique(paste(matrices$config, matrices$run, sep = "|"))) {
  df_run <- matrices %>% filter(paste(config, run, sep = "|") == key) %>%
    mutate(factor_display = factor(factor_display, levels = unique(factor_display)))
  subspace_levels <- unique(df_run$subspace)
  n_factors <- length(unique(df_run$factor))
  for (m in c("P", "dT")) {
    p <- plot_matrix(df_run %>% filter(matrix == m), subspace_levels, matrix_limits[[m]], matrix_legend[[m]])
    save_plot(p, paste0(matrix_file[[m]], "_", df_run$config[1], "_", df_run$run[1]),
              4 + 2.4 * n_factors, 2 + 1.6 * length(subspace_levels))
  }
}

summary <- read_csv(file.path(load_dir, "config_summary.csv"), show_col_types = FALSE) %>%
  filter(dataset != "sim_vowels", quantity == "selectivity") %>%
  mutate(run_display = run_display(dataset, run),
         partition_display = factor(display_partition_names[partition], levels = display_partition_names))
summary <- summary %>% mutate(run_display = factor(run_display, levels = unique(run_display)))
colors <- viridis(n = 2, option = "turbo", end = yellow_block_threshold)
names(colors) <- display_partition_names

for (m in c("P", "dT")) {
  p <- ggplot(summary %>% filter(matrix == m), aes(x = run_display, y = mean, fill = partition_display)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = mean - std, ymax = mean + std), position = position_dodge(width = 0.8),
                  width = 0.5, linewidth = 1.2, color = "black", na.rm = TRUE) +
    scale_fill_manual(values = colors, name = "") +
    scale_y_continuous(breaks = seq(0, 1, by = 0.1), labels = label_number(accuracy = 0.1),
                       expand = expansion(mult = c(0, 0.05))) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(title = "", x = "", y = paste("Selectivity of", matrix_file[[m]])) +
    family_a_theme
  save_plot(p, paste0("selectivity_", matrix_file[[m]]), 12, 8)
}
