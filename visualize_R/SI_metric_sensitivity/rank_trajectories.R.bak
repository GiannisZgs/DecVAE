#' Metric sensitivity (SI): rank position of each ranked model under every condition, one panel per
#' aggregation (Flat, Flat without MI and GCN, 4-family, 3-family). Family A line style (SI Fig. 12).
#' Plots data/metric_sensitivity/rank_trajectories.csv and models.csv, written by
#' scripts/post-training/metric_sensitivity_real.py. One PNG per aggregation.

library(vscDebugger)
library(ggplot2)
library(dplyr)
library(readr)
library(viridis)

plot_font_family <- "Arial"
plot_title_size <- 28
title_font_face <- "plain"
axis_title_size <- 28
axis_text_size <- 22
axis_text_x_size <- 18  # many condition labels
legend_title_size <- 20
legend_text_size <- 18
legend_font_face <- "plain"
line_size <- 1.2
point_size <- 2.5

yellow_block_threshold <- 1.0

# Load data from
load_dir <- file.path('..', 'data', 'metric_sensitivity')

# Save data at
save_dir <- file.path('..', 'supplementary_figures', 'SI_metric_sensitivity', 'rank_trajectories')
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

models <- read_csv(file.path(load_dir, "models.csv"), show_col_types = FALSE) %>% arrange(order)
positions <- read_csv(file.path(load_dir, "rank_trajectories.csv"), show_col_types = FALSE)

ranked <- models %>% filter(model %in% positions$model)
colors <- setNames(viridis(n = nrow(ranked), option = "turbo", end = yellow_block_threshold), ranked$model)
shapes <- setNames(rep(c(15:18, 7:14, 0:6), length.out = nrow(ranked)), ranked$model)
labels <- setNames(ranked$label, ranked$model)
n_models <- nrow(ranked)

for (agg in unique(positions$aggregation[order(positions$aggregation_order)])) {
  df <- positions %>% filter(aggregation == agg) %>%
    mutate(condition = factor(condition, levels = unique(condition[order(condition_order)])),
           model = factor(model, levels = ranked$model))
  p <- ggplot(df, aes(x = condition, y = position, color = model, shape = model, group = model)) +
    geom_line(linewidth = line_size, alpha = 0.8) +
    geom_point(size = point_size, alpha = 0.9) +
    scale_color_manual(values = colors, labels = labels, name = "") +
    scale_shape_manual(values = shapes, labels = labels, name = "") +
    scale_y_reverse(breaks = seq_len(n_models), limits = c(n_models + 0.5, 0.5), expand = expansion(mult = c(0, 0))) +
    labs(title = "", x = "", y = "Rank position (1 is best)") +
    theme_minimal(base_size = 14, base_family = plot_font_family) +
    theme(
      plot.title = element_text(size = plot_title_size, face = title_font_face),
      axis.title = element_text(size = axis_title_size),
      axis.text = element_text(size = axis_text_size),
      axis.text.x = element_text(angle = 45, hjust = 1, size = axis_text_x_size),
      legend.title = element_text(size = legend_title_size, face = legend_font_face),
      legend.text = element_text(size = legend_text_size),
      panel.grid.minor = element_blank(),
      legend.position = "right"
    )
  save_path <- file.path(save_dir, paste0("rank_trajectories_", gsub("[^A-Za-z0-9]+", "_", agg), ".png"))
  ggsave(filename = save_path, plot = p, width = 14, height = 8, dpi = 600, bg = "white")
  cat("Saved plot to:", save_path, "\n")
}
