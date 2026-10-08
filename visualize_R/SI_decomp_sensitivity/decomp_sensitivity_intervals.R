#' SI Figure (decomposition sensitivity): FD peak-search interval variants on SimVowels at C = 3.
#' Plots data/decomp_sensitivity/sweep_runs_long.csv and sweep_groups_long.csv, written by
#' scripts/post-training/decomp_sensitivity_summary.py.
#' One PNG per metric. Bars: group mean; points: individual runs; error bars: std over the three seeds of the
#' power-law base intervals and over the three random draws (the other variants are single runs).
#' Mutual information and Gaussian correlation are drawn as 1 - value, as in SI Fig. 10.

library(vscDebugger)
library(ggplot2)
library(dplyr)
library(readr)
library(stringr)
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
point_size <- 2.5
yellow_block_threshold <- 1.0

display_group_names <- c(C3 = "Power law (3 seeds)", lin = "Linear", int3 = "3 intervals", int8 = "8 intervals",
                         rand = "Random (3 draws)")
metrics <- tribble(
  ~metric, ~display, ~one_minus, ~bounded,
  "IRS", "Robustness (IRS)", FALSE, TRUE,
  "mutual_info_score", "1 - Mutual Info.", TRUE, TRUE,
  "gaussian_total_correlation_norm", "1 - Gaussian Correlation", TRUE, TRUE,
  "disentanglement", "Disentanglement", FALSE, TRUE,
  "completeness", "Completeness", FALSE, TRUE,
  "modularity_score", "Modularity", FALSE, TRUE,
  "informativeness_test", "Informativeness", FALSE, TRUE,
  "explicitness_score_test", "Explicitness", FALSE, TRUE,
  "sel_OCs:P", "Subspace selectivity (prediction, OCs)", FALSE, TRUE,
  "sel_OCs:dT", "Subspace selectivity (swap, OCs)", FALSE, TRUE,
  "n_eff:P:vowel", "Effective OCs, vowel (prediction)", FALSE, FALSE,
  "n_eff:P:speaker_frame", "Effective OCs, speaker (prediction)", FALSE, FALSE
)

# Load data from
load_dir <- file.path('..', 'data', 'decomp_sensitivity')

# Save data at
save_dir <- file.path('..', 'supplementary_figures', 'SI_decomp_sensitivity', 'intervals')
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
    legend.position = "none"
  )

save_plot <- function(p, name, width = 12, height = 8) {
  save_path <- file.path(save_dir, paste0(name, ".png"))
  ggsave(filename = save_path, plot = p, width = width, height = height, dpi = 600, bg = "white")
  cat("Saved plot to:", save_path, "\n")
}

with_display <- function(df) {
  df %>% filter(dataset == "sim_vowels", group %in% names(display_group_names)) %>%
    mutate(group_display = factor(display_group_names[group], levels = display_group_names))
}
runs <- with_display(read_csv(file.path(load_dir, "sweep_runs_long.csv"), show_col_types = FALSE))
groups <- with_display(read_csv(file.path(load_dir, "sweep_groups_long.csv"), show_col_types = FALSE))
print(groups %>% distinct(group, n))

group_colors <- viridis(n = length(display_group_names), option = "turbo", end = yellow_block_threshold)
names(group_colors) <- display_group_names

for (i in seq_len(nrow(metrics))) {
  m <- metrics[i, ]
  g <- groups %>% filter(metric == m$metric)
  r <- runs %>% filter(metric == m$metric)
  if (nrow(g) == 0) next
  if (m$one_minus) {
    g <- g %>% mutate(mean = 1 - mean)
    r <- r %>% mutate(value = 1 - value)
  }
  p <- ggplot(g, aes(x = group_display, y = mean, fill = group_display)) +
    geom_col(width = 0.7) +
    geom_errorbar(aes(ymin = mean - std, ymax = mean + std), width = 0.5, linewidth = 1.2, color = "black", na.rm = TRUE) +
    geom_point(data = r, aes(x = group_display, y = value), inherit.aes = FALSE, size = point_size, color = "black",
               position = position_jitter(width = 0.08, height = 0, seed = 0)) +
    scale_fill_manual(values = group_colors, drop = TRUE) +
    labs(title = "", x = "", y = m$display) +
    family_a_theme
  if (m$bounded) {
    p <- p + scale_y_continuous(breaks = seq(0, 1, by = 0.1), labels = label_number(accuracy = 0.1),
                                expand = expansion(mult = c(0, 0.05))) +
      coord_cartesian(ylim = c(0, 1))
  } else {
    p <- p + scale_y_continuous(breaks = pretty_breaks(n = 6), labels = label_number(accuracy = 0.1),
                                expand = expansion(mult = c(0, 0.05)))
  }
  save_plot(p, paste0("intervals_", str_replace_all(m$metric, "[^A-Za-z0-9]+", "_")))
}
