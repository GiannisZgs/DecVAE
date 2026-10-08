#' Subspace analysis figure (SimVowels configurations): selectivity of the prediction and swap
#' matrices of the DecVAE subspaces against random partitions, and the alignment MAE of the swap
#' matrix with the generator, per configuration (EWT and FD at beta 0, 0.1, 1; EMD; VMD).
#' Plots data/subspace/config_summary.csv, written by scripts/post-training/subspace_analysis.py.
#' Error bars: std over seeds where a configuration has several, over folds otherwise (std_over column).

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
yellow_block_threshold <- 1.0

display_decomp_names <- c(ewt = "EWT", fd = "FD", emd = "EMD", vmd = "VMD")
display_partition_names <- c(own = "DecVAE subspaces", null = "Random partitions")
matrix_file <- c(P = "prediction", dT = "swap")

# Load data from
load_dir <- file.path('..', 'data', 'subspace')

# Save data at
save_dir <- file.path('..', 'supplementary_figures', 'SI_subspace', 'subspace_configs')
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

save_plot <- function(p, name, width, height) {
  save_path <- file.path(save_dir, paste0(name, ".png"))
  ggsave(filename = save_path, plot = p, width = width, height = height, dpi = 600, bg = "white")
  cat("Saved plot to:", save_path, "\n")
}

summary <- read_csv(file.path(load_dir, "config_summary.csv"), show_col_types = FALSE) %>%
  filter(dataset == "sim_vowels", run == "main") %>%
  mutate(config_display = paste0(display_decomp_names[decomposition], " β = ", format(beta)),
         partition_display = factor(display_partition_names[partition], levels = display_partition_names))
config_levels <- summary %>% distinct(order, config_display) %>% arrange(order) %>% pull(config_display)
summary <- summary %>% mutate(config_display = factor(config_display, levels = config_levels))
print(summary %>% distinct(config_display, partition, std_over, n))

colors <- viridis(n = 2, option = "turbo", end = yellow_block_threshold)
names(colors) <- display_partition_names

for (m in c("P", "dT")) {
  df <- summary %>% filter(quantity == "selectivity", matrix == m)
  p <- ggplot(df, aes(x = config_display, y = mean, fill = partition_display)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = mean - std, ymax = mean + std), position = position_dodge(width = 0.8),
                  width = 0.5, linewidth = 1.2, color = "black", na.rm = TRUE) +
    scale_fill_manual(values = colors, name = "") +
    scale_y_continuous(breaks = seq(0, 1, by = 0.1), labels = label_number(accuracy = 0.1),
                       expand = expansion(mult = c(0, 0.05))) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(title = "", x = "", y = paste("Selectivity of", matrix_file[[m]])) +
    family_a_theme
  save_plot(p, paste0("selectivity_", matrix_file[[m]]), 14, 8)
}

df <- summary %>% filter(quantity == "alignment_mae", matrix == "dT")
if (nrow(df) > 0) {
  p <- ggplot(df, aes(x = config_display, y = mean)) +
    geom_col(width = 0.7, fill = colors[[1]]) +
    geom_errorbar(aes(ymin = mean - std, ymax = mean + std), width = 0.5, linewidth = 1.2, color = "black", na.rm = TRUE) +
    scale_y_continuous(breaks = pretty_breaks(n = 6), labels = label_number(accuracy = 0.01),
                       expand = expansion(mult = c(0, 0.05))) +
    labs(title = "", x = "", y = "Alignment MAE of swap") +
    family_a_theme
  save_plot(p, "alignment_mae_swap", 12, 8)
}
