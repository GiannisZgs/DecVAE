#' SI Figure (decomposition sensitivity): metrics against the number of components C, on SimVowels and SimCoupled.
#' Plots data/decomp_sensitivity/sweep_groups_long.csv, written by scripts/post-training/decomp_sensitivity_summary.py.
#' One PNG per metric. Error bars: std over the three seeds at C = 3 (the other C values are single runs).
#' Dashed vertical line: the SimVowels generator's C (3). Mutual information and Gaussian correlation are drawn
#' as 1 - value, as in SI Fig. 10. The effective number of OC subspaces per factor is drawn with the line y = C
#' (factor spread evenly over all OCs) for reference. SimCoupled's unseen (indep) split gets its own PNGs.

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
line_size <- 1.2
point_size <- 2.5
yellow_block_threshold <- 1.0
generator_C <- 3

display_dataset_names <- c(sim_vowels = "SimVowels", sim_coupled = "SimCoupled")
display_factor_names <- c(vowel = "vowel", speaker_frame = "speaker", lag = "lag", gain = "gain")
# metric key -> display name; one_minus: drawn as 1 - value; bounded: on [0, 1]
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
  "sel_OCs:dT", "Subspace selectivity (swap, OCs)", FALSE, TRUE
)
neff_matrices <- c(P = "prediction", dT = "swap")

# Load data from
load_dir <- file.path('..', 'data', 'decomp_sensitivity')

# Save data at
save_dir <- file.path('..', 'supplementary_figures', 'SI_decomp_sensitivity', 'c_sweep')
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

family_a_theme <- theme_minimal(base_size = 14, base_family = plot_font_family) +
  theme(
    plot.title = element_text(size = plot_title_size, face = title_font_face),
    axis.title = element_text(size = axis_title_size),
    axis.text = element_text(size = axis_text_size),
    legend.title = element_text(size = legend_title_size, face = legend_font_face),
    legend.text = element_text(size = legend_text_size),
    panel.grid.minor = element_blank(),
    legend.position = "right"
  )

save_plot <- function(p, name, width = 14, height = 8) {
  save_path <- file.path(save_dir, paste0(name, ".png"))
  ggsave(filename = save_path, plot = p, width = width, height = height, dpi = 600, bg = "white")
  cat("Saved plot to:", save_path, "\n")
}

groups <- read_csv(file.path(load_dir, "sweep_groups_long.csv"), show_col_types = FALSE) %>%
  filter(str_detect(group, "^C[0-9]+$"), dataset %in% names(display_dataset_names)) %>%
  mutate(dataset_display = factor(display_dataset_names[dataset], levels = display_dataset_names))
C_levels <- sort(unique(groups$C))
generator_x <- match(generator_C, C_levels)
print(groups %>% distinct(dataset, group, C, n) %>% arrange(dataset, C))

dataset_colors <- viridis(n = length(display_dataset_names), option = "turbo", end = yellow_block_threshold)
names(dataset_colors) <- display_dataset_names

for (split_prefix in c("", "indep:")) {
  for (i in seq_len(nrow(metrics))) {
    m <- metrics[i, ]
    df <- groups %>% filter(metric == paste0(split_prefix, m$metric))
    if (nrow(df) == 0) next
    if (m$one_minus) df <- df %>% mutate(mean = 1 - mean)
    p <- ggplot(df, aes(x = factor(C, levels = C_levels), y = mean, color = dataset_display, group = dataset_display)) +
      geom_vline(xintercept = generator_x, linetype = "dashed", linewidth = 0.8, color = "grey40") +
      geom_line(linewidth = line_size, alpha = 0.8) +
      geom_point(size = point_size, alpha = 0.9) +
      geom_errorbar(aes(ymin = mean - std, ymax = mean + std), width = 0.2, linewidth = line_size, na.rm = TRUE) +
      scale_color_manual(values = dataset_colors, name = "Dataset", drop = TRUE) +
      labs(title = "", x = "Number of components C", y = m$display) +
      family_a_theme
    if (m$bounded) {
      p <- p + scale_y_continuous(breaks = seq(0, 1, by = 0.1), labels = label_number(accuracy = 0.1),
                                  expand = expansion(mult = c(0, 0.05))) +
        coord_cartesian(ylim = c(0, 1))
    }
    save_plot(p, paste0("c_sweep_", str_replace_all(paste0(split_prefix, m$metric), "[^A-Za-z0-9]+", "_")))
  }

  # Effective number of OC subspaces carrying each factor, against C (fully spread)
  for (mat in names(neff_matrices)) {
    df <- groups %>% filter(str_starts(metric, fixed(paste0(split_prefix, "n_eff:", mat, ":")))) %>%
      mutate(factor_name = str_remove(metric, fixed(paste0(split_prefix, "n_eff:", mat, ":"))),
             series = paste0(dataset_display, ": ", display_factor_names[factor_name]))
    if (nrow(df) == 0) next
    series_levels <- unique(df$series)
    series_colors <- viridis(n = length(series_levels), option = "turbo", end = yellow_block_threshold)
    names(series_colors) <- series_levels
    reference <- tibble(C = C_levels, mean = C_levels)
    p <- ggplot(df, aes(x = factor(C, levels = C_levels), y = mean)) +
      geom_vline(xintercept = generator_x, linetype = "dashed", linewidth = 0.8, color = "grey40") +
      geom_line(data = reference, aes(group = 1), linetype = "dotted", linewidth = line_size, color = "grey40") +
      geom_line(aes(color = series, group = series), linewidth = line_size, alpha = 0.8) +
      geom_point(aes(color = series), size = point_size, alpha = 0.9) +
      geom_errorbar(aes(ymin = mean - std, ymax = mean + std, color = series), width = 0.2, linewidth = line_size,
                    na.rm = TRUE) +
      scale_color_manual(values = series_colors, name = "Factor") +
      scale_y_continuous(breaks = pretty_breaks(n = 6), labels = label_number(accuracy = 0.1),
                         expand = expansion(mult = c(0, 0.05))) +
      expand_limits(y = 0) +
      labs(title = "", x = "Number of components C", y = paste("Effective OCs per factor,", neff_matrices[[mat]])) +
      family_a_theme
    save_plot(p, paste0("c_sweep_", str_replace_all(paste0(split_prefix, "n_eff_", mat), "[^A-Za-z0-9]+", "_")))
  }
}
