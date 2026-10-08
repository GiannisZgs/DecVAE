#' SI Figure (decomposition sensitivity): components FD keeps per frame (K) against the number of components C.
#' Plots data/decomp_sensitivity/component_counts_distribution.csv, component_counts_by_C.csv and
#' component_counts_summary.csv, written by scripts/post-training/fd_component_counts.py.
#' Top row: share of frames per K, per dataset (bars: base peak-search intervals; lines: the SimVowels interval
#' variants; dashed: generator C). Bottom row: for each C, share of frames grouped (K > C), exact (K = C),
#' padded with empty components (0 < K < C) or without a component (K = 0).

library(vscDebugger)
library(tidyverse)
library(patchwork)
library(viridis)

# Style and font parameters
plot_font_family <- "Arial"
plot_background_color <- "#F4F5F1"
plot_text_color <- "black"

# Axis titles
axis_title_size <- 22
axis_title_face <- "plain"
axis_title_margin <- margin(t = 15, r = 15, b = 15, l = 15)

# Axis tick labels
axis_text_size <- 21
axis_text_face <- "plain"
axis_text_color <- "black"

# Facet text
facet_text_size <- 40
facet_text_face <- "plain"

# Legend text
legend_text_size <- 20
legend_title_size <- 22

# Plot margins and spacing
plot_margin <- margin(15, 15, 15, 15)
panel_spacing <- unit(4, "lines")

# Lines and points
line_size <- 1.2
point_size <- 2.5
generator_line_width <- 1

fig_width <- 20
row_height <- 6
fig_dpi <- 600

display_dataset_names <- c(sim_vowels = "SimVowels", sim_coupled = "SimCoupled", timit = "TIMIT")
display_variant_names <- c(base = "Power law (base)", lin = "Linear", int3 = "3 intervals", int8 = "8 intervals",
                           rand0 = "Random 1", rand1 = "Random 2", rand2 = "Random 3")
display_outcome_names <- c(exact = "Exact (K = C)", grouped = "Grouped (K > C)",
                           padded = "Padded (0 < K < C)", none = "No component (K = 0)")

# Load data from
load_dir <- file.path('..', 'data', 'decomp_sensitivity')

# Save data at
save_dir <- file.path('..', 'supplementary_figures', 'SI_decomp_sensitivity', 'component_counts')
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

base_theme <- theme_minimal() +
  theme(
    text = element_text(family = plot_font_family),
    plot.background = element_rect(color = plot_background_color, fill = plot_background_color),
    panel.background = element_rect(fill = NA, color = NA),
    panel.grid.minor = element_blank(),
    plot.margin = plot_margin,
    panel.spacing = panel_spacing,
    strip.text = element_text(size = facet_text_size, face = facet_text_face, color = plot_text_color,
                              family = plot_font_family),
    axis.text = element_text(size = axis_text_size, face = axis_text_face, color = axis_text_color,
                             family = plot_font_family),
    axis.title = element_text(size = axis_title_size, face = axis_title_face, color = plot_text_color,
                              margin = axis_title_margin, family = plot_font_family),
    legend.text = element_text(size = legend_text_size, family = plot_font_family),
    legend.title = element_text(size = legend_title_size, family = plot_font_family),
    legend.position = "right"
  )

save_plot <- function(plot, fname, n_rows) {
  save_path <- file.path(save_dir, paste0(fname, ".png"))
  ggsave(save_path, plot, width = fig_width, height = row_height * n_rows, dpi = fig_dpi, bg = plot_background_color)
  cat("Saved plot to:", save_path, "\n")
}

dataset_levels <- function(df) {
  unname(display_dataset_names[names(display_dataset_names) %in% unique(df$dataset)])
}

dist <- read_csv(file.path(load_dir, "component_counts_distribution.csv"), show_col_types = FALSE)
by_C <- read_csv(file.path(load_dir, "component_counts_by_C.csv"), show_col_types = FALSE)
summ <- read_csv(file.path(load_dir, "component_counts_summary.csv"), show_col_types = FALSE)
levels_ds <- dataset_levels(summ)

variant_levels <- unname(display_variant_names[names(display_variant_names) %in% unique(dist$variant)])
base_bar_color <- "grey60"
variant_colors <- c(base_bar_color, viridis(n = length(variant_levels) - 1, option = "plasma", begin = 0, end = 0.8))
names(variant_colors) <- variant_levels

dist <- dist %>%
  mutate(dataset_display = factor(display_dataset_names[dataset], levels = levels_ds),
         variant_display = factor(display_variant_names[variant], levels = variant_levels))
generator <- summ %>% filter(!is.na(true_C)) %>% distinct(dataset, true_C) %>%
  mutate(dataset_display = factor(display_dataset_names[dataset], levels = levels_ds))

# Top row: distribution of K
p_k <- ggplot() +
  geom_col(data = dist %>% filter(variant == "base"), aes(x = K, y = share, fill = variant_display),
           width = 0.7, alpha = 0.6) +
  geom_line(data = dist %>% filter(variant != "base"), aes(x = K, y = share, color = variant_display),
            linewidth = line_size, alpha = 0.8) +
  geom_point(data = dist %>% filter(variant != "base"), aes(x = K, y = share, color = variant_display),
             size = point_size, alpha = 0.9) +
  geom_vline(data = generator, aes(xintercept = true_C), linetype = "dashed", linewidth = generator_line_width) +
  facet_wrap(~dataset_display, nrow = 1, scales = "free_x") +
  scale_fill_manual(values = variant_colors, name = "Peak-search intervals", drop = TRUE) +
  scale_color_manual(values = variant_colors, name = "", drop = TRUE) +
  scale_x_continuous(breaks = function(lim) seq(0, ceiling(lim[2]), by = 1)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2), labels = scales::label_number(accuracy = 0.1),
                     expand = expansion(mult = c(0, 0.05))) +
  labs(x = "Components kept per frame (K)", y = "Share of frames") +
  base_theme
save_plot(p_k, "component_counts_K", 1)

# Bottom row: outcome per C, base intervals
outcome_levels <- unname(display_outcome_names)
outcome_colors <- viridis(n = length(outcome_levels), option = "plasma", begin = 0, end = 0.8)
names(outcome_colors) <- outcome_levels
by_C_base <- by_C %>% filter(variant == "base") %>%
  mutate(dataset_display = factor(display_dataset_names[dataset], levels = levels_ds),
         outcome_display = factor(display_outcome_names[outcome], levels = rev(outcome_levels)))

p_c <- ggplot(by_C_base, aes(x = factor(C), y = share, fill = outcome_display)) +
  geom_col(width = 0.7) +
  facet_wrap(~dataset_display, nrow = 1) +
  scale_fill_manual(values = outcome_colors, name = "Frames", breaks = outcome_levels) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2), labels = scales::label_number(accuracy = 0.1),
                     expand = expansion(mult = c(0, 0.05))) +
  labs(x = "Number of components C", y = "Share of frames") +
  base_theme
save_plot(p_c, "component_counts_by_C", 1)

# Composite of both rows
p_all <- p_k / p_c
save_plot(p_all, "component_counts", 2)
