#' Metric sensitivity (SI): every disentanglement metric against the latent dimensionality d, in the
#' style of the model scatter plots (Figs 2-5). Gaussian noise chance level (mean +- std over draws),
#' the dim_subset curve of each model that has one (random subsets of its own dimensions, mean +- std,
#' ending at its full ref_sub score), and the ref_sub score of every other model.
#' Plots data/metric_sensitivity/dimensionality_curves.csv and models.csv, written by
#' scripts/post-training/metric_sensitivity_real.py. One PNG per metric, legend included, and all
#' metrics combined in one figure with a single legend at the bottom.

library(vscDebugger)
library(ggplot2)
library(dplyr)
library(readr)
library(viridis)
library(ggrepel)
library(patchwork)

# Style and font parameters
plot_font_family <- "Arial"
plot_background_color <- "white"

# Colour palette
palette <- "plasma"
yellow_block_threshold <- 0.8
chance_line_color <- "gray45"
chance_band_color <- "gray85"

# Axis titles and tick labels
axis_title_size_scatter <- 30
axis_text_size_scatter <- 30

# Legend elements
legend_text_size_scatter <- 20
legend_title_size_scatter <- 22
legend_key_size_scatter <- 1.2  # in cm
max_per_row <- 5

# Geom elements
point_size_scatter <- 7
line_size <- 1.2
errorbar_width <- 0.08  # in log2 units of d
direct_labels <- TRUE
text_size_scatter <- 6
line_size_repel <- 1
line_alpha_repel <- 0.6

plot_margin_scatter <- margin(10, 10, 10, 10, "pt")

# Combined figure: every size scaled down so the panels fit one figure
combined_ncol <- 4
combined_scale <- 0.45
combined_panel_width <- 6.5   # in inches
combined_panel_height <- 5    # in inches
combined_legend_height <- 1.5 # in inches
combined_max_per_row <- 10

metric_order <- c("MI", "GCN", "DCI-D", "DCI-C", "DCI-I", "Modularity", "Explicitness", "IRS")
metric_labels <- c("MI" = "Mutual Information", "DCI-D" = "Disentanglement", "DCI-C" = "Completeness",
                   "DCI-I" = "Informativeness", "Modularity" = "Modularity", "Explicitness" = "Explicitness",
                   "IRS" = "Robustness")
gcn_labels <- c("gaussian_total_correlation" = "Gaussian Correlation",
                "gaussian_total_correlation_norm" = "Gaussian Correlation Norm.")
chance_name <- "Gaussian noise"

# Load data from
load_dir <- file.path('..', 'data', 'metric_sensitivity')

# Save data at
save_dir <- file.path('..', 'supplementary_figures', 'SI_metric_sensitivity', 'dimensionality_curves')
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

models <- read_csv(file.path(load_dir, "models.csv"), show_col_types = FALSE) %>% arrange(order)
curves <- read_csv(file.path(load_dir, "dimensionality_curves.csv"), show_col_types = FALSE)

# One colour and shape per model, fixed by its position in the models file
all_shapes <- c(15:18, 7:14, 0:6)
model_colors <- setNames(viridis(n = nrow(models), option = palette, end = yellow_block_threshold), models$model)
model_shapes <- setNames(rep(all_shapes, length.out = nrow(models)), models$model)
model_labels <- setNames(models$label, models$model)

chance <- curves %>% filter(kind == "chance") %>% mutate(std = ifelse(is.na(std), 0, std))
points <- curves %>% filter(kind != "chance", model %in% models$model) %>%
  mutate(std = ifelse(is.na(std), 0, std), model = factor(model, levels = models$model))
d_breaks <- sort(unique(chance$latent_dim))
if (length(d_breaks) == 0) d_breaks <- sort(unique(points$latent_dim))

scatter_theme <- function(k) {
  theme_minimal() +
    theme(
      legend.position = "bottom",
      legend.box = "vertical",
      legend.title = element_text(size = k * legend_title_size_scatter, face = "plain", family = plot_font_family),
      legend.text = element_text(size = k * legend_text_size_scatter, family = plot_font_family),
      legend.key.size = unit(k * legend_key_size_scatter, "cm"),
      axis.title = element_text(size = k * axis_title_size_scatter, family = plot_font_family),
      axis.text = element_text(size = k * axis_text_size_scatter, family = plot_font_family),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(fill = NA, color = "gray80"),
      panel.background = element_rect(fill = plot_background_color, color = NA),
      plot.background = element_rect(fill = plot_background_color, color = NA),
      plot.margin = k * plot_margin_scatter
    )
}

# k scales every size; per_row is the number of models per legend row
dimensionality_plot <- function(m, k = 1, per_row = max_per_row) {
  y_label <- if (m == "GCN") gcn_labels[[curves$metric_source[curves$metric == m][1]]] else metric_labels[[m]]
  ch <- chance %>% filter(metric == m)
  pts <- points %>% filter(metric == m)
  curve_pts <- pts %>% filter(curve) %>% arrange(model, latent_dim)
  sub_pts <- pts %>% filter(kind == "dim_subset")
  ref_pts <- pts %>% filter(kind == "ref_sub")

  # y axis starting and ending on a labelled tick, around everything drawn
  y_range <- range(c(ch$mean - ch$std, ch$mean + ch$std, pts$mean - pts$std, pts$mean + pts$std), na.rm = TRUE)
  y_breaks <- pretty(y_range, n = 5)

  p <- ggplot() +
    geom_ribbon(data = ch, aes(x = latent_dim, ymin = mean - std, ymax = mean + std), fill = chance_band_color) +
    geom_line(data = ch, aes(x = latent_dim, y = mean, linetype = chance_name), color = chance_line_color, linewidth = k * line_size) +
    geom_line(data = curve_pts, aes(x = latent_dim, y = mean, color = model, group = model), linewidth = k * line_size) +
    geom_errorbar(data = sub_pts, aes(x = latent_dim, ymin = mean - std, ymax = mean + std, color = model),
                  width = errorbar_width, linewidth = k * line_size) +
    geom_point(data = pts, aes(x = latent_dim, y = mean, color = model, shape = model),
               size = k * point_size_scatter, alpha = 0.9) +
    scale_color_manual(values = model_colors, labels = model_labels, limits = models$model, breaks = models$model, name = "Model") +
    scale_shape_manual(values = model_shapes, labels = model_labels, limits = models$model, breaks = models$model, name = "Model") +
    scale_linetype_manual(values = setNames("solid", chance_name), name = "") +
    guides(color = guide_legend(nrow = ceiling(nrow(models) / per_row), byrow = TRUE, order = 1),
           shape = guide_legend(nrow = ceiling(nrow(models) / per_row), byrow = TRUE, order = 1),
           linetype = guide_legend(order = 2)) +
    scale_x_continuous(trans = "log2", breaks = d_breaks, labels = d_breaks) +
    scale_y_continuous(limits = range(y_breaks), breaks = y_breaks, expand = expansion(mult = 0.02)) +
    labs(x = "Latent dimensionality d", y = y_label, title = NULL) +
    scatter_theme(k)

  if (direct_labels) {
    p <- p + geom_text_repel(
      data = ref_pts,
      aes(x = latent_dim, y = mean, label = model_labels[as.character(model)], color = model),
      direction = "both", seed = 42, segment.size = k * line_size_repel, segment.alpha = line_alpha_repel,
      segment.color = "gray30", size = k * text_size_scatter, fontface = "bold", family = plot_font_family,
      box.padding = k * 1.2, point.padding = k * 0.8, force = 15, force_pull = 0, max.iter = 20000, max.time = 10,
      max.overlaps = Inf, min.segment.length = 0, show.legend = FALSE
    )
  }
  p
}

metrics_to_plot <- intersect(metric_order, unique(curves$metric))

for (m in metrics_to_plot) {
  save_path <- file.path(save_dir, paste0("dimensionality_", gsub("[^A-Za-z]", "", m), ".png"))
  ggsave(filename = save_path, plot = dimensionality_plot(m), width = 14, height = 11, dpi = 600, bg = "white")
  cat("Saved plot to:", save_path, "\n")
}

# All metrics in one figure, one legend at the bottom
combined <- wrap_plots(lapply(metrics_to_plot, dimensionality_plot, k = combined_scale, per_row = combined_max_per_row),
                       ncol = combined_ncol, guides = "collect") &
  theme(legend.position = "bottom")
n_rows <- ceiling(length(metrics_to_plot) / combined_ncol)
save_path <- file.path(save_dir, "dimensionality_all_metrics.png")
ggsave(filename = save_path, plot = combined, width = combined_ncol * combined_panel_width,
       height = n_rows * combined_panel_height + combined_legend_height, dpi = 600, bg = "white")
cat("Saved plot to:", save_path, "\n")
