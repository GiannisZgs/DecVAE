#' SI Figure (SimCoupled): characterization of the SimCoupled dataset, in the style of SI_fig_4a.
#' Plots the CSVs written by scripts/simulations/simulated_coupled_characterization.py:
#' mean log-power spectra and autocovariances per lag and gain class, example waveforms per class,
#' and one example sequence with its segment-level labels.

library(vscDebugger)
library(tidyverse)
library(MetBrewer)
library(patchwork)

# Style and font parameters
plot_font_family <- "Arial"
plot_background_color <- "#F4F5F1"
plot_text_color <- "black"

# Axis titles
axis_title_size <- 35
axis_title_face <- "plain"
axis_title_margin <- margin(t = 15, r = 15, b = 15, l = 15)

# Axis tick labels
axis_text_y_size <- 30
axis_text_y_face <- "plain"
axis_text_x_size <- 30
axis_text_x_face <- "plain"
axis_text_color <- "black"

# Plot title
plot_title_size <- 0
plot_title_face <- "bold"
plot_title_hjust <- 0.5
plot_title_vjust <- 0.5
plot_title_margin <- margin(b = 20)

# Facet text
facet_text_size <- 40
facet_text_face <- "plain"

# Plot margins and spacing
plot_margin <- margin(10, 10, 10, 10)
panel_spacing <- unit(2, "lines")

# Line widths
class_line_width <- 1
context_line_width <- 0.4
context_line_color <- "grey70"
label_line_width <- 1.2
label_line_color <- met.brewer("Redon")[2]

# Load data
data_dir <- file.path('..', '..', 'sim_coupled', 'characterization')
# Set and create directory and filepaths to save
save_dir <- file.path('..', 'supplementary_figures', 'SI_sim_coupled')
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

splits <- c('indep', 'test')
lag_levels <- c(2, 4, 8, 16, 32)
gain_levels <- c(-4, -2, 0, 2, 4)
lag_labels <- paste0("k = ", lag_levels)
gain_labels <- paste0(ifelse(gain_levels > 0, "+", ""), gain_levels, " dB")
gain_example_lag <- 8 # lag of the gain waveform examples, as in the Python script

fig_width <- 20
row_height <- 6
fig_dpi <- 600

# Define a base theme for all plots - with font passed directly to element_text
base_theme <- theme_minimal() +
  theme(
    text = element_text(family = plot_font_family),
    plot.background = element_rect(color = plot_background_color, fill = plot_background_color),
    panel.background = element_rect(fill = NA, color = NA),
    plot.margin = plot_margin,
    panel.spacing = panel_spacing,
    strip.text = element_text(
      size = facet_text_size,
      face = facet_text_face,
      color = plot_text_color,
      family = plot_font_family
    ),
    plot.title = element_text(
      size = plot_title_size,
      face = plot_title_face,
      hjust = plot_title_hjust,
      vjust = plot_title_vjust,
      margin = plot_title_margin,
      family = plot_font_family
    ),
    axis.text.y = element_text(
      size = axis_text_y_size,
      face = axis_text_y_face,
      color = axis_text_color,
      family = plot_font_family
    ),
    axis.text.x = element_text(
      size = axis_text_x_size,
      face = axis_text_x_face,
      color = axis_text_color,
      family = plot_font_family
    ),
    axis.title = element_text(
      size = axis_title_size,
      face = axis_title_face,
      color = plot_text_color,
      margin = axis_title_margin,
      family = plot_font_family
    ),
    legend.position = "none"
  )

# Facet label per class, ordered by class index
add_class_label <- function(data, factor_name) {
  labels <- if (factor_name == 'lag') lag_labels else gain_labels
  data %>% mutate(class_label = factor(labels[class + 1], levels = labels))
}

save_plot <- function(plot, fname, n_rows) {
  if (interactive()) print(plot)  # Rscript's default device has no Arial; ggsave uses ragg
  ggsave(
    file.path(save_dir, paste0(fname, ".png")),
    plot,
    width = fig_width,
    height = row_height * n_rows,
    dpi = fig_dpi,
    bg = plot_background_color
  )
}

#' One facet per class; the other classes are drawn in grey behind it for comparison.
create_class_curve_plot <- function(data, x_col, y_col, y_label, show_strip_text = TRUE,
                                    show_x_axis = TRUE, x_label = "", vlines = NULL, x_breaks = waiver()) {
  context <- data %>% select(-class_label) %>% rename(context_class = class)
  p <- ggplot(data) +
    geom_line(data = context, aes(x = .data[[x_col]], y = .data[[y_col]], group = context_class),
              color = context_line_color, linewidth = context_line_width) +
    geom_line(aes(x = .data[[x_col]], y = .data[[y_col]], color = class_label), linewidth = class_line_width)
  if (!is.null(vlines)) {
    p <- p + geom_vline(data = vlines, aes(xintercept = x, color = class_label), linetype = "dashed", linewidth = 0.5)
  }
  p +
    scale_color_met_d(name = "Redon") +
    scale_x_continuous(breaks = x_breaks) +
    facet_wrap(~ class_label, nrow = 1) +
    coord_cartesian(clip = "off") +
    labs(x = x_label, y = y_label) +
    base_theme +
    theme(
      axis.title.x = if (show_x_axis) element_text(size = axis_title_size, family = plot_font_family) else element_blank(),
      axis.text.x = if (show_x_axis) element_text(size = axis_text_x_size, family = plot_font_family) else element_blank(),
      strip.text.x = if (show_strip_text)
                       element_text(face = facet_text_face, size = facet_text_size, family = plot_font_family)
                     else
                       element_blank(),
      axis.title.y = element_text(size = axis_title_size, face = axis_title_face, angle = 90,
                                  vjust = 0.5, family = plot_font_family, margin = margin(r = 25))
    )
}

#' Rows = splits, facets = classes
create_split_rows <- function(data, factor_name, x_col, y_col, y_label, x_label, vlines = NULL, x_breaks = waiver()) {
  plots <- lapply(seq_along(splits), function(i) {
    split_data <- data %>% filter(split == splits[i]) %>% add_class_label(factor_name)
    create_class_curve_plot(split_data, x_col, y_col, paste0(splits[i], "\n", y_label),
                            show_strip_text = (i == 1), show_x_axis = (i == length(splits)),
                            x_label = x_label, vlines = vlines, x_breaks = x_breaks)
  })
  wrap_plots(plots, ncol = 1)
}

# Mean log-power spectra; the DC and Nyquist bins are dropped (halved by Welch, DC also detrended)
read_psd <- function(fname) {
  read_csv(file.path(data_dir, fname), show_col_types = FALSE) %>%
    filter(freq_hz > 0, freq_hz < max(freq_hz)) %>%
    mutate(freq_khz = freq_hz / 1000)
}
psd_lag <- read_psd("psd_by_lag.csv")
psd_gain <- read_psd("psd_by_gain.csv")

save_plot(create_split_rows(psd_lag, 'lag', "freq_khz", "log_power_db", "Log-power (dB)", "Frequency (kHz)"),
          "sim_coupled_psd_by_lag", length(splits))
save_plot(create_split_rows(psd_gain, 'gain', "freq_khz", "log_power_db", "Log-power (dB)", "Frequency (kHz)"),
          "sim_coupled_psd_by_gain", length(splits))

# Mean autocovariance
acf_lag <- read_csv(file.path(data_dir, "acf_by_lag.csv"), show_col_types = FALSE)
acf_gain <- read_csv(file.path(data_dir, "acf_by_gain.csv"), show_col_types = FALSE)
acf_breaks <- c(0, 16, 32, 48)
lag_vlines <- data.frame(class = seq_along(lag_levels) - 1, x = lag_levels) %>% add_class_label('lag')

save_plot(create_split_rows(acf_lag, 'lag', "lag_samples", "autocov", "Autocorrelation", "Lag (samples)", vlines = lag_vlines,
                            x_breaks = acf_breaks),
          "sim_coupled_acf_by_lag", length(splits))
save_plot(create_split_rows(acf_gain, 'gain', "lag_samples", "autocov", "Autocovariance", "Lag (samples)",
                            x_breaks = acf_breaks),
          "sim_coupled_acf_by_gain", length(splits))

# Example waveforms per class
create_signal_plot <- function(data, signal_label) {
  ggplot(data) +
    geom_hline(yintercept = 0, linetype = "solid", linewidth = .25) +
    geom_line(aes(x = time_ms, y = value, color = class_label)) +
    scale_color_met_d(name = "Redon") +
    scale_y_continuous(labels = scales::number_format(accuracy = 0.1)) +
    scale_x_continuous(breaks = c(0, 5, 10)) +
    facet_wrap(~ class_label, nrow = 1) +
    coord_cartesian(clip = "off") +
    labs(x = "Time (ms)", y = signal_label) +
    base_theme +
    theme(
      axis.title.y = element_text(size = axis_title_size, face = axis_title_face, angle = 90,
                                  vjust = 0.5, family = plot_font_family)
    )
}

wave_lag <- read_csv(file.path(data_dir, "waveforms_by_lag.csv"), show_col_types = FALSE) %>% add_class_label('lag')
wave_gain <- read_csv(file.path(data_dir, "waveforms_by_gain.csv"), show_col_types = FALSE) %>% add_class_label('gain')

save_plot(create_signal_plot(wave_lag, "X + Y, gain removed"), "sim_coupled_waveforms_by_lag", 1)
save_plot(create_signal_plot(wave_gain, paste0("Observation, k = ", gain_example_lag)), "sim_coupled_waveforms_by_gain", 1)

# Example sequence with its segment-level labels
seq_audio <- read_csv(file.path(data_dir, "example_sequence.csv"), show_col_types = FALSE)
seq_labels <- read_csv(file.path(data_dir, "example_sequence_labels.csv"), show_col_types = FALSE) %>%
  mutate(lag_label = factor(lag_labels[lag + 1], levels = lag_labels),
         gain_label = factor(gain_labels[gain + 1], levels = gain_labels))

#' Segment labels as a step function; the last value is repeated at the end so the final segment is drawn
create_label_plot <- function(data, y_col, y_label, breaks, log_scale = FALSE) {
  steps <- bind_rows(
    data %>% transmute(t = t_start_s, value = .data[[y_col]]),
    data %>% slice_tail(n = 1) %>% transmute(t = t_end_s, value = .data[[y_col]])
  )
  p <- ggplot(steps) +
    geom_step(aes(x = t, y = value), direction = "hv", color = label_line_color, linewidth = label_line_width) +
    scale_x_continuous(limits = c(0, max(data$t_end_s)), expand = c(0, 0)) +
    labs(x = "Time (s)", y = y_label) +
    base_theme
  if (log_scale) {
    p + scale_y_continuous(trans = "log2", breaks = breaks)
  } else {
    p + scale_y_continuous(breaks = breaks)
  }
}

for (split_name in splits) {
  audio <- seq_audio %>% filter(split == split_name)
  labels <- seq_labels %>% filter(split == split_name)
  p_audio <- ggplot(audio) +
    geom_hline(yintercept = 0, linetype = "solid", linewidth = .25) +
    geom_line(aes(x = time_s, y = value), color = met.brewer("Redon")[2], linewidth = 0.1) +
    scale_x_continuous(limits = c(0, max(labels$t_end_s)), expand = c(0, 0)) +
    labs(y = "Observation") +
    base_theme +
    theme(axis.title.x = element_blank(), axis.text.x = element_blank())
  p_lag <- create_label_plot(labels, "lag_samples", "Lag (samples)", lag_levels, log_scale = TRUE) +
    theme(axis.title.x = element_blank(), axis.text.x = element_blank())
  p_gain <- create_label_plot(labels, "gain_db", "Gain (dB)", gain_levels)
  final_plot <- p_audio / p_lag / p_gain + plot_layout(heights = c(2, 1, 1))
  save_plot(final_plot, paste0("sim_coupled_example_sequence_", split_name), 2)
}
