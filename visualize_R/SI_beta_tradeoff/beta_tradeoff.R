#' SI Figure (R1 #6): the beta trade-off between component separation and downstream performance.
#' (a, b) positive and negative divergences of the trained checkpoints against beta, SimVowels and TIMIT;
#' (c-e) SimVowels trade-off: negative divergence against vowel accuracy, speaker accuracy and DCI
#' disentanglement, one point per beta; (f) TIMIT: against DCI disentanglement; (g) IEMOCAP: against
#' emotion accuracy. Same style as the pre-training divergence figures (SI Figs 22-24).
#' Reads data/beta_tradeoff/tradeoff_wide.csv written by scripts/post-training/beta_tradeoff_tables.py.

library(ggplot2)
library(vscDebugger)
library(dplyr)
library(tidyr)
library(readr)
library(scales)
library(ggnewscale)
library(ggrepel)

# Style and font parameters
plot_font_family <- "Arial"
plot_title_size <- 28
title_font_face <- "plain"
plot_subtitle_size <- 22
axis_title_size <- 28
axis_text_size <- 22
legend_title_size <- 28
legend_text_size <- 28
legend_font_face <- "plain"
line_size <- 1.2
point_size <- 2.5
label_size <- 7
errorbar_linewidth <- 1.2

# Cold colours for positive divergences, warm for negative (SI Figs 22-23); FD red and EWT blue (SI Fig 24)
positive_colors <- c("#2166ac", "#4575b4", "#74add1", "#abd9e9")
negative_colors <- c("#d73027", "#f46d43", "#fdae61", "#fee090")
decomp_palette <- c("FD" = "#d73027", "EWT" = "#2166ac", "EMD" = "#5aae61", "VMD" = "#35978f")

# "checkpoint": the model's own div_neg and div_pos of the evaluated checkpoints (decomposition_divergences.py),
# as in SI Fig 22. "wandb_last_epoch": provisional, the same values logged at the last pre-training epoch.
divergence_source <- "checkpoint"

# Load data from
load_dir <- file.path('..', 'data', 'beta_tradeoff')
# Save data at
save_dir <- file.path('..', 'supplementary_figures', 'SI_beta_tradeoff', divergence_source)
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

selected_betas <- c(0, 0.1, 1, 5, 10, 20, 40)
models <- c("FD", "EWT", "EMD", "VMD")
transfer_names <- c(vowels = "SimVowels", timit = "TIMIT")
x_ortho <- "model_div_neg"; x_recon <- "model_div_pos"
lab_div <- "Jensen-Shannon Divergence"
lab_ortho <- "Jensen-Shannon Divergence (negatives)"
if (divergence_source == "wandb_last_epoch") {
  lab_div <- "Jensen-Shannon Divergence (last epoch)"
  lab_ortho <- "Jensen-Shannon Divergence (negatives, last epoch)"
}

dat <- read_csv(file.path(load_dir, "tradeoff_wide.csv"), show_col_types = FALSE) %>%
  filter(source == divergence_source) %>%
  mutate(Model = factor(model, levels = models),
         Beta_label = as.character(beta),
         Transfer = unname(transfer_names[as.character(transfer_from)]))

theme_divergence <- function(show_legend = TRUE) {
  theme_minimal() +
    theme(
      plot.title = element_text(size = plot_title_size, face = title_font_face, family = plot_font_family,
                                margin = margin(b = 10)),
      plot.subtitle = element_text(size = plot_subtitle_size, family = plot_font_family, margin = margin(b = 20)),
      axis.title = element_text(size = axis_title_size, family = plot_font_family),
      axis.text = element_text(size = axis_text_size, family = plot_font_family),
      legend.title = element_text(size = legend_title_size, face = legend_font_face, family = plot_font_family),
      legend.text = element_text(size = legend_text_size, family = plot_font_family),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.5),
      panel.grid.minor = element_line(color = "grey95", linewidth = 0.25),
      legend.position = if (show_legend) "right" else "none",
      legend.box.background = element_rect(color = "grey80", fill = "white"),
      legend.margin = margin(10, 10, 10, 10),
      plot.margin = margin(20, 20, 20, 20)
    )
}

save_plot <- function(p, name) {
  save_path <- file.path(save_dir, paste0(name, ".png"))
  ggsave(filename = save_path, plot = p, width = 12, height = 8, dpi = 600, bg = "white")
  cat("Plot saved to:", save_path, "\n")
}

# (a, b) Positive (cold) and negative (warm) divergences against beta, one line per decomposition
divergence_beta_plot <- function(d, name) {
  if (!all(c(x_ortho, x_recon) %in% names(d))) return(invisible(NULL))
  d <- d %>% filter(!is.na(.data[[x_ortho]]) | !is.na(.data[[x_recon]]))
  if (nrow(d) == 0) return(invisible(NULL))
  present <- models[models %in% d$model]
  pos_pal <- setNames(positive_colors[match(present, models)], present)
  neg_pal <- setNames(negative_colors[match(present, models)], present)
  betas <- selected_betas[selected_betas %in% d$beta]
  d <- d %>% mutate(Beta_f = factor(beta, levels = betas))
  p <- ggplot() +
    geom_line(data = d %>% filter(!is.na(.data[[x_recon]])),
              aes(x = Beta_f, y = .data[[x_recon]], color = Model, group = Model), linewidth = line_size, alpha = 0.8) +
    geom_point(data = d %>% filter(!is.na(.data[[x_recon]])),
               aes(x = Beta_f, y = .data[[x_recon]], color = Model), size = point_size, alpha = 0.9) +
    scale_color_manual(name = "Positive", values = pos_pal, guide = guide_legend(order = 1)) +
    new_scale_color() +
    geom_line(data = d %>% filter(!is.na(.data[[x_ortho]])),
              aes(x = Beta_f, y = .data[[x_ortho]], color = Model, group = Model), linewidth = line_size, alpha = 0.8) +
    geom_point(data = d %>% filter(!is.na(.data[[x_ortho]])),
               aes(x = Beta_f, y = .data[[x_ortho]], color = Model), size = point_size, alpha = 0.9) +
    scale_color_manual(name = "Negative", values = neg_pal, guide = guide_legend(order = 2)) +
    scale_x_discrete(labels = betas) +
    scale_y_continuous(breaks = pretty_breaks(n = 8), labels = label_number(accuracy = 0.1)) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(title = "", x = "β value", y = lab_div) +
    theme_divergence()
  save_plot(p, name)
}

# (c-g) Trade-off: one point per beta, joined in beta order, labelled with beta
tradeoff_plot <- function(d, yvar, ylab, name, ci = NULL, shape_by_transfer = FALSE) {
  if (!all(c(x_ortho, yvar) %in% names(d))) return(invisible(NULL))
  if (!is.null(ci) && !(ci %in% names(d))) ci <- NULL
  d <- d %>% filter(!is.na(.data[[x_ortho]]), !is.na(.data[[yvar]])) %>% arrange(Model, Transfer, beta)
  if (nrow(d) == 0) return(invisible(NULL))
  d$grp <- if (shape_by_transfer) interaction(d$Model, d$Transfer) else d$Model
  p <- ggplot(d, aes(x = .data[[x_ortho]], y = .data[[yvar]], color = Model)) +
    geom_path(aes(group = grp), linewidth = line_size, alpha = 0.8)
  if (!is.null(ci)) {
    p <- p + geom_errorbar(aes(ymin = .data[[yvar]] - .data[[ci]], ymax = .data[[yvar]] + .data[[ci]]),
                           width = 0, linewidth = errorbar_linewidth, alpha = 0.6, na.rm = TRUE)
  }
  p <- p + (if (shape_by_transfer) geom_point(aes(shape = Transfer), size = 2 * point_size, alpha = 0.9)
            else geom_point(size = 2 * point_size, alpha = 0.9)) +
    geom_text_repel(aes(label = paste0("β=", Beta_label)), size = label_size, family = plot_font_family,
                    show.legend = FALSE, seed = 1, max.overlaps = Inf, box.padding = 0.5) +
    scale_color_manual(name = "Decomposition", values = decomp_palette, drop = TRUE) +
    scale_x_continuous(breaks = pretty_breaks(n = 6), labels = label_number(accuracy = 0.01)) +
    scale_y_continuous(breaks = pretty_breaks(n = 6), labels = label_number(accuracy = 0.01)) +
    labs(title = "", x = lab_ortho, y = ylab, shape = "Pre-trained on") +
    theme_divergence()
  save_plot(p, name)
}

sv <- dat %>% filter(dataset == "sim_vowels")
tm <- dat %>% filter(dataset == "timit")
divergence_beta_plot(sv, "a_sim_vowels_divergences_vs_beta")
divergence_beta_plot(tm, "b_timit_divergences_vs_beta")
tradeoff_plot(sv, "accuracy_vowel", "Accuracy (vowel)", "c_sim_vowels_tradeoff_vowel")
tradeoff_plot(sv, "accuracy_speaker", "Accuracy (speaker)", "d_sim_vowels_tradeoff_speaker")
tradeoff_plot(sv, "disentanglement", "Disentanglement", "e_sim_vowels_tradeoff_disentanglement")
tradeoff_plot(tm, "disentanglement", "Disentanglement", "f_timit_tradeoff_disentanglement")
tradeoff_plot(dat %>% filter(dataset == "iemocap"), "accuracy_emotion", "Weighted Accuracy (ER)",
              "g_iemocap_tradeoff_emotion", ci = "accuracy_emotion_ci", shape_by_transfer = TRUE)

cat("Completed beta trade-off figures\n")
