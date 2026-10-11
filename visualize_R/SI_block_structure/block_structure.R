#' SI Figure (Proposition 2, SI Section 7): block structure of the decomposition embeddings.
#' (a) heatmaps of the dataset-mean component distributions R_bar_i(u), units ordered by owner and then by purity,
#' one panel per model x beta, S and Z branches separately; (b) consistency index CI per model x beta x branch,
#' with the null (CI = 0) and the untrained control; (c) per-sample JS of the negatives and positives against
#' beta (S branch), with the references ln 2 and j_C of Proposition 2. Same style as beta_tradeoff.R.
#' Reads data/block_structure/block_structure_<dataset>.csv and mean_distributions_<dataset>_<run_id>_<branch>_<split>.csv
#' written by scripts/post-training/block_structure.py.

library(ggplot2)
library(vscDebugger)
library(dplyr)
library(tidyr)
library(readr)
library(scales)
library(ggnewscale)

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
strip_text_size <- 22
line_size <- 1.2
point_size <- 2.5
ref_line_size <- 0.8

# Cold colours for positive divergences, warm for negative (SI Figs 22-23); FD red and EWT blue (SI Fig 24)
positive_colors <- c("#2166ac", "#4575b4", "#74add1", "#abd9e9")
negative_colors <- c("#d73027", "#f46d43", "#fdae61", "#fee090")
decomp_palette <- c("FD" = "#d73027", "EWT" = "#2166ac", "EMD" = "#5aae61", "VMD" = "#35978f")

# Load data from
load_dir <- file.path('..', 'data', 'block_structure')
# Save data at
save_dir <- file.path('..', 'supplementary_figures', 'SI_block_structure')
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

datasets <- c(sim_vowels = "SimVowels", timit = "TIMIT")
selected_split <- "test"
selected_betas <- c(0, 0.1, 1)
models <- c("FD", "EWT", "EMD", "VMD")
branch_names <- c(s = "S branch (sequence)", z = "Z branch (frame)")
lab_div <- "Jensen-Shannon Divergence"

theme_block <- function(show_legend = TRUE) {
  theme_minimal() +
    theme(
      plot.title = element_text(size = plot_title_size, face = title_font_face, family = plot_font_family,
                                margin = margin(b = 10)),
      plot.subtitle = element_text(size = plot_subtitle_size, family = plot_font_family, margin = margin(b = 20)),
      axis.title = element_text(size = axis_title_size, family = plot_font_family),
      axis.text = element_text(size = axis_text_size, family = plot_font_family),
      legend.title = element_text(size = legend_title_size, face = legend_font_face, family = plot_font_family),
      legend.text = element_text(size = legend_text_size, family = plot_font_family),
      strip.text = element_text(size = strip_text_size, family = plot_font_family),
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

save_plot <- function(p, name, width = 12, height = 8) {
  save_path <- file.path(save_dir, paste0(name, ".png"))
  ggsave(filename = save_path, plot = p, width = width, height = height, dpi = 600, bg = "white")
  cat("Plot saved to:", save_path, "\n")
}

read_dataset <- function(ds) {
  f <- file.path(load_dir, paste0("block_structure_", ds, ".csv"))
  if (!file.exists(f)) return(NULL)
  read_csv(f, show_col_types = FALSE, col_types = cols(comp_i = col_double(), comp_j = col_double())) %>%
    filter(split == selected_split) %>%
    mutate(Model = factor(model, levels = models), Branch = unname(branch_names[branch]))
}

# (a) Heatmap of R_bar_i(u); units ordered by owner (argmax_i R_bar_i(u)), then by purity a_owner(u)
heatmap_plot <- function(d, ds, br) {
  runs <- d %>% filter(branch == br, !untrained, beta %in% selected_betas) %>%
    distinct(run_id, Model, beta)
  if (nrow(runs) == 0) return(invisible(NULL))
  hm <- bind_rows(lapply(seq_len(nrow(runs)), function(k) {
    f <- file.path(load_dir, paste0("mean_distributions_", ds, "_", runs$run_id[k], "_", br, "_", selected_split, ".csv"))
    if (!file.exists(f)) return(NULL)
    read_csv(f, show_col_types = FALSE) %>% filter(row != "P") %>%
      pivot_longer(-row, names_to = "unit", values_to = "mass") %>%
      group_by(unit) %>%
      mutate(owner = row[which.max(mass)], purity = max(mass) / sum(mass)) %>%
      ungroup() %>%
      mutate(Model = runs$Model[k], beta = runs$beta[k])
  }))
  if (nrow(hm) == 0) return(invisible(NULL))
  hm <- hm %>%
    group_by(Model, beta) %>%
    mutate(unit_rank = match(unit, unique(unit[order(owner, -purity)]))) %>%
    ungroup() %>%
    mutate(Component = factor(sub("R", "", row), levels = rev(sort(unique(sub("R", "", row))))),
           Panel = factor(paste0(Model, ", β=", beta),
                          levels = unique(paste0(Model, ", β=", beta)[order(Model, beta)])))
  p <- ggplot(hm, aes(x = unit_rank, y = Component, fill = mass)) +
    geom_tile() +
    facet_wrap(~ Panel, ncol = length(unique(hm$beta))) +
    scale_fill_gradient(name = expression(bar(R)[i](u)), low = "white", high = positive_colors[1],
                        trans = "log10", breaks = breaks_log(n = 4), labels = label_scientific(digits = 2),
                        guide = guide_colorbar(barheight = unit(8, "cm"))) +
    scale_x_continuous(expand = c(0, 0)) +
    labs(title = "", x = "Hidden unit (ordered by owner, then purity)", y = "Component") +
    theme_block() +
    theme(panel.grid = element_blank())
  save_plot(p, paste0("a_", ds, "_heatmap_", br), width = 16, height = 8)
}

# (b) Consistency index; the null is CI = 0 by construction, the untrained control as a dashed line
ci_plot <- function(d, ds) {
  ci <- d %>% filter(measure == "CI")
  trained <- ci %>% filter(!untrained, beta %in% selected_betas)
  if (nrow(trained) == 0) return(invisible(NULL))
  betas <- selected_betas[selected_betas %in% trained$beta]
  trained <- trained %>% mutate(Beta_f = factor(beta, levels = betas))
  control <- ci %>% filter(untrained) %>% mutate(Control = paste0("Untrained ", model))
  p <- ggplot(trained, aes(x = Beta_f, y = value, color = Model, group = Model)) +
    geom_hline(yintercept = 0, color = "grey40", linewidth = ref_line_size) +
    geom_line(linewidth = line_size, alpha = 0.8) +
    geom_point(size = 2 * point_size, alpha = 0.9) +
    scale_color_manual(name = "Decomposition", values = decomp_palette, drop = TRUE, guide = guide_legend(order = 1)) +
    facet_wrap(~ Branch)
  if (nrow(control) > 0) {
    p <- p + new_scale_color() +
      geom_hline(data = control, aes(yintercept = value, linetype = Control), color = "grey30",
                 linewidth = ref_line_size) +
      scale_linetype_manual(name = "Control", values = c("dashed", "dotted"), guide = guide_legend(order = 2))
  }
  p <- p +
    scale_y_continuous(breaks = pretty_breaks(n = 6), labels = label_number(accuracy = 0.1)) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(title = "", x = "β value", y = "Consistency index (CI)") +
    theme_block()
  save_plot(p, paste0("b_", ds, "_consistency_index"), width = 16, height = 8)
}

# (c) Per-sample JS of the positives (cold) and negatives (warm), S branch, with ln 2 and j_C
js_beta_plot <- function(d, ds) {
  s <- d %>% filter(branch == "s", !untrained, beta %in% selected_betas)
  js <- s %>% filter(measure %in% c("js_pos_mean", "js_neg_mean")) %>%
    select(Model, model, beta, measure, value) %>% pivot_wider(names_from = measure, values_from = value)
  if (nrow(js) == 0) return(invisible(NULL))
  ref <- tibble(value = c(log(2), (s %>% filter(measure == "j_C"))$value[1]),
                Reference = c("ln 2", paste0("j_C (C = ", max(s$comp_i, na.rm = TRUE), ")")))
  present <- models[models %in% js$model]
  pos_pal <- setNames(positive_colors[match(present, models)], present)
  neg_pal <- setNames(negative_colors[match(present, models)], present)
  betas <- selected_betas[selected_betas %in% js$beta]
  js <- js %>% mutate(Beta_f = factor(beta, levels = betas))
  p <- ggplot() +
    geom_hline(data = ref, aes(yintercept = value, linetype = Reference), color = "grey30", linewidth = ref_line_size) +
    scale_linetype_manual(name = "Proposition 2", values = c("dashed", "dotted"), guide = guide_legend(order = 3)) +
    geom_line(data = js, aes(x = Beta_f, y = js_pos_mean, color = Model, group = Model), linewidth = line_size, alpha = 0.8) +
    geom_point(data = js, aes(x = Beta_f, y = js_pos_mean, color = Model), size = point_size, alpha = 0.9) +
    scale_color_manual(name = "Positive", values = pos_pal, guide = guide_legend(order = 1)) +
    new_scale_color() +
    geom_line(data = js, aes(x = Beta_f, y = js_neg_mean, color = Model, group = Model), linewidth = line_size, alpha = 0.8) +
    geom_point(data = js, aes(x = Beta_f, y = js_neg_mean, color = Model), size = point_size, alpha = 0.9) +
    scale_color_manual(name = "Negative", values = neg_pal, guide = guide_legend(order = 2)) +
    scale_x_discrete(labels = betas) +
    scale_y_continuous(breaks = pretty_breaks(n = 8), labels = label_number(accuracy = 0.1)) +
    coord_cartesian(ylim = c(0, 0.8)) +
    labs(title = "", x = "β value", y = lab_div) +
    theme_block()
  save_plot(p, paste0("c_", ds, "_js_vs_beta_s"))
}

for (ds in names(datasets)) {
  d <- read_dataset(ds)
  if (is.null(d) || nrow(d) == 0) next
  heatmap_plot(d, ds, "s")
  heatmap_plot(d, ds, "z")
  ci_plot(d, ds)
  js_beta_plot(d, ds)
}

cat("Completed block structure figures\n")
