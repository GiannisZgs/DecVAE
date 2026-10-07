#' Subspace analysis figure (SimVowels): (a) generator expectation, (b)-(d) prediction, ablation and
#' swap matrices of the headline configuration (mean over seeds), (e) selectivity of the DecVAE
#' subspaces against random partitions per configuration, (f) CKA across seeds.
#' Plots the CSVs written by scripts/post-training/subspace_analysis.py to data/subspace/:
#' generator_expectation.csv, subspace_matrices.csv, selectivity_null.csv, cka.csv. One PNG per panel.

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
point_size <- 2.5
cell_text_size <- 7
yellow_block_threshold <- 1.0

# Fixed colour limits per matrix type, so models can be compared
matrix_limits <- list(G = c(0, 1), P = c(0, 1), A = c(0, 0.5), dT = c(0, 0.5), CKA = c(0, 1))
matrix_legend <- c(G = "η²", P = "Prediction", A = "Ablation", dT = "Swap\nT − Tc", CKA = "Linear CKA")
matrix_file <- c(P = "prediction", A = "ablation", dT = "swap")
display_factor_names <- c(vowel = "Vowel", speaker = "Speaker", speaker_frame = "Speaker", phoneme = "Phoneme",
                          phoneme_frame = "Phoneme", king_stage_frame = "King's stage", cat_emotion_frame = "Emotion")
display_decomp_names <- c(ewt = "EWT", fd = "FD", emd = "EMD", vmd = "VMD")

# Load data from
load_dir <- file.path('..', 'data', 'subspace')

# Save data at
save_dir <- file.path('..', 'supplementary_figures', 'SI_subspace', 'subspace_simvowels')
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

config_display <- function(decomposition, beta) {
  paste0(display_decomp_names[decomposition], " β = ", format(beta))
}

plot_matrix <- function(df, x_col, y_col, y_levels, limits, legend_title, y_title) {
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
  ggplot(df, aes(x = .data[[x_col]], y = .data[[y_col]], fill = mean)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = label, color = text_color), size = cell_text_size, family = plot_font_family, lineheight = 0.9) +
    scale_color_identity() +
    scale_fill_viridis(option = "plasma", limits = limits, oob = scales::squish, na.value = "grey85",
                       name = legend_title) +
    scale_y_discrete(limits = rev(y_levels)) +
    labs(title = "", x = "", y = y_title) +
    family_a_theme +
    theme(panel.grid = element_blank())
}

save_plot <- function(p, name, width, height) {
  save_path <- file.path(save_dir, paste0(name, ".png"))
  ggsave(filename = save_path, plot = p, width = width, height = height, dpi = 600, bg = "white")
  cat("Saved plot to:", save_path, "\n")
}

matrices <- read_csv(file.path(load_dir, "subspace_matrices.csv"), show_col_types = FALSE) %>%
  filter(dataset == "sim_vowels", run == "main") %>%
  mutate(factor_display = factor(display_factor_names[factor], levels = unique(display_factor_names[factor])))
headline <- matrices %>% filter(headline)
if (nrow(headline) == 0) stop("No headline configuration in subspace_matrices.csv")
headline_name <- config_display(headline$decomposition[1], headline$beta[1])
subspace_levels <- unique(headline$subspace)
cat("Headline configuration:", headline$config[1], "(", headline_name, "), std over", headline$std_over[1],
    ", n =", headline$n[1], "\n")

# (a) Generator expectation
generator <- read_csv(file.path(load_dir, "generator_expectation.csv"), show_col_types = FALSE) %>%
  mutate(factor_display = factor(display_factor_names[factor], levels = c("Vowel", "Speaker")),
         std = NA_real_)  # identical across seeds: it depends on the labels only
p <- plot_matrix(generator, "factor_display", "formant", c("F1", "F2", "F3"), matrix_limits$G,
                 matrix_legend[["G"]], "Formant")
save_plot(p, "a_generator_expectation", 8, 8)

# (b)-(d) Prediction, ablation and swap matrices of the headline configuration
for (m in c("P", "A", "dT")) {
  p <- plot_matrix(headline %>% filter(matrix == m), "factor_display", "subspace", subspace_levels,
                   matrix_limits[[m]], matrix_legend[[m]], "Subspace")
  save_plot(p, paste0(c(P = "b", A = "c", dT = "d")[[m]], "_", matrix_file[[m]], "_", headline$config[1]), 8, 9)
}

# (e) Selectivity of the DecVAE subspaces against random partitions
selectivity <- read_csv(file.path(load_dir, "selectivity_null.csv"), show_col_types = FALSE) %>%
  filter(dataset == "sim_vowels", run == "main") %>%
  mutate(config_display = config_display(decomposition, beta))
config_levels <- selectivity %>% distinct(order, config_display) %>% arrange(order) %>% pull(config_display)
selectivity <- selectivity %>% mutate(config_display = factor(config_display, levels = config_levels))
seed_levels <- paste("Seed", sort(unique(selectivity$seed)))
shape_levels <- c("Random partitions", seed_levels)
colors <- viridis(n = length(config_levels), option = "turbo", end = yellow_block_threshold)
names(colors) <- config_levels

for (m in c("P", "dT")) {
  own <- selectivity %>% filter(matrix == m, partition == "own") %>%
    mutate(point_type = factor(paste("Seed", seed), levels = shape_levels))
  null <- selectivity %>% filter(matrix == m, partition == "null") %>%
    mutate(point_type = factor("Random partitions", levels = shape_levels))
  p <- ggplot() +
    geom_jitter(data = null, aes(x = config_display, y = value, shape = point_type),
                color = "grey60", size = point_size, width = 0.15, height = 0, alpha = 0.9) +
    geom_point(data = own, aes(x = config_display, y = value, shape = point_type, color = config_display),
               size = point_size * 2.5, position = position_nudge(x = 0.3), alpha = 0.9) +
    scale_color_manual(values = colors, guide = "none") +
    scale_shape_manual(values = c(16, 15, 17, 18, 8)[seq_along(shape_levels)], limits = shape_levels, name = "",
                       guide = guide_legend(override.aes = list(color = c("grey60", rep("black", length(seed_levels))),
                                                                size = point_size * 2))) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1), labels = label_number(accuracy = 0.1),
                       expand = expansion(mult = c(0, 0.05))) +
    labs(title = "", x = "", y = paste("Selectivity of", c(P = "prediction", dT = "swap")[[m]])) +
    family_a_theme
  save_plot(p, paste0("e_selectivity_vs_null_", matrix_file[[m]]), 14, 8)
}

# (f) CKA across seeds, and the optional CKA between configurations
cka_path <- file.path(load_dir, "cka.csv")
if (file.exists(cka_path)) {
  cka <- read_csv(cka_path, show_col_types = FALSE) %>% filter(dataset == "sim_vowels")
  for (comp in unique(cka$comparison)) {
    df <- cka %>% filter(comparison == comp) %>%
      mutate(block_b = factor(block_b, levels = subspace_levels), std = ifelse(n > 1, std, NA_real_))
    x_title <- if (grepl(" vs ", comp)) df$config_b[1] else "Subspace (seed j)"
    y_title <- if (grepl(" vs ", comp)) df$config_a[1] else "Subspace (seed i)"
    p <- plot_matrix(df, "block_b", "block_a", subspace_levels, matrix_limits$CKA, matrix_legend[["CKA"]], y_title) +
      labs(x = x_title)
    save_plot(p, paste0("f_cka_", gsub(" ", "_", comp)), 10, 9)
  }
}
