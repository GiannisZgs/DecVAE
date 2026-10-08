#' Subspace analysis figure (fold stability): per-dimension probe importance across the CV folds of
#' the headline SimVowels model, one panel per factor, subspace boundaries marked. Each fold's
#' importances are divided by their maximum. The stability numbers for the caption are printed.
#' Plots data/subspace/fold_importance.csv and fold_stability_headline.csv, written by
#' scripts/post-training/subspace_analysis.py.

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

display_factor_names <- c(vowel = "Vowel", speaker_frame = "Speaker")

# Load data from
load_dir <- file.path('..', 'data', 'subspace')

# Save data at
save_dir <- file.path('..', 'supplementary_figures', 'SI_subspace', 'subspace_fold_stability')
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
    panel.grid = element_blank(),
    legend.position = "right"
  )

importance <- read_csv(file.path(load_dir, "fold_importance.csv"), show_col_types = FALSE)
stability <- read_csv(file.path(load_dir, "fold_stability_headline.csv"), show_col_types = FALSE)

blocks <- importance %>% group_by(subspace) %>% summarise(start = min(dim), end = max(dim), .groups = "drop") %>%
  arrange(start)
n_folds <- length(unique(importance$fold))

for (fac in unique(importance$factor)) {
  df <- importance %>% filter(factor == fac) %>% mutate(fold = factor(fold, levels = rev(sort(unique(fold)))))
  p <- ggplot(df, aes(x = dim, y = fold, fill = importance)) +
    geom_tile() +
    geom_vline(xintercept = blocks$start[-1] - 0.5, color = "white", linewidth = 1.2) +
    scale_fill_viridis(option = "plasma", limits = c(0, 1), oob = scales::squish, name = "Importance\n(fold max = 1)") +
    scale_x_continuous(breaks = (blocks$start + blocks$end) / 2, labels = blocks$subspace, expand = c(0, 0)) +
    scale_y_discrete(labels = function(x) paste("Fold", as.integer(x) + 1)) +
    labs(title = "", x = "", y = "") +
    family_a_theme
  label <- ifelse(is.na(display_factor_names[fac]), fac, display_factor_names[fac])
  save_path <- file.path(save_dir, paste0("fold_importance_", tolower(label), "_", df$model[1], ".png"))
  ggsave(filename = save_path, plot = p, width = 18, height = 1.5 + 0.9 * n_folds, dpi = 600, bg = "white")
  cat("Saved plot to:", save_path, "\n")
}

cat("\nFold stability (", stability$model[1], ", seed ", stability$seed[1], "):\n", sep = "")
for (i in seq_len(nrow(stability))) {
  cat(sprintf("  %s: Spearman %.2f, top-%d%% Jaccard %.2f\n", stability$factor[i], stability$spearman[i],
              round(100 * stability$top_frac[i]), stability$topk_jaccard[i]))
}
cat(sprintf("  Same dominant factor in every fold: %.0f%% of dimensions\n", 100 * stability$assignment_consistency[1]))
