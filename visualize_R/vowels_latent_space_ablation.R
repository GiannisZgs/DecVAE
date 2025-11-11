library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(scales)
library(viridis)
library(ggnewscale)
library(stringr)
library(cowplot)

plot_font_family <- "Arial"
plot_title_size <- 28
title_font_face <- "plain"
plot_subtitle_size <- 22
axis_title_size <- 28
axis_text_size <- 22  
legend_title_size <- 20
legend_text_size <- 18
legend_font_face <- "plain"
line_size <- 1.2
point_size <- 2.5

yellow_block_threshold <- 1.0
colors <- viridis(n = 8, option = "turbo", end = yellow_block_threshold)

# Load data from
parent_load_dir <- file.path("D:", "wandb_exports_for_figures", "vowels_posttraining")
current_script_experiment <- "latent_spaces_ablation"
decomp <- "ewt"
selected_beta <- 0
branch <- "Z_branch"
display_latent_names <- c("β-VAE X", "X", "OC1", "OC2", "OC3", "joint OCs", "projected OCs", "X+OCs")
true_latent_names <- c("X_VAE", "X", "OC1", "OC2", "OC3", "OCs_joint", "OCs_proj", "All")
display_metric_names <- c("Accuracy (speaker)", "Accuracy (vowel)", "Robustness", "1 - Mutual Info.", "1 - Gaussian Correlation", "Disentanglement", "Completeness", "Modularity", "Informativeness", "Explicitness")
true_metric_names <- c("accuracy_speaker","accuracy_vowel","IRS","mi","gaussian_tc_norm","disentanglement","completeness","modularity","informativeness","explicitness")

#data_dir
load_dir <- file.path(parent_load_dir, current_script_experiment, branch)

# Save data at
parent_save_dir <- file.path('..','figures','post-training','vowels')
save_dir <- file.path(parent_save_dir, current_script_experiment)
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

save_path <- file.path(save_dir, paste0("latent_spaces_ablation_",branch,"_",decomp,"_b",selected_beta,".png"))


# Helper functions
beta_regex <- function(beta) {
  if (is.numeric(beta)) {
    if (beta == 0.1) return("b(0\\.1|01)(_|$)")
    if (beta == 0.01) return("b(0\\.01|001)(_|$)")
    return(paste0("b", beta, "(_|$)"))
  }
  if (beta == "0.1") return("b(0\\.1|01)(_|$)")
  if (beta == "0.01") return("b(0\\.01|001)(_|$)")
  paste0("b", beta, "(_|$)")
}

decomp_regex <- function(decomp) {
  d <- tolower(decomp)
  if (d == "filter") return("(filter|FD)")
  if (d == "ewt") return("(ewt|EWT)")
  return(decomp)
}

parse_metric_latent <- function(fname_no_ext) {
  # Try to match against known latent names first (from longest to shortest to avoid partial matches)
  latent_names_sorted <- true_latent_names[order(nchar(true_latent_names), decreasing = TRUE)]
  
  for (latent in latent_names_sorted) {
    if (endsWith(fname_no_ext, paste0("_", latent))) {
      # Found a matching latent name, extract the metric part
      metric <- sub(paste0("_", latent, "$"), "", fname_no_ext)
      # Verify the metric is in our known list
      if (metric %in% true_metric_names) {
        return(list(metric = metric, latent = latent))
      }
    }
  }
  
  # Fallback: if no known pattern matches, use original logic
  parts <- strsplit(fname_no_ext, "_")[[1]]
  if (length(parts) < 2) {
    return(list(metric = fname_no_ext, latent = ""))
  }
  latent <- parts[length(parts)]
  metric <- paste(parts[1:(length(parts)-1)], collapse = "_")
  list(metric = metric, latent = latent)
}

# Load and process files
files <- list.files(load_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(files) == 0) {
  stop("No CSV files found in: ", load_dir)
}

b_pat <- beta_regex(selected_beta)
d_pat <- decomp_regex(decomp)

rows <- list()

for (f in files) {
  base <- tools::file_path_sans_ext(basename(f))
  parsed <- parse_metric_latent(base)
  print(parsed)
  metric <- parsed$metric
  latent <- parsed$latent
  
  # Only keep if latent is one of desired
  if (!(latent %in% true_latent_names)) next
  
  df <- suppressMessages(readr::read_csv(f, show_col_types = FALSE))
  if (nrow(df) < 1) next
  
  # Remove columns that contain MIN or MAX
  original_cols <- colnames(df)
  min_max_cols <- grep("MIN|MAX", original_cols, ignore.case = TRUE, value = TRUE)
  if (length(min_max_cols) > 0) {
    cat("Removing MIN/MAX columns from", basename(f), ":", paste(min_max_cols, collapse = ", "), "\n")
    df <- df %>% select(-any_of(min_max_cols))
  }
  
  cols <- colnames(df)
  match_cols <- cols[grepl(d_pat, cols, ignore.case = TRUE) & grepl(b_pat, cols, ignore.case = TRUE)]
  
  # Special case for X_VAE: if no decomp found in columns, use default (first column with beta)
  if (length(match_cols) == 0 && latent == "X_VAE") {
    beta_only_cols <- cols[grepl(b_pat, cols, ignore.case = TRUE)]
    if (length(beta_only_cols) > 0) {
      match_cols <- beta_only_cols[1]  # Take first column with matching beta
      cat("Using default column for X_VAE:", match_cols, "\n")
    }
  }
  
  if (length(match_cols) == 0) {
    message("No matching columns for ", basename(f), " (decomp=", decomp, ", beta=", selected_beta, ")")
    next
  }
  
  # Single row; take mean across matched columns if multiple
  vals <- suppressWarnings(as.numeric(df[1, match_cols]))
  value <- mean(vals, na.rm = TRUE)
  
  # Transform specific metrics: replace with 1 - value
  if (metric %in% c("mi", "gaussian_tc_norm")) {
    value <- 1 - value
    if (value < 0) value <- 0
    if (value > 1) value <- 1
    cat("Transformed", metric, "value to 1 - value for", latent, "\n")
  }
  
  rows[[length(rows) + 1]] <- data.frame(
    Latent = latent,
    Metric = metric,
    Value = value,
    stringsAsFactors = FALSE
  )
}

if (length(rows) == 0) {
  stop("No valid metric-latent values found for the given filters.")
}

dat <- bind_rows(rows)

# Map to display names
latent_mapping <- setNames(display_latent_names, true_latent_names)
metric_mapping <- setNames(display_metric_names, true_metric_names)

dat$Latent_Display <- latent_mapping[dat$Latent]
dat$Metric_Display <- metric_mapping[dat$Metric]

# Handle unmapped values
dat$Latent_Display[is.na(dat$Latent_Display)] <- dat$Latent[is.na(dat$Latent_Display)]
dat$Metric_Display[is.na(dat$Metric_Display)] <- dat$Metric[is.na(dat$Metric_Display)]

# Ensure ordering
dat$Latent_Display <- factor(dat$Latent_Display, levels = display_latent_names)
dat$Metric_Display <- factor(dat$Metric_Display, levels = display_metric_names)

# Create grouped bar chart
p <- ggplot(dat, aes(x = Latent_Display, y = Value, fill = Metric_Display)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1),
                     expand = expansion(mult = c(0, 0.05)), 
                     labels = label_number(accuracy = 0.1)) +
  scale_fill_viridis_d(option = "turbo", end = yellow_block_threshold) +
  labs(
    title ="",
    x = "", #Latent Space
    y = "Metric Value",
    fill = "Metric"
  ) +
  theme_minimal(base_size = 14, base_family = plot_font_family) +
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

# Save plot
ggsave(filename = save_path, plot = p, width = 12, height = 8, dpi = 600, bg = "white")
cat("Saved plot to:", save_path, "\n")


