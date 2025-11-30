#' SI Figure 14c: post-training evaluation on TIMIT. Ablation of the number of
#' components used w.r.t. disentanglement metrics.

library(ggplot2)
library(vscDebugger)
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
parent_load_dir <- file.path('..', 'data', 'results_wandb_exports_for_figures', 'timit_posttraining')
current_script_experiment <- "NoC_ablation"
selected_beta <- 1
use_row <- -1
branch <- "Z_branch"
display_combination_names <- c("FD NoC 3", "FD NoC 4", "FD NoC 5", "EWT NoC 3", "EWT NoC 4", "EWT NoC 5")
true_combination_names <- c("filter_dual_NoC3", "filter_dual_NoC4", "filter_dual_NoC5", "ewt_dual_NoC3", "ewt_dual_NoC4", "ewt_dual_NoC5")
display_metric_names <- c( "Robustness", "1 - Mutual Info.", "1 - Gaussian Correlation", "Disentanglement", "Completeness", "Informativeness", "Modularity", "Explicitness")
true_metric_names <- c("IRS","mi","gaussian_tc_norm","disentanglement","completeness","modularity","informativeness","explicitness")

#data_dir
load_dir <- file.path(parent_load_dir, current_script_experiment, branch)

# Save data at
parent_save_dir <- file.path('..','supplementary_figures','SI_fig_14_timit_post-training')
save_dir <- file.path(parent_save_dir, current_script_experiment)
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

save_path <- file.path(save_dir, paste0("NoC_ablation_",branch,"_b",selected_beta,".png"))

# Helper functions
beta_regex <- function(beta) {
  if (is.numeric(beta)) {
    if (beta == 0.1) return("(b(0\\.1|01)|bz01_bs01)(_|$)")
    if (beta == 0.01) return("(b(0\\.01|001)|bz001_bs001)(_|$)")
    if (beta == 0) return("(b0|bz0_bs0)(_|$)")
    return(paste0("(b", beta, "|bz", beta, "_bs", beta, ")(_|$)"))
  }
  if (beta == "0.1") return("(b(0\\.1|01)|bz01_bs01)(_|$)")
  if (beta == "0.01") return("(b(0\\.01|001)|bz001_bs001)(_|$)")
  if (beta == "0") return("(b0|bz0_bs0)(_|$)")
  paste0("(b", beta, "|bz", beta, "_bs", beta, ")(_|$)")
}

get_beta_match_cols <- function(cols, selected_beta) {
  # Create patterns for both formats - match anywhere in string but be more specific
  if (selected_beta == 0.1) {
    b_pattern <- "_b(0\\.1|01)(_|$| )"
    bz_bs_pattern <- "(bz01_bs01|bz0\\.1_bs0\\.1)"
  } else if (selected_beta == 0.01) {
    b_pattern <- "_b(0\\.01|001)(_|$| )"
    bz_bs_pattern <- "(bz001_bs001|bz0\\.01_bs0\\.01)"
  } else if (selected_beta == 0) {
    b_pattern <- "_b0(_|$| )"
    bz_bs_pattern <- "bz0_bs0"
  } else {
    b_pattern <- paste0("_b", selected_beta, "(_|$| )")
    bz_bs_pattern <- paste0("bz", selected_beta, "_bs", selected_beta)
  }
  
  # Find matches for both patterns
  b_matches <- cols[grepl(b_pattern, cols, ignore.case = TRUE)]
  bz_bs_matches <- cols[grepl(bz_bs_pattern, cols, ignore.case = TRUE)]
  
  # Log which formats were found
  if (length(b_matches) > 0) {
    cat("Found standard b format columns:", paste(b_matches, collapse = ", "), "\n")
  }
  if (length(bz_bs_matches) > 0) {
    cat("Found bz_bs format columns:", paste(bz_bs_matches, collapse = ", "), "\n")
  }
  
  # Combine all matches
  all_matches <- c(b_matches, bz_bs_matches)
  return(unique(all_matches))
}

extract_metric_from_filename <- function(fname_no_ext) {
  # Find which metric name appears in the filename
  for (metric in true_metric_names) {
    if (grepl(metric, fname_no_ext, ignore.case = TRUE)) {
      return(metric)
    }
  }
  return(NULL)
}

# Load and process files
files <- list.files(load_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(files) == 0) {
  stop("No CSV files found in: ", load_dir)
}

b_pat <- beta_regex(selected_beta)
rows <- list()

for (f in files) {
  base <- tools::file_path_sans_ext(basename(f))
  metric <- extract_metric_from_filename(base)
  
  if (is.null(metric)) {
    message("No known metric found in filename: ", basename(f))
    next
  }
  
  df <- suppressMessages(readr::read_csv(f, show_col_types = FALSE))
  if (nrow(df) < 1) next
  
  # Remove columns that contain MIN or MAX
  original_cols <- colnames(df)
  min_max_cols <- grep("MIN|MAX", original_cols, ignore.case = TRUE, value = TRUE)
  if (length(min_max_cols) > 0) {
    cat("Removing MIN/MAX columns from", basename(f), ":", paste(min_max_cols, collapse = ", "), "\n")
    df <- df %>% select(-any_of(min_max_cols))
  }
  
  # Filter columns by beta value using improved function
  cols <- colnames(df)
  beta_match_cols <- get_beta_match_cols(cols, selected_beta)
  
  if (length(beta_match_cols) == 0) {
    message("No columns with beta ", selected_beta, " found in ", basename(f))
    next
  }
  
  # For each combination, find matching column and extract value
  for (combo in true_combination_names) {
    # Create pattern for this combination (decomp_dual_NoC pattern)
    combo_parts <- strsplit(combo, "_")[[1]]
    decomp_part <- combo_parts[1]
    model_part <- combo_parts[2]  # "dual"
    noc_part <- combo_parts[3]    # "NoC3", "NoC4", etc.
    
    # Look for columns that match the exact combination pattern
    combo_pattern <- paste0(decomp_part, ".*", model_part, ".*", noc_part)
    combo_cols <- beta_match_cols[grepl(combo_pattern, beta_match_cols, ignore.case = TRUE)]
    
    # If no match with dual, try without dual part
    if (length(combo_cols) == 0) {
      combo_pattern_alt <- paste0(decomp_part, ".*", noc_part)
      combo_cols <- beta_match_cols[grepl(combo_pattern_alt, beta_match_cols, ignore.case = TRUE)]
    }
    
    if (length(combo_cols) > 0) {
      # Take the most specific match (longest column name)
      selected_col <- combo_cols[which.max(nchar(combo_cols))]
      
      cat("Matched", combo, "to column:", selected_col, "in", basename(f), "\n")
      
      # Check if column has multiple non-NA values
      col_values <- df[[selected_col]]
      non_na_values <- col_values[!is.na(col_values)]
      
      if (length(non_na_values) > 1) {
        # Multiple rows with data - use use_row to select
        if (use_row < 0) {
          # Negative index: count from end
          row_index <- length(non_na_values) + use_row + 1
          cat("Using row", row_index, "from", length(non_na_values), "available rows (use_row =", use_row, ") for", combo, "in", basename(f), "\n")
        } else {
          # Positive index: count from start
          row_index <- use_row
          cat("Using row", row_index, "from", length(non_na_values), "available rows (use_row =", use_row, ") for", combo, "in", basename(f), "\n")
        }
        
        # Ensure row_index is within bounds
        if (row_index > 0 && row_index <= length(non_na_values)) {
          value <- non_na_values[row_index]
        } else {
          cat("Warning: row_index", row_index, "out of bounds, using first row for", combo, "in", basename(f), "\n")
          value <- non_na_values[1]
        }
      } else {
        # Single row or only one non-NA value - use the non-NA value
        if (length(non_na_values) == 1) {
          value <- non_na_values[1]
        } else {
          # Fallback for truly empty columns
          value <- suppressWarnings(as.numeric(df[1, selected_col]))
        }
      }
      
      # Transform specific metrics: replace with 1 - value
      if (metric %in% c("mi", "gaussian_tc_norm")) {
        # Ensure value is numeric to handle scientific notation like 8E-17
        value <- as.numeric(value)
        if (!is.na(value)) {
          value <- 1 - value
          cat("Transformed", metric, "value to 1 - value for", combo, "\n")
        }
      } else {
        # Ensure all values are numeric
        value <- as.numeric(value)
      }
      
      # Skip if conversion failed
      if (is.na(value)) {
        next
      }
      
      rows[[length(rows) + 1]] <- data.frame(
        Combination = combo,
        Metric = metric,
        Value = value,
        stringsAsFactors = FALSE
      )
    }
  }
}

if (length(rows) == 0) {
  stop("No valid metric-combination values found for the given filters.")
}

dat <- bind_rows(rows)

# Map to display names
combination_mapping <- setNames(display_combination_names, true_combination_names)
metric_mapping <- setNames(display_metric_names, true_metric_names)

dat$Combination_Display <- combination_mapping[dat$Combination]
dat$Metric_Display <- metric_mapping[dat$Metric]

# Handle unmapped values
dat$Combination_Display[is.na(dat$Combination_Display)] <- dat$Combination[is.na(dat$Combination_Display)]
dat$Metric_Display[is.na(dat$Metric_Display)] <- dat$Metric[is.na(dat$Metric_Display)]

# Ensure ordering
dat$Combination_Display <- factor(dat$Combination_Display, levels = display_combination_names)
dat$Metric_Display <- factor(dat$Metric_Display, levels = display_metric_names)

# Create grouped bar chart
p <- ggplot(dat, aes(x = Combination_Display, y = Value, fill = Metric_Display)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1),
                     expand = expansion(mult = c(0, 0.05)), 
                     labels = label_number(accuracy = 0.1)) +
  scale_fill_viridis_d(option = "turbo", end = yellow_block_threshold) +
  labs(
    title = "",
    x = "", #Decomposition & Components
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
ggsave(filename = save_path, plot = p, width = 14, height = 8, dpi = 600, bg = "white")
cat("Saved plot to:", save_path, "\n")
