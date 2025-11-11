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
current_script_experiment <- "SSL_loss_ablation"
selected_betas <- c(0,0.1,1)
use_row <- -2
decomp <- "filter" # filter only
branches <- c("Z_branch","S_branch")
display_metric_names <- c("Accuracy (speaker)", "Accuracy (vowel)", "Robustness", "1 - Mutual Info.", "1 - Gaussian Correlation", "Disentanglement", "Completeness", "Modularity", "Informativeness", "Explicitness")
true_metric_names <- c("accuracy_speaker","accuracy_vowel","IRS","mi","gaussian_tc_norm","disentanglement","completeness","modularity","informativeness","explicitness")
metric_y_limits <- list(
  "accuracy_speaker" = c(0.55,1),
  "accuracy_vowel" = c(0.85,1),
  "IRS" = c(0.25,0.75),
  "mi" = c(0.9,1),
  "gaussian_tc_norm" = c(0,1),
  "disentanglement" = c(0.4,0.65),
  "completeness" = c(0.1,0.35),
  "modularity" = c(0.75,0.95),
  "informativeness" = c(0.6,1),
  "explicitness" = c(0.8,1)
)

# Save data at
parent_save_dir <- file.path('..','figures','post-training','vowels')
save_dir <- file.path(parent_save_dir, current_script_experiment)
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

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

extract_percentage_from_column <- function(col_name) {
  # Extract percentage from column name (e.g., "60percent" -> 60)
  percent_match <- str_extract(col_name, "([0-9]+)percent")
  if (!is.na(percent_match)) {
    return(as.numeric(str_extract(percent_match, "[0-9]+")))
  }
  return(50)  # Default percentage
}

# Collect all data across branches and betas
all_rows <- list()

for (branch in branches) {
  load_dir <- file.path(parent_load_dir, current_script_experiment, branch)
  
  # Load and process files
  files <- list.files(load_dir, pattern = "\\.csv$", full.names = TRUE)
  if (length(files) == 0) {
    cat("No CSV files found in:", load_dir, "\n")
    next
  }
  
  for (selected_beta in selected_betas) {
    cat("Processing branch:", branch, "beta:", selected_beta, "\n")
    
    for (f in files) {
      base <- tools::file_path_sans_ext(basename(f))
      metric <- extract_metric_from_filename(base)
      
      if (is.null(metric)) {
        next
      }
      
      df <- suppressMessages(readr::read_csv(f, show_col_types = FALSE))
      if (nrow(df) < 1) next
      
      # Remove columns that contain MIN or MAX
      original_cols <- colnames(df)
      min_max_cols <- grep("MIN|MAX", original_cols, ignore.case = TRUE, value = TRUE)
      if (length(min_max_cols) > 0) {
        df <- df %>% select(-any_of(min_max_cols))
      }
      
      # Filter columns by beta value and decomposition
      cols <- colnames(df)
      beta_match_cols <- get_beta_match_cols(cols, selected_beta)
      
      # Further filter by decomposition
      decomp_pattern <- paste0("(", decomp, "|", toupper(decomp), ")")
      decomp_beta_cols <- beta_match_cols[grepl(decomp_pattern, beta_match_cols, ignore.case = TRUE)]
      
      if (length(decomp_beta_cols) == 0) {
        next
      }
      
      # Process each matching column
      for (selected_col in decomp_beta_cols) {
        # Extract percentage from column name
        percentage <- extract_percentage_from_column(selected_col)
        
        # Extract value using use_row logic
        col_values <- df[[selected_col]]
        non_na_values <- col_values[!is.na(col_values)]
        
        if (length(non_na_values) > 1) {
          if (use_row < 0) {
            row_index <- length(non_na_values) + use_row + 1
          } else {
            row_index <- use_row
          }
          
          if (row_index > 0 && row_index <= length(non_na_values)) {
            value <- non_na_values[row_index]
          } else {
            value <- non_na_values[1]
          }
        } else if (length(non_na_values) == 1) {
          value <- non_na_values[1]
        } else {
          value <- suppressWarnings(as.numeric(df[1, selected_col]))
        }
        
        # Transform specific metrics
        if (metric %in% c("mi", "gaussian_tc_norm")) {
          value <- 1 - value
        }
        
        all_rows[[length(all_rows) + 1]] <- data.frame(
          Branch = branch,
          Beta = selected_beta,
          Metric = metric,
          Percentage = percentage,
          Value = value,
          stringsAsFactors = FALSE
        )
      }
    }
  }
}

if (length(all_rows) == 0) {
  stop("No valid data found for any branch/beta combination.")
}

# Combine all data
dat <- bind_rows(all_rows)

# Create combination labels for legend
dat$Branch_Beta <- paste0(str_remove(dat$Branch, "_branch"), ", β", dat$Beta)

# Map metric names
metric_mapping <- setNames(display_metric_names, true_metric_names)
dat$Metric_Display <- metric_mapping[dat$Metric]
dat$Metric_Display[is.na(dat$Metric_Display)] <- dat$Metric[is.na(dat$Metric_Display)]

# Create line plots for each metric
unique_metrics <- unique(dat$Metric_Display)

for (metric_name in unique_metrics) {
  metric_data <- dat %>% filter(Metric_Display == metric_name)
  
  if (nrow(metric_data) == 0) next
  
  # Get the corresponding true metric name for y-limits
  true_metric <- names(metric_mapping)[metric_mapping == metric_name]
  if (length(true_metric) == 0) true_metric <- metric_name
  
  # Get y-limits for this metric
  y_limits <- metric_y_limits[[true_metric[1]]]
  if (is.null(y_limits)) y_limits <- c(0, 1)  # Default if not specified
  
  # Create colors for different branch-beta combinations
  unique_combinations <- unique(metric_data$Branch_Beta)
  n_combinations <- length(unique_combinations)
  combination_colors <- viridis(n_combinations, option = "turbo", end = yellow_block_threshold)
  names(combination_colors) <- unique_combinations
  
  p <- ggplot(metric_data, aes(x = Percentage, y = Value, color = Branch_Beta, group = Branch_Beta)) +
    geom_line(size = line_size, alpha = 0.8) +
    geom_point(size = point_size, alpha = 0.9) +
    scale_color_manual(values = combination_colors) +
    scale_x_continuous(breaks = pretty_breaks(n = 8)) +
    scale_y_continuous(limits = y_limits,
                       breaks = pretty_breaks(n = 6),
                       expand = expansion(mult = c(0, 0.05)), 
                       labels = label_number(accuracy = 0.01)) +
    labs(
      title = "",
      x = "Frames used in SSL loss (%)",
      y = metric_name,
      color = "Latent Branch & β"
    ) +
    theme_minimal(base_size = 14, base_family = plot_font_family) +
    theme(
      plot.title = element_text(size = plot_title_size, face = title_font_face),
      axis.title = element_text(size = axis_title_size),
      axis.text = element_text(size = axis_text_size),
      legend.title = element_text(size = legend_title_size, face = legend_font_face),
      legend.text = element_text(size = legend_text_size),
      panel.grid.minor = element_blank(),
      legend.position = "right"
    )
  
  # Save individual metric plot
  save_path_metric <- file.path(save_dir, paste0("SSL_loss_ablation_", decomp, "_", gsub("[^A-Za-z0-9]", "_", metric_name), ".png"))
  ggsave(filename = save_path_metric, plot = p, width = 14, height = 8, dpi = 600, bg = "white")
  cat("Saved plot for", metric_name, "to:", save_path_metric, "\n")
}

cat("Completed SSL loss ablation analysis for", decomp, "decomposition\n")