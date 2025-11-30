#' Fig.4: This script generates figure 4h in the main paper containing an overview of the main
#' results for the VOC-ALS dataset.

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
axis_text_size <- 16 
axis_text_x_angle <- 60
legend_title_size <- 20
legend_text_size <- 18
legend_font_face <- "plain"
line_size <- 1.2
point_size <- 2.5

yellow_block_threshold <- 1.0
colors <- viridis(n = 8, option = "turbo", end = yellow_block_threshold)

# Load data from
current_script_experiment <- "total_results"
selected_betas <- c(0.1)
selected_components <- c(3,4)
branch <- "Z_branch"
parent_load_dir <- file.path('..', 'data', 'results_wandb_exports_for_figures', 'voc_als_posttraining')
use_row <- -2
models <- c("raw_mels", "PCA", "ICA", "VAE_vowels", "VAE_timit", "filter_vowels", "ewt_vowels", "filter_timit", "ewt_timit")
display_model_names <- c("Raw Mel Fbank","PCA", "ICA", "Vowels β-VAE", "TIMIT β-VAE","Vowels β-DecVAE + FD","Vowels β-DecVAE + EWT", "TIMIT β-DecVAE + FD","TIMIT β-DecVAE + EWT")
display_metric_names <- c("King's Stage Detection (Acc.)","Disease Duration Detection (Acc.)", "Phoneme Recognition (Acc.)","Speaker Identification (Acc.)", "King's Stage Detection (F1-Macro)","Disease Duration Detection (F1-Macro)", "Phoneme Recognition (F1-Macro)","Speaker Identification (F1-Macro)")
true_metric_names <- c("accuracy_kings_stage","accuracy_disease_duration","accuracy_phoneme", "accuracy_speaker", "f1_macro_kings_stage", "f1_macro_disease_duration", "f1_macro_phoneme", "f1_macro_speaker")
model_order <- c("Raw Mel Fbank", "PCA", "ICA", "Vowels β-VAE", "TIMIT β-VAE", "Vowels β-DecVAE + FD, NoC 3","Vowels β-DecVAE + EWT, NoC 3", "Vowels β-DecVAE + FD, NoC 4","Vowels β-DecVAE + EWT, NoC 4", "TIMIT β-DecVAE + FD, NoC 4","TIMIT β-DecVAE + EWT, NoC 4")

# Save data at
parent_save_dir <- file.path('..','figures','fig_4_voc_als_results','fig_4h_voc_als_post-training')
save_dir <- file.path(parent_save_dir, current_script_experiment, branch)
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
    bz_pattern <- "bz(0\\.1|01)(_|$| )"
    bs_pattern <- "bs(0\\.1|01)(_|$| )"
  } else if (selected_beta == 0.01) {
    b_pattern <- "_b(0\\.01|001)(_|$| )"
    bz_bs_pattern <- "(bz001_bs001|bz0\\.01_bs0\\.01)"
    bz_pattern <- "bz(0\\.01|001)(_|$| )"
    bs_pattern <- "bs(0\\.01|001)(_|$| )"
  } else if (selected_beta == 0) {
    b_pattern <- "_b0(_|$| )"
    bz_bs_pattern <- "bz0_bs0"
    bz_pattern <- "bz0(_|$| )"
    bs_pattern <- "bs0(_|$| )"
  } else {
    b_pattern <- paste0("_b", selected_beta, "(_|$| )")
    bz_bs_pattern <- paste0("bz", selected_beta, "_bs", selected_beta)
    bz_pattern <- paste0("bz", selected_beta, "(_|$| )")
    bs_pattern <- paste0("bs", selected_beta, "(_|$| )")
  }
  
  # Find matches for all patterns
  b_matches <- cols[grepl(b_pattern, cols, ignore.case = TRUE)]
  bz_bs_matches <- cols[grepl(bz_bs_pattern, cols, ignore.case = TRUE)]
  bz_matches <- cols[grepl(bz_pattern, cols, ignore.case = TRUE)]
  bs_matches <- cols[grepl(bs_pattern, cols, ignore.case = TRUE)]
  
  # Log which formats were found
  if (length(b_matches) > 0) {
    cat("Found standard b format columns:", paste(b_matches, collapse = ", "), "\n")
  }
  if (length(bz_bs_matches) > 0) {
    cat("Found bz_bs format columns:", paste(bz_bs_matches, collapse = ", "), "\n")
  }
  if (length(bz_matches) > 0) {
    cat("Found bz format columns:", paste(bz_matches, collapse = ", "), "\n")
  }
  if (length(bs_matches) > 0) {
    cat("Found bs format columns:", paste(bs_matches, collapse = ", "), "\n")
  }
  
  # Combine all matches
  all_matches <- c(b_matches, bz_bs_matches, bz_matches, bs_matches)
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

extract_model_from_column <- function(col_name, models) {
  # Find which model name appears in the column name
  for (model in models) {
    model_pattern <- paste0("(", model, "|", toupper(model), ")")
    if (grepl(model_pattern, col_name, ignore.case = TRUE)) {
      return(model)
    }
  }
  return(NULL)
}

extract_beta_from_column <- function(col_name) {
  # Extract beta value from column name - check all possible formats
  if (grepl("_b(0\\.1|01)(_|$| )", col_name, ignore.case = TRUE) ||
      grepl("bz(0\\.1|01)(_|$| )", col_name, ignore.case = TRUE) ||
      grepl("bs(0\\.1|01)(_|$| )", col_name, ignore.case = TRUE) ||
      grepl("bz0\\.1_bs0\\.1", col_name, ignore.case = TRUE) ||
      grepl("bz01_bs01", col_name, ignore.case = TRUE)) {
    return(0.1)
  } else if (grepl("_b(0\\.01|001)(_|$| )", col_name, ignore.case = TRUE) ||
             grepl("bz(0\\.01|001)(_|$| )", col_name, ignore.case = TRUE) ||
             grepl("bs(0\\.01|001)(_|$| )", col_name, ignore.case = TRUE) ||
             grepl("bz0\\.01_bs0\\.01", col_name, ignore.case = TRUE) ||
             grepl("bz001_bs001", col_name, ignore.case = TRUE)) {
    return(0.01)
  } else if (grepl("_b0(_|$| )", col_name, ignore.case = TRUE) ||
             grepl("bz0(_|$| )", col_name, ignore.case = TRUE) ||
             grepl("bs0(_|$| )", col_name, ignore.case = TRUE) ||
             grepl("bz0_bs0", col_name, ignore.case = TRUE)) {
    return(0)
  } else {
    # Try to extract numeric beta from standard formats
    for (pattern in c("_b([0-9]+)(_|$| )", "bz([0-9]+)(_|$| )", "bs([0-9]+)(_|$| )")) {
      beta_match <- str_extract(col_name, pattern)
      if (!is.na(beta_match)) {
        return(as.numeric(str_extract(beta_match, "[0-9]+")))
      }
    }
  }
  return(NULL)
}

extract_components_from_column <- function(col_name) {
  # Extract NoC value from column name (e.g., "NoC3" -> 3)
  noc_match <- str_extract(col_name, "NoC([0-9]+)")
  if (!is.na(noc_match)) {
    return(as.numeric(str_extract(noc_match, "[0-9]+")))
  }
  return(NULL)
}

# Collect all data across files
all_rows <- list()

load_dir <- file.path(parent_load_dir, current_script_experiment, branch)

# Load and process files
files <- list.files(load_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(files) == 0) {
  # Try .xls files if no CSV files found
  files <- list.files(load_dir, pattern = "\\.xls$", full.names = TRUE)
  if (length(files) == 0) {
    stop("No CSV or XLS files found in: ", load_dir)
  }
  cat("No CSV files found, using XLS files instead\n")
}

for (f in files) {
  base <- tools::file_path_sans_ext(basename(f))
  metric <- extract_metric_from_filename(base)
  
  if (is.null(metric)) {
    next
  }
  
  cat("Processing:", basename(f), "for metric:", metric, "\n")
  
  # Read file based on extension
  if (grepl("\\.csv$", f, ignore.case = TRUE)) {
    df <- suppressMessages(readr::read_csv(f, show_col_types = FALSE))
  } else if (grepl("\\.xls$", f, ignore.case = TRUE)) {
    if (!requireNamespace("readxl", quietly = TRUE)) {
      stop("readxl package is required to read XLS files. Please install it with: install.packages('readxl')")
    }
    df <- suppressMessages(readxl::read_excel(f))
  } else {
    next
  }
  
  if (nrow(df) < 1) next
  
  # Remove columns that contain MIN or MAX
  original_cols <- colnames(df)
  min_max_cols <- grep("MIN|MAX", original_cols, ignore.case = TRUE, value = TRUE)
  if (length(min_max_cols) > 0) {
    df <- df %>% select(-any_of(min_max_cols))
  }
  
  # Process each column
  cols <- colnames(df)
  for (col in cols) {
    # Extract model from column name
    model_val <- extract_model_from_column(col, models)
    
    if (is.null(model_val)) {
      next
    }
    
    # For beta-related models (filter, ewt, VAE), extract beta value
    if (model_val %in% c("filter_vowels", "ewt_vowels", "filter_timit", "ewt_timit", "VAE_vowels", "VAE_timit")) {
      beta_val <- extract_beta_from_column(col)
      if (is.null(beta_val) || !(beta_val %in% selected_betas)) {
        next
      }
      
      # Extract components for decomposition models
      if (model_val %in% c("filter_vowels", "ewt_vowels", "filter_timit", "ewt_timit")) {
        components_val <- extract_components_from_column(col)
        
        if (is.null(components_val) || !(components_val %in% selected_components)) {
          next
        }
      } else {
        components_val <- NA
      }
    } else {
      # For non-beta models (raw_mels), use NA for beta and components
      beta_val <- NA
      components_val <- NA
    }
    
    # Extract value using use_row logic
    col_values <- df[[col]]
    non_na_values <- col_values[!is.na(col_values)]
    
    mean_value <- non_na_values[1]
    ci_value <- non_na_values[2]

    # Ensure values are numeric to handle scientific notation like 8E-17
    mean_value <- as.numeric(mean_value)
    ci_value <- as.numeric(ci_value)
    
    # Skip if conversion failed
    if (is.na(mean_value)) {
      next
    }
    
    # Transform specific metrics
    if (metric %in% c("mi", "gaussian_tc_norm")) {
      mean_value <- 1 - mean_value
      if (mean_value < 0) mean_value <- 0
    }
    
    all_rows[[length(all_rows) + 1]] <- data.frame(
      Metric = metric,
      Beta = beta_val,
      Model = model_val,
      Components = components_val,
      Value = mean_value,
      CI = ci_value,
      stringsAsFactors = FALSE
    )
  }
}

if (length(all_rows) == 0) {
  stop("No valid data found for any combination.")
}

# Combine all data
dat <- bind_rows(all_rows)

# Map model and metric names
model_mapping <- setNames(display_model_names, models)
metric_mapping <- setNames(display_metric_names, true_metric_names)

dat$Model_Display <- model_mapping[dat$Model]
dat$Model_Display[is.na(dat$Model_Display)] <- dat$Model[is.na(dat$Model_Display)]

dat$Metric_Display <- metric_mapping[dat$Metric]
dat$Metric_Display[is.na(dat$Metric_Display)] <- dat$Metric[is.na(dat$Metric_Display)]

# Create combination labels for grouping
dat$Model_Beta_Components <- ifelse(
  is.na(dat$Components), 
  dat$Model_Display,
  paste0(dat$Model_Display, ", NoC ", dat$Components)
)

# Calculate error bar bounds
dat$CI_lower <- dat$Value - dat$CI
dat$CI_upper <- dat$Value + dat$CI

# Ensure proper ordering
dat$Model_Beta_Components <- factor(dat$Model_Beta_Components, levels = model_order)
dat$Metric_Display <- factor(dat$Metric_Display, levels = display_metric_names)

p <- ggplot(dat, aes(x = Model_Beta_Components, y = Value, fill = Metric_Display)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper), 
                position = position_dodge(width = 0.8), 
                width = 0.5, 
                linewidth = 1.2,
                color = "black") +
  scale_fill_viridis_d(option = "turbo", end = yellow_block_threshold) +
  scale_y_continuous(limits = c(0, max(dat$CI_upper, na.rm = TRUE) * 1.1),
                     breaks = pretty_breaks(n = 6),
                     expand = expansion(mult = c(0, 0.05)), 
                     labels = label_number(accuracy = 0.01)) +
  labs(
    title = "",
    x =  "",
    y = "Metric Value",
    fill = "Metric"
  ) +
  theme_minimal(base_size = 14, base_family = plot_font_family) +
  theme(
    plot.title = element_text(size = plot_title_size, face = title_font_face),
    axis.title = element_text(size = axis_title_size),
    axis.text = element_text(size = axis_text_size),
    axis.text.x = element_text(angle = axis_text_x_angle, hjust = 1),
    legend.title = element_text(size = legend_title_size, face = legend_font_face),
    legend.text = element_text(size = legend_text_size),
    panel.grid.minor = element_blank(),
    legend.position = "right"
  )

# Save grouped bar chart
save_path_plot <- file.path(save_dir, "total_model_results_grouped.png")
ggsave(filename = save_path_plot, plot = p, width = 18, height = 8, dpi = 600, bg = "white")
cat("Saved grouped bar chart to:", save_path_plot, "\n")

cat("Completed total model results analysis\n")
