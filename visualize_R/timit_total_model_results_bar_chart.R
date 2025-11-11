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
axis_text_size <- 20 
axis_text_x_angle <- 60
legend_title_size <- 20
legend_text_size <- 20
legend_font_face <- "plain"
line_size <- 1.2
point_size <- 2.5

yellow_block_threshold <- 1.0
colors <- viridis(n = 8, option = "turbo", end = yellow_block_threshold)

# Load data from
parent_load_dir <- file.path("D:", "wandb_exports_for_figures", "timit_posttraining")
current_script_experiment <- "total_results"
selected_betas <- c(0,0.1,1)
use_row <- -1
models <- c("raw_mels","ICA","PCA","VAE","filter","ewt")
display_model_names <- c("Raw Mel Fbank",  "ICA", "PCA", "β-VAE", "β-DecVAE + FD","β-DecVAE + EWT")
display_metric_names <- c( "1 - Mutual Info.",  "1 - Gaussian Correlation", "Disentanglement", "Completeness", "Informativeness", "Modularity",  "Explicitness", "Robustness") #"Accuracy (speaker)", "Accuracy (phoneme)",
true_metric_names <- c("mi","gaussian_tc_norm","disentanglement","completeness","informativeness","modularity","explicitness","IRS") #"accuracy_speaker","accuracy_phoneme",
    
# Save data at
parent_save_dir <- file.path('..','figures','post-training','TIMIT')
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

# Collect all data across files
all_rows <- list()

load_dir <- file.path(parent_load_dir, current_script_experiment)

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
    if (model_val %in% c("filter", "ewt", "VAE")) {
      beta_val <- extract_beta_from_column(col)
      if (is.null(beta_val) || !(beta_val %in% selected_betas)) {
        next
      }
    } else {
      # For non-beta models (ICA, PCA, raw_mels), use NA for beta
      beta_val <- NA
    }
    
    # Extract value using use_row logic
    col_values <- df[[col]]
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
      next  # Skip columns with no valid data
    }
    
    # Transform specific metrics
    if (metric %in% c("mi", "gaussian_tc_norm")) {
      value <- 1 - value
      if (value < 0) value <- 0
    }
    
    all_rows[[length(all_rows) + 1]] <- data.frame(
      Metric = metric,
      Beta = beta_val,
      Model = model_val,
      Value = value,
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
dat$Model_Beta <- ifelse(is.na(dat$Beta), 
                         dat$Model_Display,
                         paste0(dat$Model_Display, ", β = ", dat$Beta))

# Create colors for different metrics
unique_metrics <- unique(dat$Metric_Display)
n_metrics <- length(unique_metrics)
metric_colors <- viridis(n_metrics, option = "turbo", end = yellow_block_threshold)
names(metric_colors) <- unique_metrics

# Create proper ordering based on display_model_names
model_beta_order <- c()
for (model_display in display_model_names) {
  # Add non-beta version if it exists in data
  if (any(dat$Model_Display == model_display & is.na(dat$Beta))) {
    model_beta_order <- c(model_beta_order, model_display)
  }
  # Add beta versions if they exist in data
  for (beta in selected_betas) {
    beta_combo <- paste0(model_display, ", β = ", beta)
    if (beta_combo %in% dat$Model_Beta) {
      model_beta_order <- c(model_beta_order, beta_combo)
    }
  }
}

# Ensure proper ordering
dat$Model_Beta <- factor(dat$Model_Beta, levels = model_beta_order)
dat$Metric_Display <- factor(dat$Metric_Display, levels = display_metric_names)

p <- ggplot(dat, aes(x = Model_Beta, y = Value, fill = Metric_Display)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = metric_colors) +
  scale_y_continuous(limits = c(0, 1),
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
