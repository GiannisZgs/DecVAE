#' SI Figure 17: disentanglement evaluation results on the IEMOCAP dataset.

library(ggplot2)
library(dplyr)
library(vscDebugger)
library(readr)
library(scales)
library(viridis)

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
parent_load_dir <- file.path('..', 'data', 'results_wandb_exports_for_figures', 'iemocap_posttraining', 'total_results')
current_script_experiment <- "Z_branch"
selected_betas <- c(0,0.1,1)
use_row <- 1  # Use first row (mean values)
models <- c("raw_mels","ICA","PCA","VAE","filter","ewt")
display_model_names <- c("Raw Mel Fbank",  "ICA", "PCA", "β-VAE", "β-DecVAE + FD","β-DecVAE + EWT")
display_metric_names <- c("1 - Mutual Info.",  "1 - Gaussian Correlation", "Disentanglement", "Completeness", "Informativeness", "Modularity",  "Explicitness", "Robustness")
true_metric_names <- c("mi","gaussian_tc_norm","disentanglement","completeness","informativeness","modularity","explicitness","IRS")
true_transfer_from_names <- c("vowels", "timit")
display_transfer_from_names <- c("Vowels", "TIMIT")
# Save data at
parent_save_dir <- file.path('..','supplementary_figures','SI_fig_17_disentanglement_barcharts','iemocap')
save_dir <- file.path(parent_save_dir, current_script_experiment)
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

# Helper functions
extract_metric_from_filename <- function(fname_no_ext) {
  # Sort metrics by length (longest first) to avoid substring matches
  sorted_metrics <- true_metric_names[order(nchar(true_metric_names), decreasing = TRUE)]
  
  # Find which metric name appears in the filename
  for (metric in sorted_metrics) {
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
  } else if (grepl("_b0(_|$| )", col_name, ignore.case = TRUE) ||
             grepl("bz0(_|$| )", col_name, ignore.case = TRUE) ||
             grepl("bs0(_|$| )", col_name, ignore.case = TRUE) ||
             grepl("bz0_bs0", col_name, ignore.case = TRUE)) {
    return(0)
  } else if (grepl("_b1(_|$| )", col_name, ignore.case = TRUE) ||
             grepl("bz1(_|$| )", col_name, ignore.case = TRUE) ||
             grepl("bs1(_|$| )", col_name, ignore.case = TRUE) ||
             grepl("bz1_bs1", col_name, ignore.case = TRUE)) {
    return(1)
  }
  return(NULL)
}

extract_transfer_from_column <- function(col_name) {
  # Check for transfer_from indicators in column name
  for (transfer_from in true_transfer_from_names) {
    if (grepl(transfer_from, col_name, ignore.case = TRUE)) {
      return(transfer_from)
    }
  }
  return(NA)  # Return NA instead of NULL to ensure consistent length
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
    cat("Skipping file:", basename(f), "- no metric found\n")
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
  
  if (nrow(df) < 1) {
    cat("Warning: File", basename(f), "has no data rows. Skipping.\n")
    next
  }
  
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
    
    # Extract transfer_from information
    transfer_from_val <- extract_transfer_from_column(col)
    
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
    
    # Extract value from the specified row
    col_values <- df[[col]]
    non_na_values <- col_values[!is.na(col_values)]
    
    if (length(non_na_values) < use_row) {
      next
    }
    
    value <- non_na_values[use_row]
    
    # Ensure value is numeric to handle scientific notation like 8E-17
    value <- as.numeric(value)
    
    # Skip if conversion failed
    if (is.na(value)) {
      next
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
      Transfer_From = transfer_from_val,
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

# Handle transfer_from comparison - keep both variants for all models
transfer_from_mapping <- setNames(display_transfer_from_names, true_transfer_from_names)

# Map model and metric names
model_mapping <- setNames(display_model_names, models)
metric_mapping <- setNames(display_metric_names, true_metric_names)

dat$Model_Display <- model_mapping[dat$Model]
dat$Model_Display[is.na(dat$Model_Display)] <- dat$Model[is.na(dat$Model_Display)]

dat$Metric_Display <- metric_mapping[dat$Metric]
dat$Metric_Display[is.na(dat$Metric_Display)] <- dat$Metric[is.na(dat$Metric_Display)]

dat$Transfer_From_Display <- transfer_from_mapping[dat$Transfer_From]
# Replace NA with empty string for proper concatenation
dat$Transfer_From_Display[is.na(dat$Transfer_From_Display)] <- ""

# Create model names with beta information and transfer_from
dat$Model_Beta <- ifelse(
  is.na(dat$Beta), 
  paste0(dat$Model_Display,
         ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), "")),
  ifelse(dat$Beta == 0,
         ifelse(dat$Model_Display == "β-VAE", 
                paste0("AE", ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), "")),
                ifelse(dat$Model_Display %in% c("β-DecVAE + FD", "β-DecVAE + EWT"), 
                       paste0(gsub("β-DecVAE", "DecAE", dat$Model_Display), 
                              ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), "")), 
                       dat$Model_Display)),
         ifelse(dat$Beta == 1,
                ifelse(dat$Model_Display == "β-VAE", 
                       paste0("VAE", ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), "")),
                       ifelse(dat$Model_Display %in% c("β-DecVAE + FD", "β-DecVAE + EWT"), 
                              paste0(gsub("β-DecVAE", "DecVAE", dat$Model_Display),
                                     ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), "")), 
                              dat$Model_Display)),
                paste0(dat$Model_Display, ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), ""))))  # Keep β-VAE for β=0.1
)

# Create proper ordering by checking what's actually in the data
all_model_betas <- unique(dat$Model_Beta)
model_beta_order <- c()

# Add non-variational models first - use more flexible matching
for (model_display in c("Raw Mel Fbank", "ICA", "PCA")) {
  matching_models <- all_model_betas[grepl(paste0("^", model_display), all_model_betas)]
  model_beta_order <- c(model_beta_order, matching_models)
}

# Add VAE variants in correct order: AE, VAE, β-VAE (both TIMIT and Vowels)
ae_models <- all_model_betas[grepl("^AE", all_model_betas) & !grepl("DecAE", all_model_betas)]
vae_models <- all_model_betas[grepl("^VAE", all_model_betas) & !grepl("DecVAE", all_model_betas)]
beta_vae_models <- all_model_betas[grepl("^β-VAE", all_model_betas) & !grepl("β-DecVAE", all_model_betas)]

model_beta_order <- c(model_beta_order, ae_models, vae_models, beta_vae_models)

# Add DecVAE variants (both TIMIT and Vowels)
decae_fd <- all_model_betas[grepl("^DecAE \\+ FD", all_model_betas)]
decae_ewt <- all_model_betas[grepl("^DecAE \\+ EWT", all_model_betas)]
decvae_fd <- all_model_betas[grepl("^DecVAE \\+ FD", all_model_betas)]
decvae_ewt <- all_model_betas[grepl("^DecVAE \\+ EWT", all_model_betas)]
beta_decvae_fd <- all_model_betas[grepl("^β-DecVAE \\+ FD", all_model_betas)]
beta_decvae_ewt <- all_model_betas[grepl("^β-DecVAE \\+ EWT", all_model_betas)]

model_beta_order <- c(model_beta_order, decae_fd, decae_ewt, decvae_fd, decvae_ewt, beta_decvae_fd, beta_decvae_ewt)

# Remove duplicates and empty strings
model_beta_order <- unique(model_beta_order[model_beta_order != ""])

# Ensure proper ordering
dat$Model_Beta <- factor(dat$Model_Beta, levels = model_beta_order)
dat$Metric_Display <- factor(dat$Metric_Display, levels = display_metric_names)

# Create colors for different metrics
unique_metrics <- unique(dat$Metric_Display)
n_metrics <- length(unique_metrics)
metric_colors <- viridis(n_metrics, option = "turbo", end = yellow_block_threshold)
names(metric_colors) <- unique_metrics

p <- ggplot(dat, aes(x = Model_Beta, y = Value, fill = Metric_Display)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = metric_colors) +
  scale_y_continuous(limits = c(0, max(dat$Value) * 1.1),
                     breaks = pretty_breaks(n = 6),
                     expand = expansion(mult = c(0, 0.05)), 
                     labels = label_number(accuracy = 0.01)) +
  labs(
    title = "",
    x = "",
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
save_path_plot <- file.path(save_dir, "disentanglement_results_grouped.png")
ggsave(filename = save_path_plot, plot = p, width = 18, height = 8, dpi = 600, bg = "white")
cat("Saved grouped bar chart to:", save_path_plot, "\n")
cat("Completed disentanglement results analysis\n")