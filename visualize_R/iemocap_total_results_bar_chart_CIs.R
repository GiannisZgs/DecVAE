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
legend_text_size <- 20
legend_font_face <- "plain"
line_size <- 1.2
point_size <- 2.5

yellow_block_threshold <- 1.0
colors <- viridis(n = 8, option = "turbo", end = yellow_block_threshold)

# Load data from
parent_load_dir <- file.path("D:", "wandb_exports_for_figures", "iemocap_posttraining", "total_results")
current_script_experiment <- "supervised_performance"
selected_betas <- c(0,0.1,1)
use_row <- -1
models <- c("raw_mels","ICA","PCA","VAE","filter","ewt")
display_model_names <- c("Raw Mel Fbank",  "ICA", "PCA", "β-VAE", "β-DecVAE + FD","β-DecVAE + EWT")
display_metric_names <- c("Weighted Accuracy (ER)", "Unweighted Accuracy (ER)", "Weighted F1 (ER)",  
    "Weighted Accuracy (SI)", "Weighted F1 (SI)", "Weighted Accuracy (PR)", "Weighted F1 (PR)")
true_metric_names <- c("accuracy_emotion", "unweighted_accuracy_emotion", "f1_emotion", 
    "accuracy_speaker", "f1_speaker", "accuracy_phoneme", "f1_phoneme") #"accuracy_speaker","accuracy_phoneme",
true_branch_names <- c("Z_branch", "S_branch")
display_branch_names <- c("Fr.", "Seq.")
true_transfer_from_names <- c("vowels", "timit")
display_transfer_from_names <- c("Vowels", "TIMIT")
# Save data at
parent_save_dir <- file.path('..','figures','post-training','iemocap')
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

extract_branch_from_filename <- function(fname_no_ext) {
  # Check for branch indicators in filename
  for (branch in true_branch_names) {
    if (grepl(branch, fname_no_ext, ignore.case = TRUE)) {
      return(branch)
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
  branch <- extract_branch_from_filename(base)
  
  if (is.null(metric)) {
    cat("Skipping file:", basename(f), "- no metric found\n")
    next
  }
  
  cat("Processing:", basename(f), "for metric:", metric, "branch:", branch, "\n")
  
  # Debug: Check if this is unweighted accuracy
  if (metric == "unweighted_accuracy_emotion") {
    cat("DEBUG: Found unweighted accuracy emotion file with branch:", branch, "\n")
  }
  
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
  
  if (nrow(df) < 2) {
    cat("Warning: File", basename(f), "has less than 2 rows. Skipping.\n")
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
  processed_any <- FALSE
  for (col in cols) {
    # Extract model from column name
    model_val <- extract_model_from_column(col, models)
    
    if (is.null(model_val)) {
      next
    }
    
    # Extract transfer_from information
    transfer_from_val <- extract_transfer_from_column(col)
    
    # Check if this model requires branch information
    requires_branch <- model_val %in% c("filter", "ewt")
    
    # Skip if branch is required but not found, or if branch found but not required
    if (requires_branch && is.null(branch)) {
      if (metric == "unweighted_accuracy_emotion") {
        cat("DEBUG: Skipping", col, "- DecVAE model requires branch but branch is NULL\n")
      }
      next
    }
    if (!requires_branch && !is.null(branch)) {
      # For non-DecVAE models, only process if this is a Z_branch file or no branch specified
      if (branch != "Z_branch") {
        if (metric == "unweighted_accuracy_emotion") {
          cat("DEBUG: Skipping", col, "- non-DecVAE model with S_branch\n")
        }
        next
      }
    }
    
    # For beta-related models (filter, ewt, VAE), extract beta value
    if (model_val %in% c("filter", "ewt", "VAE")) {
      beta_val <- extract_beta_from_column(col)
      if (is.null(beta_val) || !(beta_val %in% selected_betas)) {
        if (metric == "unweighted_accuracy_emotion") {
          cat("DEBUG: Skipping", col, "- beta not found or not in selected betas\n")
        }
        next
      }
    } else {
      # For non-beta models (ICA, PCA, raw_mels), use NA for beta
      beta_val <- NA
    }
    
    # Extract mean value (first row) and CI (second row)
    col_values <- df[[col]]
    non_na_values <- col_values[!is.na(col_values)]
    
    if (length(non_na_values) < 2) {
      if (metric == "unweighted_accuracy_emotion") {
        cat("DEBUG: Skipping", col, "- insufficient data, only", length(non_na_values), "non-NA values\n")
      }
      next  # Skip columns without both mean and CI
    }
    
    mean_value <- non_na_values[1]
    ci_value <- non_na_values[2]
    
    # Transform specific metrics for both mean and CI
    if (metric %in% c("mi", "gaussian_tc_norm")) {
      mean_value <- 1 - mean_value
      if (mean_value < 0) mean_value <- 0
      # CI remains the same as it represents the range
    }
    
    if (metric == "unweighted_accuracy_emotion") {
      cat("DEBUG: Processing", col, "- model:", model_val, "beta:", beta_val, "branch:", branch, "transfer_from:", transfer_from_val, "value:", mean_value, "\n")
    }
    
    all_rows[[length(all_rows) + 1]] <- data.frame(
      Metric = metric,
      Beta = beta_val,
      Model = model_val,
      Branch = ifelse(requires_branch, branch, NA),
      Transfer_From = transfer_from_val,
      Value = mean_value,
      CI = ci_value,
      stringsAsFactors = FALSE
    )
    processed_any <- TRUE
  }
  
  if (metric == "unweighted_accuracy_emotion" && !processed_any) {
    cat("DEBUG: No columns processed for unweighted accuracy emotion in file:", basename(f), "\n")
    cat("DEBUG: Available columns:", paste(cols, collapse = ", "), "\n")
  }
}

if (length(all_rows) == 0) {
  stop("No valid data found for any combination.")
}

# Combine all data
dat <- bind_rows(all_rows)

# Handle transfer_from comparison - use specific datasets for different model types
transfer_from_mapping <- setNames(display_transfer_from_names, true_transfer_from_names)

# First, identify groups that have both vowels and timit variants
dat_with_transfer <- dat %>%
  filter(!is.na(Transfer_From)) %>%
  group_by(Metric, Beta, Model, Branch) %>%
  mutate(n_transfer_variants = n_distinct(Transfer_From)) %>%
  ungroup()

# Apply specific selection rules based on model type and branch
dat_transfer_selected <- dat_with_transfer %>%
  group_by(Metric, Beta, Model, Branch) %>%
  filter(
    # For DecVAE Z_branch: use Vowels
    (Model %in% c("filter", "ewt") & Branch == "Z_branch" & Transfer_From == "vowels") |
    # For DecVAE S_branch: use TIMIT  
    (Model %in% c("filter", "ewt") & Branch == "S_branch" & Transfer_From == "timit") |
    # For VAE: use TIMIT
    (Model == "VAE" & Transfer_From == "timit") |
    # For single variant groups, keep the only one
    (n_transfer_variants == 1)
  ) %>%
  slice(1) %>%  # In case of ties, take the first one
  ungroup() %>%
  select(-n_transfer_variants)

# For data without transfer_from information (raw_mels, ICA, PCA), keep as is
dat_no_transfer <- dat %>%
  filter(is.na(Transfer_From))

# Combine back - this should include raw_mels, ICA, PCA
dat <- bind_rows(dat_no_transfer, dat_transfer_selected)

# Debug: Check what models we have after filtering
cat("DEBUG: Models after transfer_from filtering:\n")
print(table(dat$Model))
cat("DEBUG: Sample of data after filtering:\n")
print(head(dat[, c("Model", "Beta", "Branch", "Transfer_From")]))

# Map model and metric names first, before any aggregation
model_mapping <- setNames(display_model_names, models)
metric_mapping <- setNames(display_metric_names, true_metric_names)
branch_mapping <- setNames(display_branch_names, true_branch_names)

dat$Model_Display <- model_mapping[dat$Model]
dat$Model_Display[is.na(dat$Model_Display)] <- dat$Model[is.na(dat$Model_Display)]

dat$Metric_Display <- metric_mapping[dat$Metric]
dat$Metric_Display[is.na(dat$Metric_Display)] <- dat$Metric[is.na(dat$Metric_Display)]

dat$Branch_Display <- branch_mapping[dat$Branch]
# Replace NA with empty string for proper concatenation
dat$Branch_Display[is.na(dat$Branch_Display)] <- ""

dat$Transfer_From_Display <- transfer_from_mapping[dat$Transfer_From]
# Replace NA with empty string for proper concatenation
dat$Transfer_From_Display[is.na(dat$Transfer_From_Display)] <- ""

# Debug: Check the data before creating Model_Beta
cat("DEBUG: Sample of data before Model_Beta creation:\n")
print(head(dat[, c("Model", "Model_Display", "Beta", "Branch", "Branch_Display", "Transfer_From", "Transfer_From_Display")]))

# Create combination labels for grouping - include branch and transfer_from information
# Use shortened model names based on beta values and preserve FD/EWT distinction
dat$Model_Beta <- ifelse(
  !is.na(dat$Branch) & dat$Branch_Display != "",  # Has branch info (DecVAE models)
  ifelse(is.na(dat$Beta), 
         paste0(dat$Branch_Display, " ", dat$Model_Display,
                ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), "")),
         ifelse(dat$Beta == 0,
                paste0(dat$Branch_Display, " DecAE + ", 
                       ifelse(dat$Model == "filter", "FD", "EWT"),
                       ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), "")),
                ifelse(dat$Beta == 1,
                       paste0(dat$Branch_Display, " DecVAE + ", 
                              ifelse(dat$Model == "filter", "FD", "EWT"),
                              ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), "")),
                       paste0(dat$Branch_Display, " β-DecVAE + ", 
                              ifelse(dat$Model == "filter", "FD", "EWT"),
                              ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), ""))))),
  # No branch info (other models)
  ifelse(is.na(dat$Beta), 
         paste0(dat$Model_Display,
                ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), "")),
         ifelse(dat$Beta == 0,
                ifelse(dat$Model_Display == "β-VAE", 
                       paste0("AE", ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), "")), 
                       paste0(dat$Model_Display, ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), ""))),
                ifelse(dat$Beta == 1,
                       ifelse(dat$Model_Display == "β-VAE", 
                               paste0("VAE", ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), "")), 
                               paste0(dat$Model_Display, ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), ""))),
                       paste0(dat$Model_Display, ifelse(!is.na(dat$Transfer_From) & dat$Transfer_From_Display != "", paste0(" (", dat$Transfer_From_Display, ")"), "")))))  # Keep β-VAE for β=0.1
)

# Debug: Check the Model_Beta after creation
cat("DEBUG: Sample of Model_Beta after creation:\n")
print(head(dat[, c("Model", "Model_Display", "Beta", "Branch", "Transfer_From", "Model_Beta")]))
cat("DEBUG: Unique Model_Beta values:\n")
print(unique(dat$Model_Beta))

# Remove duplicates based on all key columns INCLUDING the full Model_Beta to preserve all distinctions
dat <- dat %>%
  group_by(Metric, Beta, Model, Branch, Transfer_From, Model_Beta, Metric_Display) %>%
  summarise(
    Value = mean(Value, na.rm = TRUE),
    CI = mean(CI, na.rm = TRUE),
    .groups = 'drop'
  )

# Use only metric display names for legend (no branch suffix)
dat$Metric_Legend <- dat$Metric_Display

# Create metric-branch combination for grouping in significance testing
# For models without branches, use just the metric name
dat$Metric_Branch <- ifelse(
  !is.na(dat$Branch),
  paste0(dat$Metric_Display, " - ", dat$Branch_Display),
  dat$Metric_Display
)

# Calculate error bar bounds
dat$CI_lower <- dat$Value - dat$CI
dat$CI_upper <- dat$Value + dat$CI

# Create colors for different metrics (without branch distinction in legend)
unique_metrics_legend <- unique(dat$Metric_Legend)
n_metrics_legend <- length(unique_metrics_legend)
metric_colors <- viridis(n_metrics_legend, option = "turbo", end = yellow_block_threshold)
names(metric_colors) <- unique_metrics_legend

# Create proper ordering - fix the order to include all FD/EWT variants
model_beta_order <- c()

# Get all unique Model_Beta values from the data
all_model_betas <- unique(dat$Model_Beta)
cat("DEBUG: All Model_Beta values in data:\n")
print(all_model_betas)

# Add non-variational models first - use more flexible matching
for (model_display in c("Raw Mel Fbank", "ICA", "PCA")) {
  # Use grepl without fixed=TRUE for more flexible matching
  matching_models <- all_model_betas[grepl(paste0("^", model_display), all_model_betas)]
  model_beta_order <- c(model_beta_order, matching_models)
}

# Add VAE variants in correct order: AE, VAE, β-VAE
# Look for models that start with "AE", "VAE", or "β-VAE" (but not preceded by "Dec")
ae_models <- all_model_betas[grepl("^AE", all_model_betas) & !grepl("DecAE", all_model_betas)]
vae_models <- all_model_betas[grepl("^VAE", all_model_betas) & !grepl("DecVAE", all_model_betas)]
beta_vae_models <- all_model_betas[grepl("^β-VAE", all_model_betas) & !grepl("β-DecVAE", all_model_betas)]


model_beta_order <- c(model_beta_order, ae_models, vae_models, beta_vae_models)

# Add DecVAE variants for each branch in correct order
for (branch_display in display_branch_names) {
  # Add DecAE, DecVAE, β-DecVAE variants for this branch with both FD and EWT
  decae_pattern <- paste0("^", branch_display, " DecAE")
  decvae_pattern <- paste0("^", branch_display, " DecVAE")
  beta_decvae_pattern <- paste0("^", branch_display, " β-DecVAE")
  
  decae_models <- all_model_betas[grepl(decae_pattern, all_model_betas)]
  decvae_models <- all_model_betas[grepl(decvae_pattern, all_model_betas) & !grepl("β-DecVAE", all_model_betas)]
  beta_decvae_models <- all_model_betas[grepl(beta_decvae_pattern, all_model_betas)]
  
  # Within each type, order by FD first, then EWT
  decae_fd <- decae_models[grepl("\\+ FD", decae_models)]
  decae_ewt <- decae_models[grepl("\\+ EWT", decae_models)]
  decvae_fd <- decvae_models[grepl("\\+ FD", decvae_models)]
  decvae_ewt <- decvae_models[grepl("\\+ EWT", decvae_models)]
  beta_decvae_fd <- beta_decvae_models[grepl("\\+ FD", beta_decvae_models)]
  beta_decvae_ewt <- beta_decvae_models[grepl("\\+ EWT", beta_decvae_models)]
  
  model_beta_order <- c(model_beta_order, decae_fd, decae_ewt, decvae_fd, decvae_ewt, beta_decvae_fd, beta_decvae_ewt)
}

# Remove any remaining duplicates and filter out empty strings
model_beta_order <- unique(model_beta_order[model_beta_order != ""])

cat("DEBUG: Final model_beta_order:\n")
print(model_beta_order)

# Ensure proper ordering
dat$Model_Beta <- factor(dat$Model_Beta, levels = model_beta_order)
dat$Metric_Legend <- factor(dat$Metric_Legend, levels = display_metric_names)

p <- ggplot(dat, aes(x = Model_Beta, y = Value, fill = Metric_Legend)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper), 
                position = position_dodge(width = 0.8), 
                width = 0.5, 
                linewidth = 1.2,
                color = "black") +
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
save_path_plot <- file.path(save_dir, "total_model_results_grouped_with_CIs.png")
ggsave(filename = save_path_plot, plot = p, width = 18, height = 8, dpi = 600, bg = "white")
cat("Saved grouped bar chart with confidence intervals to:", save_path_plot, "\n")
cat("Completed total model results analysis with confidence intervals\n")
