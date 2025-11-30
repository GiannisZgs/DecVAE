#' SI Figure 22: pre-training phase ablations for the SimVowels dataset.
#' The current script generates figure SI_fig22d, which is a demonstration of
#' the SSL decomposition contrastive loss comparing waveform with Mel fbank inputs.

library(vscDebugger)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(scales)
library(viridis)
library(ggnewscale)
library(stringr)
library(cowplot)

# Style and font parameters
plot_font_family <- "Arial"
plot_title_size <- 28
title_font_face <- "plain"
plot_subtitle_size <- 22
axis_title_size <- 28
axis_text_size <- 18  
legend_title_size <- 28
legend_text_size <- 28
legend_font_face <- "plain"
line_size <- 1.2
point_size <- 2.5

yellow_block_threshold <- 1.0
colors <- viridis(n = 8, option = "turbo", end = yellow_block_threshold)

# Load data from
parent_load_dir <- file.path('..', 'data', 'results_wandb_exports_for_figures', 'vowels_pretraining')
current_script_experiment <- "FD_input_type_ablation_loss_demo"
model_type <- "dual"
set <- "validation"
current_focus <- "input_type"
selected_betas <- c(0, 1, 5) # can be one or more values
loss_file1 <- paste0(set, "_divergence_negative_Z.csv")
loss_file2 <- paste0(set, "_divergence_positive_Z.csv")
loss_file3 <- paste0(set, "_cross_entropy_negative_Z.csv")
loss_file4 <- paste0(set, "_cross_entropy_positive_Z.csv")
loss_file5 <- paste0(set, "_decomposition_loss_Z.csv")
loss_file6 <- paste0(set, "_decomposition_loss_S.csv")
loss_file7 <- paste0(set, "_prior_loss_Z.csv")

#loss_file
csv_file_path1 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file1)
csv_file_path2 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file2)
csv_file_path3 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file3)
csv_file_path4 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file4)
csv_file_path5 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file5)
csv_file_path6 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file6)
csv_file_path7 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file7)

# Save data at
parent_save_dir <- file.path('..','supplementary_figures','SI_fig_22_pre-training_losses_vowels','pre-training_losses_vowels')
save_dir <- file.path(parent_save_dir, current_script_experiment, model_type, set)
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

save_path1 <- file.path(save_dir, "input_type_divergences_FD_beta_0.png")
save_path2 <- file.path(save_dir, "input_type_cross_entropies_FD_beta_0.png")
save_path3 <- file.path(save_dir, "input_type_decomposition_losses_FD_beta_0.png")
save_path4 <- file.path(save_dir, "input_type_prior_loss_Z_FD_beta_0.png")
save_path_panel <- file.path(save_dir, "input_type_losses_panel_figure.png")

# Function to create line chart from multiple CSV data files
create_line_chart_multi <- function(data_paths, 
                                   x_col = "Epoch", 
                                   value_cols = NULL,
                                   title = "Training Progress",
                                   subtitle = NULL,
                                   x_label = "Epoch",
                                   y_label = "Value",
                                   legend_title = "Variables",
                                   y_limits = NULL,
                                   zoom_inset = NULL,
                                   save_path = NULL,
                                   width = 12,
                                   height = 8,
                                   show_legend = TRUE) {  # Add legend control parameter
  
  # Load and concatenate data from multiple files
  cat("Loading data from", length(data_paths), "files...\n")
  all_data <- list()
  
  for (i in seq_along(data_paths)) {
    cat("Loading file", i, ":", data_paths[i], "\n")
    data <- read_csv(data_paths[i], show_col_types = FALSE)
    
    # Apply preprocessing to each file individually
    cat("Preprocessing file", i, "...\n")
    
    # Step 1: Remove columns that contain MIN or MAX
    original_cols <- colnames(data)
    min_max_cols <- grep("MIN|MAX", original_cols, ignore.case = TRUE, value = TRUE)
    if (length(min_max_cols) > 0) {
      cat("Removing MIN/MAX columns from file", i, ":", paste(min_max_cols, collapse = ", "), "\n")
      data <- data %>% select(-any_of(min_max_cols))
    }
    
    # Step 2: Clean column names by removing everything after " - "
    new_col_names <- colnames(data)
    cleaned_col_names <- gsub(" - .*$", "", new_col_names)
    colnames(data) <- cleaned_col_names
    
    # Step 3: Filter by selected betas and rename for components, negative-positive, or input_type case
    if (exists("current_focus") && current_focus == "components") {
      if (exists("selected_betas") && length(selected_betas) > 0) {
        processed_data_list <- list()
        
        for (beta_idx in seq_along(selected_betas)) {
          beta_val <- selected_betas[beta_idx]
          beta_filter_pattern <- paste0("_[bB]", beta_val, "_")
          cat("Filtering columns for beta =", beta_val, "using pattern:", beta_filter_pattern, "\n")
          
          # Find columns that contain the selected beta value
          beta_filter_matches <- grepl(beta_filter_pattern, cleaned_col_names, ignore.case = TRUE)
          
          if (any(beta_filter_matches)) {
            # Find the data column (not Step/Epoch)
            data_cols <- cleaned_col_names[beta_filter_matches & !cleaned_col_names %in% c("Step", "Epoch")]
            
            if (length(data_cols) > 0) {
              # Take the first matching data column
              selected_col <- data_cols[1]
              cat("Selected column from file", i, "for beta", beta_val, ":", selected_col, "\n")
              
              # Extract component pattern and create new name
              component_pattern <- "_([0-9]+)_([0-9]+)_"
              component_match <- regmatches(data_paths[i], regexec(component_pattern, data_paths[i], ignore.case = TRUE))
              
              if (length(component_match[[1]]) > 1) {
                x_value <- as.numeric(component_match[[1]][2])
                y_value <- as.numeric(component_match[[1]][3])
                
                # Convert numbers to labels
                first_label <- if (x_value == 0) "X" else paste0("OC", x_value)
                second_label <- if (y_value == 0) "X" else paste0("OC", y_value)
                
                # Create the new column name with beta info
                new_col_name <- paste0(first_label, "-", second_label, "_β", beta_val)
                cat("Creating column", new_col_name, "\n")
                
                # Keep only the selected data column and rename it
                beta_data <- data %>% select(all_of(selected_col))
                colnames(beta_data) <- new_col_name
                processed_data_list[[paste0("beta_", beta_val)]] <- beta_data
              }
            }
          }
        }
        
        # Combine all beta data for this file
        if (length(processed_data_list) > 0) {
          data <- bind_cols(processed_data_list)
        } else {
          # No matching data found, create empty data frame
          data <- data.frame()
        }
      } else {
        # No selected_betas, create empty data frame
        data <- data.frame()
      }
    } else if (exists("current_focus") && current_focus == "input_type") {
      if (exists("selected_betas") && length(selected_betas) > 0) {
        processed_data_list <- list()
        
        # Determine if this is negative or positive file based on filename
        is_negative <- grepl("negative", data_paths[i], ignore.case = TRUE)
        is_positive <- grepl("positive", data_paths[i], ignore.case = TRUE)
        
        # Determine branch type for decomposition losses
        is_Z_branch <- grepl("_Z\\.csv$", data_paths[i], ignore.case = TRUE)
        is_S_branch <- grepl("_S\\.csv$", data_paths[i], ignore.case = TRUE)
        
        if (is_negative || is_positive) {
          file_type <- if (is_negative) "neg" else "pos"
          cat("Processing", file_type, "file\n")
          
          for (beta_idx in seq_along(selected_betas)) {
            beta_val <- selected_betas[beta_idx]
            beta_filter_pattern <- paste0("_[bB]", beta_val, "_")
            cat("Filtering columns for beta =", beta_val, "using pattern:", beta_filter_pattern, "\n")
            
            # Find columns that contain the selected beta value
            beta_filter_matches <- grepl(beta_filter_pattern, cleaned_col_names, ignore.case = TRUE)
            
            if (any(beta_filter_matches)) {
              # Find the data columns (not Step/Epoch)
              data_cols <- cleaned_col_names[beta_filter_matches & !cleaned_col_names %in% c("Step", "Epoch")]
              
              for (selected_col in data_cols) {
                cat("Selected column for beta", beta_val, ":", selected_col, "\n")
                
                # Extract input type from column name
                # Be more explicit about detection
                has_waveform <- grepl("waveform", selected_col, ignore.case = TRUE)
                has_mel <- grepl("mel", selected_col, ignore.case = TRUE)
                
                if (has_waveform) {
                  input_type <- "waveform"
                } else if (has_mel) {
                  input_type <- "mel"
                } else {
                  # If neither waveform nor mel is explicitly mentioned, assume mel
                  # This handles cases where column names might be like "dual_b0_Z" without input type
                  input_type <- "mel"
                  cat("Warning: Could not detect input type from column name", selected_col, ", defaulting to mel\n")
                }
                
                cat("Detected input type:", input_type, "for column:", selected_col, "\n")
                
                # Create descriptive name: input_type_pos/neg_bX (e.g., "waveform_pos_b0", "mel_neg_b0")
                new_col_name <- paste0(input_type, "_", file_type, "_b", beta_val)
                cat("Creating column", new_col_name, "\n")
                
                # Keep only the selected data column and rename it
                beta_data <- data %>% select(all_of(selected_col))
                colnames(beta_data) <- new_col_name
                processed_data_list[[paste0(input_type, "_", file_type, "_beta_", beta_val)]] <- beta_data
              }
            }
          }
        } else if (is_Z_branch || is_S_branch) {
          # Handle decomposition loss files (Z or S branch)
          branch_type <- if (is_Z_branch) "Z" else "S"
          cat("Processing", branch_type, "branch file\n")
          
          for (beta_idx in seq_along(selected_betas)) {
            beta_val <- selected_betas[beta_idx]
            beta_filter_pattern <- paste0("_[bB]", beta_val, "_")
            cat("Filtering columns for beta =", beta_val, "using pattern:", beta_filter_pattern, "\n")
            
            # Find columns that contain the selected beta value
            beta_filter_matches <- grepl(beta_filter_pattern, cleaned_col_names, ignore.case = TRUE)
            
            if (any(beta_filter_matches)) {
              # Find the data columns (not Step/Epoch)
              data_cols <- cleaned_col_names[beta_filter_matches & !cleaned_col_names %in% c("Step", "Epoch")]
              
              for (selected_col in data_cols) {
                cat("Selected column for beta", beta_val, ":", selected_col, "\n")
                
                # Extract input type from column name
                input_type <- if (grepl("waveform", selected_col, ignore.case = TRUE)) "waveform" else "mel"
                
                # Create descriptive name: input_type_branch_bX (e.g., "waveform_Z_b0", "mel_S_b0")
                new_col_name <- paste0(input_type, "_", branch_type, "_b", beta_val)
                cat("Creating column", new_col_name, "\n")
                
                # Keep only the selected data column and rename it
                beta_data <- data %>% select(all_of(selected_col))
                colnames(beta_data) <- new_col_name
                processed_data_list[[paste0(input_type, "_", branch_type, "_beta_", beta_val)]] <- beta_data
              }
            }
          }
        }
        
        # Combine all beta data for this file
        if (length(processed_data_list) > 0) {
          data <- bind_cols(processed_data_list)
        } else {
          # No matching data found, create empty data frame
          data <- data.frame()
        }
      } else {
        # No selected_betas, create empty data frame
        data <- data.frame()
      }
    } else if (exists("current_focus") && current_focus == "negative-positive") {
      if (exists("selected_betas") && length(selected_betas) > 0) {
        processed_data_list <- list()
        
        # Determine if this is negative or positive file based on filename
        is_negative <- grepl("negative", data_paths[i], ignore.case = TRUE)
        is_positive <- grepl("positive", data_paths[i], ignore.case = TRUE)
        
        if (is_negative || is_positive) {
          file_type <- if (is_negative) "Negative" else "Positive"
          cat("Processing", file_type, "file\n")
          
          for (beta_idx in seq_along(selected_betas)) {
            beta_val <- selected_betas[beta_idx]
            beta_filter_pattern <- paste0("_[bB]", beta_val, "_")
            cat("Filtering columns for beta =", beta_val, "using pattern:", beta_filter_pattern, "\n")
            
            # Find columns that contain the selected beta value
            beta_filter_matches <- grepl(beta_filter_pattern, cleaned_col_names, ignore.case = TRUE)
            
            if (any(beta_filter_matches)) {
              # Find the data column (not Step/Epoch)
              data_cols <- cleaned_col_names[beta_filter_matches & !cleaned_col_names %in% c("Step", "Epoch")]
              
              if (length(data_cols) > 0) {
                # Take the first matching data column
                selected_col <- data_cols[1]
                cat("Selected column from file", i, "for beta", beta_val, ":", selected_col, "\n")
                
                # Create the new column name with beta info and file type
                new_col_name <- paste0(file_type, "_β", beta_val)
                cat("Creating column", new_col_name, "\n")
                
                # Keep only the selected data column and rename it
                beta_data <- data %>% select(all_of(selected_col))
                colnames(beta_data) <- new_col_name
                processed_data_list[[paste0("beta_", beta_val)]] <- beta_data
              }
            }
          }
        }
        
        # Combine all beta data for this file
        if (length(processed_data_list) > 0) {
          data <- bind_cols(processed_data_list)
        } else {
          # No matching data found, create empty data frame
          data <- data.frame()
        }
      } else {
        # No selected_betas, create empty data frame
        data <- data.frame()
      }
    } else if (exists("current_focus") && current_focus == "branch_comparison") {
      if (exists("selected_betas") && length(selected_betas) > 0) {
        processed_data_list <- list()
        
        # Determine file type (negative/positive) and model type based on file index and filename
        is_negative <- grepl("negative", data_paths[i], ignore.case = TRUE)
        is_positive <- grepl("positive", data_paths[i], ignore.case = TRUE)
        is_Z_branch <- grepl("_Z\\.csv$", data_paths[i], ignore.case = TRUE)
        is_S_branch <- grepl("_S\\.csv$", data_paths[i], ignore.case = TRUE)
        
        if (is_negative || is_positive) {
          file_type <- if (is_negative) "negative" else "positive"
          
          # Determine model type and branch based on file index
          if (i <= 2) {
            # Files 7_1 and 7_2: dual model, Z branch
            model_type_label <- "dual"
            branch_label <- "Z"
          } else if (i <= 4) {
            # Files 7_3 and 7_4: dual model, S branch  
            model_type_label <- "dual"
            branch_label <- "S"
          } else if (i <= 6) {
            # Files 7_5 and 7_6: Z_only model, Z branch
            model_type_label <- "single"
            branch_label <- "Z"
          } else {
            # Files 7_7 and 7_8: S_only model, S branch
            model_type_label <- "single"
            branch_label <- "S"
          }
          
          cat("Processing", file_type, "file with", model_type_label, "model type and", branch_label, "branch\n")
          
          for (beta_idx in seq_along(selected_betas)) {
            beta_val <- selected_betas[beta_idx]
            beta_filter_pattern <- paste0("_[bB]", beta_val, "_")
            cat("Filtering columns for beta =", beta_val, "using pattern:", beta_filter_pattern, "\n")
            
            # Find columns that contain the selected beta value
            beta_filter_matches <- grepl(beta_filter_pattern, cleaned_col_names, ignore.case = TRUE)
            
            if (any(beta_filter_matches)) {
              # Find the data column (not Step/Epoch)
              data_cols <- cleaned_col_names[beta_filter_matches & !cleaned_col_names %in% c("Step", "Epoch")]
              
              if (length(data_cols) > 0) {
                # Take the first matching data column
                selected_col <- data_cols[1]
                cat("Selected column from file", i, "for beta", beta_val, ":", selected_col, "\n")
                
                # Create the new column name with model type, branch, and file type
                new_col_name <- paste0("divergence_", file_type, "_", model_type_label, "_", branch_label, "_β", beta_val)
                cat("Creating column", new_col_name, "\n")
                
                # Keep only the selected data column and rename it
                beta_data <- data %>% select(all_of(selected_col))
                colnames(beta_data) <- new_col_name
                processed_data_list[[paste0("beta_", beta_val)]] <- beta_data
              }
            }
          }
        }
        
        # Combine all beta data for this file
        if (length(processed_data_list) > 0) {
          data <- bind_cols(processed_data_list)
        } else {
          # No matching data found, create empty data frame
          data <- data.frame()
        }
      } else {
        # No selected_betas, create empty data frame
        data <- data.frame()
      }
    } else {
      # Not components or negative-positive focus, create empty data frame
      data <- data.frame()
    }
    
    all_data[[i]] <- data
  }
  
  # Check row lengths and trim to minimum length to ensure compatibility
  cat("Checking row lengths across all datasets...\n")
  row_lengths <- sapply(all_data, function(x) if(is.data.frame(x) && nrow(x) > 0) nrow(x) else 0)
  cat("Row lengths:", paste(row_lengths, collapse = ", "), "\n")
  
  # Filter out empty datasets
  non_empty_data <- all_data[row_lengths > 0]
  non_empty_lengths <- row_lengths[row_lengths > 0]
  
  if (length(non_empty_data) == 0) {
    stop("No valid data found in any of the files")
  }
  
  # Find minimum length
  min_length <- min(non_empty_lengths)
  cat("Minimum row length:", min_length, "\n")
  
  # Trim all datasets to minimum length
  if (any(non_empty_lengths != min_length)) {
    cat("Trimming datasets to match minimum length of", min_length, "rows\n")
    for (i in seq_along(non_empty_data)) {
      if (nrow(non_empty_data[[i]]) > min_length) {
        cat("Trimming dataset", i, "from", nrow(non_empty_data[[i]]), "to", min_length, "rows\n")
        non_empty_data[[i]] <- non_empty_data[[i]][1:min_length, , drop = FALSE]
      }
    }
  }
  
  # Concatenate all datasets by columns (now guaranteed to have same number of rows)
  data <- bind_cols(non_empty_data)
  cat("Concatenated data has", nrow(data), "rows and", ncol(data), "columns\n")
  
  # Preprocessing steps (minimal since files already preprocessed)
  cat("Applying final preprocessing steps...\n")
  
  # Create Epoch column as integer sequence (like in create_line_chart)
  data$Epoch <- 0:(nrow(data)-1)
  cat("Created new Epoch column with values 0 to", nrow(data)-1, "\n")

  # Print column names to help user identify available columns
  cat("Available columns after preprocessing:\n")
  print(colnames(data))
  
  # If value_cols is not specified, use all numeric columns except x_col
  if (is.null(value_cols)) {
    numeric_cols <- sapply(data, is.numeric)
    value_cols <- names(data)[numeric_cols & names(data) != x_col]
    cat("Auto-detected numeric columns for plotting:", paste(value_cols, collapse = ", "), "\n")
  }
  
  # Check if specified columns exist
  missing_cols <- setdiff(c(x_col, value_cols), colnames(data))
  if (length(missing_cols) > 0) {
    stop("Missing columns in data: ", paste(missing_cols, collapse = ", "))
  }
  
  # Prepare data for plotting by reshaping to long format
  plot_data <- data %>%
    select(all_of(c(x_col, value_cols))) %>%
    pivot_longer(cols = all_of(value_cols), 
                 names_to = "Variable", 
                 values_to = "Value") %>%
    filter(!is.na(Value))  # Remove any NA values
  
  # Extract beta values and component pairs from variable names
  plot_data$Beta <- str_extract(plot_data$Variable, "b[0-9\\.]+")
  
  # Handle different focus types
  if (exists("current_focus") && current_focus == "negative-positive") {
    # For negative-positive focus, extract the file type (Negative/Positive)
    plot_data$Component_Pair <- str_remove(plot_data$Variable, "_b[0-9\\.]+")
    plot_data$Pair_Type <- plot_data$Component_Pair  # Use Negative/Positive directly
  } else if (exists("current_focus") && current_focus == "input_type") {
    # For input_type focus, extract information from column names
    # Column format: "waveform_pos_b0", "mel_neg_b0", "waveform_Z_b0", "mel_S_b0"
    
    # Extract input type (waveform or mel)
    plot_data$Input_Type <- str_extract(plot_data$Variable, "^(waveform|mel)")
    
    # Extract pos/neg or Z/S from the middle part
    plot_data$File_Type <- str_extract(plot_data$Variable, "_(pos|neg|Z|S)_")
    plot_data$File_Type <- str_remove_all(plot_data$File_Type, "_")  # Remove underscores
    
    # For grouping and legend purposes
    plot_data$Component_Pair <- paste0(plot_data$Input_Type, "_", plot_data$File_Type)
    plot_data$Pair_Type <- plot_data$File_Type  # Use pos/neg/Z/S for separation
  } else if (exists("current_focus") && current_focus == "branch_comparison") {
    # For branch comparison focus, extract model type, branch, and divergence type
    plot_data$Component_Pair <- str_remove(plot_data$Variable, "_β[0-9\\.]+")
    plot_data$Pair_Type <- ifelse(grepl("negative", plot_data$Component_Pair), "Negative", "Positive")
    
    # Extract model type (dual/single) and branch (Z/S) for better grouping
    plot_data$Model_Type <- ifelse(grepl("_dual_", plot_data$Component_Pair), "Dual", "Single")
    plot_data$Branch <- str_extract(plot_data$Component_Pair, "[ZS](?=_β|$)")
  } else {
    # For components focus, extract component pairs
    plot_data$Component_Pair <- str_remove(plot_data$Variable, "_β[0-9\\.]+")
    plot_data$Pair_Type <- ifelse(grepl("X", plot_data$Component_Pair), "Positive", "Negative")
  }
  
  # Clean beta values for legend (remove b prefix)
  plot_data$Beta_Clean <- str_remove(plot_data$Beta, "b")
  
  # Create linetype mapping for beta values (only for components focus)
  if (exists("current_focus") && current_focus == "components") {
    beta_values <- unique(plot_data$Beta_Clean)
    linetype_mapping <- c("solid", "dashed", "dotted")[1:length(beta_values)]
    names(linetype_mapping) <- beta_values
  }
  
  # Create color palettes for different cases
  if (exists("current_focus") && (current_focus == "negative-positive" || current_focus == "components")) {
    # Cold colors for positive, warm colors for negative
    positive_colors <- c("#4575b4", "#74add1", "#abd9e9")  # Blues
    negative_colors <- c("#d73027", "#f46d43", "#fdae61")  # Reds/oranges
  } else if (exists("current_focus") && current_focus == "input_type") {
    # Create distinct colors for input types and beta values
    # Cool colors for waveform, warm colors for mel
    waveform_colors <- c("#2166ac", "#4575b4", "#74add1", "#abd9e9")  # Blues
    mel_colors <- c("#d73027", "#f46d43", "#fdae61", "#fee090")  # Reds/oranges
  } else if (exists("current_focus") && current_focus == "branch_comparison") {
    # For branch comparison, create distinct colors for each model_type + branch combination
    # Cold colors for positive divergences, warm colors for negative divergences
    branch_colors <- c(
      "dual_z" = "#2166ac",     # Dark blue for dual Z
      "dual_s" = "#5aae61",     # Green for dual S  
      "single_z" = "#053061",   # Very dark blue for single Z
      "single_s" = "#35978f"    # Teal for single S
    )
    
    # Create model_branch combination in plot_data for mapping
    plot_data$Model_Branch <- paste0(tolower(plot_data$Model_Type), "_", tolower(plot_data$Branch))
    
    # Create color palette based on the model_branch combinations
    unique_combinations <- unique(plot_data$Model_Branch)
    color_palette <- branch_colors[unique_combinations]
    names(color_palette) <- unique_combinations
    
    # Create formal legend labels
    formal_labels <- c(
      "dual_z" = "Dual Latent - Z",
      "dual_s" = "Dual Latent - S", 
      "single_z" = "Single Latent - Z",
      "single_s" = "Single Latent - S"
    )
    
    # Map formal labels to the color palette
    color_palette_formal <- color_palette
    names(color_palette_formal) <- formal_labels[names(color_palette)]
    
    # Add formal labels to plot_data for plotting
    plot_data$Model_Branch_Formal <- formal_labels[plot_data$Model_Branch]
    
    # Debug: print the mapping to verify colors are assigned
    cat("Model-Branch combinations found:", paste(unique_combinations, collapse = ", "), "\n")
    cat("Color mapping:", paste(names(color_palette_formal), "=", color_palette_formal, collapse = ", "), "\n")
  }
  
  # Separate different input types for input_type focus
  if (exists("current_focus") && current_focus == "input_type") {
    # Since each plot now handles only one file (either positive or negative), 
    # we just separate by input type
    waveform_data <- plot_data %>% filter(Input_Type == "waveform")
    mel_data <- plot_data %>% filter(Input_Type == "mel")
  } else {
    # For other focus types, separate positive and negative pairs
    positive_pairs <- plot_data %>% filter(Pair_Type == "positive")
    negative_pairs <- plot_data %>% filter(Pair_Type == "negative")
  }
  
  # Create palettes after separating pairs
  if (exists("current_focus") && (current_focus == "negative-positive" || current_focus == "components")) {
    if (current_focus == "negative-positive") {
      # Assign colors based on number of beta values
      n_betas <- length(unique(plot_data$Beta_Clean))
      positive_palette <- positive_colors[1:n_betas]
      negative_palette <- negative_colors[1:n_betas]
    } else if (current_focus == "components") {
      # Assign colors based on number of component pairs
      positive_vars <- unique(positive_pairs$Component_Pair)
      negative_vars <- unique(negative_pairs$Component_Pair)
      positive_palette <- positive_colors[1:length(positive_vars)]
      negative_palette <- negative_colors[1:length(negative_vars)]
    }
  } else if (exists("current_focus") && current_focus == "input_type") {
    # Assign colors based on number of beta values for each input type
    n_betas <- length(unique(plot_data$Beta_Clean))
    waveform_palette <- waveform_colors[1:n_betas]
    mel_palette <- mel_colors[1:n_betas]
  }
  
  # Sort legend entries based on current focus
  if (exists("current_focus") && current_focus == "components") {
    # Sort component legend entries logically within categories
    positive_vars <- unique(positive_pairs$Component_Pair)
    negative_vars <- unique(negative_pairs$Component_Pair)
    
    positive_pairs$Component_Pair <- factor(positive_pairs$Component_Pair, levels = positive_vars)
    negative_pairs$Component_Pair <- factor(negative_pairs$Component_Pair, levels = negative_vars)
  } else if (exists("current_focus") && current_focus == "negative-positive") {
    # For negative-positive, use the type names directly
    positive_vars <- unique(positive_pairs$Component_Pair)
    negative_vars <- unique(negative_pairs$Component_Pair)
    
    positive_pairs$Component_Pair <- factor(positive_pairs$Component_Pair, levels = positive_vars)
    negative_pairs$Component_Pair <- factor(negative_pairs$Component_Pair, levels = negative_vars)
  } 
  
  # Create the line plot with appropriate styling based on focus type
  if (exists("current_focus") && current_focus == "negative-positive") {
    # Negative-positive case: use color only, no line styles
    p <- ggplot() +
      # Plot positive pairs with cold colors
      geom_line(data = positive_pairs, 
                aes(x = !!sym(x_col), y = Value, color = Beta_Clean), 
                size = line_size, alpha = 0.8) +
      geom_point(data = positive_pairs %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1), 
                 aes(x = !!sym(x_col), y = Value, color = Beta_Clean),
                 size = point_size, alpha = 0.9) +
      scale_color_manual(name = "Positive β",
                         values = positive_palette,
                         guide = guide_legend(order = 1)) +
      new_scale_color() +
      # Plot negative pairs with warm colors
      geom_line(data = negative_pairs, 
                aes(x = !!sym(x_col), y = Value, color = Beta_Clean), 
                size = line_size, alpha = 0.8) +
      geom_point(data = negative_pairs %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1), 
                 aes(x = !!sym(x_col), y = Value, color = Beta_Clean),
                 size = point_size, alpha = 0.9) +
      scale_color_manual(name = "Negative β",
                         values = negative_palette,
                         guide = guide_legend(order = 2))
  } else if (exists("current_focus") && current_focus == "input_type") {
    # Input type case: create comprehensive plot with separate legend areas
    
    # Create linetype mapping for beta values
    unique_betas <- sort(unique(plot_data$Beta_Clean))
    beta_linetypes <- c("solid", "dashed", "dotted", "dotdash")[1:length(unique_betas)]
    names(beta_linetypes) <- unique_betas
    
    # Create combination variable for color mapping (Input_Type + File_Type)
    plot_data$Type_Combo <- paste0(plot_data$Input_Type, "_", plot_data$File_Type)
    
    # Create base color palette for input types only
    input_type_colors <- c(
      "waveform" = "#4575b4",   # Blue for waveform
      "mel" = "#d73027"         # Red for mel
    )
    
    # Create condition/branch labels mapping
    condition_labels <- c(
      "pos" = "Positive",
      "neg" = "Negative", 
      "Z" = "Z-branch",
      "S" = "S-branch"
    )
    
    # Add separate columns for input type and condition
    plot_data$Input_Type_Display <- plot_data$Input_Type
    plot_data$Input_Type_Display <- factor(plot_data$Input_Type_Display, 
                                          levels = c("waveform", "mel"))
    
    plot_data$Condition_Display <- condition_labels[plot_data$File_Type]
    plot_data$Condition_Display <- factor(plot_data$Condition_Display, 
                                         levels = c("Positive", "Negative", "Z-branch", "S-branch"))
    
    # Create shape mapping for conditions (to distinguish them visually)
    condition_shapes <- c(
      "Positive" = 16,    # Solid circle
      "Negative" = 17,    # Solid triangle
      "Z-branch" = 15,    # Solid square
      "S-branch" = 18     # Solid diamond
    )
    
    # Debug output
    cat("Input types found:", paste(unique(plot_data$Input_Type_Display), collapse = ", "), "\n")
    cat("Conditions found:", paste(unique(plot_data$Condition_Display), collapse = ", "), "\n")
    
    p <- ggplot(plot_data, aes(x = !!sym(x_col), y = Value)) +
      geom_line(aes(color = Input_Type_Display, linetype = Beta_Clean, group = Variable), 
                size = line_size, alpha = 0.8) +
      geom_point(data = plot_data %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1), 
                 aes(color = Input_Type_Display, shape = Condition_Display),
                 size = point_size * 1.5, alpha = 0.9) +
      scale_color_manual(name = "Input Type",
                         values = input_type_colors,
                         guide = guide_legend(order = 1, 
                                            override.aes = list(shape = 16, size = 4, linetype = "solid"))) +
      scale_shape_manual(name = "Condition",
                         values = condition_shapes,
                         guide = guide_legend(order = 2,
                                            override.aes = list(size = 4, linetype = "blank"))) +
      scale_linetype_manual(name = "β Value", 
                            values = beta_linetypes,
                            guide = guide_legend(order = 3,
                                               override.aes = list(shape = NA, size = 1)))
  } else if (exists("current_focus") && current_focus == "branch_comparison") {
    # Branch comparison case: use different color schemes for positive vs negative divergences
    # Create linetype mapping for beta values
    beta_values <- unique(plot_data$Beta_Clean)
    linetype_mapping <- c("solid", "dashed", "dotted")[1:length(beta_values)]
    names(linetype_mapping) <- beta_values
    
    # Create separate color palettes for positive (cold) and negative (warm) divergences
    positive_branch_colors <- c(
      "dual_z" = "#2166ac",     # Dark blue for dual Z
      "dual_s" = "#5aae61",     # Green for dual S  
      "single_z" = "#053061",   # Very dark blue for single Z
      "single_s" = "#35978f"    # Teal for single S
    )
    
    negative_branch_colors <- c(
      "dual_z" = "#d73027",     # Red for dual Z
      "dual_s" = "#f46d43",     # Orange-red for dual S
      "single_z" = "#a50026",   # Dark red for single Z
      "single_s" = "#fd8d3c"    # Orange for single S
    )
    
    # Create formal labels for positive and negative pairs
    positive_formal_labels <- c(
      "dual_z" = "Dual Latent - Z",
      "dual_s" = "Dual Latent - S", 
      "single_z" = "Single Latent - Z",
      "single_s" = "Single Latent - S"
    )
    
    negative_formal_labels <- c(
      "dual_z" = "Dual Latent - Z",
      "dual_s" = "Dual Latent - S", 
      "single_z" = "Single Latent - Z",
      "single_s" = "Single Latent - S"
    )
    
    # Create color palettes with formal labels
    positive_combinations <- unique(positive_pairs$Model_Branch)
    negative_combinations <- unique(negative_pairs$Model_Branch)
    
    positive_palette <- positive_branch_colors[positive_combinations]
    names(positive_palette) <- positive_formal_labels[positive_combinations]
    
    negative_palette <- negative_branch_colors[negative_combinations]
    names(negative_palette) <- negative_formal_labels[negative_combinations]
    
    p <- ggplot() +
      # Plot positive pairs with cold colors
      geom_line(data = positive_pairs, 
                aes(x = !!sym(x_col), y = Value, color = Model_Branch_Formal, linetype = Beta_Clean), 
                size = line_size, alpha = 0.8) +
      geom_point(data = positive_pairs %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1), 
                 aes(x = !!sym(x_col), y = Value, color = Model_Branch_Formal),
                 size = point_size, alpha = 0.9) +
      scale_color_manual(name = "Positive Divergences",
                         values = positive_palette,
                         guide = guide_legend(order = 1)) +
      scale_linetype_manual(name = "β Values",
                            values = linetype_mapping,
                            guide = guide_legend(order = 3)) +
      new_scale_color() +
      # Plot negative pairs with warm colors
      geom_line(data = negative_pairs, 
                aes(x = !!sym(x_col), y = Value, color = Model_Branch_Formal, linetype = Beta_Clean), 
                size = line_size, alpha = 0.8) +
      geom_point(data = negative_pairs %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1), 
                 aes(x = !!sym(x_col), y = Value, color = Model_Branch_Formal),
                 size = point_size, alpha = 0.9) +
      scale_color_manual(name = "Negative Divergences",
                         values = negative_palette,
                         guide = guide_legend(order = 2))
  } else {
    # Components case: use colors for pairs and line styles for betas
    p <- ggplot() +
      # Plot positive pairs with cold colors
      geom_line(data = positive_pairs, 
                aes(x = !!sym(x_col), y = Value, color = Component_Pair, linetype = Beta_Clean), 
                size = line_size, alpha = 0.8) +
      geom_point(data = positive_pairs %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1), 
                 aes(x = !!sym(x_col), y = Value, color = Component_Pair),
                 size = point_size, alpha = 0.9) +
      scale_color_manual(name = "Positive Pairs",
                         values = positive_palette,
                         guide = guide_legend(order = 1)) +
      scale_linetype_manual(name = "β Values",
                            values = linetype_mapping,
                            guide = guide_legend(order = 3)) +
      new_scale_color() +
      # Plot negative pairs with warm colors
      geom_line(data = negative_pairs, 
                aes(x = !!sym(x_col), y = Value, color = Component_Pair, linetype = Beta_Clean), 
                size = line_size, alpha = 0.8) +
      geom_point(data = negative_pairs %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1), 
                 aes(x = !!sym(x_col), y = Value, color = Component_Pair),
                 size = point_size, alpha = 0.9) +
      scale_color_manual(name = "Negative Pairs",
                         values = negative_palette,
                         guide = guide_legend(order = 2))
  }
  
  # Continue with common plot elements
  p <- p +
    scale_x_continuous(breaks = pretty_breaks(n = 10)) +
    scale_y_continuous(breaks = pretty_breaks(n = 8), 
                       labels = label_number(accuracy = 0.1)) +
    coord_cartesian(ylim = y_limits) +
    labs(
      title = title,
      subtitle = subtitle,
      x = x_label,
      y = y_label
    ) +
    theme_minimal() +
    theme(
      # Text formatting
      plot.title = element_text(
        size = plot_title_size, 
        face = title_font_face, 
        family = plot_font_family,
        margin = margin(b = 10)
      ),
      plot.subtitle = element_text(
        size = plot_subtitle_size, 
        family = plot_font_family,
        margin = margin(b = 20)
      ),
      axis.title = element_text(
        size = axis_title_size, 
        family = plot_font_family
      ),
      axis.text = element_text(
        size = axis_text_size, 
        family = plot_font_family
      ),
      legend.title = element_text(
        size = legend_title_size, 
        face = legend_font_face, 
        family = plot_font_family
      ),
      legend.text = element_text(
        size = legend_text_size, 
        family = plot_font_family
      ),
      
      # Background and grid
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major = element_line(color = "grey90", size = 0.5),
      panel.grid.minor = element_line(color = "grey95", size = 0.25),
      
      # Legend positioning - controlled by show_legend parameter
      legend.position = if(show_legend) "right" else "none",
      legend.box.background = element_rect(color = "grey80", fill = "white"),
      legend.margin = margin(10, 10, 10, 10),
      
      # Plot margins
      plot.margin = margin(20, 20, 20, 20)
    )
  
  # Add zoom inset if specified
  if (!is.null(zoom_inset)) {
    # Create filtered data for the zoom region
    zoom_data <- plot_data %>%
      filter(!!sym(x_col) >= zoom_inset$x_limits[1] & !!sym(x_col) <= zoom_inset$x_limits[2])
    
    # Create the zoom inset plot with appropriate styling based on focus type
    if (exists("current_focus") && current_focus == "input_type") {
      p_zoom <- ggplot(zoom_data, aes(x = !!sym(x_col), y = Value)) +
        geom_line(aes(color = Input_Type_Display, linetype = Beta_Clean, group = Variable), 
                  size = line_size * 0.8, alpha = 0.9) +
        geom_point(data = zoom_data %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1),
                   aes(color = Input_Type_Display, shape = Condition_Display),
                   size = point_size * 0.8, alpha = 1.0) +
        scale_color_manual(values = input_type_colors, guide = "none") +
        scale_shape_manual(values = condition_shapes, guide = "none") +
        scale_linetype_manual(values = beta_linetypes, guide = "none")
    } else {
      # Generic zoom plot for other focus types
      p_zoom <- ggplot(zoom_data, aes(x = !!sym(x_col), y = Value, color = Variable)) +
        geom_line(size = line_size * 0.8, alpha = 0.9) +
        geom_point(data = zoom_data %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1),
                   size = point_size * 0.8, alpha = 1.0) +
        scale_color_manual(values = colors[1:length(unique(zoom_data$Variable))], guide = "none")
    }
    
    p_zoom <- p_zoom +
      coord_cartesian(xlim = zoom_inset$x_limits, ylim = zoom_inset$y_limits) +
      scale_x_continuous(breaks = seq(zoom_inset$x_limits[1], zoom_inset$x_limits[2], by = 1)) +
      scale_y_continuous(breaks = pretty_breaks(n = 4),
                         labels = label_number(accuracy = 0.01)) +
      theme_minimal() +
      theme(
        panel.background = element_rect(fill = "white", color = "black", size = 0.8),
        plot.background = element_rect(fill = "white", color = NA),
        panel.grid.major = element_line(color = "grey95", size = 0.3),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size = axis_text_size * 0.6, family = plot_font_family),
        axis.title = element_blank(),
        legend.position = "none",  # Explicitly remove legend from zoom inset
        plot.margin = margin(2, 2, 2, 2)
      )
    
    # Combine main plot with zoom inset and connection elements
    if (requireNamespace("cowplot", quietly = TRUE)) {
      # Calculate positions for connection elements
      zoom_region_x1 <- zoom_inset$x_limits[1]
      zoom_region_x2 <- zoom_inset$x_limits[2]
      zoom_region_y1 <- zoom_inset$y_limits[1]
      zoom_region_y2 <- zoom_inset$y_limits[2]
      
      # Make the rectangle more relaxed by expanding it by 50% on each side (width only)
      x_padding <- (zoom_region_x2 - zoom_region_x1) * 0.5
      
      # Create connection lines and rectangle overlay
      p_with_annotations <- p +
        # Add rectangle around zoom region (expanded width only)
        annotate("rect", 
                xmin = zoom_region_x1 - x_padding, xmax = zoom_region_x2 + x_padding,
                ymin = zoom_region_y1, ymax = zoom_region_y2,
                fill = NA, color = "black", size = 0.8, linetype = "dashed")
      
      # Use cowplot for inset positioning and add connection lines
      # Calculate relative positions of the expanded rectangle edges
      plot_x_range <- range(plot_data[[x_col]], na.rm = TRUE)
      plot_y_range <- if (!is.null(y_limits)) y_limits else range(plot_data$Value, na.rm = TRUE)
      
      # Convert zoom region coordinates to relative plot coordinates
      rect_left_x <- (zoom_region_x1 - x_padding - plot_x_range[1]) / (plot_x_range[2] - plot_x_range[1])
      rect_right_x <- (zoom_region_x2 + x_padding - plot_x_range[1]) / (plot_x_range[2] - plot_x_range[1])
      rect_bottom_y <- (zoom_region_y1 - plot_y_range[1]) / (plot_y_range[2] - plot_y_range[1])
      rect_top_y <- (zoom_region_y2 - plot_y_range[1]) / (plot_y_range[2] - plot_y_range[1])
      
      # Adjust for plot area within the overall figure (accounting for margins/axes)
      plot_area_x_start <- 0.08  # Left margin for y-axis
      plot_area_x_end <- 0.75   # Right margin for legend
      plot_area_y_start <- 0.12  # Bottom margin for x-axis
      plot_area_y_end <- 0.88    # Top margin for title
      
      # Convert arrow start coordinates to relative plot coordinates
      arrow_start_y <- (zoom_region_y2 - plot_y_range[1]) / (plot_y_range[2] - plot_y_range[1])
      
      # Convert to figure coordinates for arrow start
      arrow_start_x_fig <- (zoom_inset$x_limits[1] + (zoom_inset$x_limits[2] - zoom_inset$x_limits[1])/2) / (plot_x_range[2] - plot_x_range[1]) - (1-plot_area_x_end)
      arrow_start_y_fig <- plot_area_y_start + arrow_start_y * (plot_area_y_end - plot_area_y_start)
      
      # Calculate arrow end coordinates (bottom-right corner of inset)
      arrow_end_x_fig <- zoom_inset$position[1] + (zoom_inset$position[3] - zoom_inset$position[1])/2
      arrow_end_y_fig <- zoom_inset$position[2]
      
      p_final <- cowplot::ggdraw(p_with_annotations) + 
        cowplot::draw_plot(p_zoom, 
                          x = zoom_inset$position[1], 
                          y = zoom_inset$position[2],
                          width = zoom_inset$position[3] - zoom_inset$position[1], 
                          height = zoom_inset$position[4] - zoom_inset$position[2]) +
        # Add a dotted arrow from rectangle corner to inset corner
        cowplot::draw_line(
          x = c(arrow_start_x_fig, arrow_end_x_fig),  # From rectangle upper-right to inset bottom-right
          y = c(arrow_start_y_fig, arrow_end_y_fig),  # From rectangle top to inset bottom
          color = "black", size = 0.6, linetype = 3,
          arrow = arrow(length = unit(0.15, "inches"), type = "closed")
        )
      
      p <- p_final
    } else {
      warning("cowplot package not available. Install it to use zoom insets.")
    }
  }
  
  # Print summary statistics
  cat("\nSummary statistics for each variable:\n")
  summary_stats <- plot_data %>%
    group_by(Variable) %>%
    summarise(
      Min = min(Value, na.rm = TRUE),
      Mean = mean(Value, na.rm = TRUE),
      Max = max(Value, na.rm = TRUE),
      .groups = 'drop'
    )
  print(summary_stats)
  
  # Save plot if path is provided
  if (!is.null(save_path)) {
    ggsave(
      filename = save_path,
      plot = p,
      width = width,
      height = height,
      dpi = 600,
      bg = "white"
    )
    cat("Plot saved to:", save_path, "\n")
  }
  
  return(p)
}

# Function to create line chart from CSV data
create_line_chart <- function(data_path, 
                             x_col = "Epoch", 
                             value_cols = NULL,
                             title = "Training Progress",
                             subtitle = NULL,
                             x_label = "Epoch",
                             y_label = "Value",
                             legend_title = "Variables",
                             y_limits = NULL,
                             zoom_inset = NULL,  # list(x_limits = c(110, 115), y_limits = c(1.1, 1.8), position = c(0.7, 0.9, 0.95, 1.0))
                             save_path = NULL,
                             width = 12,
                             height = 8,
                             custom_theme = NULL,  # Add custom theme parameter
                             show_legend = TRUE) {  # Add legend control parameter
  
  # Load the data
  cat("Loading data from:", data_path, "\n")
  data <- read_csv(data_path, show_col_types = FALSE)
  
  # Preprocessing steps
  cat("Applying preprocessing steps...\n")
  
  # Step 1: Remove columns that contain MIN or MAX
  original_cols <- colnames(data)
  min_max_cols <- grep("MIN|MAX", original_cols, ignore.case = TRUE, value = TRUE)
  if (length(min_max_cols) > 0) {
    cat("Removing MIN/MAX columns:", paste(min_max_cols, collapse = ", "), "\n")
    data <- data %>% select(-any_of(min_max_cols))
  }
  
  # Step 2: Clean column names by removing everything after " - "
  new_col_names <- colnames(data)
  cleaned_col_names <- gsub(" - .*$", "", new_col_names)
  
  # Step 3: Extract focus-specific information from column names and clean ALL data columns
  # General cleaning for all data columns (not Step/Epoch)
  data_column_indices <- which(!cleaned_col_names %in% c("Step", "Epoch"))
  
  if (length(data_column_indices) > 0) {
    cat("Cleaning all data column names...\n")
    
    for (i in data_column_indices) {
      original_name <- cleaned_col_names[i]
      
      # Check if this column contains beta information
      beta_pattern <- "_[bB]([0-9\\.]+)_"
      beta_match <- regmatches(original_name, regexec(beta_pattern, original_name, ignore.case = TRUE))
      
      if (length(beta_match[[1]]) > 1) {
        beta_value <- beta_match[[1]][2]  # The captured group (the number)
        
        # Extract input type information
        input_type <- if (grepl("waveform", original_name, ignore.case = TRUE)) "waveform" else "mel"
        
        # Create short name: input_type + beta value (e.g., "waveform_b0", "mel_b1")
        cleaned_col_names[i] <- paste0(input_type, "_b", beta_value)
        cat("Cleaned column:", original_name, "->", cleaned_col_names[i], "\n")
      }
    }
  }
  
  # Focus-specific processing (kept for compatibility but most work done above)
  if (exists("current_focus") && current_focus == "beta") {
    cat("Beta focus: column cleaning already applied above\n")
  } else if (exists("current_focus") && current_focus == "input_type") {
    cat("Input type focus: column cleaning already applied above\n")
  }
  
  # Show the column name changes
  name_changes <- data.frame(
    Original = new_col_names,
    Cleaned = cleaned_col_names,
    stringsAsFactors = FALSE
  ) %>%
    filter(Original != Cleaned)
  
  if (nrow(name_changes) > 0) {
    cat("Cleaning column names (removing text after ' - '):\n")
    print(name_changes)
  }
  
  # Apply the cleaned column names
  colnames(data) <- cleaned_col_names
  
  # Remove Step column if it exists and create new integer-based epoch column
  if ("Step" %in% colnames(data)) {
    data <- data %>% select(-Step)
    cat("Removed Step column\n")
  }
  
  # Create new Epoch column as integer sequence
  data$Epoch <- 0:(nrow(data)-1)
  cat("Created new Epoch column with values 0 to", nrow(data)-1, "\n")
  
  # Print column names to help user identify available columns
  cat("Available columns after preprocessing:\n")
  print(colnames(data))
  
  # If value_cols is not specified, use all numeric columns except x_col
  if (is.null(value_cols)) {
    numeric_cols <- sapply(data, is.numeric)
    all_value_cols <- names(data)[numeric_cols & names(data) != x_col]
    
    # Filter columns by selected betas if they exist
    if (exists("selected_betas") && length(selected_betas) > 0) {
      cat("Filtering columns by selected betas:", paste(selected_betas, collapse = ", "), "\n")
      
      # Create precise patterns for selected betas (exact match after _b)
      beta_patterns <- paste0("_b", selected_betas, "$", collapse = "|")
      cat("Using beta filter pattern:", beta_patterns, "\n")
      
      # Filter value_cols to only include selected betas
      value_cols <- all_value_cols[grepl(beta_patterns, all_value_cols)]
      
      if (length(value_cols) == 0) {
        cat("No columns found matching selected betas. Available columns:\n")
        print(all_value_cols)
        stop("No data columns found for selected beta values: ", paste(selected_betas, collapse = ", "))
      }
      
      cat("Filtered columns for plotting:", paste(value_cols, collapse = ", "), "\n")
    } else {
      # Use all numeric columns if no beta filtering
      value_cols <- all_value_cols
      cat("No beta filtering applied. Using all numeric columns:", paste(value_cols, collapse = ", "), "\n")
    }
  }
  
  # Check if specified columns exist
  missing_cols <- setdiff(c(x_col, value_cols), colnames(data))
  if (length(missing_cols) > 0) {
    stop("Missing columns in data: ", paste(missing_cols, collapse = ", "))
  }
  
  # Prepare data for plotting by reshaping to long format
  plot_data <- data %>%
    select(all_of(c(x_col, value_cols))) %>%
    pivot_longer(cols = all_of(value_cols), 
                 names_to = "Variable", 
                 values_to = "Value") %>%
    filter(!is.na(Value))  # Remove any NA values
  
  # Sort legend entries by numeric value for proper ordering
  if (exists("current_focus") && current_focus == "beta") {
    # Extract numeric values from variable names for sorting
    plot_data$Variable <- factor(plot_data$Variable, 
                                levels = value_cols[order(as.numeric(value_cols))])
  } else {
    plot_data$Variable <- factor(plot_data$Variable, levels = value_cols)
  }
  
  # Create the line plot with separate aesthetics for input type and beta
  # First, add input type and beta information to plot_data
  plot_data$Input_Type <- str_extract(plot_data$Variable, "^(waveform|mel)")
  plot_data$Beta_Value <- str_extract(plot_data$Variable, "b([0-9\\.]+)")
  plot_data$Beta_Clean <- str_remove(plot_data$Beta_Value, "b")
  
  # Create color mapping for input types - more distinct cold vs warm colors
  input_type_colors <- c("waveform" = "#4575b4", "mel" = "#d73027")  # Cool blue for waveform, warm red for mel
  
  # Create linetype mapping for beta values
  unique_betas <- sort(unique(plot_data$Beta_Clean))
  beta_linetypes <- c("solid", "dashed", "dotted", "dotdash", "longdash", "twodash")[1:length(unique_betas)]
  names(beta_linetypes) <- unique_betas
  
  p <- ggplot(plot_data, aes(x = !!sym(x_col), y = Value, 
                             color = Input_Type, linetype = Beta_Clean)) +
    geom_line(size = line_size, alpha = 0.8) +
    # Add points only for the last sample of each curve
    geom_point(data = plot_data %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1), 
               size = point_size, alpha = 0.9, aes(color = Input_Type)) +
    scale_color_manual(name = "Input Type", 
                       values = input_type_colors,
                       guide = guide_legend(order = 1)) +
    scale_linetype_manual(name = "β Value", 
                          values = beta_linetypes,
                          guide = guide_legend(order = 2)) +
    scale_x_continuous(breaks = pretty_breaks(n = 10)) +
    scale_y_continuous(breaks = pretty_breaks(n = 8), 
                       labels = label_number(accuracy = 0.1)) +
    coord_cartesian(ylim = y_limits) +
    labs(
      title = title,
      subtitle = subtitle,
      x = x_label,
      y = y_label
    )
  
  # Apply theme - use custom theme if provided, otherwise use default
  if (!is.null(custom_theme)) {
    p <- p + custom_theme
  } else {
    p <- p + theme_minimal() +
      theme(
        # Text formatting
        plot.title = element_text(
          size = plot_title_size, 
          face = title_font_face, 
          family = plot_font_family,
          margin = margin(b = 10)
        ),
        plot.subtitle = element_text(
          size = plot_subtitle_size, 
          family = plot_font_family,
          margin = margin(b = 20)
        ),
        axis.title = element_text(
          size = axis_title_size, 
          family = plot_font_family
        ),
        axis.text = element_text(
          size = axis_text_size, 
          family = plot_font_family
        ),
        legend.title = element_text(
          size = legend_title_size, 
          face = legend_font_face, 
          family = plot_font_family
        ),
        legend.text = element_text(
          size = legend_text_size, 
          family = plot_font_family
        ),
        
        # Background and grid
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        panel.grid.major = element_line(color = "grey90", size = 0.5),
        panel.grid.minor = element_line(color = "grey95", size = 0.25),
        
        # Legend positioning - controlled by show_legend parameter
        legend.position = if(show_legend) "right" else "none",
        legend.box.background = element_rect(color = "grey80", fill = "white"),
        legend.margin = margin(10, 10, 10, 10),
        
        # Plot margins
        plot.margin = margin(20, 20, 20, 20)
      )
  }
  
  # Add zoom inset if specified
  if (!is.null(zoom_inset)) {
    # Create filtered data for the zoom region
    zoom_data <- plot_data %>%
      filter(!!sym(x_col) >= zoom_inset$x_limits[1] & !!sym(x_col) <= zoom_inset$x_limits[2])
    
    # Create the zoom inset plot
    p_zoom <- ggplot(zoom_data, aes(x = !!sym(x_col), y = Value, color = Variable)) +
      geom_line(size = line_size * 0.8, alpha = 0.9) +
      geom_point(data = zoom_data %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1),
                 size = point_size * 0.8, alpha = 1.0) +
      scale_color_manual(values = colors[1:length(value_cols)]) +
      coord_cartesian(xlim = zoom_inset$x_limits, ylim = zoom_inset$y_limits) +
      scale_x_continuous(breaks = seq(zoom_inset$x_limits[1], zoom_inset$x_limits[2], by = 1)) +
      scale_y_continuous(breaks = pretty_breaks(n = 4),
                         labels = label_number(accuracy = 0.01)) +
      theme_minimal() +
      theme(
        panel.background = element_rect(fill = "white", color = "black", size = 0.8),
        plot.background = element_rect(fill = "white", color = NA),
        panel.grid.major = element_line(color = "grey95", size = 0.3),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size = axis_text_size * 0.6, family = plot_font_family),
        axis.title = element_blank(),
        legend.position = "none",
        plot.margin = margin(2, 2, 2, 2)
      )
    
    # Combine main plot with zoom inset and connection elements
    if (requireNamespace("cowplot", quietly = TRUE)) {
      # Calculate positions for connection elements
      zoom_region_x1 <- zoom_inset$x_limits[1]
      zoom_region_x2 <- zoom_inset$x_limits[2]
      zoom_region_y1 <- zoom_inset$y_limits[1]
      zoom_region_y2 <- zoom_inset$y_limits[2]
      
      # Make the rectangle more relaxed by expanding it by 50% on each side (width only)
      x_padding <- (zoom_region_x2 - zoom_region_x1) * 0.5
      
      # Create connection lines and rectangle overlay
      p_with_annotations <- p +
        # Add rectangle around zoom region (expanded width only)
        annotate("rect", 
                xmin = zoom_region_x1 - x_padding, xmax = zoom_region_x2 + x_padding,
                ymin = zoom_region_y1, ymax = zoom_region_y2,
                fill = NA, color = "black", size = 0.8, linetype = "dashed")
      
      # Use cowplot for inset positioning and add connection lines
      # Calculate relative positions of the expanded rectangle edges
      plot_x_range <- range(plot_data[[x_col]], na.rm = TRUE)
      plot_y_range <- if (!is.null(y_limits)) y_limits else range(plot_data$Value, na.rm = TRUE)
      
      # Convert zoom region coordinates to relative plot coordinates
      rect_left_x <- (zoom_region_x1 - x_padding - plot_x_range[1]) / (plot_x_range[2] - plot_x_range[1])
      rect_right_x <- (zoom_region_x2 + x_padding - plot_x_range[1]) / (plot_x_range[2] - plot_x_range[1])
      rect_bottom_y <- (zoom_region_y1 - plot_y_range[1]) / (plot_y_range[2] - plot_y_range[1])
      rect_top_y <- (zoom_region_y2 - plot_y_range[1]) / (plot_y_range[2] - plot_y_range[1])
      
      # Adjust for plot area within the overall figure (accounting for margins/axes)
      plot_area_x_start <- 0.08  # Left margin for y-axis
      plot_area_x_end <- 0.83    # Right margin for legend
      plot_area_y_start <- 0.12  # Bottom margin for x-axis
      plot_area_y_end <- 0.88    # Top margin for title
      
      # Convert arrow start coordinates to relative plot coordinates
      #arrow_start_x <- 
      arrow_start_y <- (zoom_region_y2 - plot_y_range[1]) / (plot_y_range[2] - plot_y_range[1])
      
      # Convert to figure coordinates for arrow start
      arrow_start_x_fig <- (zoom_inset$x_limits[1] + (zoom_inset$x_limits[2] - zoom_inset$x_limits[1])/2) / (plot_x_range[2] - plot_x_range[1]) - (1-plot_area_x_end)
      arrow_start_y_fig <- plot_area_y_start + arrow_start_y * (plot_area_y_end - plot_area_y_start)
      
      # Calculate arrow end coordinates (bottom-right corner of inset)
      arrow_end_x_fig <- zoom_inset$position[1] + (zoom_inset$position[3] - zoom_inset$position[1])/2
      arrow_end_y_fig <- zoom_inset$position[2]
      
      p_final <- cowplot::ggdraw(p_with_annotations) + 
        cowplot::draw_plot(p_zoom, 
                          x = zoom_inset$position[1], 
                          y = zoom_inset$position[2],
                          width = zoom_inset$position[3] - zoom_inset$position[1], 
                          height = zoom_inset$position[4] - zoom_inset$position[2]) +
        # Add a dotted arrow from rectangle corner to inset corner
        cowplot::draw_line(
          x = c(arrow_start_x_fig, arrow_end_x_fig),  # From rectangle upper-right to inset bottom-right
          y = c(arrow_start_y_fig, arrow_end_y_fig),  # From rectangle top to inset bottom
          color = "black", size = 0.6, linetype = 3,
          arrow = arrow(length = unit(0.15, "inches"), type = "closed")
        )
      
      p <- p_final
    } else {
      warning("cowplot package not available. Install it to use zoom insets.")
    }
  }
  
  # Print summary statistics
  cat("\nSummary statistics for each variable:\n")
  summary_stats <- plot_data %>%
    group_by(Variable) %>%
    summarise(
      Min = min(Value, na.rm = TRUE),
      Mean = mean(Value, na.rm = TRUE),
      Max = max(Value, na.rm = TRUE),
      .groups = 'drop'
    )
  print(summary_stats)
  
  # Save plot if path is provided
  if (!is.null(save_path)) {
    ggsave(
      filename = save_path,
      plot = p,
      width = width,
      height = height,
      dpi = 600,
      bg = "white"
    )
    cat("Plot saved to:", save_path, "\n")
  }
  
  return(p)
}

plot1_combined <- create_line_chart_multi(
    data_paths = c(csv_file_path1, csv_file_path2),  # negative and positive divergences
    title = NULL,
    legend_title = "Input Type",
    x_col = "Epoch", 
    x_label = "Epoch",
    y_label = "Jensen-Shannon Divergence",
    y_limits = c(0, 1),
    save_path = save_path1
)

plot2_combined <- create_line_chart_multi(
    data_paths = c(csv_file_path3, csv_file_path4),  # negative and positive cross-entropies
    title = NULL,
    legend_title = "Input Type",
    x_col = "Epoch", 
    x_label = "Epoch",
    y_label = "Cross-Entropy",
    y_limits = c(0, 9),
    zoom_inset = list(
        x_limits = c(110, 115), 
        y_limits = c(0.15, 1.85), 
        position = c(0.35, 0.65, 0.6, 0.9)  # x_start, y_start, x_end, y_end (relative coords)
    ),
    save_path = save_path2
)

plot3_combined <- create_line_chart_multi(
    data_paths = c(csv_file_path5, csv_file_path6),  # Z and S branch decomposition losses
    title = NULL,
    legend_title = "Input Type",
    x_col = "Epoch", 
    x_label = "Epoch",
    y_label = "Decomposition Loss", 
    y_limits = c(0, 10),
    zoom_inset = list(
        x_limits = c(110, 115), 
        y_limits = c(0.45, 1.15), 
        position = c(0.5, 0.65, 0.75, 0.9)  # x_start, y_start, x_end, y_end (relative coords)
    ),
    save_path = save_path3
)

# Change focus to beta for the prior loss plot
# current_focus <- "beta"

# plot4 <- create_line_chart(
#    data_path = csv_file_path7,
#    title = "",
#    subtitle = "",
#    x_col = "Epoch",
#    x_label = "Epoch",
#    y_label = "Prior Loss",
#    legend_title = "β",
#    save_path = save_path4,
#    y_limits = c(0, 200),
#    zoom_inset = list(
#      x_limits = c(110, 115), 
#      y_limits = c(0.0, 0.6), 
#      position = c(0.5, 0.65, 0.75, 0.9)  # x_start, y_start, x_end, y_end (relative coords)
#    )
# )

