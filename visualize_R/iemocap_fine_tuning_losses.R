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
axis_text_size <- 18  # Reduced from 22 to prevent overlapping
legend_title_size <- 28
legend_text_size <- 28
legend_font_face <- "plain"
line_size <- 1.2
point_size <- 2.5

yellow_block_threshold <- 1.0
colors <- viridis(n = 8, option = "turbo", end = yellow_block_threshold)

# Load data from
parent_load_dir <- file.path("D:", "wandb_exports_for_figures", "iemocap_fine_tuning")
current_script_experiment <- "from_dataset_ablation"
model_type <- "dual"
set <- "validation"
current_focus <- "transfer_dataset"
decomp <- "filter"
selected_betas <- c(0, 0.1, 1) # can be one or more values
transfer_datasets <- c("from_vowels", "from_timit")  # Add transfer dataset filter
loss_file1 <- paste0(set, "_divergence_negative_Z.csv")
loss_file2 <- paste0(set, "_divergence_positive_Z.csv")
loss_file3 <- paste0(set, "_divergence_negative_S.csv")
loss_file4 <- paste0(set, "_divergence_positive_S.csv")
loss_file5 <- paste0(set, "_cross_entropy_negative_Z.csv")
loss_file6 <- paste0(set, "_cross_entropy_positive_Z.csv")
loss_file7 <- paste0(set, "_cross_entropy_negative_S.csv")
loss_file8 <- paste0(set, "_cross_entropy_positive_S.csv")
loss_file9 <- paste0(set, "_decomposition_loss_Z.csv")
loss_file10 <- paste0(set, "_decomposition_loss_S.csv")
loss_file11 <- paste0(set, "_prior_loss_Z.csv")
loss_file12 <- paste0(set, "_prior_loss_S.csv")


#loss_file
csv_file_path1 <- file.path(parent_load_dir, model_type, loss_file1)
csv_file_path2 <- file.path(parent_load_dir, model_type, loss_file2)
csv_file_path3 <- file.path(parent_load_dir, model_type, loss_file3)
csv_file_path4 <- file.path(parent_load_dir, model_type, loss_file4)
csv_file_path5 <- file.path(parent_load_dir, model_type, loss_file5)
csv_file_path6 <- file.path(parent_load_dir, model_type, loss_file6)
csv_file_path7 <- file.path(parent_load_dir, model_type, loss_file7)
csv_file_path8 <- file.path(parent_load_dir, model_type, loss_file8)
csv_file_path9 <- file.path(parent_load_dir, model_type, loss_file9)
csv_file_path10 <- file.path(parent_load_dir, model_type, loss_file10)
csv_file_path11 <- file.path(parent_load_dir, model_type, loss_file11)
csv_file_path12 <- file.path(parent_load_dir, model_type, loss_file12)

# Save data at
parent_save_dir <- file.path('..','figures','fine_tuning_losses','IEMOCAP')
save_dir <- file.path(parent_save_dir, current_script_experiment, model_type, set)
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

save_path1 <- file.path(save_dir, "number_of_components_divergence_negative_Z_FD_betas_0_01_1.png")
save_path2 <- file.path(save_dir, "number_of_components_divergence_positive_Z_FD_betas_0_01_1.png")
save_path3 <- file.path(save_dir, "number_of_components_divergence_negative_S_FD_betas_0_01_1.png")
save_path4 <- file.path(save_dir, "number_of_components_divergence_positive_S_FD_betas_0_01_1.png")
save_path5 <- file.path(save_dir, "number_of_components_cross_entropy_negative_Z_FD_betas_0_01_1.png")
save_path6 <- file.path(save_dir, "number_of_components_cross_entropy_positive_Z_FD_betas_0_01_1.png")
save_path7 <- file.path(save_dir, "number_of_components_cross_entropy_negative_S_FD_betas_0_01_1.png")
save_path8 <- file.path(save_dir, "number_of_components_cross_entropy_positive_S_FD_betas_0_01_1.png")
save_path9 <- file.path(save_dir, "number_of_components_decomposition_loss_Z_FD_betas_0_01_1.png")
save_path10 <- file.path(save_dir, "number_of_components_decomposition_loss_S_FD_betas_0_01_1.png")
save_path11 <- file.path(save_dir, "number_of_components_prior_loss_Z_FD_betas_0_01_1.png")
save_path12 <- file.path(save_dir, "number_of_components_prior_loss_S_FD_betas_0_01_1.png")

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
          # Create pattern that matches both decimal and integer representations
          # For beta = 0.1, match both "_b0.1_" and "_b01_"
          if (beta_val == 0.1) {
            beta_filter_pattern <- paste0("_[bB](0\\.1|01)($|_)")
          } else if (beta_val == 0.01) {
            beta_filter_pattern <- paste0("_[bB](0\\.01|001)($|_)")
          } else {
            # For other values, use standard pattern
            beta_filter_pattern <- paste0("_[bB]", beta_val, "($|_)")
          }
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
            beta_filter_pattern <- paste0("_[bB]", beta_val, "($|_)")
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
            # Create pattern that matches both decimal and integer representations
            if (beta_val == 0.1) {
              beta_filter_pattern <- paste0("_[bB](0\\.1|01)($|_)")
            } else if (beta_val == 0.01) {
              beta_filter_pattern <- paste0("_[bB](0\\.01|001)($|_)")
            } else {
              beta_filter_pattern <- paste0("_[bB]", beta_val, "($|_)")
            }
            cat("Filtering columns for beta =", beta_val, "using pattern:", beta_filter_pattern, "\n")
            
            # Find columns that contain the selected beta value
            beta_filter_matches <- grepl(beta_filter_pattern, cleaned_col_names, ignore.case = TRUE)
            
            if (any(beta_filter_matches)) {
              # Find the data columns (not Step/Epoch)
              data_cols <- cleaned_col_names[beta_filter_matches & !cleaned_col_names %in% c("Step", "Epoch")]
              
              for (selected_col in data_cols) {
                cat("Selected column for beta", beta_val, ":", selected_col, "\n")
                
                # Extract NoC (Number of Components) from column name
                noc_pattern <- "NoC([0-9]+)"
                noc_match <- regmatches(selected_col, regexec(noc_pattern, selected_col, ignore.case = TRUE))
                
                if (length(noc_match[[1]]) > 1) {
                  noc_value <- noc_match[[1]][2]  # The captured group (the number)
                } else {
                  # Default to NoC3 if pattern not found
                  noc_value <- "3"
                  cat("Warning: Could not detect NoC value from column name", selected_col, ", defaulting to NoC3\n")
                }
                
                # Detect decomposition type from column name
                decomp_type <- ifelse(grepl("filter", selected_col, ignore.case = TRUE), "FD", 
                                     ifelse(grepl("ewt", selected_col, ignore.case = TRUE), "EWT", "FD"))
                
                # Create descriptive name: NoC_decomp_branch_bX (e.g., "NoC3_FD_Z_b1", "NoC4_EWT_S_b1")
                new_col_name <- paste0("NoC", noc_value, "_", decomp_type, "_", branch_type, "_b", beta_val)
                cat("Creating column", new_col_name, "\n")
                
                # Keep only the selected data column and rename it
                beta_data <- data %>% select(all_of(selected_col))
                colnames(beta_data) <- new_col_name
                processed_data_list[[paste0("NoC", noc_value, "_", decomp_type, "_", branch_type, "_beta_", beta_val)]] <- beta_data
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
    } else if (exists("current_focus") && current_focus == "number_of_components") {
      if (exists("selected_betas") && length(selected_betas) > 0) {
        processed_data_list <- list()
        
        # Determine if this is negative or positive file based on filename
        is_negative <- grepl("negative", data_paths[i], ignore.case = TRUE)
        is_positive <- grepl("positive", data_paths[i], ignore.case = TRUE)
        
        # Determine branch type for decomposition losses
        is_Z_branch <- grepl("_Z\\.csv$", data_paths[i], ignore.case = TRUE)
        is_S_branch <- grepl("_S\\.csv$", data_paths[i], ignore.case = TRUE)
        
        if (is_negative || is_positive) {
          file_type <- if (is_negative) "negative" else "positive"
          cat("Processing", file_type, "file\n")
          
          for (beta_idx in seq_along(selected_betas)) {
            beta_val <- selected_betas[beta_idx]
            if (beta_val == 0.1) {
              beta_filter_pattern <- paste0("_[bB](0\\.1|01)($|_)")
            } else if (beta_val == 0.01) {
              beta_filter_pattern <- paste0("_[bB](0\\.01|001)($|_)")
            } else {
              # For other values, use standard pattern
              beta_filter_pattern <- paste0("_[bB]", beta_val, "($|_)")
            }
            cat("Filtering columns for beta =", beta_val, "using pattern:", beta_filter_pattern, "\n")
            
            # Find columns that contain the selected beta value
            beta_filter_matches <- grepl(beta_filter_pattern, cleaned_col_names, ignore.case = TRUE)
            
            if (any(beta_filter_matches)) {
              # Find the data columns (not Step/Epoch)
              data_cols <- cleaned_col_names[beta_filter_matches & !cleaned_col_names %in% c("Step", "Epoch")]
              
              for (selected_col in data_cols) {
                cat("Selected column for beta", beta_val, ":", selected_col, "\n")
                
                # Extract NoC (Number of Components) from column name
                noc_pattern <- "NoC([0-9]+)"
                noc_match <- regmatches(selected_col, regexec(noc_pattern, selected_col, ignore.case = TRUE))
                
                if (length(noc_match[[1]]) > 1) {
                  noc_value <- noc_match[[1]][2]  # The captured group (the number)
                } else {
                  # Default to NoC3 if pattern not found
                  noc_value <- "3"
                  cat("Warning: Could not detect NoC value from column name", selected_col, ", defaulting to NoC3\n")
                }
                
                # Detect decomposition type from column name
                decomp_type <- ifelse(grepl("filter", selected_col, ignore.case = TRUE), "FD", 
                                     ifelse(grepl("ewt", selected_col, ignore.case = TRUE), "EWT", "FD"))
                
                # Create descriptive name: NoC_decomp_positive/negative_bX (e.g., "NoC2_FD_positive_b0", "NoC3_EWT_negative_b0")
                new_col_name <- paste0("NoC", noc_value, "_", decomp_type, "_", file_type, "_b", beta_val)
                cat("Creating column", new_col_name, "\n")
                
                # Keep only the selected data column and rename it
                beta_data <- data %>% select(all_of(selected_col))
                colnames(beta_data) <- new_col_name
                processed_data_list[[paste0("NoC", noc_value, "_", decomp_type, "_", file_type, "_beta_", beta_val)]] <- beta_data
              }
            }
          }
        } else if (is_Z_branch || is_S_branch) {
          # Handle decomposition loss files (Z or S branch)
          branch_type <- if (is_Z_branch) "Z" else "S"
          cat("Processing", branch_type, "branch file\n")
          
          for (beta_idx in seq_along(selected_betas)) {
            beta_val <- selected_betas[beta_idx]
            # Create pattern that matches both decimal and integer representations
            if (beta_val == 0.1) {
              beta_filter_pattern <- paste0("_[bB](0\\.1|01)($|_)")
            } else if (beta_val == 0.01) {
              beta_filter_pattern <- paste0("_[bB](0\\.01|001)($|_)")
            } else {
              beta_filter_pattern <- paste0("_[bB]", beta_val, "($|_)")
            }
            cat("Filtering columns for beta =", beta_val, "using pattern:", beta_filter_pattern, "\n")
            
            # Find columns that contain the selected beta value
            beta_filter_matches <- grepl(beta_filter_pattern, cleaned_col_names, ignore.case = TRUE)
            
            if (any(beta_filter_matches)) {
              # Find the data columns (not Step/Epoch)
              data_cols <- cleaned_col_names[beta_filter_matches & !cleaned_col_names %in% c("Step", "Epoch")]
              
              for (selected_col in data_cols) {
                cat("Selected column for beta", beta_val, ":", selected_col, "\n")
                
                # Extract NoC (Number of Components) from column name
                noc_pattern <- "NoC([0-9]+)"
                noc_match <- regmatches(selected_col, regexec(noc_pattern, selected_col, ignore.case = TRUE))
                
                if (length(noc_match[[1]]) > 1) {
                  noc_value <- noc_match[[1]][2]  # The captured group (the number)
                } else {
                  # Default to NoC3 if pattern not found
                  noc_value <- "3"
                  cat("Warning: Could not detect NoC value from column name", selected_col, ", defaulting to NoC3\n")
                }
                
                # Detect decomposition type from column name
                decomp_type <- ifelse(grepl("filter", selected_col, ignore.case = TRUE), "FD", 
                                     ifelse(grepl("ewt", selected_col, ignore.case = TRUE), "EWT", "FD"))
                
                # Create descriptive name: NoC_decomp_branch_bX (e.g., "NoC3_FD_Z_b1", "NoC4_EWT_S_b1")
                new_col_name <- paste0("NoC", noc_value, "_", decomp_type, "_", branch_type, "_b", beta_val)
                cat("Creating column", new_col_name, "\n")
                
                # Keep only the selected data column and rename it
                beta_data <- data %>% select(all_of(selected_col))
                colnames(beta_data) <- new_col_name
                processed_data_list[[paste0("NoC", noc_value, "_", decomp_type, "_", branch_type, "_beta_", beta_val)]] <- beta_data
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
    } else if (exists("current_focus") && current_focus == "transfer_dataset") {
      if (exists("selected_betas") && length(selected_betas) > 0) {
        processed_data_list <- list()
        
        for (beta_idx in seq_along(selected_betas)) {
          beta_val <- selected_betas[beta_idx]
          # Create pattern that matches both decimal and integer representations
          if (beta_val == 0.1) {
            beta_filter_pattern <- paste0("_[bB](0\\.1|01)($|_)")
          } else if (beta_val == 0.01) {
            beta_filter_pattern <- paste0("_[bB](0\\.01|001)($|_)")
          } else {
            beta_filter_pattern <- paste0("_[bB]", beta_val, "($|_)")
          }
          cat("Filtering columns for beta =", beta_val, "using pattern:", beta_filter_pattern, "\n")
          
          # Find columns that contain the selected beta value
          beta_filter_matches <- grepl(beta_filter_pattern, cleaned_col_names, ignore.case = TRUE)
          
          if (any(beta_filter_matches)) {
            # Find the data columns (not Step/Epoch)
            data_cols <- cleaned_col_names[beta_filter_matches & !cleaned_col_names %in% c("Step", "Epoch")]
            
            # Filter by decomposition type
            decomp_pattern <- if (decomp == "filter") "filter" else "ewt"
            decomp_cols <- grep(decomp_pattern, data_cols, ignore.case = TRUE, value = TRUE)
            
            for (selected_col in decomp_cols) {
              cat("Selected column for beta", beta_val, ":", selected_col, "\n")
              
              # Extract transfer dataset from column name
              transfer_dataset <- NULL
              if (grepl("vowels", selected_col, ignore.case = TRUE)) {
                transfer_dataset <- "from_vowels"
              } else if (grepl("timit", selected_col, ignore.case = TRUE)) {
                transfer_dataset <- "from_timit"
              }
              
              if (!is.null(transfer_dataset) && transfer_dataset %in% transfer_datasets) {
                # Create descriptive name: transfer_dataset_bX (e.g., "from_vowels_b0", "from_timit_b1")
                new_col_name <- paste0(transfer_dataset, "_β", beta_val)
                cat("Creating column", new_col_name, "\n")
                
                # Keep only the selected data column and rename it
                beta_data <- data %>% select(all_of(selected_col))
                colnames(beta_data) <- new_col_name
                processed_data_list[[paste0(transfer_dataset, "_beta_", beta_val)]] <- beta_data
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
            beta_filter_pattern <- paste0("_[bB]", beta_val, "($|_)")
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
            beta_filter_pattern <- paste0("_[bB]", beta_val, "($|_)")
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
  
  # Create Epoch column as integer sequence
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
  plot_data$Beta <- str_extract(plot_data$Variable, "β[0-9\\.]+")
  
  # Handle different focus types
  if (exists("current_focus") && current_focus == "transfer_dataset") {
    # For transfer_dataset focus, extract transfer dataset and beta from variable names
    plot_data$Transfer_Dataset <- str_extract(plot_data$Variable, "(from_vowels|from_timit)")
    plot_data$Component_Pair <- plot_data$Transfer_Dataset
    plot_data$Pair_Type <- plot_data$Transfer_Dataset
  } else if (exists("current_focus") && current_focus == "negative-positive") {
    # For negative-positive focus, extract the file type (Negative/Positive)
    plot_data$Component_Pair <- str_remove(plot_data$Variable, "_b[0-9\\.]+")
    plot_data$Pair_Type <- plot_data$Component_Pair  # Use Negative/Positive directly
  } else if (exists("current_focus") && current_focus == "input_type") {
    # For input_type focus, extract information from column names
    # Column format: "waveform_pos_b0", "mel_neg_b0", "waveform_Z_b0", "mel_S_b0"
    
    # Extract input type (waveform or mel)
    plot_data$Input_Type <- str_extract(plot_data$Variable, "^(waveform|mel)")
    
    # Extract pos/neg or Z/S from the middle part
    plot_data$File_Type <- str_extract(plot_data$Variable, "_(positive|negative|Z|S)_")
    plot_data$File_Type <- str_remove_all(plot_data$File_Type, "_")  # Remove underscores
    
    # For grouping and legend purposes
    plot_data$Component_Pair <- paste0(plot_data$Input_Type, "_", plot_data$File_Type)
    plot_data$Pair_Type <- plot_data$File_Type  # Use pos/neg/Z/S for separation
  } else if (exists("current_focus") && current_focus == "number_of_components") {
    # For number_of_components focus, extract NoC and decomposition type from column names
    # Column format: "NoC4_filter_b1", "NoC4_ewt_b1", etc.
    plot_data$NoC_Value <- str_extract(plot_data$Variable, "NoC([0-9]+)")
    plot_data$NoC_Value <- str_remove(plot_data$NoC_Value, "NoC")
    
    # Detect decomposition type from column names and map to display labels
    plot_data$Decomposition <- ifelse(grepl("filter", plot_data$Variable, ignore.case = TRUE), "FD", 
                                     ifelse(grepl("ewt", plot_data$Variable, ignore.case = TRUE), "EWT", "FD"))
    
    plot_data$File_Type <- str_extract(plot_data$Variable, "_(positive|negative|Z|S)_")
    plot_data$File_Type <- str_remove_all(plot_data$File_Type, "_")
    
    # Keep separate for individual legends
    plot_data$Component_Only <- plot_data$NoC_Value
    plot_data$Decomposition_Only <- plot_data$Decomposition
    plot_data$Pair_Type <- plot_data$File_Type
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
  
  # Clean beta values for legend (remove β prefix)
  plot_data$Beta_Clean <- str_remove(plot_data$Beta, "β")
  
  # Debug: Print what we found
  cat("Beta values found:", paste(unique(plot_data$Beta), collapse = ", "), "\n")
  cat("Beta clean values:", paste(unique(plot_data$Beta_Clean), collapse = ", "), "\n")
  cat("Transfer datasets found:", paste(unique(plot_data$Transfer_Dataset), collapse = ", "), "\n")
  
  # Filter out rows with NA values in key columns
  if (exists("current_focus") && current_focus == "transfer_dataset") {
    plot_data <- plot_data %>% 
      filter(!is.na(Beta_Clean) & !is.na(Transfer_Dataset))
    
    if (nrow(plot_data) == 0) {
      cat("ERROR: No valid data after filtering. Check column names and patterns.\n")
      cat("Original variables:", paste(unique(data %>% select(-Epoch) %>% colnames()), collapse = ", "), "\n")
      return(NULL)
    }
  }
  
  # Create the plot
  if (exists("current_focus") && current_focus == "transfer_dataset") {
    # Create linetype mapping for beta values - solid for 0, dashed for 0.1, dotted for 1
    beta_linetypes <- c("0" = "solid", "0.1" = "dashed", "1" = "dotted")
    
    # Create shape mapping for beta values
    beta_shapes <- c("0" = 16, "0.1" = 17, "1" = 18)  # circle, triangle, square
    
    # Create color mapping for transfer datasets - blue for vowels, red for TIMIT
    transfer_colors <- c("from_vowels" = "#2166ac", "from_timit" = "#d73027")
    
    p <- ggplot(plot_data, aes_string(x = x_col, y = "Value")) +
      geom_line(aes_string(color = "Transfer_Dataset", linetype = "Beta_Clean"), 
                size = line_size, alpha = 0.8) +
      geom_point(data = plot_data %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1), 
                 aes_string(color = "Transfer_Dataset", shape = "Beta_Clean"),
                 size = point_size, alpha = 0.9) +
      scale_color_manual(
        name = "Transfer from",
        values = transfer_colors,
        labels = c("from_vowels" = "Vowels", "from_timit" = "TIMIT")
      ) +
      scale_linetype_manual(
        name = "β Value", 
        values = beta_linetypes
      ) +
      scale_shape_manual(
        name = "β Value",
        values = beta_shapes
      ) +
      coord_cartesian(ylim = y_limits) +
      labs(
        title = title,
        subtitle = subtitle,
        x = x_label,
        y = y_label
      ) +
      theme_minimal(base_family = plot_font_family) +
      theme(
        plot.title = element_text(size = plot_title_size, face = title_font_face),
        plot.subtitle = element_text(size = plot_subtitle_size),
        axis.title = element_text(size = axis_title_size),
        axis.text = element_text(size = axis_text_size),
        legend.title = element_text(size = legend_title_size, face = legend_font_face),
        legend.text = element_text(size = legend_text_size),
        legend.position = ifelse(show_legend, "right", "none"),
        legend.box.background = element_rect(color = "grey80", fill = "white"),
        legend.margin = margin(10, 10, 10, 10),
        plot.margin = margin(20, 20, 20, 20)
      )
  } else {
    # Default plot for other focus types
    p <- ggplot(plot_data, aes_string(x = x_col, y = "Value")) +
      geom_line(aes_string(color = "Variable", linetype = "Beta_Clean"), size = line_size) +
      geom_point(aes_string(color = "Variable", shape = "Beta_Clean"), size = point_size) +
      scale_color_viridis(discrete = TRUE, option = "turbo") +
      scale_linetype_manual(values = c("0" = "solid", "0.1" = "dashed", "1" = "dotted")) +
      scale_shape_manual(values = c("0" = 16, "0.1" = 17, "1" = 18)) +
      coord_cartesian(ylim = y_limits) +
      labs(
        title = title,
        subtitle = subtitle,
        x = x_label,
        y = y_label,
        color = legend_title,
        linetype = "β Value",
        shape = "β Value"
      ) +
      theme_minimal(base_family = plot_font_family) +
      theme(
        plot.title = element_text(size = plot_title_size, face = title_font_face),
        plot.subtitle = element_text(size = plot_subtitle_size),
        axis.title = element_text(size = axis_title_size),
        axis.text = element_text(size = axis_text_size),
        legend.title = element_text(size = legend_title_size, face = legend_font_face),
        legend.text = element_text(size = legend_text_size),
        legend.position = ifelse(show_legend, "right", "none"),
        legend.box.background = element_rect(color = "grey80", fill = "white"),
        legend.margin = margin(10, 10, 10, 10),
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
      scale_color_manual(values = colors[1:length(unique(zoom_data$Variable))], guide = "none") +
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

# Updated plot generation - one call per CSV file, focusing on NoC and decomposition comparison

# Plot 1: Z Divergence Negative
plot1_z_divergence_negative <- create_line_chart_multi(
    data_paths = csv_file_path1,  # Z-branch divergence negative only
    title = NULL,
    x_col = "Epoch", 
    x_label = "Epoch",
    y_label = "Jensen-Shannon Divergence",
    y_limits = c(0.5, 1),
    save_path = save_path1
)

# Plot 2: Z Divergence Positive
plot2_z_divergence_positive <- create_line_chart_multi(
    data_paths = csv_file_path2,  # Z-branch divergence positive only
    title = NULL,
    x_col = "Epoch", 
    x_label = "Epoch", 
    y_label = "Jensen-Shannon Divergence",
    y_limits = c(0, 0.6),
    save_path = save_path2
)

# Plot 3: S-branch Negative Divergences
plot3_s_negative_divergence <- create_line_chart_multi(
    data_paths = csv_file_path3,  # S-branch negative divergence
    title = NULL,
    x_col = "Epoch",
    x_label = "Epoch",
    y_label = "Jensen-Shannon Divergence", 
    y_limits = c(0, 0.75),
    save_path = save_path3
)

# Plot 4: S-branch Positive Divergences
plot4_s_positive_divergence <- create_line_chart_multi(
    data_paths = csv_file_path4,  # S-branch positive divergence
    title = NULL,
    x_col = "Epoch",
    x_label = "Epoch",
    y_label = "Jensen-Shannon Divergence",
    y_limits = c(0, 0.6), 
    save_path = save_path4
)

# Plot 5: Z-branch negative Cross-Entropies
plot5_z_negative_cross_entropy <- create_line_chart_multi(
    data_paths = csv_file_path5,  # Z-branch negative cross-entropy
    title = NULL,
    x_col = "Epoch",
    x_label = "Epoch", 
    y_label = "Cross-Entropy",
    y_limits = c(0, 5),
    save_path = save_path5
)

# Plot 6: Z-branch positive Cross-Entropies
plot6_z_positive_cross_entropy <- create_line_chart_multi(
    data_paths = csv_file_path6,  # Z-branch positive cross-entropy
    title = NULL,
    x_col = "Epoch",
    x_label = "Epoch", 
    y_label = "Cross-Entropy",
    y_limits = c(0, 4.5),
    save_path = save_path6
)

# Plot 8: S-branch negative Cross-Entropies
plot7_s_negative_cross_entropy <- create_line_chart_multi(
    data_paths = csv_file_path7,  # S-branch negative cross-entropy
    title = NULL,
    x_col = "Epoch",
    x_label = "Epoch", 
    y_label = "Cross-Entropy",
    y_limits = c(0, 10),
    save_path = save_path7
)

# Plot 8: S-branch positive Cross-Entropies
plot8_s_positive_cross_entropy <- create_line_chart_multi(
    data_paths = csv_file_path8,  # S-branch positive cross-entropy
    title = NULL,
    x_col = "Epoch",
    x_label = "Epoch", 
    y_label = "Cross-Entropy",
    y_limits = c(0, 2.5),
    save_path = save_path8
)

# Plot 9: Z-branch Decomposition Loss
plot9_z_decomposition_loss <- create_line_chart_multi(
    data_paths = csv_file_path9,  # Z-branch decomposition loss
    title = NULL,
    x_col = "Epoch",
    x_label = "Epoch",
    y_label = "Decomposition Loss", 
    y_limits = c(0, 2),
    save_path = save_path9
)

# Plot 10: S-branch Decomposition Loss
plot10_s_decomposition_loss <- create_line_chart_multi(
    data_paths = csv_file_path10,  # S-branch decomposition loss
    title = NULL,
    x_col = "Epoch",
    x_label = "Epoch",
    y_label = "Decomposition Loss",
    y_limits = c(0, 2), 
    save_path = save_path10
)

# Plot 11: Z-branch Prior Loss
plot11_z_prior_loss <- create_line_chart_multi(
    data_paths = csv_file_path11,  # Z-branch prior loss
    title = NULL,
    x_col = "Epoch",
    x_label = "Epoch",
    y_label = "Prior Approximation Loss",
    y_limits = c(0, 0.02), 
    save_path = save_path11
)

# Plot 12: S-branch Prior Loss
plot12_s_prior_loss <- create_line_chart_multi(
    data_paths = csv_file_path12,  # S-branch prior loss
    title = NULL,
    x_col = "Epoch",
    x_label = "Epoch",
    y_label = "Prior Approximation Loss",
    y_limits = c(0, 0.1), 
    save_path = save_path12
)