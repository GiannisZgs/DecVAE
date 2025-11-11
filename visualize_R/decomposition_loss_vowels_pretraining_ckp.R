
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(scales)
library(viridis)
library(ggnewscale)

# Style and font parameters
plot_font_family <- "Arial"
plot_title_size <- 28
title_font_face <- "plain"
plot_subtitle_size <- 22
axis_title_size <- 24
axis_text_size <- 20
legend_title_size <- 22
legend_text_size <- 18
legend_font_face <- "plain"
line_size <- 1.2
point_size <- 2.5

yellow_block_threshold <- 1.0
colors <- viridis(n = 8, option = "turbo", end = yellow_block_threshold)

# Load data from
parent_load_dir <- file.path("D:", "wandb_exports_for_figures", "vowels_pretraining")
current_script_experiment <- "FD_decomposition_loss_demo_betas"
model_type <- "dual"
set <- "validation"
current_focus <- "components"
selected_beta <- 0
loss_file1_1 <- paste0(set, "_divergence_0_1_Z.csv")
loss_file1_2 <- paste0(set, "_divergence_0_2_Z.csv")
loss_file1_3 <- paste0(set, "_divergence_0_3_Z.csv")
loss_file1_4 <- paste0(set, "_divergence_2_1_Z.csv")
loss_file1_5 <- paste0(set, "_divergence_3_1_Z.csv")
loss_file1_6 <- paste0(set, "_divergence_3_2_Z.csv")
loss_file4 <- paste0(set, "_decomposition_loss_Z.csv")

#loss_file
csv_file_path1_1 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file1_1)
csv_file_path1_2 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file1_2)
csv_file_path1_3 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file1_3)
csv_file_path1_4 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file1_4)
csv_file_path1_5 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file1_5)
csv_file_path1_6 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file1_6)
csv_file_path4 <- file.path(parent_load_dir, current_script_experiment, model_type, loss_file4)

# Save data at
parent_save_dir <- file.path('..','figures','pre-training_losses')
save_dir <- file.path(parent_save_dir, current_script_experiment, model_type)
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}
save_path1 <- file.path(save_dir, "component_divergences_Z_FD_b0.png")
save_path4 <- file.path(save_dir, "decomposition_loss_Z_betas.png")

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
                                   height = 8) {
  
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
    
    # Step 3: Filter by selected beta and rename for components case
    if (exists("current_focus") && current_focus == "components") {
      if (exists("selected_beta")) {
        beta_filter_pattern <- paste0("_[bB]", selected_beta, "_")
        cat("Filtering columns for beta =", selected_beta, "using pattern:", beta_filter_pattern, "\n")
        
        # Find columns that contain the selected beta value
        beta_filter_matches <- grepl(beta_filter_pattern, cleaned_col_names, ignore.case = TRUE)
        
        if (any(beta_filter_matches)) {
          # Find the data column (not Step/Epoch)
          data_cols <- cleaned_col_names[beta_filter_matches & !cleaned_col_names %in% c("Step", "Epoch")]
          
          if (length(data_cols) > 0) {
            # Take the first matching data column
            selected_col <- data_cols[1]
            cat("Selected column from file", i, ":", selected_col, "\n")
            
            # Extract component pattern and create new name
            component_pattern <- "_([0-9]+)_([0-9]+)_"
            component_match <- regmatches(data_paths[i], regexec(component_pattern, data_paths[i], ignore.case = TRUE))
            
            if (length(component_match[[1]]) > 1) {
              x_value <- as.numeric(component_match[[1]][2])
              y_value <- as.numeric(component_match[[1]][3])
              
              # Convert numbers to labels
              first_label <- if (x_value == 0) "X" else paste0("OC", x_value)
              second_label <- if (y_value == 0) "X" else paste0("OC", y_value)
              
              # Create the new column name
              new_col_name <- paste0(first_label, "-", second_label)
              cat("Renaming column", selected_col, "to", new_col_name, "\n")
              
              # Keep only the renamed data column and remove Step and everything else
              data <- data %>% select(all_of(selected_col))
              colnames(data)[colnames(data) == selected_col] <- new_col_name              
            } else {
              # No component match, create empty data frame
              data <- data.frame()
            }
          } else {
            warning("No data columns found for beta = ", selected_beta, " in file ", i)
            # Create empty data frame
            data <- data.frame()
          }
        } else {
          warning("No columns found for beta = ", selected_beta, " in file ", i)
          # Create empty data frame
          data <- data.frame()
        }
      } else {
        # No selected_beta, create empty data frame
        data <- data.frame()
      }
    } else {
      # Not components focus, create empty data frame
      data <- data.frame()
    }
    
    all_data[[i]] <- data
  }
  
  # Concatenate all datasets by columns (assuming same number of rows)
  data <- bind_cols(all_data)
  cat("Concatenated data has", nrow(data), "rows and", ncol(data), "columns\n")
  
  # Preprocessing steps (minimal since files already preprocessed)
  cat("Applying final preprocessing steps...\n")
  
  # Create Epoch column as integer sequence (like in create_line_chart)
  data$Epoch <- 1:nrow(data)
  cat("Created new Epoch column with values 1 to", nrow(data), "\n")
  
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
  
  # Separate positive and negative pairs
  positive_pairs <- plot_data %>% filter(grepl("X", Variable))
  negative_pairs <- plot_data %>% filter(!grepl("X", Variable))
  
  # Sort legend entries based on current focus
  if (exists("current_focus") && current_focus == "beta") {
    # Extract numeric values from variable names for sorting
    plot_data$Variable <- factor(plot_data$Variable, 
                                levels = value_cols[order(as.numeric(value_cols))])
  } else if (exists("current_focus") && current_focus == "components") {
    # Sort component legend entries logically within categories
    positive_vars <- value_cols[grepl("X", value_cols)]
    negative_vars <- value_cols[!grepl("X", value_cols)]
    
    positive_pairs$Variable <- factor(positive_pairs$Variable, levels = positive_vars)
    negative_pairs$Variable <- factor(negative_pairs$Variable, levels = negative_vars)
  } 
  
  # Create the line plot with two separate legends
  p <- ggplot() +
    # Plot positive pairs with first color scale
    geom_line(data = positive_pairs, 
              aes(x = !!sym(x_col), y = Value, color = Variable), 
              size = line_size, alpha = 0.8) +
    geom_point(data = positive_pairs %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1), 
               aes(x = !!sym(x_col), y = Value, color = Variable),
               size = point_size, alpha = 0.9) +
    scale_color_manual(name = "Positive Pairs",
                       values = colors[1:length(unique(positive_pairs$Variable))],
                       guide = guide_legend(order = 1)) +
    new_scale_color() +
    # Plot negative pairs with second color scale
    geom_line(data = negative_pairs, 
              aes(x = !!sym(x_col), y = Value, color = Variable), 
              size = line_size, alpha = 0.8) +
    geom_point(data = negative_pairs %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1), 
               aes(x = !!sym(x_col), y = Value, color = Variable),
               size = point_size, alpha = 0.9) +
    scale_color_manual(name = "Negative Pairs",
                       values = colors[(length(unique(positive_pairs$Variable))+1):length(value_cols)],
                       guide = guide_legend(order = 2)) +
    scale_x_continuous(breaks = pretty_breaks(n = 10)) +
    scale_y_continuous(breaks = pretty_breaks(n = 8), 
                       labels = label_number(accuracy = 0.001)) +
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
      
      # Legend positioning
      legend.position = "right",
      legend.box.background = element_rect(color = "grey80", fill = "white"),
      legend.margin = margin(10, 10, 10, 10),
      
      # Plot margins
      plot.margin = margin(20, 20, 20, 20)
    )
  
  # Add zoom inset if specified (same as original function)
  if (!is.null(zoom_inset)) {
    # ... (copy the zoom inset code from the original function)
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
                             height = 8) {
  
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
  
  # Step 3: Extract focus-specific information from column names
  if (exists("current_focus") && current_focus == "beta") {
    # Extract beta values from column names (case insensitive)
    beta_pattern <- "_[bB]([0-9\\.]+)_"
    
    # Find columns that match the beta pattern
    beta_matches <- regexpr(beta_pattern, cleaned_col_names, ignore.case = TRUE)
    has_beta <- beta_matches > 0
    
    if (any(has_beta)) {
      cat("Processing beta-focused column names...\n")
      
      # Extract the beta values for matching columns
      for (i in which(has_beta)) {
        beta_match <- regmatches(cleaned_col_names[i], regexec(beta_pattern, cleaned_col_names[i], ignore.case = TRUE))
        if (length(beta_match[[1]]) > 1) {
          beta_value <- beta_match[[1]][2]  # The captured group (the number)
          cleaned_col_names[i] <- paste0("", beta_value)
        }
      }
    }
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
  data$Epoch <- 1:nrow(data)
  cat("Created new Epoch column with values 1 to", nrow(data), "\n")
  
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
  
  # Sort legend entries by numeric value for proper ordering
  if (exists("current_focus") && current_focus == "beta") {
    # Extract numeric values from variable names for sorting
    plot_data$Variable <- factor(plot_data$Variable, 
                                levels = value_cols[order(as.numeric(value_cols))])
  } else {
    plot_data$Variable <- factor(plot_data$Variable, levels = value_cols)
  }
  
  # Create the line plot
  p <- ggplot(plot_data, aes(x = !!sym(x_col), y = Value, color = Variable)) +
    geom_line(size = line_size, alpha = 0.8) +
    # Add points only for the last sample of each curve
    geom_point(data = plot_data %>% group_by(Variable) %>% slice_max(!!sym(x_col), n = 1), 
               size = point_size, alpha = 0.9) +
    scale_color_manual(values = colors[1:length(value_cols)]) +
    scale_x_continuous(breaks = pretty_breaks(n = 10)) +
    scale_y_continuous(breaks = pretty_breaks(n = 8), 
                       labels = label_number(accuracy = 0.001)) +
    coord_cartesian(ylim = y_limits) +
    labs(
      title = title,
      subtitle = subtitle,
      x = x_label,
      y = y_label,
      color = legend_title
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
      
      # Legend positioning
      legend.position = "right",
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
      plot_area_x_end <- 0.75    # Right margin for legend
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

plot1 <- create_line_chart_multi(
    data_paths = c(csv_file_path1_1, csv_file_path1_2, csv_file_path1_3, 
                    csv_file_path1_4, csv_file_path1_5, csv_file_path1_6),
    title = NULL,
    legend_title = "Contrasting Pairs",
    x_col = "Epoch", 
    x_label = "Epoch",
    y_label = "Jensen-Shannon Divergence",
    y_limits = NULL,
    zoom_inset = NULL,
    save_path = save_path1,
)

# Change focus to beta for the decomposition loss plot
current_focus <- "beta"

plot4 <- create_line_chart(
   data_path = csv_file_path4,
   title = "",
   subtitle = "",
   x_col = "Epoch",
   x_label = "Epoch",
   y_label = "Decomposition Loss",
   legend_title = "Compression β",
   save_path = save_path4,
   y_limits = c(0, 5),
   zoom_inset = list(
     x_limits = c(110, 115), 
     y_limits = c(0.5, 1.2), 
     position = c(0.5, 0.65, 0.75, 0.9)  # x_start, y_start, x_end, y_end (relative coords)
   )
)


