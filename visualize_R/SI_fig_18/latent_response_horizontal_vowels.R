#' SI Figure 18: latent response (traversals) and interpretability analysis for the 
#' SimVowels dataset.

library(vscDebugger)
library(jsonlite)
library(ggplot2)
library(dplyr)
library(viridis)
library(gridExtra)
library(grid)
library(cowplot)
library(patchwork)

# Style and font parameters
plot_font_family <- "Arial"
plot_background_color <- "white"
plot_text_color <- "black"
palette <- "plasma"
yellow_block_threshold <- 0.8  

# Plot type-specific parameters
axis_x_title_size <- 50
axis_y_title_size <- 60 
axis_text_size_x <- 30 
axis_text_size_y <- 40
plot_title_size <- 70
plot_subtitle_size <- 40
plot_sup_title_size <- 60 
plot_title_face <- "plain"
plot_title_hjust <- 0.5

# Legend elements
legend_text_size <- 40
legend_title_size <- 40
legend_text_face <- "plain"
legend_title_face <- "plain"
legend_key_size <- 2.5  # in cm

# Geom elements
line_size <- 1.2
grid_line_color <- "gray90"

# Plot margins
plot_margin <- margin(8, 8, 8, 8)

parent_save_dir <-  file.path('..','supplementary_figures','SI_fig_18_vowels_latent_traversals')
experiment <- 'fixed_speakers_free_vowels_5' #fixed_vowels_5, fixed_speakers_free_vowels_5
model <- 'vae1D_FC_mel' #'vae1D_FC_mel', 'ewt', 'vmd', 'filter', 'emd'
beta <- '1'
ckp <- 'training_ckp_epoch_109' 
latent <- 'X'

if (grepl("fixed_speakers", experiment)) {
  varying <- 'vowels'
  fixed <- 'speakers'
} else {
  varying <- 'speakers'
  fixed <- 'vowels'
}
parent_load_dir <- file.path('..','data','latent_traversal_data','vowels')
if (grepl("vae",model)) {
  load_dir <- file.path(parent_load_dir,'latent_responses_vae/sim_vowels_traversals')
} else {
  load_dir <- file.path(parent_load_dir,'latent_responses/sim_vowels_traversals')
}

save_dir <- file.path(parent_save_dir, experiment, model, paste0('b', beta), ckp)
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}
save_path <- file.path(save_dir, paste0(latent,'_varying_',varying,'_horizontal_norm.png'))

CANONICAL_FORMANT_FREQUENCIES <- list(
  'a' = c(710, 1100, 2540),
  'e' = c(550, 1770, 2490),
  'I' = c(400, 1920, 2560),
  'aw' = c(590, 880, 2540),
  'u' = c(310, 870, 2250)
)

CANONICAL_FORMANT_FREQUENCIES_EXTENDED <- list(
  'i' = c(280, 2250, 2890),
  'I' = c(400, 1920, 2560),
  'e' = c(550, 1770, 2490),
  'ae' = c(690, 1660, 2490),
  'a' = c(710, 1100, 2540),
  'aw' = c(590, 880, 2540),
  'y' = c(450, 1030, 2380),
  'u' = c(310, 870, 2250)
)

# Create vowel-to-integer and integer-to-vowel mappings
INT_TO_VOWEL <- names(CANONICAL_FORMANT_FREQUENCIES)
VOWEL_TO_INT <- setNames(seq(0, length(INT_TO_VOWEL) - 1), INT_TO_VOWEL)

INT_TO_VOWEL_EXTENDED <- names(CANONICAL_FORMANT_FREQUENCIES_EXTENDED)
VOWEL_TO_INT_EXTENDED <- setNames(seq(0, length(INT_TO_VOWEL_EXTENDED) - 1), INT_TO_VOWEL_EXTENDED)

load_json_data <- function(file_path) {
    json_text <- readLines(file_path)
    json_text <- gsub("\\bNaN\\b", "\"NaN\"", json_text)
    fromJSON(json_text, simplifyDataFrame = TRUE, simplifyMatrix = TRUE)
}

# Helper function to read the data from a JSON file
load_data_from_json <- function(json_file) {

  data <- load_json_data(json_file)
  # Extract relevant fields
  mu <- data$mu
  logvar <- data$logvar
  speaker <- data$speaker
  vowel <- data$vowel
  var_dims <- data$var_dims
  min_var_latents <- data$min_var_latents
  
  # Convert data to appropriate formats if needed
  if (is.list(mu)) mu <- do.call(rbind, mu)
  if (is.list(logvar)) logvar <- do.call(rbind, logvar)
  if (is.list(speaker)) speaker <- unlist(speaker)
  if (is.list(vowel)) vowel <- unlist(vowel)
  if (is.list(var_dims)) var_dims <- unlist(var_dims)
  if (is.list(min_var_latents)) min_var_latents <- unlist(min_var_latents)
  
  return(list(
    mu = mu,
    logvar = logvar,
    speaker = speaker,
    vowel = vowel,
    var_dims = var_dims,
    min_var_latents = min_var_latents
  ))
}



# Function to analyze latent responses with respect to factors
analyze_latents_wrt_factor <- function(mu, logvar, varying_factor_values, fixed_factor_values, 
                                      latent_ids, fname, min_var_latents = 2, varying = "vowels", 
                                      max_speakers = 10) {
  # Get dimensions
  latent_dim <- ncol(mu)
  
  #Variance standardization for comparability
  all_values <- as.vector(mu)  # Flatten matrix into a single vector
  global_mean <- mean(all_values)
  global_sd <- sd(all_values)

  # Only standardize if there's variance in the data
  if (global_sd > 0) {
    # Apply the same standardization to all dimensions
    mu_std <- (mu - global_mean) / global_sd
  } else {
    mu_std <- mu  # Keep original if no variance
  }

  # Replace original mu with globally standardized version
  mu <- mu_std

  # Set fixed factor based on varying factor
  fixed_factor <- if (varying == "vowels") "speakers" else "vowels"
  
  # Filter to only include first 4 and last 2 latent dimensions
  if (length(latent_ids) > 6) {
    selected_latent_ids <- c(1:4, (latent_dim-1):latent_dim)
  } else {
    selected_latent_ids <- latent_ids
  }
  
  # Get unique factors
  unique_factors <- unique(varying_factor_values)
  unique_factors_fixed <- unique(fixed_factor_values)
  
  # Create a mapping from original speaker values to "SPKR X" labels
  if (fixed_factor == "speakers") {
    # Sort the unique fixed factors (speakers) to ensure consistent order
    unique_factors_fixed <- sort(unique_factors_fixed)
    # Create the mapping from speaker value to "SPKR X" label
    speaker_labels <- paste0("SPKR ", 1:length(unique_factors_fixed))
    names(speaker_labels) <- as.character(unique_factors_fixed)
  }
  
  if (varying == "speakers" && length(unique_factors) > max_speakers) {
    # Sample evenly from available speakers
    speaker_indices <- round(seq(1, length(unique_factors), length.out = max_speakers))
    selected_speakers <- unique_factors[speaker_indices]
    
    # Filter data to only include selected speakers
    keep_indices <- which(varying_factor_values %in% selected_speakers)
    mu <- mu[keep_indices, ]
    logvar <- logvar[keep_indices, ]
    varying_factor_values <- varying_factor_values[keep_indices]
    fixed_factor_values <- fixed_factor_values[keep_indices]
    
    # Update unique factors
    unique_factors <- unique(varying_factor_values)
  }
  
  # Limit number of speakers if needed
  if (varying == "vowels" && length(unique_factors_fixed) > max_speakers) {
    # Create a sequence to sample evenly from the speakers
    speaker_indices <- round(seq(1, length(unique_factors_fixed), length.out = max_speakers))
    unique_factors_fixed <- unique_factors_fixed[speaker_indices]
    
    # Also filter data to only include selected speakers
    keep_indices <- which(fixed_factor_values %in% unique_factors_fixed)
    mu <- mu[keep_indices, ]
    logvar <- logvar[keep_indices, ]
    varying_factor_values <- varying_factor_values[keep_indices]
    fixed_factor_values <- fixed_factor_values[keep_indices]
  }
  
  # Create a color palette mapping
  colors <- viridis(length(unique_factors), option = "D")
  names(colors) <- as.character(unique_factors)
  
  # Calculate x-axis range
  xaxis <- as.numeric(unique(fixed_factor_values))
  
  # Calculate global mu range for consistent y-axis limits
  mu_min <- min(mu[, selected_latent_ids])
  mu_max <- max(mu[, selected_latent_ids])
  y_padding <- 0.2 #(mu_max - mu_min) / 10
  y_limits <- c(-3 - y_padding, 3 + y_padding) #c(mu_min - y_padding, mu_max + y_padding)
  
  # Create plots for each selected latent dimension
  plots <- lapply(1:length(selected_latent_ids), function(i_idx) {
    i <- selected_latent_ids[i_idx]
    z_id <- latent_ids[i]
    
    # Create data frame for this latent dimension
    plot_data <- data.frame()
    
    for (j in 1:length(unique_factors)) {
      factor_value <- unique_factors[j]
      
      # Format the varying factor label
      if (varying == "vowels") {
        if (length(unique_factors) <= 5) {
          varying_factor <- INT_TO_VOWEL[j]
        } else if (length(unique_factors) <= 8) {
          varying_factor <- INT_TO_VOWEL_EXTENDED[j]
        } else {
          varying_factor <- as.character(factor_value)
        }
      } else if (varying == "speakers") {
        varying_factor <- paste0("SPKR", j)  # Keep original speaker IDs
      }
      
      # Find indices where the varying factor equals the current value
      factor_indices <- which(varying_factor_values == factor_value)
      
      # Add data for this factor value
      if (length(factor_indices) > 0) {
        temp_data <- data.frame(
          x = as.numeric(fixed_factor_values[factor_indices]),
          y = mu[factor_indices, i],
          group = varying_factor  # Use the factor value for grouping
        )
        plot_data <- rbind(plot_data, temp_data)
      }
    }
    
    # Create plot title
    is_unused <- i_idx > length(selected_latent_ids) - min_var_latents
    # Remove latent type from title text
    title_text <- bquote(z[.(z_id)])
    if (is_unused) {
      subtitle_text <- "'unused'"
    } else {
      subtitle_text <- ""
    }

    # Create the plot with improved styling
    p <- ggplot(plot_data, aes(x = x, y = y, color = group, group = group)) +
      # Lines only, no points
      geom_line(size = line_size) +
      ylim(y_limits) +
      labs(title = title_text, subtitle = subtitle_text) +
      theme_minimal() +
      theme(
        legend.position = "none",
        plot.margin = plot_margin,
        plot.title = element_text(size = plot_title_size, face = plot_title_face, hjust = plot_title_hjust, family = plot_font_family),
        plot.subtitle = element_text(size = plot_subtitle_size, face = plot_title_face, hjust = plot_title_hjust, family = plot_font_family),
        # Separate axis title styling for x and y
        axis.title.x = element_blank(),  # Still blank for sub-plots
        axis.title.y = element_blank(),  # Still blank for sub-plots
        axis.text.x = element_text(angle = 45, hjust = 1, size = axis_text_size_x, family = plot_font_family),
        axis.text.y = element_text(size = axis_text_size_y, family = plot_font_family),
        panel.grid = element_line(color = grid_line_color),
        panel.background = element_rect(fill = plot_background_color, color = NA),
        plot.background = element_rect(fill = plot_background_color, color = NA)
      )
    
    # Add custom x-axis labels
    if (fixed_factor == "vowels") {
      # When vowels are fixed, use vowel names on x-axis
      vowel_labels <- if (length(unique_factors_fixed) <= 5) {
        INT_TO_VOWEL[1:length(unique_factors_fixed)]
      } else {
        INT_TO_VOWEL_EXTENDED[1:length(unique_factors_fixed)]
      }
      p <- p + scale_x_continuous(
        breaks = as.numeric(xaxis),
        labels = vowel_labels
      )
    } else {
      # When speakers are fixed, use speaker labels
      p <- p + scale_x_continuous(
        breaks = as.numeric(xaxis),
        labels = paste0("", 1:length(xaxis))
      )
    }
    
    # Apply consistent color mapping with threshold to exclude bright yellows
    p <- p + scale_color_viridis_d(
      option = palette, 
      end = yellow_block_threshold  # Ensure the threshold is applied here
    )
    
    return(p)
  })
  
  # Create a common legend with proper labels
  legend_data <- data.frame(
    x = 1:length(unique_factors),
    y = rep(1, length(unique_factors))
  )
  
  # Add the group column with appropriate factor labels
  if (varying == "vowels") {
    if (length(unique_factors) <= 5) {
      legend_data$group <- INT_TO_VOWEL[1:length(unique_factors)]
    } else {
      legend_data$group <- INT_TO_VOWEL_EXTENDED[1:length(unique_factors)]
    }
  } else {
    # For speaker case, we use the original speaker IDs
    legend_data$group <- paste0("", 1:length(unique_factors))
  }
  
  n_plots <- length(plots)
  n_cols <- min(n_plots, 6)  # Cap at 6 columns (for 4 first + 2 last dimensions)
  n_rows <- ceiling(n_plots / n_cols)

  legend_title <- if (varying == "vowels") "Vowel" else "Speaker"

  # Create a proper legend
  legend_plot <- ggplot(legend_data, aes(x = x, y = y, color = group)) +
    geom_line(aes(group = 1), size = line_size) +
    scale_color_viridis_d(
      option = palette, 
      name = legend_title, 
      end = yellow_block_threshold  # Ensure the threshold is applied here
    ) +
    theme_void() +
    theme(
      legend.position = "right",  # Changed from "bottom" to "right"
      legend.direction = "vertical",  # Changed from "horizontal" to "vertical"
      legend.title = element_text(size = legend_title_size, face = legend_title_face, family = plot_font_family),
      legend.text = element_text(size = legend_text_size, face = legend_text_face, family = plot_font_family),
      legend.key.size = unit(legend_key_size * 1.5, "cm"),
      legend.key.width = unit(1.5, "cm"),  # Added explicit key width
      legend.key.height = unit(0.5, "cm"),  # Reduced key height for vertical layout
      legend.spacing.y = unit(0.3, "cm"),  # Changed from spacing.x to spacing.y
      legend.margin = margin(10, 10, 10, 10),
      legend.background = element_rect(fill = plot_background_color, color = NA)
    )

  if (!requireNamespace("cowplot", quietly = TRUE)) {
    install.packages("cowplot")
    library(cowplot)
  }

  # Extract the legend
  legend_grob <- cowplot::get_legend(legend_plot)  

  # Add a title plot for the first row with consistent styling
  title_grob <- textGrob(
    "", #paste0("Latent response across ", varying) 
    gp = gpar(fontsize = plot_sup_title_size, fontface = plot_title_face, family = plot_font_family)
  )

  plots_grid <- do.call(gridExtra::arrangeGrob, c(
    plots,
    list(ncol = n_cols)
  ))

  # Add bottom label with different x-axis title size
  bottom_grob <- textGrob(fixed_factor, 
                         gp = gpar(fontsize = axis_x_title_size, 
                                  family = plot_font_family))

  # Now create the full layout with title, plots, and bottom label
  main_grid <- gridExtra::arrangeGrob(
    title_grob,
    plots_grid,
    bottom_grob,
    ncol = 1,
    heights = unit(c(0.1, 0.8, 0.05), "npc")
  )

  # Combine with legend
  combined_plot <- gridExtra::arrangeGrob(
    main_grid, legend_grob,
    ncol = 2,
    widths = c(4, 1)  # 4:1 ratio for main grid to legend
  )

  # Save the plot with consistent background color
  ggsave(
    fname, 
    combined_plot, 
    width = min(28, 5 * n_cols + 4),  # Add space for legend
    height = 4 * n_rows + 2,  # More height for clarity
    limitsize = FALSE,
    bg = plot_background_color,
    dpi = 600
  )
  
  return(combined_plot)
}

# Update the run_analysis function with the min_var_latents parameter
run_analysis <- function(json_file, output_file, varying = "speakers", max_speakers = 10) {
  # Load data
  data <- load_data_from_json(json_file)
  
  # Run analysis with fixed min_var_latents = 2
  if (varying == "speakers") {
    analyze_latents_wrt_factor(
      data$mu, 
      data$logvar, 
      data$speaker,  # varying factor values (speakers)
      data$vowel,    # fixed factor values (vowels)
      data$var_dims, 
      output_file, 
      min_var_latents = 2, 
      varying = "speakers"
    )
  } else {
    analyze_latents_wrt_factor(
      data$mu, 
      data$logvar, 
      data$vowel,    # varying factor values (vowels)
      data$speaker,  # fixed factor values (speakers) 
      data$var_dims, 
      output_file, 
      min_var_latents = 2, 
      varying = "vowels",
      max_speakers = max_speakers
    )
  }
}

# Function to check if a latent file exists
check_latent_file <- function(base_dir, experiment, model, beta, ckp, latent_type, varying) {
  file_path <- file.path(base_dir, experiment, model, paste0('b', beta), ckp,
                         paste0(latent_type, '_varying_', varying, '.json'))
  return(file.exists(file_path))
}

# Function to get file path for a latent type
get_latent_file_path <- function(base_dir, experiment, model, beta, ckp, latent_type, varying) {
  return(file.path(base_dir, experiment, model, paste0('b', beta), ckp,
                  paste0(latent_type, '_varying_', varying, '.json')))
}

# Main function to create combined latent response plots
create_combined_latent_plot <- function(load_dir, save_dir, experiment, model, beta, ckp, varying, max_speakers = 10) {
  # Define all possible latent types to check
  latent_types <- c('X', 'OC1', 'OC2', 'OC3')
  
  # Check which latent files exist
  available_latents <- latent_types[sapply(latent_types, function(lt) {
    check_latent_file(load_dir, experiment, model, beta, ckp, lt, varying)
  })]
  
  if (length(available_latents) == 0) {
    stop("No latent files found!")
  }
  
  # Process each available latent
  latent_plots <- list()
  latent_data <- list()
  
  for (i in seq_along(available_latents)) {
    latent_type <- available_latents[i]
    # Get file path for this latent
    file_path <- get_latent_file_path(load_dir, experiment, model, beta, ckp, latent_type, varying)
    
    # Load data
    data <- load_data_from_json(file_path)
    latent_data[[latent_type]] <- data
    
    # Determine if this is the last latent type
    is_last_latent <- (i == length(available_latents))
    
    # Create individual plots for this latent (without legends or titles)
    latent_plots[[latent_type]] <- analyze_latents_without_legend(
      data$mu,
      data$logvar,
      if(varying == "speakers") data$speaker else data$vowel,
      if(varying == "speakers") data$vowel else data$speaker,
      data$var_dims,
      latent_type,
      min_var_latents = 2,
      varying = varying,
      max_speakers = max_speakers,
      is_last_latent = is_last_latent  # Pass this new parameter
    )
  }
  
  # Create a common legend
  legend <- create_common_legend(latent_data[[1]], varying)
  
  # Combine plots
  combined_plot <- combine_plots(latent_plots, available_latents, varying, fixed)
  
  # Save combined plot
  combined_save_path <- file.path(save_dir, paste0('combined_latent_responses_', varying, '.png'))
  ggsave(
    combined_save_path,
    combined_plot,
    width = min(28, 5 * 6 + 4),  # Width based on max 6 columns
    height = 4 * length(available_latents) * 1.5, # Height based on number of latent types
    limitsize = FALSE,
    bg = plot_background_color,
    dpi = 600
  )
  
  # Save separate legend
  legend_save_path <- file.path(save_dir, paste0('latent_responses_legend_', varying, '.png'))
  ggsave(
    legend_save_path,
    legend,
    width = 3,     # Reduced width for vertical legend
    height = 8,    # Increased height for vertical legend
    bg = plot_background_color,
    dpi = 600
  )
  
  return(list(combined_plot = combined_plot, legend = legend))
}

# Function to create plots without legends for combining
analyze_latents_without_legend <- function(mu, logvar, varying_factor_values, fixed_factor_values, 
                                          latent_ids, latent_type, min_var_latents = 2, 
                                          varying = "vowels", max_speakers = 10,
                                          is_last_latent = FALSE) {  # Add new parameter
  # Similar to analyze_latents_wrt_factor but returns only the plots list without combining them
  # Get dimensions
  latent_dim <- ncol(mu)
  
  # Variance standardization for comparability
  all_values <- as.vector(mu)
  global_mean <- mean(all_values)
  global_sd <- sd(all_values)

  # Only standardize if there's variance in the data
  if (global_sd > 0) {
    mu_std <- (mu - global_mean) / global_sd
  } else {
    mu_std <- mu
  }

  mu <- mu_std
  fixed_factor <- if (varying == "vowels") "speakers" else "vowels"
  
  # Filter to include first 4 and last 2 latent dimensions
  if (length(latent_ids) > 6) {
    selected_latent_ids <- c(1:4, (latent_dim-1):latent_dim)
  } else {
    selected_latent_ids <- latent_ids
  }
  
  # Sample processing code (similar to analyze_latents_wrt_factor)
  unique_factors <- unique(varying_factor_values)
  unique_factors_fixed <- unique(fixed_factor_values)
  
  # Rest of the filtering code...
  if (fixed_factor == "speakers") {
    unique_factors_fixed <- sort(unique_factors_fixed)
    speaker_labels <- paste0("SPKR ", 1:length(unique_factors_fixed))
    names(speaker_labels) <- as.character(unique_factors_fixed)
  }
  
  if (varying == "speakers" && length(unique_factors) > max_speakers) {
    speaker_indices <- round(seq(1, length(unique_factors), length.out = max_speakers))
    selected_speakers <- unique_factors[speaker_indices]
    
    keep_indices <- which(varying_factor_values %in% selected_speakers)
    mu <- mu[keep_indices, ]
    logvar <- logvar[keep_indices, ]
    varying_factor_values <- varying_factor_values[keep_indices]
    fixed_factor_values <- fixed_factor_values[keep_indices]
    
    unique_factors <- unique(varying_factor_values)
  }
  
  if (varying == "vowels" && length(unique_factors_fixed) > max_speakers) {
    speaker_indices <- round(seq(1, length(unique_factors_fixed), length.out = max_speakers))
    unique_factors_fixed <- unique_factors_fixed[speaker_indices]
    
    keep_indices <- which(fixed_factor_values %in% unique_factors_fixed)
    mu <- mu[keep_indices, ]
    logvar <- logvar[keep_indices, ]
    varying_factor_values <- varying_factor_values[keep_indices]
    fixed_factor_values <- fixed_factor_values[keep_indices]
  }
  
  xaxis <- as.numeric(unique(fixed_factor_values))
  
  mu_min <- min(mu[, selected_latent_ids])
  mu_max <- max(mu[, selected_latent_ids])
  y_padding <- 0.2
  y_limits <- c(-3 - y_padding, 3 + y_padding)
  
  # Create plots for each selected latent dimension
  plots <- lapply(1:length(selected_latent_ids), function(i_idx) {
    i <- selected_latent_ids[i_idx]
    z_id <- latent_ids[i]
    
    plot_data <- data.frame()
    
    for (j in 1:length(unique_factors)) {
      factor_value <- unique_factors[j]
      
      if (varying == "vowels") {
        if (length(unique_factors) <= 5) {
          varying_factor <- INT_TO_VOWEL[j]
        } else if (length(unique_factors) <= 8) {
          varying_factor <- INT_TO_VOWEL_EXTENDED[j]
        } else {
          varying_factor <- as.character(factor_value)
        }
      } else if (varying == "speakers") {
        varying_factor <- paste0("SPKR", j)
      }
      
      factor_indices <- which(varying_factor_values == factor_value)
      
      if (length(factor_indices) > 0) {
        temp_data <- data.frame(
          x = as.numeric(fixed_factor_values[factor_indices]),
          y = mu[factor_indices, i],
          group = varying_factor
        )
        plot_data <- rbind(plot_data, temp_data)
      }
    }
    
    # Determine if this plot is the first in a row (columns 1, 7, 13, etc.)
    is_first_in_row <- (i_idx - 1) %% 6 == 0
    
    # Determine if this plot is in the last row of the current latent type
    n_cols <- 6  # We're using 6 columns for layout
    total_plots <- length(selected_latent_ids)
    last_row_start <- (ceiling(total_plots / n_cols) - 1) * n_cols + 1
    is_in_last_row <- i_idx >= last_row_start
    
    # Only show x-axis elements if this is the last row of the last latent type
    show_x_elements <- is_in_last_row && is_last_latent
    
    is_unused <- i_idx > length(selected_latent_ids) - min_var_latents
    # Remove latent type from title text
    title_text <- bquote(z[.(z_id)])
    if (is_unused) {
      subtitle_text <- "'unused'"
    } else {
      subtitle_text <- ""
    }
    
    p <- ggplot(plot_data, aes(x = x, y = y, color = group, group = group)) +
      geom_line(size = line_size) +
      ylim(y_limits) +
      labs(
        title = title_text,
        subtitle = subtitle_text,
        # Add y-axis title only for the first plot in each row
        y = if(is_first_in_row) latent_type else NULL,
        # Add x-axis title only for plots in the last row of the last latent
        x = if(show_x_elements) fixed_factor else NULL
      ) +
      theme_minimal() +
      theme(
        legend.position = "none",
        plot.margin = plot_margin,
        plot.title = element_text(size = plot_title_size, face = plot_title_face, hjust = plot_title_hjust, family = plot_font_family),
        plot.subtitle = element_text(size = plot_subtitle_size, face = plot_title_face, hjust = plot_title_hjust, family = plot_font_family),
        # Y-axis title visible only for first plot in row
        axis.title.y = if(is_first_in_row) 
                      element_text(size = axis_y_title_size, family = plot_font_family, angle = 90)
                    else 
                      element_blank(),
        # X-axis title visible only for last row of last latent type
        axis.title.x = if(show_x_elements) 
                      element_text(size = axis_x_title_size, family = plot_font_family)
                    else 
                      element_blank(),
        # X-axis text visible only for last row of last latent type
        axis.text.x = if(show_x_elements) 
                     element_text(angle = 45, hjust = 1, size = axis_text_size_x, family = plot_font_family)
                   else 
                     element_blank(),
        axis.ticks.x = if(show_x_elements) element_line() else element_blank(),
        axis.text.y = element_text(size = axis_text_size_y, family = plot_font_family),
        panel.grid = element_line(color = grid_line_color),
        panel.background = element_rect(fill = plot_background_color, color = NA),
        plot.background = element_rect(fill = plot_background_color, color = NA)
      )
    
    # Add custom x-axis labels - make sure they only appear in the last row of the last latent type
    if (fixed_factor == "vowels") {
      # When vowels are fixed, use vowel names on x-axis
      vowel_labels <- if (length(unique_factors_fixed) <= 5) {
        INT_TO_VOWEL[1:length(unique_factors_fixed)]
      } else {
        INT_TO_VOWEL_EXTENDED[1:length(unique_factors_fixed)]
      }
      p <- p + scale_x_continuous(
        breaks = if(show_x_elements) as.numeric(xaxis) else NULL,
        labels = if(show_x_elements) vowel_labels else NULL
      )
    } else {
      # When speakers are fixed, use speaker labels
      p <- p + scale_x_continuous(
        breaks = if(show_x_elements) as.numeric(xaxis) else NULL,
        labels = if(show_x_elements) paste0("", 1:length(xaxis)) else NULL
      )
    }
    
    # Apply consistent color mapping with threshold
    p <- p + scale_color_viridis_d(option = palette, end = yellow_block_threshold)
    
    return(p)
  })
  
  return(plots)
}

# Function to create a common legend
create_common_legend <- function(data, varying) {
  unique_factors <- unique(if(varying == "vowels") data$vowel else data$speaker)
  
  legend_data <- data.frame(
    x = 1:length(unique_factors),
    y = rep(1, length(unique_factors))
  )
  
  if (varying == "vowels") {
    if (length(unique_factors) <= 5) {
      legend_data$group <- INT_TO_VOWEL[1:length(unique_factors)]
    } else {
      legend_data$group <- INT_TO_VOWEL_EXTENDED[1:length(unique_factors)]
    }
  } else {
    legend_data$group <- paste0(" ", 1:length(unique_factors))
  }
  
  legend_title <- if (varying == "vowels") "Vowel" else "Speaker"
  
  legend_plot <- ggplot(legend_data, aes(x = x, y = y, color = group)) +
    geom_line(aes(group = 1), size = line_size * 2) + # Larger lines in legend
    scale_color_viridis_d(option = palette, name = legend_title, end = yellow_block_threshold) +
    theme_void() +
    theme(
      legend.position = "right",  # Changed from "bottom" to "right"
      legend.direction = "vertical",  # Changed from "horizontal" to "vertical"
      legend.title = element_text(size = legend_title_size, face = legend_title_face, family = plot_font_family),
      legend.text = element_text(size = legend_text_size, face = legend_text_face, family = plot_font_family),
      legend.key.size = unit(legend_key_size * 1.5, "cm"),
      legend.key.width = unit(1.5, "cm"),  # Added explicit key width
      legend.key.height = unit(0.5, "cm"),  # Reduced key height for vertical layout
      legend.spacing.y = unit(0.3, "cm"),  # Changed from spacing.x to spacing.y
      legend.margin = margin(10, 10, 10, 10),
      legend.background = element_rect(fill = plot_background_color, color = NA)
    )
  
  # Extract just the legend
  legend_grob <- cowplot::get_legend(legend_plot)
  legend_only <- cowplot::ggdraw() + cowplot::draw_grob(legend_grob)
  
  return(legend_only)
}

# Function to combine individual plots
combine_plots <- function(latent_plots, available_latents, varying, fixed) {
  if (length(available_latents) == 1) {
    # If only one latent type, arrange its plots in a grid
    plots_grid <- wrap_plots(latent_plots[[available_latents[1]]], ncol = 6)
    
    # Add title only (x-axis labels are handled by individual plots)
    final_plot <- plots_grid +
      plot_annotation(
        title = "", #paste0("Latent response across ", varying)
        theme = theme(
          plot.title = element_text(size = plot_sup_title_size, family = plot_font_family, hjust = 0.5),
          plot.background = element_rect(fill = plot_background_color, color = NA)
        )
      )
  } else {
    # For multiple latent types, create a grid for each type
    latent_grids <- lapply(available_latents, function(latent_type) {
      grid <- wrap_plots(latent_plots[[latent_type]], ncol = 6) +
        plot_annotation(
          title = paste0(latent_type, " Response"),  # Removed "Latent" from title
          theme = theme(
            plot.title = element_text(size = plot_title_size, family = plot_font_family, hjust = 0.5),
            plot.background = element_rect(fill = plot_background_color, color = NA)
          )
        )
      return(grid)
    })
    
    # Stack all grids vertically
    final_plot <- wrap_plots(latent_grids, ncol = 1) +
      plot_annotation(
        title = "", #paste0("Latent response across ", varying)
        # Remove caption as we now have axis labels on individual plots
        theme = theme(
          plot.title = element_text(size = plot_sup_title_size, family = plot_font_family, hjust = 0.5),
          plot.background = element_rect(fill = plot_background_color, color = NA)
        )
      )
  }
  
  return(final_plot)
}

# Update the main run_analysis function to use the combined plot function
run_combined_analysis <- function(varying = "speakers", max_speakers = 10) {
  result <- create_combined_latent_plot(
    load_dir, save_dir, experiment, model, beta, ckp, varying, max_speakers
  )
  return(result)
}

# Call the modified function instead of the original
run_combined_analysis(varying = varying, max_speakers = 10)
