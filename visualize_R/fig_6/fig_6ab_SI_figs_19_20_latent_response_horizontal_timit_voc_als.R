#' Figure 6: This script generates figures 6a and 6b in the main paper,
#' and supplementary figures 19 and 20 in the supplementary material.
#' Appropriate setting of the parameters in lines 46-57 is required to reproduce the figures.

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
legend_text_size <- 30
legend_title_size <- 30
legend_text_face <- "plain"
legend_title_face <- "plain"
legend_key_size <- 2.5  # in cm

# Geom elements
line_size <- 1.2
grid_line_color <- "gray90"

# Plot margins
plot_margin <- margin(8, 8, 8, 8)

parent_save_dir <-  file.path('..','figures','fig_6','fig_6ab_SI_figs_19_20_latent_traversals_timit_voc_als')
dataset <- 'VOC_ALS'
model <- 'ewt' #'vae1D_FC_mel' vae1d_fc
transfer_from <- 'timit' # timit, sim_vowels / for VOC_ALS and iemocap
experiment <- 'fixed_phoneme' # fixed_phoneme, fixed_speaker, fixed_emotion_phoneme_speaker
emotion <- 'ang' # only needed for iemocap fixed_emotion_phoneme_speaker
# 'ang', 'hap', 'neu', 'sad'
feature <- 'mel'
NoC <- 4
dec_vae_type <- 'dual' # single_z, single_s, dual
betas <- 'bz01_bs01' # bz1_bs1 for DecVAEs or b1 for VAEs - 'bz1_bs1' / 'b1'
ckp <- 'training_ckp_epoch_99'
latent <- 'X' #set to X, then function will find all available latents in the same directory

#available experiments:
# timit - fixed_phoneme, fixed_speaker
# VOC_ALS - fixed_phoneme, fixed_kings_stage

#Set up paths
if (grepl("vae",model)) {
    parent_load_dir <- file.path('..','data','latent_traversal_data_vae')
} else {
    parent_load_dir <- file.path('..','data','latent_traversal_data')
}

if (grepl("VOC_ALS",dataset) || grepl("iemocap",dataset)) {
  experiment_load_dir <- file.path(parent_load_dir, paste0(dataset,'_', model,'_transfer_from_',transfer_from,'_',experiment))
} else {
  experiment_load_dir <- file.path(parent_load_dir, paste0(dataset,'_', model,'_',experiment))
}

if (grepl("vae",model)) {
  exact_model <- paste0(betas, '_', feature,'_',model)
} else {
  exact_model <- paste0(betas,'_NoC',NoC, '_', feature,'_',dec_vae_type)
}

ckp_dir <- file.path(experiment_load_dir, exact_model, ckp)

if (grepl("VOC_ALS",dataset) || grepl("iemocap",dataset)) {
  save_dir <- file.path(parent_save_dir, paste0(dataset,"_from_",transfer_from), experiment, model, exact_model, ckp)
} else {
  save_dir <- file.path(parent_save_dir, dataset, experiment, model, exact_model, ckp)
}
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

if (grepl("fixed_phoneme", experiment) && grepl("timit",dataset)) {
  varying <- 'speaker'
  varying_for_title <- 'speakers'
  fixed <- 'phoneme'
  fixed_for_axis <- 'phoneme'
} else if (grepl("fixed_phoneme", experiment) && grepl("VOC_ALS",dataset)) {
  varying <- 'king_stage'
  varying_for_title <- "King's Stage"
  fixed <- 'phoneme'
  fixed_for_axis <- 'phoneme'
} else if (grepl("fixed_speaker", experiment)) {
  varying <- 'phoneme'
  varying_for_title <- 'phonemes'
  fixed <- 'speaker'
  fixed_for_axis <- 'speaker'
} else if (grepl("fixed_kings_stage", experiment)) {
  varying <- 'phoneme'
  varying_for_title <- 'phonemes'
  fixed <- 'king_stage'
  fixed_for_axis <- "King's stage"
} else if (grepl("fixed_speaker_emotion", experiment)) {
  varying <- 'emotion'
  fixed <- 'speaker'
  fixed_for_axis <- 'speaker'
} else if (grepl("fixed_phoneme_emotion", experiment)) {
  varying <- 'emotion'
  varying_for_title <- 'emotions'
  fixed <- 'phoneme'
  fixed_for_axis <- 'phoneme'
} else if (grepl("fixed_emotion_phoneme_speaker", experiment)) {
  varying1 <- 'phoneme'
  varying2 <- 'speaker'
  varying1_for_title <- 'phonemes'
  varying2_for_title <- 'speakers'
  varying <- 'phonemes_speakers'
  fixed <- 'emotion'
  fixed_for_axis <- 'emotion'
}


# Create a general dictionary mapping function to select the right INT_TO_X dictionary
get_factor_mapping <- function(factor_type, dataset) {
  
  # Select the appropriate dictionary based on factor_type and dataset
  if (factor_type == "speaker") {
    if (exists("INT_TO_SPEAKER")) {
      mapping <- function(x) {
        result <- INT_TO_SPEAKER[x+1]
        if (is.na(result)) paste0("Speaker ", x) else result
      }
    }
  } else if (factor_type == "phoneme") {
    if (exists("INT_TO_PHONEME")) {
      mapping <- function(x) {
        result <- INT_TO_PHONEME[x+1]
        if (is.na(result)) paste0("Phoneme ", x) else result
      }
    }
  } else if (factor_type == "emotion") {
    if (exists("INT_TO_EMOTION")) {
      mapping <- function(x) {
        result <- INT_TO_EMOTION[x+1]
        if (is.na(result)) paste0("Emotion ", x) else result
      }
    }
  } else if (factor_type == "king_stage") {
    if (exists("INT_TO_KINGS_STAGE")) {
      mapping <- function(x) {
        result <- INT_TO_KINGS_STAGE[x+1]
        if (is.na(result)) paste0("Stage ", x) else result
      }
    }
  } else if (factor_type == "vowel") {
    if (exists("INT_TO_VOWEL")) {
      mapping <- function(x) {
        result <- INT_TO_VOWEL[x+1]
        if (is.na(result)) paste0("Vowel ", x) else result
      }
    }
  }
  
  return(mapping)
}

# Load factor mappings for each dataset
load_json_data <- function(file_path) {
    json_text <- readLines(file_path)
    json_text <- gsub("\\bNaN\\b", "\"NaN\"", json_text)
    fromJSON(json_text, simplifyDataFrame = TRUE, simplifyMatrix = TRUE)
}

vocab_path <- file.path('..','vocabularies')

# Load dictionaries based on dataset
if (grepl("timit", dataset)) {
    speaker_dict <- load_json_data(file.path(vocab_path, 'timit_speaker_dict.json'))
    phoneme_dict <- load_json_data(file.path(vocab_path, 'timit_phon39_dict.json'))
    INT_TO_SPEAKER <- names(speaker_dict)
    INT_TO_PHONEME <- names(phoneme_dict)
} else if (grepl("VOC_ALS", dataset)) {
    voc_als_dict <- load_json_data(file.path(vocab_path, 'voc_als_encodings.json'))
    kings_stage_values <- names(voc_als_dict$KingClinicalStage)
    phoneme_values <- names(voc_als_dict$phoneme)
    
    # Create both direction mappings
    KINGS_STAGE_TO_INT <- setNames(as.numeric(kings_stage_values), 
                                  as.character(voc_als_dict$KingClinicalStage))
    INT_TO_KINGS_STAGE <- names(KINGS_STAGE_TO_INT)
    
    PHONEME_TO_INT <- setNames(as.numeric(phoneme_values), 
                               as.character(voc_als_dict$phoneme))
    INT_TO_PHONEME <- names(PHONEME_TO_INT)
} else if (grepl("iemocap", dataset)) {
    speaker_dict <- load_json_data(file.path(vocab_path, 'iemocap_speaker_dict.json'))
    emotion_dict <- load_json_data(file.path(vocab_path, 'iemocap_emotion_dict.json'))
    phoneme_dict <- load_json_data(file.path(vocab_path, 'iemocap_phone_dict.json'))
    INT_TO_SPEAKER <- names(speaker_dict)
    INT_TO_EMOTION <- names(emotion_dict)
    INT_TO_PHONEME <- names(phoneme_dict)
}

# Modified helper function to read JSON data with support for multiple varying factors
load_data_from_json <- function(json_file) {
  data <- load_json_data(json_file)
  
  # Extract basic fields (common to all datasets)
  result <- list(
    mu = data$mu,
    logvar = data$logvar,
    var_dims = data$var_dims,
    min_var_latents = data$min_var_latents
  )
  
  # Handle matrix conversion if needed
  if (is.list(result$mu)) result$mu <- do.call(rbind, result$mu)
  if (is.list(result$logvar)) result$logvar <- do.call(rbind, result$logvar)
  if (is.list(result$var_dims)) result$var_dims <- unlist(result$var_dims)
  if (is.list(result$min_var_latents)) result$min_var_latents <- unlist(result$min_var_latents)
  
  # Extract factor values based on experiment type
  if (exists("varying1") && exists("varying2")) {
    # Special case: two varying factors (iemocap fixed_emotion_phoneme_speaker)
    result[[varying1]] <- data[[varying1]]
    result[[varying2]] <- data[[varying2]]
    result[[fixed]] <- data[[fixed]]
    
    # Convert list to vector if needed
    if (is.list(result[[varying1]])) result[[varying1]] <- unlist(result[[varying1]])
    if (is.list(result[[varying2]])) result[[varying2]] <- unlist(result[[varying2]])
    if (is.list(result[[fixed]])) result[[fixed]] <- unlist(result[[fixed]])
  } else {
    # Standard case: one varying, one fixed factor
    # Extract based on experiment configuration
    for (factor_name in c(varying, fixed)) {
      if (!is.null(data[[factor_name]])) {
        result[[factor_name]] <- data[[factor_name]]
        if (is.list(result[[factor_name]])) {
          result[[factor_name]] <- unlist(result[[factor_name]])
        }
      }
    }
  }
  
  return(result)
}


analyze_latents_wrt_factor <- function(mu, logvar, varying_factor_values, fixed_factor_values, 
                                      latent_ids, fname, min_var_latents = 2, varying = "phoneme", 
                                      max_items = 10) {
  # Get dimensions
  latent_dim <- ncol(mu)
  
  # Variance standardization for comparability
  all_values <- as.vector(mu)
  global_mean <- mean(all_values)
  global_sd <- sd(all_values)

  if (global_sd > 0) {
    mu_std <- (mu - global_mean) / global_sd
  } else {
    mu_std <- mu
  }

  mu <- mu_std

  # Filter latent dimensions
  if (length(latent_ids) > 6) {
    selected_latent_ids <- c(1:4, (latent_dim-1):latent_dim)
  } else {
    selected_latent_ids <- latent_ids
  }
  
  # Get unique factors
  unique_factors <- unique(varying_factor_values)
  unique_factors_fixed <- unique(fixed_factor_values)
  
  # Limit number of items if needed
  if (length(unique_factors) > max_items) {
    # Sample evenly from available items
    factor_indices <- round(seq(1, length(unique_factors), length.out = max_items))
    selected_factors <- unique_factors[factor_indices]
    
    # Filter data
    keep_indices <- which(varying_factor_values %in% selected_factors)
    mu <- mu[keep_indices, ]
    logvar <- logvar[keep_indices, ]
    varying_factor_values <- varying_factor_values[keep_indices]
    fixed_factor_values <- fixed_factor_values[keep_indices]
    
    # Update unique factors
    unique_factors <- unique(varying_factor_values)
  }
  
  # Limit number of fixed factors if needed
  if (length(unique_factors_fixed) > max_items) {
    fixed_indices <- round(seq(1, length(unique_factors_fixed), length.out = max_items))
    selected_fixed <- unique_factors_fixed[fixed_indices]
    
    keep_indices <- which(fixed_factor_values %in% selected_fixed)
    mu <- mu[keep_indices, ]
    logvar <- logvar[keep_indices, ]
    varying_factor_values <- varying_factor_values[keep_indices]
    fixed_factor_values <- fixed_factor_values[keep_indices]
    
    unique_factors_fixed <- unique(fixed_factor_values)
  }
  
  # Calculate x-axis range
  xaxis <- as.numeric(unique(fixed_factor_values))
  
  # Calculate y-axis limits
  mu_min <- min(mu[, selected_latent_ids])
  mu_max <- max(mu[, selected_latent_ids])
  y_padding <- 0.2
  y_limits <- c(-3 - y_padding, 3 + y_padding)
  
  # Get factor mapping functions
  varying_mapper <- get_factor_mapping(varying, dataset)
  fixed_mapper <- get_factor_mapping(fixed, dataset)
  
  # Create plots for each selected latent dimension
  plots <- lapply(1:length(selected_latent_ids), function(i_idx) {
    i <- selected_latent_ids[i_idx]
    z_id <- latent_ids[i]
    
    # Create data frame for this latent dimension
    plot_data <- data.frame()
    
    for (j in 1:length(unique_factors)) {
      factor_value <- unique_factors[j]
      
      # Format the varying factor label using the appropriate mapping
      varying_factor <- varying_mapper(factor_value)
      if (is.na(varying_factor)) {
        varying_factor <- paste0(varying, " ", j)
      }
      
      # Find indices where the varying factor equals the current value
      factor_indices <- which(varying_factor_values == factor_value)
      
      # Add data for this factor value
      if (length(factor_indices) > 0) {
        temp_data <- data.frame(
          x = as.numeric(fixed_factor_values[factor_indices]),
          y = mu[factor_indices, i],
          group = varying_factor
        )
        plot_data <- rbind(plot_data, temp_data)
      }
    }
    
    # Create plot title with mathematical notation
    is_unused <- i_idx > length(selected_latent_ids) - min_var_latents
    # Remove latent type from title text
    title_text <- bquote(z[.(z_id)])
    if (is_unused) {
      subtitle_text <- "'unused'"
    } else {
      subtitle_text <- ""
    }
    
    # Create the plot
    p <- ggplot(plot_data, aes(x = x, y = y, color = group, group = group)) +
      geom_line(size = line_size) +
      ylim(y_limits) +
      labs(title = title_text, subtitle = subtitle_text) +
      theme_minimal() +
      theme(
        legend.position = "none",
        plot.margin = plot_margin,
        plot.title = element_text(size = plot_title_size, face = plot_title_face, hjust = plot_title_hjust, family = plot_font_family),
        plot.subtitle = element_text(size = plot_subtitle_size, face = plot_title_face, hjust = plot_title_hjust, family = plot_font_family),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1, size = axis_text_size_x, family = plot_font_family),
        axis.text.y = element_text(size = axis_text_size_y, family = plot_font_family),
        panel.grid = element_line(color = grid_line_color),
        panel.background = element_rect(fill = plot_background_color, color = NA),
        plot.background = element_rect(fill = plot_background_color, color = NA)
      )
    
    # Add custom x-axis labels using the appropriate mapping
    fixed_labels <- sapply(xaxis, fixed_mapper)
    # Fallback to numeric labels if mapping returns NA
    fixed_labels[is.na(fixed_labels)] <- paste0(fixed, " ", which(is.na(fixed_labels)))
    
    p <- p + scale_x_continuous(
      breaks = as.numeric(xaxis),
      labels = fixed_labels
    )
    
    # Apply color palette with threshold
    p <- p + scale_color_viridis_d(
      option = palette, 
      end = yellow_block_threshold
    )
    
    return(p)
  })
  
  # Create legend data
  legend_data <- data.frame(
    x = 1:length(unique_factors),
    y = rep(1, length(unique_factors))
  )
  
  # Format legend labels using the appropriate mapping
  legend_labels <- sapply(unique_factors, varying_mapper)
  # Fallback to numeric labels if mapping returns NA
  legend_labels[is.na(legend_labels)] <- paste0(varying, " ", which(is.na(legend_labels)))
  legend_data$group <- legend_labels
  
  legend_title <- tools::toTitleCase(varying)  # Capitalize first letter
  
  # Create legend
  legend_plot <- ggplot(legend_data, aes(x = x, y = y, color = group)) +
    geom_line(aes(group = 1), size = line_size) +
    scale_color_viridis_d(
      option = palette, 
      name = legend_title, 
      end = yellow_block_threshold
    ) +
    theme_void() +
    theme(
      legend.position = "right",
      legend.direction = "vertical",
      legend.title = element_text(size = legend_title_size, face = legend_title_face, family = plot_font_family),
      legend.text = element_text(size = legend_text_size, face = legend_text_face, family = plot_font_family),
      legend.key.size = unit(legend_key_size * 1.5, "cm"),
      legend.key.width = unit(1.5, "cm"),
      legend.key.height = unit(0.5, "cm"),
      legend.spacing.y = unit(0.3, "cm"),
      legend.margin = margin(10, 10, 10, 10),
      legend.background = element_rect(fill = plot_background_color, color = NA)
    )

  # Extract the legend
  legend_grob <- cowplot::get_legend(legend_plot)  

  # Add a title plot for the first row with consistent styling
  title_grob <- textGrob(
    "", #paste0("Latent response across ", varying_for_title) 
    gp = gpar(fontsize = plot_sup_title_size, fontface = plot_title_face, family = plot_font_family)
  )

  plots_grid <- do.call(gridExtra::arrangeGrob, c(
    plots,
    list(ncol = 6)
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


# Function to check if a latent file exists
check_latent_file <- function(base_dir, latent_type, varying, fixed) {
  if (grepl("fixed_emotion_phoneme_speaker", experiment)) {
    file_path <- file.path(base_dir,
            paste0(latent_type, '_varying_', varying, '_fixed_emotion_', emotion, '.json'))
  } else {
    file_path <- file.path(base_dir, 
            paste0(latent_type, '_varying_', varying, '_fixed_', fixed, '.json'))
  }

  return(file.exists(file_path))
}

# Function to get file path for a latent type
get_latent_file_path <- function(base_dir, latent_type, varying, fixed) {
    if (grepl("fixed_emotion_phoneme_speaker", experiment)) {
        return(file.path(base_dir, paste0(latent_type, '_varying_', varying, '_fixed_emotion_', emotion, '.json')))
    } else {
        return(file.path(base_dir, paste0(latent_type, '_varying_', varying, '_fixed_', fixed, '.json')))
    }
}

# Main function to create combined latent response plots
create_combined_latent_plot <- function(load_dir, save_dir, varying, fixed, max_items = 10) {
  # Define all possible latent types to check
  latent_types <- c('X', 'OC1', 'OC2', 'OC3','OC4')
  
  # Check which latent files exist
  available_latents <- latent_types[sapply(latent_types, function(lt) {
    check_latent_file(load_dir, lt, varying, fixed)
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
    file_path <- get_latent_file_path(load_dir, latent_type, varying, fixed)
    
    # Load data
    data <- load_data_from_json(file_path)
    latent_data[[latent_type]] <- data
    
    # Determine if this is the last latent type
    is_last_latent <- (i == length(available_latents))
    
    # Create individual plots for this latent (without legends or titles)
    latent_plots[[latent_type]] <- analyze_latents_without_legend(
      data$mu,
      data$logvar,
      data[[varying]],  # Use double brackets here
      data[[fixed]],    # Use double brackets here
      data$var_dims,
      latent_type,
      min_var_latents = 2,
      varying_factor = varying,
      fixed_factor = fixed,
      max_items = max_items,
      is_last_latent = is_last_latent
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
    width = 3,    
    height = 8,    
    bg = plot_background_color,
    dpi = 600
  )
  
  return(list(combined_plot = combined_plot, legend = legend))
}

# Function to create plots without legends for combining
analyze_latents_without_legend <- function(mu, logvar, varying_factor_values, fixed_factor_values, 
                                          latent_ids, latent_type, min_var_latents = 2, 
                                          varying_factor = "phoneme", fixed_factor = "speaker", max_items = 10,
                                          is_last_latent = FALSE) {
  # Get dimensions
  latent_dim <- ncol(mu)
  
  # Variance standardization for comparability
  all_values <- as.vector(mu)
  global_mean <- mean(all_values)
  global_sd <- sd(all_values)

  if (global_sd > 0) {
    mu_std <- (mu - global_mean) / global_sd
  } else {
    mu_std <- mu
  }

  mu <- mu_std
  
  # Filter to include first 4 and last 2 latent dimensions
  if (length(latent_ids) > 6) {
    selected_latent_ids <- c(1:4, (latent_dim-1):latent_dim)
  } else {
    selected_latent_ids <- latent_ids
  }
  
  # Sample processing code (similar to analyze_latents_wrt_factor)
  unique_factors <- unique(varying_factor_values)
  unique_factors_fixed <- unique(fixed_factor_values)
  
  # Get factor mapping functions
  if (grepl("fixed_emotion_phoneme_speaker", experiment)) {
    varying_mapper1 <- get_factor_mapping(varying1, dataset)
    varying_mapper2 <- get_factor_mapping(varying2, dataset)
  } else {
    varying_mapper <- get_factor_mapping(varying_factor, dataset)
  }
  fixed_mapper <- get_factor_mapping(fixed_factor, dataset)
  
  # Limit number of items if needed
  if (length(unique_factors) > max_items) {
    factor_indices <- round(seq(1, length(unique_factors), length.out = max_items))
    selected_factors <- unique_factors[factor_indices]
    
    keep_indices <- which(varying_factor_values %in% selected_factors)
    mu <- mu[keep_indices, ]
    logvar <- logvar[keep_indices, ]
    varying_factor_values <- varying_factor_values[keep_indices]
    fixed_factor_values <- fixed_factor_values[keep_indices]
    
    unique_factors <- unique(varying_factor_values)
  }
  
  # Limit number of fixed factors if needed
  if (length(unique_factors_fixed) > max_items) {
    fixed_indices <- round(seq(1, length(unique_factors_fixed), length.out = max_items))
    selected_fixed <- unique_factors_fixed[fixed_indices]
    
    keep_indices <- which(fixed_factor_values %in% selected_fixed)
    mu <- mu[keep_indices, ]
    logvar <- logvar[keep_indices, ]
    varying_factor_values <- varying_factor_values[keep_indices]
    fixed_factor_values <- fixed_factor_values[keep_indices]
    
    unique_factors_fixed <- unique(fixed_factor_values)
  }
  
  # Create an evenly-spaced sequence for the x-axis
  # Instead of using actual numeric values that might have gaps
  xaxis_original <- as.numeric(unique(fixed_factor_values))
  xaxis_positions <- seq_along(xaxis_original)  # Creates 1, 2, 3, ... for each unique fixed factor
  
  # Create a mapping from original factor values to positions
  xaxis_mapping <- setNames(xaxis_positions, xaxis_original)
  
  # Calculate y-axis limits
  mu_min <- min(mu[, selected_latent_ids])
  mu_max <- max(mu[, selected_latent_ids])
  y_padding <- 0.2
  y_limits <- c(-3 - y_padding, 3 + y_padding)
  
  # Create plots for each selected latent dimension
  plots <- lapply(1:length(selected_latent_ids), function(i_idx) {
    i <- selected_latent_ids[i_idx]
    z_id <- latent_ids[i]
    
    plot_data <- data.frame()
    
    # Format the varying factor label using the appropriate mapping
    for (j in 1:length(unique_factors)) {
        factor_value <- unique_factors[j]
        
        # Get the mapped label, using direct mapping to avoid NA issues
        if (varying_factor == "king_stage" && exists("INT_TO_KINGS_STAGE")) {
            varying_label <- INT_TO_KINGS_STAGE[as.character(factor_value)]
            if (is.na(varying_label)) varying_label <- paste0("Stage ", j)
        } else if (varying_factor == "phoneme" && exists("INT_TO_PHONEME")) {
            varying_label <- INT_TO_PHONEME[as.character(factor_value)]
            if (is.na(varying_label)) varying_label <- paste0("Phoneme ", j)
        } else {
            varying_label <- varying_mapper(factor_value)
            if (is.na(varying_label)) varying_label <- paste0(tools::toTitleCase(varying_factor), " ", j)
        }
        
        # Find indices where the varying factor equals the current value
        factor_indices <- which(varying_factor_values == factor_value)
        
        # Add data for this factor value
        if (length(factor_indices) > 0) {
            temp_data <- data.frame(
              x_original = as.numeric(fixed_factor_values[factor_indices]),
              y = mu[factor_indices, i],
              group = varying_label
            )
            # Map the original x values to evenly-spaced positions
            temp_data$x <- xaxis_mapping[as.character(temp_data$x_original)]
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
        x = if(show_x_elements) fixed_for_axis else NULL  # Use formatted title
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
    if (show_x_elements) {
      # Get fixed factor labels using direct dictionary access based on factor type
      if (fixed_factor == "phoneme" && exists("INT_TO_PHONEME")) {
        fixed_labels <- sapply(xaxis_original, function(x) {
          INT_TO_PHONEME[x+1] 
        })
      } else if (fixed_factor == "king_stage" && exists("INT_TO_KINGS_STAGE")) {
        fixed_labels <- sapply(xaxis_original, function(x) {
          INT_TO_KINGS_STAGE[x+1]
        })
      } else {
        # For other factor types, use the mapper function
        fixed_labels <- sapply(xaxis_original, fixed_mapper)
      }
      
      # Handle any NA values with simple numbering (without prefixes)
      fixed_labels[is.na(fixed_labels)] <- as.character(which(is.na(fixed_labels)))
    } else {
      fixed_labels <- NULL
    }
    
    # Use the evenly-spaced positions for the breaks, but maintain the original labels
    p <- p + scale_x_continuous(
      breaks = if(show_x_elements) xaxis_positions else NULL,
      labels = if(show_x_elements) fixed_labels else NULL
    )
    
    # Apply consistent color mapping with threshold
    p <- p + scale_color_viridis_d(option = palette, end = yellow_block_threshold)
    
    return(p)
  })
  
  return(plots)
}

# Function to create a common legend
create_common_legend <- function(data, varying) {
  unique_factors <- unique(if(exists("varying1") && exists("varying2")) 
                          c(data[[varying1]], data[[varying2]]) 
                        else data[[varying]])
  
  legend_data <- data.frame(
    x = 1:length(unique_factors),
    y = rep(1, length(unique_factors))
  )
  
  # Apply direct mapping for legend labels based on factor type
  if (varying == "king_stage" && exists("INT_TO_KINGS_STAGE")) {
    legend_labels <- sapply(unique_factors, function(x) {
      # Get the actual King Stage label without prefix
      INT_TO_KINGS_STAGE[x+1]
    })
  } else if (varying == "phoneme" && exists("INT_TO_PHONEME")) {
    legend_labels <- sapply(unique_factors, function(x) {
      # Get the actual Phoneme label without prefix
      INT_TO_PHONEME[x+1]
    })
  } else {
    # Fall back to generic mapper
    varying_mapper <- get_factor_mapping(varying, dataset)
    legend_labels <- sapply(unique_factors, varying_mapper)
  }
  
  # Handle any NA values with simple numbering (without prefixes)
  legend_labels[is.na(legend_labels)] <- paste0(varying, " ", which(is.na(legend_labels)))
  
  legend_data$group <- legend_labels
  
  # Format the legend title properly
  legend_title <- tools::toTitleCase(gsub("_", " ", varying_for_title))
  
  # Create legend with correctly mapped values
  legend_plot <- ggplot(legend_data, aes(x = x, y = y, color = group)) +
    geom_line(aes(group = 1), size = line_size * 2) +
    scale_color_viridis_d(
      option = palette, 
      name = legend_title,  # Use formatted title 
      end = yellow_block_threshold
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
        title = "", #paste0("Latent response across ", varying_for_title)
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
        title = "", #paste0("Latent response across ", varying_for_title)
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
run_combined_analysis <- function(varying = "phoneme", fixed = "speaker", max_items = 10) {
  if (exists("varying1") && exists("varying2")) {
    # Special case: two varying factors (e.g., iemocap fixed_emotion_phoneme_speaker)
    # This would require a custom implementation
    message("Multiple varying factors detected. Using primary varying factor:", varying)
  }
  
  result <- create_combined_latent_plot(
    ckp_dir, save_dir, varying, fixed, max_items
  )
  return(result)
}

run_combined_analysis(varying = varying, fixed = fixed, max_items = 10)

