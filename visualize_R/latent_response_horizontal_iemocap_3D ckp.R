library(jsonlite)
library(ggplot2)
library(dplyr)
library(purrr)
library(tidyr)
library(viridis)
library(gridExtra)
library(grid)
library(cowplot)
library(patchwork)
library(plotly)  # For 3D plots
library(reshape2) # For data manipulation

# Enhanced style and font parameters - consolidated for all plot types
plot_font_family <- "Arial"
plot_background_color <- "white"
plot_text_color <- "black"
palette <- "turbo"
# For turbo palette, we want to skip just the yellow portion in the middle (0.4-0.6)
# We'll handle this differently for ggplot vs plotly
# For ggplot discrete scale, we'll use option 3 below
custom_palette_option <- 3  # 1=blues only, 2=reds only, 3=both with yellow gap

# For turbo palette, yellow is in the middle, so we need different approach
# Option 1: Use begin/end to skip the yellow section (appears around 0.4-0.6 of the range)
palette_begin <- 0.0  # Start from blue
palette_end <- 0.4    # Stop before yellow (or use 0.6-1.0 for orange-red part)
# Option 2: Use direction=-1 to reverse the palette if you prefer the red-to-blue direction
palette_direction <- 1  # 1 for regular direction, -1 to reverse
# yellow_block_threshold is not useful for turbo since yellow is in the middle

# Plot type-specific parameters - expanded for all plots
axis_title_size <- 23        # Used for all axis titles
axis_text_size <- 15         # Base size for axis text
axis_text_size_small <- 15   # For 3D plots where space is limited
plot_title_size <- 35
plot_sup_title_size <- 60 
plot_title_face <- "plain"
plot_title_hjust <- 0.5

# Legend elements
legend_text_size <- 30
legend_title_size <- 30
legend_text_face <- "plain"
legend_title_face <- "plain"
legend_key_size <- 3  # in cm

# Geom elements
line_size <- 1.2
point_size <- 2
grid_line_color <- "gray90"

# 3D plot specific
zoom_level <- 3 # Zoom level for 3D plots
surface_opacity <- 0.9       # Opacity for 3D surfaces
camera_eye_x <- 1.8          # Camera position X
camera_eye_y <- 1.8          # Camera position Y
camera_eye_z <- 0.7         # Camera position Z
aspect_ratio_x <- 1.5        # X dimension aspect ratio for 3D
aspect_ratio_y <- 1.5        # Y dimension aspect ratio for 3D
aspect_ratio_z <- 0.8        # Z dimension aspect ratio for 3D

# Plot margins
plot_margin <- margin(8, 8, 8, 8)
plot_margin_small <- margin(5, 5, 5, 5)

parent_save_dir <-  file.path('..','figures','latent_traversals')
dataset <- 'iemocap'
model <- 'ewt' #'vae1D_FC_mel'
transfer_from <- 'timit' # from_timit, from_sim_vowels / for VOC_ALS and iemocap
experiment <- 'fixed_emotion_phoneme_speaker'
feature <- 'mel'
NoC <- 4
dec_vae_type <- 'dual' # single_z, single_s, dual
betas <- 'bz1_bs1' # bz1_bs1 for DecVAEs or b1 for VAEs
ckp <- 'training_ckp_epoch_24'
latent <- 'X' #set to X, then function will find all available latents in the same directory

# Available emotions in iemocap
available_emotions <- c('ang', 'hap', 'neu', 'sad')

# Create emotion display name mapping
emotion_display_names <- c(
  'ang' = 'Angry',
  'hap' = 'Happy',
  'neu' = 'Neutral',
  'sad' = 'Sad'
)

# Set up paths
if (grepl("vae",model)) {
    parent_load_dir <- file.path('D:','latent_traversal_data_vae')
} else {
    parent_load_dir <- file.path('D:','latent_traversal_data')
}

if (grepl("VOC_ALS",dataset) || grepl("iemocap",dataset)) {
  experiment_load_dir <- file.path(parent_load_dir, paste0(dataset,'_', model,'_transfer_from_',transfer_from,'_',experiment))
} else {
  experiment_load_dir <- file.path(parent_load_dir, paste0(dataset,'_', model,'_',experiment))
}

exact_model <- paste0(betas,'_NoC',NoC, '_', feature,'_',dec_vae_type)
ckp_dir <- file.path(experiment_load_dir, exact_model, ckp)

save_dir <- file.path(parent_save_dir, dataset, experiment, model, exact_model, ckp)
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

# Setting up the iemocap specific parameters
varying1 <- 'phoneme'
varying2 <- 'speaker'
varying1_for_title <- 'phonemes'
varying2_for_title <- 'speakers'
varying <- 'phoneme_speaker'
fixed <- 'emotion'

# Create a general dictionary mapping function to select the right INT_TO_X dictionary
get_factor_mapping <- function(factor_type, dataset) {
  # Select the appropriate dictionary based on factor_type and dataset
  if (factor_type == "speaker") {
    if (exists("INT_TO_SPEAKER")) {
      mapping <- function(x) {
        result <- INT_TO_SPEAKER[x+1]
        if (is.na(result)) paste0("Speaker ", x) else result
      }
      return(mapping)
    }
  } else if (factor_type == "phoneme") {
    if (exists("INT_TO_PHONEME")) {
      mapping <- function(x) {
        result <- INT_TO_PHONEME[x+1]
        if (is.na(result)) paste0("Phoneme ", x) else result
      }
      return(mapping)
    }
  } else if (factor_type == "emotion") {
    if (exists("INT_TO_EMOTION")) {
      mapping <- function(x) {
        result <- INT_TO_EMOTION[x+1]
        if (is.na(result)) paste0("Emotion ", x) else result
      }
      return(mapping)
    }
  }
  
  # Default case if no mapping was found
  warning(paste("No mapping found for factor type:", factor_type))
  return(function(x) paste0(factor_type, " ", x))
}

# Load factor mappings for each dataset
load_json_data <- function(file_path) {
    json_text <- readLines(file_path)
    json_text <- gsub("\\bNaN\\b", "\"NaN\"", json_text)
    fromJSON(json_text, simplifyDataFrame = TRUE, simplifyMatrix = TRUE)
}

vocab_path <- file.path('..','vocabularies')

# Load iemocap dictionaries
speaker_dict <- load_json_data(file.path(vocab_path, 'iemocap_speaker_dict.json'))
emotion_dict <- load_json_data(file.path(vocab_path, 'iemocap_emotion_dict.json'))
phoneme_dict <- load_json_data(file.path(vocab_path, 'iemocap_phone_dict.json'))
INT_TO_SPEAKER <- names(speaker_dict)
INT_TO_EMOTION <- names(emotion_dict)
INT_TO_PHONEME <- names(phoneme_dict)

# Function to check if a latent file exists for a specific emotion
check_latent_file <- function(base_dir, latent_type, emotion) {
  file_path <- file.path(base_dir, paste0(latent_type, '_varying_phonemes_speakers_fixed_emotion_', emotion, '.json'))
  return(file.exists(file_path))
}

# Function to get file path for a latent type and emotion
get_latent_file_path <- function(base_dir, latent_type, emotion) {
  return(file.path(base_dir, paste0(latent_type, '_varying_phonemes_speakers_fixed_emotion_', emotion, '.json')))
}

# Function to load data from all emotion files for a specific latent type
load_all_emotions_data <- function(base_dir, latent_type, available_emotions) {
  all_data <- list()
  
  for (emotion in available_emotions) {
    # Check if file exists
    if (check_latent_file(base_dir, latent_type, emotion)) {
      # Get file path
      file_path <- get_latent_file_path(base_dir, latent_type, emotion)
      
      # Load data
      data <- load_json_data(file_path)
      
      # Extract basic fields
      result <- list(
        mu = data$mu,
        logvar = data$logvar,
        var_dims = data$var_dims,
        min_var_latents = data$min_var_latents,
        phoneme = data$phoneme,
        speaker = data$speaker,
        emotion = rep(as.numeric(which(available_emotions == emotion) - 1), length(data$phoneme))
      )
      
      # Handle matrix conversion if needed
      if (is.list(result$mu)) result$mu <- do.call(rbind, result$mu)
      if (is.list(result$logvar)) result$logvar <- do.call(rbind, result$logvar)
      if (is.list(result$var_dims)) result$var_dims <- unlist(result$var_dims)
      if (is.list(result$min_var_latents)) result$min_var_latents <- unlist(result$min_var_latents)
      if (is.list(result$phoneme)) result$phoneme <- unlist(result$phoneme)
      if (is.list(result$speaker)) result$speaker <- unlist(result$speaker)
      
      all_data[[emotion]] <- result
    }
  }
  
  return(all_data)
}

# Function to combine data from all emotions into a single dataset
combine_emotion_data <- function(all_data) {
  # Initialize combined data
  combined <- list(
    mu = NULL,
    logvar = NULL,
    phoneme = NULL,
    speaker = NULL,
    emotion = NULL
  )
  
  # Combine data from all emotions
  for (emotion_data in all_data) {
    combined$mu <- rbind(combined$mu, emotion_data$mu)
    combined$logvar <- rbind(combined$logvar, emotion_data$logvar)
    combined$phoneme <- c(combined$phoneme, emotion_data$phoneme)
    combined$speaker <- c(combined$speaker, emotion_data$speaker)
    combined$emotion <- c(combined$emotion, emotion_data$emotion)
  }
  
  # Use var_dims and min_var_latents from the first emotion data
  first_emotion <- names(all_data)[1]
  combined$var_dims <- all_data[[first_emotion]]$var_dims
  combined$min_var_latents <- all_data[[first_emotion]]$min_var_latents
  
  return(combined)
}

# Function to create a 2D plot of latent responses for a specific latent dimension and emotion
create_2D_line_plots <- function(data, latent_idx, emotion_idx, latent_type, varying_factor = "phoneme", max_items = 10, 
                                is_last_row = FALSE, is_first_column = FALSE, is_unused = FALSE, show_legend = FALSE) {
  # Filter data for the specific emotion
  filtered_data <- list(
    mu = data$mu[data$emotion == emotion_idx, ],
    phoneme = data$phoneme[data$emotion == emotion_idx],
    speaker = data$speaker[data$emotion == emotion_idx]
  )
  
  # Standardize latent response values (like in latent_response_horizontal.R)
  all_values <- as.vector(filtered_data$mu)
  global_mean <- mean(all_values)
  global_sd <- sd(all_values)

  if (global_sd > 0) {
    filtered_data$mu <- (filtered_data$mu - global_mean) / global_sd
  }
  
  # Determine which factor is varying and which is fixed
  if (varying_factor == "phoneme") {
    varying_values <- filtered_data$phoneme
    fixed_values <- filtered_data$speaker
    varying_mapper <- get_factor_mapping("phoneme", dataset)
    fixed_mapper <- get_factor_mapping("speaker", dataset)
    x_title <- "Phoneme"
    legend_title <- "Speaker"
  } else {
    varying_values <- filtered_data$speaker
    fixed_values <- filtered_data$phoneme
    varying_mapper <- get_factor_mapping("speaker", dataset)
    fixed_mapper <- get_factor_mapping("phoneme", dataset)
    x_title <- "Speaker"
    legend_title <- "Phoneme"
  }
  
  # Get unique values for both factors
  unique_varying <- sort(unique(varying_values))
  unique_fixed <- sort(unique(fixed_values))
  
  # Randomly sample varying factors to improve visibility
  if (length(unique_varying) > max_items) {
    set.seed(42)  # For reproducibility
    selected_indices <- sample(length(unique_varying), max_items)
    unique_varying <- unique_varying[selected_indices]
  }
  
  # Create evenly-spaced x-axis values
  x_positions <- seq_along(unique_varying)
  x_mapping <- setNames(x_positions, as.character(unique_varying))
  
  # Create a data frame for plotting
  plot_data <- data.frame()
  
  # Get the actual dimension value from var_dims for the plot title only
  dimension_value <- data$var_dims[latent_idx]
  
  for (fixed_val in unique_fixed) {
    # fixed_label <- fixed_mapper(fixed_val)
    fixed_label <- fixed_mapper(fixed_val)
    if (is.na(fixed_label)) fixed_label <- paste0(ifelse(varying_factor == "phoneme", "Speaker ", "Phoneme "), fixed_val)
    
    # Create a line for each fixed value
    line_data <- data.frame(
      x_original = numeric(),
      x = numeric(),
      y = numeric(),
      group = character()
    )
    
    for (varying_val in unique_varying) {
      # Find indices where both values match
      if (varying_factor == "phoneme") {
        indices <- which(filtered_data$phoneme == varying_val & filtered_data$speaker == fixed_val)
      } else {
        indices <- which(filtered_data$speaker == varying_val & filtered_data$phoneme == fixed_val)
      }
      
      if (length(indices) > 0) {
        # IMPORTANT: Use latent_idx directly for data access, not dimension_value
        latent_value <- mean(filtered_data$mu[indices, latent_idx], na.rm = TRUE)
        
        # Map to evenly-spaced x position
        x_pos <- x_mapping[as.character(varying_val)]
        
        # Add to line data
        line_data <- rbind(line_data, data.frame(
          x_original = varying_val,
          x = x_pos,
          y = latent_value,
          group = fixed_label
        ))
      }
    }
    
    # Add complete line to plot data
    plot_data <- rbind(plot_data, line_data)
  }
  
  # Apply consistent y-axis limits
  y_padding <- 0.2
  y_limits <- c(-3 - y_padding, 3 + y_padding)
  
  # Create the 2D line plot using dimension_value only for display
  p <- ggplot(plot_data, aes(x = x, y = y, color = group, group = group)) +
    geom_line(size = line_size) +
    geom_point(size = point_size) +
    ylim(y_limits) +
    # Handle color palette based on selected option
    {
      if (custom_palette_option == 1) {
        # Option 1: Blues/greens only (0.0-0.4)
        scale_color_viridis_d(
          option = palette, 
          begin = 0.0,
          end = 0.4,
          direction = 1
        )
      } else if (custom_palette_option == 2) {
        # Option 2: Oranges/reds only (0.6-1.0)  
        scale_color_viridis_d(
          option = palette,
          begin = 0.6,
          end = 1.0,
          direction = 1
        )
      } else {
        # Option 3: Custom blues->greens->oranges->reds (skipping yellow)
        scale_color_manual(
          values = colorRampPalette(c(
            viridis::turbo(n = 100, begin = 0.0, end = 0.4),  # Blues to greens
            viridis::turbo(n = 100, begin = 0.6, end = 1.0)   # Oranges to reds
          ))(length(unique(plot_data$group)))
        )
      }
    } +
    labs(
      title = paste0("z_", dimension_value, if(is_unused) " (unused)" else ""),  # Add 'unused' label if marked
      x = if(is_last_row) x_title else "",
      y = if(is_first_column) "Latent Response" else "",
      color = legend_title
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = plot_title_size, face = plot_title_face, hjust = plot_title_hjust, family = plot_font_family, color = plot_text_color),
      axis.title.x = element_text(size = axis_title_size, family = plot_font_family, color = plot_text_color),
      axis.title.y = element_text(size = axis_title_size, family = plot_font_family, color = plot_text_color),
      axis.text = element_text(size = axis_text_size, family = plot_font_family, color = plot_text_color),
      axis.text.x = element_text(
        size = axis_text_size, 
        family = plot_font_family,
        angle = 45,  # Add angle to x-axis tick labels
        hjust = 1,
        vjust = 1,
        margin = margin(t = 10, r = 0, b = 0, l = 0),  # Add top margin to push labels away from axis
        color = if(is_last_row) plot_text_color else "transparent"  # Only show ticks on last row
      ),
      axis.text.y = element_text(
        color = if(is_first_column) plot_text_color else "transparent" # Only show y-axis text for first column
      ),
      axis.ticks.x = element_line(color = if(is_last_row) "gray50" else "transparent"),
      axis.ticks.y = element_line(color = if(is_first_column) "gray50" else "transparent"),
      axis.ticks.length.x = unit(0.3, "cm"),  # Make ticks longer to create more space
      legend.position = if(show_legend) "right" else "none",  # Conditionally show legend
      panel.grid = element_line(color = grid_line_color),
      panel.background = element_rect(fill = plot_background_color, color = NA),
      plot.background = element_rect(fill = plot_background_color, color = NA),
      plot.margin = plot_margin_small
    )
  
  # Map original values to labels for x-axis
  varying_labels <- sapply(unique_varying, varying_mapper)
  varying_labels[is.na(varying_labels)] <- paste0(ifelse(varying_factor == "phoneme", "Phoneme ", "Speaker "), 
                                              unique_varying[is.na(varying_labels)])
  
  # Add custom x-axis with evenly spaced ticks
  p <- p + scale_x_continuous(
    breaks = x_positions,
    labels = varying_labels,
    expand = c(0.05, 0.05)
  )
  
  return(list(plot = p, data = plot_data))
}

# Function to create a 3D surface plot
create_3D_surface_plot <- function(data, latent_idx, emotion_idx, latent_type, is_unused = FALSE, show_legend = FALSE, 
                                  is_last_row = FALSE, is_first_column = FALSE) {
  # Filter data for the specific emotion
  filtered_data <- list(
    mu = data$mu[data$emotion == emotion_idx, ],
    phoneme = data$phoneme[data$emotion == emotion_idx],
    speaker = data$speaker[data$emotion == emotion_idx]
  )
  
  # Standardize latent response values
  all_values <- as.vector(filtered_data$mu)
  global_mean <- mean(all_values)
  global_sd <- sd(all_values)

  if (global_sd > 0) {
    filtered_data$mu <- (filtered_data$mu - global_mean) / global_sd
  }
  
  # Get unique phonemes and speakers
  unique_phonemes <- sort(unique(filtered_data$phoneme))
  unique_speakers <- sort(unique(filtered_data$speaker))
  
  # Randomly sample phonemes for better visualization
  if (length(unique_phonemes) > 20) {
    set.seed(42)  # For reproducibility
    selected_indices <- sample(length(unique_phonemes), 20)
    unique_phonemes <- unique_phonemes[selected_indices]
  }
  
  # Create evenly-spaced indices for visualization
  phoneme_positions <- seq_along(unique_phonemes)
  speaker_positions <- seq_along(unique_speakers)
  
  # Get the actual dimension value from var_dims for the plot title only
  dimension_value <- data$var_dims[latent_idx]
  
  # Initialize a matrix to store latent values
  z_matrix <- matrix(NA, nrow = length(unique_phonemes), ncol = length(unique_speakers))
  
  # Fill the matrix with latent values using even spacing
  for (i in seq_along(unique_phonemes)) {
    for (j in seq_along(unique_speakers)) {
      phoneme_val <- unique_phonemes[i]
      speaker_val <- unique_speakers[j]
      
      # Find indices where phoneme and speaker match
      indices <- which(filtered_data$phoneme == phoneme_val & filtered_data$speaker == speaker_val)
      
      if (length(indices) > 0) {
        # IMPORTANT: Use latent_idx directly for data access, not dimension_value
        z_matrix[i, j] <- mean(filtered_data$mu[indices, latent_idx], na.rm = TRUE)
      } else {
        # If no data for this combination, interpolate from neighbors
        nearby_indices <- which(abs(filtered_data$phoneme - phoneme_val) <= 2 & 
                               abs(filtered_data$speaker - speaker_val) <= 2)
        if (length(nearby_indices) > 0) {
          z_matrix[i, j] <- mean(filtered_data$mu[nearby_indices, latent_idx], na.rm = TRUE)
        }
      }
    }
  }
  
  # Fill any remaining NA values with interpolation
  for (i in seq_along(unique_phonemes)) {
    for (j in seq_along(unique_speakers)) {
      if (is.na(z_matrix[i, j])) {
        # Simple average of non-NA neighbors
        neighbors <- c()
        if (i > 1 && !is.na(z_matrix[i-1, j])) neighbors <- c(neighbors, z_matrix[i-1, j])
        if (i < length(unique_phonemes) && !is.na(z_matrix[i+1, j])) neighbors <- c(neighbors, z_matrix[i+1, j])
        if (j > 1 && !is.na(z_matrix[i, j-1])) neighbors <- c(neighbors, z_matrix[i, j-1])
        if (j < length(unique_speakers) && !is.na(z_matrix[i, j+1])) neighbors <- c(neighbors, z_matrix[i, j+1])
        
        if (length(neighbors) > 0) {
          z_matrix[i, j] <- mean(neighbors)
        } else {
          z_matrix[i, j] <- 0  # Default value if no neighbors
        }
      }
    }
  }
  
  # Apply consistent y-axis limits to 3D plot
  y_padding <- 0.2
  z_limits <- c(-3 - y_padding, 3 + y_padding)
  
  # Ensure z_matrix values stay within limits (after standardization)
  z_matrix[z_matrix < z_limits[1]] <- z_limits[1]
  z_matrix[z_matrix > z_limits[2]] <- z_limits[2]
  
  # Map phonemes and speakers to readable labels
  phoneme_mapper <- get_factor_mapping("phoneme", dataset)
  speaker_mapper <- get_factor_mapping("speaker", dataset)
  emotion_mapper <- get_factor_mapping("emotion", dataset)
  
  phoneme_labels <- sapply(unique_phonemes, phoneme_mapper)
  speaker_labels <- sapply(unique_speakers, speaker_mapper)
  emotion_label <- emotion_mapper(emotion_idx)
  
  # Create the 3D surface plot with improved zoomed-out view and consistent styling
  p <- plot_ly(
    x = phoneme_positions, 
    y = speaker_positions, 
    z = z_matrix,
    type = "surface",
    # Update colorscale to skip yellow portion (0.4-0.6)
    colorscale = list(
      c(0.00, "rgb(23,27,230)"),  # Blue
      c(0.15, "rgb(0,152,223)"),  # Cyan
      c(0.30, "rgb(0,206,137)"),  # Green
      # Skip yellow section (would be around 0.4-0.6)
      c(0.70, "rgb(238,91,13)"),  # Orange
      c(0.85, "rgb(220,50,32)"),  # Light red
      c(1.00, "rgb(197,0,11)")    # Dark red
    ),
    showscale = show_legend, # Only show colorscale when legend is requested
    opacity = surface_opacity
  ) %>%
    layout(
      title = list(
        text = paste0("z_", dimension_value, if(is_unused) " (unused)" else ""),
        font = list(
          family = plot_font_family,
          size = plot_title_size,
          color = plot_text_color
        )
      ),
      scene = list(
        xaxis = list(
          title = list(
            text = "Phoneme",
            font = list(
              family = plot_font_family,
              size = axis_title_size,
              color = plot_text_color
            ),
            standoff = 25  # Increased standoff to push title further away
          ),
          tickvals = phoneme_positions,
          ticktext = phoneme_labels,
          tickfont = list(
            family = plot_font_family, 
            size = axis_text_size_small,
            color = plot_text_color
          ),
          tickangle = 0,  # Keep labels horizontal
          ticklen = 8,     # Make ticks longer
          tickwidth = 2,   # Make ticks thicker
          showticklabels = TRUE,
          showline = TRUE,
          linecolor = 'black',
          showgrid = TRUE,
          standoff = 15    # Push labels away from the axis
        ),
        yaxis = list(
          title = list(
            text = "Speaker",
            font = list(
              family = plot_font_family,
              size = axis_title_size,
              color = plot_text_color
            ),
            standoff = 25  # Increased standoff to push title further away
          ),
          tickvals = speaker_positions,
          ticktext = speaker_labels,
          tickfont = list(
            family = plot_font_family, 
            size = axis_text_size_small,
            color = plot_text_color
          ),
          tickangle = 0,   # Keep labels horizontal
          ticklen = 8,      # Make ticks longer
          tickwidth = 2,    # Make ticks thicker
          showticklabels = TRUE,
          showline = TRUE,
          linecolor = 'black',
          showgrid = TRUE,
          standoff = 15     # Push labels away from the axis
        ),
        zaxis = list(
          title = list(
            text = "Latent Response",
            font = list(
              family = plot_font_family,
              size = axis_title_size,
              color = plot_text_color
            ),
            standoff = 35  # Increased from 15 to move title further away
          ),
          range = z_limits,
          tickfont = list(
            family = plot_font_family, 
            size = axis_text_size_small,
            color = plot_text_color
          )
        ),
        aspectratio = list(x = aspect_ratio_x, y = aspect_ratio_y, z = aspect_ratio_z),
        camera = list(
          eye = list(x = camera_eye_x, y = camera_eye_y, z = camera_eye_z + 0.2),  # Increase z slightly to view from higher angle
          center = list(x = 0, y = 0, z = -0.3),
          up = list(x = 0, y = 0, z = 1)
        ),
        bgcolor = plot_background_color
      ),
      paper_bgcolor = plot_background_color,
      margin = list(l = 0, r = 0, b = 0, t = 50),
      font = list(
        family = plot_font_family,
        color = plot_text_color
      )
    )
  
  # Create heatmap data for heatmap visualization
  heatmap_data <- expand.grid(
    Phoneme = seq_along(unique_phonemes),
    Speaker = seq_along(unique_speakers)
  )
  heatmap_data$Value <- as.vector(z_matrix)
  
  # Create mapping dictionaries for axis labels
  phoneme_tick_positions <- seq_along(unique_phonemes)
  speaker_tick_positions <- seq_along(unique_speakers)
  phoneme_tick_labels <- sapply(unique_phonemes, phoneme_mapper)
  speaker_tick_labels <- sapply(unique_speakers, speaker_mapper)
  
  # Also create a heatmap visualization with consistent styling
  # Create a custom color vector that skips yellow (0.4-0.6)
  custom_colors <- c(
    viridis::turbo(n = 100, begin = 0.0, end = 0.4),  # Blues to greens
    viridis::turbo(n = 100, begin = 0.6, end = 1.0)   # Oranges to reds
  )
  
  heatmap_plot <- ggplot(heatmap_data, aes(x = Phoneme, y = Speaker, fill = Value)) +
    geom_tile() +
    # Use the custom color scale for the heatmap
    scale_fill_gradientn(
      colors = custom_colors,
      limits = z_limits,
      name = "Latent Response"
    ) +
    # Add proper x and y axis scales with actual labels
    scale_x_continuous(
      breaks = phoneme_tick_positions,
      labels = phoneme_tick_labels
    ) +
    scale_y_continuous(
      breaks = speaker_tick_positions,
      labels = speaker_tick_labels
    ) +
    labs(
      title = paste0("z_", dimension_value, if(is_unused) " (unused)" else ""),
      x = if(is_last_row) "Phoneme" else "",
      y = if(is_first_column) "Speaker" else ""
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = plot_title_size, hjust = 0.5, family = plot_font_family, color = plot_text_color),
      axis.title = element_text(size = axis_title_size, family = plot_font_family, color = plot_text_color),
      axis.text = element_text(size = axis_text_size, family = plot_font_family, color = plot_text_color),
      axis.text.x = element_text(
        size = axis_text_size, 
        family = plot_font_family,
        angle = 45,  # Add angle to x-axis tick labels
        hjust = 1,
        vjust = 1,
        margin = margin(t = 10, r = 0, b = 0, l = 0),  # Add top margin to push labels away
        color = if(is_last_row) plot_text_color else "transparent"  # Only show ticks on last row
      ),
      axis.text.y = element_text(
        margin = margin(t = 0, r = 10, b = 0, l = 0),   # Add right margin to push labels away
        color = if(is_first_column) plot_text_color else "transparent" # Only show y-axis text for first column
      ),
      axis.ticks.x = element_line(color = if(is_last_row) "gray50" else "transparent"),
      axis.ticks.y = element_line(color = if(is_first_column) "gray50" else "transparent"),
      axis.ticks.length = unit(0.3, "cm"),  # Make ticks longer to create more space
      legend.position = "none",  # Fixed: use "none" directly 
      panel.grid = element_line(color = grid_line_color),
      panel.background = element_rect(fill = plot_background_color, color = NA),
      plot.background = element_rect(fill = plot_background_color, color = NA),
      plot.margin = plot_margin_small
    )
  
  return(list(plotly = p, heatmap = heatmap_plot, heatmap_data = heatmap_data, z_limits = z_limits, custom_colors = custom_colors, 
              phoneme_tick_positions = phoneme_tick_positions, speaker_tick_positions = speaker_tick_positions,
              phoneme_tick_labels = phoneme_tick_labels, speaker_tick_labels = speaker_tick_labels))
}

# Function to create a combined plot with all emotions and latent types
create_emotion_latent_comparison_plot <- function(all_snapshots, save_dir) {
  # Check if we have data to plot
  if (length(all_snapshots) == 0) {
    message("No snapshot data available for comparison plot")
    return(NULL)
  }
  
  # Get available latent types and emotions
  available_latent_types <- names(all_snapshots)
  
  # Initialize a list to store the plots in the right arrangement
  comparison_grid <- list()
  
  # For each emotion (will be rows)
  for (emotion_name in available_emotions) {
    emotion_plots <- list()
    
    # For each latent type (will be columns)
    for (latent_type_idx in seq_along(available_latent_types)) {
      latent_type <- available_latent_types[latent_type_idx]
      # Check if this is the first column
      is_first_column <- (latent_type_idx == 1)
      
      # Skip if this emotion doesn't have data for this latent type
      if (!emotion_name %in% names(all_snapshots[[latent_type]])) {
        # Create empty plot as placeholder
        empty_plot <- ggplot() + 
          theme_void() + 
          labs(title = paste0(latent_type, " - No Data"))
        
        emotion_plots[[latent_type]] <- empty_plot
        next
      }
      
      # Get the first latent dimension plot for this emotion/latent combination
      first_dim_plot <- all_snapshots[[latent_type]][[emotion_name]][[1]]
      
      # Skip if the plot is NULL or invalid
      if (is.null(first_dim_plot) || !inherits(first_dim_plot, "ggplot")) {
        empty_plot <- ggplot() + 
          theme_void() + 
          labs(title = paste0(latent_type, " - Invalid Data"))
        
        emotion_plots[[latent_type]] <- empty_plot
        next
      }
      
      # Use the plot as is but add a more informative title
      modified_plot <- first_dim_plot +
        labs(title = latent_type) +
        theme(
          plot.title = element_text(
            size = plot_title_size,
            hjust = 0.5,
            family = plot_font_family,
            color = plot_text_color
          ),
          # Hide y-axis text and title for all columns except the first
          axis.text.y = element_text(
            color = if(is_first_column) plot_text_color else "transparent",
            size = axis_text_size,
            family = plot_font_family
          ),
          axis.title.y = element_text(
            color = if(is_first_column) plot_text_color else "transparent",
            size = axis_title_size,
            family = plot_font_family
          )
        )
      
      # Store the plot
      emotion_plots[[latent_type]] <- modified_plot
    }
    
    # Add row title (emotion name) to the first plot in the row
    if (length(emotion_plots) > 0) {
      first_latent <- names(emotion_plots)[1]
      emotion_plots[[first_latent]] <- emotion_plots[[first_latent]] +
        labs(subtitle = emotion_display_names[emotion_name]) +
        theme(
          plot.subtitle = element_text(
            size = plot_title_size * 0.8,
            hjust = 0,
            family = plot_font_family,
            color = plot_text_color,
            margin = margin(b = 10)
          )
        )
    }
    
    # Store the row of plots
    comparison_grid[[emotion_name]] <- emotion_plots
  }
  
  # Convert the nested list to a flat list for patchwork
  plot_grid <- list()
  for (emotion_name in names(comparison_grid)) {
    # Get plots for this emotion and convert to a row
    emotion_row <- wrap_plots(comparison_grid[[emotion_name]], ncol = length(available_latent_types))
    plot_grid[[emotion_name]] <- emotion_row
  }
  
  # Combine all rows into a single plot
  combined_plot <- wrap_plots(plot_grid, ncol = 1) +
    plot_annotation(
      title = "",
      subtitle = "",
      theme = theme(
        plot.title = element_text(
          size = plot_sup_title_size,
          hjust = 0.5,
          family = plot_font_family,
          color = plot_text_color,
          face = plot_title_face,
          margin = margin(b = 20)
        ),
        plot.subtitle = element_text(
          size = plot_title_size,
          hjust = 0.5,
          family = plot_font_family,
          color = plot_text_color,
          margin = margin(b = 20)
        ),
        plot.background = element_rect(fill = plot_background_color, color = NA)
      )
    )
  
  # Save the combined plot
  save_path <- file.path(save_dir, "emotion_latent_comparison_first_dimension.png")
  tryCatch({
    ggsave(
      save_path, 
      combined_plot, 
      width = 7 * length(available_latent_types), 
      height = 5 * length(available_emotions), 
      dpi = 600, 
      bg = plot_background_color
    )
    message("Successfully saved emotion-latent comparison plot")
  }, error = function(e) {
    message(paste("Error saving emotion-latent comparison plot:", e$message))
  })
  
  return(combined_plot)
}

# Function to create a combined plot with heatmaps across emotions and latent types
create_emotion_latent_heatmap_comparison_plot <- function(all_heatmaps, save_dir) {
  # Check if we have data to plot
  if (length(all_heatmaps) == 0) {
    message("No heatmap data available for comparison plot")
    return(NULL)
  }
  
  # Get available latent types and emotions
  available_latent_types <- names(all_heatmaps)
  
  # Initialize a list to store the plots in the right arrangement
  comparison_grid <- list()
  
  # For each emotion (will be rows)
  for (emotion_idx in seq_along(available_emotions)) {
    emotion_name <- available_emotions[emotion_idx]
    emotion_plots <- list()
    
    # Check if this is the last row
    is_last_row <- (emotion_idx == length(available_emotions))
    
    # For each latent type (will be columns)
    for (latent_type_idx in seq_along(available_latent_types)) {
      latent_type <- available_latent_types[latent_type_idx]
      # Check if this is the first column
      is_first_column <- (latent_type_idx == 1)
      
      # Skip if this emotion doesn't have data for this latent type
      if (!emotion_name %in% names(all_heatmaps[[latent_type]])) {
        # Create empty plot as placeholder
        empty_plot <- ggplot() + 
          theme_void() + 
          labs(title = paste0(latent_type, " - No Data"))
        
        emotion_plots[[latent_type]] <- empty_plot
        next
      }
      
      # Get the first latent dimension plot for this emotion/latent combination
      first_dim_plot <- all_heatmaps[[latent_type]][[emotion_name]][[1]]
      
      # Skip if the plot is NULL or invalid
      if (is.null(first_dim_plot) || !inherits(first_dim_plot, "ggplot")) {
        empty_plot <- ggplot() + 
          theme_void() + 
          labs(title = paste0(latent_type, " - Invalid Data"))
        
        emotion_plots[[latent_type]] <- empty_plot
        next
      }
      
      # Use the plot as is but add a more informative title only for the first row
      # And ensure axis text/titles are shown for the last row and first column
      modified_plot <- first_dim_plot +
        labs(title = if(emotion_idx == 1) latent_type else "") +
        theme(
          plot.title = element_text(
            size = plot_title_size,
            hjust = 0.5,
            family = plot_font_family,
            color = plot_text_color
          ),
          axis.text.x = element_text(
            color = if(is_last_row) plot_text_color else "transparent",
            size = axis_text_size, 
            family = plot_font_family,
            angle = 45,
            hjust = 1
          ),
          axis.title.x = element_text(
            color = if(is_last_row) plot_text_color else "transparent",
            size = axis_title_size,
            family = plot_font_family
          ),
          axis.text.y = element_text(
            color = if(is_first_column) plot_text_color else "transparent",
            size = axis_text_size, 
            family = plot_font_family
          ),
          axis.title.y = element_text(
            color = if(is_first_column) plot_text_color else "transparent",
            size = axis_title_size,
            family = plot_font_family
          )
        )
      
      # Store the plot
      emotion_plots[[latent_type]] <- modified_plot
    }
    
    # Add row title (emotion name) to the first plot in the row
    if (length(emotion_plots) > 0) {
      first_latent <- names(emotion_plots)[1]
      emotion_plots[[first_latent]] <- emotion_plots[[first_latent]] +
        labs(subtitle = emotion_display_names[emotion_name]) +
        theme(
          plot.subtitle = element_text(
            size = plot_title_size * 0.8,
            hjust = 0,
            family = plot_font_family,
            color = plot_text_color,
            margin = margin(b = 10)
          )
        )
    }
    
    # Store the row of plots
    comparison_grid[[emotion_name]] <- emotion_plots
  }
  
  # Convert the nested list to a flat list for patchwork
  plot_grid <- list()
  for (emotion_name in names(comparison_grid)) {
    # Get plots for this emotion and convert to a row
    emotion_row <- wrap_plots(comparison_grid[[emotion_name]], ncol = length(available_latent_types))
    plot_grid[[emotion_name]] <- emotion_row
  }
  
  # Combine all rows into a single plot
  combined_plot <- wrap_plots(plot_grid, ncol = 1) +
    plot_annotation(
      title = "",
      subtitle = "",
      theme = theme(
        plot.title = element_text(
          size = plot_sup_title_size,
          hjust = 0.5,
          family = plot_font_family,
          color = plot_text_color,
          face = plot_title_face,
          margin = margin(b = 20)
        ),
        plot.subtitle = element_text(
          size = plot_title_size,
          hjust = 0.5,
          family = plot_font_family,
          color = plot_text_color,
          margin = margin(b = 20)
        ),
        plot.background = element_rect(fill = plot_background_color, color = NA)
      )
    )
  
  # Save the combined plot
  save_path <- file.path(save_dir, "emotion_latent_comparison_heatmaps.png")
  tryCatch({
    ggsave(
      save_path, 
      combined_plot, 
      width = 7 * length(available_latent_types), 
      height = 5 * length(available_emotions), 
      dpi = 300, 
      bg = plot_background_color
    )
    message("Successfully saved emotion-latent heatmap comparison plot")
  }, error = function(e) {
    message(paste("Error saving emotion-latent heatmap comparison plot:", e$message))
  })
  
  return(combined_plot)
}

# Function to create a combined plot with phoneme line plots across emotions and latent types
create_emotion_latent_phoneme_comparison_plot <- function(all_latent_plots_phoneme, save_dir) {
  # Check if we have data to plot
  if (length(all_latent_plots_phoneme) == 0) {
    message("No phoneme plot data available for comparison plot")
    return(NULL)
  }
  
  # Get available latent types and emotions
  available_latent_types <- names(all_latent_plots_phoneme)
  
  # Initialize a list to store the plots in the right arrangement
  comparison_grid <- list()
  
  # For each emotion (will be rows)
  for (emotion_idx in seq_along(available_emotions)) {
    emotion_name <- available_emotions[emotion_idx]
    emotion_plots <- list()
    
    # Check if this is the last row
    is_last_row <- (emotion_idx == length(available_emotions))
    
    # For each latent type (will be columns)
    for (latent_type_idx in seq_along(available_latent_types)) {
      latent_type <- available_latent_types[latent_type_idx]
      # Check if this is the first column
      is_first_column <- (latent_type_idx == 1)
      
      # Skip if this emotion doesn't have data for this latent type
      if (!emotion_name %in% names(all_latent_plots_phoneme[[latent_type]])) {
        # Create empty plot as placeholder
        empty_plot <- ggplot() + 
          theme_void() + 
          labs(title = paste0(latent_type, " - No Data"))
        
        emotion_plots[[latent_type]] <- empty_plot
        next
      }
      
      # Get the first latent dimension plot for this emotion/latent combination
      first_dim_plot <- all_latent_plots_phoneme[[latent_type]][[emotion_name]][[1]]
      
      # Skip if the plot is NULL or invalid
      if (is.null(first_dim_plot) || !inherits(first_dim_plot, "ggplot")) {
        empty_plot <- ggplot() + 
          theme_void() + 
          labs(title = paste0(latent_type, " - Invalid Data"))
        
        emotion_plots[[latent_type]] <- empty_plot
        next
      }
      
      # Use the plot as is but add a more informative title only for the first row
      # And ensure axis text/titles are shown for the last row and first column
      modified_plot <- first_dim_plot +
        labs(title = if(emotion_idx == 1) latent_type else "") +
        theme(
          plot.title = element_text(
            size = plot_title_size,
            hjust = 0.5,
            family = plot_font_family,
            color = plot_text_color
          ),
          axis.text.x = element_text(
            color = if(is_last_row) plot_text_color else "transparent",
            size = axis_text_size, 
            family = plot_font_family,
            angle = 45,
            hjust = 1
          ),
          axis.title.x = element_text(
            color = if(is_last_row) plot_text_color else "transparent",
            size = axis_title_size,
            family = plot_font_family
          ),
          axis.text.y = element_text(
            color = if(is_first_column) plot_text_color else "transparent",
            size = axis_text_size, 
            family = plot_font_family
          ),
          axis.title.y = element_text(
            color = if(is_first_column) plot_text_color else "transparent",
            size = axis_title_size,
            family = plot_font_family
          )
        )
      
      # Store the plot
      emotion_plots[[latent_type]] <- modified_plot
    }
    
    # Add row title (emotion name) to the first plot in the row
    if (length(emotion_plots) > 0) {
      first_latent <- names(emotion_plots)[1]
      emotion_plots[[first_latent]] <- emotion_plots[[first_latent]] +
        labs(subtitle = emotion_display_names[emotion_name]) +
        theme(
          plot.subtitle = element_text(
            size = plot_title_size * 0.8,
            hjust = 0,
            family = plot_font_family,
            color = plot_text_color,
            margin = margin(b = 10)
          )
        )
    }
    
    # Store the row of plots
    comparison_grid[[emotion_name]] <- emotion_plots
  }
  
  # Convert the nested list to a flat list for patchwork
  plot_grid <- list()
  for (emotion_name in names(comparison_grid)) {
    # Get plots for this emotion and convert to a row
    emotion_row <- wrap_plots(comparison_grid[[emotion_name]], ncol = length(available_latent_types))
    plot_grid[[emotion_name]] <- emotion_row
  }
  
  # Combine all rows into a single plot
  combined_plot <- wrap_plots(plot_grid, ncol = 1) +
    plot_annotation(
      title = "",
      subtitle = "",
      theme = theme(
        plot.title = element_text(
          size = plot_sup_title_size,
          hjust = 0.5,
          family = plot_font_family,
          color = plot_text_color,
          face = plot_title_face,
          margin = margin(b = 20)
        ),
        plot.subtitle = element_text(
          size = plot_title_size,
          hjust = 0.5,
          family = plot_font_family,
          color = plot_text_color,
          margin = margin(b = 20)
        ),
        plot.background = element_rect(fill = plot_background_color, color = NA)
      )
    )
  
  # Save the combined plot
  save_path <- file.path(save_dir, "emotion_latent_comparison_phoneme_lines.png")
  tryCatch({
    ggsave(
      save_path, 
      combined_plot, 
      width = 7 * length(available_latent_types), 
      height = 5 * length(available_emotions), 
      dpi = 300, 
      bg = plot_background_color
    )
    message("Successfully saved emotion-latent phoneme line comparison plot")
  }, error = function(e) {
    message(paste("Error saving emotion-latent phoneme line comparison plot:", e$message))
  })
  
  return(combined_plot)
}

# Function to create a combined plot with speaker line plots across emotions and latent types
create_emotion_latent_speaker_comparison_plot <- function(all_latent_plots_speaker, save_dir) {
  # Check if we have data to plot
  if (length(all_latent_plots_speaker) == 0) {
    message("No speaker plot data available for comparison plot")
    return(NULL)
  }
  
  # Get available latent types and emotions
  available_latent_types <- names(all_latent_plots_speaker)
  
  # Initialize a list to store the plots in the right arrangement
  comparison_grid <- list()
  
  # For each emotion (will be rows)
  for (emotion_idx in seq_along(available_emotions)) {
    emotion_name <- available_emotions[emotion_idx]
    emotion_plots <- list()
    
    # Check if this is the last row
    is_last_row <- (emotion_idx == length(available_emotions))
    
    # For each latent type (will be columns)
    for (latent_type_idx in seq_along(available_latent_types)) {
      latent_type <- available_latent_types[latent_type_idx]
      # Check if this is the first column
      is_first_column <- (latent_type_idx == 1)
      
      # Skip if this emotion doesn't have data for this latent type
      if (!emotion_name %in% names(all_latent_plots_speaker[[latent_type]])) {
        # Create empty plot as placeholder
        empty_plot <- ggplot() + 
          theme_void() + 
          labs(title = paste0(latent_type, " - No Data"))
        
        emotion_plots[[latent_type]] <- empty_plot
        next
      }
      
      # Get the first latent dimension plot for this emotion/latent combination
      first_dim_plot <- all_latent_plots_speaker[[latent_type]][[emotion_name]][[1]]
      
      # Skip if the plot is NULL or invalid
      if (is.null(first_dim_plot) || !inherits(first_dim_plot, "ggplot")) {
        empty_plot <- ggplot() + 
          theme_void() + 
          labs(title = paste0(latent_type, " - Invalid Data"))
        
        emotion_plots[[latent_type]] <- empty_plot
        next
      }
      
      # Use the plot as is but add a more informative title only for the first row
      # And ensure axis text/titles are shown for the last row and first column
      modified_plot <- first_dim_plot +
        labs(title = if(emotion_idx == 1) latent_type else "") +
        theme(
          plot.title = element_text(
            size = plot_title_size,
            hjust = 0.5,
            family = plot_font_family,
            color = plot_text_color
          ),
          axis.text.x = element_text(
            color = if(is_last_row) plot_text_color else "transparent",
            size = axis_text_size, 
            family = plot_font_family,
            angle = 45,
            hjust = 1
          ),
          axis.title.x = element_text(
            color = if(is_last_row) plot_text_color else "transparent",
            size = axis_title_size,
            family = plot_font_family
          ),
          axis.text.y = element_text(
            color = if(is_first_column) plot_text_color else "transparent",
            size = axis_text_size, 
            family = plot_font_family
          ),
          axis.title.y = element_text(
            color = if(is_first_column) plot_text_color else "transparent",
            size = axis_title_size,
            family = plot_font_family
          )
        )
      
      # Store the plot
      emotion_plots[[latent_type]] <- modified_plot
    }
    
    # Add row title (emotion name) to the first plot in the row
    if (length(emotion_plots) > 0) {
      first_latent <- names(emotion_plots)[1]
      emotion_plots[[first_latent]] <- emotion_plots[[first_latent]] +
        labs(subtitle = emotion_display_names[emotion_name]) +
        theme(
          plot.subtitle = element_text(
            size = plot_title_size * 0.8,
            hjust = 0,
            family = plot_font_family,
            color = plot_text_color,
            margin = margin(b = 10)
          )
        )
    }
    
    # Store the row of plots
    comparison_grid[[emotion_name]] <- emotion_plots
  }
  
  # Convert the nested list to a flat list for patchwork
  plot_grid <- list()
  for (emotion_name in names(comparison_grid)) {
    # Get plots for this emotion and convert to a row
    emotion_row <- wrap_plots(comparison_grid[[emotion_name]], ncol = length(available_latent_types))
    plot_grid[[emotion_name]] <- emotion_row
  }
  
  # Combine all rows into a single plot
  combined_plot <- wrap_plots(plot_grid, ncol = 1) +
    plot_annotation(
      title = "",
      subtitle = "",
      theme = theme(
        plot.title = element_text(
          size = plot_sup_title_size,
          hjust = 0.5,
          family = plot_font_family,
          color = plot_text_color,
          face = plot_title_face,
          margin = margin(b = 20)
        ),
        plot.subtitle = element_text(
          size = plot_title_size,
          hjust = 0.5,
          family = plot_font_family,
          color = plot_text_color,
          margin = margin(b = 20)
        ),
        plot.background = element_rect(fill = plot_background_color, color = NA)
      )
    )
  
  # Save the combined plot
  save_path <- file.path(save_dir, "emotion_latent_comparison_speaker_lines.png")
  tryCatch({
    ggsave(
      save_path, 
      combined_plot, 
      width = 7 * length(available_latent_types), 
      height = 5 * length(available_emotions), 
      dpi = 300, 
      bg = plot_background_color
    )
    message("Successfully saved emotion-latent speaker line comparison plot")
  }, error = function(e) {
    message(paste("Error saving emotion-latent speaker line comparison plot:", e$message))
  })
  
  return(combined_plot)
}

# Main function to analyze latent responses for iemocap
analyze_iemocap_latent_responses <- function(ckp_dir, save_dir, available_emotions, latent_types = c('X', 'OC1', 'OC2', 'OC3', 'OC4')) {
  # Check which latent types exist for at least one emotion
  available_latents <- latent_types[sapply(latent_types, function(lt) {
    any(sapply(available_emotions, function(e) check_latent_file(ckp_dir, lt, e)))
  })]
  
  if (length(available_latents) == 0) {
    stop("No latent files found for any emotion!")
  }
  
  # Create separate directories for 3D and 2D plots
  plots_3d_dir <- file.path(save_dir, "3D_plots")
  
  if (!dir.exists(plots_3d_dir)) {
    dir.create(plots_3d_dir, recursive = TRUE, showWarnings = FALSE)
  }

  # Create a legends directory
  legends_dir <- file.path(save_dir, "legends")
  if (!dir.exists(legends_dir)) {
    dir.create(legends_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  # Initialize collection variables at the function level
  all_latent_plots_phoneme <- list()
  all_latent_plots_speaker <- list()
  all_heatmaps <- list()
  all_3d_snapshots <- list()
  
  # Process each available latent type
  for (latent_type in available_latents) {
    # Load data from all emotions
    all_data <- load_all_emotions_data(ckp_dir, latent_type, available_emotions)
    
    # Skip if no data loaded
    if (length(all_data) == 0) {
      message(paste("No data found for latent type:", latent_type))
      next
    }
    
    # Combine data from all emotions
    combined_data <- combine_emotion_data(all_data)
    
    # Get important latent dimensions
    var_dims <- combined_data$var_dims
    min_var_latents <- combined_data$min_var_latents
    
    # Log dimensions for debugging
    message(paste("Latent type:", latent_type, 
                 "- Dimensions:", paste(var_dims, collapse=", "),
                 "- Min var latents:", paste(min_var_latents, collapse=", ")))
    
    # Select a subset of latent dimensions to plot using LIST INDICES (not actual dimension values)
    # These indices will be used to index into var_dims to get the actual dimension values
    num_dims <- length(var_dims)
    if (num_dims > 6) {
      # Take first 4 and last 2 indices (not dimensions)
      selected_latent_ids <- c(1:4, (num_dims-1):num_dims)
      # Mark the last two indices as 'unused'
      is_unused <- rep(FALSE, length(selected_latent_ids))
      is_unused[(length(selected_latent_ids)-1):length(selected_latent_ids)] <- TRUE
    } else {
      # Use all indices if 6 or fewer dimensions
      selected_latent_ids <- 1:num_dims
      is_unused <- rep(FALSE, length(selected_latent_ids))
    }
    
    # Log selected indices and their corresponding dimension values
    message(paste("Selected latent indices for", latent_type, ":", paste(selected_latent_ids, collapse=", ")))
    message(paste("Which correspond to dimensions:", paste(var_dims[selected_latent_ids], collapse=", ")))
    
    # Initialize the lists for this latent type
    latent_plots_by_emotion_phoneme <- list()
    latent_plots_by_emotion_speaker <- list()
    heatmaps_by_emotion <- list()
    snapshots_by_emotion <- list()
    
    # Create horizontal stacked plots for each emotion
    for (emotion_idx in 0:(length(available_emotions)-1)) {
      emotion_name <- available_emotions[emotion_idx + 1]
      
      # Check if we have sufficient data for this emotion
      unique_phonemes <- unique(combined_data$phoneme[combined_data$emotion == emotion_idx])
      unique_speakers <- unique(combined_data$speaker[combined_data$emotion == emotion_idx])
      
      if (length(unique_phonemes) < 3 || length(unique_speakers) < 3) {
        message(paste("Skipping emotion", emotion_name, "for latent type", latent_type, "- insufficient unique factors"))
        next
      }
      
      # Extract and save legends once per emotion/latent type combination
      if (length(unique_phonemes) >= 3 && length(unique_speakers) >= 3) {
        # Save separate 2D legends for speakers and phonemes
        extract_2D_speaker_legend(combined_data, latent_type, emotion_name, legends_dir)
        extract_2D_phoneme_legend(combined_data, latent_type, emotion_name, legends_dir)
        
        # Save 3D heatmap legend (now vertical)
        extract_3D_heatmap_legend(combined_data, latent_type, emotion_name, legends_dir)
      }
      
      # Create 2D line plots without legends
      phoneme_plots <- list()
      for (i in seq_along(selected_latent_ids)) {
        latent_idx <- selected_latent_ids[i]
        is_first_column <- (i == 1)
        
        # We'll still calculate this, but it will be overridden in the create_combined_plot_by_emotion function
        is_last_row_in_grid <- (i > length(selected_latent_ids) - (length(selected_latent_ids) %% 6 || 6))
        
        # Pass the is_unused flag to the plotting function, with show_legend=FALSE
        result <- create_2D_line_plots(combined_data, latent_idx, emotion_idx, latent_type, "phoneme", 
                                     max_items = 15, is_last_row = is_last_row_in_grid, 
                                     is_first_column = is_first_column, is_unused = is_unused[i], show_legend = FALSE)
        phoneme_plots[[i]] <- result$plot + 
                          theme(
                            plot.margin = margin(5, 5, 5, 5)
                          )
      }
      
      # Create 2D line plots for varying speaker without legends
      speaker_plots <- list()
      for (i in seq_along(selected_latent_ids)) {
        latent_idx <- selected_latent_ids[i]
        is_first_column <- (i == 1)
        is_last_row_in_grid <- (i > length(selected_latent_ids) - (length(selected_latent_ids) %% 6 || 6))
        result <- create_2D_line_plots(combined_data, latent_idx, emotion_idx, latent_type, "speaker", 
                                    max_items = 15, is_last_row = is_last_row_in_grid, 
                                    is_first_column = is_first_column, is_unused = is_unused[i], show_legend = FALSE)
        speaker_plots[[i]] <- result$plot + 
                          theme(
                            plot.margin = margin(5, 5, 5, 5)
                          )
      }
      
      # Create 3D plots (both heatmaps and snapshots) without legends
      heatmap_plots <- list()
      snapshot_plots <- list()
      
      for (i in seq_along(selected_latent_ids)) {
        latent_idx <- selected_latent_ids[i]
        dimension_value <- combined_data$var_dims[latent_idx]
        is_first_column <- (i == 1)
        is_last_row_in_grid <- (i > length(selected_latent_ids) - (length(selected_latent_ids) %% 6 || 6))
        
        # Create 3D surface plot with both plotly and heatmap versions - pass is_unused flag
        plots_3d <- create_3D_surface_plot(combined_data, latent_idx, emotion_idx, latent_type, 
                                         is_unused = is_unused[i], show_legend = FALSE,
                                         is_last_row = is_last_row_in_grid, 
                                         is_first_column = is_first_column)
        
        # Save interactive 3D plot for reference
        html_file <- file.path(plots_3d_dir, paste0(latent_type, "_z", latent_idx, "_emotion_", emotion_name, "_3D.html"))
        
        # Try to save the plotly plot
        tryCatch({
          htmlwidgets::saveWidget(plots_3d$plotly, html_file, selfcontained = FALSE)
          
          # Create a static image of the 3D plot using webshot2 if available
          snapshot_file <- file.path(plots_3d_dir, paste0(latent_type, "_z", latent_idx, "_emotion_", emotion_name, "_3D_snapshot.png"))
          
          if (requireNamespace("webshot2", quietly = TRUE)) {
            webshot2::webshot(html_file, snapshot_file, 
                            delay = 2.0,  # Longer delay to ensure full rendering
                            zoom = zoom_level,   # Zoom level
                            vwidth = 1200, 
                            vheight = 800)
            
            # Read the snapshot and convert to a ggplot object for the combined plot
            if (file.exists(snapshot_file) && requireNamespace("png", quietly = TRUE) && requireNamespace("grid", quietly = TRUE)) {
              img <- png::readPNG(snapshot_file)
              snapshot_plot <- tryCatch({
                p <- ggplot() + 
                  annotation_custom(grid::rasterGrob(img, interpolate = TRUE), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf) +
                  labs(title = paste0("z_", dimension_value, if(is_unused[i]) " (unused)" else "")) +
                  theme_void() +
                  theme(
                    plot.title = element_text(size = plot_title_size, hjust = 0.5, family = plot_font_family)
                  )
                p # Return the plot object
              }, error = function(e) {
                message(paste("Error creating snapshot plot:", e$message))
                return(NULL)
              })
              
              # Only add to collection if valid
              if (!is.null(snapshot_plot) && inherits(snapshot_plot, "ggplot")) {
                snapshot_plots[[i]] <- snapshot_plot
              }
            }
          }
        }, error = function(e) {
          message("Error saving 3D plot: ", e$message)
        })
        
        # Add heatmap to collection - ensure it's a valid ggplot object
        heatmap_plot <- tryCatch({
          # Use the data from plots_3d that was returned from create_3D_surface_plot
          ggplot(plots_3d$heatmap_data, aes(x = Phoneme, y = Speaker, fill = Value)) +
            geom_tile() +
            scale_fill_gradientn(
              colors = plots_3d$custom_colors,
              limits = plots_3d$z_limits,
              name = "Latent Response"
            ) +
            # Add proper x and y axis scales with actual labels
            scale_x_continuous(
              breaks = plots_3d$phoneme_tick_positions,
              labels = plots_3d$phoneme_tick_labels
            ) +
            scale_y_continuous(
              breaks = plots_3d$speaker_tick_positions,
              labels = plots_3d$speaker_tick_labels
            ) +
            labs(
              title = paste0("z_", dimension_value, if(is_unused[i]) " (unused)" else ""),
              x = if(is_last_row_in_grid) "Phoneme" else "",
              y = if(is_first_column) "Speaker" else ""
            ) +
            theme_minimal() +
            theme(
              plot.title = element_text(size = plot_title_size, hjust = 0.5, family = plot_font_family, color = plot_text_color),
              axis.title = element_text(size = axis_title_size, family = plot_font_family, color = plot_text_color),
              axis.text = element_text(size = axis_text_size, family = plot_font_family, color = plot_text_color),
              axis.text.x = element_text(
                size = axis_text_size, 
                family = plot_font_family,
                angle = 45,  # Add angle to x-axis tick labels
                hjust = 1,
                vjust = 1,
                margin = margin(t = 10, r = 0, b = 0, l = 0),  # Add top margin to push labels away
                color = if(is_last_row_in_grid) plot_text_color else "transparent"  # Only show ticks on last row
              ),
              axis.text.y = element_text(
                margin = margin(t = 0, r = 10, b = 0, l = 0),   # Add right margin to push labels away
                color = if(is_first_column) plot_text_color else "transparent" # Only show y-axis text for first column
              ),
              axis.ticks.x = element_line(color = if(is_last_row_in_grid) "gray50" else "transparent"),
              axis.ticks.y = element_line(color = if(is_first_column) "gray50" else "transparent"),
              axis.ticks.length = unit(0.3, "cm"),  # Make ticks longer to create more space
              legend.position = "none",  # Fixed: use "none" directly 
              panel.grid = element_line(color = grid_line_color),
              panel.background = element_rect(fill = plot_background_color, color = NA),
              plot.background = element_rect(fill = plot_background_color, color = NA),
              plot.margin = plot_margin_small
            )
        }, error = function(e) {
          message(paste("Error creating heatmap plot:", e$message))
          return(NULL)
        })
        
        # Only add to collection if valid
        if (!is.null(heatmap_plot) && inherits(heatmap_plot, "ggplot")) {
          heatmap_plots[[i]] <- heatmap_plot
        }
      }
      
      # Store the plots for this emotion
      latent_plots_by_emotion_phoneme[[emotion_name]] <- phoneme_plots
      latent_plots_by_emotion_speaker[[emotion_name]] <- speaker_plots
      heatmaps_by_emotion[[emotion_name]] <- heatmap_plots
      if (length(snapshot_plots) > 0) {
        snapshots_by_emotion[[emotion_name]] <- snapshot_plots
      }
    }
    
    # Store all plots for this latent type
    all_latent_plots_phoneme[[latent_type]] <- latent_plots_by_emotion_phoneme
    all_latent_plots_speaker[[latent_type]] <- latent_plots_by_emotion_speaker
    all_heatmaps[[latent_type]] <- heatmaps_by_emotion
    if (length(snapshots_by_emotion) > 0) {
      all_3d_snapshots[[latent_type]] <- snapshots_by_emotion
    }
  }


  # Create combined plots for each emotion across all latent types
  for (emotion_name in available_emotions) {
    # Check if we have data for this emotion in any latent type
    emotion_exists <- FALSE
    
    # Check phoneme plots
    if (length(all_latent_plots_phoneme) > 0) {
      emotion_exists <- any(sapply(all_latent_plots_phoneme, function(x) emotion_name %in% names(x)))
      if (emotion_exists) {
        create_combined_plot_by_emotion(all_latent_plots_phoneme, "", 
                                       "phoneme_plots_by_emotion", "Phoneme", emotion_name)
      }
    }
    
    # Check speaker plots
    if (length(all_latent_plots_speaker) > 0) {
      emotion_exists <- any(sapply(all_latent_plots_speaker, function(x) emotion_name %in% names(x)))
      if (emotion_exists) {
        create_combined_plot_by_emotion(all_latent_plots_speaker, "", 
                                      "speaker_plots_by_emotion", "Speaker", emotion_name)
      }
    }
    
    # Check heatmaps
    if (length(all_heatmaps) > 0) {
      emotion_exists <- any(sapply(all_heatmaps, function(x) emotion_name %in% names(x)))
      if (emotion_exists) {
        create_combined_plot_by_emotion(all_heatmaps, "", 
                                      "heatmap_plots_by_emotion", "Phoneme (x) vs Speaker (y)", emotion_name)
      }
    }
    
    # Check 3D snapshots
    if (length(all_3d_snapshots) > 0) {
      emotion_exists <- any(sapply(all_3d_snapshots, function(x) emotion_name %in% names(x)))
      if (emotion_exists) {
        create_combined_plot_by_emotion(all_3d_snapshots, "", 
                                      "3d_snapshot_plots_by_emotion", "Phoneme (x) vs Speaker (y) vs Response (z)", emotion_name)
      }
    }
  }
  
  # After all plots have been created and saved, add our new comparison plots
  if (length(all_3d_snapshots) > 0) {
    # Create the emotion-latent comparison plot for 3D snapshots
    create_emotion_latent_comparison_plot(all_3d_snapshots, save_dir)
  }
  
  if (length(all_heatmaps) > 0) {
    # Create the heatmap comparison plot
    create_emotion_latent_heatmap_comparison_plot(all_heatmaps, save_dir)
  }
  
  if (length(all_latent_plots_phoneme) > 0) {
    # Create the phoneme line comparison plot
    create_emotion_latent_phoneme_comparison_plot(all_latent_plots_phoneme, save_dir)
  }
  
  if (length(all_latent_plots_speaker) > 0) {
    # Create the speaker line comparison plot
    create_emotion_latent_speaker_comparison_plot(all_latent_plots_speaker, save_dir)
  }
  
  message("All plots have been created and saved.")
}

# Function to create a combined plot from a nested list structure - with consistent styling
create_combined_plot_by_emotion <- function(plot_list, title_prefix, save_suffix, varying_factor, emotion_name) {
  # Create a list to store plots for this emotion across all latent types
  emotion_rows <- list()
  row_count <- 1
  
  # Get all latent types with plots for this emotion
  available_latent_types <- names(plot_list)[sapply(plot_list, function(x) emotion_name %in% names(x))]
  last_latent_type <- tail(available_latent_types, 1)
  
  for (latent_type in names(plot_list)) {
    # Check if this latent type has plots for the specified emotion
    if (emotion_name %in% names(plot_list[[latent_type]])) {
      # Create a row title
      row_title <- paste0(latent_type)
      
      # Get the plots for this emotion
      row_plots <- plot_list[[latent_type]][[emotion_name]]
      
      # Check if this is the last latent type (bottom row in final combined plot)
      is_last_latent_type <- (latent_type == last_latent_type)
      
      # Validate that all elements in row_plots are ggplot objects and modify axis visibility
      valid_plots <- list()
      for (i in seq_along(row_plots)) {
        if (inherits(row_plots[[i]], "ggplot")) {
          # Modify the plot to show x-axis labels only if this is the last latent type
          plot <- row_plots[[i]] +
            theme(
              axis.text.x = element_text(
                size = axis_text_size, 
                family = plot_font_family,
                angle = 45,
                hjust = 1,
                vjust = 1,
                margin = margin(t = 10, r = 0, b = 0, l = 0),
                color = if(is_last_latent_type) plot_text_color else "transparent"
              ),
              axis.title.x = element_text(
                size = axis_title_size,
                family = plot_font_family,
                color = if(is_last_latent_type) plot_text_color else "transparent"
              ),
              axis.ticks.x = element_line(
                color = if(is_last_latent_type) "gray50" else "transparent"
              )
            )
          valid_plots[[length(valid_plots) + 1]] <- plot
        } else {
          message(paste("Skipping invalid plot object for", latent_type, emotion_name, "at position", i))
        }
      }
      
      # Skip if no valid plots
      if (length(valid_plots) == 0) {
        message(paste("No valid plots for", latent_type, emotion_name, "- skipping row"))
        next
      }
      
      # Add y-axis title for the first column showing latent type
      valid_plots[[1]] <- valid_plots[[1]] + 
        labs(y = latent_type) +
        theme(
          axis.title.y = element_text(
            size = axis_title_size, 
            family = plot_font_family,
            color = plot_text_color,
            angle = 90,
            hjust = 0.5
          )
        )
      
      # Safely combine the plots in this row using tryCatch
      row_plot <- tryCatch({
        wrap_plots(valid_plots, ncol = length(valid_plots)) +
          plot_annotation(
            title = row_title,
            theme = theme(
              plot.title = element_text(
                size = plot_title_size, 
                hjust = 0.5, 
                family = plot_font_family,
                color = plot_text_color,
                face = plot_title_face
              )
            )
          )
      }, error = function(e) {
        message(paste("Error creating row plot for", latent_type, emotion_name, ":", e$message))
        return(NULL)
      })
      
      # Only add the row if plot creation was successful
      if (!is.null(row_plot)) {
        emotion_rows[[row_count]] <- row_plot
        row_count <- row_count + 1
      }
    }
  }
  
  if (length(emotion_rows) == 0) {
    message(paste("No valid rows for emotion:", emotion_name))
    return(NULL)  # No plots for this emotion
  }
  
  # Combine rows for this emotion into a single plot
  combined_plot <- tryCatch({
    wrap_plots(emotion_rows, ncol = 1) +
      plot_annotation(
        title = paste0(title_prefix, " - ", emotion_name),
        caption = paste0(varying_factor),
        theme = theme(
          plot.title = element_text(
            size = plot_sup_title_size, 
            hjust = 0.5, 
            family = plot_font_family,
            color = plot_text_color,
            face = plot_title_face
          ),
          plot.caption = element_text(
            size = axis_title_size, 
            hjust = 0.5, 
            family = plot_font_family,
            color = plot_text_color
          ),
          plot.background = element_rect(fill = plot_background_color, color = NA)
        )
      )
  }, error = function(e) {
    message(paste("Error creating combined plot for emotion", emotion_name, ":", e$message))
    return(NULL)
  })
  
  # Save the combined plot if creation was successful
  if (!is.null(combined_plot)) {
    save_path <- file.path(save_dir, paste0("combined_", save_suffix, "_", emotion_name, ".png"))
    tryCatch({
      ggsave(save_path, combined_plot, width = 24, height = 6 * length(emotion_rows), dpi = 300, bg = plot_background_color)
      message(paste("Successfully saved combined plot for emotion:", emotion_name))
    }, error = function(e) {
      message(paste("Error saving combined plot for emotion", emotion_name, ":", e$message))
    })
  }
  
  return(combined_plot)
}

# Add a new function to extract and save legends for 2D plots
extract_2D_legend <- function(data, latent_type, emotion_name, save_dir) {
  # Create a dummy data frame for the plot with more varied data for better legend
  dummy_data <- data.frame(
    x = rep(1:5, each = 2),
    y = rep(1:2, 5),
    group = factor(1:10, levels = 1:10)
  )
  
  # Create a temporary plot with all the styling we want in the legend - using geom_point to ensure visibility
  temp_plot <- ggplot(dummy_data, aes(x = x, y = y, color = group)) +
    geom_point(size = 3) +  # Add visible elements to generate legend from
    # Handle color palette based on selected option
    {
      if (custom_palette_option == 1) {
        # Option 1: Blues/greens only (0.0-0.4)
        scale_color_viridis_d(
          option = palette, 
          begin = 0.0,
          end = 0.4,
          direction = 1,
          name = "Speaker/Phoneme"
        )
      } else if (custom_palette_option == 2) {
        # Option 2: Oranges/reds only (0.6-1.0)  
        scale_color_viridis_d(
          option = palette,
          begin = 0.6,
          end = 1.0,
          direction = 1,
          name = "Speaker/Phoneme"
        )
      } else {
        # Option 3: Custom blues->greens->oranges->reds (skipping yellow)
        scale_color_manual(
          values = colorRampPalette(c(
            viridis::turbo(n = 100, begin = 0.0, end = 0.4),  # Blues to greens
            viridis::turbo(n = 100, begin = 0.6, end = 1.0)   # Oranges to reds
          ))(10),
          name = "Speaker/Phoneme"
        )
      }
    } +
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.text = element_text(size = legend_text_size, family = plot_font_family, color = plot_text_color),
      legend.title = element_text(size = legend_title_size, family = plot_font_family, color = plot_text_color),
      legend.key.size = unit(legend_key_size, "cm"),
      legend.background = element_rect(fill = plot_background_color)
    )
  
  # Extract just the legend
  legend <- cowplot::get_legend(temp_plot)
  
  # Create a plot with just the legend
  legend_plot <- cowplot::ggdraw() + 
    cowplot::draw_grob(legend)
  
  # Save the legend
  legend_file <- file.path(save_dir, paste0(latent_type, "_", emotion_name, "_2D_legend.png"))
  ggsave(legend_file, legend_plot, width = 10, height = 2, dpi = 300, bg = plot_background_color)
  
  return(legend_file)
}

# Add a function to extract and save speaker legend for 2D plots
extract_2D_speaker_legend <- function(data, latent_type, emotion_name, save_dir) {
  # Get unique speaker IDs from the data
  emotion_idx <- which(available_emotions == emotion_name) - 1
  filtered_data <- data$speaker[data$emotion == emotion_idx]
  unique_speakers <- sort(unique(filtered_data))
  
  # Get proper speaker labels using INT_TO_SPEAKER mapping
  speaker_labels <- sapply(unique_speakers, function(x) {
    speaker_name <- INT_TO_SPEAKER[x + 1]
    if (is.na(speaker_name)) paste0("Speaker ", x) else speaker_name
  })
  
  # Create a data frame for the legend
  dummy_data <- data.frame(
    x = 1:length(unique_speakers),
    y = rep(1, length(unique_speakers)),
    speaker = factor(unique_speakers, levels = unique_speakers)
  )
  
  # Create a temporary plot with the speakers as colors
  temp_plot <- ggplot(dummy_data, aes(x = x, y = y, color = speaker)) +
    geom_point(size = 3) +
    # Handle color palette based on selected option
    {
      if (custom_palette_option == 1) {
        scale_color_viridis_d(
          option = palette, 
          begin = 0.0,
          end = 0.4,
          direction = 1,
          name = "Speaker",
          labels = speaker_labels
        )
      } else if (custom_palette_option == 2) {
        scale_color_viridis_d(
          option = palette,
          begin = 0.6,
          end = 1.0,
          direction = 1,
          name = "Speaker",
          labels = speaker_labels
        )
      } else {
        scale_color_manual(
          values = colorRampPalette(c(
            viridis::turbo(n = 100, begin = 0.0, end = 0.4),  # Blues to greens
            viridis::turbo(n = 100, begin = 0.6, end = 1.0)   # Oranges to reds
          ))(length(unique_speakers)),
          name = "Speaker",
          labels = speaker_labels
        )
      }
    } +
    theme(
      legend.position = "right",
      legend.direction = "vertical",
      legend.text = element_text(size = legend_text_size, family = plot_font_family, color = plot_text_color),
      legend.title = element_text(size = legend_title_size, family = plot_font_family, color = plot_text_color),
      legend.key.size = unit(legend_key_size/2, "cm"),
      legend.background = element_rect(fill = plot_background_color)
    )
  
  # Extract just the legend
  legend <- cowplot::get_legend(temp_plot)
  
  # Create a plot with just the legend
  legend_plot <- cowplot::ggdraw() + 
    cowplot::draw_grob(legend)
  
  # Save the legend
  legend_file <- file.path(save_dir, paste0(latent_type, "_", emotion_name, "_2D_speaker_legend.png"))
  ggsave(legend_file, legend_plot, width = 6, height = 10, dpi = 300, bg = plot_background_color)
  
  return(legend_file)
}

# Add a function to extract and save phoneme legend for 2D plots
extract_2D_phoneme_legend <- function(data, latent_type, emotion_name, save_dir) {
  # Get unique phoneme IDs from the data
  emotion_idx <- which(available_emotions == emotion_name) - 1
  filtered_data <- data$phoneme[data$emotion == emotion_idx]
  unique_phonemes <- sort(unique(filtered_data))
  
  # Get proper phoneme labels using INT_TO_PHONEME mapping
  phoneme_labels <- sapply(unique_phonemes, function(x) {
    phoneme_name <- INT_TO_PHONEME[x + 1]
    if (is.na(phoneme_name)) paste0("Phoneme ", x) else phoneme_name
  })
  
  # Create a data frame for the legend
  dummy_data <- data.frame(
    x = 1:length(unique_phonemes),
    y = rep(1, length(unique_phonemes)),
    phoneme = factor(unique_phonemes, levels = unique_phonemes)
  )
  
  # Create a temporary plot with the phonemes as colors
  temp_plot <- ggplot(dummy_data, aes(x = x, y = y, color = phoneme)) +
    geom_point(size = 3) +
    # Handle color palette based on selected option
    {
      if (custom_palette_option == 1) {
        scale_color_viridis_d(
          option = palette, 
          begin = 0.0,
          end = 0.4,
          direction = 1,
          name = "Phoneme",
          labels = phoneme_labels
        )
      } else if (custom_palette_option == 2) {
        scale_color_viridis_d(
          option = palette,
          begin = 0.6,
          end = 1.0,
          direction = 1,
          name = "Phoneme",
          labels = phoneme_labels
        )
      } else {
        scale_color_manual(
          values = colorRampPalette(c(
            viridis::turbo(n = 100, begin = 0.0, end = 0.4),  # Blues to greens
            viridis::turbo(n = 100, begin = 0.6, end = 1.0)   # Oranges to reds
          ))(length(unique_phonemes)),
          name = "Phoneme",
          labels = phoneme_labels
        )
      }
    } +
    theme(
      legend.position = "right",
      legend.direction = "vertical",
      legend.text = element_text(size = legend_text_size, family = plot_font_family, color = plot_text_color),
      legend.title = element_text(size = legend_title_size, family = plot_font_family, color = plot_text_color),
      legend.key.size = unit(legend_key_size/2, "cm"),
      legend.background = element_rect(fill = plot_background_color)
    )
  
  # Extract just the legend
  legend <- cowplot::get_legend(temp_plot)
  
  # Create a plot with just the legend
  legend_plot <- cowplot::ggdraw() + 
    cowplot::draw_grob(legend)
  
  # Save the legend
  legend_file <- file.path(save_dir, paste0(latent_type, "_", emotion_name, "_2D_phoneme_legend.png"))
  ggsave(legend_file, legend_plot, width = 6, height = 10, dpi = 300, bg = plot_background_color)
  
  return(legend_file)
}

# Add a function to extract and save 3D heatmap legends - completing the function
extract_3D_heatmap_legend <- function(data, latent_type, emotion_name, save_dir) {
  # Define z limits consistent with the main plot
  y_padding <- 0.2
  z_limits <- c(-3 - y_padding, 3 + y_padding)
  
  # Create a custom color vector that skips yellow (0.4-0.6)
  custom_colors <- c(
    viridis::turbo(n = 100, begin = 0.0, end = 0.4),  # Blues to greens
    viridis::turbo(n = 100, begin = 0.6, end = 1.0)   # Oranges to reds
  )
  
  # Create a sample dataset with a range of values
  sample_data <- expand.grid(
    x = seq(1, 10, length.out = 20),
    y = seq(1, 10, length.out = 20)
  )
  sample_data$z <- with(sample_data, sin(x/2) * cos(y/2) * 3) # Values that span our z_limits
  
  # Create a temporary heatmap plot with real data to ensure legend generation
  temp_heatmap <- ggplot(sample_data, aes(x = x, y = y, fill = z)) +
    geom_tile() +
    scale_fill_gradientn(
      colors = custom_colors,
      limits = z_limits,
      name = "Latent Response",
      guide = guide_colorbar(
        direction = "vertical",
        title.position = "top",
        title.hjust = 0.5,
        barwidth = 1.5,
        barheight = 6
      )
    ) +
    theme_void() +
    theme(
      legend.position = "right",
      legend.direction = "vertical",
      legend.justification = "center",
      legend.box.just = "center",
      legend.text = element_text(size = legend_text_size * 0.6, family = plot_font_family, color = plot_text_color),
      legend.title = element_text(size = legend_title_size * 0.6, family = plot_font_family, color = plot_text_color),
      legend.margin = margin(0, 0, 0, 0),
      plot.margin = margin(0, 0, 0, 0),
      plot.background = element_rect(fill = plot_background_color, color = NA)
    )
  
  # Extract just the legend
  legend_grob <- cowplot::get_legend(temp_heatmap)
  
  # Make sure we actually got a legend
  if (is.null(legend_grob)) {
    stop("Failed to extract legend - legend_grob is NULL")
  }
  
  # Create a new plot with just the legend
  final_legend_plot <- ggplot() +
    theme_void() +
    theme(plot.background = element_rect(fill = plot_background_color, color = NA))
  
  # Use cowplot instead of annotation_custom which seems to be causing issues
  final_legend_plot <- cowplot::ggdraw() + 
    cowplot::draw_grob(legend_grob, x = 0.5, y = 0.5, hjust = 0.5, vjust = 0.5)
  
  # Save the legend
  legend_file <- file.path(save_dir, paste0(latent_type, "_", emotion_name, "_3D_heatmap_legend.png"))
  ggsave(legend_file, final_legend_plot, width = 3, height = 6, dpi = 300, bg = plot_background_color)
  
  return(legend_file)
}

# Run the analysis
analyze_iemocap_latent_responses(ckp_dir, save_dir, available_emotions)