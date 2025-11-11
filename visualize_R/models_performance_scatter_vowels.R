library(dplyr)
library(ggplot2)
library(stringr) 
library(tidyr) 
library(fmsb)  # For radar charts
library(RColorBrewer)
library(ggbreak)
library(ggrepel)
library(gridExtra)
library(grid)

# Set and create directory and filepaths to save
parent_save_dir <-  file.path('..','figures','model_performance_vowels_only_mel_models')

scatter_2d_save_dir <- file.path(parent_save_dir, "scatterplots_two_metrics")
if (!dir.exists(scatter_2d_save_dir)) {
  dir.create(scatter_2d_save_dir, recursive = TRUE, showWarnings = FALSE)
}

scatter_3d_save_dir <- file.path(parent_save_dir, "scatterplots_three_metrics")
if (!dir.exists(scatter_3d_save_dir)) {
  dir.create(scatter_3d_save_dir, recursive = TRUE, showWarnings = FALSE)
}

# Style and font parameters
plot_font_family <- "Arial"
plot_background_color <- "white"
plot_text_color <- "black"

#showtext_auto(enable = TRUE)

# Plot types
plot_2D_scatterplots <- TRUE
plot_3D_scatterplots <- TRUE

#color palette
palette <- "plasma"
palette_w_yellows <- "Set1"
yellow_block_threshold <- 0.8

# Axis titles
axis_title_size_scatter <- 30
axis_title_size_bubble <- 30
axis_title_face <- "plain"

# Axis tick labels
axis_text_size_scatter <- 30
axis_text_size_bubble <- 30
axis_text_face <- "plain"

# Plot title
plot_title_size_scatter <- 0
plot_title_size_bubble <- 0
plot_title_face <- "plain"
plot_title_hjust <- 0.5
plot_title_vjust <- 0.5

# Legend elements
legend_text_size_scatter <- 12
legend_text_size_bubble <- 12
legend_title_size_scatter <- 13
legend_title_size_bubble <- 13
legend_text_face <- "plain"
legend_title_face <- "plain"
legend_key_size_scatter <- 1  # in cm
legend_key_size_bubble <- 1  # in cm

# Geom elements
point_size_scatter <- 10
point_size_bubble_min <- 3
point_size_bubble_max <- 15
text_size_scatter <- 9
text_size_bubble <- 9
line_size_repel <- 1
line_alpha_repel <- 0.6

# Margin settings
plot_margin_scatter <- margin(10, 10, 10, 10, "pt")
plot_margin_bubble <- margin(10, 10, 10, 10, "pt")

metrics <- c("mutual_info", "gaussian_corr_norm", "disentanglement", "completeness", 
             "informativeness", "explicitness", "modularity","IRS", 
             "supervised_phoneme_recognition", "supervised_speaker_identification",
             "unsupervised_phoneme_recognition", "unsupervised_speaker_identification")

#Waveform models 
#selected_models <- c("PCA_wav",  "ICA_wav",  
#                     "fcAE_wav", "fc_betaVAE_wav", "MDecAE_EWT_wav", 
#                     "MDecVAE_FD_wav", "betaMDecVAE_EWT_wav")
#Mel models 
selected_models <- c("PCA_mel", "ICA_mel", 
                     "fcAE_mel", "fcVAE_mel", "fc_betaVAE_mel", "MDecAE_EWT_mel", "MDecAE_FD_mel", 
                     "MDecVAE_FD_mel", "MDecVAE_EWT_mel", "betaMDecVAE_FD_mel", "betaMDecVAE_EWT_mel")

# Store model data 
model_data <- data.frame(
  metric = metrics,
  PCA_wav = c(0.014,0.001,0.01,0.065,0.525,0.009,0.817,0.611,0.794, 0.421, 0.259, 0.219),
  PCA_mel = c(0.004,0.035,0.230,0.239,0.801,0.894,0.645,0.477,0.974, 0.741, 0.433, 0.499),
  ICA_wav = c(0.023,0.052,0.127,0.066,0.578,0.009,0.968,0.703,0.824, 0.471, 0.260, 0.220),
  ICA_mel = c(0.005,4.85,0.086,0.156,0.812,0.894,0.605,0.538,0.974, 0.741, 0.455, 0.506),

  cAE_wav = c(0.070,0.486,0.081,0.212,0.690,0.664,0.963,0.526,0.957, 0.720, 0.567, 0.333),
  cVAE_wav = c(0.004,0.264,0.017,0.773,0.755,0.690,0.759,0.449,0.903, 0.258 , 0.349, 0.573),
  c_betaVAE_wav = c(0.020,0.072,0.065,0.534,0.631,0.597,0.742,0.458,0.908, 0.561, 0.513, 0.250),

  fcAE_wav = c(0.027,3.171,0.183,0.214,0.433,0.615,0.966,0.520, 0.650, 0.353, 0.458, 0.270),
  fcVAE_wav = c(0.097,0.316,0.106,0.563,0.361,0.403,0.960,0.535,0.523, 0.226, 0.227, 0.209),
  fc_betaVAE_wav = c(0.011,0.256,0.039,0.615,0.425,0.488,0.834,0.482,0.589,0.299,0.246,0.225),
  
  fcAE_mel = c(0.044,1.715,0.329,0.147,0.794,0.902,0.779,0.570,0.976, 0.744, 0.561, 0.316),
  fcVAE_mel = c(0.011,0.044,0.03,0.578,0.781,0.789,0.450,0.517,0.939, 0.738, 0.373, 0.332),
  fc_betaVAE_mel = c(0.007,0.061,0.062,0.615,0.805,0.772,0.563,0.552,0.963, 0.732, 0.367, 0.265),
  
  MDecAE_FD_mel = c(0.064,0.459,0.526,0.219,0.805,0.947,0.817,0.521,0.944, 0.938, 0.317,0.588),
  MDecAE_EWT_mel = c(0.089,0.565,0.529,0.26,0.761,0.913,0.871,0.619,0.974, 0.956, 0.388,0.550),

  MDecVAE_FD_mel = c(0.031,0.102,0.618,0.22,0.761,0.921,0.828,0.655,0.916, 0.769, 0.391, 0.516),
  MDecVAE_EWT_mel = c(0.040,0.114,0.553,0.233,0.717,0.918,0.779,0.626,0.974, 0.581, 0.412, 0.366),

  betaMDecVAE_FD_mel = c(0.051,0.173,0.526,0.196,0.809,0.929,0.809,0.600,0.924, 0.750, 0.334, 0.536),
  betaMDecVAE_EWT_mel = c(0.036,0.147,0.582,0.222,0.742,0.915,0.905,0.523,0.977, 0.713, 0.377, 0.638),

  ideal_model = c(0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)
)

model_display_names <- c(
  "PCA_wav" = "PCA (Waveform)",
  "PCA_mel" = "PCA",
  "ICA_wav" = "ICA (Waveform)",
  "ICA_mel" = "ICA",
  "cAE_wav" = "Conv AE (Waveform)",
  "cVAE_wav" = "Conv VAE (Waveform)",
  "c_betaVAE_wav" = "Conv β-VAE (Waveform)",
  "fcAE_wav" = "FC AE (Waveform)",
  "fcVAE_wav" = "FC VAE (Waveform)",
  "fc_betaVAE_wav" = "FC β-VAE (Waveform)",
  "fcAE_mel" = "AE",
  "fcVAE_mel" = "VAE",
  "fc_betaVAE_mel" = "β-VAE",
  "MDecAE_FD_mel" = "DecAE + FD",
  "MDecAE_EWT_mel" = "DecAE + EWT",
  "MDecVAE_FD_mel" = "DecVAE + FD",
  "MDecVAE_EWT_mel" = "DecVAE + EWT",
  "betaMDecVAE_FD_mel" = "β-DecVAE + FD",
  "betaMDecVAE_EWT_mel" = "β-DecVAE + EWT",
  "ideal_model" = "Ideal Model"
)

# Normalize values to 0-1
# Fix any values above 1.0 to 1.0
model_data[, -1] <- apply(model_data[, -1], 2, function(x) {
  ifelse(x > 1, 1, x)
})


create_metric_scatterplot <- function(data, x_metric, y_metric, selected_models = NULL, save_path = NULL, colorblind = TRUE,
                      x_limits = c(0, 1), y_limits = c(0, 1), x_breaks = NULL, y_breaks = NULL) {
  # Get indices of metrics
  x_idx <- which(data$metric == x_metric)
  y_idx <- which(data$metric == y_metric)
  
  # Extract values for x_metric and y_metric for all models
  result_data <- data.frame(model = colnames(data)[-1])
  result_data[[x_metric]] <- as.numeric(data[x_idx, -1])
  result_data[[y_metric]] <- as.numeric(data[y_idx, -1])
  
  # Filter for selected models if specified
  if (!is.null(selected_models)) {
    # Remove ideal_model as requested
    selected_models <- selected_models[selected_models != "ideal_model"]
    result_data <- result_data %>%
      filter(model %in% selected_models)
  }
  
  # Define nice metric labels
  metric_labels <- c(
    "mutual_info" = "Mutual Information",
    "gaussian_corr_norm" = "Gaussian Correlation Norm.",
    "disentanglement" = "Disentanglement",
    "completeness" = "Completeness",
    "informativeness" = "Informativeness",
    "explicitness" = "Explicitness",
    "modularity" = "Modularity",
    "IRS" = "Robustness",
    "supervised_phoneme_recognition" = "Phoneme Recognition (Accuracy)",
    "supervised_speaker_identification" = "Speaker Identification (Accuracy)",
    "unsupervised_phoneme_recognition" = "Phoneme Recognition (Unsupervised)",
    "unsupervised_speaker_identification" = "Speaker Identification (Unsupervised)"
  )
  
  # Create unique shape mapping for each model
  n_models <- nrow(result_data)
  
  # Make sure we have enough shapes (recycle if needed)
  all_shapes <- c(15:18, 7:14, 0:6)  # Use all available shape codes
  shape_mapping <- setNames(
    rep(all_shapes, length.out = n_models),
    result_data$model
  )
  
  # Create the scatterplot base with points but no labels yet
  p <- ggplot(result_data, aes(x = !!sym(x_metric), y = !!sym(y_metric))) +
    # Add points with distinct colors and shapes by model
    geom_point(
      aes(color = model, shape = model),
      size = point_size_scatter,
      alpha = 0.9
    )
  
  # Apply color scales first
  if (colorblind) {
    p <- p + scale_color_viridis_d(
      option = palette,
      end = yellow_block_threshold
    )
  } else {
    p <- p + scale_color_brewer(
      palette = palette_w_yellows
    )
  }
  
  # Add shape mapping
  p <- p + scale_shape_manual(values = shape_mapping)
  
  # Add optimized text labels - using only geom_text_repel now, removing fallback mechanism
  p <- p + ggrepel::geom_text_repel(
    data = result_data,
    aes(
      x = !!sym(x_metric), 
      y = !!sym(y_metric),
      label = model_display_names[model],
      color = model
    ),
    direction = "both",
    seed = 42,
    segment.size = line_size_repel,
    segment.alpha = line_alpha_repel,
    segment.color = "gray30",
    size = text_size_scatter,
    fontface = "bold",
    family = plot_font_family,
    box.padding = 1.2,
    point.padding = 0.8,
    force = 15,                   # Increased repulsion force
    force_pull = 0,               # No pull toward point
    max.iter = 20000,             # More iterations
    max.time = 10,                # More time to find optimal positions
    max.overlaps = Inf,
    min.segment.length = 0,
    hjust = 0.5,
    vjust = 0.5,
    show.legend = FALSE
  )
  
  # Complete the plot
  p <- p +
    labs(
      x = metric_labels[x_metric],
      y = metric_labels[y_metric],
      title = NULL,
      color = "Model",
      shape = "Model"
    ) +
    theme_minimal() +
    theme(
      legend.position = "none",
      plot.title = element_text(size = plot_title_size_scatter, face = plot_title_face, 
                               hjust = plot_title_hjust, vjust = plot_title_vjust, 
                               family = plot_font_family),
      axis.title = element_text(size = axis_title_size_scatter, family = plot_font_family),
      axis.text = element_text(size = axis_text_size_scatter, family = plot_font_family),
      axis.text.x = element_text(size = axis_text_size_scatter, family = plot_font_family),
      axis.text.y = element_text(size = axis_text_size_scatter, family = plot_font_family),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(fill = NA, color = "gray80"),
      panel.background = element_rect(fill = plot_background_color, color = NA),
      plot.background = element_rect(fill = plot_background_color, color = NA),
      plot.margin = plot_margin_scatter
    )
  
  # Calculate adaptive tick marks instead of fixed increments
  x_range <- x_limits[2] - x_limits[1]
  y_range <- y_limits[2] - y_limits[1]
  
  # Calculate appropriate number of ticks based on range
  # For smaller ranges, use more ticks (at least 5)
  x_tick_count <- max(5, 5 * x_range)
  y_tick_count <- max(5, 5 * y_range)
  
  # Calculate tick positions
  x_tick_positions <- pretty(x_limits, n = x_tick_count)
  y_tick_positions <- pretty(y_limits, n = y_tick_count)
  
  # Keep only ticks within the limits
  x_tick_positions <- x_tick_positions[x_tick_positions >= x_limits[1] & x_tick_positions <= x_limits[2]]
  y_tick_positions <- y_tick_positions[y_tick_positions >= y_limits[1] & y_tick_positions <= y_limits[2]]
  
  # Apply the adaptive tick marks
  base_p <- p + 
    scale_x_continuous(limits = x_limits, breaks = x_tick_positions) +
    scale_y_continuous(limits = y_limits, breaks = y_tick_positions)
  
  # Check if we should apply breaks
  if (!is.null(x_breaks) || !is.null(y_breaks)) {
    # Prepare break ranges if needed
    x_break_ranges <- NULL
    y_break_ranges <- NULL
    
    if (!is.null(x_breaks)) {
      # Find empty ranges for x-axis
      x_data <- result_data[[x_metric]]
      x_break_ranges <- list()
      
      for (i in 1:(length(x_breaks)-1)) {
        range_start <- x_breaks[i]
        range_end <- x_breaks[i+1]
        
        # Check if there's any data in this range
        if (!any(x_data >= range_start & x_data <= range_end) && 
            (range_end - range_start) > 0.1) {
          x_break_ranges <- c(x_break_ranges, list(c(range_start, range_end)))
        }
      }
    }
    
    if (!is.null(y_breaks)) {
      # Find empty ranges for y-axis
      y_data <- result_data[[y_metric]]
      y_break_ranges <- list()
      
      for (i in 1:(length(y_breaks)-1)) {
        range_start <- y_breaks[i]
        range_end <- y_breaks[i+1]
        
        # Check if there's any data in this range
        if (!any(y_data >= range_start & y_data <= range_end) && 
            (range_end - range_start) > 0.1) {
          y_break_ranges <- c(y_break_ranges, list(c(range_start, range_end)))
        }
      }
    }
    
    # Apply breaks using ggbreak
    if (length(x_break_ranges) > 0 || length(y_break_ranges) > 0) {
      # Try to use ggbreak::ggplot_break
      tryCatch({
        # Apply breaks at the end all at once
        if (length(x_break_ranges) > 0 && length(y_break_ranges) > 0) {
          # Both axes need breaks
          p <- ggplot_break(p, x_breaks = x_break_ranges, y_breaks = y_break_ranges)
        } else if (length(x_break_ranges) > 0) {
          # Only x-axis needs breaks
          p <- ggplot_break(p, x_breaks = x_break_ranges)
        } else if (length(y_break_ranges) > 0) {
          # Only y-axis needs breaks
          p <- ggplot_break(p, y_breaks = y_break_ranges)
        }
      }, error = function(e) {
        # If ggbreak fails, fall back to the base plot with standard axes
        message("Warning: ggbreak failed with error: ", e$message)
        message("Falling back to standard axes without breaks")
        p <- base_p
      })
      p <- p + theme(
        axis.text = element_text(size = 20),
        axis.text.x = element_text(size = 20),
        axis.text.y = element_text(size = 20)
      )
    } else {
      # No breaks needed, use standard axes
      p <- base_p
    }
  } else {
    # No breaks requested, use standard axes
    p <- base_p
  }


  # Save plot if path is provided
  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, p, width = 12, height = 8, dpi = 600, bg = "white")
  }
  
  return(p)
}

# Update the create_metric_bubble_chart function to use filled circles
create_metric_bubble_chart <- function(data, x_metric, y_metric, size_metric, selected_models = NULL, save_path = NULL, colorblind = TRUE,
                            x_limits = c(0, 1), y_limits = c(0, 1), x_breaks = NULL, y_breaks = NULL) {

  # Get indices of metrics
  x_idx <- which(data$metric == x_metric)
  y_idx <- which(data$metric == y_metric)
  size_idx <- which(data$metric == size_metric)
  
  # Extract values for all three metrics for all models
  result_data <- data.frame(model = colnames(data)[-1])
  result_data[[x_metric]] <- as.numeric(data[x_idx, -1])
  result_data[[y_metric]] <- as.numeric(data[y_idx, -1])
  result_data[[size_metric]] <- as.numeric(data[size_idx, -1])
  
  # Filter for selected models if specified
  if (!is.null(selected_models)) {
    # Remove ideal_model as requested
    selected_models <- selected_models[selected_models != "ideal_model"]
    result_data <- result_data %>%
      filter(model %in% selected_models)
  }
  
  # Define nice metric labels
  metric_labels <- c(
    "mutual_info" = "Mutual Information",
    "gaussian_corr_norm" = "Gaussian Correlation Norm.",
    "disentanglement" = "Disentanglement",
    "completeness" = "Completeness",
    "informativeness" = "Informativeness",
    "explicitness" = "Explicitness",
    "modularity" = "Modularity",
    "IRS" = "Robustness",
    "supervised_phoneme_recognition" = "Phoneme Recognition (Accuracy)",
    "supervised_speaker_identification" = "Speaker Identification (Accuracy)",
    "unsupervised_phoneme_recognition" = "Phoneme Recognition (Unsupervised)",
    "unsupervised_speaker_identification" = "Speaker Identification (Unsupervised)"
  )
  
  # Create the bubble chart
  p <- ggplot(result_data, aes(x = !!sym(x_metric), y = !!sym(y_metric))) +
    # Add bubbles with distinct colors by model
    geom_point(
      aes(size = !!sym(size_metric), color = model, fill = model),
      shape = 21,
      stroke = 0.8,
      alpha = 0.7
    )
  
  # Apply color scales
  if (colorblind) {
    p <- p + 
      scale_color_viridis_d(
        option = palette,
        end = yellow_block_threshold
      ) +
      scale_fill_viridis_d(
        option = palette,   
        end = yellow_block_threshold
      )
  } else {
    p <- p + 
      scale_color_brewer(
        palette = palette_w_yellows
      ) +
      scale_fill_brewer(
        palette = palette_w_yellows
      )
  }
  
  # Add optimized text labels - using only geom_text_repel now, removing fallback mechanism
  p <- p + ggrepel::geom_text_repel(
    data = result_data,
    aes(
      x = !!sym(x_metric), 
      y = !!sym(y_metric),
      label = model_display_names[model],
      color = model
    ),
    direction = "both",
    seed = 42,
    segment.size = line_size_repel,
    segment.alpha = line_alpha_repel,
    segment.color = "gray30",
    size = text_size_bubble,
    fontface = "bold",
    family = plot_font_family,
    box.padding = 1.2,
    point.padding = 0.8,
    force = 15,                   # Increased repulsion force
    force_pull = 0,               # No pull toward point
    max.iter = 20000,             # More iterations
    max.time = 10,                # More time to find optimal positions
    max.overlaps = Inf,
    min.segment.length = 0,
    hjust = 0.5,
    vjust = 0.5,
    show.legend = FALSE
  )
  
  # Complete the plot
  p <- p +
    scale_size_continuous(
      name = metric_labels[size_metric],
      range = c(point_size_bubble_min, point_size_bubble_max)
    ) +
    labs(
      x = metric_labels[x_metric],
      y = metric_labels[y_metric],
      title = NULL,
      color = "Model",
      fill = "Model"
    ) +
    theme_minimal() +
    theme(
      legend.position = "none",
      plot.title = element_text(size = plot_title_size_bubble, face = plot_title_face, 
                              hjust = plot_title_hjust, vjust = plot_title_vjust,
                              family = plot_font_family),
      axis.title = element_text(size = axis_title_size_bubble, family = plot_font_family),
      axis.text = element_text(size = axis_text_size_bubble, family = plot_font_family),
      axis.text.x = element_text(size = axis_text_size_bubble, family = plot_font_family),
      axis.text.y = element_text(size = axis_text_size_bubble, family = plot_font_family),
      panel.grid.major = element_line(color = "gray90"),
      panel.grid.minor = element_line(color = "gray95"),
      panel.background = element_rect(fill = plot_background_color, color = NA),
      plot.background = element_rect(fill = plot_background_color, color = NA),
      plot.margin = plot_margin_bubble
    )
  
  # Calculate adaptive tick marks instead of fixed increments
  x_range <- x_limits[2] - x_limits[1]
  y_range <- y_limits[2] - y_limits[1]
  
  # Calculate appropriate number of ticks based on range
  # For smaller ranges, use more ticks (at least 5)
  x_tick_count <- max(5, 5 * x_range)
  y_tick_count <- max(5, 5 * y_range)
  
  # Calculate tick positions
  x_tick_positions <- pretty(x_limits, n = x_tick_count)
  y_tick_positions <- pretty(y_limits, n = y_tick_count)
  
  # Keep only ticks within the limits
  x_tick_positions <- x_tick_positions[x_tick_positions >= x_limits[1] & x_tick_positions <= x_limits[2]]
  y_tick_positions <- y_tick_positions[y_tick_positions >= y_limits[1] & y_tick_positions <= y_limits[2]]
  
  # Apply the adaptive tick marks
  base_p <- p + 
    scale_x_continuous(limits = x_limits, breaks = x_tick_positions) +
    scale_y_continuous(limits = y_limits, breaks = y_tick_positions)
  
  # Check if we should apply breaks
  if (!is.null(x_breaks) || !is.null(y_breaks)) {
    # Prepare break ranges if needed
    x_break_ranges <- NULL
    y_break_ranges <- NULL
    
    if (!is.null(x_breaks)) {
      # Find empty ranges for x-axis
      x_data <- result_data[[x_metric]]
      x_break_ranges <- list()
      
      for (i in 1:(length(x_breaks)-1)) {
        range_start <- x_breaks[i]
        range_end <- x_breaks[i+1]
        
        # Check if there's any data in this range
        if (!any(x_data >= range_start & x_data <= range_end) && 
            (range_end - range_start) > 0.1) {
          x_break_ranges <- c(x_break_ranges, list(c(range_start, range_end)))
        }
      }
    }
    
    if (!is.null(y_breaks)) {
      # Find empty ranges for y-axis
      y_data <- result_data[[y_metric]]
      y_break_ranges <- list()
      
      for (i in 1:(length(y_breaks)-1)) {
        range_start <- y_breaks[i]
        range_end <- y_breaks[i+1]
        
        # Check if there's any data in this range
        if (!any(y_data >= range_start & y_data <= range_end) && 
            (range_end - range_start) > 0.1) {
          y_break_ranges <- c(y_break_ranges, list(c(range_start, range_end)))
        }
      }
    }
    
    # Apply breaks using ggbreak
    if (length(x_break_ranges) > 0 || length(y_break_ranges) > 0) {
      # Try to use ggbreak::ggplot_break
      tryCatch({
        # Apply breaks at the end all at once
        if (length(x_break_ranges) > 0 && length(y_break_ranges) > 0) {
          # Both axes need breaks
          p <- ggplot_break(p, x_breaks = x_break_ranges, y_breaks = y_break_ranges)
        } else if (length(x_break_ranges) > 0) {
          # Only x-axis needs breaks
          p <- ggplot_break(p, x_breaks = x_break_ranges)
        } else if (length(y_break_ranges) > 0) {
          # Only y-axis needs breaks
          p <- ggplot_break(p, y_breaks = y_break_ranges)
        }
      }, error = function(e) {
        # If ggbreak fails, fall back to the base plot with standard axes
        message("Warning: ggbreak failed with error: ", e$message)
        message("Falling back to standard axes without breaks")
        p <- base_p
      })
      p <- p + theme(
        axis.text = element_text(size = 20),
        axis.text.x = element_text(size = 20),
        axis.text.y = element_text(size = 20)
      )
    } else {
      # No breaks needed, use standard axes
      p <- base_p
    }
  } else {
    # No breaks requested, use standard axes
    p <- base_p
  }

  # Save plot if path is provided
  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, p, width = 12, height = 8, dpi = 600, bg = "white")
  }
  
  return(p)
}


create_legend <- function(selected_models, size_metric = NULL, 
                         model_display_names = NULL, colorblind = TRUE, max_per_row = 5, 
                         is_bubble_chart = FALSE) {

  # Load necessary packages
  library(ggplot2)
  
  if (is.null(model_display_names)) {
    model_display_names <- setNames(selected_models, selected_models)
  }

  if (!is.null(selected_models)) {
    selected_models <- selected_models[selected_models != "ideal_model"]
  }

  selected_models <- sort(selected_models)

  # Prepare data for plotting
  df <- data.frame(
    x = 1:length(selected_models),
    y = rep(1, length(selected_models)),
    model = factor(selected_models, levels = selected_models)
  )
  
  # Define nice metric labels
  metric_labels <- c(
    "mutual_info" = "Mutual Information",
    "gaussian_corr_norm" = "Gaussian Correlation Norm.",
    "disentanglement" = "Disentanglement",
    "completeness" = "Completeness",
    "informativeness" = "Informativeness",
    "explicitness" = "Explicitness",
    "modularity" = "Modularity",
    "IRS" = "Robustness",
    "supervised_phoneme_recognition" = "Phoneme Recognition (Accuracy)",
    "supervised_speaker_identification" = "Speaker Identification (Accuracy)",
    "unsupervised_phoneme_recognition" = "Phoneme Recognition (Unsupervised)",
    "unsupervised_speaker_identification" = "Speaker Identification (Unsupervised)"
  )
  
  # Create a plot based on the chart type
  if (is_bubble_chart) {
    # For bubble charts - use filled circles (shape 16)
    p <- ggplot(df, aes(x, y)) +
      geom_point(aes(color = model), size = 5, shape = 16) +  # Shape 16 is a filled circle
      scale_color_discrete(
        name = "Model",
        labels = function(x) model_display_names[x]
      )
  } else {
    # For scatter plots - use various shapes
    all_shapes <- c(15:18, 7:14, 0:6)
    shape_values <- all_shapes[1:length(selected_models)]
    
    p <- ggplot(df, aes(x, y)) +
      geom_point(aes(color = model, shape = model), size = 5) +
      # Use model_display_names for both color and shape legends
      scale_shape_manual(
        values = shape_values,
        name = "Model",
        labels = function(x) model_display_names[x]
      ) +
      scale_color_discrete(
        name = "Model",
        labels = function(x) model_display_names[x]
      )
  }
  
  # Apply color palette
  if (colorblind) {
    p <- p + scale_color_viridis_d(
      option = palette,
      end = yellow_block_threshold,
      name = "Model",
      labels = function(x) model_display_names[x],
    )
  } else {
    p <- p + scale_color_brewer(
      palette = palette_w_yellows,
      name = "Model",
      labels = function(x) model_display_names[x]
    )
  }
  
  # Add size legend if needed
  if (!is.null(size_metric)) {
    # Create a dataframe for size
    size_df <- data.frame(
      x = rep(1, 5),
      y = rep(2, 5),
      size_value = seq(0.2, 1, length.out = 5)
    )
    
    # Add size points
    p <- p + 
      geom_point(
        data = size_df,
        aes(x = x, y = y, size = size_value),
        shape = 16,  # Filled circle for size legend
        color = "black",
        inherit.aes = FALSE
      ) +
      scale_size_continuous(
        name = metric_labels[size_metric],
        range = c(point_size_bubble_min, point_size_bubble_max)
      )
  }
  
  # Format the legend appearance
  p <- p + theme_void() +
    theme(
      text = element_text(family = plot_font_family),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.title = element_text(size = ifelse(is_bubble_chart, 
                                               legend_title_size_bubble, 
                                               legend_title_size_scatter),
                                 face = legend_title_face, 
                                 family = plot_font_family),
      legend.text = element_text(size = ifelse(is_bubble_chart, 
                                              legend_text_size_bubble, 
                                              legend_text_size_scatter),
                                family = plot_font_family),
      legend.key.size = unit(ifelse(is_bubble_chart, 
                                   legend_key_size_bubble, 
                                   legend_key_size_scatter), "cm"),
      plot.background = element_rect(fill = plot_background_color, color = NA)
    )
  
  # Configure legend rows if there are many models
  if (length(selected_models) > max_per_row) {
    if (is_bubble_chart) {
      p <- p + guides(
        color = guide_legend(
          nrow = ceiling(length(selected_models) / max_per_row),
          byrow = TRUE
        )
      )
    } else {
      p <- p + guides(
        color = guide_legend(
          nrow = ceiling(length(selected_models) / max_per_row),
          byrow = TRUE
        ),
        shape = guide_legend(
          nrow = ceiling(length(selected_models) / max_per_row),
          byrow = TRUE
        )
      )
    }
    
    # Configure size legend if present
    if (!is.null(size_metric)) {
      p <- p + guides(
        size = guide_legend(
          nrow = 1,
          byrow = TRUE
        )
      )
    }
  }
  
  # Make the plot area tiny so only the legend shows
  p <- p + 
    lims(x = c(0, 0.001), y = c(0, 0.001)) + 
    theme(
      plot.margin = unit(c(0, 0, 0, 0), "cm"),
      legend.box = "vertical"  # Stack legends vertically
    )
  
  return(p)
}

# Create output directory for the new plots
dir.create("model_performance/comparison_plots", recursive = TRUE, showWarnings = FALSE)


# Create scatterplots

if(plot_2D_scatterplots) {

  disentanglement_informativeness_plot <- create_metric_scatterplot(
    model_data, 
    "disentanglement", 
    "informativeness", 
    selected_models,
    file.path(scatter_2d_save_dir, "disentanglement_informativeness.png"),
    colorblind = TRUE,
    x_limits = c(0, 1),
    y_limits = c(0.5, 1),
    x_breaks = c(0, 0.7, 0.95, 1),
    y_breaks = c(0, 0.1, 0.4, 1)
  )

  # Modularity vs Explicitness
  modularity_explicitness_plot <- create_metric_scatterplot(
    model_data, 
    "modularity", 
    "explicitness", 
    x_limits = c(0.4, 1),
    y_limits = c(0.6, 1),
    selected_models,
    file.path(scatter_2d_save_dir, "modularity_explicitness.png"),
    colorblind = TRUE,
  )

  disentanglement_informativeness_plot <- create_metric_scatterplot(
    model_data, 
    "disentanglement", 
    "informativeness", 
    x_limits = c(0, 1),
    y_limits = c(0.5, 1),
    selected_models,
    file.path(scatter_2d_save_dir, "disentanglement_informativeness_colorblind.png"),
    colorblind = TRUE
  )

  # Modularity vs Explicitness
  modularity_explicitness_plot <- create_metric_scatterplot(
    model_data, 
    "modularity", 
    "explicitness", 
    x_limits = c(0.4, 1),
    y_limits = c(0.6, 1),
    selected_models,
    file.path(scatter_2d_save_dir, "modularity_explicitness_colorblind.png"),
    colorblind = TRUE
  )

  sup_speaker_phoneme_plot <- create_metric_scatterplot(
    model_data, 
    "supervised_speaker_identification", 
    "supervised_phoneme_recognition", 
    x_limits = c(0.7, 1),
    y_limits = c(0.8, 1),
    selected_models,
    file.path(scatter_2d_save_dir, "sup_speaker_phoneme_colorblind.png"),
    colorblind = TRUE
  )
}


if(plot_3D_scatterplots) {

  #Colorblind friendly bubble charts
  # Mutual Information vs Gaussianity with Robustness as size
  mi_gcn_robustness_plot <- create_metric_bubble_chart(
    model_data, 
    "mutual_info", 
    "gaussian_corr_norm", 
    "IRS", 
    selected_models,
    file.path(scatter_3d_save_dir, "mi_gcn_robustness_bubble_colorblind.png"),
    colorblind = TRUE,
    x_limits = c(0, 0.22),
    y_limits = c(0, 1)
  )

  # Disentanglement vs Informativeness with Robustness as size
  disent_info_robustness_plot <- create_metric_bubble_chart(
    model_data, 
    "disentanglement", 
    "informativeness", 
    "IRS", 
    selected_models,
    file.path(scatter_3d_save_dir, "disent_info_robustness_bubble_colorblind.png"),
    colorblind = TRUE,
    x_limits = c(0, 0.7),
    y_limits = c(0.48, 1)
  )

  # Supervised Speaker vs Supervised Phoneme bubble chart with Disentanglement as size
  sup_speaker_phoneme_disent_plot <- create_metric_bubble_chart(
    model_data, 
    "supervised_speaker_identification", 
    "supervised_phoneme_recognition", 
    "disentanglement", 
    selected_models,
    file.path(scatter_3d_save_dir, "sup_speaker_phoneme_disent_bubble_colorblind.png"),
    colorblind = TRUE,
    x_limits = c(0.5, 1),
    y_limits = c(0.85, 1)
  )

  modularity_explicitness_robustness_plot <- create_metric_bubble_chart(
    model_data, 
    "modularity", 
    "explicitness", 
    "IRS", 
    selected_models,
    file.path(scatter_3d_save_dir, "modularity_explicitness_robustness_bubble_colorblind.png"),
    colorblind = TRUE,
    x_limits = c(0.4, 1),
    y_limits = c(0.7, 1)
  )

}

scatter_legend <- create_legend(
  selected_models,
  size_metric = NULL,
  model_display_names = model_display_names,
  colorblind = TRUE,
  max_per_row = 5,
  is_bubble_chart = FALSE
)

bubble_legend_irs <- create_legend(
  selected_models,
  size_metric = "IRS",
  model_display_names = model_display_names,
  colorblind = TRUE,
  max_per_row = 5,
  is_bubble_chart = TRUE
)

bubble_legend_disent <- create_legend(
  selected_models,
  size_metric = "disentanglement",
  model_display_names = model_display_names,
  colorblind = TRUE,
  max_per_row = 5,
  is_bubble_chart = TRUE
)

# Save these plots directly - they should show just the legend with white background
ggsave(
  file.path(scatter_2d_save_dir, "scatter_legend.png"), 
  scatter_legend, 
  width = 11, 
  height = 2, 
  dpi = 600, 
  bg = "white"
)

ggsave(
  file.path(scatter_3d_save_dir, "bubble_legend_irs.png"), 
  bubble_legend_irs, 
  width = 11, 
  height = 2, 
  dpi = 600, 
  bg = "white"
)

ggsave(
  file.path(scatter_3d_save_dir, "bubble_legend_disent.png"), 
  bubble_legend_disent, 
  width = 11, 
  height = 2, 
  dpi = 600, 
  bg = "white"
)