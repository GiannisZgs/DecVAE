#' Fig.3: This script generates figures 3g in the main paper,
#' and supplementary figures 7b in the supplementary material. Creates metric scatterplots 
#' for the TIMIT dataset

library(vscDebugger)
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(ggbreak)
library(ggrepel)

# Set and create directory and filepaths to save
parent_save_dir <-  file.path('..','figures','fig_3_timit_results','fig_3g_timit_metrics_scatter')

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
legend_text_size_scatter <- 11
legend_text_size_bubble <- 10.5
legend_title_size_scatter <- 13
legend_title_size_bubble <- 12
legend_text_face <- "plain"
legend_title_face <- "plain"
legend_key_size_scatter <- 1.5  # in cm
legend_key_size_bubble <- 1.5  # in cm

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
             "phoneme_recognition", "speaker_identification")


selected_models <- c("PCA", "ICA", "raw_mels",
                     "AE", "VAE", "betaVAE", "DecAE_EWT", "DecVAE_EWT", "betaDecVAE_EWT",
                     "DecAE_FD", "DecVAE_FD", "betaDecVAE_FD")


# Store model data 
model_data <- data.frame(
  metric = metrics,
  PCA = c(0.0021, 0.0009266, 0.22912 , 0.15596 , 0.25837 ,0.64766 , 0.7293, 0.5509, 0.43106, 0.22338),
  ICA = c(0.0028, 0.21263, 0.13873, 0.084868 , 0.26239 , 0.64766 , 0.71436, 0.554, 0.44622 , 0.22337),
  raw_mels = c(0.1905, 4.02072, 0.10904 , 0.035566 , 0.26486 , 0.67851 , 0.98013 , 0.55682, 0.44593 , 0.24499 ),
  
  AE = c(0.093878, 2.37793, 0.1531, 0.062913, 0.19079, 0.57943, 0.95742, 0.5439, 0.4485, 0.31066),
  VAE = c(0.041274, 0.084475, 0.19096, 0.11677, 0.18826, 0.4918, 0.98014, 0.68308 , 0.46553 , 0.26989),
  betaVAE = c(0.021917,0.074006 , 0.20172 , 0.16403 , 0.18385 , 0.48599 , 0.93309 , 0.66886 , 0.45524 , 0.28806 ),
  
  DecAE_FD = c(0.035628, 0.21727, 0.23037 , 0.071132, 0.4079, 0.47188, 0.65718, 0.62506 , 0.46945 , 0.73253),
  DecAE_EWT = c(0.0335, 0.23648, 0.25427 , 0.085775 , 0.38272 , 0.47531, 0.65032, 0.67547, 0.46346, 0.68683),

  DecVAE_FD = c(0.060716, 0.18443, 0.2033 , 0.065371 , 0.4914 , 0.66606, 0.56884, 0.77173, 0.49421, 0.84889),
  DecVAE_EWT = c(0.065449, 0.18572, 0.23251, 0.072382, 0.4724, 0.67334, 0.54146, 0.70637, 0.49479 , 0.82426),

  betaDecVAE_FD = c(0.051969, 0.1475, 0.17166, 0.06661 , 0.3937, 0.46672 , 0.73948 , 0.87209, 0.43673 , 0.74837),
  betaDecVAE_EWT = c(0.051242, 0.15457, 0.25561, 0.087143, 0.376 , 0.50385, 0.57539, 0.87499, 0.44764 , 0.67773)
)

model_display_names <- c(
  "PCA" = "PCA",
  "ICA" = "ICA",
  "raw_mels" = "Raw Mel Fbank",
  "AE" = "AE",
  "VAE" = "VAE",
  "betaVAE" = "β-VAE",
  "DecAE_FD" = "DecAE + FD",
  "DecAE_EWT" = "DecAE + EWT",
  "DecVAE_FD" = "DecVAE + FD",
  "DecVAE_EWT" = "DecVAE + EWT",
  "betaDecVAE_FD" = "β-DecVAE + FD",
  "betaDecVAE_EWT" = "β-DecVAE + EWT"
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
  
  # Always remove ideal_model
  result_data <- result_data %>%
    filter(model != "ideal_model")
  
  # Filter for selected models if specified
  if (!is.null(selected_models)) {
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
    "supervised_phoneme_recognition" = "Phoneme Recognition (Supervised)",
    "supervised_speaker_identification" = "Speaker Identification (Supervised)",
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
    force = 15,                  
    force_pull = 0,               
    max.iter = 20000,             
    max.time = 10,                
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
  
  # Always remove ideal_model
  result_data <- result_data %>%
    filter(model != "ideal_model")
  
  # Filter for selected models if specified
  if (!is.null(selected_models)) {
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
    "supervised_phoneme_recognition" = "Phoneme Recognition (Supervised)",
    "supervised_speaker_identification" = "Speaker Identification (Supervised)",
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
    force = 15,                   
    force_pull = 0,               
    max.iter = 20000,             
    max.time = 10,                
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
    "supervised_phoneme_recognition" = "Phoneme Recognition (Supervised)",
    "supervised_speaker_identification" = "Speaker Identification (Supervised)",
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
    x_limits = c(0, 0.3),
    y_limits = c(0, 0.5),
    x_breaks = c(0, 0.7, 0.95, 1),
    y_breaks = c(0, 0.1, 0.4, 1)
  )

  # Modularity vs Explicitness
  modularity_explicitness_plot <- create_metric_scatterplot(
    model_data, 
    "modularity", 
    "explicitness", 
    x_limits = c(0.4, 1),
    y_limits = c(0.4, 1),
    selected_models,
    file.path(scatter_2d_save_dir, "modularity_explicitness.png"),
    colorblind = TRUE,
  )

  disentanglement_informativeness_plot <- create_metric_scatterplot(
    model_data, 
    "disentanglement", 
    "informativeness", 
    x_limits = c(0, 0.3),
    y_limits = c(0, 0.5),
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
    y_limits = c(0.4, 1),
    selected_models,
    file.path(scatter_2d_save_dir, "modularity_explicitness_colorblind.png"),
    colorblind = TRUE
  )

  sup_speaker_phoneme_plot <- create_metric_scatterplot(
    model_data, 
    "supervised_speaker_identification", 
    "supervised_phoneme_recognition", 
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
    x_limits = c(0, 0.25),
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
    x_limits = c(0, 0.5),
    y_limits = c(0, 0.5)
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
    x_limits = c(0, 1),
    y_limits = c(0, 1)
  )

  modularity_explicitness_robustness_plot <- create_metric_bubble_chart(
    model_data, 
    "modularity", 
    "explicitness", 
    "IRS", 
    selected_models,
    file.path(scatter_3d_save_dir, "modularity_explicitness_robustness_bubble_colorblind.png"),
    colorblind = TRUE,
    x_limits = c(0.5, 1),
    y_limits = c(0.3, 1)
  )

}

scatter_legend <- create_legend(
  selected_models,
  size_metric = NULL,
  model_display_names = model_display_names,
  colorblind = TRUE,
  max_per_row = 6,
  is_bubble_chart = FALSE
)

bubble_legend_irs <- create_legend(
  selected_models,
  size_metric = "IRS",
  model_display_names = model_display_names,
  colorblind = TRUE,
  max_per_row = 6,
  is_bubble_chart = TRUE
)

bubble_legend_disent <- create_legend(
  selected_models,
  size_metric = "disentanglement",
  model_display_names = model_display_names,
  colorblind = TRUE,
  max_per_row = 6,
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