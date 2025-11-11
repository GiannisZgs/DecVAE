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
parent_save_dir <-  file.path('..','figures','model_performance_voc_als')

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
axis_title_size_scatter <- 29
axis_title_size_bubble <- 29
axis_title_face <- "plain"

# Axis tick labels
axis_text_size_scatter <- 29
axis_text_size_bubble <- 29
axis_text_face <- "plain"

# Plot title
plot_title_size_scatter <- 0
plot_title_size_bubble <- 0
plot_title_face <- "plain"
plot_title_hjust <- 0.5
plot_title_vjust <- 0.5

# Legend elements
legend_text_size_scatter <- 9
legend_text_size_bubble <- 9
legend_title_size_scatter <- 9
legend_title_size_bubble <- 9
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
             "king_stage_weighted_accuracy", "king_stage_weighted_f1", "king_stage_weighted_f1_macro",
             "disease_duration_weighted_accuracy", "disease_duration_weighted_f1", "disease_duration_weighted_f1_macro",
             "phoneme_weighted_accuracy", "phoneme_weighted_f1", "phoneme_weighted_f1_macro",
             "speaker_id_weighted_accuracy", "speaker_id_weighted_f1", "speaker_id_weighted_f1_macro")

selected_models <- c("raw_mels","PCA", "ICA",
                     "AE_vowels", "AE_timit", "VAE_vowels", "VAE_timit", "betaVAE_vowels", "betaVAE_timit",
                     "betaDecVAE_FD_vowels", "betaDecVAE_EWT_vowels", "betaDecVAE_FD_timit","betaDecVAE_EWT_timit")
#betaDecVAE FD NoC = 4
#betaDecVAE EWT NoC = 3 

# Store model data 
model_data <- data.frame(
  metric = metrics,
  raw_mels = c(0.17913 , 2.88385 , 0.1234 , 0.082694 , 0.50649 , 0.62357 , 0.84538 , 0.65187 , 0.700 , 0.694 , 0.620 , 0.706 , 0.697 , 0.672 , 0.777 , 0.777 , 0.779 , 0.705 , 0.704 , 0.699),
  ICA = c(0.004, 0.000001 , 0.130 , 0.154 , 0.434 , 0.590 , 0.863 , 0.620 , 0.584 , 0.564 , 0.462 , 0.587 , 0.562 , 0.514 , 0.711 , 0.710 , 0.713 , 0.550 , 0.548 , 0.542 ),
  PCA = c(0.004, 0.000001 , 0.082 , 0.069 , 0.439 , 0.590 , 0.842 , 0.611 , 0.575 , 0.554 , 0.453 , 0.577 , 0.551 , 0.501 , 0.700 , 0.700 , 0.703 , 0.523 , 0.520 , 0.515),

  AE_vowels = c(0.091397 , 2.98506 , 0.1073 , 0.070382 , 0.37438 , 0.56997, 0.86571, 0.55294, 0.481 , 0.450 , 0.360 , 0.489 , 0.454 , 0.391 , 0.648 , 0.647 , 0.649 , 0.322 , 0.318 , 0.315),
  AE_timit = c( 0.09161, 3.01362, 0.1029 , 0.067915 , 0.36919 , 0.57386, 0.79231, 0.74264 , 0.475 , 0.440 , 0.331 , 0.483 , 0.443 , 0.376 , 0.648 , 0.647 , 0.650 , 0.332 , 0.329 , 0.324 ),

  VAE_vowels = c(0.035782 , 0.1344 , 0.074728 , 0.069139 , 0.3499 , 0.51626 , 0.85327, 0.46771 , 0.463 , 0.424 , 0.326 , 0.474 , 0.432 , 0.364 , 0.632 , 0.630 , 0.632 , 0.300 , 0.297 , 0.295),
  VAE_timit = c( 0.1217, 0.34092, 0.076864 , 0.046739 , 0.34015 , 0.54566 , 0.88843 , 0.5, 0.415 , 0.383 , 0.285 , 0.425 , 0.387 , 0.318 , 0.622 , 0.622 , 0.624 , 0.201 , 0.197 , 0.199),

  betaVAE_vowels = c( 0.042769, 0.21856 , 0.073676  , 0.06515 , 0.36475 , 0.54348 , 0.77463 , 0.48127 , 0.476 , 0.437 , 0.342 , 0.485 , 0.442 , 0.372 , 0.648 , 0.645 , 0.647 , 0.341 , 0.338 , 0.336),
  betaVAE_timit = c(0.067234 ,0.33216 , 0.086466  , 0.067478 , 0.35007 , 0.55535 , 0.86305, 0.48909 , 0.439 , 0.403 , 0.300 , 0.447 , 0.407 , 0.337 , 0.635 , 0.636 ,0.639 , 0.243 , 0.239 , 0.239),

  betaDecVAE_FD_vowels = c( 0.043877, 0.1547, 0.097035 , 0.085435 , 0.65322 , 0.62568 , 0.75035 , 0.72901 , 0.864 , 0.866 , 0.846 , 0.864 , 0.863 , 0.853 , 0.913 , 0.913 , 0.913 , 0.953 , 0.953 , 0.952 ),
  betaDecVAE_EWT_vowels = c( 0.036032, 0.14952, 0.099141 , 0.081047 , 0.65913  , 0.66163, 0.75033 , 0.65006 , 0.879 , 0.880 , 0.852 , 0.882 , 0.881 , 0.873 , 0.918 , 0.918 , 0.918 , 0.948 , 0.948 , 0.947),

  betaDecVAE_FD_timit = c(0.039401 , 0.20365, 0.1721 , 0.1013 , 0.6118 , 0.6703, 0.76702 , 0.62374 , 0.710 , 0.753 , 0.738 , 0.762 , 0.760 , 0.749 , 0.791 , 0.794 , 0.794 , 0.787 , 0.790 , 0.791),
  betaDecVAE_EWT_timit = c(0.045773 , 0.22885, 0.17755 , 0.093022 , 0.57109 , 0.65385 , 0.77598, 0.61062, 0.700 , 0.698 , 0.683 , 0.711 , 0.704 , 0.685 , 0.778 , 0.778 , 0.780 , 0.712 , 0.715 , 0.718)
)

model_display_names <- c(
  "PCA" = "PCA",
  "ICA" = "ICA",
  "raw_mels" = "Raw Mel Fbank",
  "AE_vowels" = "Vowels AE",
  "VAE_vowels" = "Vowels VAE",
  "betaVAE_vowels" = "Vowels β-VAE",
  "AE_timit" = "TIMIT AE",
  "VAE_timit" = "TIMIT VAE",
  "betaVAE_timit" = "TIMIT β-VAE",
  "betaDecVAE_FD_vowels" = "Vowels β-DecVAE + FD",
  "betaDecVAE_EWT_vowels" = "Vowels β-DecVAE + EWT",
  "betaDecVAE_FD_timit" = "TIMIT β-DecVAE + FD",
  "betaDecVAE_EWT_timit" = "TIMIT β-DecVAE + EWT"
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
    "king_stage_weighted_accuracy" = "King's Clinical Stage Detection (Accuracy)",
    "king_stage_weighted_f1" = "King's Clinical Stage Detection (F1)",
    "king_stage_weighted_f1_macro" = "King's Clinical Stage Detection (F1 Macro)",
    "disease_duration_weighted_accuracy" = "Disease Duration Detection(Accuracy)",
    "disease_duration_weighted_f1" = "Disease Duration Detection(F1)",
    "disease_duration_weighted_f1_macro" = "Disease Duration Detection(F1 Macro)",
    "phoneme_weighted_accuracy" = "Phoneme Recognition (Accuracy)",
    "phoneme_weighted_f1" = "Phoneme Recognition (F1)",
    "phoneme_weighted_f1_macro" = "Phoneme Recognition (F1 Macro)",
    "speaker_id_weighted_accuracy" = "Speaker Identification (Accuracy)",
    "speaker_id_weighted_f1" = "Speaker Identification (F1)",
    "speaker_id_weighted_f1_macro" = "Speaker Identification (F1 Macro)"
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
    "king_stage_weighted_accuracy" = "King's Clinical Stage Detection (Accuracy)",
    "king_stage_weighted_f1" = "King's Clinical Stage Detection (F1)",
    "king_stage_weighted_f1_macro" = "King's Clinical Stage Detection (F1 Macro)",
    "disease_duration_weighted_accuracy" = "Disease Duration Detection(Accuracy)",
    "disease_duration_weighted_f1" = "Disease Duration Detection(F1)",
    "disease_duration_weighted_f1_macro" = "Disease Duration Detection(F1 Macro)",
    "phoneme_weighted_accuracy" = "Phoneme Recognition (Accuracy)",
    "phoneme_weighted_f1" = "Phoneme Recognition (F1)",
    "phoneme_weighted_f1_macro" = "Phoneme Recognition (F1 Macro)",
    "speaker_id_weighted_accuracy" = "Speaker Identification (Accuracy)",
    "speaker_id_weighted_f1" = "Speaker Identification (F1)",
    "speaker_id_weighted_f1_macro" = "Speaker Identification (F1 Macro)"
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
    "king_stage_weighted_accuracy" = "King's Clinical Stage Detection (Accuracy)",
    "king_stage_weighted_f1" = "King's Clinical Stage Detection (F1)",
    "king_stage_weighted_f1_macro" = "King's Clinical Stage Detection (F1 Macro)",
    "disease_duration_weighted_accuracy" = "Disease Duration Detection(Accuracy)",
    "disease_duration_weighted_f1" = "Disease Duration Detection(F1)",
    "disease_duration_weighted_f1_macro" = "Disease Duration Detection(F1 Macro)",
    "phoneme_weighted_accuracy" = "Phoneme Recognition (Accuracy)",
    "phoneme_weighted_f1" = "Phoneme Recognition (F1)",
    "phoneme_weighted_f1_macro" = "Phoneme Recognition (F1 Macro)",
    "speaker_id_weighted_accuracy" = "Speaker Identification (Accuracy)",
    "speaker_id_weighted_f1" = "Speaker Identification (F1)",
    "speaker_id_weighted_f1_macro" = "Speaker Identification (F1 Macro)"
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
    y_limits = c(0, 1),
    x_breaks = c(0, 0.7, 0.95, 1),
    y_breaks = c(0, 0.1, 0.4, 1)
  )

  # Modularity vs Explicitness
  modularity_explicitness_plot <- create_metric_scatterplot(
    model_data, 
    "modularity", 
    "explicitness", 
    selected_models,
    file.path(scatter_2d_save_dir, "modularity_explicitness.png"),
    colorblind = TRUE,
  )

  disentanglement_informativeness_plot <- create_metric_scatterplot(
    model_data, 
    "disentanglement", 
    "informativeness", 
    selected_models,
    file.path(scatter_2d_save_dir, "disentanglement_informativeness_colorblind.png"),
    colorblind = TRUE
  )

  # Modularity vs Explicitness
  modularity_explicitness_plot <- create_metric_scatterplot(
    model_data, 
    "modularity", 
    "explicitness", 
    selected_models,
    file.path(scatter_2d_save_dir, "modularity_explicitness_colorblind.png"),
    colorblind = TRUE
  )

  sup_kings_phoneme_plot <- create_metric_scatterplot(
    model_data, 
    "king_stage_weighted_accuracy", 
    "phoneme_weighted_accuracy", 
    selected_models,
    file.path(scatter_2d_save_dir, "sup_kings_phoneme_colorblind.png"),
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
    x_limits = c(0, 0.3),
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
    x_limits = c(0, 0.4),
    y_limits = c(0.2, 1)
  )

  # Supervised Kings Staging vs Supervised Phoneme bubble chart with Disentanglement as size
  sup_kings_phoneme_disent_plot <- create_metric_bubble_chart(
    model_data, 
    "king_stage_weighted_accuracy", 
    "phoneme_weighted_accuracy", 
    "disentanglement", 
    selected_models,
    file.path(scatter_3d_save_dir, "sup_kings_phoneme_disent_bubble_colorblind.png"),
    colorblind = TRUE,
    x_limits = c(0.3, 1),
    y_limits = c(0.5, 1)
  )

  sup_kings_phoneme_disent_plot <- create_metric_bubble_chart(
    model_data, 
    "speaker_id_weighted_accuracy", 
    "phoneme_weighted_accuracy", 
    "king_stage_weighted_accuracy",
    selected_models,
    file.path(scatter_3d_save_dir, "sup_speaker_phoneme_kings_bubble_colorblind.png"),
    colorblind = TRUE,
    x_limits = c(0.1, 1),
    y_limits = c(0.5, 1)
  )

  sup_kings_phoneme_disent_plot <- create_metric_bubble_chart(
    model_data, 
    "king_stage_weighted_accuracy",
    "speaker_id_weighted_accuracy", 
    "phoneme_weighted_accuracy", 
    selected_models,
    file.path(scatter_3d_save_dir, "sup_kings_speaker_phoneme_bubble_colorblind.png"),
    colorblind = TRUE,
    x_limits = c(0.1, 1),
    y_limits = c(0.1, 1)
  )

  sup_kings_phoneme_disent_plot <- create_metric_bubble_chart(
    model_data, 
    "speaker_id_weighted_accuracy", 
    "disease_duration_weighted_accuracy", 
    "phoneme_weighted_accuracy",
    selected_models,
    file.path(scatter_3d_save_dir, "sup_speaker_disease_phoneme_bubble_colorblind.png"),
    colorblind = TRUE,
    x_limits = c(0.1, 1),
    y_limits = c(0.3, 1)
  )

  # Supervised Kings Staging vs Sup. Disease duration bubble chart with Phoneme Rec. as size
  sup_kings_phoneme_disent_plot <- create_metric_bubble_chart(
    model_data, 
    "king_stage_weighted_accuracy", 
    "disease_duration_weighted_accuracy", 
    "phoneme_weighted_accuracy", 
    selected_models,
    file.path(scatter_3d_save_dir, "sup_kings_disease_disent_bubble_colorblind.png"),
    colorblind = TRUE,
    x_limits = c(0.3, 1),
    y_limits = c(0.3, 1)
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
    y_limits = c(0.4, 1)
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

bubble_legend_phoneme_wacc <- create_legend(
  selected_models,
  size_metric = "phoneme_weighted_accuracy",
  model_display_names = model_display_names,
  colorblind = TRUE,
  max_per_row = 6,
  is_bubble_chart = TRUE
)

bubble_legend_king <- create_legend(
  selected_models,
  size_metric = "king_stage_weighted_accuracy",
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

ggsave(
  file.path(scatter_3d_save_dir, "bubble_legend_phoneme.png"), 
  bubble_legend_phoneme_wacc, 
  width = 11, 
  height = 2, 
  dpi = 600, 
  bg = "white"
)

ggsave(
  file.path(scatter_3d_save_dir, "bubble_legend_king.png"), 
  bubble_legend_king, 
  width = 11, 
  height = 2, 
  dpi = 600, 
  bg = "white"
)