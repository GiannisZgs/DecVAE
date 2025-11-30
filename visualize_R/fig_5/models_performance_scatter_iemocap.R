#' Fig.5: This script generates figures 5g in the main paper,
#' and supplementary figures 7d in the supplementary material. Creates metric scatterplots 
#' for the IEMOCAP dataset

library(vscDebugger)
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(ggrepel)

# Set and create directory and filepaths to save
parent_save_dir <-  file.path('..','figures','fig_5_iemocap_results','fig_5g_iemocap_metrics_scatter')
transfer_from <- "vowels"
scatter_2d_save_dir <- file.path(parent_save_dir, paste0("scatterplots_two_metrics_from_",transfer_from))
if (!dir.exists(scatter_2d_save_dir)) {
  dir.create(scatter_2d_save_dir, recursive = TRUE, showWarnings = FALSE)
}

scatter_3d_save_dir <- file.path(parent_save_dir, paste0("scatterplots_three_metrics_from_",transfer_from))
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
legend_text_size_bubble <- 10
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
             "informativeness", "explicitness", "modularity", "IRS",
             "weighted_accuracy_emotion", "unweighted_accuracy_emotion", "f1_emotion",
             "weighted_accuracy_phoneme", "f1_phoneme", "weighted_accuracy_speaker", "f1_speaker")
#"PCA",
selected_models <- c("raw_mels","ICA","PCA", "VAE_vowels", "VAE_timit", "betaVAE_vowels", "betaVAE_timit",
                      "DecVAE_EWT_vowels", "DecVAE_FD_timit", "betaDecVAE_EWT_vowels", "betaDecVAE_FD_timit",
                     "DecAE_EWT_vowels","DecAE_FD_timit")
#"DecVAE_FD_vowels","DecVAE_EWT_timit","betaDecVAE_FD_vowels","betaDecVAE_EWT_timit",

# Store model data 
model_data <- data.frame(
    metric = metrics,
    raw_mels = c(0.226, 4.571, 0.254 , 0.224 , 0.365 , 0.398 , 0.981 , 0.635, 0.405 , 0.403 , 0.405 , 0.508 , 0.401 , 0.247 , 0.242 ),
    PCA = c(0.003, 0.00001, 0.120 , 0.215 , 0.358 ,0.375 , 0.907, 0.610, 0.395 , 0.391 ,0.394 ,0.504 ,0.399 ,0.211 ,0.206 ),
    ICA = c(0.003, 0.00001, 0.128, 0.132 , 0.358 , 0.375 , 0.895, 0.610, 0.388 , 0.381 , 0.385 , 0.499 , 0.383 , 0.206 , 0.201 ),        
    VAE_vowels = c(0.034, 0.0699 , 0.09 , 0.221 , 0.327 , 0.226 , 0.963 , 0.567 , 0.367 , 0.360 ,0.363 , 0.482, 0.359, 0.163, 0.158),
    VAE_timit = c(0.049 , 0.104 ,0.155 ,0.129 ,0.330 ,0.290 ,0.971 ,0.617  , 0.363  , 0.360 , 0.362 , 0.481 , 0.376 , 0.154 , 0.151 ),
    betaVAE_vowels = c(0.019 , 0.084 , 0.071 , 0.238 , 0.332 ,0.266 ,0.907, 0.556   ,0.372 ,0.366 ,0.369 ,0.489 ,0.371 ,0.170 ,0.166 ),
    betaVAE_timit = c(0.024 ,0.078 ,0.118 , 0.199 , 0.330 ,0.265 ,0.966 ,0.607  , 0.366 , 0.359 , 0.362 , 0.484 , 0.365 , 0.166 ,0.163 ),

    DecAE_FD_timit = c(0.057, 0.341,0.445,0.309,0.443,0.278,0.876,0.658,0.462, 0.461, 0.462, 0.534, 0.447, 0.727, 0.731),
    DecAE_EWT_vowels = c(0.086, 0.334, 0.253, 0.159, 0.439, 0.253, 0.921, 0.731,0.459, 0.461, 0.457, 0.541, 0.458, 0.712, 0.713),
    DecVAE_FD_vowels = c( 0.017,0.091 ,0.233 , 0.141 , 0.341 ,0.277 ,0.940 ,0.809  , 0.440 , 0.440 , 0.439 , 0.536, 0.451, 0.737, 0.741),
    DecVAE_EWT_vowels = c(0.017 ,0.111 ,0.240 ,0.153 , 0.341 , 0.278 ,0.909 , 0.576  , 0.420, 0.446, 0.414,0.537 ,0.453 ,0.766 ,0.768 ),
    DecVAE_FD_timit = c(0.017 ,0.134 ,0.289 ,0.209 ,0.341 ,0.285 ,0.831 ,0.778  ,0.429 , 0.425, 0.430 ,0.534 ,0.446 ,0.718 ,0.722 ),
    DecVAE_EWT_timit = c( 0.019, 0.130 ,0.238 ,0.158 ,0.335 ,0.297 ,0.849 ,0.808  ,0.411 ,0.361 ,0.411 ,0.534 ,0.446 ,0.718 ,0.722 ),

    betaDecVAE_FD_vowels = c(0.019 ,0.116 ,0.451 , 0.293, 0.367, 0.277, 0.755, 0.683 ,0.436 ,0.432 ,0.436 ,0.533 ,0.447 ,0.737 ,0.740 ),
    betaDecVAE_EWT_vowels = c(0.019 , 0.053 ,0.28 ,0.160 ,0.359 ,0.290 ,0.889 ,0.799  ,0.448 ,0.446 ,0.447 ,0.533 ,0.443 ,0.766 ,0.714 ),
    betaDecVAE_FD_timit = c( 0.025, 0.127 ,0.231 ,0.118 ,0.316 ,0.243 ,0.788 ,0.713  , 0.396, 0.389, 0.396, 0.528, 0.437,0.677 ,0.685 ),
    betaDecVAE_EWT_timit = c(0.022, 0.136,0.254 ,0.149 ,0.317 ,0.242 ,0.879 ,0.755  , 0.366, 0.361,0.365 ,0.534 ,0.436 ,0.672 ,0.678 )
)


model_display_names <- c(
"raw_mels" = "Raw Mel Fbank",
"ICA" = "ICA",
"PCA" = "PCA",
"VAE_vowels" = "Vowels VAE",
"VAE_timit" = "TIMIT VAE",
"betaVAE_vowels" = "Vowels β-VAE",
"betaVAE_timit" = "TIMIT β-VAE",
"DecVAE_FD_vowels" = "Vowels DecVAE + FD",
"DecVAE_EWT_vowels" = "Vowels DecVAE + EWT",
"DecVAE_FD_timit" = "TIMIT DecVAE + FD",
"DecVAE_EWT_timit" = "TIMIT DecVAE + EWT",
"betaDecVAE_FD_vowels" = "Vowels β-DecVAE + FD",
"betaDecVAE_EWT_vowels" = "Vowels β-DecVAE + EWT",
"betaDecVAE_FD_timit" = "TIMIT β-DecVAE + FD",
"betaDecVAE_EWT_timit" = "TIMIT β-DecVAE + EWT",
"DecAE_EWT_vowels" = "Vowels DecAE + EWT",
"DecAE_FD_timit" = "TIMIT DecAE + FD"
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
    "weighted_accuracy_emotion" = "Emotion Recognition (Accuracy)", 
    "weighted_accuracy_phoneme" = "Phoneme Recognition (Accuracy)", 
    "weighted_accuracy_speaker" = "Speaker Identification (Accuracy)"  
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
    "weighted_accuracy_emotion" = "Emotion Recognition (Accuracy)", 
    "weighted_accuracy_phoneme" = "Phoneme Recognition (Accuracy)", 
    "weighted_accuracy_speaker" = "Speaker Identification (Accuracy)"  
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
    "weighted_accuracy_emotion" = "Emotion Recognition (Accuracy)", 
    "weighted_accuracy_phoneme" = "Phoneme Recognition (Accuracy)", 
    "weighted_accuracy_speaker" = "Speaker Identification (Accuracy)"  
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

  #sup_speaker_phoneme_plot <- create_metric_scatterplot(
  #  model_data, 
  #  "supervised_speaker_identification", 
  #  "supervised_phoneme_recognition", 
  #  selected_models,
  #  file.path(scatter_2d_save_dir, "sup_speaker_phoneme_colorblind.png"),
  #  colorblind = TRUE
  #)
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
  sup_speaker_phoneme_emotion_plot <- create_metric_bubble_chart(
    model_data, 
    "weighted_accuracy_phoneme", 
    "weighted_accuracy_speaker", 
    "weighted_accuracy_emotion", 
    selected_models,
    file.path(scatter_3d_save_dir, "sup_speaker_phoneme_emotion_bubble_colorblind.png"),
    colorblind = TRUE,
    x_limits = c(0, 0.6),
    y_limits = c(0, 1)
  )

  sup_speaker_phoneme_disent_plot <- create_metric_bubble_chart(
    model_data, 
    "weighted_accuracy_phoneme", 
    "weighted_accuracy_emotion", 
    "disentanglement", 
    selected_models,
    file.path(scatter_3d_save_dir, "sup_phoneme_emotion_disent_bubble_colorblind.png"),
    colorblind = TRUE,
    x_limits = c(0, 0.6),
    y_limits = c(0, 0.6)
  )

  modularity_explicitness_robustness_plot <- create_metric_bubble_chart(
    model_data, 
    "modularity", 
    "explicitness", 
    "IRS", 
    selected_models,
    file.path(scatter_3d_save_dir, "modularity_explicitness_robustness_bubble_colorblind.png"),
    colorblind = TRUE,
    x_limits = c(0.6, 1.0),
    y_limits = c(0.0, 0.5)
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

bubble_legend_emotion <- create_legend(
  selected_models,
  size_metric = "weighted_accuracy_emotion",
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
  height = 3, 
  dpi = 600, 
  bg = "white"
)

ggsave(
  file.path(scatter_3d_save_dir, "bubble_legend_irs.png"), 
  bubble_legend_irs, 
  width = 11, 
  height = 3, 
  dpi = 600, 
  bg = "white"
)

ggsave(
  file.path(scatter_3d_save_dir, "bubble_legend_disent.png"), 
  bubble_legend_disent, 
  width = 11, 
  height = 3, 
  dpi = 600, 
  bg = "white"
)

ggsave(
  file.path(scatter_3d_save_dir, "bubble_legend_emotion.png"), 
  bubble_legend_emotion, 
  width = 11, 
  height = 3, 
  dpi = 600, 
  bg = "white"
)