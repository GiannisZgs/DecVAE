library(dplyr)
library(ggplot2)
library(stringr) 
library(tidyr) 
library(fmsb)  # For radar charts
library(RColorBrewer)

# Set and create directory and filepaths to save
save_dir <-  file.path('..','figures','model_performance_vowels', 'radar_charts')
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

# Style and font parameters
plot_font_family <- "Arial"
plot_background_color <- "white"
plot_text_color <- "black"

#color palette
palette <- "plasma"
palette_w_yellows <- "Set1"
yellow_block_threshold <- 0.8

# Radar chart specific settings
radar_axis_text_size <- 25
radar_label_size <- 25
radar_legend_text_size <- 14
radar_line_width <- 3
radar_axis_label_angle <- 45  # New parameter to control axis label angle (0 = horizontal)

# Selected models for comparison - adjust as needed
selected_models_mel <- c(
  "PCA_mel", 
  "fc_betaVAE_mel", 
  "betaMDecVAE_EWT_mel",
  "MDecAE_FD_mel"
)
selected_models_wav <- c(
  "PCA_wav", 
  "ICA_wav",
  "fc_betaVAE_wav", 
  "fcVAE_wav"
)

metrics <- c("mutual_info", "gaussian_corr_norm", "disentanglement", "completeness", 
             "informativeness", "explicitness", "modularity","IRS", 
             "supervised_phoneme_recognition", "supervised_speaker_identification",
             "unsupervised_phoneme_recognition", "unsupervised_speaker_identification")

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
  "PCA_mel" = "PCA (Mel)",
  "ICA_wav" = "ICA (Waveform)",
  "ICA_mel" = "ICA (Mel)",
  "cAE_wav" = "Conv AE (Waveform)",
  "cVAE_wav" = "Conv VAE (Waveform)",
  "c_betaVAE_wav" = "Conv β-VAE (Waveform)",
  "fcAE_wav" = "FC AE (Waveform)",
  "fcVAE_wav" = "FC VAE (Waveform)",
  "fc_betaVAE_wav" = "FC β-VAE (Waveform)",
  "fcAE_mel" = "FC AE (Mel)",
  "fcVAE_mel" = "FC VAE (Mel)",
  "fc_betaVAE_mel" = "FC β-VAE (Mel)",
  "MDecAE_FD_mel" = "DecAE + FD (Mel)",
  "MDecAE_EWT_mel" = "DecAE + EWT (Mel)",
  "MDecVAE_FD_mel" = "DecVAE + FD (Mel)",
  "MDecVAE_EWT_mel" = "DecVAE + EWT (Mel)",
  "betaMDecVAE_FD_mel" = "β-DecVAE + FD (Mel)",
  "betaMDecVAE_EWT_mel" = "β-DecVAE + EWT (Mel)",
  "ideal_model" = "Ideal Model"
)

# Normalize values to 0-1
# Fix any values above 1.0 to 1.0
model_data[, -1] <- apply(model_data[, -1], 2, function(x) {
  ifelse(x > 1, 1, x)
})

# Fix values for gaussian_corr_norm (row 2) - lower is better
# Invert the values (1 - value) so higher becomes better
gcn_row <- which(model_data$metric == "gaussian_corr_norm")
model_data[gcn_row, -1] <- 1 - model_data[gcn_row, -1]

# Same for mutual_info (row 1) - lower is better
mi_row <- which(model_data$metric == "mutual_info")
model_data[mi_row, -1] <- 1 - model_data[mi_row, -1]

# Prepare data for radar chart with prettier metric names
radar_data <- model_data
radar_data <- radar_data[rev(1:nrow(radar_data)),]

radar_data$metric <- c(
  "Speaker (Unsup.)",
  "Phoneme (Unsup.)",
  "Speaker (Sup.)",
  "Phoneme (Sup.)",
  "Robustness",
  "Modularity",
  "Explicitness",
  "Informativeness",
  "Completeness",
  "Disentanglement",
  "Gaussianity (inv.)",
  "Mutual Information (inv.)"
)


# Function to create a single radar chart with multiple models
create_radar_chart <- function(data, selected_models, save_path = NULL, create_separate_legend = FALSE) {
  # Convert to wide format and set rownames to metric names
  radar_wide <- as.data.frame(t(data[, c("metric", selected_models)]))
  colnames(radar_wide) <- radar_wide[1, ]
  radar_wide <- radar_wide[-1, ]
  
  # Convert to numeric
  radar_wide[] <- lapply(radar_wide, as.numeric)
  
  # Add max and min values required by fmsb
  radar_wide <- rbind(rep(1, ncol(radar_wide)), rep(0, ncol(radar_wide)), radar_wide)
  
  # Set up colors with transparency
  n_models <- length(selected_models)
  
  # Use viridis color palette (plasma) for colorblind friendliness
  if (yellow_block_threshold <= 0 || yellow_block_threshold > 1) {
    yellow_block_threshold <- 0.8  # Default if invalid
  }
  
  # Generate colors based on specified palette
  if (requireNamespace("viridis", quietly = TRUE)) {
    colors <- switch(palette,
                    "plasma" = viridis::plasma(n_models, end = yellow_block_threshold),
                    "viridis" = viridis::viridis(n_models, end = yellow_block_threshold),
                    "magma" = viridis::magma(n_models, end = yellow_block_threshold),
                    "inferno" = viridis::inferno(n_models, end = yellow_block_threshold),
                    "cividis" = viridis::cividis(n_models, end = yellow_block_threshold),
                    # Default to plasma if palette is not recognized
                    viridis::plasma(n_models, end = yellow_block_threshold))
  } else {
    # Fallback to brewer if viridis not available
    colors <- brewer.pal(min(n_models, 8), palette_w_yellows)
    if (n_models > 8) {
      more_colors <- colorRampPalette(colors)(n_models)
      colors <- more_colors
    }
  }
  
  # Add transparency (lower alpha value means more transparent)
  colors_transparent <- sapply(colors, function(x) {
    rgb_values <- col2rgb(x) / 255
    rgb(rgb_values[1], rgb_values[2], rgb_values[3], alpha = 0.3)
  })
  
  # Save the plot if path is provided
  if (!is.null(save_path)) {
    # First create the directory if it doesn't exist
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    
    # Save directly to PNG with larger dimensions for more space
    png(save_path, width = 2400, height = 2000, res = 300, bg = plot_background_color, family = plot_font_family)
    
    # Reset plotting parameters and use more generous margins
    par(
      mar = c(6, 6, 6, 6),  # Increase margins all around to provide more space for labels
      bg = plot_background_color,
      family = plot_font_family,
      new = FALSE,
      cex.lab = 1.2
    )
    
    # Create the radar chart with adjusted parameters to ensure labels stay outside chart area
    radarchart(
        radar_wide,
        pcol = colors,
        pfcol = colors_transparent,
        plwd = radar_line_width,
        plty = 1,
        cglcol = "grey",
        cglty = 1,
        cglwd = 0.8,
        axistype = 1,
        axislabcol = plot_text_color,
        caxislabels = c("0", "0.2", "0.4", "0.6", "0.8", "1.0"),
        maxmin = TRUE,
        title = "",
        vlcex = radar_label_size/25,     # Make labels slightly smaller
        calcex = radar_axis_text_size/20, 
        palcex = radar_axis_text_size/20, 
        pty = 32,                         
        plabcex = 0.9,
        seg = 5,
        centerzero = FALSE,               # Don't center at zero
        na.itp = FALSE,                   # No interpolation of NA values
        pdensity = -1,                    # Set fill patterns
        plabel.dist = 1.2                 # Increase distance of labels from the plot
    )
    
    # Only add a legend to the main plot if we're not creating a separate legend
    if (!create_separate_legend) {
      # Add a legend in the top right - vertical orientation
      legend(
          "topright",
          legend = sapply(selected_models, function(model) {
            if (!is.null(model_display_names[model])) {
              model_display_names[model]
            } else {
              gsub("_", " ", model)
            }
          }),
          col = colors,
          lty = 1,
          lwd = radar_line_width,
          bty = "n",
          cex = radar_legend_text_size/15,
          pt.cex = 2,
          text.col = plot_text_color,
          horiz = FALSE,             # Vertical orientation
          inset = c(0.02, 0.02),     # Small inset from the edges
          y.intersp = 1.2            # Spacing between legend items
      )
    }
    
    # Close the PNG device
    dev.off()
    
    # Optionally create a separate legend file
    if (create_separate_legend) {
      legend_path <- sub("\\.png$", "_legend.png", save_path)
      
      # Create a blank plot with just the legend - with fixed dimensions
      png(legend_path, width = 800, height = 600, res = 300, bg = plot_background_color, family = plot_font_family)
      
      # Create a plot with empty plotting region but proper margins
      par(mar = c(0, 0, 0, 0), bg = plot_background_color, family = plot_font_family)
      plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n", 
           xlab = "", ylab = "", xlim = c(0, 1), ylim = c(0, 1))
      
      # Add only the legend with improved positioning and scaling
      legend(
        "center",
        legend = sapply(selected_models, function(model) {
          if (!is.null(model_display_names[model])) {
            model_display_names[model]
          } else {
            gsub("_", " ", model)
          }
        }),
        col = colors,
        lty = 1,
        lwd = radar_line_width * 1.5,
        bty = "n",
        cex = radar_legend_text_size/15,                   # Better default scaling
        pt.cex = 2,                  # Larger points
        text.col = plot_text_color,
        horiz = FALSE,               # Vertical orientation
        y.intersp = 1.5              # More space between items
      )
      
      dev.off()
    }
  }
  
  # For interactive plotting, also adjust margins
  if (is.null(save_path)) {
    # Use generous margins for screen display too
    par(mar = c(3, 3, 4, 3))
  }
  
  return(invisible(NULL))
}

# Create the radar charts with optional separate legends
create_radar_chart(
  radar_data, 
  selected_models_mel,
  file.path(save_dir, "model_comparison_radar_mel.png"),
  create_separate_legend = TRUE
)

create_radar_chart(
  radar_data, 
  selected_models_wav,
  file.path(save_dir, "model_comparison_radar_waveform.png"),
  create_separate_legend = TRUE
)



