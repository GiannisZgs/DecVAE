library(dplyr)
library(ggplot2)
library(stringr) 
library(tidyr) 
library(patchwork) 
library(viridis) 

# Set and create directory and filepaths to save
save_dir <-  file.path('..','figures','model_performance_vowels', 'circular_barplots')
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

# Style and font parameters
plot_font_family <- "Arial"
plot_background_color <- "white"
plot_text_color <- "black"

#color palette
palette <- "plasma"
yellow_block_threshold <- 0.8

# Circular barplot specific settings
axis_text_size <- 20
plot_title_size <- 45
plot_title_face <- "plain"
plot_title_hjust <- 0.5
plot_title_vjust <- 0.5

# Legend elements
legend_text_size <- 28
legend_title_size <- 35
legend_text_face <- "plain"
legend_title_face <- "plain"
legend_key_size <- 7  # in cm

# Geom elements
bar_alpha <- 0.6
line_alpha <- 0.6
point_size_regular <- 5
text_size_regular <- 15
text_face <- "plain"

metrics <- c("mutual_info", "gaussian_corr_norm", "disentanglement", "completeness", 
             "informativeness", "explicitness", "modularity","IRS", 
             "supervised_phoneme_recognition", "supervised_speaker_identification","unsupervised_phoneme_recognition",
             "unsupervised_speaker_identification")

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

models_to_plot <- c("PCA_wav", "PCA_mel", "ICA_wav", "ICA_mel", "cAE_wav", "cVAE_wav","c_betaVAE_wav",
         "fcAE_wav", "fcVAE_wav","fc_betaVAE_wav","fcAE_mel", "fcVAE_mel","fc_betaVAE_mel",
         "MDecAE_FD_mel", "MDecAE_EWT_mel","MDecVAE_FD_mel", "MDecVAE_EWT_mel",
         "betaMDecVAE_FD_mel", "betaMDecVAE_EWT_mel","ideal_model")
         
#  fDecAE + FD mel = c(-,-,-,-,-,-,-,-,-,-,-,-),
#  fDecAE + EWT mel = c(-,-,-,-,-,-,-,-,-,-,-,-),
#  sDecAE + FD mel = c(-,-,-,-,-,-,-,-,-,-,-,-),
#  sDecAE + EWT mel = c(-,-,-,-,-,-,-,-,-,-,-,-),
#  fDecVAE + FD mel = c(-,-,-,-,-,-,-,-,-,-,-,-),
#  fDecVAE + EWT mel = c(-,-,-,-,-,-,-,-,-,-,-,-),
#  sDecVAE + FD mel = c(-,-,-,-,-,-,-,-,-,-,-,-),
#  sDecVAE + EWT mel = c(-,-,-,-,-,-,-,-,-,-,-,-),
#  fDec-beta-VAE + FD mel = c(-,-,-,-,-,-,-,-,-,-,-,-),
#  fDec-beta-VAE + EWT mel = c(-,-,-,-,-,-,-,-,-,-,-,-),
#  sDec-beta-VAE + FD mel = c(-,-,-,-,-,-,-,-,-,-,-,-),
#  sDec-beta-VAE + EWT mel = c(-,-,-,-,-,-,-,-,-,-,-,-),

model_data_long <- model_data %>%
  pivot_longer(
    cols = -metric,
    names_to = "model",
    values_to = "value"
  )

create_model_plot <- function(data, model_name, color_palette) {
  
  # Filter data for specific model
  model_df <- data %>% filter(model == model_name)
  
  # Adjust metric names for better display
  model_df <- model_df %>%
    mutate(
      # Invert mutual info and gaussian_corr_norm values so higher is better
      value = case_when(
        metric == "mutual_info" ~ 1 - value,
        metric == "gaussian_corr_norm" ~ 1 - value,
        TRUE ~ value
      ),
      # Cap values between 0 and 1
      value_display = pmax(0, pmin(value, 1)),
      short_metric = case_when(
        metric == "mutual_info" ~ "Mutual Information (inv.)",
        metric == "gaussian_corr_norm" ~ "Gaussianity (inv.)",
        metric == "disentanglement" ~ "Disentanglement",
        metric == "completeness" ~ "Completeness",
        metric == "informativeness" ~ "Informativeness",
        metric == "explicitness" ~ "Explicitness",
        metric == "modularity" ~ "Modularity",
        metric == "IRS" ~ "Robustness",
        metric == "supervised_phoneme_recognition" ~ "Phoneme (Sup.)",
        metric == "supervised_speaker_identification" ~ "Speaker (Sup.)",
        metric == "unsupervised_phoneme_recognition" ~ "Phoneme (Unsup.)",
        metric == "unsupervised_speaker_identification" ~ "Speaker (Unsup.)",
        TRUE ~ metric
      ),
      short_metric = factor(short_metric, 
                           levels = c("Mutual Information (inv.)", "Gaussianity (inv.)", "Disentanglement", "Completeness", "Informativeness", 
                                     "Explicitness", "Modularity", "Robustness", "Phoneme (Sup.)", "Speaker (Sup.)", 
                                     "Phoneme (Unsup.)", "Speaker (Unsup.)"))
    )
    #                           levels = c("MI", "GCN", "Disentang.", "Complet.", "Inform.", 
    #                                  "Explic.", "Modul.", "IRS", "SPR", "SSI", 
    #                                 "UPR", "USI"))  

  display_name <- if (!is.null(model_display_names[model_name])) {
    model_display_names[model_name]
  } else {
    # Fallback formatting if model isn't in our mapping
    formatted_name <- gsub("_", " ", model_name)
    formatted_name <- gsub("\\+", " + ", formatted_name)
    formatted_name <- gsub("beta", "β", formatted_name)  # Replace beta with β
    formatted_name <- gsub("  ", " ", formatted_name)
    formatted_name
  }


  grid_lines <- data.frame(
    y = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),
    label = c("0", "0.2", "0.4", "0.6", "0.8", "1.0")
  )

  # Create the plot
  p <- ggplot(model_df) +
    # Make custom panel grid
    geom_hline(
      aes(yintercept = y), 
      data = grid_lines,
      color = "lightgrey"
    ) + 
    geom_text(
      aes(x = 1, y = y, label = label),
      data = grid_lines,
      color = "gray20",
      size = 3.5,
      hjust = 1.1,
      family = plot_font_family,
      inherit.aes = FALSE
    ) +
    # Add bars for metric values
    geom_col(
      aes(
        x = short_metric,
        y = value_display,
        fill = value_display
      ),
      position = "dodge2",
      show.legend = TRUE,
      alpha = bar_alpha
    ) +
    # Lollipop shaft
    geom_segment(
      aes(
        x = short_metric,
        y = 0,
        xend = short_metric,
        yend = 1.0
      ),
      linetype = "dashed",
      color = "gray30",
      alpha = line_alpha
    ) + 
    # Improve label spacing and prevent overlap at top - use horizontal text with offset
    coord_polar(clip = "off", start = 0.2) +  # Adjust start angle to avoid crowding at top
    # Scale y axis so bars don't start in center
    scale_y_continuous(
      limits = c(-0.5, 1.2),
      expand = c(0, 0),
      breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ) + 
    # Remove legend from the main plot as we create it separately
    scale_fill_viridis_c(
      "Performance Score",
      option = palette,
      direction = 1,
      limits = c(0, 1),
      breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),
      labels = c("0", "0.2", "0.4", "0.6", "0.8", "1.0"),
      end = yellow_block_threshold,
      guide = "none"  # Remove the legend
    ) +
    scale_x_discrete() + 
    # Make the guide for the fill discrete
    guides(
      fill = guide_colorbar(
        barwidth = 15, barheight = 0.5, 
        title.position = "top", 
        title.hjust = 0.5
      )
    ) +
    # Add labels
    labs(
      title = display_name
    ) +
    # Customize theme
    theme_minimal() +
    theme(
      # Remove axis ticks and text
      axis.title = element_blank(),
      axis.ticks = element_blank(),
      axis.text.y = element_blank(),
      axis.text.x = element_text(
          color = plot_text_color, 
          size = axis_text_size,    
          face = text_face,
          family = plot_font_family,
          margin = margin(t=10),
          angle = 0,              # Horizontal text
          hjust = 0.5,            # Center alignment
          vjust = 0.6            # Adjust vertical position
      ),
      # Remove legend from the plot
      legend.position = "none",
      # Set default color and font family for the text
      text = element_text(color = plot_text_color, family = plot_font_family),
      # Customize the text in the title and subtitle
      plot.title = element_text(
          face = plot_title_face, 
          size = plot_title_size, 
          hjust = plot_title_hjust, 
          family = plot_font_family
      ),
      plot.subtitle = element_text(
          size = plot_title_size-8, 
          hjust = plot_title_hjust, 
          family = plot_font_family
      ),
      # Make the background white and remove extra grid lines
      panel.background = element_rect(fill = plot_background_color, color = plot_background_color),
      plot.background = element_rect(fill = plot_background_color, color = NA),
      panel.grid = element_blank(),
      panel.grid.major.x = element_blank(),
      plot.margin = margin(20, 40, 20, 30),
      # Style the legend
      legend.text = element_text(size = legend_text_size, face = legend_text_face, family = plot_font_family),
      legend.title = element_text(size = legend_title_size, face = legend_title_face, family = plot_font_family),
      legend.key.size = unit(legend_key_size, "cm")
    )
  
  return(p)
}

# Create a function to extract and save just the legend
extract_legend <- function(plot_object, save_path) {
  # Use cowplot to extract the legend
  legend <- cowplot::get_legend(
    plot_object + 
    theme(
      legend.position = "bottom",
      legend.box.margin = margin(20, 0, 20, 0),
      legend.text = element_text(
          size = legend_text_size, 
          face = legend_text_face, 
          family = plot_font_family
      ),
      legend.title = element_text(
          size = legend_title_size, 
          face = legend_title_face, 
          family = plot_font_family
      ),
      legend.key.size = unit(legend_key_size/2, "cm")
    )
  )
  
  # Create a blank canvas with just the legend
  legend_plot <- cowplot::ggdraw() + 
    cowplot::draw_grob(legend)
  
  # Save the legend
  ggsave(
    save_path,
    legend_plot,
    width = 10, 
    height = 3,
    dpi = 600, 
    bg = plot_background_color
  )
  
  return(legend_plot)
}

model_names <- colnames(model_data)[-1]


# Generate individual plots and their legends
for (model_name in models_to_plot) {
  # Use the palette parameter consistently
  plot <- create_model_plot(model_data_long, model_name, palette)
  
  # Clean up model name for file naming
  file_name <- gsub(" ", "_", model_name)
  file_name <- gsub("\\+", "plus", file_name)
  file_name <- gsub("-", "_", file_name)
  
  # Create paths for both plot and legend
  plot_path <- file.path(save_dir, paste0(file_name, "_performance.png"))
  legend_path <- file.path(save_dir, paste0(file_name, "_legend.png"))
  
  # Save plot without legend
  ggsave(
    plot_path,
    plot, width = 12, height = 9, dpi = 600, bg = plot_background_color
  )
}

# We're already saving the legend for the last plot after the loop,
# but let's make sure it's always saved for a specific model
extract_legend(create_model_plot(model_data_long, "ideal_model", palette), 
               file.path(save_dir, "reference_legend.png"))

