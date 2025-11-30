#' SI Figure 4: this script generates figure SI_fig_4b. Demonstrates the generated 
#' simulated signals of the SimVowels dataset in the mel-frequency domain
#' - the true generative model.

library(jsonlite)
library(R.utils)
library(vscDebugger)
library(tidyverse)
library(cowplot)

# Style and font parameters
plot_font_family <- "Arial"
plot_background_color <- "white"
plot_text_color <- "black"

# Axis titles
axis_title_size_single_vowel <- 55
axis_title_size <- 90
axis_title_face <- "plain"

# Axis tick labels
axis_text_size_x_vowel <- 0
axis_text_size_y_vowel <- 40
axis_text_size_x <- 0
axis_text_size_y <- 75 #also controls strip text size
axis_text_face <- "plain"

# Plot title
plot_title_size <- 0
plot_title_size_single_vowel <- 50
plot_title_face <- "plain"
plot_title_hjust <- 0.5
plot_title_vjust <- 0.5

# Legend elements
legend_text_size <- 95
legend_text_size_single_vowel <- 50
legend_title_size <- 100
legend_title_size_single_vowel <- 50
legend_title_face <- "plain"
legend_key_width <- 10  # in cm
legend_key_height <- 3    # in cm
legend_key_width_single_vowel <- 5  # in cm
legend_key_height_single_vowel <- 2 # in cm

# Load data
data_file <- file.path('..','data','sim_vowels_figures.json.gz') 
json_file_path <- gsub(".gz$", "", data_file)

# Set and create directory and filepaths to save
save_dir <-  file.path('..','supplementary_figures','SI_fig_4','generative_model','mel_features')
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}
create_legend <- FALSE
sampling_rate <- 16000  # Sampling rate in Hz
frame_to_plot <- 2 # second frame corresponds to other plots 
sample <- "_0.705"
start_ind <- 320
frame_start <- 2
frames_to_keep <- 1 #mel spectrogram samples to keep
vowel <- "a"

if (!file.exists(json_file_path)) {
  # Decompress the .json.gz file
  json_file_path <- gunzip(json_gz_file_path, remove = FALSE)
}

data <- fromJSON(json_file_path)


data_x <- data.frame()
data_f1 <- data.frame()
data_f2 <- data.frame()
data_f3 <- data.frame()

for (vowel_key in names(data)) {
  vowel_type <- sub("_.*", "", vowel_key)  # Extract vowel type (a, e, i, etc.)
  speaker_id <- sub(".*_", "", vowel_key)  # Extract speaker id
  mel_x <- data[[vowel_key]][["mel_filterbank_energies"]][,frame_start:(frame_start+frames_to_keep-1)] # nolint
  mel_f1 <- data[[vowel_key]][["mel_filterbank_energies_f1"]][,frame_start:(frame_start+frames_to_keep-1)] # nolint
  mel_f2 <- data[[vowel_key]][["mel_filterbank_energies_f2"]][,frame_start:(frame_start+frames_to_keep-1)] # nolint
  mel_f3 <- data[[vowel_key]][["mel_filterbank_energies_f3"]][,frame_start:(frame_start+frames_to_keep-1)] # nolint
  df_x <- data.frame(
    mel_x = mel_x,
    vowel = vowel_type,
    speaker = speaker_id
  )
  df_f1 <- data.frame(
    mel_f1 = mel_f1,
    vowel = vowel_type,
    speaker = speaker_id
  )
  df_f2 <- data.frame(
    mel_f2 = mel_f2,
    vowel = vowel_type,
    speaker = speaker_id
  )
  df_f3 <- data.frame(
    mel_f3 = mel_f3,
    vowel = vowel_type,
    speaker = speaker_id
  )
  data_x <- rbind(data_x, df_x)
  data_f1 <- rbind(data_f1, df_f1)
  data_f2 <- rbind(data_f2, df_f2)
  data_f3 <- rbind(data_f3, df_f3)
}

base_date <- as.Date("2025-01-01")

data_x <- data_x %>%
  rename(value = mel_x) %>%
  group_by(vowel, speaker) %>%
  mutate(time = row_number()) %>%  # Add time index
  ungroup()

data_f1 <- data_f1 %>%
  rename(value = mel_f1) %>%
  group_by(vowel, speaker) %>%
  mutate(time = row_number()) %>%  # Add time index
  ungroup()

data_f2 <- data_f2 %>%
  rename(value = mel_f2) %>%
  group_by(vowel, speaker) %>%
  mutate(time = row_number()) %>%  # Add time index
  ungroup()

data_f3 <- data_f3 %>%
  rename(value = mel_f3) %>%
  group_by(vowel, speaker) %>%
  mutate(time = row_number()) %>%  # Add time index
  ungroup()

create_mel_spectrogram <- function(data, title_suffix = "", save_suffix = "", single_vowel = NULL, create_legend = TRUE) {
  # Filter for single vowel if specified
  if (!is.null(single_vowel)) {
    data <- data %>% filter(vowel == single_vowel)
    plot_title <- paste("Mel coefficients of 60 speakers - Vowel ", toupper(single_vowel))
    text_size <- legend_text_size_single_vowel
    title_size <- plot_title_size_single_vowel
    ax_title_size <- axis_title_size_single_vowel
    x_tick_size <- axis_text_size_x_vowel
    y_tick_size <- axis_text_size_y_vowel
    key_width <- legend_key_width_single_vowel
    key_height <- legend_key_height_single_vowel
  } else {
    plot_title <- paste("Mel coefficients of 60 speakers by vowel", title_suffix)
    text_size <- legend_text_size
    title_size <- plot_title_size
    ax_title_size <- axis_title_size
    y_tick_size <- axis_text_size_y
    x_tick_size <- axis_text_size_x
    key_width <- legend_key_width
    key_height <- legend_key_height
  }
  
  value_range <- range(data$value, na.rm = TRUE)
  value_range[2] <- value_range[2] + (value_range[2] - value_range[1]) * 0.001

  p <- ggplot(data, aes(x = speaker, y = time, fill = value)) +
    geom_tile() +
    scale_fill_viridis(
      option = "plasma",
      name = "Mel Energy (dB)",
      limits = value_range,
      oob = scales::squish,
      breaks = scales::pretty_breaks(n = 8),
      guide = guide_colorbar(
        title.position = "top"
      )
    ) +
    facet_wrap(~vowel, ncol = 1, scales = "free_y", strip.position = "right") +
    scale_x_discrete(breaks = unique(data$speaker),
                    labels = 1:60) +
    theme_minimal() +
    labs(
      x = "Speaker vocal tract factor - 0.7 → 1.3",
      y = "Mel Filterbank Index",
      title = plot_title
    ) +
    theme(
      panel.grid = element_blank(),
      legend.position = if(create_legend) "bottom" else "none",
      legend.direction = "horizontal",
      legend.key.width = unit(key_width, "cm"),
      legend.key.height = unit(key_height, "cm"),
      legend.text = element_text(size = text_size, family = plot_font_family),
      legend.title = element_text(size = text_size, family = plot_font_family),
      legend.margin = margin(l = 40, unit = "pt"), 
      legend.spacing.x = unit(1, "cm"),  
      legend.box.spacing = unit(1, "cm"),
      strip.text.y = if (!is.null(single_vowel)) element_blank() 
                     else element_text(size = text_size, face = "plain", angle = 0, family = plot_font_family),
      strip.placement = "outside",
      axis.text.x = element_text(size = x_tick_size, angle = 45, hjust = 1, family = plot_font_family),
      axis.ticks.x = element_line(),
      axis.text.y = element_text(size = y_tick_size, family = plot_font_family),
      axis.title = element_text(size = ax_title_size, family = plot_font_family),
      plot.title = element_text(size = title_size, hjust = plot_title_hjust, family = plot_font_family),
      panel.spacing = unit(0.5, "cm"),
      plot.margin = margin(t = 30, r = 30, b = 20, l = 30, unit = "pt"),
      panel.background = element_rect(fill = plot_background_color, color = NA),
      plot.background = element_rect(fill = plot_background_color, color = NA)
    )
    
  # Save the plot
  save_name <- file.path(save_dir, paste0("vowels_speakers_mel_spec_heatmap",
    ifelse(!is.null(single_vowel), paste0("_vowel_", single_vowel), ""),
    save_suffix,
    ifelse(!create_legend, "_no_legend", ""),
    ".png"
  ))

  ggsave(save_name, 
        p, 
        width = ifelse(!is.null(single_vowel), 15, 35),
        height = ifelse(!is.null(single_vowel), 20, 30), 
        dpi = 600, 
        bg = "white"
  )
  
  return(p)
}

# Function to create a standalone legend
create_standalone_legend <- function(data, title_suffix = "", save_suffix = "", single_vowel = NULL) {
  # Determine which styling to use
  if (!is.null(single_vowel)) {
    data <- data %>% filter(vowel == single_vowel)
    text_size <- legend_text_size_single_vowel
    key_width <- legend_key_width_single_vowel
    key_height <- legend_key_height_single_vowel
    legend_width <- 10
    legend_height <- 20
  } else {
    text_size <- legend_text_size
    key_width <- legend_key_width
    key_height <- legend_key_height
    legend_width <- 15
    legend_height <- 30
  }
  
  value_range <- range(data$value, na.rm = TRUE)
  value_range[2] <- value_range[2] + (value_range[2] - value_range[1]) * 0.001
  
  # Create a dummy plot with the desired legend
  dummy_plot <- ggplot(data, aes(x = speaker, y = time, fill = value)) +
    geom_tile() +
    scale_fill_viridis(
      option = "plasma",
      name = "Mel Energy (dB)",
      limits = value_range,
      oob = scales::squish,
      breaks = scales::pretty_breaks(n = 8),
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        title.vjust = 4
      )
    ) +
    theme_minimal() +
    theme(
      legend.position = "right",
      legend.direction = "vertical",
      legend.key.width = unit(key_height, "cm"),  # Swapped width and height for vertical
      legend.key.height = unit(key_width, "cm"),
      legend.text = element_text(size = text_size, family = plot_font_family),
      legend.title = element_text(size = text_size, family = plot_font_family),
      legend.margin = margin(l = 40, unit = "pt"),
      legend.spacing.y = unit(1, "cm"),  # Changed from spacing.x to spacing.y for vertical
      legend.box.spacing = unit(1, "cm"),
      panel.background = element_rect(fill = plot_background_color, color = NA),
      plot.background = element_rect(fill = plot_background_color, color = NA)
    )
  
  # Extract legend
  legend <- get_legend(dummy_plot)
  
  # Create a blank plot with just the legend
  legend_plot <- ggdraw() + 
    draw_grob(legend)
  
  # Save the standalone legend
  legend_save_name <- file.path(save_dir, paste0("mel_spec_legend",
    ifelse(!is.null(single_vowel), paste0("_vowel_", single_vowel), ""),
    save_suffix,
    ".png"
  ))
  
  ggsave(legend_save_name, 
         legend_plot, 
         width = legend_width, 
         height = legend_height, 
         dpi = 600,
         bg = "white")
  
  return(legend_plot)
}

# Create plots for all vowels with or without legends
p_x <- create_mel_spectrogram(data_x, "", "_x", create_legend = create_legend)
p_f1 <- create_mel_spectrogram(data_f1, "- Formant 1", "_f1", create_legend = create_legend)
p_f2 <- create_mel_spectrogram(data_f2, "- Formant 2", "_f2", create_legend = create_legend)
p_f3 <- create_mel_spectrogram(data_f3, "- Formant 3", "_f3", create_legend = create_legend)

# Create standalone legends
if (!create_legend) {
  legend_x <- create_standalone_legend(data_x, "", "_x")
  legend_f1 <- create_standalone_legend(data_f1, "- Formant 1", "_f1")
  legend_f2 <- create_standalone_legend(data_f2, "- Formant 2", "_f2")
  legend_f3 <- create_standalone_legend(data_f3, "- Formant 3", "_f3")
}

# Create plots for individual vowels (with and without legends)
vowels <- c("a", "e", "aw", "u", "I")
for (v in vowels) {
  # With legends
  create_mel_spectrogram(data_x, "", "_x", single_vowel = v, create_legend = create_legend)
  create_mel_spectrogram(data_f1, "- Formant 1", "_f1", single_vowel = v, create_legend = create_legend)
  create_mel_spectrogram(data_f2, "- Formant 2", "_f2", single_vowel = v, create_legend = create_legend)
  create_mel_spectrogram(data_f3, "- Formant 3", "_f3", single_vowel = v, create_legend = create_legend)  
}

