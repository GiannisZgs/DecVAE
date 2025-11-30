#' SI Figure 4: this script generates figure SI_fig_4a. Demonstrates the generated 
#' simulated signals of the SimVowels dataset in the time domain - the true generative model.

library(jsonlite)
library(R.utils)
library(vscDebugger)
library(tidyverse)
library(MetBrewer)
library(patchwork)

# Style and font parameters
plot_font_family <- "Arial"
plot_background_color <- "#F4F5F1"
plot_text_color <- "black"

# Axis titles
axis_title_size <- 35
axis_title_face <- "plain"
axis_title_margin <- margin(t = 15, r = 15, b = 15, l = 15)

# Axis tick labels
axis_text_y_size <- 30
axis_text_y_face <- "plain"
axis_text_x_size <- 30
axis_text_x_face <- "plain"
axis_text_color <- "black"

# Plot title
plot_title_size <- 0
plot_title_face <- "bold"
plot_title_hjust <- 0.5
plot_title_vjust <- 0.5
plot_title_margin <- margin(b = 20)

# Facet text
facet_text_size <- 40
facet_text_face <- "plain"

# Plot margins and spacing
plot_margin <- margin(10, 10, 10, 10)
panel_spacing <- unit(2, "lines")

# Load data
data_file <- file.path('..','data','sim_vowels_figures.json.gz')
json_file_path <- gsub(".gz$", "", data_file)
# Set and create directory and filepaths to save
save_dir <-  file.path('..','supplementary_figures','SI_fig_4','generative_model','time_domain')
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

vowel_duration <- 0.025 # Duration of each vowel sample in seconds
vowel_dur_str <- as.character(vowel_duration*1000)
sampling_rate <- 16000  # Sampling rate in Hz
cycles_to_keep <- 4 # Number of cycles to keep for visualizations
frame_start <- 1
sample <- "_1.105"

if (!file.exists(json_file_path)) {
  # Decompress the .json.gz file
  json_file_path <- gunzip(data_file, remove = FALSE)
}

# Load the JSON data
data <- fromJSON(json_file_path)

keys <- names(data)
subkeys <- names(data[[keys[1]]])

# Plot all vowels for a random speaker

data_td_x <- data.frame()
data_td_f1 <- data.frame()
data_td_f2 <- data.frame()
data_td_f3 <- data.frame()
samples_to_keep <- cycles_to_keep*vowel_duration*sampling_rate
for (vowel_key in grep(sample, names(data), value = TRUE)) {
  vowel_type <- sub("_.*", "", vowel_key)  # Extract vowel type (a, e, i, etc.)
  x <- data[[vowel_key]][["time_domain_signal"]][frame_start:(frame_start+samples_to_keep)] # nolint
  f1 <- data[[vowel_key]][["formant_1_waves"]][frame_start:(frame_start+samples_to_keep)] # nolint
  f2 <- data[[vowel_key]][["formant_2_waves"]][frame_start:(frame_start+samples_to_keep)] # nolint
  f3 <- data[[vowel_key]][["formant_3_waves"]][frame_start:(frame_start+samples_to_keep)] # nolint
  df_x <- data.frame(
    x = x,
    vowel = vowel_type
  )
  df_f1 <- data.frame(
    f1 = f1,
    vowel = vowel_type
  )
  df_f2 <- data.frame(
    f2 = f2,
    vowel = vowel_type
  )
  df_f3 <- data.frame(
    f3 = f3,
    vowel = vowel_type
  )
  data_td_x <- rbind(data_td_x, df_x)
  data_td_f1 <- rbind(data_td_f1, df_f1)
  data_td_f2 <- rbind(data_td_f2, df_f2)
  data_td_f3 <- rbind(data_td_f3, df_f3)
}

base_date <- as.Date("2025-01-01")

data_td_x_long <- data_td_x %>%
  rename(value = x) %>%
  group_by(vowel) %>%
  mutate(
    sample_id = row_number(),
    time = (sample_id - 1) / 16000,  
    time = base_date + time 
  ) %>%
  ungroup() %>%
  select(-sample_id) 

data_td_f1_long <- data_td_f1 %>%
  rename(value = f1) %>%
  group_by(vowel) %>%
  mutate(
    sample_id = row_number(),
    time = (sample_id - 1) / 16000,  
    time = base_date + time 
  ) %>%
  ungroup() %>%
  select(-sample_id) 

data_td_f2_long <- data_td_f2 %>%
  rename(value = f2) %>%
  group_by(vowel) %>%
  mutate(
    sample_id = row_number(),
    time = (sample_id - 1) / 16000,  
    time = base_date + time 
  ) %>%
  ungroup() %>%
  select(-sample_id) 

data_td_f3_long <- data_td_f3 %>%
  rename(value = f3) %>%
  group_by(vowel) %>%
  mutate(
    sample_id = row_number(),
    time = (sample_id - 1) / 16000,  
    time = base_date + time 
  ) %>%
  ungroup() %>%
  select(-sample_id) 

# Define a base theme for all plots - with font passed directly to element_text
base_theme <- theme_minimal() +
  theme(
    text = element_text(family = plot_font_family),
    plot.background = element_rect(color = plot_background_color, fill = plot_background_color),
    panel.background = element_rect(fill = NA, color = NA),
    plot.margin = plot_margin,
    panel.spacing = panel_spacing,
    strip.text = element_text(
      size = facet_text_size, 
      face = facet_text_face, 
      color = plot_text_color,
      family = plot_font_family
    ),
    plot.title = element_text(
      size = plot_title_size, 
      face = plot_title_face, 
      hjust = plot_title_hjust, 
      vjust = plot_title_vjust,
      margin = plot_title_margin,
      family = plot_font_family
    ),
    axis.text.y = element_text(
      size = axis_text_y_size, 
      face = axis_text_y_face, 
      color = axis_text_color,
      family = plot_font_family
    ),
    axis.text.x = element_text(
      size = axis_text_x_size, 
      face = axis_text_x_face, 
      color = axis_text_color,
      family = plot_font_family
    ),
    axis.title = element_text(
      size = axis_title_size, 
      face = axis_title_face, 
      color = plot_text_color,
      margin = axis_title_margin,
      family = plot_font_family
    )
  )

create_signal_plot <- function(data, ymin, ymax, show_strip_text = FALSE, signal_label = "") {
  ggplot(data) +
    geom_hline(yintercept = 0, linetype="solid", size=.25) +
    geom_line(aes(x=time, y=value, color=vowel)) +
    scale_color_met_d(name="Redon") +
    scale_y_continuous(
      limits = c(ymin, ymax), 
      labels = scales::number_format(accuracy = 0.1)
    ) +
    facet_wrap(~ factor(vowel, levels=c('a','e','I','aw','u')), nrow = 1) +
    coord_cartesian(clip = "off") +
    labs(y = signal_label) +
    base_theme +
    theme(
      axis.title.x = element_blank(),
      axis.text.x = element_blank(),
      strip.text.x = if(show_strip_text) 
                      element_text(face = facet_text_face, size = facet_text_size, family = plot_font_family) 
                     else 
                      element_blank(),
      legend.position = "none",
      axis.title.y = element_text(
        size = axis_title_size, 
        face = axis_title_face, 
        angle = 90,
        vjust = 0.5,
        family = plot_font_family
      )
    )
}

p1 <- create_signal_plot(data_td_x_long, -1, 1, show_strip_text = TRUE, signal_label = "X")
p2 <- create_signal_plot(data_td_f1_long, -1, 1, show_strip_text = FALSE, signal_label = "Formant 1")
p3 <- create_signal_plot(data_td_f2_long, -1, 1, show_strip_text = FALSE, signal_label = "Formant 2")
p4 <- create_signal_plot(data_td_f3_long, -1, 1, show_strip_text = FALSE, signal_label = "Formant 3")

# Combine plots vertically
final_plot <- p1 / p2 / p3 / p4 +
  plot_layout(heights = c(1, 1, 1, 1))

print(final_plot)

ggsave(
  file.path(save_dir, paste0("vowel_generative_model_time_domain_sample_", sample, "_", vowel_dur_str, "ms.png")), 
  final_plot, 
  width = 20, 
  height = 12,
  dpi = 600,
  bg = plot_background_color
)