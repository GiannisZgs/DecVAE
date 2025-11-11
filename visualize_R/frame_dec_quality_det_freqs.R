#!/usr/bin/env Rscript # nolint
library(jsonlite)
library(R.utils)
library(vscDebugger)
library(tidyverse)
library(janitor)
library(showtext)
library(MetBrewer)
library(scico)
library(ggtext)
library(patchwork)
library(gghighlight)
library(ggExtra)
library(tidyr)
library(viridis)
library(plotly)
library(ggridges)
library(grid)

plot_agg_formants <- FALSE
plot_comparisons <- TRUE
plot_violins <- FALSE


# Style and font parameters
plot_font_family <- "Arial"
plot_background_color <- "white"
plot_text_color <- "black"

showtext_auto(enable = TRUE)

# Axis titles
axis_title_size_agg_formants <- 130
axis_title_size_comparisons <- 55
axis_title_size_violins <- 170
axis_title_face <- "plain"

# Axis tick labels
axis_text_size_agg_formants <- 130
axis_text_size_comparisons <- 45
axis_text_size_violins <- 170
axis_text_face <- "plain"

# Plot title
plot_title_size_agg_formants <- 0
plot_title_size_comparisons <- 45
plot_title_size_violins <- 150
plot_title_face <- "plain"
plot_title_hjust <- 0.5
plot_title_vjust <- 0.5

# Legend elements
legend_text_face <- "plain"
legend_title_face <- "plain"

legend_position_agg_formants <- "right"
legend_title_size_agg_formants <- 95
legend_text_size_agg_formants <- 120
legend_key_size_agg_formants <- 4  # in cm

legend_position_comparisons <- "left"
legend_title_size_comparisons <- 60
legend_text_size_comparisons <- 60
legend_key_size_comparisons <- 4

legend_position_violins <- "right"
legend_title_size_violins <- 170 
legend_text_size_violins <- 170   
legend_key_size_violins <- 4   # Reduced from 2
legend_spacing_violins <- unit(0.5, "cm") 
legend_margin_violins <- margin(0, 0, 0, 0)  

# Geom elements
point_size_regular_agg_formants <- 6
point_size_large_agg_formants <- 10
text_size_regular_agg_formants <- 30
text_size_legend_agg_formants <- 16

point_size_regular_comparisons <- 6
point_size_large_comparisons <- 10
text_size_regular_comparisons <- 30
text_size_legend_comparisons <- 16

# Color schemes
gender_colors <- c("F" = "#D81B60", "M" = "#1E88E5")
dataset_shapes <- c("sim_vowels" = 16, "timit" = 17, "ground_truth" = 8)
dataset_labels <- c("SimVowels", "TIMIT", "Ground Truth")

# Margin settings for special plots
vowel_comparison_margin <- margin(4, 8, 2, 2, "cm")
legend_box_margin <- margin(0, 2, 0, 0, "cm")

#Set parameters and paths
decomposition <- 'filter'
decomposition_display <- 'fd'
NoC <- '3'
branch <- 'frame' #'frame' for Z, 'sequence' for S
feature <- 'det_freqs'

#Load directories
load_dir <- file.path("D:","data_for_figures_freq_analysis","decomposition_quality",decomposition)
data_file <- file.path(load_dir,paste0("dec_quality_",branch,"_NoC",NoC,"_",decomposition,"_",feature,".json"))

datasets <- c('sim_vowels', 'timit')
stat <- 'Median' #or 'Mean'
include_overlap <- TRUE
if (include_overlap) {
  overlap_str <- "with_overlap"
} else {
  overlap_str <- "without_overlap"
}

# Set and create directory and filepaths to save
parent_save_dir <-  file.path('..','figures','decomposition_quality_det_freqs')
save_dir <- file.path(parent_save_dir, decomposition)
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

scatter_save_dir <- file.path(save_dir, "aggr_formants_scatter")
if (!dir.exists(scatter_save_dir)) {
  dir.create(scatter_save_dir, recursive = TRUE, showWarnings = FALSE)
}
#Vowels scatter (simulated and real data)
save_path12 <- file.path(scatter_save_dir, paste0(decomposition,"_det_formants12_scatter_",overlap_str,"_",stat,".png"))
save_path13 <- file.path(scatter_save_dir, paste0(decomposition,"_det_formants13_scatter_",overlap_str,"_",stat,".png"))
save_path23 <- file.path(scatter_save_dir, paste0(decomposition,"_det_formants23_scatter_",overlap_str,"_",stat,".png"))
#Consonants scatter (real data only)
save_path_cons12 <- file.path(scatter_save_dir, paste0(decomposition,"_det_formants12_consonants_scatter_",overlap_str,"_",stat,".png"))
save_path_cons13 <- file.path(scatter_save_dir, paste0(decomposition,"_det_formants13_consonants_scatter_",overlap_str,"_",stat,".png"))
save_path_cons23 <- file.path(scatter_save_dir, paste0(decomposition,"_det_formants23_consonants_scatter_",overlap_str,"_",stat,".png"))

# Vowel comparison scatter (between real/synth data)
vowel_comparison_dir <- file.path(save_dir,"vowel_comparisons")
if (!dir.exists(vowel_comparison_dir)) {
  dir.create(vowel_comparison_dir, recursive = TRUE, showWarnings = FALSE)
}
consonant_comparison_dir <- file.path(save_dir,"consonant_comparisons")
if (!dir.exists(consonant_comparison_dir)) {
  dir.create(consonant_comparison_dir, recursive = TRUE, showWarnings = FALSE)
}

#More parameters
freq1_upper_limit <- 1300
freq2_upper_limit <- 3000
freq3_upper_limit <- 5000

ground_truth_freqs <- data.frame(
  vowels = c("a", "e", "I", "aw", "u"),
  f1 = c(710, 550, 400, 590, 310),
  f2 = c(1100, 1770, 1920, 880, 870),
  f3 = c(2540, 2490, 2560, 2540, 2250),
  dataset = "ground_truth"
)

# sim_vowels map: {'a': 0, 'e': 1, 'I': 2, 'aw': 3, 'u': 4}
vowel_map <- c('a' = 0, 'e' = 1, 'I' = 2, 'aw' = 3, 'u' = 4)
reverse_vowel_map <- names(vowel_map)
names(reverse_vowel_map) <- vowel_map

load_json_data <- function(file_path) {
    json_text <- readLines(file_path)
    json_text <- gsub("\\bNaN\\b", "\"NaN\"", json_text)
    fromJSON(json_text, simplifyDataFrame = TRUE, simplifyMatrix = TRUE)
}


#This data file contains both the synthetic and real speech data
#(Vowels and TIMIT)
df <- load_json_data(data_file)

df$sim_vowels$gender <- df$sim_vowels$gender[1:4000]
df$sim_vowels$consonants <- NULL

# Interpolate speakers and gender
vowels_dims <- dim(df$sim_vowels$vowels)
speaker_matrix <- matrix(rep(df$sim_vowels$speaker_id, each=vowels_dims[2]), 
                        nrow=vowels_dims[1], 
                        ncol=vowels_dims[2],
                        byrow = TRUE)
df$sim_vowels$speaker_id <- speaker_matrix
#df$sim_vowels$gender <- ifelse(speaker_matrix <= 0.95, "F",
#                              ifelse(speaker_matrix >= 1.05, "M", "U"))

# Separate frequencies
sim_freqs <- df$sim_vowels$detected_frequencies
df$sim_vowels$freq1 <- as.vector(t(sim_freqs[,1,]))
df$sim_vowels$freq2 <- as.vector(t(sim_freqs[,2,]))
df$sim_vowels$freq3 <- as.vector(t(sim_freqs[,3,]))
df$sim_vowels$detected_frequencies <- NULL
# Flatten all variables
df$sim_vowels$vowels <- as.vector(t(df$sim_vowels$vowels))
df$sim_vowels$overlap_mask <- as.vector(t(df$sim_vowels$overlap_mask))
df$sim_vowels$speaker_id <- as.vector(t(df$sim_vowels$speaker_id))
df$sim_vowels$gender <- ifelse(df$sim_vowels$speaker_id <= 0.98, "F",
                              ifelse(df$sim_vowels$speaker_id >= 1.02, "M", "U"))

# Do the same for TIMIT

timit_vowels <- df$timit$vowels
timit_speakers <- df$timit$speaker_id
timit_genders <- df$timit$gender

# Interpolate speakers and gender for each vowel length
interpolated_speakers <- vector("list", length(timit_vowels))
interpolated_genders <- vector("list", length(timit_vowels))

for(i in seq_along(timit_vowels)) {
    vowel_length <- length(timit_vowels[[i]])
    interpolated_speakers[[i]] <- rep(timit_speakers[i], times=vowel_length)
    interpolated_genders[[i]] <- rep(timit_genders[i], times=vowel_length)
}

df$timit$speaker_id <- unlist(interpolated_speakers)
df$timit$gender <- unlist(interpolated_genders)
df$timit$vowels <- unlist(timit_vowels)
df$timit$consonants <- unlist(df$timit$consonants)


timit_freqs <- df$timit$detected_frequencies
df$timit$freq1 <- unlist(sapply(timit_freqs, function(x) x[1,]))
df$timit$freq2 <- unlist(sapply(timit_freqs, function(x) x[2,]))
df$timit$freq3 <- unlist(sapply(timit_freqs, function(x) x[3,]))
df$timit$detected_frequencies <- NULL
df$timit$overlap_mask <- unlist(df$timit$overlap_mask)

sim_vowels_df <- data.frame(
    dataset = "sim_vowels",
    vowels = df$sim_vowels$vowels,
    freq1 = df$sim_vowels$freq1,
    freq2 = df$sim_vowels$freq2,
    freq3 = df$sim_vowels$freq3,
    speaker_id = df$sim_vowels$speaker_id,
    gender = df$sim_vowels$gender,
    overlap_mask = df$sim_vowels$overlap_mask
)

timit_df <- data.frame(
    dataset = "timit",
    vowels = df$timit$vowels,
    freq1 = df$timit$freq1,
    freq2 = df$timit$freq2,
    freq3 = df$timit$freq3,
    speaker_id = df$timit$speaker_id,
    gender = df$timit$gender,
    overlap_mask = df$timit$overlap_mask
)

combined_df <- rbind(sim_vowels_df, timit_df)

combined_df <- as_tibble(combined_df)

# Map integers to their vowels for sim_vowels
combined_df <- combined_df %>%
  mutate(vowels = case_when(
    dataset == "sim_vowels" ~ as.character(reverse_vowel_map[vowels]),
    TRUE ~ vowels
  ))


# Figure 1: Formant 1/2 vs Formant 2/3 scatter plot for Vowels of real and synth data

#To ensure consistent ordering in plots
sim_vowel_order <- c("a", "e", "I", "aw", "u")  
timit_vowel_order <- sort(unique(df$timit$vowels[df$timit$vowels != "NO"]))  
consonant_order <- sort(unique(df$timit$consonants[df$timit$consonants != "NO"]))

avg_freqs <- combined_df %>%
  filter(
    gender != "U",
    vowels != "NO", 
    !(dataset == "sim_vowels" & (speaker_id < 0.9 | speaker_id > 1.1)),
    freq1 != 0 & freq1 < freq1_upper_limit,  # Filter out zero frequencies
    freq2 != 0 & freq2 < freq2_upper_limit,
    freq3 != 0 & freq3 < freq3_upper_limit,
    if (include_overlap) TRUE else overlap_mask != 1
    ) %>%
  group_by(dataset) %>%
  mutate(vowels = case_when(
    dataset == "sim_vowels" ~ factor(vowels, levels = sim_vowel_order),
    dataset == "timit" ~ factor(vowels, levels = timit_vowel_order),
    TRUE ~ factor(vowels)
  )) %>%
  group_by(dataset, vowels, gender) %>%
  summarise(
    avg_freq1 = if(stat == "Median") median(freq1, na.rm = TRUE) else mean(freq1, na.rm = TRUE),
    avg_freq2 = if(stat == "Median") median(freq2, na.rm = TRUE) else mean(freq2, na.rm = TRUE),
    avg_freq3 = if(stat == "Median") median(freq3, na.rm = TRUE) else mean(freq3, na.rm = TRUE),
    .groups = 'drop'
  )

if (plot_agg_formants) {

  # Plot theme with consistent styling - For the aggregated formants plots
  plot_theme_agg_formants <- theme_minimal() +
    theme(
      legend.position = legend_position_agg_formants,
      legend.text = element_text(size = legend_text_size_agg_formants, family = plot_font_family),
      legend.title = element_text(size = legend_title_size_agg_formants, family = plot_font_family),
      legend.key.size = unit(legend_key_size_agg_formants, "cm"),
      plot.title = element_text(size = plot_title_size_agg_formants, face = plot_title_face, hjust = plot_title_hjust, vjust = plot_title_vjust, family = plot_font_family),
      plot.title.position = "plot",  
      axis.title = element_text(size = axis_title_size_agg_formants, family = plot_font_family),
      axis.text.x = element_text(size = axis_text_size_agg_formants, family = plot_font_family),
      axis.text.y = element_text(size = axis_text_size_agg_formants, family = plot_font_family),
      panel.background = element_rect(fill = plot_background_color, color = NA),
      plot.background = element_rect(fill = plot_background_color, color = NA),
      legend.background = element_rect(fill = plot_background_color, color = NA)
    ) 
}

if (plot_comparisons) {
  # Plot theme with consistent styling - For the vowel_comparison plots
  plot_theme_comparisons <- theme_minimal() +
    theme(
      legend.position = legend_position_comparisons,
      legend.text = element_text(size = legend_text_size_comparisons, family = plot_font_family),
      legend.title = element_text(size = legend_title_size_comparisons, family = plot_font_family),
      legend.key.size = unit(legend_key_size_comparisons, "cm"),
      plot.title = element_text(size = plot_title_size_comparisons, face = plot_title_face, hjust = plot_title_hjust, vjust = plot_title_vjust, family = plot_font_family),
      plot.title.position = "plot",  
      axis.title = element_text(size = axis_title_size_comparisons, family = plot_font_family),
      axis.text.x = element_text(size = axis_text_size_comparisons, family = plot_font_family),
      axis.text.y = element_text(size = axis_text_size_comparisons, family = plot_font_family),
      panel.background = element_rect(fill = plot_background_color, color = NA),
      plot.background = element_rect(fill = plot_background_color, color = NA),
      legend.background = element_rect(fill = plot_background_color, color = NA)
    ) 
}

if (plot_violins) {
  # Plot theme with consistent styling - For the violin plots
  plot_theme_violins <- theme_minimal() +
    theme(
      legend.position = legend_position_violins,
      legend.text = element_text(size = legend_text_size_violins, family = plot_font_family),
      legend.title = element_text(size = legend_title_size_violins, family = plot_font_family),
      legend.key.size = unit(legend_key_size_violins, "cm"),
      plot.title = element_text(size = plot_title_size_violins, face = plot_title_face, hjust = plot_title_hjust, vjust = plot_title_vjust, family = plot_font_family),
      plot.title.position = "plot",  
      axis.title = element_text(size = axis_title_size_violins, family = plot_font_family),
      axis.text.x = element_text(size = axis_text_size_violins, family = plot_font_family),
      axis.text.y = element_text(size = axis_text_size_violins, family = plot_font_family),
      panel.background = element_rect(fill = plot_background_color, color = NA),
      plot.background = element_rect(fill = plot_background_color, color = NA),
      legend.background = element_rect(fill = plot_background_color, color = NA)
    ) 
}

if (plot_agg_formants) {
  # Formant 1 vs Formant 2 scatter plot
  p1 <- ggplot(avg_freqs, aes(x = avg_freq1, y = avg_freq2, 
                        shape = dataset, 
                        color = gender,
                        label = vowels)) +
    
    # Original plot elements
    #geom_line(aes(group = interaction(dataset, gender)), alpha = 0.3) +
    geom_point(size = point_size_regular_agg_formants, alpha = 0.8) +
    geom_text(hjust = -0.5, vjust = -0.5, size = text_size_regular_agg_formants, show.legend = FALSE) +
    # Add ground truth points
    geom_point(data = ground_truth_freqs, 
              aes(x = f1, y = f2, shape = "ground_truth"),
              size = point_size_large_agg_formants,
              color = "darkgreen") +
    geom_text(data = ground_truth_freqs,
              aes(x = f1, y = f2, label = vowels),
              hjust = -0.5, vjust = -0.5,
              size = text_size_regular_agg_formants,
              color = "darkgreen") +
    scale_shape_manual(
      values = dataset_shapes,
      labels = dataset_labels,
      breaks = c("sim_vowels", "timit", "ground_truth") 
    ) +
    scale_color_manual(values = gender_colors,
                        labels = c("Female", "Male")) +
    guides(
      shape = guide_legend(override.aes = list(size = text_size_legend_agg_formants)), 
      color = guide_legend(override.aes = list(size = text_size_legend_agg_formants))
    ) +
    labs(
      x = "First Formant (F1) - Hz",
      y = "Second Formant (F2) - Hz",
      shape = "Dataset",
      color = "Gender",
      title = paste0(stat, " Detected Formant Frequencies by ", toupper(decomposition_display))
    ) +
    plot_theme_agg_formants

  ggsave(save_path12, 
    p1, 
    width = 10, 
    height = 8, 
    dpi = 600
  )

  # Formant 1 vs Formant 3 scatter plot
  p2 <- ggplot(avg_freqs, aes(x = avg_freq1, y = avg_freq3, 
                        shape = dataset, 
                        color = gender,
                        label = vowels)) +
    # Original plot elements
    #geom_line(aes(group = interaction(dataset, gender)), alpha = 0.3) +
    geom_point(size = point_size_regular_agg_formants, alpha = 0.8) +
    geom_text(hjust = -0.5, vjust = -0.5, size = text_size_regular_agg_formants, show.legend = FALSE) +
    # Add ground truth points
    geom_point(data = ground_truth_freqs, 
              aes(x = f1, y = f3, shape = "ground_truth"),
              size = point_size_large_agg_formants,
              color = "darkgreen") +
    geom_text(data = ground_truth_freqs,
              aes(x = f1, y = f3, label = vowels),
              hjust = -0.5, vjust = -0.5,
              size = text_size_regular_agg_formants,
              color = "darkgreen") +
    scale_shape_manual(
      values = dataset_shapes,
      labels = dataset_labels,
      breaks = c("sim_vowels", "timit", "ground_truth")
    ) +
    scale_color_manual(values = gender_colors,
                        labels = c("Female", "Male")) +
    guides(
      shape = guide_legend(override.aes = list(size = text_size_legend_agg_formants)), 
      color = guide_legend(override.aes = list(size = text_size_legend_agg_formants))
    ) +
    labs(
      x = "First Formant (F1) - Hz",
      y = "Third Formant (F3) - Hz",
      shape = "Dataset",
      color = "Gender",
      title = paste0(stat, " Detected Formant Frequencies by ", toupper(decomposition_display))
    ) +
    plot_theme_agg_formants



  ggsave(save_path13,
    p2, 
    width = 10, 
    height = 8, 
    dpi = 600)


  # Formant 2 vs Formant 3 scatter plot
  p3 <- ggplot(avg_freqs, aes(x = avg_freq2, y = avg_freq3, 
                        shape = dataset, 
                        color = gender,
                        label = vowels)) +
    # Original plot elements
    #geom_line(aes(group = interaction(dataset, gender)), alpha = 0.3) +
    geom_point(size = point_size_regular_agg_formants, alpha = 0.8) +
    geom_text(hjust = -0.5, vjust = -0.5, size = text_size_regular_agg_formants, show.legend = FALSE) +
    geom_point(data = ground_truth_freqs, 
              aes(x = f2, y = f3, shape = "ground_truth"),
              size = point_size_large_agg_formants,
              color = "darkgreen") +
    geom_text(data = ground_truth_freqs,
              aes(x = f2, y = f3, label = vowels),
              hjust = -0.5, vjust = -0.5,
              size = text_size_regular_agg_formants,
              color = "darkgreen") +
    scale_shape_manual(
      values = dataset_shapes,
      labels = dataset_labels,
      breaks = c("sim_vowels", "timit", "ground_truth")
    ) +
    scale_color_manual(values = gender_colors,
                        labels = c("Female", "Male")) +
    guides(
      shape = guide_legend(override.aes = list(size = text_size_legend_agg_formants)), 
      color = guide_legend(override.aes = list(size = text_size_legend_agg_formants))
    ) +
    labs(
      x = "Second Formant (F2) - Hz",
      y = "Third Formant (F3) - Hz",
      shape = "Dataset",
      color = "Gender",
      title = paste0(stat, " Detected Formant Frequencies by ", toupper(decomposition_display))
    ) +
    plot_theme_agg_formants

  ggsave(save_path23, 
    p3, 
    width = 10, 
    height = 8, 
    dpi = 600)

  # Figure 1.5: Formants for consonants of real data
}
timit_df_cons <- data.frame(
    dataset = "timit",
    vowels = df$timit$vowels,
    freq1 = df$timit$freq1,
    freq2 = df$timit$freq2,
    freq3 = df$timit$freq3,
    speaker_id = df$timit$speaker_id,
    gender = df$timit$gender,
    overlap_mask = df$timit$overlap_mask,
    consonants = df$timit$consonants
)

avg_cons_freqs <- timit_df_cons %>%
  filter(
    consonants != "NO",
    freq1 != 0,  # Filter out zero frequencies
    freq2 != 0,
    freq3 != 0,
    if (include_overlap) TRUE else overlap_mask != 1
  ) %>%
  mutate(consonants = factor(consonants, levels = consonant_order)) %>%
  group_by(consonants, gender) %>%
  summarise(
    avg_freq1 = if(stat == "Median") median(freq1, na.rm = TRUE) else mean(freq1, na.rm = TRUE),
    avg_freq2 = if(stat == "Median") median(freq2, na.rm = TRUE) else mean(freq2, na.rm = TRUE),
    avg_freq3 = if(stat == "Median") median(freq3, na.rm = TRUE) else mean(freq3, na.rm = TRUE),
    .groups = 'drop'
  )#overlap_mask != 1

if (plot_agg_formants) {
  p_cons1 <- ggplot(avg_cons_freqs, aes(x = avg_freq1, y = avg_freq2, 
                                      color = gender,
                                      label = consonants)) +
    #geom_line(aes(group = gender), alpha = 0.3) +
    geom_point(size = point_size_regular_agg_formants, alpha = 0.8) +
    geom_text(hjust = -0.5, vjust = -0.5, size = text_size_regular_agg_formants, show.legend = FALSE) +
    scale_color_manual(values = gender_colors,
                        labels = c("Female", "Male")) +
    guides(color = guide_legend(override.aes = list(size = text_size_legend_agg_formants))) +
    labs(
      x = "Oscillatory Component 1 - Hz",
      y = "Oscillatory Component 2 - Hz",
      color = "Gender",
      title = paste0(stat, " Detected Oscillatory Frequencies in TIMIT by ", toupper(decomposition_display))
    ) +
    plot_theme_agg_formants


  ggsave(save_path_cons12, 
    p_cons1, 
    width = 10, 
    height = 8, 
    dpi = 600
  )

  p_cons2 <- ggplot(avg_cons_freqs, aes(x = avg_freq1, y = avg_freq3, 
                                      color = gender,
                                      label = consonants)) +
    #geom_line(aes(group = gender), alpha = 0.3) +
    geom_point(size = point_size_regular_agg_formants, alpha = 0.8) +
    geom_text(hjust = -0.5, vjust = -0.5, size = text_size_regular_agg_formants, show.legend = FALSE) +
    scale_color_manual(values = gender_colors,
                        labels = c("Female", "Male")) +
    guides(color = guide_legend(override.aes = list(size = text_size_legend_agg_formants))) +
    labs(
      x = "Oscillatory Component 1 - Hz",
      y = "Oscillatory Component 3 - Hz",
      color = "Gender",
      title = paste0(stat, " Detected Oscillatory Frequencies in TIMIT by ", toupper(decomposition_display))
    ) +
    plot_theme_agg_formants


  ggsave(save_path_cons13, 
    p_cons2, 
    width = 10, 
    height = 8, 
    dpi = 600
  )

  p_cons3 <- ggplot(avg_cons_freqs, aes(x = avg_freq2, y = avg_freq3, 
                                      color = gender,
                                      label = consonants)) +
    #geom_line(aes(group = gender), alpha = 0.3) +
    geom_point(size = point_size_regular_agg_formants, alpha = 0.8) +
    geom_text(hjust = -0.5, vjust = -0.5, size = text_size_regular_agg_formants, show.legend = FALSE) +
    scale_color_manual(values = gender_colors,
                        labels = c("Female", "Male")) +
    guides(color = guide_legend(override.aes = list(size = text_size_legend_agg_formants))) +
    labs(
      x = "Oscillatory Component 2 - Hz",
      y = "Oscillatory Component 3 - Hz",
      color = "Gender",
      title = paste0(stat, " Detected Oscillatory Frequencies in TIMIT by ", toupper(decomposition_display))
    ) +
    plot_theme_agg_formants


  ggsave(save_path_cons23, 
    p_cons3, 
    width = 10, 
    height = 8, 
    dpi = 600
  )
}

# Figure 2: Scatter plot of all points with Marginal Distributions of Categories
# For vowel pairs: color by gender and shape by dataset

if (plot_comparisons) {

  create_vowel_comparison_plot <- function(vowel_sim_vowels, vowel_timit, combined_df, formant_pair = c(1,2)) {
    # Validate formant pair input
    if (!all(formant_pair %in% 1:3) || length(formant_pair) != 2) {
      stop("formant_pair must be a vector of two different numbers from 1-3")
    }
    
    # Create frequency column names based on formant pair
    freq_x <- paste0("freq", formant_pair[1])
    freq_y <- paste0("freq", formant_pair[2])
    
    # Create axis labels
    axis_labels <- c(
      "freq1" = "First Formant (F1) - Hz",
      "freq2" = "Second Formant (F2) - Hz",
      "freq3" = "Third Formant (F3) - Hz"
    )
    
    # Create filtered dataset for specific vowels
    filtered_df <- combined_df %>%
      filter(
        (dataset == "sim_vowels" & vowels == vowel_sim_vowels) |
        (dataset == "timit" & vowels == vowel_timit),
        gender %in% c("F", "M"),
        freq1 != 0 & freq1 < freq1_upper_limit,  # Filter out zero frequencies
        freq2 != 0 & freq2 < freq2_upper_limit,
        freq3 != 0 & freq3 < freq3_upper_limit,
        if (include_overlap) TRUE else overlap_mask != 1
      ) 

    # Define colors and labels
    distinct_colors <- c(
      "sim_vowels.F" = "#D81B60",  # Magenta
      "sim_vowels.M" = "#1E88E5",  # Blue
      "timit.F" = "#004D40",       # Dark Teal
      "timit.M" = "#FFC107"        # Amber
    )

    legend_labels <- c(
      "sim_vowels.F" = "SimVowels - Female",
      "sim_vowels.M" = "SimVowels - Male",
      "timit.F" = "TIMIT - Female",
      "timit.M" = "TIMIT - Male"
    )

    # Create main plot with dynamic formant selection
    p_vowel_compare <- ggplot(filtered_df, 
                            aes(x = .data[[freq_x]], y = .data[[freq_y]], 
                                shape = dataset,
                                color = interaction(dataset, gender))) +
      geom_point(alpha = 0.6, size = point_size_regular_comparisons) +
      scale_shape_manual(values = c("sim_vowels" = 16, "timit" = 17), labels = c("SimVowels", "TIMIT")) +
      scale_color_manual(values = distinct_colors,
                        labels = legend_labels) +
      labs(
        x = axis_labels[freq_x],
        y = axis_labels[freq_y],
        shape = "Dataset",
        color = "Dataset-Gender",
        title = sprintf("Vowels: SimVowels=%s, TIMIT=%s", 
                      toupper(decomposition_display), vowel_sim_vowels, vowel_timit)
      )+
      plot_theme_comparisons +
      theme(
        plot.margin = vowel_comparison_margin,
        legend.position = "left",
        legend.box.margin = legend_box_margin,
        legend.key.size = unit(legend_key_size_comparisons - 1, "cm"),  # Slightly smaller than default
        plot.title.position = "plot",
        plot.title = element_text(hjust = plot_title_hjust, vjust = plot_title_vjust)
      ) +
      guides(
        shape = guide_legend(override.aes = list(size = text_size_legend_comparisons)),
        color = guide_legend(override.aes = list(size = text_size_legend_comparisons))
      )

    # Add marginal distributions
    p_vowel_compare_margin <- ggMarginal(p_vowel_compare, 
                                      type = "density",
                                      groupColour = TRUE,
                                      groupFill = TRUE,
                                      size = 15,
                                      margins = "both",
                                      xparams = list(
                                        fill = NA,
                                        alpha = 0.3
                                      ),
                                      yparams = list(
                                        fill = NA,
                                        alpha = 0.3
                                      ))

    # Save plot with formant pair in filename
    ggsave(file.path(vowel_comparison_dir, paste0(decomposition,
      "_det_formants", formant_pair[1], formant_pair[2], "_scatter_vowel_compare_",
      vowel_sim_vowels, "_", vowel_timit, "_marginal_", overlap_str,".png")), 
      p_vowel_compare_margin, 
      width = 24,
      height = 16,
      dpi = 600,
      bg = "white"
    )
    
    return(p_vowel_compare_margin)
  }


  # Create plots for different vowel pairs and formant combinations
  p1 <- create_vowel_comparison_plot("I", "iy", combined_df, c(1,2)) 
  p2 <- create_vowel_comparison_plot("e", "ey", combined_df, c(1,2))  
  p3 <- create_vowel_comparison_plot("a", "ay", combined_df, c(1,2))  
  p4 <- create_vowel_comparison_plot("aw", "aw", combined_df, c(1,2)) 
  p5 <- create_vowel_comparison_plot("u", "uw", combined_df, c(1,2)) 

  p6 <- create_vowel_comparison_plot("I", "iy", combined_df, c(2,3)) 
  p7 <- create_vowel_comparison_plot("e", "ey", combined_df, c(2,3))  
  p8 <- create_vowel_comparison_plot("a", "ay", combined_df, c(2,3))  
  p9 <- create_vowel_comparison_plot("aw", "aw", combined_df, c(2,3)) 
  p10 <- create_vowel_comparison_plot("u", "uw", combined_df, c(2,3)) 

  # Create consonant comparison only for timit dataset
  create_consonant_comparison_plot <- function(consonant, timit_df_cons, formant_pair = c(1,2)) {
    # Validate formant pair input
    if (!all(formant_pair %in% 1:3) || length(formant_pair) != 2) {
      stop("formant_pair must be a vector of two different numbers from 1-3")
    }
    
    # Create frequency column names based on formant pair
    freq_x <- paste0("freq", formant_pair[1])
    freq_y <- paste0("freq", formant_pair[2])
    
    # Create axis labels
    axis_labels <- c(
      "freq1" = "First Oscillatory Component",
      "freq2" = "Second Oscillatory Component",
      "freq3" = "Third Oscillatory Component"
    )
    
    # Create filtered dataset for specific consonant
    filtered_df <- timit_df_cons %>%
      filter(
        consonants == consonant,
        gender %in% c("F", "M"),
        freq1 != 0,  # Filter out zero frequencies
        freq2 != 0,
        freq3 != 0,
        if (include_overlap) TRUE else overlap_mask != 1
      ) 

    # Define colors for genders
    gender_colors <- c(
      "F" = "#D81B60",  # Magenta
      "M" = "#1E88E5"   # Blue
    )

    # Create main plot
    p_consonant_compare <- ggplot(filtered_df, 
                                aes(x = .data[[freq_x]], 
                                    y = .data[[freq_y]], 
                                    color = gender)) +
      geom_point(alpha = 0.6, size = point_size_regular_comparisons) +
      scale_color_manual(values = gender_colors,
                        labels = c("Female", "Male")) +
      labs(
        x = axis_labels[freq_x],
        y = axis_labels[freq_y],
        color = "Gender",
        title = sprintf("Detected Oscillatory Components in TIMIT by %s \n(Consonant: %s)", 
                  toupper(decomposition_display), consonant)
      ) +
      plot_theme_comparisons +
      theme(
        plot.margin = vowel_comparison_margin,
        legend.position = "left",
        legend.box.margin = legend_box_margin,
        legend.key.size = unit(legend_key_size_comparisons - 1, "cm"),
        plot.title.position = "plot",
        plot.title = element_text(hjust = plot_title_hjust, vjust = plot_title_vjust)
      ) +
      guides(
        color = guide_legend(override.aes = list(size = text_size_legend_comparisons))
      )

    # Add marginal distributions
    p_consonant_compare_margin <- ggMarginal(p_consonant_compare, 
                                          type = "density",
                                          groupColour = TRUE,
                                          groupFill = TRUE,
                                          size = 15,
                                          margins = "both",
                                          xparams = list(
                                            fill = NA,
                                            alpha = 0.3
                                          ),
                                          yparams = list(
                                            fill = NA,
                                            alpha = 0.3
                                          ))

    # Save plot
    ggsave(file.path(consonant_comparison_dir, paste0(decomposition,
      "_det_formants", formant_pair[1], formant_pair[2], "_scatter_consonant_",
      consonant, "_marginal_with_",overlap_str,".png")), 
      p_consonant_compare_margin, 
      width = 24,
      height = 16,
      dpi = 600,
      bg = "white"
    )
    
    return(p_consonant_compare_margin)
  }

  # Example usage:
  # Create plots for different consonants and formant combinations
  p1 <- create_consonant_comparison_plot("b", timit_df_cons, c(1,2))  
  p2 <- create_consonant_comparison_plot("d", timit_df_cons, c(1,2))  
  p3 <- create_consonant_comparison_plot("f", timit_df_cons, c(1,2)) 
  p4 <- create_consonant_comparison_plot("k", timit_df_cons, c(1,2)) 
  p5 <- create_consonant_comparison_plot("l", timit_df_cons, c(1,2))  
  p6 <- create_consonant_comparison_plot("s", timit_df_cons, c(1,2)) 

  p7 <- create_consonant_comparison_plot("b", timit_df_cons, c(2,3))  
  p8 <- create_consonant_comparison_plot("d", timit_df_cons, c(2,3))  
  p9 <- create_consonant_comparison_plot("f", timit_df_cons, c(2,3)) 
  p10 <- create_consonant_comparison_plot("k", timit_df_cons, c(2,3)) 
  p11 <- create_consonant_comparison_plot("l", timit_df_cons, c(2,3))  
  p12 <- create_consonant_comparison_plot("s", timit_df_cons, c(2,3)) 

}



# Figure 3: Use violin plots to show distribution of formants for vowels and consonants
# Make an ordered violin plot for vowels and one for consonants
# In each plot, start from f1 and end with f3
# For f1, plot male and female, and different datasets, on different violins

if (plot_violins) {

  # Create directories for individual violin plots
  vowel_violin_individual_dir <- file.path(vowel_comparison_dir, "individual_violins")
  if (!dir.exists(vowel_violin_individual_dir)) {
    dir.create(vowel_violin_individual_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  consonant_violin_individual_dir <- file.path(consonant_comparison_dir, "individual_violins")
  if (!dir.exists(consonant_violin_individual_dir)) {
    dir.create(consonant_violin_individual_dir, recursive = TRUE, showWarnings = FALSE)
  }

  create_vowel_violin_plot <- function(vowel_sim_vowels, vowel_timit, combined_df) {
    # Filter data for specific vowel
    filtered_df <- combined_df %>%
      filter(
        (dataset == "sim_vowels" & vowels == vowel_sim_vowels) |
        (dataset == "timit" & vowels == vowel_timit),
        gender %in% c("F", "M"),
        freq1 != 0 & freq1 < freq1_upper_limit,  # Filter out zero frequencies
        freq2 != 0 & freq2 < freq2_upper_limit,
        freq3 != 0 & freq3 < freq3_upper_limit,
        if (include_overlap) TRUE else overlap_mask != 1
      ) %>% 
      # Reshape data for violin plot
      pivot_longer(
        cols = c(freq1, freq2, freq3),
        names_to = "formant",
        values_to = "frequency"
      ) %>%
      # Create dataset-gender interaction
      mutate(
        group = interaction(dataset, gender),
        formant = factor(formant, 
                        levels = c("freq1", "freq2", "freq3"),
                        labels = c("F1", "F2", "F3"))
      )

    # Create violin plot
    p <- ggplot(filtered_df, 
              aes(x = formant, y = frequency, fill = group)) +
      geom_violin(position = position_dodge(width = 0.7),
                alpha = 0.7, scale = "width") +
      scale_fill_manual(
        values = c(
          "sim_vowels.F" = "#D81B60",
          "sim_vowels.M" = "#1E88E5",
          "timit.F" = "#004D40",
          "timit.M" = "#FFC107"
        ),
        labels = c(
          "sim_vowels.F" = "SimVowels - Female",
          "sim_vowels.M" = "SimVowels - Male",
          "timit.F" = "TIMIT - Female",
          "timit.M" = "TIMIT - Male"
        )
      ) +
      labs(
        x = "Formant",
        y = "Frequency (Hz)",
        fill = "Dataset-Gender",
        title = sprintf("Vowels: SimVowels=%s, TIMIT=%s",
                      toupper(decomposition_display), vowel_sim_vowels, vowel_timit)
      ) +
      guides(fill = guide_legend(
        keywidth = 3,  # Smaller key width
        keyheight = 3, # Smaller key height
        label.position = "right",
        label.hjust = 0,
        byrow = TRUE,    # Arrange legend items in rows instead of columns
        nrow = 4,        # Force all items into 4 rows (one per group)
        spacing = legend_spacing_violins
      )) +
      plot_theme_violins +
      theme(
        plot.title = element_text(size = plot_title_size_violins, face = plot_title_face, hjust = plot_title_hjust),
        legend.position = legend_position_violins,
        legend.margin = legend_margin_violins,
        legend.box.margin = margin(0, 0, 0, 0),
        legend.box.spacing = unit(0.2, "cm")
      )
    
    return(p)
  }

  # Create violin plots for different vowel pairs
  vowel_pairs <- list(
    c("I", "iy"),
    c("e", "ey"),
    c("a", "ay"),
    c("aw", "aw"),
    c("u", "uw")
  )

  # Create and save individual violin plots
  for (pair in vowel_pairs) {
    vowel_sim_vowels <- pair[1]
    vowel_timit <- pair[2]
    
    # Get the plot
    p <- create_vowel_violin_plot(vowel_sim_vowels, vowel_timit, combined_df)
    
    # Save the individual plot with legend
    individual_plot_filename <- file.path(vowel_violin_individual_dir, 
                                         paste0(decomposition, "_vowel_", 
                                               vowel_sim_vowels, "_", vowel_timit, 
                                               "_formant_distributions_violin_", 
                                               overlap_str, ".png"))
    
    ggsave(individual_plot_filename, 
           p, 
           width = 12, 
           height = 10, 
           dpi = 600)
  }

  # Create and arrange multiple violin plots for the combined figure
  violin_plots <- map(vowel_pairs, ~create_vowel_violin_plot(.x[1], .x[2], combined_df))

  # Combine plots using patchwork
  combined_violins <- wrap_plots(violin_plots, ncol = 2) +
    plot_layout(guides = "collect") &
    theme(
      legend.position = legend_position_violins,
      legend.key.size = unit(legend_key_size_violins, "cm"),    # Even smaller for combined plot
      legend.text = element_text(size = legend_text_size_violins),
      legend.title = element_text(size = legend_title_size_violins),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0)
    )

  # Save combined plot
  ggsave(file.path(vowel_comparison_dir,
    paste0(decomposition,
    "_vowel_formant_distributions_violin_",overlap_str,".png")),
    combined_violins,
    width = 20, 
    height = 25, 
    dpi = 600
  )

  # Similar function for consonants (TIMIT only)
  create_consonant_violin_plot <- function(consonant, timit_df_cons) {
    filtered_df <- timit_df_cons %>%
      filter(
        consonants == consonant,
        gender %in% c("F", "M"),
        freq1 != 0,  # Filter out zero frequencies
        freq2 != 0,
        freq3 != 0,
        if (include_overlap) TRUE else overlap_mask != 1
      ) %>% 
      pivot_longer(
        cols = c(freq1, freq2, freq3),
        names_to = "component",
        values_to = "frequency"
      ) %>%
      mutate(
        component = factor(component, 
                          levels = c("freq1", "freq2", "freq3"),
                          labels = c("C1", "C2", "C3"))
      )

    ggplot(filtered_df, aes(x = component, y = frequency, fill = gender)) +
      geom_violin(position = position_dodge(width = 0.7),
                alpha = 0.7, scale = "width") +
      scale_fill_manual(
        values = gender_colors,
        labels = c("Female", "Male")
      ) +
      labs(
        x = "Oscillatory Component",
        y = "Frequency (Hz)",
        fill = "Gender",
        title = sprintf("Detected Oscillatory Component Distributions by %s (TIMIT Consonant: %s)",
                      toupper(decomposition_display), consonant)
      )+
      guides(fill = guide_legend(
        keywidth = 3,  # Smaller key width
        keyheight = 3, # Smaller key height
        label.position = "right",
        label.hjust = 0,
        spacing = legend_spacing_violins
      )) +
      plot_theme_violins +
      theme(
        plot.title = element_text(size = plot_title_size_violins, face = plot_title_face, hjust = plot_title_hjust),
        legend.position = legend_position_violins,
        legend.margin = legend_margin_violins,
        legend.box.margin = margin(0, 0, 0, 0)
      )
  }

  # Create violin plots for different consonants
  consonant_list <- c("b", "d", "f", "k", "l", "s")
  
  # Create and save individual consonant violin plots
  for (consonant in consonant_list) {
    # Get the plot
    p <- create_consonant_violin_plot(consonant, timit_df_cons)
    
    # Save the individual plot with legend
    individual_plot_filename <- file.path(consonant_violin_individual_dir, 
                                         paste0(decomposition, "_consonant_", 
                                               consonant, 
                                               "_component_distributions_violin_", 
                                               overlap_str, ".png"))
    
    ggsave(individual_plot_filename, 
           p, 
           width = 12, 
           height = 10, 
           dpi = 600)
  }
  
  # Create plots for combined figure
  consonant_plots <- map(consonant_list, ~create_consonant_violin_plot(., timit_df_cons))

  # Combine consonant plots
  combined_consonant_violins <- wrap_plots(consonant_plots, ncol = 2) +
    plot_layout(guides = "collect") &
    theme(
      legend.position = legend_position_violins,
      legend.key.size = unit(legend_key_size_violins, "cm"),
      legend.text = element_text(size = legend_text_size_violins),
      legend.title = element_text(size = legend_title_size_violins),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0)
    )

  # Save combined consonant plot
  ggsave(file.path(consonant_comparison_dir,
    paste0(decomposition,
    "_consonant_component_distributions_violin_",overlap_str,".png")),
    combined_consonant_violins,
    width = 20, 
    height = 25, 
    dpi = 600
  )
}