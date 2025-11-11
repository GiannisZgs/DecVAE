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

#Set parameters and paths
decomposition <- 'vmd'
NoC <- '3'
branch <- 'frame' #'frame' for Z, 'sequence' for S
feature <- 'det_freqs'

#Load directories
load_dir <- file.path("D:","data_for_figures_det_freqs","decomposition_quality",decomposition)
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
#Vowels scatter (simulated and real data)
save_path12 <- file.path(save_dir, "aggr_formants_scatter", paste0(decomposition,"_det_formants12_scatter_",overlap_str,"_",stat,".png"))
save_path13 <- file.path(save_dir, "aggr_formants_scatter", paste0(decomposition,"_det_formants13_scatter_",overlap_str,"_",stat,".png"))
save_path23 <- file.path(save_dir, "aggr_formants_scatter", paste0(decomposition,"_det_formants23_scatter_",overlap_str,"_",stat,".png"))
#Consonants scatter (real data only)
save_path_cons12 <- file.path(save_dir, "aggr_formants_scatter", paste0(decomposition,"_det_formants12_consonants_scatter_",overlap_str,"_",stat,".png"))
save_path_cons13 <- file.path(save_dir, "aggr_formants_scatter", paste0(decomposition,"_det_formants13_consonants_scatter_",overlap_str,"_",stat,".png"))
save_path_cons23 <- file.path(save_dir, "aggr_formants_scatter", paste0(decomposition,"_det_formants23_consonants_scatter_",overlap_str,"_",stat,".png"))
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

#Set fonts and style
font <- "Gudea"
font_add_google(family=font, font, db_cache = TRUE)
fa_path <- systemfonts::font_info(family = "Font Awesome 6 Brands")[["path"]]
font_add(family = "fa-brands", regular = fa_path)
theme_set(theme_minimal(base_family = font, base_size = 10))
bg <- "#F4F5F1"
txt_col <- "black"
showtext_auto(enable = TRUE)

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


# Try 3D scatter plot with all formants

p3d_vowels <- plot_ly() %>%
  add_trace(data = subset(avg_freqs, dataset == "sim_vowels" & gender == "F"),
            x = ~avg_freq1, y = ~avg_freq2, z = ~avg_freq3,
            type = 'scatter3d', mode = 'lines',
            line = list(color = '#D81B60', width = 2),
            opacity = 0.3,
            showlegend = FALSE) %>%
  add_trace(data = subset(avg_freqs, dataset == "sim_vowels" & gender == "M"),
            x = ~avg_freq1, y = ~avg_freq2, z = ~avg_freq3,
            type = 'scatter3d', mode = 'lines',
            line = list(color = '#1E88E5', width = 2),
            opacity = 0.3,
            showlegend = FALSE) %>%
  add_trace(data = subset(avg_freqs, dataset == "timit" & gender == "F"),
            x = ~avg_freq1, y = ~avg_freq2, z = ~avg_freq3,
            type = 'scatter3d', mode = 'lines',
            line = list(color = '#D81B60', width = 2),
            opacity = 0.3,
            showlegend = FALSE) %>%
  add_trace(data = subset(avg_freqs, dataset == "timit" & gender == "M"),
            x = ~avg_freq1, y = ~avg_freq2, z = ~avg_freq3,
            type = 'scatter3d', mode = 'lines',
            line = list(color = '#1E88E5', width = 2),
            opacity = 0.3,
            showlegend = FALSE) %>%
  # Add points with solid colors
  add_trace(data = subset(avg_freqs, dataset == "sim_vowels" & gender == "F"),
            x = ~avg_freq1, y = ~avg_freq2, z = ~avg_freq3,
            type = 'scatter3d', mode = 'markers+text',
            marker = list(
              symbol = 'sphere',
              color = '#D81B60',
              size = 8
            ),
            text = ~vowels,
            name = "sim_vowels - Female",
            textposition = "top right") %>%
  add_trace(data = subset(avg_freqs, dataset == "sim_vowels" & gender == "M"),
            x = ~avg_freq1, y = ~avg_freq2, z = ~avg_freq3,
            type = 'scatter3d', mode = 'markers+text',
            marker = list(
              symbol = 'sphere',
              color = '#1E88E5',
              size = 8
            ),
            text = ~vowels,
            name = "sim_vowels - Male",
            textposition = "top right") %>%
  add_trace(data = subset(avg_freqs, dataset == "timit" & gender == "F"),
            x = ~avg_freq1, y = ~avg_freq2, z = ~avg_freq3,
            type = 'scatter3d', mode = 'markers+text',
            marker = list(
              symbol = 'diamond',
              color = '#D81B60',
              size = 8
            ),
            text = ~vowels,
            name = "timit - Female",
            textposition = "top right") %>%
  add_trace(data = subset(avg_freqs, dataset == "timit" & gender == "M"),
            x = ~avg_freq1, y = ~avg_freq2, z = ~avg_freq3,
            type = 'scatter3d', mode = 'markers+text',
            marker = list(
              symbol = 'diamond',
              color = '#1E88E5',
              size = 8
            ),
            text = ~vowels,
            name = "timit - Male",
            textposition = "top right") %>%
  
  # Add ground truth points
  add_trace(data = ground_truth_freqs,
            x = ~f1, y = ~f2, z = ~f3,
            type = 'scatter3d', mode = 'markers+text',
            marker = list(
              symbol = 'star',
              color = 'darkgreen',
              size = 12
            ),
            text = ~vowels,
            name = "Ground Truth",
            textposition = "top right") %>%

  layout(
    scene = list(
      xaxis = list(title = "First Formant (F1)"),
      yaxis = list(title = "Second Formant (F2)"),
      zaxis = list(title = "Third Formant (F3)")
    ),
    title = paste0("3D Plot of ", toupper(decomposition), " Detected Average Formant Frequencies by Dataset, Vowel, and Gender"),
    showlegend = TRUE,
    legend = list(
      x = 0.8, y = 0.9,
      traceorder = "normal",  # Ensure consistent legend order
      itemsizing = "constant"  # Keep legend symbol sizes consistent
    )
  )

# Create 3D scatter plot for consonants with solid colors
p3d_cons <- plot_ly() %>%
  # Add lines
  add_trace(data = subset(avg_cons_freqs, gender == "F"),
            x = ~avg_freq1, y = ~avg_freq2, z = ~avg_freq3,
            type = 'scatter3d', mode = 'lines',
            line = list(color = '#D81B60', width = 2),
            opacity = 0.3,
            showlegend = FALSE) %>%
  add_trace(data = subset(avg_cons_freqs, gender == "M"),
            x = ~avg_freq1, y = ~avg_freq2, z = ~avg_freq3,
            type = 'scatter3d', mode = 'lines',
            line = list(color = '#1E88E5', width = 2),
            opacity = 0.3,
            showlegend = FALSE) %>%
  # Add points with solid colors
  add_trace(data = subset(avg_cons_freqs, gender == "F"),
            x = ~avg_freq1, y = ~avg_freq2, z = ~avg_freq3,
            type = 'scatter3d', mode = 'markers+text',
            marker = list(
              symbol = 'circle',
              color = '#D81B60',
              size = 8
            ),
            text = ~consonants,
            name = "Female",
            textposition = "top right") %>%
  add_trace(data = subset(avg_cons_freqs, gender == "M"),
            x = ~avg_freq1, y = ~avg_freq2, z = ~avg_freq3,
            type = 'scatter3d', mode = 'markers+text',
            marker = list(
              symbol = 'circle',
              color = '#1E88E5',
              size = 8
            ),
            text = ~consonants,
            name = "Male",
            textposition = "top right") %>%
  layout(
    scene = list(
      xaxis = list(title = "Oscillatory Component 1"),
      yaxis = list(title = "Oscillatory Component 2"),
      zaxis = list(title = "Oscillatory Component 3")
    ),
    title = paste0("3D Plot of ", toupper(decomposition), " Detected Average Oscillatory Frequencies of TIMIT Consonants by Gender"),
    showlegend = TRUE,
    legend = list(x = 0.8, y = 0.9)
  )

# Save the plots as HTML files for interactivity
htmlwidgets::saveWidget(p3d_vowels, 
                       paste0("decomposition_quality_frame/",decomposition,"/",decomposition,"_det_formants_3d_vowels.html"), selfcontained = TRUE)
htmlwidgets::saveWidget(p3d_cons, 
                       paste0("decomposition_quality_frame/",decomposition,"/",decomposition,"_det_formants_3d_consonants.html"), selfcontained = TRUE)
