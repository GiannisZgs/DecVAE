#' SI Figure 9: decomposition quality assessment at the input time domain. This 
#' script draws the correlation distributions between pairs of components. It also 
#' draws the distribution of the correlation between correlations of pairs - how likely
#' it is for two different pairs to be correlated at the same time. Draws the above 
#' for four different decomposition models (EWT,EMD,VMD,FD).

library(jsonlite)
library(vscDebugger)
library(tidyverse)
library(patchwork)
library(viridis)
library(ggExtra)

# Style and font parameters
plot_font_family <- "Arial"
plot_background_color <- "#F4F5F1"
plot_text_color <- "black"

# Axis titles
axis_title_size <- 22
axis_title_face <- "plain"
axis_title_margin <- margin(t = 15, r = 15, b = 15, l = 15)

# Axis tick labels
axis_text_y_size <- 21
axis_text_y_face <- "plain"
axis_text_x_size <- 21
axis_text_x_face <- "plain"
axis_text_color <- "black"

# Plot title
plot_title_size <- 40
plot_title_face <- "plain"
plot_title_hjust <- 0.5
plot_title_vjust <- 0.5
plot_title_margin <- margin(b = 20)

# Facet text
facet_text_size <- 40
facet_text_face <- "plain"

# Plot margins and spacing
plot_margin <- margin(15, 15, 15, 15)
panel_spacing <- unit(4, "lines")

# Legend text
legend_text_size <- 30
legend_title_size <- 35

# Annotation text
annotation_text_size <- 8

# Define a base theme for all plots
base_theme <- theme_minimal() +
  theme(
    text = element_text(family = plot_font_family),
    plot.background = element_rect(color = plot_background_color, fill = plot_background_color),
    panel.background = element_rect(fill = NA, color = NA),
    plot.margin = plot_margin,
    panel.spacing = panel_spacing,
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
    ),
    legend.text = element_text(
      size = legend_text_size,
      family = plot_font_family
    ),
    legend.title = element_text(
      size = legend_title_size,
      family = plot_font_family
    )
  )

decomposition <- 'ewt'
NoC <- '3'
level <- 'frame'
feature <- 'NRMSEs_correlograms'
load_dir <- file.path('..','data','decomposition_quality',decomposition)
data_file <- file.path(load_dir,paste0("dec_quality_",level,"_NoC",NoC,"_",decomposition,"_",feature,".json"))

nrmse_violins_save_dir <- file.path('..','supplementary_figures','SI_fig_9_decomposition_quality_orthogonality','NRMSE_violins',level,decomposition, paste0("NoC",  NoC))
corr_save_dir <- file.path('..','supplementary_figures','SI_fig_9_decomposition_quality_orthogonality','correlograms', decomposition, level, paste0("NoC",  NoC))
if (!dir.exists(nrmse_violins_save_dir)) {
  dir.create(nrmse_violins_save_dir, recursive = TRUE, showWarnings = FALSE)
}
if (!dir.exists(corr_save_dir)) {
  dir.create(corr_save_dir, recursive = TRUE, showWarnings = FALSE)
}
datasets <- c('sim_vowels', 'timit')

vowel_map <- c('a' = 0, 'e' = 1, 'I' = 2, 'aw' = 3, 'u' = 4)
reverse_vowel_map <- names(vowel_map)
names(reverse_vowel_map) <- vowel_map

load_json_data <- function(file_path) {
    json_text <- readLines(file_path)
    json_text <- gsub("\\bNaN\\b", "\"NaN\"", json_text)
    fromJSON(json_text, simplifyDataFrame = TRUE, simplifyMatrix = TRUE)
}

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

# Flatten all variables
df$sim_vowels$vowels <- as.vector(t(df$sim_vowels$vowels))
df$sim_vowels$overlap_mask <- as.vector(t(df$sim_vowels$overlap_mask))
df$sim_vowels$speaker_id <- as.vector(t(df$sim_vowels$speaker_id))
df$sim_vowels$gender <- ifelse(df$sim_vowels$speaker_id <= 0.98, "F",
                              ifelse(df$sim_vowels$speaker_id >= 1.02, "M", "U"))

df$sim_vowels$NRMSEs <- as.vector(t(df$sim_vowels$NRMSEs))                             
df$sim_vowels$corr_12 <- as.vector(t(df$sim_vowels$correlograms[,,1,2]))
df$sim_vowels$corr_13 <- as.vector(t(df$sim_vowels$correlograms[,,1,3]))
df$sim_vowels$corr_23 <- as.vector(t(df$sim_vowels$correlograms[,,2,3]))


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
df$timit$overlap_mask <- unlist(df$timit$overlap_mask)

df$timit$NRMSEs <- unlist(df$timit$NRMSEs)   
timit_correlograms <- df$timit$correlograms                          
df$timit$corr_12 <- unlist(sapply(timit_correlograms, function(x) x[,1,2]))
df$timit$corr_13 <- unlist(sapply(timit_correlograms, function(x) x[,1,3]))
df$timit$corr_23 <- unlist(sapply(timit_correlograms, function(x) x[,2,3]))

sim_vowels_df <- data.frame(
    dataset = "sim_vowels",
    NRMSEs = df$sim_vowels$NRMSEs,
    corr_12 = df$sim_vowels$corr_12,
    corr_13 = df$sim_vowels$corr_13,
    corr_23 = df$sim_vowels$corr_23,
    vowels = df$sim_vowels$vowels,
    speaker_id = df$sim_vowels$speaker_id,
    gender = df$sim_vowels$gender,
    overlap_mask = df$sim_vowels$overlap_mask
)

timit_df <- data.frame(
    dataset = "timit",
    NRMSEs = df$timit$NRMSEs,
    corr_12 = df$timit$corr_12,
    corr_13 = df$timit$corr_13,
    corr_23 = df$timit$corr_23,
    vowels = df$timit$vowels,
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

#Toensure consistent ordering in plots
sim_vowel_order <- c("a", "e", "I", "aw", "u")  
timit_vowel_order <- sort(unique(df$timit$vowels[df$timit$vowels != "NO"]))  

filtered_combined_df <- combined_df %>%
  filter(
    gender != "U",
    vowels != "NO",
    !is.na(corr_12),
    !is.na(corr_13), 
    !is.na(corr_23),
    NRMSEs >= 0
    ) %>%
  group_by(dataset) %>%
  mutate(vowels = case_when(
    dataset == "sim_vowels" ~ factor(vowels, levels = sim_vowel_order),
    dataset == "timit" ~ factor(vowels, levels = timit_vowel_order),
    TRUE ~ factor(vowels)
  )) %>%
  group_by(dataset, vowels, gender)

#----------------------------------------------------------------------------
#Correlation heatmaps
#----------------------------------------------------------------------------

# Function to create correlation plot for a specific vowel and dataset
create_corr_plot <- function(data, phoneme_val, dataset_name, type = "vowel") {
  # Filter and prepare data
  plot_data <- data %>%
    filter(dataset == dataset_name,
           if(type == "vowel") {
             vowels == phoneme_val
           } else {
             consonants == phoneme_val
           }) %>%
    ungroup() %>%
    select(corr_12, corr_13, corr_23) %>%
    rename(
      "Comp 1-2" = corr_12,
      "Comp 1-3" = corr_13,
      "Comp 2-3" = corr_23
    )
  
  # Calculate means
  means <- colMeans(plot_data, na.rm = TRUE)
  
  correlation_base_theme <- base_theme +
    theme(
      plot.margin = margin(5, 5, 5, 5),
      panel.spacing = unit(0, "lines"),
      aspect.ratio = 1,
      plot.background = element_rect(color = NA, fill = "white"),
      panel.background = element_rect(color = NA, fill = "white")
    )

  axis_theme <- correlation_base_theme +
    theme(
      axis.text = element_text(size = axis_text_x_size, family = plot_font_family),
      axis.title = element_text(size = axis_title_size, family = plot_font_family)
    )
  
  # Density plots for diagonal with fixed dimensions
  p11 <- ggplot(plot_data, aes(x = `Comp 1-2`)) +
    geom_density(fill = "lightblue", alpha = 0.5) +
    scale_x_continuous(limits = c(0, 1), labels = scales::number_format(accuracy = 0.1)) +
    labs(x = "Corr. 1-2", y = "PDF") +
    axis_theme

  p22 <- ggplot(plot_data, aes(x = `Comp 1-3`)) +
    geom_density(fill = "lightblue", alpha = 0.5) +
    scale_x_continuous(limits = c(0, 1), labels = scales::number_format(accuracy = 0.1)) +
    labs(x = "Corr. 1-3", y = "PDF") +
    axis_theme

  p33 <- ggplot(plot_data, aes(x = `Comp 2-3`)) +
    geom_density(fill = "lightblue", alpha = 0.5) +
    scale_x_continuous(limits = c(0, 1), labels = scales::number_format(accuracy = 0.1)) +
    labs(x = "Corr. 2-3", y = "PDF") +
    axis_theme

  # Mean value plots for upper triangle
  box_theme <- correlation_base_theme +
    theme(
      panel.grid.major = element_line(color = "white", linewidth = 0.5),
      panel.grid.minor = element_line(color = "white", linewidth = 0.25),
      axis.text.x = element_blank(),
      axis.text.y = element_blank(),
      axis.title.y = element_blank(),
      axis.title.x = element_text(size = axis_title_size, family = plot_font_family),
      axis.ticks = element_blank(),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "gray95", color = NA),
      plot.margin = margin(0.1, 0.1, 0.1, 0.1)
    )

  p12 <- ggplot() +
    # Add grid first
    geom_hline(yintercept = seq(0, 1, 0.25), color = "white", linewidth = 0.5) +
    geom_vline(xintercept = seq(0, 1, 0.25), color = "white", linewidth = 0.5) +
    # Add text
    labs(x = "Corr. 1-2") +
    annotate("text", x = 0.5, y = 0.5,
             label = sprintf("Avg. Corr: %.3f", means[1]),
             size = annotation_text_size) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    box_theme

  p13 <- ggplot() +
    # Add grid first
    geom_hline(yintercept = seq(0, 1, 0.25), color = "white", linewidth = 0.5) +
    geom_vline(xintercept = seq(0, 1, 0.25), color = "white", linewidth = 0.5) +
    # Add text
    labs(x = "Corr. 1-3") +
    annotate("text", x = 0.5, y = 0.5,
             label = sprintf("Avg. Corr: %.3f", means[2]),
             size = annotation_text_size) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    box_theme

  p23 <- ggplot() +
    # Add grid first
    geom_hline(yintercept = seq(0, 1, 0.25), color = "white", linewidth = 0.5) +
    geom_vline(xintercept = seq(0, 1, 0.25), color = "white", linewidth = 0.5) +
    # Add text
    labs(x = "Corr. 2-3") +
    annotate("text", x = 0.5, y = 0.5,
             label = sprintf("Avg. Corr: %.3f", means[3]),
             size = annotation_text_size) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    box_theme

  # Create scatter plots for lower triangle with axis labels
  p21 <- ggplot(plot_data, aes(x = `Comp 1-2`, y = `Comp 1-3`)) +
    geom_density_2d_filled(alpha = 0.8) +
    scale_fill_viridis_d(option = "plasma", begin = 0, end = 0.8) +
    scale_x_continuous(limits = c(0, 0.4), labels = scales::number_format(accuracy = 0.1)) +
    scale_y_continuous(limits = c(0, 0.4), labels = scales::number_format(accuracy = 0.1)) +
    labs(x = "Corr. 1-2", y = "Corr. 1-3") +
    axis_theme +
    theme(legend.position = "none")

  p31 <- ggplot(plot_data, aes(x = `Comp 1-2`, y = `Comp 2-3`)) +
    geom_density_2d_filled(alpha = 0.8) +
    scale_fill_viridis_d(option = "plasma", begin = 0, end = 0.8) +
    scale_x_continuous(limits = c(0, 0.4), labels = scales::number_format(accuracy = 0.1)) +
    scale_y_continuous(limits = c(0, 0.4), labels = scales::number_format(accuracy = 0.1)) +
    labs(x = "Corr. 1-2", y = "Corr. 2-3") +
    axis_theme +
    theme(legend.position = "none")

  p32 <- ggplot(plot_data, aes(x = `Comp 1-3`, y = `Comp 2-3`)) +
    geom_density_2d_filled(alpha = 0.8) +
    scale_fill_viridis_d(option = "plasma", begin = 0, end = 0.8) +
    scale_x_continuous(limits = c(0, 0.4), labels = scales::number_format(accuracy = 0.1)) +
    scale_y_continuous(limits = c(0, 0.4), labels = scales::number_format(accuracy = 0.1)) +
    labs(x = "Corr. 1-3", y = "Corr. 2-3") +
    axis_theme +
    theme(legend.position = "none")

  # Combine plots with explicit sizing
  combined_plot <- wrap_plots(
    p11, p12, p13,
    p21, p22, p23,
    p31, p32, p33,
    ncol = 3
  ) +
  plot_layout(
    heights = rep(1, 3),
    widths = rep(1, 3),
    design = "
    123
    456
    789
    "
  ) +
  plot_annotation(
    title = "", #paste0("Correlations for '", phoneme_val, "'")
    theme = theme(
      plot.title = element_text(
        size = plot_title_size, 
        hjust = plot_title_hjust, 
        margin = plot_title_margin,
        family = plot_font_family
      ),
      text = element_text(family = plot_font_family)
    )
  ) & 
  theme(
    plot.margin = margin(0.1, 0.1, 0.1, 0.1),
    panel.spacing = unit(0, "lines")
  )

  return(combined_plot)
}

# Function to create just the legend as a separate plot
create_correlation_legend <- function() {
  # Create a simple gradient data for the colorbar
  gradient_data <- data.frame(
    x = rep(1, 100),
    y = seq(0, 1, length.out = 100),
    fill_value = seq(0, 1, length.out = 100)
  )
  
  # Create the colorbar plot
  colorbar_plot <- ggplot(gradient_data, aes(x = x, y = y, fill = fill_value)) +
    geom_tile(width = 1, height = 0.01) +
    scale_fill_gradientn(
      colors = viridis::plasma(100, begin = 0, end = 0.8),
      name = "PDF",
      breaks = c(0, 0.25, 0.5, 0.75, 1),
      labels = c("0.0", "0.2", "0.5", "0.8", "1.0"),
      guide = guide_colorbar(
        title = "PDF",
        title.theme = element_text(size = legend_title_size, family = plot_font_family),
        label.theme = element_text(size = legend_text_size, family = plot_font_family),
        barwidth = unit(12, "cm"),
        barheight = unit(1.2, "cm"),
        title.position = "top",
        title.hjust = 0.5,
        label.position = "bottom",
        frame.colour = "black",
        frame.linewidth = 0.5
      )
    ) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    theme_void() +
    theme(
      legend.position = "bottom",
      legend.justification = "center",
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(40, 40, 40, 40),
      legend.margin = margin(t = 20, b = 20),
      legend.box.margin = margin(0, 0, 0, 0)
    ) +
    labs(x = NULL, y = NULL) +
    coord_cartesian(xlim = c(0.5, 1.5), ylim = c(0, 1), clip = "off")
  
  return(colorbar_plot)
}

# Create and display plot for vowel 'a'
a_vowel_plot <- create_corr_plot(filtered_combined_df, "a", "sim_vowels")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_sim_vowels_a_correlation_matrix.png")),
  a_vowel_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)

aw_vowel_plot <- create_corr_plot(filtered_combined_df, "aw", "sim_vowels")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_sim_vowels_aw_correlation_matrix.png")),
  aw_vowel_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)

e_vowel_plot <- create_corr_plot(filtered_combined_df, "e", "sim_vowels")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_sim_vowels_e_correlation_matrix.png")),
  e_vowel_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)

I_vowel_plot <- create_corr_plot(filtered_combined_df, "I", "sim_vowels")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_sim_vowels_I_correlation_matrix.png")),
  I_vowel_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)

u_vowel_plot <- create_corr_plot(filtered_combined_df, "u", "sim_vowels")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_sim_vowels_u_correlation_matrix.png")),
  u_vowel_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)



#Do the same for timit
timit_vowels <- c("iy", "eh", "ax", "aw", "ow", "uw", "y")
consonant_list <- c("b", "d", "f", "k", "l", "s")
timit_df_cons <- data.frame(
    dataset = "timit",
    NRMSEs = df$timit$NRMSEs,
    corr_12 = df$timit$corr_12,
    corr_13 = df$timit$corr_13,
    corr_23 = df$timit$corr_23,
    speaker_id = df$timit$speaker_id,
    gender = df$timit$gender,
    overlap_mask = df$timit$overlap_mask,
    consonants = df$timit$consonants
)

timit_df_cons <- as_tibble(timit_df_cons)

filtered_timit_cons_df <- timit_df_cons %>%
  filter(
    gender != "U",
    consonants != "NO",
    !is.na(corr_12),
    !is.na(corr_13), 
    !is.na(corr_23),
    NRMSEs >= 0
  ) %>%
  group_by(dataset, consonants, gender)

iy_vowel_plot <- create_corr_plot(filtered_combined_df, "iy", "timit")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_timit_iy_correlation_matrix.png")),
  iy_vowel_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)

eh_vowel_plot <- create_corr_plot(filtered_combined_df, "eh", "timit")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_timit_eh_correlation_matrix.png")),
  eh_vowel_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)
ax_vowel_plot <- create_corr_plot(filtered_combined_df, "ax", "timit")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_timit_ax_correlation_matrix.png")),
  ax_vowel_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)

aw_timit_plot <- create_corr_plot(filtered_combined_df, "aw", "timit")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_timit_aw_correlation_matrix.png")),
  aw_vowel_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)
ow_vowel_plot <- create_corr_plot(filtered_combined_df, "ow", "timit")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_timit_ow_correlation_matrix.png")),
  ow_vowel_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)
uw_vowel_plot <- create_corr_plot(filtered_combined_df, "uw", "timit")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_timit_uw_correlation_matrix.png")),
  uw_vowel_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)
y_vowel_plot <- create_corr_plot(filtered_combined_df, "y", "timit")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_timit_y_correlation_matrix.png")),
  y_vowel_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)



b_cons_plot <- create_corr_plot(filtered_timit_cons_df, "b", "timit", type = "consonant")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_timit_b_correlation_matrix.png")),
  b_cons_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)
d_cons_plot <- create_corr_plot(filtered_timit_cons_df, "d", "timit", type = "consonant")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_timit_d_correlation_matrix.png")),
  d_cons_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)
f_cons_plot <- create_corr_plot(filtered_timit_cons_df, "f", "timit", type = "consonant")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_timit_f_correlation_matrix.png")),
  f_cons_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)
k_cons_plot <- create_corr_plot(filtered_timit_cons_df, "k", "timit", type = "consonant")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_timit_k_correlation_matrix.png")),
  k_cons_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)
l_cons_plot <- create_corr_plot(filtered_timit_cons_df, "l", "timit", type = "consonant")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_timit_l_correlation_matrix.png")),
  l_cons_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)
s_cons_plot <- create_corr_plot(filtered_timit_cons_df, "s", "timit", type = "consonant")
ggsave(
  file.path(corr_save_dir,paste0(decomposition, "_timit_s_correlation_matrix.png")),
  s_cons_plot,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"  # Explicit white background
)

#----------------------------------------------------------------------------
#NRMSE Violin Plots
#----------------------------------------------------------------------------

# Combine plots using patchwork
#combined_violins <- wrap_plots(violin_plots, ncol = 2) +
#  plot_layout(guides = "collect") &
#  theme(legend.position = "right")

# Create violin plots for different consonants
#consonant_list <- c("b", "d", "f", "k", "l", "s")
#consonant_plots <- map(consonant_list, ~create_consonant_violin_plot(., timit_df_cons))


white_theme <- theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA)
  )

# Case 1: Within gender comparison for vowels
create_vowel_violin_plot <- function(data, gender_val, plot_title) {
  vowel_pairs_map <- data.frame(
    original = c("I", "e", "a", "aw", "u"),
    timit = c("iy", "eh", "ax", "ow", "uw"),
    pair_id = 1:5
  )
  
  filtered_df <- data %>%
    filter(
      gender == gender_val,
      vowels != "NO",
      (dataset == "sim_vowels" & vowels %in% vowel_pairs_map$original) |
      (dataset == "timit" & vowels %in% vowel_pairs_map$timit)
    ) %>%
    mutate(
      dataset = factor(dataset, levels = c("sim_vowels", "timit")),
      pair_id = case_when(
        dataset == "sim_vowels" ~ match(vowels, vowel_pairs_map$original),
        dataset == "timit" ~ match(vowels, vowel_pairs_map$timit)
      )
    )

  ggplot(filtered_df, aes(x = factor(pair_id), y = NRMSEs, fill = dataset)) +
    geom_violin(position = position_dodge(width = 0.7),
                alpha = 0.7, scale = "width") +
    scale_fill_manual(
      values = c(
        "sim_vowels" = "#1E88E5",
        "timit" = "#FFC107"
      ),
      labels = c(
        "sim_vowels" = "Synthetic",
        "timit" = "TIMIT"
      )
    ) +
    scale_x_discrete(
      labels = paste(vowel_pairs_map$original, "-", vowel_pairs_map$timit)
    ) +
    labs(
      x = "Vowel Pairs",
      y = "NRMSE",
      fill = "Dataset",
      title = plot_title
    ) +
    base_theme +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid.major.x = element_blank()
    )
}

# Case 2: Within dataset comparison for vowels
create_vowel_dataset_violin_plot <- function(data, dataset_name, plot_title) {
  filtered_df <- data %>%
    filter(
      dataset == dataset_name,
      vowels != "NO"
    ) %>%
    mutate(
      vowels = as.factor(vowels),
      gender = factor(gender, levels = c("F", "M"))
    )

  ggplot(filtered_df, aes(x = vowels, y = NRMSEs, fill = gender)) +
    geom_violin(position = position_dodge(width = 0.7),
                alpha = 0.7, scale = "width") +
    scale_fill_manual(
      values = c(
        "F" = "#D81B60",
        "M" = "#1E88E5"
      ),
      labels = c(
        "F" = "Female",
        "M" = "Male"
      )
    ) +
    labs(
      x = "Vowels",
      y = "NRMSE",
      fill = "Gender",
      title = plot_title
    ) +
    base_theme +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA)
    )
}

# Case 1: Within gender comparison for consonants
create_consonant_violin_plot <- function(data, gender_val, plot_title) {
  filtered_df <- data %>%
    filter(
      gender == gender_val,
      consonants != "NO"
    ) %>%
    mutate(
      consonants = as.factor(consonants)
    )

  ggplot(filtered_df, aes(x = consonants, y = NRMSEs)) +
    geom_violin(fill = "#FFC107",
                alpha = 0.7, scale = "width") +
    labs(
      x = "Consonants",
      y = "NRMSE",
      title = plot_title
    ) +
    base_theme +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA)
    )
}

# Case 2: Within dataset comparison for consonants (TIMIT only)
create_consonant_gender_violin_plot <- function(data, plot_title) {
  filtered_df <- data %>%
    filter(
      consonants != "NO"
    ) %>%
    mutate(
      consonants = as.factor(consonants),
      gender = factor(gender, levels = c("F", "M"))
    )

  ggplot(filtered_df, aes(x = consonants, y = NRMSEs, fill = gender)) +
    geom_violin(position = position_dodge(width = 0.7),
                alpha = 0.7, scale = "width") +
    scale_fill_manual(
      values = c(
        "F" = "#D81B60",
        "M" = "#1E88E5"
      ),
      labels = c(
        "F" = "Female",
        "M" = "Male"
      )
    ) +
    labs(
      x = "Consonants",
      y = "NRMSE",
      fill = "Gender",
      title = plot_title
    ) +
    base_theme +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA)
    )
}

# Example usage:
# Case 1: Within gender plots
female_vowels <- create_vowel_violin_plot(filtered_combined_df, "F", "Female Vowel NRMSEs Across Datasets")
male_vowels <- create_vowel_violin_plot(filtered_combined_df, "M", "Male Vowel NRMSEs Across Datasets")
female_consonants <- create_consonant_violin_plot(filtered_timit_cons_df, "F", "Female Consonant NRMSEs in TIMIT")
male_consonants <- create_consonant_violin_plot(filtered_timit_cons_df, "M", "Male Consonant NRMSEs in TIMIT")

# Case 2: Within dataset plots
sim_vowels_plot <- create_vowel_dataset_violin_plot(filtered_combined_df, "sim_vowels", "Synthetic Vowels NRMSEs by Gender")
timit_vowels_plot <- create_vowel_dataset_violin_plot(filtered_combined_df, "timit", "TIMIT Vowels NRMSEs by Gender")
timit_consonants_plot <- create_consonant_gender_violin_plot(filtered_timit_cons_df, "TIMIT Consonants NRMSEs by Gender")

# Save plots
ggsave(file.path(nrmse_violins_save_dir,"female_vowels_comparison.png"), female_vowels, width = 15, height = 10, dpi = 300, bg = "white")
ggsave(file.path(nrmse_violins_save_dir,"male_vowels_comparison.png"), male_vowels, width = 15, height = 10, dpi = 300, bg = "white")
ggsave(file.path(nrmse_violins_save_dir,"female_consonants.png"), female_consonants, width = 15, height = 10, dpi = 300, bg = "white")
ggsave(file.path(nrmse_violins_save_dir,"male_consonants.png"), male_consonants, width = 15, height = 10, dpi = 300, bg = "white")
ggsave(file.path(nrmse_violins_save_dir,"sim_vowels_gender.png"), sim_vowels_plot, width = 15, height = 10, dpi = 300, bg = "white")
ggsave(file.path(nrmse_violins_save_dir,"timit_vowels_gender.png"), timit_vowels_plot, width = 15, height = 10, dpi = 300, bg = "white")
ggsave(file.path(nrmse_violins_save_dir,"timit_consonants_gender.png"), timit_consonants_plot, width = 15, height = 10, dpi = 300, bg = "white")


create_vowel_pair_violin_plot <- function(data, orig_vowel, timit_vowel, plot_title) {
  # Filter data for the specific vowel pair
  filtered_df <- data %>%
    filter(
      (dataset == "sim_vowels" & vowels == orig_vowel) |
      (dataset == "timit" & vowels == timit_vowel),
      gender != "U"
    ) %>%
    mutate(
      dataset = factor(dataset, levels = c("sim_vowels", "timit")),
      gender = factor(gender, levels = c("F", "M")),
      # Create combined factor for dataset-gender combination
      dataset_gender = factor(paste(dataset, gender))
    )

  ggplot(filtered_df, aes(x = dataset_gender, y = NRMSEs, fill = dataset_gender)) +
    geom_violin(alpha = 0.7, scale = "width") +
    scale_fill_manual(
      values = c(
        "sim_vowels F" = "#FF69B4",  # Pink for synthetic female
        "sim_vowels M" = "#4169E1",  # Royal blue for synthetic male
        "timit F" = "#FF8C00",       # Dark orange for TIMIT female
        "timit M" = "#228B22"        # Forest green for TIMIT male
      ),
      labels = c(
        "sim_vowels F" = "Synthetic Female",
        "sim_vowels M" = "Synthetic Male",
        "timit F" = "TIMIT Female",
        "timit M" = "TIMIT Male"
      )
    ) +
    scale_x_discrete(
      labels = c(
        "sim_vowels F" = paste0(orig_vowel, " (F)"),
        "sim_vowels M" = paste0(orig_vowel, " (M)"),
        "timit F" = paste0(timit_vowel, " (F)"),
        "timit M" = paste0(timit_vowel, " (M)")
      )
    ) +
    labs(
      x = "Dataset-Gender Combination",
      y = "NRMSE",
      fill = "Dataset-Gender",
      title = plot_title
    ) +
    base_theme +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid.major.x = element_blank()
    )
}

vowel_pairs <- list(
  c("I", "iy"),
  c("e", "eh"),
  c("a", "ax"),
  c("aw", "ow"),
  c("u", "uw"),
  c("u", "y")
)

# Create and save plots for each vowel pair
for (pair in vowel_pairs) {
  orig_vowel <- pair[1]
  timit_vowel <- pair[2]
  plot_title <- sprintf("NRMSE Distribution for Vowel Pair '%s-%s'", orig_vowel, timit_vowel)
  
  pair_plot <- create_vowel_pair_violin_plot(
    filtered_combined_df, 
    orig_vowel, 
    timit_vowel, 
    plot_title
  )
  
  ggsave(
    file.path(nrmse_violins_save_dir,
      paste0("vowel_pair_", orig_vowel, "_", timit_vowel, ".png")
    ),
    pair_plot,
    width = 15,
    height = 10,
    dpi = 600,
    bg = "white"
  )
}

# Create and save the legend separately
correlation_legend <- create_correlation_legend()
ggsave(
  file.path(corr_save_dir, "correlation_legend.png"),
  correlation_legend,
  width = 8,
  height = 2,
  dpi = 600,
  bg = "white"
)