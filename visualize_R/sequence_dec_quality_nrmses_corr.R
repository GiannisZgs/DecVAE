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
library(GGally)
library(gridExtra)
library(grid)

decomposition <- 'emd'
NoC <- '3'
level <- 'sequence'
feature <- 'NRMSEs_correlograms'
data_path <- file.path(paste0("/home/giannis/Documents/data_for_figures/decomposition_quality"),decomposition)
datasets <- c('sim_vowels', 'timit')

"Set plot styling options"
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


synth_data_file <- paste0(data_path,"/dec_quality_",level,"_NoC",NoC,"_",decomposition,"_",feature,".json")
df <- load_json_data(synth_data_file)

"Prepare data for sim_vowels"
df$sim_vowels$gender <- df$sim_vowels$gender[1:4000]
df$sim_vowels$consonants <- NULL
df$sim_vowels$speaker_id <- as.vector(t(df$sim_vowels$speaker_id))
df$sim_vowels$gender <- ifelse(df$sim_vowels$speaker_id <= 0.98, "F",
                              ifelse(df$sim_vowels$speaker_id >= 1.02, "M", "U"))
df$sim_vowels$NRMSEs <- as.vector(t(df$sim_vowels$NRMSEs))                             
df$sim_vowels$corr_12 <- as.vector(t(df$sim_vowels$correlograms[,1,2]))
df$sim_vowels$corr_13 <- as.vector(t(df$sim_vowels$correlograms[,1,3]))
df$sim_vowels$corr_23 <- as.vector(t(df$sim_vowels$correlograms[,2,3]))

"Prepare data for TIMIT"
df$timit$speaker_id <- as.vector(t(df$timit$speaker_id))
df$timit$gender <- as.vector(t(df$timit$gender))
df$timit$NRMSEs <- as.vector(t(df$timit$NRMSEs))                            
df$timit$corr_12 <- as.vector(t(df$timit$correlograms[,1,2]))
df$timit$corr_13 <- as.vector(t(df$timit$correlograms[,1,3]))
df$timit$corr_23 <- as.vector(t(df$timit$correlograms[,2,3]))

sim_vowels_df <- data.frame(
    dataset = "sim_vowels",
    NRMSEs = df$sim_vowels$NRMSEs,
    corr_12 = df$sim_vowels$corr_12,
    corr_13 = df$sim_vowels$corr_13,
    corr_23 = df$sim_vowels$corr_23,
    speaker_id = df$sim_vowels$speaker_id,
    gender = df$sim_vowels$gender
)

timit_df <- data.frame(
    dataset = "timit",
    NRMSEs = df$timit$NRMSEs,
    corr_12 = df$timit$corr_12,
    corr_13 = df$timit$corr_13,
    corr_23 = df$timit$corr_23,
    speaker_id = df$timit$speaker_id,
    gender = df$timit$gender
)

combined_df <- rbind(sim_vowels_df, timit_df)

combined_df <- as_tibble(combined_df)

filtered_combined_df <- combined_df %>%
  filter(
    gender != "U",
    !is.na(corr_12),
    !is.na(corr_13), 
    !is.na(corr_23),
    NRMSEs >= 0
    ) %>%
  group_by(dataset, gender)

#----------------------------------------------------------------------------
#Correlation heatmaps
#----------------------------------------------------------------------------

# Function to create correlation plot for a specific vowel and dataset
create_corr_plot <- function(data, dataset_name, gender_val = NULL, plot_title = NULL) {
  # Filter and prepare data
  plot_data <- data %>%
    filter(dataset == dataset_name) %>%
    {if (!is.null(gender_val)) filter(., gender == gender_val) else .} %>%
    ungroup() %>%
    select(corr_12, corr_13, corr_23) %>%
    rename(
      "Comp 1-2" = corr_12,
      "Comp 1-3" = corr_13,
      "Comp 2-3" = corr_23
    )
  
  # Create automatic title if none provided
  if (is.null(plot_title)) {
    plot_title <- if (is.null(gender_val)) {
      paste0("Correlations for ", dataset_name)
    } else {
      paste0("Correlations for ", dataset_name, " (", gender_val, ")")
    }
  }
  
  # Calculate means
  means <- colMeans(plot_data, na.rm = TRUE)
  
  base_theme <- theme_minimal(base_family = font) +
    theme(
      plot.margin = margin(0.1, 0.1, 0.1, 0.1),
      panel.spacing = unit(0, "lines"),
      aspect.ratio = 1,
      plot.background = element_rect(color = NA, fill = "white"),  # Explicit white background
      panel.background = element_rect(color = NA, fill = "white"),
      axis.text = element_text(size = 40),         # Larger tick labels
      axis.title = element_text(size = 40),        # Larger axis titles
      plot.title = element_text(size = 40)         # Larger plot title
    )

  axis_theme <- base_theme +
    theme(
      axis.text = element_text(size = 40),
      axis.title = element_text(size = 40)
    )
  
  # Density plots for diagonal with fixed dimensions
  p11 <- ggplot(plot_data, aes(x = `Comp 1-2`)) +
    geom_density(fill = "lightblue", alpha = 0.5) +
    scale_x_continuous(limits = c(0, 1)) +
    labs(x = "Components Corr. 1-2") +
    axis_theme

  p22 <- ggplot(plot_data, aes(x = `Comp 1-3`)) +
    geom_density(fill = "lightblue", alpha = 0.5) +
    scale_x_continuous(limits = c(0, 1)) +
    labs(x = "Components Corr. 1-3") +
    axis_theme

  p33 <- ggplot(plot_data, aes(x = `Comp 2-3`)) +
    geom_density(fill = "lightblue", alpha = 0.5) +
    scale_x_continuous(limits = c(0, 1)) +
    labs(x = "Components Corr. 2-3") +
    axis_theme

  # Mean value plots for upper triangle
  box_theme <- base_theme +
    theme(
      panel.grid.major = element_line(color = "white", linewidth = 0.5),
      panel.grid.minor = element_line(color = "white", linewidth = 0.25),
      axis.text = element_blank(),
      axis.title = element_text(size = 40),
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
    labs(x = "Components 1-2", y = NULL) + 
    annotate("text", x = 0.5, y = 0.5,
             label = sprintf("Avg. Corr: %.3f", means[1]),
             size = 20) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    box_theme

  p13 <- ggplot() +
    # Add grid first
    geom_hline(yintercept = seq(0, 1, 0.25), color = "white", linewidth = 0.5) +
    geom_vline(xintercept = seq(0, 1, 0.25), color = "white", linewidth = 0.5) +
    # Add text
    labs(x = "Components 1-3", y = NULL) +
    annotate("text", x = 0.5, y = 0.5,
             label = sprintf("Avg. Corr: %.3f", means[2]),
             size = 20) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    box_theme

  p23 <- ggplot() +
    # Add grid first
    geom_hline(yintercept = seq(0, 1, 0.25), color = "white", linewidth = 0.5) +
    geom_vline(xintercept = seq(0, 1, 0.25), color = "white", linewidth = 0.5) +
    # Add text
    labs(x = "Components 2-3", y = NULL) + 
    annotate("text", x = 0.5, y = 0.5,
             label = sprintf("Avg. Corr: %.3f", means[3]),
             size = 20) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    box_theme

  # Create scatter plots for lower triangle with axis labels
  p21 <- ggplot(plot_data, aes(x = `Comp 1-2`, y = `Comp 1-3`)) +
    geom_density_2d_filled(alpha = 0.7) +
    scale_x_continuous(limits = c(0, 0.25)) +
    scale_y_continuous(limits = c(0, 0.25)) +
    labs(x = "Corr. 1-2", y = "Corr. 1-3") +
    axis_theme +
    theme(legend.position = "none")  # Remove legend

  p31 <- ggplot(plot_data, aes(x = `Comp 1-2`, y = `Comp 2-3`)) +
    geom_density_2d_filled(alpha = 0.7) +
    scale_x_continuous(limits = c(0, 0.25)) +
    scale_y_continuous(limits = c(0, 0.25)) +
    labs(x = "Corr. 1-2", y = "Corr. 2-3") +
    axis_theme +
    theme(legend.position = "none")  # Remove legend

  p32 <- ggplot(plot_data, aes(x = `Comp 1-3`, y = `Comp 2-3`)) +
    geom_density_2d_filled(alpha = 0.7) +
    scale_x_continuous(limits = c(0, 0.25)) +
    scale_y_continuous(limits = c(0, 0.25)) +
    labs(x = "Corr. 1-3", y = "Corr. 2-3") +
    axis_theme +
    theme(legend.position = "none")  # Remove legend


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
    guides = "collect",
    design = "
    123
    456
    789
    "
  ) +
  plot_annotation(
    title <- if (is.null(gender_val)) {
      paste0("Correlations for ", dataset_name)
    } else {
      paste0("Correlations for ", dataset_name, " (", gender_val, ")")
    },
    theme = theme(
      plot.title = element_text(size = 40, hjust = 0.5, margin = margin(b = 10)),
      text = element_text(family = font)
    )
  ) & 
  theme(
    plot.margin = margin(0.1, 0.1, 0.1, 0.1),
    panel.spacing = unit(0, "lines")
  )

  return(combined_plot)
}

# Create plots for all dataset-gender combinations
datasets <- c("sim_vowels", "timit")
genders <- c("F", "M")

# Plot for each dataset (all genders)
for (dataset in datasets) {
  plot <- create_corr_plot(filtered_combined_df, dataset)
  ggsave(
    paste0("decomposition_quality_", level, "/", decomposition, "/correlograms/", 
           decomposition, "_", dataset, "_all_genders_correlation_matrix.png"),
    plot,
    width = 10, height = 10, dpi = 300, bg = "white"
  )
}

# Plot for each dataset-gender combination
for (dataset in datasets) {
  for (gender in genders) {
    plot <- create_corr_plot(filtered_combined_df, dataset, gender)
    ggsave(
      paste0("decomposition_quality_", level, "/", decomposition, "/correlograms/", 
             decomposition, "_", dataset, "_", gender, "_correlation_matrix.png"),
      plot,
      width = 10, height = 10, dpi = 300, bg = "white"
    )
  }
}

#----------------------------------------------------------------------------
#NRMSE Violin Plots
#----------------------------------------------------------------------------
white_theme <- theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA)
  )
create_violin_plot <- function(data, plot_title) {
  filtered_df <- data %>%
    filter(
      gender != "U"
    ) %>%
    mutate(
      dataset = factor(dataset, levels = c("sim_vowels", "timit")),
      gender = factor(gender, levels = c("F", "M")),
      dataset_gender = factor(paste(dataset, gender),
                            levels = c("sim_vowels F", "sim_vowels M",
                                     "timit F", "timit M"))
    )

  ggplot(filtered_df, aes(x = dataset_gender, y = NRMSEs, fill = dataset_gender)) +
    geom_violin(alpha = 0.7, scale = "width") +
    scale_fill_manual(
      values = c(
        "sim_vowels F" = "#FF69B4",
        "sim_vowels M" = "#4169E1",
        "timit F" = "#FF8C00",
        "timit M" = "#228B22"
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
        "sim_vowels F" = "Synthetic (F)",
        "sim_vowels M" = "Synthetic (M)",
        "timit F" = "TIMIT (F)",
        "timit M" = "TIMIT (M)"
      )
    ) +
    labs(
      x = "Dataset-Gender Combination",
      y = "NRMSE",
      fill = "Dataset-Gender",
      title = plot_title
    ) +
    white_theme +
    theme(
      plot.title = element_text(size = 40, face = "bold", hjust = 0.5),
      axis.text.x = element_text(size = 40, angle = 45, hjust = 1),
      axis.text.y = element_text(size = 40),
      axis.title = element_text(size = 40),
      legend.text = element_text(size = 30),
      legend.title = element_text(size = 35),
      panel.grid.major.x = element_blank(),
      legend.position = "right",  # Position legend on the right
      legend.box.margin = margin(0, 0, 0, 20)  # Add margin to separate legend from plot
    )
}

# Create and save the plot
combined_plot <- create_violin_plot(
  filtered_combined_df,
  "NRMSE Distribution Across Datasets and Genders"
)

# Save the plot
ggsave(
  paste0(
    "decomposition_quality_", level, "/",
    decomposition,
    "/NRMSE_violins/combined_dataset_gender_distribution.png"
  ),
  combined_plot,
  width = 15,
  height = 10,
  dpi = 300,
  bg = "white"
)