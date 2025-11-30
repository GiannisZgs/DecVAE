#' Figure 6: this script generates Fig.6d. It demonstrates the overall disentanglement performance
#' of VAE-based and DecVAE-based models across multiple datasets using circular bar charts.

library(dplyr)
library(ggplot2)
library(stringr) 
library(tidyr) 
library(RColorBrewer)
library(viridis)
library(gridExtra)
library(grid)

# Load the data from the existing file
source(file.path('fig_6', 'models_performance_across_all_datasets_disent.R'))

# Set save directory
save_dir <- file.path('..', 'figures', 'fig_6', 'fig_6d_circular_barcharts_datasets')
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

# Font and text size parameters
plot_font_family <- "Arial"
dataset_text_size <- 7  # Parameter to control dataset label text size
model_label_size <- 5.5  # Parameter to control model label text size
transfer_data_text_size <- 6.5  # Parameter to control transfer dataset label text size

# Function to prepare data for circular bar chart
prepare_circular_data <- function(datasets_list, model_type = "VAE") {
  all_data <- data.frame()
  
  for (dataset_name in names(datasets_list)) {
    dataset_info <- datasets_list[[dataset_name]]
    model_data <- dataset_info$data
    selected_models <- dataset_info$models
    
    # Get display names mapping
    display_names_var <- paste0("model_display_names_", tolower(gsub("-", "_", dataset_name)))
    if (exists(display_names_var)) {
      display_mapping <- get(display_names_var)
    } else {
      display_mapping <- setNames(selected_models, selected_models)
    }
    
    # Filter for selected models based on type
    if (model_type == "VAE") {
      vae_models <- selected_models[grepl("VAE|betaVAE", selected_models) & 
                                   !grepl("DecVAE|betaDecVAE", selected_models)]
    } else {
      vae_models <- selected_models[grepl("DecVAE|betaDecVAE", selected_models)]
    }
    
    if (length(vae_models) == 0) next
    
    # Prepare data for this dataset
    dataset_long <- model_data %>%
      filter(metric %in% selected_metrics_all) %>%
      select(metric, all_of(vae_models)) %>%
      pivot_longer(cols = -metric, names_to = "model", values_to = "value") %>%
      mutate(
        # Transform gaussian_corr_norm to 1 - gaussian_corr_norm
        value = ifelse(metric == "gaussian_corr_norm", 1 - value, value),
        # Ensure values are between 0 and 1
        value = pmax(0, pmin(1, value)),
        dataset = dataset_name,
        model_metric = paste(model, metric, sep = "_"),
        # Apply display mapping and fix formatting issues
        model_display = ifelse(model %in% names(display_mapping),
                              display_mapping[model], model),
        # Fix specific formatting issues
        model_display = case_when(
          # Fix β-VAE formatting
          grepl("betaVAE", model) ~ gsub("β-VAE", "β-VAE", model_display),
          # Fix DecVAE formatting (replace underscores with spaces and +)
          grepl("DecVAE_FD", model) ~ gsub("DecVAE_FD", "DecVAE + FD", model_display),
          grepl("DecVAE_EWT", model) ~ gsub("DecVAE_EWT", "DecVAE + EWT", model_display),
          grepl("betaDecVAE_FD", model) ~ gsub("betaDecVAE_FD", "β-DecVAE + FD", model_display),
          grepl("betaDecVAE_EWT", model) ~ gsub("betaDecVAE_EWT", "β-DecVAE + EWT", model_display),
          TRUE ~ model_display
        ),
        # Use proper metric labels from selected_metrics_all_labels
        metric_label = ifelse(metric %in% names(selected_metrics_all_labels),
                             selected_metrics_all_labels[metric], metric),
        model_metric_display = paste(model_display, metric_label, sep = " - ")
      )
    
    all_data <- rbind(all_data, dataset_long)
  }
  
  return(all_data)
}

# Create datasets list
datasets_list <- list(
  "SimVowels" = list(data = model_data_vowels, models = selected_models_vowels),
  "TIMIT" = list(data = model_data_timit, models = selected_models_timit),
  "VOC-ALS" = list(data = model_data_voc_als, models = selected_models_voc_als),
  "IEMOCAP" = list(data = model_data_iemocap, models = selected_models_iemocap)
)

# Prepare data for both types
vae_data <- prepare_circular_data(datasets_list, "VAE")
decvae_data <- prepare_circular_data(datasets_list, "DecVAE")

# Function to create circular bar chart in the reference style
create_circular_barchart <- function(data, title) {
  if (nrow(data) == 0) {
    return(ggplot() + 
           annotate("text", x = 0, y = 0, label = "No data found", size = 6) +
           theme_void() +
           labs(title = title))
  }
  
  # Set number of empty bars between groups
  empty_bar <- 3
  
  # Get unique datasets (main groups)
  datasets <- unique(data$dataset)
  
  # Add empty bars between dataset groups
  to_add <- data.frame()
  for (dataset_name in datasets) {
    empty_rows <- data.frame(
      metric = rep(NA, empty_bar),
      model = rep(NA, empty_bar),
      value = rep(NA, empty_bar),
      dataset = rep(dataset_name, empty_bar),
      model_metric = rep(NA, empty_bar),
      model_display = rep(NA, empty_bar),
      metric_label = rep(NA, empty_bar),  # Add missing column
      model_metric_display = rep(NA, empty_bar)
    )
    to_add <- rbind(to_add, empty_rows)
  }
  
  # Combine data with empty bars and arrange by dataset, then by model-metric
  plot_data <- rbind(data, to_add) %>%
    arrange(dataset, model_metric) %>%
    mutate(id = row_number())
  
  # Prepare label data
  label_data <- plot_data
  number_of_bar <- nrow(label_data)
  angle <- 90 - 360 * (label_data$id - 0.5) / number_of_bar
  label_data$hjust <- ifelse(angle < -90, 1, 0)
  label_data$angle <- ifelse(angle < -90, angle + 180, angle)
  
  # Prepare base data for dataset group labels
  base_data <- plot_data %>% 
    filter(!is.na(value)) %>%
    group_by(dataset) %>% 
    summarize(start = min(id), end = max(id), .groups = 'drop') %>% 
    rowwise() %>% 
    mutate(title = mean(c(start, end)))
  
  # Prepare grid data for separating dataset groups
  grid_data <- base_data
  if (nrow(grid_data) > 1) {
    grid_data$end <- c(grid_data$end[-1], grid_data$end[1]) + 1
    grid_data$start <- grid_data$start - 1
    grid_data <- grid_data[-1, ]
  }
  
  # Create sophisticated color mapping using continuous viridis
  # Determine if this is VAE or DecVAE data
  is_vae_data <- any(grepl("VAE", data$model_display) & !grepl("DecVAE", data$model_display))
  
  # Choose color palette based on model type
  if (is_vae_data) {
    # Cold colors for VAEs (blues, greens)
    color_palette <- viridis(100, option = "viridis")[1:70]  # Use cooler part of viridis
  } else {
    # Hot colors for DecVAEs (reds, oranges, yellows)
    color_palette <- viridis(100, option = "plasma")[30:100]  # Use warmer part of plasma
  }
  
  # Create color mapping for model-metric combinations
  all_combinations <- unique(data$model_metric_display)
  n_combinations <- length(all_combinations)
  
  # Assign colors from the selected palette
  color_indices <- seq(1, length(color_palette), length.out = n_combinations)
  color_mapping <- setNames(color_palette[round(color_indices)], all_combinations)

  # Create the plot
  p <- ggplot(plot_data, aes(x = id, y = value, fill = model_metric_display)) +
    
    # Add complete circular grid lines in background with lighter grey color
    geom_segment(aes(x = 1, y = 1.0, xend = max(plot_data$id), yend = 1.0), 
                 colour = "grey70", alpha = 0.6, size = 0.5, inherit.aes = FALSE) +
    geom_segment(aes(x = 1, y = 0.8, xend = max(plot_data$id), yend = 0.8), 
                 colour = "grey70", alpha = 0.6, size = 0.5, inherit.aes = FALSE) +
    geom_segment(aes(x = 1, y = 0.6, xend = max(plot_data$id), yend = 0.6), 
                 colour = "grey70", alpha = 0.6, size = 0.5, inherit.aes = FALSE) +
    geom_segment(aes(x = 1, y = 0.4, xend = max(plot_data$id), yend = 0.4), 
                 colour = "grey70", alpha = 0.6, size = 0.5, inherit.aes = FALSE) +
    geom_segment(aes(x = 1, y = 0.2, xend = max(plot_data$id), yend = 0.2), 
                 colour = "grey70", alpha = 0.6, size = 0.5, inherit.aes = FALSE) +
    geom_segment(aes(x = 1, y = 0.0, xend = max(plot_data$id), yend = 0.0), 
                 colour = "grey70", alpha = 0.6, size = 0.5, inherit.aes = FALSE) +
    
    # Add bars on top of grid lines
    geom_bar(stat = "identity", alpha = 0.8, width = 0.9) +
    
    # Add grid value labels - include 0 and 1.0 with better positioning
    annotate("text", x = rep(max(plot_data$id, na.rm = TRUE), 6), 
             y = c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), 
             label = c("0.0", "0.2", "0.4", "0.6", "0.8", "1.0"), 
             color = "black", size = 5, angle = 0, fontface = "bold", hjust = 1,
             vjust = 0.5, family = plot_font_family) +
    
    # Add model-metric labels with cleaned names (remove transfer info)
    geom_text(data = filter(label_data, !is.na(value)), 
              aes(x = id, y = value + 0.12, 
                  label = paste(
                    gsub("(TIMIT |Vowels |SimVowels )", "", model_display), 
                    metric_label
                  ), 
                  hjust = hjust), 
              color = "black", fontface = "plain", alpha = 0.8, size = model_label_size, 
              family = plot_font_family,
              angle = label_data$angle[!is.na(label_data$value)], inherit.aes = FALSE) +
    
    # Add base lines for dataset groups with moved closer to center
    {if (nrow(base_data) > 0) {
      list(
        geom_segment(data = base_data, aes(x = start, y = -0.05, xend = end, yend = -0.05), 
                     colour = "black", alpha = 0.8, size = 1.2, inherit.aes = FALSE),
        geom_text(data = base_data, 
                  aes(x = title, y = -0.18, label = dataset, 
                      angle = case_when(
                        dataset %in% c("IEMOCAP", "TIMIT") ~ -40,
                        dataset %in% c("VOC-ALS", "SimVowels") ~ 50,
                        TRUE ~ 0
                      )), 
                  hjust = 0.5, colour = "black", alpha = 0.9, size = dataset_text_size, 
                  fontface = "bold", family = plot_font_family, inherit.aes = FALSE)
      )
    }} +
    
    # Styling without legend - reduced upper limit since transfer labels are removed
    scale_fill_manual(values = color_mapping, na.value = "transparent") +
    scale_x_continuous(limits = c(0.5, max(plot_data$id) + 0.5), expand = c(0, 0)) +
    ylim(-0.5, 1.4) +  # Reduced upper limit since transfer segments are removed
    theme_minimal() +
    theme(
      legend.position = "none",  # Remove legend completely
      axis.text = element_blank(),
      axis.title = element_blank(),
      panel.grid = element_blank(),
      plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm"),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold", 
                               margin = margin(b = 20), family = plot_font_family),
      text = element_text(family = plot_font_family)
    ) +
    coord_polar(theta = "x") +
    labs(title = title)
  
  return(p)
}

# Create the plots without separate color arguments
if (nrow(vae_data) > 0) {
  p_vae <- create_circular_barchart(vae_data, "")
} else {
  p_vae <- ggplot() + 
    annotate("text", x = 0, y = 0, label = "No VAE models found", size = 6) +
    theme_void() +
    labs(title = "VAE-based Models Performance Across Datasets")
}

if (nrow(decvae_data) > 0) {
  p_decvae <- create_circular_barchart(decvae_data, "")
} else {
  p_decvae <- ggplot() + 
    annotate("text", x = 0, y = 0, label = "No DecVAE models found", size = 6) +
    theme_void() +
    labs(title = "DecVAE-based Models Performance Across Datasets")
}

# Save individual plots
ggsave(file.path(save_dir, "circular_barchart_VAE_models.png"), 
       p_vae, width = 16, height = 16, dpi = 300, bg = "white")

ggsave(file.path(save_dir, "circular_barchart_DecVAE_models.png"), 
       p_decvae, width = 16, height = 16, dpi = 300, bg = "white")

# Create combined plot
combined_plot <- grid.arrange(p_vae, p_decvae, ncol = 2)

ggsave(file.path(save_dir, "circular_barchart_combined.png"), 
       combined_plot, width = 32, height = 16, dpi = 300, bg = "white")

cat("Circular bar charts saved to:", save_dir, "\n")
cat("- VAE models chart: circular_barchart_VAE_models.png\n")
cat("- DecVAE models chart: circular_barchart_DecVAE_models.png\n") 
cat("- Combined chart: circular_barchart_combined.png\n")

# Print data summary
if (nrow(vae_data) > 0 || nrow(decvae_data) > 0) {
  cat("\nData Summary:\n")
  cat("VAE model-metric combinations:", ifelse(nrow(vae_data) > 0, n_distinct(vae_data$model_metric_display), 0), "\n")
  cat("DecVAE model-metric combinations:", ifelse(nrow(decvae_data) > 0, n_distinct(decvae_data$model_metric_display), 0), "\n")
  cat("Total datasets:", n_distinct(c(vae_data$dataset, decvae_data$dataset)), "\n")
} else {
  p_vae <- ggplot() + 
    annotate("text", x = 0, y = 0, label = "No VAE models found", size = 6) +
    theme_void() +
    labs(title = "" )
} #"VAE-based Models Performance Across Datasets"

if (nrow(decvae_data) > 0) {
  p_decvae <- create_circular_barchart(decvae_data, "")
} else {
  p_decvae <- ggplot() + 
    annotate("text", x = 0, y = 0, label = "No DecVAE models found", size = 6) +
    theme_void() +
    labs(title = "")
} #"DecVAE-based Models Performance Across Datasets"

# Save individual plots
ggsave(file.path(save_dir, "circular_barchart_VAE_models.png"), 
       p_vae, width = 16, height = 16, dpi = 600, bg = "white")

ggsave(file.path(save_dir, "circular_barchart_DecVAE_models.png"), 
       p_decvae, width = 16, height = 16, dpi = 600, bg = "white")

# Create combined plot
combined_plot <- grid.arrange(p_vae, p_decvae, ncol = 2)

ggsave(file.path(save_dir, "circular_barchart_combined.png"), 
       combined_plot, width = 32, height = 16, dpi = 600, bg = "white")

cat("Circular bar charts saved to:", save_dir, "\n")
cat("- VAE models chart: circular_barchart_VAE_models.png\n")
cat("- DecVAE models chart: circular_barchart_DecVAE_models.png\n") 
cat("- Combined chart: circular_barchart_combined.png\n")

ggsave(file.path(save_dir, "circular_barchart_combined.png"), 
       combined_plot, width = 32, height = 16, dpi = 600, bg = "white")

cat("Circular bar charts saved to:", save_dir, "\n")
cat("- VAE models chart: circular_barchart_VAE_models.png\n")
cat("- DecVAE models chart: circular_barchart_DecVAE_models.png\n") 
cat("- Combined chart: circular_barchart_combined.png\n")

