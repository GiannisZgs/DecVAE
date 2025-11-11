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

# Style and font parameters
plot_font_family <- "Arial"
plot_background_color <- "#F4F5F1"
plot_text_color <- "black"
bg <- plot_background_color
txt_col <- plot_text_color
showtext_auto(enable = TRUE)

# Font sizes for different elements
axis_text_size_large <- 200
axis_text_size_small <- 200
axis_y_title_angle <- 90
strip_text_size <- 195
axis_title_size <- 200
plot_title_size <- 200

# Line and point sizes
line_size_regular <- 2.5
line_size_thin <- 1.2

# Margin settings
plot_margin <- margin(10, 10, 10, 10, "pt")
title_margin_top <- margin(t = 20)
ylabel_margin_right <- margin(r = 15)

# Layout parameters
coord_clip <- "off"

# Plots color palette
# Using MetBrewer palette - will be applied with scale_color_met_d
palette <- "Redon"

parent_data_dir <- file.path("D:","data_for_figures_freq_analysis")

json_file_path_filter <- file.path(parent_data_dir, "decomposition_results_NoC3_filter_single_speaker.json") 
json_file_path_emd <- file.path(parent_data_dir, "decomposition_results_NoC3_emd_single_speaker.json") 
json_file_path_ewt <- file.path(parent_data_dir, "decomposition_results_NoC3_ewt_single_speaker.json") 
json_file_path_vmd <- file.path(parent_data_dir, "decomposition_results_NoC3_vmd_single_speaker.json") 
json_file_path_vmd_corrected <- file.path(parent_data_dir, "decomposition_results_NoC3_vmd_corrected_single_speaker.json") 
json_gz_file_path <- file.path(parent_data_dir, "sim_vowels_figures.json.gz") 
json_file_path <- gsub(".gz$", "", json_gz_file_path)

# Set and create directory and filepaths to save
save_dir <-  file.path('..','figures','recognition_model','sequence')
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}
savename <- file.path(save_dir, "speaker_recognition_model_time_domain_seq_single_speaker.png")

sampling_rate <- 16000  # Sampling rate in Hz
sample <- "_0.705"


# Load original generative model
if (!file.exists(json_file_path)) {
  # Decompress the .json.gz file
  json_file_path <- gunzip(json_gz_file_path, remove = FALSE)
}

# Load the JSON data
data_original <- fromJSON(json_file_path)


load_json_data <- function(file_path) {
    json_text <- readLines(file_path)
    json_text <- gsub("\\bNaN\\b", "\"NaN\"", json_text)
    fromJSON(json_text, simplifyDataFrame = TRUE, simplifyMatrix = TRUE)
}

create_decomposition_tibbles <- function(data, vowel) {
    # Components to extract    
    is_input <- length(names(data)) == 1 

    components <- c("Reconstructed", "OC1", "OC2", "OC3")
    spectral_components <- paste0(components, "_PSD")
    if (is_input) {
        spectral_components_prev <- paste0("spectral_density_input_seq_", components)
        spectral_components_prev[1] <- "spectral_density_input_seq_X"
    } else {
        spectral_components_prev <- paste0("spectral_density_", components,"_seq")
        spectral_components_prev[1] <- "spectral_density_X_seq"
    }

    # Initialize lists
    time_domain_values <- list()
    freq_domain_values <- list()

    if (is_input) {
        time_domain_values[["Reconstructed"]] <- as.numeric(data[[vowel]][["sequence"]][1,])
        for (i in 2:length(components)) {
            time_domain_values[[components[i]]] <- as.numeric(data[[vowel]][["sequence"]][i,])
        }
    } else {
        for (i in 2:length(components)) {
            time_domain_values[[components[i]]] <- as.numeric(data[[vowel]][["sequence"]][i, ])
        }
        # Get reconstructed X as sum of OCs    
        x_sum <- Reduce("+", time_domain_values)
        time_domain_values <- c(list(Reconstructed = x_sum), time_domain_values)
    }

    # Get max of X spectral density for normalization
    if (is_input) {
        x_spectral <- as.numeric(data[[vowel]][["Reconstructed_PSD"]])
    } else {
        oc_spectral_values <- lapply(spectral_components_prev[-1], function(comp) {
        as.numeric(data[[vowel]][[comp]])
        })
        x_spectral <- Reduce("+", oc_spectral_values)    
    }

    if (is_input) {
      max_amplitude <- max(abs(x_spectral))
    } else {
      max_amplitude <- max(abs(as.numeric(data[[vowel]][["spectral_density_X_seq"]])))
    }

    for (i in 1:length(spectral_components_prev)) {
        values <- if (is_input) {
            as.numeric(data[[vowel]][[spectral_components[i]]][1:floor(length(x_spectral)/2+1)])
        } else {
            as.numeric(data[[vowel]][[spectral_components_prev[i]]][1:floor(length(x_spectral)/2+1)])
        }
        freq_domain_values[[spectral_components[i]]] <- values / max_amplitude  
    }
    
    # Create time vector
    n_samples_td <- length(time_domain_values[[1]])
    time_vec <- (0:(n_samples_td-1)) / sampling_rate

    # Get frequencies vector
    frequencies <- data[[vowel]][["frequencies"]][1:floor(length(x_spectral)/2+1)]

    # Return list of two tibbles
    list(
        time_domain = tibble(
            value = unlist(time_domain_values),
            signal_type = rep(components, each = n_samples_td),
            time = rep(time_vec, length(components))
        ),
        freq_domain = tibble(
            value = unlist(freq_domain_values),
            signal_type = rep(components, each = length(frequencies)),
            time = rep(frequencies, length(components))
        )
    )
}

# Load raw data
data_raw <- list(
    filter = load_json_data(json_file_path_filter),
    vmd = load_json_data(json_file_path_vmd),
    vmd_corrected = load_json_data(json_file_path_vmd_corrected),
    emd = load_json_data(json_file_path_emd),
    ewt = load_json_data(json_file_path_ewt)
)

# For input signal, extract it from some decomposition and then merge with higher levels
input <- list()
input[["all"]] <- list(
    sequence = data_raw[["filter"]][["all"]][["input_seq"]],
    frequencies = data_raw[["filter"]][["all"]][["frequencies"]],
    Reconstructed_PSD = data_raw[["filter"]][["all"]][["spectral_density_input_seq_X"]],
    OC1_PSD = data_raw[["filter"]][["all"]][["spectral_density_input_seq_OC1"]],
    OC2_PSD = data_raw[["filter"]][["all"]][["spectral_density_input_seq_OC2"]],
    OC3_PSD = data_raw[["filter"]][["all"]][["spectral_density_input_seq_OC3"]]
)

# Merge input with data_raw
data_raw <- c(list(input=input),data_raw)
original_data <- create_decomposition_tibbles(data_raw[["input"]], "all")
filter_data <- create_decomposition_tibbles(data_raw[["filter"]], "all")
vmd_data <- create_decomposition_tibbles(data_raw[["vmd"]], "all")
vmd_corrected_data <- create_decomposition_tibbles(data_raw[["vmd_corrected"]], "all")
emd_data <- create_decomposition_tibbles(data_raw[["emd"]], "all")
ewt_data <- create_decomposition_tibbles(data_raw[["ewt"]], "all")


create_original_freq_plot <- function(decomposition_method = "original", ymin, ymax, show_strip_text = FALSE, is_bottom = FALSE) {
  # Select data based on decomposition method
  data <- switch(decomposition_method,
                "original" = original_data[["freq_domain"]],
                "filter" = filter_data[["freq_domain"]],
                "vmd" = vmd_data[["freq_domain"]],
                "vmd_corrected" = vmd_corrected_data[["freq_domain"]],
                "emd" = emd_data[["freq_domain"]],
                "ewt" = ewt_data[["freq_domain"]])
  
  # Keep only Reconstructed component and add PSD suffix
  data <- data %>%
    mutate(signal_type = paste0(signal_type, "_PSD")) %>%
    filter(signal_type == "Reconstructed_PSD")
  
  # Define frequency breaks
  freq_breaks_major <- seq(0, 8000, by = 500)
  freq_breaks_minor <- seq(0, 8000, by = 100)

  ggplot(data) +
    geom_hline(yintercept = 0, linetype="solid", size=line_size_thin) +
    geom_line(aes(x=time, y=value), color = met.brewer(palette, n=1)[1], size = line_size_regular) +
    scale_y_continuous(limits = c(ymin, ymax)) +
    coord_cartesian(clip = coord_clip) +
    scale_x_continuous(
      breaks = freq_breaks_major,
      minor_breaks = freq_breaks_minor,
      labels = if(is_bottom) freq_breaks_major else NULL
    ) +
    theme(
      axis.text.x = if(is_bottom) 
                    element_text(color=txt_col, size=axis_text_size_large, family=plot_font_family) 
                    else 
                    element_blank(),
      axis.text.y = element_text(color=txt_col, size=axis_text_size_large, family=plot_font_family),
      strip.text.x = if(show_strip_text) 
                     element_text(face="plain", size=strip_text_size, family=plot_font_family) 
                     else element_blank(),
      strip.background = element_rect(fill = plot_background_color, color = NA),
      plot.background = element_rect(color=bg, fill=bg),
      plot.margin = plot_margin,
      legend.position = "none",
      axis.title.x = if(is_bottom)                            
                    element_text(size=axis_title_size, color=txt_col, margin=title_margin_top, family=plot_font_family)     
                    else element_blank(),
      axis.title.y = element_blank(),
      plot.title = if(show_strip_text)
                   element_text(face="plain", size=plot_title_size, hjust=0.5, family=plot_font_family)
                   else element_blank()
    ) +
    {
      if(show_strip_text) labs(title = "Power Spectral Density (Normalized)") else labs(title = NULL)
    } +
    {if(is_bottom) xlab("Frequency (Hz)") else xlab("")}
}

create_signal_plot <- function(decomposition_method, domain = "time_domain", ymin, ymax, 
                             show_strip_text = FALSE, signal_label = "", is_bottom = FALSE) {
  # Select the appropriate data based on decomposition method and domain
  data <- switch(decomposition_method,
                "original" = original_data[[domain]],
                "filter" = filter_data[[domain]],
                "vmd" = vmd_data[[domain]],
                "vmd_corrected" = vmd_corrected_data[[domain]],
                "emd" = emd_data[[domain]],
                "ewt" = ewt_data[[domain]])

  if (domain == "freq_domain") {
    data <- data %>% mutate(signal_type = paste0(signal_type,"_PSD"))
    facet_levels <- c('Reconstructed_PSD','OC1_PSD','OC2_PSD','OC3_PSD')
  } else {
    facet_levels <- c('Reconstructed','OC1','OC2','OC3')
  }

  if (domain == "freq_domain") {
    freq_breaks_major <- seq(0, 8000, by = 1000)    # 0 to 8000 Hz
    freq_breaks_minor <- seq(0, 8000, by = 500)     # Steps of 500 Hz
  }


  ggplot(data) +
    geom_hline(yintercept = 0, linetype="solid", size=line_size_thin) +
    geom_line(aes(x=time, y=value, color=signal_type)) +  # Changed to color by signal_type
    scale_color_met_d(name=palette) +
    scale_y_continuous(limits = c(ymin, ymax)) +
    facet_wrap(
      ~ factor(signal_type, levels = facet_levels), 
      nrow = 1, 
      ncol = 4
    ) + 
    coord_cartesian(clip = coord_clip) +
    {if(domain == "freq_domain") 
      scale_x_continuous(
        breaks = freq_breaks_major,
        minor_breaks = freq_breaks_minor,
        labels = if(is_bottom) freq_breaks_major/1000 else NULL
      )
    } +
    theme(
      axis.title = element_blank(),
      axis.text.x = if(domain == "freq_domain" && is_bottom) 
                      element_text(color=txt_col, size=axis_text_size_large, family=plot_font_family) 
                    else 
                      element_blank(),
      axis.text.y = element_text(color=txt_col, size=axis_text_size_large, family=plot_font_family),
      axis.title.x = if(domain == "freq_domain" && is_bottom)
                    element_text(size=axis_text_size_small, color=txt_col, family=plot_font_family)
                    else element_blank(),
      strip.text.x = if(show_strip_text) 
                     element_text(face="plain", size=strip_text_size, family=plot_font_family) 
                     else element_blank(),
      strip.background = element_rect(fill = plot_background_color, color = NA),
      plot.background = element_rect(color=bg, fill=bg),
      plot.margin = plot_margin,
      legend.position = "none",
      axis.title.y = element_text(size=axis_title_size, face="plain", angle=axis_y_title_angle,
                                 vjust=0.5, margin=ylabel_margin_right, family=plot_font_family),
    ) +
    # Conditionally add x-axis label
    {if(domain == "freq_domain" && is_bottom) xlab("Frequency (kHz)") else xlab("")} +
    ylab(signal_label)
}

ymin <- -2.7
ymax <- 2.7
p0 <- create_signal_plot("original", "time_domain", ymin, ymax, TRUE, "Ground Truth")
p1 <- create_signal_plot("filter", "time_domain", ymin, ymax, FALSE, "FD")
p2 <- create_signal_plot("vmd", "time_domain", ymin, ymax, FALSE, "VMD")
#p3 <- create_signal_plot("vmd_corrected", "time_domain", ymin, ymax, FALSE, "VMD Corrected")
p4 <- create_signal_plot("emd", "time_domain", ymin, ymax, FALSE, "EMD")
p5 <- create_signal_plot("ewt", "time_domain", ymin, ymax, FALSE, "EWT")

ymin <- 0
ymax <- 1
pf0 <- create_original_freq_plot("original", ymin, ymax, TRUE, FALSE)
pf1 <- create_original_freq_plot("filter", ymin, ymax, FALSE, FALSE)
pf2 <- create_original_freq_plot("vmd", ymin, ymax, FALSE, FALSE)
pf4 <- create_original_freq_plot("emd", ymin, ymax, FALSE, FALSE)
pf5 <- create_original_freq_plot("ewt", ymin, ymax, FALSE, TRUE)


total_plot <- (p0 | pf0) / (p1 | pf1) / (p2 | pf2) / (p4 | pf4) / (p5 | pf5) +
  plot_layout(heights = c(1, 1, 1, 1, 1)) 

print(total_plot)

ggsave(savename,
 total_plot, 
 width = 24, 
 height = 15,
 dpi = 600
)
