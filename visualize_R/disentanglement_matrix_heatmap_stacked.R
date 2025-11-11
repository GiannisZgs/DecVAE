library(jsonlite)
library(dplyr)
library(purrr)
library(tidyr)
library(ggplot2)
library(tibble)
library(viridis)
library(gridExtra)
library(grid)
library(cowplot)

# Style and font parameters
plot_font_family <- "Arial"

# Axis titles
axis_title_size <- 26
axis_title_face <- "plain"
axis_title_y_margin <- margin(r = 25)
axis_title_x_margin <- margin(t = 15)
axis_title_y_angle <- 0
axis_title_y_vjust_horizontal <- 0.5
axis_title_y_hjust_horizontal <- 0.1

# Axis tick labels
axis_text_y_size <- 24
axis_text_y_face <- "plain"
axis_text_x_size_horizontal <- 24
axis_text_x_size_vertical <- 20
axis_text_x_face <- "plain"
axis_text_x_angle_horizontal <- 45
axis_text_x_angle_vertical <- 45
axis_text_x_hjust_horizontal <- 1
axis_text_x_vjust_horizontal <- 0.5
axis_text_x_hjust_vertical <- 1

# Legend elements
legend_title_size <- 22
legend_title_face <- "plain"
legend_text_size <- 18
legend_text_face <- "plain"

# Colorbar (bottom legend)
colorbar_title_size <- 22
colorbar_title_face <- "plain"
colorbar_text_size <- 18
colorbar_text_face <- "plain"
colorbar_title_vjust <- 0.8
colorbar_title_hjust <- 5
colorbar_title_position <- "left"
colorbar_width <- 25
colorbar_height <- 2
colorbar_margin <- margin(t = 10, r = 10, b = 0, l = 10)

# Plot margins
plot_margin <- margin(20, 20, 20, 20)
final_plot_margin <- margin(t = 5, r = 5, b = 15, l = 25, unit = "mm")

#Ensure that VAEs and other models use the same color levels
#VAE_FC_mel_b0: global_max_ -> 0.1041, global_max_instance_ -> 0.0565
#VAE_FC_mel_b1: global_max_ -> 0.1542, global_max_instance_ -> 0.0508
#VAE_FC_mel_b01: global_max_ -> 0.1137, global_max_instance_ -> 0.0513
#VAE_FC_mel_b10: global_max_ -> 0.129035060206159, global_max_instance_ -> 0.268367785567103
#VAE_FC_wf_b0: global_max_ -> 0.0895, global_max_instance_ -> 0.04014
#VAE_FC_wf_b01: global_max_ -> 0.0917, global_max_instance_ -> 0.0534
#VAE_FC_wf_b1: global_max_ -> 0.0897, global_max_instance_ -> 0.0422
#VAE_FC_wf_b10: global_max_ -> 0.0703, global_max_instance_ -> 0.02541
#DecVAE_EWT_b0: global_max_ -> 0.1234, global_max_instance_ -> 0.0599
#DecVAE_EWT_b01: global_max_ -> 0.118, global_max_instance_ -> 0.1514
#DecVAE_EWT_b1: global_max_ -> 0.1807, global_max_instance_ -> 0.1904
#DecVAE_EWT_b5: global_max_ -> 0.0939, global_max_instance_ -> 0.1733
#DecVAE_FD_b0: global_max_ -> 0.1170, global_max_instance_ -> 0.0717
#DecVAE_FD_b01: global_max_ -> 0.0856, global_max_instance_ -> 0.1300
#DecVAE_FD_b1: global_max_ -> 0.0668, global_max_instance_ -> 0.0952
#DecVAE_FD_b5: global_max_ -> 0.0786, global_max_instance_ -> 0.1462

current_dir <- getwd()
dataset <- 'voc_als'
model <- 'filter' #'vae_1D_waveform'
beta <- 'snr15_bz01_bs01_NoC4_mel_dual-bs4' # add 'snr15_' in front in case of vowels
transfer_from <- 'vowels' # from_timit, from_sim_vowels / for VOC_ALS and iemocap
analysis_level <- 'frame' #'sequence' #'frame'
ckp <- 'training_ckp_epoch_109'
mi_matrix_type <- 'total'  # Options: 'total' or 'instance'
instance_type <- "king_stage"  # Options: "phoneme", "speaker"
set_global_max <- 0.18
max_phonemes <- 15
max_speakers <- 15

# Define vocabulary path
vocab_path <- file.path('..','vocabularies')
parent_load_dir <- file.path("D:",'latent_disentanglement_matrices') 
#Define the loading dir - where to load the data from
if (grepl('vowels',dataset)) {
  load_dir <- paste0(parent_load_dir,'_vowels')
} else if (grepl("voc_als",dataset) || grepl("iemocap",dataset)) {
  load_dir <- file.path(parent_load_dir, paste0(dataset,'_from_',transfer_from))
} else {
  load_dir <- file.path(parent_load_dir,dataset)
}

if (grepl('vae',model)) {
  data_dir <- file.path(load_dir, beta, analysis_level, 
                    paste0('checkpoint-', ckp), 'latent_vis_dict_z.json')
} else {
  if (grepl("frame",analysis_level)) {
    data_dir <- file.path(load_dir, model, beta, analysis_level, 
                      paste0('checkpoint-', ckp), 'latent_vis_dict_z.json')
  } else if (grepl("sequence",analysis_level)) {
    data_dir <- file.path(load_dir, model, beta, analysis_level, 
                      paste0('checkpoint-', ckp), 'latent_vis_dict_s.json')
  } 
}


# Create save directory
if (grepl("voc_als",dataset) || grepl("iemocap",dataset)) {
  parent_save_dir <- file.path('..','figures','latent_disentanglement_matrices', paste0(dataset,'_from_',transfer_from))
} else {
  parent_save_dir <- file.path('..','figures','latent_disentanglement_matrices', dataset)
}
save_dir <- file.path(parent_save_dir, model, beta, analysis_level, ckp)
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}

# Define save path for stacked visualization
if (mi_matrix_type == "instance") {
  save_path <- file.path(save_dir, paste0("stacked_base_latents_mi_matrix_", mi_matrix_type, "_", instance_type, ".png"))
}
if (mi_matrix_type == "total") {
  save_path <- file.path(save_dir, paste0("stacked_base_latents_mi_matrix_", mi_matrix_type, ".png"))
}

load_json_data <- function(file_path) {
    json_text <- readLines(file_path)
    json_text <- gsub("\\bNaN\\b", "\"NaN\"", json_text)
    fromJSON(json_text, simplifyDataFrame = TRUE, simplifyMatrix = TRUE)
}

# Load vocabulary files for various datasets
if (grepl("timit", dataset)) {
    speaker_dict <- load_json_data(file.path(vocab_path, 'timit_speaker_dict.json'))
    phoneme_dict <- load_json_data(file.path(vocab_path, 'timit_phon39_dict.json'))
    INT_TO_SPEAKER <- names(speaker_dict)
    INT_TO_PHONEME <- names(phoneme_dict)
} else if (grepl("voc_als", dataset)) {
    voc_als_dict <- load_json_data(file.path(vocab_path, 'voc_als_encodings.json'))
    king_stage_values <- names(voc_als_dict$KingClinicalStage)
    phoneme_values <- names(voc_als_dict$phoneme)
    speaker_values <- names(voc_als_dict$speaker)
    
    # Create both direction mappings
    KING_STAGE_TO_INT <- setNames(as.numeric(king_stage_values), 
                                  as.character(voc_als_dict$KingClinicalStage))
    INT_TO_KING_STAGE <- names(KING_STAGE_TO_INT)
    
    PHONEME_TO_INT <- setNames(as.numeric(phoneme_values), 
                               as.character(voc_als_dict$phoneme))
    INT_TO_PHONEME <- names(PHONEME_TO_INT)
    
    SPEAKER_TO_INT <- setNames(as.numeric(speaker_values), 
                               as.character(voc_als_dict$speaker))
    INT_TO_SPEAKER <- names(SPEAKER_TO_INT)
} else if (grepl("iemocap", dataset)) {
    speaker_dict <- load_json_data(file.path(vocab_path, 'iemocap_speaker_dict.json'))
    emotion_dict <- load_json_data(file.path(vocab_path, 'iemocap_emotion_dict.json'))
    phoneme_dict <- load_json_data(file.path(vocab_path, 'iemocap_phone_dict.json'))
    INT_TO_SPEAKER <- names(speaker_dict)
    INT_TO_EMOTION <- names(emotion_dict)
    INT_TO_PHONEME <- names(phoneme_dict)
}


# Function to load all latent types for stacked visualization
load_all_latent_types <- function(file_path) {
  # Load the JSON data
  all_data <- load_json_data(file_path)
  
  # List of base latent types we want to include
  base_latent_types <- c("X", "OC1", "OC2", "OC3", "OC4")
  
  # Create a result list to hold all latent data
  result <- list()
  
  # Extract common data that applies across all latent types
  if (!is.null(all_data$phonemes_unique_values)) {
    result$phonemes_unique_values <- all_data$phonemes_unique_values
  } else if (!is.null(all_data$phonemes)) {
    result$phonemes_unique_values <- sort(unique(all_data$phonemes))
  }
  
  if (!is.null(all_data$speaker_unique_values)) {
    result$speaker_unique_values <- all_data$speaker_unique_values
  } else if (!is.null(all_data$speakers)) {
    result$speaker_unique_values <- sort(unique(all_data$speakers))
  }
  
  # VOC_ALS specific: king_stage
  if (!is.null(all_data$king_stage_unique_values)) {
    result$king_stage_unique_values <- all_data$king_stage_unique_values
  } else if (!is.null(all_data$king_stage)) {
    result$king_stage_unique_values <- sort(unique(all_data$king_stage))
  }
  
  # IEMOCAP specific: emotion
  if (!is.null(all_data$emotion_unique_values)) {
    result$emotion_unique_values <- all_data$emotion_unique_values
  } else if (!is.null(all_data$emotion)) {
    result$emotion_unique_values <- sort(unique(all_data$emotion))
  }
  
  # Store original data
  if (!is.null(all_data$phonemes)) {
    result$phonemes <- all_data$phonemes
  }
  
  if (!is.null(all_data$speakers)) {
    result$speakers <- all_data$speakers
  }
  
  if (!is.null(all_data$king_stage)) {
    result$king_stage <- all_data$king_stage
  }
  
  if (!is.null(all_data$emotion)) {
    result$emotion <- all_data$emotion
  }
  
  # Global max tracking variables
  global_max_total <- 0
  global_max_instance <- 0
  
  # Process each latent type
  for (latent in base_latent_types) {
    # Check and load total MI matrix
    if (!is.null(all_data[[latent]])) {
      
      # Different structure for VOC_ALS and IEMOCAP (pairwise) vs TIMIT (direct)
      if (grepl("voc_als", dataset) || (grepl("iemocap", dataset) && grepl("frame", analysis_level))) {
        # For VOC_ALS and IEMOCAP, extract from pairwise_mi
        if (!is.null(all_data[[latent]]$pairwise_mi)) {
          # Extract pairwise MI matrices separately and combine properly
          combined_rows <- list()
          
          # For VOC_ALS: phonemes, speakers, king_stage
          if (grepl("voc_als", dataset)) {
            # Debug: Print available pairwise combinations
            cat("Available pairwise combinations for", latent, ":", names(all_data[[latent]]$pairwise_mi), "\n")
            
            # Extract phonemes row from phonemes_speakers
            if (!is.null(all_data[[latent]]$pairwise_mi$phonemes_speakers$mi_matrix)) {
              combined_rows[["phonemes"]] <- all_data[[latent]]$pairwise_mi$phonemes_speakers$mi_matrix[, 1, drop = FALSE]
            }
            
            # Extract speakers row from phonemes_speakers  
            if (!is.null(all_data[[latent]]$pairwise_mi$phonemes_speakers$mi_matrix)) {
              combined_rows[["speakers"]] <- all_data[[latent]]$pairwise_mi$phonemes_speakers$mi_matrix[, 2, drop = FALSE]
            }
            
            # Try multiple possible locations for king_stage
            king_stage_extracted <- FALSE
            
            # Option 1: From phonemes_king_stage (row 2)
            if (!is.null(all_data[[latent]]$pairwise_mi$phonemes_king_stage$mi_matrix)) {
              combined_rows[["king_stage"]] <- all_data[[latent]]$pairwise_mi$phonemes_king_stage$mi_matrix[, 2, drop = FALSE]
              king_stage_extracted <- TRUE
              cat("Extracted king_stage from phonemes_king_stage (row 2)\n")
            }
            # Option 2: From speakers_king_stage (row 2)
            else if (!is.null(all_data[[latent]]$pairwise_mi$speakers_king_stage$mi_matrix)) {
              combined_rows[["king_stage"]] <- all_data[[latent]]$pairwise_mi$speakers_king_stage$mi_matrix[, 2, drop = FALSE]
              king_stage_extracted <- TRUE
              cat("Extracted king_stage from speakers_king_stage (row 2)\n")
            }
            
            if (!king_stage_extracted) {
              cat("WARNING: Could not extract king_stage for", latent, "\n")
              cat("Available pairwise keys:", names(all_data[[latent]]$pairwise_mi), "\n")
            }
          }
          
          # For IEMOCAP: phonemes, speakers, emotion
          if (grepl("iemocap", dataset)) {
            # Extract phonemes row from phonemes_speakers
            if (!is.null(all_data[[latent]]$pairwise_mi$phonemes_speakers$mi_matrix)) {
              combined_rows[["phonemes"]] <- all_data[[latent]]$pairwise_mi$phonemes_speakers$mi_matrix[, 1, drop = FALSE]
            }
            
            # Extract speakers row from phonemes_speakers
            if (!is.null(all_data[[latent]]$pairwise_mi$phonemes_speakers$mi_matrix)) {
              combined_rows[["speakers"]] <- all_data[[latent]]$pairwise_mi$phonemes_speakers$mi_matrix[, 2, drop = FALSE]
            }
            
            # Extract emotion row from phonemes_emotion
            if (!is.null(all_data[[latent]]$pairwise_mi$phonemes_emotion$mi_matrix)) {
              combined_rows[["emotion"]] <- all_data[[latent]]$pairwise_mi$phonemes_emotion$mi_matrix[, 2, drop = FALSE]
            }
          }
          
          # Combine the rows into a single matrix
          if (length(combined_rows) > 0) {
            # Debug: Print what we're combining
            cat("Combining rows for", latent, ":", names(combined_rows), "\n")
            cat("Row dimensions:", sapply(combined_rows, function(x) paste(dim(x), collapse="x")), "\n")
            
            combined_mi <- do.call(cbind, combined_rows)
            result[[latent]] <- combined_mi
            global_max_total <- max(global_max_total, max(combined_mi, na.rm = TRUE))
            
            # Debug: Print final matrix dimensions
            cat("Final matrix dimensions for", latent, ":", dim(combined_mi), "\n")
          }
        }
      } else if (grepl("iemocap", dataset) && grepl("sequence", analysis_level)) {
        # For IEMOCAP sequence-level, extract directly from mi_matrix (only speaker and emotion)
        total_mi <- all_data[[latent]]$mi_matrix
        
        if (!is.null(total_mi)) {
          result[[latent]] <- total_mi
          global_max_total <- max(global_max_total, max(total_mi, na.rm = TRUE))
        }
      
      } else {
        # For TIMIT, use the original structure
        total_mi <- all_data[[latent]]$mi_matrix
        
        if (!is.null(total_mi)) {
          result[[latent]] <- total_mi
          global_max_total <- max(global_max_total, max(total_mi, na.rm = TRUE))
        }
      }
    }
    
    # Check and load instance MI matrices
    instance_key <- paste0(latent, "_instance")

    if (grepl("voc_als", dataset) || (grepl("iemocap", dataset) && grepl("frame", analysis_level))) {
      # For VOC_ALS and IEMOCAP, extract from pairwise_instance_mi
      if (!is.null(all_data[[latent]]$pairwise_instance_mi)) {
        result[[instance_key]] <- list()
        
        # Process phonemes_speakers combination
        if (!is.null(all_data[[latent]]$pairwise_instance_mi$phonemes_speakers)) {
          if (!is.null(all_data[[latent]]$pairwise_instance_mi$phonemes_speakers$phonemes$matrix)) {
            phoneme_mi <- all_data[[latent]]$pairwise_instance_mi$phonemes_speakers$phonemes$matrix
            result[[instance_key]]$phonemes_instance_mi <- phoneme_mi
            global_max_instance <- max(global_max_instance, max(phoneme_mi, na.rm = TRUE))
          }
          
          if (!is.null(all_data[[latent]]$pairwise_instance_mi$phonemes_speakers$speakers$matrix)) {
            speaker_mi <- all_data[[latent]]$pairwise_instance_mi$phonemes_speakers$speakers$matrix
            result[[instance_key]]$speakers_instance_mi <- speaker_mi
            global_max_instance <- max(global_max_instance, max(speaker_mi, na.rm = TRUE))
          }
        }
        
        # For VOC_ALS: Process phonemes_king_stage and speakers_king_stage combinations
        if (grepl("voc_als", dataset)) {
          if (!is.null(all_data[[latent]]$pairwise_instance_mi$phonemes_king_stage)) {
            if (!is.null(all_data[[latent]]$pairwise_instance_mi$phonemes_king_stage$king_stage$matrix)) {
              king_stage_mi <- all_data[[latent]]$pairwise_instance_mi$phonemes_king_stage$king_stage$matrix
              result[[instance_key]]$king_stage_instance_mi <- king_stage_mi
              global_max_instance <- max(global_max_instance, max(king_stage_mi, na.rm = TRUE))
            }
          }
        }
        
        # For IEMOCAP: Process phonemes_emotion and speakers_emotion combinations
        if (grepl("iemocap", dataset)) {
          if (!is.null(all_data[[latent]]$pairwise_instance_mi$phonemes_emotion)) {
            if (!is.null(all_data[[latent]]$pairwise_instance_mi$phonemes_emotion$emotion$matrix)) {
              emotion_mi <- all_data[[latent]]$pairwise_instance_mi$phonemes_emotion$emotion$matrix
              result[[instance_key]]$emotion_instance_mi <- emotion_mi
              global_max_instance <- max(global_max_instance, max(emotion_mi, na.rm = TRUE))
            }
          }
        }
      }
    } else if (grepl("iemocap", dataset) && grepl("sequence", analysis_level)) {
      # For IEMOCAP sequence-level, extract directly from instance_mi (only speaker and emotion)
      if (!is.null(all_data[[latent]]$instance_mi)) {
        result[[instance_key]] <- list()
        
        # Speaker instance MI
        if (!is.null(all_data[[latent]]$instance_mi$speakers$matrix)) {
          speaker_mi <- all_data[[latent]]$instance_mi$speakers$matrix
          if (!is.null(speaker_mi)) {
            result[[instance_key]]$speakers_instance_mi <- speaker_mi
            global_max_instance <- max(global_max_instance, max(speaker_mi, na.rm = TRUE))
          }
        }
        
        # Emotion instance MI
        if (!is.null(all_data[[latent]]$instance_mi$emotions$matrix)) {
          emotion_mi <- all_data[[latent]]$instance_mi$emotions$matrix
          if (!is.null(emotion_mi)) {
            result[[instance_key]]$emotion_instance_mi <- emotion_mi
            global_max_instance <- max(global_max_instance, max(emotion_mi, na.rm = TRUE))
          }
        }
      }
    
    } else {
      # For TIMIT, use the original structure
      if (!is.null(all_data[[latent]]$instance_mi)) {
        result[[instance_key]] <- list()
        
        # Phoneme instance MI
        if (!is.null(all_data[[latent]]$instance_mi$phonemes$matrix)) {
          phoneme_mi <- all_data[[latent]]$instance_mi$phonemes$matrix
          if (!is.null(phoneme_mi)) {
            result[[instance_key]]$phonemes_instance_mi <- phoneme_mi
            global_max_instance <- max(global_max_instance, max(phoneme_mi, na.rm = TRUE))
          }
        }
        
        # Speaker instance MI
        if (!is.null(all_data[[latent]]$instance_mi$speakers$matrix)) {
          speaker_mi <- all_data[[latent]]$instance_mi$speakers$matrix
          if (!is.null(speaker_mi)) {
            result[[instance_key]]$speakers_instance_mi <- speaker_mi
            global_max_instance <- max(global_max_instance, max(speaker_mi, na.rm = TRUE))
          }
        }
      }
    }
  }
  
  # Store global maxima
  result$global_max_total <- global_max_total
  result$global_max_instance <- global_max_instance
  
  return(result)
}

# Function to create MI heatmap
create_mi_heatmap <- function(mi_matrix, factor_names = NULL, dim_names = NULL, horizontal = FALSE, global_max = NULL) {
  # Ensure mi_matrix is actually a matrix
  if (is.list(mi_matrix)) {
    mi_matrix <- do.call(rbind, mi_matrix)
  }
  if (!is.matrix(mi_matrix)) {
    mi_matrix <- as.matrix(mi_matrix)
  }
  
  # For total MI matrix, transpose if horizontal layout is requested
  if (horizontal && ncol(mi_matrix) <= 5) {  # Changed condition: check rows instead of columns
    # Transpose the matrix to get factors as rows and dimensions as columns
    mi_matrix <- t(mi_matrix)
    
    # Prepare data for horizontal plotting (factors as rows, dimensions as columns)
    mi_df <- as.data.frame(mi_matrix) %>%
      # Add factor labels as row names
      rownames_to_column("Factor")
    
    # Assign factor names if provided - handle the number of factors correctly
    if (!is.null(factor_names)) {
      # Ensure we have the right number of factor names for the matrix rows
      n_rows <- nrow(mi_matrix)
      if (length(factor_names) >= n_rows) {
        # Use sequential assignment based on row position
        mi_df$Factor <- factor_names[1:n_rows]
      } else {
        warning("Not enough factor names provided. Using default row names.")
        # Keep original factor values if not enough names provided
      }
    }
    
    mi_df <- mi_df %>%
      # Convert to long format
      pivot_longer(cols = -Factor, 
                   names_to = "Dimension", 
                   values_to = "MI") %>%
      # Ensure dimensions are ordered numerically
      mutate(Dimension = factor(
        Dimension, 
        levels = paste0("V", 1:ncol(mi_matrix))
      )) %>%
      # Ensure Factor is ordered correctly to maintain the order
      mutate(Factor = factor(Factor, levels = if(!is.null(factor_names)) factor_names[1:nrow(mi_matrix)] else unique(Factor)))
    
    # Replace V1, V2, etc. with actual dimension names if provided
    if (!is.null(dim_names)) {
      levels(mi_df$Dimension) <- dim_names
    } else {
      # Default to z_1, z_2, etc.
      levels(mi_df$Dimension) <- paste0("z_", 1:ncol(mi_matrix))
    }

    p <- ggplot(mi_df, aes(x = Dimension, y = Factor, fill = MI)) + 
        theme_minimal() +
        geom_tile(color = "white", linewidth = 0.1) +
        scale_fill_viridis(
          option = "plasma",
          name = "Normalized Mutual\nInformation",
          limits = c(0, if(is.null(global_max)) max(mi_df$MI, na.rm = TRUE) else global_max),
          direction = 1
        ) + 
        scale_y_discrete(
          limits = if(!is.null(factor_names)) rev(factor_names[1:nrow(mi_matrix)]) else NULL
        ) + 
        scale_x_discrete(
          labels = function(x) {
            # Extract numbers after z_
            nums <- as.numeric(gsub("z_", "", x))
            
            # Show labels only at positions 1, 8, 16, 24, 32, 40, 48, etc.
            # Fix the error by ensuring max(nums) >= 8 before creating sequence
            if (length(nums) > 0 && max(nums) >= 8) {
              show_positions <- c(1, seq(8, max(nums), by = 8))
            } else {
              show_positions <- c(1)
            }
            labels <- rep("", length(nums))
            labels[nums %in% show_positions] <- parse(text = paste0("z[", nums[nums %in% show_positions], "]"))
            return(labels)
          }
        ) +
        labs(
            x = "Latent Dimension",
            y = "Factor"
        ) +
        theme(
            panel.background = element_rect(fill = "white", color = NA),
            plot.background = element_rect(fill = "white", color = NA),
            # Set consistent font family for tick labels using parameters
            axis.text.y = element_text(
              size = axis_text_y_size, 
              face = axis_text_y_face, 
              family = plot_font_family
            ),
            axis.text.x = element_text(
              angle = axis_text_x_angle_horizontal, 
              hjust = axis_text_x_hjust_horizontal, 
              vjust = axis_text_x_vjust_horizontal, 
              size = axis_text_x_size_horizontal, 
              face = axis_text_x_face, 
              family = plot_font_family
            ),
            # Set consistent font family for titles with parameters
            axis.title.y = element_text(
                size = axis_title_size, 
                face = axis_title_face, 
                family = plot_font_family, 
                margin = axis_title_y_margin,
                angle = axis_title_y_angle,
                vjust = axis_title_y_vjust_horizontal,
                hjust = axis_title_y_hjust_horizontal
            ),
            axis.title.x = element_text(
              size = axis_title_size, 
              face = axis_title_face, 
              family = plot_font_family, 
              margin = axis_title_x_margin
            ),
            legend.position = "right",
            # Consistent font for legend text and title using parameters
            legend.title = element_text(
              family = plot_font_family, 
              face = legend_title_face, 
              size = legend_title_size
            ),
            legend.text = element_text(
              family = plot_font_family, 
              size = legend_text_size, 
              face = legend_text_face
            ),
            panel.grid = element_blank(),
            plot.margin = plot_margin
      )
    
    return(p)
  } else {
    # Original vertical implementation with your requested changes
    mi_df <- as.data.frame(mi_matrix) %>%
      # Add dimension labels
      mutate(Dimension = if(is.null(dim_names)) paste0("z_", 1:nrow(mi_matrix)) else dim_names) %>%
      # Convert to long format
      pivot_longer(cols = -Dimension, 
                   names_to = "Factor", 
                   values_to = "MI") %>%
      mutate(Dimension = factor(Dimension, 
                    levels = paste0("z_", 1:nrow(mi_matrix)),
                   ordered = TRUE))
    
    # Apply factor labels if provided
    if(!is.null(factor_names)) {
        # Get column indices
        col_indices <- as.numeric(gsub("V", "", mi_df$Factor))
        
        # Check if we have any NA values from the conversion
        if(any(is.na(col_indices))) {
            warning("Some column indices couldn't be converted to numeric. Using positional mapping instead.")
            # Try using positional mapping instead
            unique_factors <- unique(mi_df$Factor)
            factor_map <- setNames(factor_names[1:length(unique_factors)], unique_factors)
            mi_df$Factor <- factor_map[mi_df$Factor]
        } else {
            # Use numeric indices if conversion worked
            mi_df$Factor <- factor_names[col_indices]
        }
    }
    
    # Create heatmap
    p <- ggplot(mi_df, aes(x = Factor, y = Dimension, fill = MI)) + 
        theme_minimal() +
        geom_tile(color = "white", linewidth = 0.1) +
        # Use expression() for y-axis labels with subscripts
        scale_y_discrete(
            limits = rev(paste0("z_", 1:nrow(mi_matrix))),
            labels = function(x) {
                # Extract numbers after z_
                nums <- as.numeric(gsub("z_", "", x))
                
                # For instance MI matrices (typically with many dimensions), only show labels at specific positions
                if (nrow(mi_matrix) > 10) { # This condition identifies instance MI matrices
                # Show labels at positions 1, 8, 16, 24, 32, 40, 48, 56, 64, etc. (every 8th)
                # Fix the error by ensuring max(nums) >= 8 before creating sequence
                if (length(nums) > 0 && max(nums) >= 8) {
                  show_positions <- c(1, seq(8, max(nums), by = 8))
                } else {
                  show_positions <- c(1)
                }
                labels <- rep("", length(nums))
                labels[nums %in% show_positions] <- parse(text = paste0("z[", nums[nums %in% show_positions], "]"))
                return(labels)
                } else {
                # For smaller matrices, show all labels with subscripts
                return(parse(text = paste0("z[", nums, "]")))
                }
            }
        ) +
        scale_fill_viridis(
          option = "plasma",
          name = "Normalized Mutual\nInformation",
          limits = c(0, if(is.null(global_max)) max(mi_df$MI, na.rm = TRUE) else global_max),
          direction = 1
        ) + 
        labs(
            x = "Factor",
            y = "Latent Dimension"
        ) +
        theme(
            panel.background = element_rect(fill = "white", color = NA),
            plot.background = element_rect(fill = "white", color = NA),
            # Set consistent font family for tick labels using parameters
            axis.text.y = element_text(
              size = axis_text_y_size, 
              face = axis_text_y_face, 
              family = plot_font_family
            ),
            axis.text.x = element_text(
              angle = axis_text_x_angle_vertical, 
              hjust = axis_text_x_hjust_vertical, 
              size = axis_text_x_size_vertical, 
              face = axis_text_x_face, 
              family = plot_font_family
            ),
            # Set consistent font family for titles with parameters
            axis.title.y = element_text(
                size = axis_title_size, 
                face = axis_title_face, 
                family = plot_font_family, 
                margin = axis_title_y_margin,
                angle = axis_title_y_angle,
                vjust = axis_title_y_vjust_horizontal,
                hjust = axis_title_y_hjust_horizontal
            ),
            axis.title.x = element_text(
              size = axis_title_size, 
              face = axis_title_face, 
              family = plot_font_family, 
              margin = axis_title_x_margin
            ),
            legend.position = "right",
            # Consistent font for legend text and title using parameters
            legend.title = element_text(
              family = plot_font_family, 
              face = legend_title_face, 
              size = legend_title_size
            ),
            legend.text = element_text(
              family = plot_font_family, 
              size = legend_text_size,
              face = legend_text_face
            ),
            panel.grid = element_blank()
    )
    
    return(p)
  }
}

create_stacked_mi_heatmaps <- function(data, mi_matrix_type = "total", global_max = NULL, instance_type = "both") {
  
  # Update valid instance types based on dataset
  if (grepl("voc_als", dataset)) {
    valid_types <- c("phoneme", "speaker", "king_stage", "both")
  } else if (grepl("iemocap", dataset)) {
    valid_types <- c("phoneme", "speaker", "emotion", "both")
  } else {
    valid_types <- c("phoneme", "speaker", "both")
  }
  
  if (mi_matrix_type == "instance" && !instance_type %in% valid_types) {
    stop(paste("For", dataset, "instance MI, instance_type must be one of:", paste(valid_types, collapse = ", ")))
  }
  # List of base latent types to include (excluding composite types)
  base_latent_types <- c("X", "OC1", "OC2", "OC3", "OC4")
  
  # Filter to include only available latent types
  available_latents <- base_latent_types[base_latent_types %in% names(data)]
  # Fix the sapply call to ensure it returns a logical vector
  latent_exists <- vapply(available_latents, function(lat) !is.null(data[[lat]]), logical(1))
  available_latents <- available_latents[latent_exists]
  
  if (length(available_latents) == 0) {
    stop("No base latent types found in the data.")
  }
  
  # Create a list to store plots
  plot_list <- list()
  
  # Determine global max value if not provided
  if (is.null(global_max)) {
    if (mi_matrix_type == "total") {
      global_max <- 0
      for (latent in available_latents) {
        if (!is.null(data[[latent]])) {
          global_max <- max(global_max, max(data[[latent]], na.rm = TRUE))
        }
      }
    } else { # instance
      global_max <- 0
      for (latent in available_latents) {
        instance_key <- paste0(latent, "_instance")
        if (!is.null(data[[instance_key]])) {
          for (factor_type in c("phonemes_instance_mi", "speakers_instance_mi", "king_stage_instance_mi", "emotion_instance_mi")) {
            if (!is.null(data[[instance_key]][[factor_type]])) {
              global_max <- max(global_max, max(data[[instance_key]][[factor_type]], na.rm = TRUE))
            }
          }
        }
      }
    }
  }
  
  # Total number of latent types to plot
  n_latents <- length(available_latents)
  
  # Generate plots for each latent type
  for (i in 1:n_latents) {
    latent <- available_latents[i]
    is_last <- (i == n_latents)  # Check if this is the last plot
    
    if (mi_matrix_type == "total") {
      # Get the MI matrix for this latent
      mi_matrix <- data[[latent]]
      if (is.list(mi_matrix)) mi_matrix <- do.call(rbind, mi_matrix)
      
      # Check if matrix exists
      if (is.null(mi_matrix) || length(mi_matrix) == 0) {
        next
      }
      
      # Debug: Print matrix info
      cat("Processing", latent, "- Matrix dimensions:", dim(mi_matrix), "\n")

      # Create dimension labels
      dim_names <- paste0("z_", 1:nrow(mi_matrix))  
      
      # Factor labels based on dataset
      if (grepl("voc_als", dataset)) {
        if (ncol(mi_matrix) == 3) {
          factor_labels <- c("Phoneme", "Speaker", "King Stage")
        } else if (ncol(mi_matrix) == 2) {
          factor_labels <- c("Phoneme", "Speaker")
        } else {
          factor_labels <- paste("Factor", 1:ncol(mi_matrix))
        }
      } else if (grepl("iemocap", dataset)) {
        if (grepl("sequence", analysis_level)) {
          # For IEMOCAP sequence-level: only speaker and emotion
          if (ncol(mi_matrix) == 2) {
            factor_labels <- c("Speaker", "Emotion")
          } else {
            factor_labels <- paste("Factor", 1:ncol(mi_matrix))
          }
        } else {
          # For IEMOCAP frame-level: phoneme, speaker, and emotion
          if (ncol(mi_matrix) == 3) {
            factor_labels <- c("Phoneme", "Speaker", "Emotion")
          } else {
            factor_labels <- paste("Factor", 1:ncol(mi_matrix))
          }
        }
      } else {
        # TIMIT and others
        factor_labels <- c("Phoneme", "Speaker")
      }
      
      # Create the horizontal heatmap
      p <- create_mi_heatmap(
        mi_matrix, 
        factor_labels,
        dim_names,
        horizontal = TRUE,
        global_max = global_max
      )
      
      # Replace y-axis label with latent name
      p <- p + labs(y = latent) +
        theme(
            axis.title.y = element_text(
                size = axis_title_size, 
                face = axis_title_face, 
                family = plot_font_family,
                margin = axis_title_y_margin,
                angle = axis_title_y_angle,
                vjust = axis_title_y_vjust_horizontal,
                hjust = axis_title_y_hjust_horizontal
            ),
            axis.title.x = element_text(
                size = axis_title_size, 
                face = axis_title_face, 
                family = plot_font_family,
                margin = axis_title_x_margin
            )
        )      
      # Remove legend from all plots
      p <- p + theme(legend.position = "none")
      
      # Hide x-axis text and title except for the last plot
      if (!is_last) {
        p <- p + theme(
          axis.text.x = element_blank(),
          axis.title.x = element_blank()
        )
      }
      
      plot_list[[length(plot_list) + 1]] <- p
      
    } else if (mi_matrix_type == "instance") {
      # For instance MI, create visualizations for each factor type
      instance_key <- paste0(latent, "_instance")
      
      # Phoneme plots (only for frame-level analysis or non-IEMOCAP datasets)
      if ((instance_type == "phoneme" || instance_type == "both") && 
          !(grepl("iemocap", dataset) && grepl("sequence", analysis_level)) &&
          !is.null(data[[instance_key]]) && 
          !is.null(data[[instance_key]]$phonemes_instance_mi)) {
        
        # Process phoneme instance MI
        phoneme_matrix <- data[[instance_key]]$phonemes_instance_mi
        if (is.list(phoneme_matrix)) phoneme_matrix <- do.call(rbind, phoneme_matrix)
        
        # Select subset of phonemes 
        n_phonemes <- min(ncol(phoneme_matrix), max_phonemes)
        
        # Select top phonemes by variance or first N phonemes
        if (ncol(phoneme_matrix) > max_phonemes) {
          # Calculate variance for each phoneme column to select most informative ones
          phoneme_variance <- apply(phoneme_matrix, 2, var, na.rm = TRUE)
          top_phoneme_indices <- order(phoneme_variance, decreasing = TRUE)[1:max_phonemes]
          phoneme_matrix <- phoneme_matrix[, top_phoneme_indices, drop = FALSE]
        }
        
        # Generate phoneme labels using the dictionary
        if (exists("INT_TO_PHONEME") && !is.null(INT_TO_PHONEME)) {
          # Use actual phoneme names from dictionary
          if (ncol(phoneme_matrix) <= length(INT_TO_PHONEME)) {
            if (exists("top_phoneme_indices")) {
              phoneme_labels <- INT_TO_PHONEME[top_phoneme_indices]
            } else {
              phoneme_labels <- INT_TO_PHONEME[1:ncol(phoneme_matrix)]
            }
          } else {
            # Fallback if not enough phonemes in dictionary
            phoneme_labels <- c(INT_TO_PHONEME, paste0("P", (length(INT_TO_PHONEME) + 1):ncol(phoneme_matrix)))
          }
        } else {
          # Fallback to original logic if no dictionary available
          if (ncol(phoneme_matrix) == 5) {
            phoneme_labels <- c("a", "e", "I", "aw", "u")
          } else if (ncol(phoneme_matrix) == 8) {
            phoneme_labels <- c("i", "I", "e", "ae", "a", "aw", "y", "u")
          } else {
            phoneme_labels <- paste0("P", 1:ncol(phoneme_matrix))
          }
        }
        
        # Create dimension labels
        dim_names <- paste0("z_", 1:nrow(phoneme_matrix))
        
        # Create the phoneme heatmap
        p_phoneme <- create_mi_heatmap(
            phoneme_matrix,
            phoneme_labels,
            dim_names,
            global_max = global_max
            )
        
        # Replace y-axis label with latent name and x-axis title with factor name
        p_phoneme <- p_phoneme + labs(y = latent, x = "Phoneme") +
            theme(
                axis.title.y = element_text(
                    size = axis_title_size, 
                    face = axis_title_face, 
                    family = plot_font_family,
                    margin = axis_title_y_margin,
                    angle = axis_title_y_angle,
                    vjust = axis_title_y_vjust_horizontal,
                    hjust = axis_title_y_hjust_horizontal
                ),
                axis.title.x = element_text(
                    size = axis_title_size, 
                    face = axis_title_face, 
                    family = plot_font_family,
                    margin = axis_title_x_margin
                )
            )
        
        # Remove legend from all plots
        p_phoneme <- p_phoneme + theme(legend.position = "none")
        
        # Hide x-axis text and title if not the last plot
        if (!is_last) {
        p_phoneme <- p_phoneme + theme(
            axis.text.x = element_blank(),
            axis.title.x = element_blank()
        )
        }
        
        plot_list[[length(plot_list) + 1]] <- p_phoneme
      }
      
      # Speaker plots (TIMIT and IEMOCAP)
      if ((instance_type == "speaker" || instance_type == "both") && 
          !grepl("voc_als", dataset) &&
          !is.null(data[[instance_key]]) && 
          !is.null(data[[instance_key]]$speakers_instance_mi)) {
        
        # Process speaker instance MI
        speaker_matrix <- data[[instance_key]]$speakers_instance_mi
        if (is.list(speaker_matrix)) speaker_matrix <- do.call(rbind, speaker_matrix)
        
        # Select subset of speakers
        n_speakers <- min(ncol(speaker_matrix), max_speakers)
        
        # Select top speakers by variance or first N speakers
        if (ncol(speaker_matrix) > max_speakers) {
          # Calculate variance for each speaker column to select most informative ones
          speaker_variance <- apply(speaker_matrix, 2, var, na.rm = TRUE)
          top_speaker_indices <- order(speaker_variance, decreasing = TRUE)[1:max_speakers]
          speaker_matrix <- speaker_matrix[, top_speaker_indices, drop = FALSE]
        }
        
        # Generate speaker labels using the dictionary
        if (exists("INT_TO_SPEAKER") && !is.null(INT_TO_SPEAKER)) {
          # Use actual speaker names from dictionary
          if (ncol(speaker_matrix) <= length(INT_TO_SPEAKER)) {
            if (exists("top_speaker_indices")) {
              speaker_labels <- INT_TO_SPEAKER[top_speaker_indices]
            } else {
              speaker_labels <- INT_TO_SPEAKER[1:ncol(speaker_matrix)]
            }
          } else {
            # Fallback if not enough speakers in dictionary
            speaker_labels <- c(INT_TO_SPEAKER, paste0("S", (length(INT_TO_SPEAKER) + 1):ncol(speaker_matrix)))
          }
        } else {
          # Fallback to generic labels if no dictionary available
          speaker_labels <- paste0("S", 1:ncol(speaker_matrix))
        }
        
        # Create dimension labels
        dim_names <- paste0("z_", 1:nrow(speaker_matrix))
        
        # Create the speaker heatmap
        p_speaker <- create_mi_heatmap(
        speaker_matrix,
        speaker_labels,
        dim_names,
        global_max = global_max
        ) + 
            scale_x_discrete(limits = speaker_labels)
        
        # Replace y-axis label with latent name and x-axis title with factor name
        p_speaker <- p_speaker + labs(y = latent, x = "Speaker") +
            theme(
                axis.title.y = element_text(
                    size = axis_title_size, 
                    face = axis_title_face, 
                    family = plot_font_family,
                    margin = axis_title_y_margin,
                    angle = axis_title_y_angle,
                    vjust = axis_title_y_vjust_horizontal,
                    hjust = axis_title_y_hjust_horizontal
                ),
                axis.title.x = element_text(
                    size = axis_title_size, 
                    face = axis_title_face, 
                    family = plot_font_family,
                    margin = axis_title_x_margin
                )
            )
        # Remove legend from all plots
        p_speaker <- p_speaker + theme(legend.position = "none")
        
        # Hide x-axis text and title if not the last plot
        if (!is_last) {
        p_speaker <- p_speaker + theme(
            axis.text.x = element_blank(),
            axis.title.x = element_blank()
        )
        }
        
        plot_list[[length(plot_list) + 1]] <- p_speaker
      }
      
      # king Stage plots (VOC_ALS only)
      if ((instance_type == "king_stage" || instance_type == "both") && 
          grepl("voc_als", dataset) &&
          !is.null(data[[instance_key]]) && 
          !is.null(data[[instance_key]]$king_stage_instance_mi)) {

        # Process king_stage instance MI
        king_stage_matrix <- data[[instance_key]]$king_stage_instance_mi
        if (is.list(king_stage_matrix)) king_stage_matrix <- do.call(rbind, king_stage_matrix)
        
        # Generate king_stage labels using the dictionary
        if (exists("INT_TO_KING_STAGE") && !is.null(INT_TO_KING_STAGE)) {
          king_stage_labels <- INT_TO_KING_STAGE[1:ncol(king_stage_matrix)]
        } else {
          king_stage_labels <- paste0("KS", 1:ncol(king_stage_matrix))
        }
        
        # Create dimension labels
        dim_names <- paste0("z_", 1:nrow(king_stage_matrix))
        
        # Create the king_stage heatmap
        p_king_stage <- create_mi_heatmap(
          king_stage_matrix,
          king_stage_labels,
          dim_names,
          global_max = global_max
        ) + 
          scale_x_discrete(limits = king_stage_labels)
        
        # Replace y-axis label with latent name and x-axis title with factor name
        p_king_stage <- p_king_stage + labs(y = latent, x = "King Stage") +
          theme(
            axis.title.y = element_text(
              size = axis_title_size, 
              face = axis_title_face, 
              family = plot_font_family,
              margin = axis_title_y_margin,
              angle = axis_title_y_angle,
              vjust = axis_title_y_vjust_horizontal,
              hjust = axis_title_y_hjust_horizontal
            ),
            axis.title.x = element_text(
              size = axis_title_size, 
              face = axis_title_face, 
              family = plot_font_family,
              margin = axis_title_x_margin
            )
          )
        
        # Remove legend from all plots
        p_king_stage <- p_king_stage + theme(legend.position = "none")
        
        # Hide x-axis text and title if not the last plot
        if (!is_last) {
          p_king_stage <- p_king_stage + theme(
            axis.text.x = element_blank(),
            axis.title.x = element_blank()
          )
        }
        
        plot_list[[length(plot_list) + 1]] <- p_king_stage
      }
      
      # Emotion plots (IEMOCAP only)
      if ((instance_type == "emotion" || instance_type == "both") && 
          grepl("iemocap", dataset) &&
          !is.null(data[[instance_key]]) && 
          !is.null(data[[instance_key]]$emotion_instance_mi)) {

        # Process emotion instance MI
        emotion_matrix <- data[[instance_key]]$emotion_instance_mi
        if (is.list(emotion_matrix)) emotion_matrix <- do.call(rbind, emotion_matrix)
        
        # Generate emotion labels using the dictionary
        if (exists("INT_TO_EMOTION") && !is.null(INT_TO_EMOTION)) {
          emotion_labels <- INT_TO_EMOTION[1:ncol(emotion_matrix)]
        } else {
          emotion_labels <- paste0("E", 1:ncol(emotion_matrix))
        }
        
        # Create dimension labels
        dim_names <- paste0("z_", 1:nrow(emotion_matrix))
        
        # Create the emotion heatmap
        p_emotion <- create_mi_heatmap(
          emotion_matrix,
          emotion_labels,
          dim_names,
          global_max = global_max
        ) + 
          scale_x_discrete(limits = emotion_labels)
        
        # Replace y-axis label with latent name and x-axis title with factor name
        p_emotion <- p_emotion + labs(y = latent, x = "Emotion") +
          theme(
            axis.title.y = element_text(
              size = axis_title_size, 
              face = axis_title_face, 
              family = plot_font_family,
              margin = axis_title_y_margin,
              angle = axis_title_y_angle,
              vjust = axis_title_y_vjust_horizontal,
              hjust = axis_title_y_hjust_horizontal
            ),
            axis.title.x = element_text(
              size = axis_title_size, 
              face = axis_title_face, 
              family = plot_font_family,
              margin = axis_title_x_margin
            )
          )
        
        # Remove legend from all plots
        p_emotion <- p_emotion + theme(legend.position = "none")
        
        # Hide x-axis text and title if not the last plot
        if (!is_last) {
          p_emotion <- p_emotion + theme(
            axis.text.x = element_blank(),
            axis.title.x = element_blank()
          )
        }
        
        plot_list[[length(plot_list) + 1]] <- p_emotion
      }
    }
  }
  
  # Only proceed if we have plots to display
  if (length(plot_list) > 0) {

    # Create a more carefully designed colorbar
    color_bar <- ggplot(data.frame(x=1:100, y=1, z=seq(0, global_max, length.out=100)), aes(x=x, y=y, fill=z)) +
        geom_tile() +
        scale_fill_viridis(
          option = "plasma",
          name = "Normalized Mutual Information",
          limits = c(0, global_max),
          direction = 1,
          guide = guide_colorbar(
            title.position = colorbar_title_position,
            title.hjust = colorbar_title_hjust,
            title.vjust = colorbar_title_vjust,
            barwidth = colorbar_width,
            barheight = colorbar_height
          )
        ) + 
        theme_void() +
        theme(
        legend.position = "bottom",
        # Use consistent font for legend title and text using parameters
        legend.title = element_text(
            size = colorbar_title_size, 
            face = colorbar_title_face, 
            family = plot_font_family
        ),
        legend.text = element_text(
            size = colorbar_text_size, 
            face = colorbar_text_face, 
            family = plot_font_family
        ),
        legend.margin = colorbar_margin
    )
    
    # Extract just the legend
    legend_grob <- get_legend(color_bar)
    
    # Use plot_grid from cowplot for better control
    plot_heights <- rep(1, length(plot_list))
    if (length(plot_list) > 1) {
      plot_heights[length(plot_list)] <- 1.2  
    }
    
    # Combine plots first
    plots_combined <- plot_grid(
      plotlist = plot_list,
      ncol = 1,
      rel_heights = plot_heights,
      align = 'v'
    )
    
    # Then add the legend with proper spacing
    final_plot <- plot_grid(
      plots_combined,
      legend_grob,
      ncol = 1,
      # Better legend proportions
      rel_heights = c(0.90, 0.10),
      # Add explicit vertical spacing
      align = 'v',
      axis = 'l',
      # Add less padding between the plots and legend
      vjust = 1.2  # Reduced from 2
    )
    
    # Add a white background with reduced margins
    final_plot_with_margin <- ggdraw() + 
      draw_plot(final_plot, x = 0, y = 0, width = 1, height = 1) +
      theme(
        plot.margin = final_plot_margin,
        plot.background = element_rect(fill = "white", color = NA)
      )
    
    return(final_plot_with_margin)
  } else {
    stop("No plots were created. Check if the data contains valid MI matrices.")
  }
}

# Main execution section
# Load all latent types data
all_latent_data <- load_all_latent_types(data_dir)

# Create stacked heatmaps for all available base latent types
mi_heatmap <- create_stacked_mi_heatmaps(
  all_latent_data, 
  mi_matrix_type, 
  instance_type = instance_type,
  global_max = set_global_max
)
#global_max = if(mi_matrix_type == "total") all_latent_data$global_max_total else all_latent_data$global_max_instance,


# Count number of available base latent types for sizing
base_latent_count <- sum(c("X", "OC1", "OC2", "OC3", "OC4", "OC5", "OC6") %in% names(all_latent_data))


# Save with appropriate dimensions
ggsave(
  save_path,
  mi_heatmap,
  width = max(15, if(!is.null(all_latent_data$X)) nrow(all_latent_data$X) * 0.2 else 12),
  # Slightly less height with better proportions
  height = 2 + 2 * base_latent_count + 1.5,  # Reduced extra height from 3 to 1.5
  dpi = 600,
  limitsize = FALSE,
  bg = "white",
  device = "png"  # Explicitly set the device to PNG
)

# Print information about created visualization
cat("Created stacked visualization with", base_latent_count, "latent types\n")
cat("Saved to:", save_path, "\n")
cat("Matrix type:", mi_matrix_type, "\n")
cat("Global max value:", if(mi_matrix_type == "total") all_latent_data$global_max_total else all_latent_data$global_max_instance, "\n")