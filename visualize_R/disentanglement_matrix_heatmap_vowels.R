library(jsonlite)
library(dplyr)
library(purrr)
library(tidyr)
library(ggplot2)
library(hrbrthemes)
library(tibble)
library(viridis)

parent_save_dir <- file.path('/home/giannis/Documents/DecSSL/R_vis/latent_responses')
model <- 'vae1D_FC_mel' #'vae1D_FC_mel'
beta <- 'snr15_b01_vae1d_fc_mel_bs16'
ckp <- 'training_ckp_epoch_109'
latent <- 'X'
mi_matrix_type <- 'instance' # 'total'


if (grepl("vae",model)) {
  parent_dir <- file.path('/home/giannis/Documents/latent_responses_vae')
} else {
  parent_dir <- file.path('/home/giannis/Documents/latent_responses')
}

data_dir <- file.path(parent_dir, model, beta, 'disentanglement_matrices', paste0('checkpoint-',ckp) ,'latent_vis_dict.json')

save_dir <- file.path(parent_save_dir, model, beta, ckp)
if (!dir.exists(save_dir)) {
  dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
}
save_path <- file.path(save_dir, paste0(latent,'_mi_matrix_',mi_matrix_type,'.png'))


load_json_data <- function(file_path) {
    json_text <- readLines(file_path)
    json_text <- gsub("\\bNaN\\b", "\"NaN\"", json_text)
    fromJSON(json_text, simplifyDataFrame = TRUE, simplifyMatrix = TRUE)
}


load_latent_vis_data <- function(file_path,latents = 'X') {
  # Use the existing load_json_data function to handle NaN values
  data <- load_json_data(file_path)
  
  # Extract relevant fields
  if (latents == 'X') {
    total_mi_mat <- data$X
    phonemes_instance_mi_mat <- data$X_instance$phonemes_instance_mi
    speakers_instance_mi_mat <- data$X_instance$speakers_instance_mi
  } else if (latents == 'OC1') {
    total_mi_mat <- data$OC1
    phonemes_instance_mi_mat <- data$OC1_instance$phonemes_instance_mi
    speakers_instance_mi_mat <- data$OC1_instance$speakers_instance_mi
  } else if (latents == 'OC2') {
    total_mi_mat <- data$OC2
    phonemes_instance_mi_mat <- data$OC2_instance$phonemes_instance_mi
    speakers_instance_mi_mat <- data$OC2_instance$speakers_instance_mi
  } else if (latents == 'OC3') {
    total_mi_mat <- data$OC3
    phonemes_instance_mi_mat <- data$OC3_instance$phonemes_instance_mi
    speakers_instance_mi_mat <- data$OC3_instance$speakers_instance_mi
  } else if (latents == 'OC4') {
    total_mi_mat <- data$OC4
    phonemes_instance_mi_mat <- data$OC4_instance$phonemes_instance_mi
    speakers_instance_mi_mat <- data$OC4_instance$speakers_instance_mi
  } else if (latents == 'OCs_proj') {
    total_mi_mat <- data$OCs_proj
    phonemes_instance_mi_mat <- data$OCs_proj_instance$phonemes_instance_mi
    speakers_instance_mi_mat <- data$OCs_proj_instance$speakers_instance_mi
  } else if (latents == 'OCs_joint') {
    total_mi_mat <- data$OCs_joint
    phonemes_instance_mi_mat <- data$OCs_joint_instance$phonemes_instance_mi
    speakers_instance_mi_mat <- data$OCs_joint_instance$speakers_instance_mi
  } else if (latents == 'all') {
    total_mi_mat <- data$all
    phonemes_instance_mi_mat <- data$all_instance$phonemes_instance_mi
    speakers_instance_mi_mat <- data$all_instance$speakers_instance_mi
  } else {
    stop("Invalid latents value. Choose from 'X', 'OC1', 'OC2', 'OC3', 'OCs_proj', 'OCs_joint', or 'all'.")
  }
  
  if (!is.null(data$phonemes_unique_values)) {
    phoneme <- data$phonemes_unique_values
  } else if (!is.null(data$phonemes)) {
    phoneme <- sort(unique(data$phonemes))
  }

  if (!is.null(data$speaker_unique_values)) {
    speaker <- data$speaker_unique_values
  } else if (!is.null(data$speakers)) {
    speaker <- sort(unique(data$speakers))
  }

  # Convert data to appropriate formats if needed
  if (is.list(total_mi_mat)) total_mi_mat <- do.call(rbind, total_mi_mat)
  if (is.list(phonemes_instance_mi_mat)) phonemes_instance_mi_mat <- do.call(rbind, phonemes_instance_mi_mat)
  if (is.list(speakers_instance_mi_mat)) speakers_instance_mi_mat <- do.call(rbind, speakers_instance_mi_mat)
  if (is.list(phoneme)) phoneme <- unlist(phoneme)
  if (is.list(speaker)) speaker <- unlist(speaker)
  
  # Calculate global max values across all latent types
  global_max_total <- 0
  global_max_instance <- 0
  
  latent_types <- c("X", "OC1", "OC2", "OC3", "OC4", "OCs_proj", "OCs_joint", "all")
  
  for (type in latent_types) {
    if (!is.null(data[[type]])) {
      total_mat <- data[[type]]
      if (is.list(total_mat)) total_mat <- do.call(rbind, total_mat)
      global_max_total <- max(global_max_total, max(total_mat, na.rm = TRUE))
    }
    
    instance_key <- paste0(type, "_instance")
    if (!is.null(data[[instance_key]])) {
      if (!is.null(data[[instance_key]]$phonemes_instance_mi)) {
        phoneme_mat <- data[[instance_key]]$phonemes_instance_mi
        if (is.list(phoneme_mat)) phoneme_mat <- do.call(rbind, phoneme_mat)
        global_max_instance <- max(global_max_instance, max(phoneme_mat, na.rm = TRUE))
      }
      
      if (!is.null(data[[instance_key]]$speakers_instance_mi)) {
        speaker_mat <- data[[instance_key]]$speakers_instance_mi
        if (is.list(speaker_mat)) speaker_mat <- do.call(rbind, speaker_mat)
        global_max_instance <- max(global_max_instance, max(speaker_mat, na.rm = TRUE))
      }
    }
  }

  # Return structured data with global maxima
  return(list(
    total_mi_mat = total_mi_mat,  
    phonemes_instance_mi_mat = phonemes_instance_mi_mat, 
    speakers_instance_mi_mat = speakers_instance_mi_mat,            
    phoneme = phoneme,    # Phoneme labels
    speaker = speaker,    # Speaker labels
    global_max_total = global_max_total,         # Global max for total MI
    global_max_instance = global_max_instance    # Global max for instance MI
  ))
}

# Load the data
vis_data <- load_latent_vis_data(data_dir, latent = latent)



create_mi_heatmap <- function(mi_matrix, factor_names = NULL, dim_names = NULL, horizontal = FALSE, global_max = NULL) {
  # For total MI matrix, transpose if horizontal layout is requested
  if (horizontal && ncol(mi_matrix) <= 5) {  # Assuming total MI has 2 factors
    # Transpose the matrix to get factors as rows and dimensions as columns
    mi_matrix <- t(mi_matrix)
    
    # Prepare data for horizontal plotting (factors as rows, dimensions as columns)
    mi_df <- as.data.frame(mi_matrix) %>%
      # Add factor labels as row names
      rownames_to_column("Factor") %>%
      # Assign factor names if provided
      mutate(Factor = if(!is.null(factor_names)) factor_names else Factor) %>%
      # Convert to long format
      pivot_longer(cols = -Factor, 
                   names_to = "Dimension", 
                   values_to = "MI") %>%
      # Ensure dimensions are ordered numerically
      mutate(Dimension = factor(
        Dimension, 
        levels = paste0("V", 1:ncol(mi_matrix))
      ))
    
    # Replace V1, V2, etc. with actual dimension names if provided
    if (!is.null(dim_names)) {
      levels(mi_df$Dimension) <- dim_names
    } else {
      # Default to z_1, z_2, etc.
      levels(mi_df$Dimension) <- paste0("z_", 1:ncol(mi_matrix))
    }

    p <- ggplot(mi_df, aes(x = Dimension, y = Factor, fill = MI)) + 
      # White background for the plot
      theme_minimal() +
      # Add white tiles with borders
      geom_tile(color = "white", linewidth = 0.1) +
      # Set color scale to start at 0 and go to global max
      scale_fill_distiller(
        palette = "RdPu", 
        name = "Mutual\nInformation",
        limits = c(0, if(is.null(global_max)) max(mi_df$MI, na.rm = TRUE) else global_max),
        direction = 1
      ) +
      # Removed title as requested
      labs(
        x = "Latent Dimension",
        y = "Factor"
      ) +
      theme(
        # Add white background
        panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA),
        # Other formatting
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8),
        axis.text.y = element_text(size = 12, face = "bold"),
        legend.position = "right",
        panel.grid = element_blank(),
        # Ensure there's enough room for dimension labels
        plot.margin = margin(5, 5, 20, 5)
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
      # White background for the plot
      theme_minimal() +
      # Add white tiles with borders
      geom_tile(color = "white", linewidth = 0.1) +
      scale_y_discrete(labels = function(x) {
            # Extract numbers after z_
            nums <- as.numeric(gsub("z_", "", x))
            # Create expressions with subscripts
            parse(text = paste0("z[", nums, "]"))
        }) +
      #scale_y_discrete(limits = rev(paste0("z_", 1:nrow(mi_matrix)))) +
      # Set color scale to start at 0 and go to global max
      scale_fill_distiller(
        palette = "RdPu", 
        name = "Mutual\nInformation",
        limits = c(0, if(is.null(global_max)) max(mi_df$MI, na.rm = TRUE) else global_max),
        direction = 1,
      ) +
      # Removed title as requested
      labs(
        x = "Factor",
        y = "Latent Dimension"
      ) +
      theme(
        # Add white background
        panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA),
        # Other formatting
        axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        axis.text.y = element_text(size = 10),
        legend.position = "right",
        panel.grid = element_blank()
      )
    
    return(p)
  }
}



# 1 & 2. Create total MI heatmap (total mutual information per dimension)
    if (mi_matrix_type == "total") {
    # For total MI, we want a horizontal layout with dimensions as columns and factors as rows
    # Ensure total_mi_mat is properly oriented (rows = dimensions, cols = factors)
    if (ncol(vis_data$total_mi_mat) != 2) {
        warning("Expected 2 columns in total_mi_mat (one for each factor). Check matrix orientation.")
    }
    
    # Create dimension labels (1 to 48 or actual row count)
    dim_names <- paste0("z_", 1:nrow(vis_data$total_mi_mat))
    
    # Factor labels
    factor_labels <- c("Phoneme", "Speaker")
    
    # Create the horizontal heatmap using global max
    mi_heatmap <- create_mi_heatmap(
        vis_data$total_mi_mat, 
        factor_labels,
        dim_names,
        horizontal = TRUE,
        global_max = vis_data$global_max_total
    )
    
    # Adjust plot size for many dimensions
    plot_width <- max(10, nrow(vis_data$total_mi_mat) * 0.2)
    ggsave(
        save_path,
        mi_heatmap,
        width = plot_width,
        height = 3,  # Even smaller height for just 2 factors without title
        dpi = 300,
        limitsize = FALSE
    )
} else if (mi_matrix_type == "instance") {
    # Create instance-specific MI heatmaps using the global max

    if (ncol(vis_data$phonemes_instance_mi_mat) == 5) {
        phoneme_labels <- c("a", "e", "I", "aw","u")
        } 
    else if (ncol(vis_data$phonemes_instance_mi_mat) == 8) {
        phoneme_labels <- c("i","I","e","ae","a","aw","y","u")
    }

    speaker_labels <- paste0("SPKR", 1:length(vis_data$speaker))
    
    phoneme_heatmap <- create_mi_heatmap(
        vis_data$phonemes_instance_mi_mat,
        phoneme_labels,
        paste0("z_", 1:nrow(vis_data$phonemes_instance_mi_mat)),
        global_max = vis_data$global_max_instance
    )
    #Give vowel/speaker strings in 2nd argument
    speaker_heatmap <- create_mi_heatmap(
        vis_data$speakers_instance_mi_mat,
        speaker_labels,
        paste0("z_", 1:nrow(vis_data$speakers_instance_mi_mat)),
        global_max = vis_data$global_max_instance
    )

    # Combine the two heatmaps vertically
    if (requireNamespace("gridExtra", quietly = TRUE)) {
        # Add white background to the combined plot
        valid_phoneme <- !is.null(phoneme_heatmap) && inherits(phoneme_heatmap, "ggplot")
        valid_speaker <- !is.null(speaker_heatmap) && inherits(speaker_heatmap, "ggplot")
        
        if (valid_phoneme && valid_speaker) {
            # Both heatmaps are valid, arrange them together
            mi_heatmap <- gridExtra::grid.arrange(
            phoneme_heatmap, 
            speaker_heatmap, 
            ncol = 1,
            heights = c(0.6, 0.4),
            top = ""  # No title
            )
        } else if (valid_phoneme) {
            # Only phoneme heatmap is valid
            mi_heatmap <- phoneme_heatmap
            warning("Speaker heatmap could not be created. Showing only phoneme heatmap.")
        } else if (valid_speaker) {
            # Only speaker heatmap is valid
            mi_heatmap <- speaker_heatmap
            warning("Phoneme heatmap could not be created. Showing only speaker heatmap.")
        } else {
            # Neither heatmap is valid
            stop("Could not create either phoneme or speaker heatmaps.")
        }
        } else {
        # gridExtra not available, use first valid heatmap
        if (!is.null(phoneme_heatmap) && inherits(phoneme_heatmap, "ggplot")) {
            mi_heatmap <- phoneme_heatmap
        } else if (!is.null(speaker_heatmap) && inherits(speaker_heatmap, "ggplot")) {
            mi_heatmap <- speaker_heatmap
        } else {
            stop("Could not create either phoneme or speaker heatmaps.")
        }
        }
    
    # Save the instance-specific plot
    ggsave(
        save_path,
        mi_heatmap,
        width = 10,
        height = 8,
        dpi = 300
    )
}

# We need to use a consistent color level for all heatmaps
# Add white background to the heatmap
# Remove title
# Fix colorbar 0 - max 


# Plan: DONE - 1. Make horizontal heatmap of single latent with all dimensions - both factors (phoneme/speaker)
#       2. Make horizontal heatmap of single latent with all dimensions - single factors (phoneme/speaker)
#       3. Make combined horizontal heatmap of multiple latents for both factors (phoneme/speaker) - OCs_joint
#       4. Make combined horizontal heatmap of multiple latents for single factor (phoneme/speaker) - OCs_joint
#       5. Make rectangular heatmap of multiple latents for single factor (phoneme/speaker) - OCs_proj
