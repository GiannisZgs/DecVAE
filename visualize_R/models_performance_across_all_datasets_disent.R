library(dplyr)
library(ggplot2)
library(stringr) 
library(tidyr) 
library(fmsb)  # For radar charts
library(RColorBrewer)
library(ggbreak)
library(ggrepel)
library(gridExtra)
library(grid)

# Set and create directory and filepaths to save
parent_save_dir <-  file.path('..','figures','model_performance_vowels_only_mel_models')

scatter_2d_save_dir <- file.path(parent_save_dir, "scatterplots_two_metrics")
if (!dir.exists(scatter_2d_save_dir)) {
  dir.create(scatter_2d_save_dir, recursive = TRUE, showWarnings = FALSE)
}

scatter_3d_save_dir <- file.path(parent_save_dir, "scatterplots_three_metrics")
if (!dir.exists(scatter_3d_save_dir)) {
  dir.create(scatter_3d_save_dir, recursive = TRUE, showWarnings = FALSE)
}

# Style and font parameters
plot_font_family <- "Arial"
plot_background_color <- "white"
plot_text_color <- "black"

#showtext_auto(enable = TRUE)

# Plot types
plot_2D_scatterplots <- TRUE
plot_3D_scatterplots <- TRUE

#color palette
palette <- "plasma"
palette_w_yellows <- "Set1"
yellow_block_threshold <- 0.8

# Axis titles
axis_title_size_scatter <- 30
axis_title_size_bubble <- 30
axis_title_face <- "plain"

# Axis tick labels
axis_text_size_scatter <- 30
axis_text_size_bubble <- 30
axis_text_face <- "plain"

# Plot title
plot_title_size_scatter <- 0
plot_title_size_bubble <- 0
plot_title_face <- "plain"
plot_title_hjust <- 0.5
plot_title_vjust <- 0.5

# Legend elements
legend_text_size_scatter <- 12
legend_text_size_bubble <- 12
legend_title_size_scatter <- 13
legend_title_size_bubble <- 13
legend_text_face <- "plain"
legend_title_face <- "plain"
legend_key_size_scatter <- 1  # in cm
legend_key_size_bubble <- 1  # in cm

# Geom elements
point_size_scatter <- 10
point_size_bubble_min <- 3
point_size_bubble_max <- 15
text_size_scatter <- 9
text_size_bubble <- 9
line_size_repel <- 1
line_alpha_repel <- 0.6

# Margin settings
plot_margin_scatter <- margin(10, 10, 10, 10, "pt")
plot_margin_bubble <- margin(10, 10, 10, 10, "pt")

selected_metrics_all <- c("gaussian_corr_norm", "disentanglement", "completeness", "informativeness", "IRS")

selected_metrics_all_labels <- c(
    "mutual_info" = "Mutual Information",
    "gaussian_corr_norm" = "Gaussian Corr.",
    "disentanglement" = "Disentanglement",
    "completeness" = "Completeness",
    "informativeness" = "Informativeness",
    "explicitness" = "Explicitness",
    "modularity" = "Modularity",
    "IRS" = "Robustness",
    "supervised_phoneme_recognition" = "Phoneme Recognition (Supervised)",
    "supervised_speaker_identification" = "Speaker Identification (Supervised)",
    "unsupervised_phoneme_recognition" = "Phoneme Recognition (Unsupervised)",
    "unsupervised_speaker_identification" = "Speaker Identification (Unsupervised)"
  )
  

#Sim Vowels
metrics_vowels <- c("mutual_info", "gaussian_corr_norm", "disentanglement", "completeness", 
             "informativeness", "explicitness", "modularity","IRS", 
             "supervised_phoneme_recognition", "supervised_speaker_identification",
             "unsupervised_phoneme_recognition", "unsupervised_speaker_identification")

selected_models_vowels <- c("PCA",
                     "VAE", "betaVAE", "DecVAE_FD", "betaDecVAE_FD")



# Store model data 
model_data_vowels <- data.frame(
  metric = metrics_vowels,
  PCA = c(0.004,0.035,0.230,0.239,0.801,0.894,0.645,0.477,0.974, 0.741, 0.433, 0.499),
  ICA = c(0.005,4.85,0.086,0.156,0.812,0.894,0.605,0.538,0.974, 0.741, 0.455, 0.506),
  
  AE = c(0.044,1.715,0.329,0.147,0.794,0.902,0.779,0.570,0.976, 0.744, 0.561, 0.316),
  VAE = c(0.011,0.044,0.03,0.578,0.781,0.789,0.450,0.517,0.939, 0.738, 0.373, 0.332),
  betaVAE = c(0.007,0.061,0.062,0.615,0.805,0.772,0.563,0.552,0.963, 0.732, 0.367, 0.265),
  
  DecAE_FD = c(0.064,0.459,0.526,0.219,0.805,0.947,0.817,0.521,0.944, 0.938, 0.317,0.588),
  DecAE_EWT = c(0.089,0.565,0.529,0.26,0.761,0.913,0.871,0.619,0.974, 0.956, 0.388,0.550),

  DecVAE_FD= c(0.031,0.102,0.618,0.22,0.761,0.921,0.828,0.655,0.916, 0.769, 0.391, 0.516),
  DecVAE_EWT = c(0.040,0.114,0.553,0.233,0.717,0.918,0.779,0.626,0.974, 0.581, 0.412, 0.366),

  betaDecVAE_FD= c(0.051,0.173,0.526,0.196,0.809,0.929,0.809,0.600,0.924, 0.750, 0.334, 0.536),
  betaDecVAE_EWT = c(0.036,0.147,0.582,0.222,0.742,0.915,0.905,0.523,0.977, 0.713, 0.377, 0.638)
)

model_display_names_simvowels <- c(
  "PCA" = "PCA",
  "AE" = "AE",
  "VAE" = "VAE",
  "betaVAE" = "β-VAE",
  "DecAE_FD" = "DecAE + FD",
  "DecAE_EWT" = "DecAE + EWT",
  "DecVAE_FD" = "DecVAE + FD",
  "DecVAE_EWT" = "DecVAE + EWT",
  "betaDecVAE_FD" = "β-DecVAE + FD",
  "betaDecVAE_EWT" = "β-DecVAE + EWT"
)

#TIMIT


metrics_timit <- c("mutual_info", "gaussian_corr_norm", "disentanglement", "completeness", 
             "informativeness", "explicitness", "modularity","IRS", 
             "phoneme_recognition", "speaker_identification")


selected_models_timit <- c("PCA", 
                     "VAE", "betaVAE", "DecVAE_EWT", "betaDecVAE_EWT")


# Store model data 
model_data_timit <- data.frame(
  metric = metrics_timit,
  PCA = c(0.0021, 0.0009266, 0.22912 , 0.15596 , 0.25837 ,0.64766 , 0.7293, 0.5509, 0.43106, 0.22338),
  ICA = c(0.0028, 0.21263, 0.13873, 0.084868 , 0.26239 , 0.64766 , 0.71436, 0.554, 0.44622 , 0.22337),
  raw_mels = c(0.1905, 4.02072, 0.10904 , 0.035566 , 0.26486 , 0.67851 , 0.98013 , 0.55682, 0.44593 , 0.24499 ),
  
  AE = c(0.093878, 2.37793, 0.1531, 0.062913, 0.19079, 0.57943, 0.95742, 0.5439, 0.4485, 0.31066),
  VAE = c(0.041274, 0.084475, 0.19096, 0.11677, 0.18826, 0.4918, 0.98014, 0.68308 , 0.46553 , 0.26989),
  betaVAE = c(0.021917,0.074006 , 0.20172 , 0.16403 , 0.18385 , 0.48599 , 0.93309 , 0.66886 , 0.45524 , 0.28806 ),
  
  DecAE_FD = c(0.035628, 0.21727, 0.23037 , 0.071132, 0.4079, 0.47188, 0.65718, 0.62506 , 0.46945 , 0.73253),
  DecAE_EWT = c(0.0335, 0.23648, 0.25427 , 0.085775 , 0.38272 , 0.47531, 0.65032, 0.67547, 0.46346, 0.68683),

  DecVAE_FD = c(0.060716, 0.18443, 0.2033 , 0.065371 , 0.4914 , 0.66606, 0.56884, 0.77173, 0.49421, 0.84889),
  DecVAE_EWT = c(0.065449, 0.18572, 0.23251, 0.072382, 0.4724, 0.67334, 0.54146, 0.70637, 0.49479 , 0.82426),

  betaDecVAE_FD = c(0.051969, 0.1475, 0.17166, 0.06661 , 0.3937, 0.46672 , 0.73948 , 0.87209, 0.43673 , 0.74837),
  betaDecVAE_EWT = c(0.051242, 0.15457, 0.25561, 0.087143, 0.376 , 0.50385, 0.57539, 0.87499, 0.44764 , 0.67773)
)

model_display_names_timit <- c(
  "PCA" = "PCA",
  "ICA" = "ICA",
  "raw_mels" = "Raw Mel Fbank",
  "AE" = "AE",
  "VAE" = "VAE",
  "betaVAE" = "β-VAE",
  "DecAE_FD" = "DecAE + FD",
  "DecAE_EWT" = "DecAE + EWT",
  "DecVAE_FD" = "DecVAE + FD",
  "DecVAE_EWT" = "DecVAE + EWT",
  "betaDecVAE_FD" = "β-DecVAE + FD",
  "betaDecVAE_EWT" = "β-DecVAE + EWT"
)

#VOC-ALS


metrics_voc_als <- c("mutual_info", "gaussian_corr_norm", "disentanglement", "completeness", 
             "informativeness", "explicitness", "modularity","IRS", 
             "phoneme_recognition", "disease_duration_prediction",
             "kings_staging_prediction")

selected_models_voc_als <- c("PCA",
                      "betaVAE_vowels", "betaVAE_timit",
                     "betaDecVAE_FD_vowels", "betaDecVAE_FD_timit")
#"PCA", "ICA", 

# Store model data 
model_data_voc_als <- data.frame(
  metric = metrics_voc_als,
  PCA = c( 0.180 , 2.901  , 0.128  , 0.0805  , 0.513  , 0.623 , 0.839, 0.658,0 ,0  ,0 ),
  raw_mels = c(0.17913 , 2.88385 , 0.1234 , 0.082694 , 0.50649 , 0.62357 , 0.84538 , 0.65187 , 0.76381 , 0.69026 , 0.68286 ),
  AE_vowels = c(0.091397 , 2.98506 , 0.1073 , 0.070382 , 0.37438 , 0.56997, 0.86571, 0.55294, 0.71823 , 0.59022 , 0.57977 ),
  AE_timit = c( 0.09161, 3.01362, 0.1029 , 0.067915 , 0.36919 , 0.57386, 0.79231, 0.74264 , 0.64683 , 0.48051 , 0.47396 ),

  VAE_vowels = c(0.035782 , 0.1344 , 0.074728 , 0.069139 , 0.3499 , 0.51626 , 0.85327, 0.46771 , 0.63206 , 0.47382 , 0.4629),
  VAE_timit = c( 0.1217, 0.34092, 0.076864 , 0.046739 , 0.34015 , 0.54566 , 0.88843 , 0.5, 0.61966 , 0.44513 , 0.4145),

  betaVAE_vowels = c( 0.042769, 0.21856 , 0.073676  , 0.06515 , 0.36475 , 0.54348 , 0.77463 , 0.48127 , 0.71775 , 0.58774 , 0.58447 ),
  betaVAE_timit = c(0.067234 ,0.33216 , 0.086466  , 0.067478 , 0.35007 , 0.55535 , 0.86305, 0.48909 , 0.63419 , 0.44513 , 0.43826 ),

  betaDecVAE_FD_vowels = c( 0.043877, 0.1547, 0.097035 , 0.085435 , 0.65322 , 0.62568 , 0.75035 , 0.72901 , 0.9154 ,0.89257 , 0.89133),
  betaDecVAE_EWT_vowels = c( 0.036032, 0.14952, 0.099141 , 0.081047 , 0.65913  , 0.66163, 0.75033 , 0.65006 , 0.93824 , 0.91233 , 0.91017 ),

  betaDecVAE_FD_timit = c(0.039401 , 0.20365, 0.1721 , 0.1013 , 0.6118 , 0.6703, 0.76702 , 0.62374 , 0.84144 , 0.81664 , 0.80862 ),
  betaDecVAE_EWT_timit = c(0.045773 , 0.22885, 0.17755 , 0.093022 , 0.57109 , 0.65385 , 0.77598, 0.61062 , 0.83067 , 0.77928 , 0.7733)
)

model_display_names_voc_als <- c(
  "PCA" = "PCA",
  "ICA" = "ICA",
  "raw_mels" = "Raw MFCCs",
  "AE_vowels" = "Vowels AE",
  "VAE_vowels" = "Vowels VAE",
  "betaVAE_vowels" = "Vowels β-VAE",
  "AE_timit" = "TIMIT AE",
  "VAE_timit" = "TIMIT VAE",
  "betaVAE_timit" = "TIMIT β-VAE",
  "betaDecVAE_FD_vowels" = "Vowels β-DecVAE + FD",
  "betaDecVAE_EWT_vowels" = "Vowels β-DecVAE + EWT",
  "betaDecVAE_FD_timit" = "TIMIT β-DecVAE + FD",
  "betaDecVAE_EWT_timit" = "TIMIT β-DecVAE + EWT"
)




#IEMOCAP
metrics_iemocap <- c("mutual_info", "gaussian_corr_norm", "disentanglement", "completeness", 
             "informativeness", "explicitness", "modularity", "IRS",
             "weighted_accuracy_emotion", "unweighted_accuracy_emotion", "f1_emotion",
             "weighted_accuracy_phoneme", "f1_phoneme", "weighted_accuracy_speaker", "f1_speaker")
#"PCA",
selected_models_iemocap <- c("PCA", "VAE_vowels", "betaVAE_timit",
                     "DecVAE_FD_vowels",  "betaDecVAE_FD_timit")


# Store model data 
#ICA = c(,  ,  ,  ,  ,  ,  ,  , 0.405 , 0.403  ,0.405 ,0.508 ,0.401 ,0.247 ,0.242),
#PCA = c(,  ,  ,  ,  ,  ,  ,  , 0.405 , 0.403  ,0.405 ,0.508 ,0.401 ,0.247 ,0.242),
model_data_iemocap <- data.frame(
    metric = metrics_iemocap,
    PCA = c( 0, 0 ,0  , 0 , 0 ,0  ,0  ,0  , 0.405 , 0.403  ,0.405 ,0.508 ,0.401 ,0.247 ,0.242),
    VAE_vowels = c(0.034, 0.0699 , 0.09 , 0.221 , 0.327 , 0.226 , 0.963 , 0.567 , 0.367 , 0.360 ,0.363 , 0.482, 0.359, 0.163, 0.158),
    VAE_timit = c(0.049 , 0.104 ,0.155 ,0.129 ,0.330 ,0.290 ,0.971 ,0.617  , 0.363  , 0.360 , 0.362 , 0.481 , 0.376 , 0.154 , 0.151 ),
    betaVAE_vowels = c(0.019 , 0.084 , 0.071 , 0.238 , 0.332 ,0.266 ,0.907, 0.556   ,0.372 ,0.366 ,0.369 ,0.489 ,0.371 ,0.170 ,0.166 ),
    betaVAE_timit = c(0.024 ,0.078 ,0.118 , 0.199 , 0.330 ,0.265 ,0.966 ,0.607  , 0.366 , 0.359 , 0.362 , 0.484 , 0.365 , 0.166 ,0.163 ),

    DecVAE_FD_vowels = c( 0.017,0.091 ,0.233 , 0.141 , 0.341 ,0.277 ,0.940 ,0.809  , 0.357 , 0.350 , 0.353 , 0.536, 0.451, 0.737, 0.741),
    DecVAE_EWT_vowels = c(0.017 ,0.111 ,0.240 ,0.153 , 0.341 , 0.278 ,0.909 , 0.576  , 0.359, 0.359, 0.357,0.537 ,0.453 ,0.766 ,0.768 ),
    DecVAE_FD_timit = c(0.017 ,0.134 ,0.289 ,0.209 ,0.341 ,0.285 ,0.831 ,0.778  ,0.365 , 0.361,0.354 ,0.534 ,0.446 ,0.718 ,0.722 ),
    DecVAE_EWT_timit = c( 0.019, 0.130 ,0.238 ,0.158 ,0.335 ,0.297 ,0.849 ,0.808  ,0.367 ,0.361 ,0.354 ,0.534 ,0.446 ,0.718 ,0.722 ),

    betaDecVAE_FD_vowels = c(0.019 ,0.116 ,0.451 , 0.293, 0.367, 0.277, 0.755, 0.683 ,0.354 ,0.332 ,0.351 ,0.533 ,0.447 ,0.737 ,0.740 ),
    betaDecVAE_EWT_vowels = c(0.019 , 0.053 ,0.28 ,0.160 ,0.359 ,0.290 ,0.889 ,0.799  ,0.360 ,0.357 ,0.357 ,0.533 ,0.443 ,0.766 ,0.714 ),
    betaDecVAE_FD_timit = c( 0.025, 0.127 ,0.231 ,0.118 ,0.316 ,0.243 ,0.788 ,0.713  , 0.365, 0.339, 0.363, 0.528, 0.437,0.677 ,0.685 ),
    betaDecVAE_EWT_timit = c(0.022, 0.136,0.254 ,0.149 ,0.317 ,0.242 ,0.879 ,0.755  , 0.341, 0.332,0.338 ,0.534 ,0.436 ,0.672 ,0.678 )
)


model_display_names_iemocap <- c(
#"PCA" = "PCA",
#"ICA" = "ICA",
#"raw_mels" = "Raw Mel Fbank",
"VAE_vowels" = "Vowels VAE",
"VAE_timit" = "TIMIT VAE",
"betaVAE_vowels" = "Vowels β-VAE",
"betaVAE_timit" = "TIMIT β-VAE",
"DecVAE_FD_vowels" = "Vowels DecVAE + FD",
"DecVAE_EWT_vowels" = "Vowels DecVAE + EWT",
"DecVAE_FD_timit" = "TIMIT DecVAE + FD",
"DecVAE_EWT_timit" = "TIMIT DecVAE + EWT",
"betaDecVAE_FD_vowels" = "Vowels β-DecVAE + FD",
"betaDecVAE_EWT_vowels" = "Vowels β-DecVAE + EWT",
"betaDecVAE_FD_timit" = "TIMIT β-DecVAE + FD",
"betaDecVAE_EWT_timit" = "TIMIT β-DecVAE + EWT"
)
