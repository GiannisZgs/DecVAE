# Latent analysis utilities initialization
from .classification_utils import prediction_eval, greedy_cluster_mapping, calculate_unweighted_accuracy
from .visualization_utils import visualize
from .latent_response_utils import (
    analyze_latents_wrt_factor,
    visualize_latent_response,
    calculate_variance_dimensions,
    save_latent_representation,
    average_latent_representations
)
