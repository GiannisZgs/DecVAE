from dataclasses import dataclass, field
from typing import List, Optional
from utils import list_field

@dataclass
class VisualizationsArguments:
    """
    Arguments pertaining to the visualizations.
    """
    comment_vis_args: str = field(
        default = "",
        metadata={"help": "A comment to add to the visualization arguments."},
    )
    random_seed_vis: int = field(
        default=42,
        metadata={"help": "The random seed used for visualization reproducibility."},
    )
    save_vis_dir: str = field(
        default=None,
        metadata={"help": "The directory where visualizations will be saved."},
    )
    seq_to_vis: int = field(
        default=1000,
        metadata={"help": "The number of utterances to visualize for the sequence variable."},
    )
    frames_to_vis: int = field(
        default=100,
        metadata={"help": "The number of utterances to visualize for the frame variable."},
    )
    vis_td_frames: bool = field(
        default=False,
        metadata={"help": "Whether to visualize time-domain (raw waveform) features for the frame variable."},
    )
    vis_mel_frames: bool = field(
        default=False,
        metadata={"help": "Whether to visualize mel-filterbank features for the frame variable."},
    )
    vis_td_seq: bool = field(
        default=False,
        metadata={"help": "Whether to visualize time-domain (raw waveform) features for the sequence variable."},
    )
    vis_mel_seq: bool = field(
        default=False,
        metadata={"help": "Whether to visualize mel-filterbank features for the sequence variable."},
    )
    set_to_use_for_vis: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset split to use for visualization (e.g., 'train', 'dev', 'test', 'all')."},
    )
    use_umap: bool = field(
        default=False,
        metadata={"help": "Whether to use UMAP for dimensionality reduction in visualizations in addition to the default T-SNE."},
    )
    variables_to_plot: List[str] = list_field(
        default=["speaker_id", "phoneme", "king_stage"],
        metadata={"help": "The list of variables to plot in the visualizations for the VOC-ALS dataset. Available are: "
        "speaker_id, phoneme, king_stage, disease_duration, cantagallo, group, alsfrs_total, alsfrs_speech"},
    )
    variables_to_plot_seq: List[str] = list_field(
        default=["speaker_id", "vowel", "king_stage"],
        metadata={"help": "The list of variables to plot in the visualizations for the sequence variable in the VOC-ALS dataset."},
    )
    sel_vowels_list_timit: List[str] = list_field(
        default=['iy','ey','ay','aw','ow','uh','uw'],
        metadata={"help": "The list of vowels to select for visualization in the TIMIT dataset."},
    )
    sel_consonants_list_timit: List[str] = list_field(
        default = ['b','d','f','k','l','s'],
        metadata={"help": "The list of consonants to select for visualization in the TIMIT dataset."},
    )
    sel_phonemes_list_timit: List[str] = list_field(
        default=['iy','ey','ay','aw','ow','uh','uw','b','d','f','k','l','s'],
        metadata={"help": "The list of phonemes to select for visualization in the TIMIT dataset."},
    )
    sel_phonemes_list_iemocap: List[str] = list_field(
        default = ['IY', 'EY', ' AY', 'OW', 'UH', 'UW', 'B', 'D', 'F', 'K', 'L', 'S'],
        metadata={"help": "The list of phonemes to select for visualization in the IEMOCAP dataset."},
    )
    sel_non_verbal_phonemes_iemocap: List[str] = list_field(
        default=['SIL', '+BREATHING+', "+LIPSMACK", "+LAUGHTER+"],
        metadata={"help": "The list of non-verbal phonemes to select for visualization in the IEMOCAP dataset."},
    )

    visualize_train_set: bool = field(
        default=False,
        metadata={"help": "Whether to visualize latent samples from the training set."},
    )
    visualize_dev_set: bool = field(
        default=True,
        metadata={"help": "Whether to visualize latent samples from the development set."},
    )
    visualize_test_set: bool = field(
        default=False,
        metadata={"help": "Whether to visualize latent samples from the test set."},
    )
    latent_train_set_frames_to_vis: int = field(
        default=30,
        metadata={"help": "The number of utterances samples to visualize from the latent frame Z variable, in the training set."},
    )
    latent_dev_set_frames_to_vis: int = field(
        default=30,
        metadata={"help": "The number of utterances samples to visualize from the latent frame Z variable, in the development set."},
    )
    latent_test_set_frames_to_vis: int = field(
        default=30,
        metadata={"help": "The number of utterances samples to visualize from the latent frame Z variable, in the test set."},
    )
    latent_train_set_seq_to_vis: int = field(
        default=1000,
        metadata={"help": "The number of utterances samples to visualize from the latent sequence S variable, in the training set."},
    )
    latent_dev_set_seq_to_vis: int = field(
        default=300,
        metadata={"help": "The number of utterances samples to visualize from the latent sequence S variable, in the development set."},
    )
    latent_test_set_seq_to_vis: int = field(
        default=100,
        metadata={"help": "The number of utterances samples to visualize from the latent sequence S variable, in the test set."},
    )
    visualize_latent_frame: bool = field(
        default=False,
        metadata={"help": "Whether to visualize the latent frame Z variable."},
    )
    visualize_latent_sequence: bool = field(
        default=False,
        metadata={"help": "Whether to visualize the latent sequence S variable."},
    )
    vis_isotropic_gaussian_sphere: bool = field(
        default=True,
        metadata={"help": "Whether to visualize isotropic Gaussian sphere samples in the latent space."},
    )
    plot_3d:  bool = field(
        default=False,
        metadata={"help": "Whether to plot the latent space in 3D."},
    )
    variables_to_plot_latent: List[str] = list_field(
        default=["vowel", "speaker_id"],
        metadata={"help": "The list of variables to plot in the latent space visualizations."},
    )
    variables_to_plot_latent_seq: List[str] = list_field(
        default=["speaker_id"],
        metadata={"help": "The list of variables to plot in the latent sequence space visualizations."},
    )
    aggregation_strategies_to_plot_frame: List[str] = list_field(
        default=["all", "X_OCs_freq"],
        metadata={"help": "The latent subspace aggregation strategies to plot latent space frame visualizations."},
    )
    aggregation_strategies_to_plot_seq: List[str] = list_field(
        default=["all", "X_OCs_freq"],
        metadata={"help": "The latent subspace aggregation strategies to plot latent space sequence visualizations."},
    )
