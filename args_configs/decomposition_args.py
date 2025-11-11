from dataclasses import dataclass, field
from typing import List, Optional
from utils import list_field

@dataclass
class DecompositionArguments:
    """
    Arguments pertaining to the decomposition model.
    """
    comment_decomp_args: str = field(
        metadata={"help": "A comment to add to the decomposition arguments."},
    )
    decomp_to_perform: Optional[str] = field(
        default="filter",
        metadata={"help": "The decomposition operation to perform as part of the pre-processing."},
    )
    frame_decomp: bool = field(
        default=True,
        metadata={"help": "Whether to perform the decomposition operation on the frame level."},
    )
    seq_decomp: bool = field(
        default=False,
        metadata={"help": "Whether to perform the decomposition operation on the sequence level."},
    )
    NoC: int = field(
        default=3,
        metadata={"help": "The number of components to decompose the input frames into."},
    )
    NoC_seq: int = field(
        default=3,
        metadata={"help": "The number of components to decompose the input sequence into."},
    )
    group_OCs_by_frame: str = field(
        default="equally_distribute",
        metadata={"help": "The way to group the output frame components: high_freqs_first, low_freqs_first, equally_distribute."},
    )
    group_OCs_by_seq: str = field(
        default="equally_distribute",
        metadata={"help": "The way to group the output sequence components: high_freqs_first, low_freqs_first, equally_distribute."},
    )
    fs: int = field(
        default=16000,
        metadata={"help": "The sampling frequency of the input audio."},
    )
    freq_groups: List[int] = list_field(
        default= [[0,  750],
           [750, 1500],
           [1500, 2250],
           [2250, 3000],
           [3000, 3750],
           [3750, 4500],
           [4500, 5200],
           [5250, 6000],
           [6000, 6750],
           [6750, 7500]],
        metadata={"help": "The frequency groups used in the simulation procedure."},
    )
    receptive_field: float = field(
        default=0.025,
        metadata={"help": "The receptive field of the decomposition operation - Must match the receptive field of the convolution encoder."},
    )
    stride: float = field(
        default=0.02,
        metadata={"help": "The stride of the decomposition operation - Must match the stride of the convolution encoder."},
    )
    lower_speech_freq: int = field(
        default=50,
        metadata={"help": "Assumption about the lower frequency of the speech signal."},
    )
    higher_speech_freq: int = field(
        default=7500,
        metadata={"help": "Assumption about the higher possible frequency of the speech signal."},
    )
    max_silence_freq: int = field(
        default=250,
        metadata={"help": "The maximum frequency to be considered as a silence component."},
    )
    notch_band_low: int = field(
        default=55,
        metadata={"help": "The lower limit of the notch filter."},
    )
    notch_band_high: int = field(
        default=70,
        metadata={"help": "The upper limit of the notch filter."},
    )
    use_notch_filter: bool = field(
        default=True,
        metadata={"help": "Whether to use a notch filter to remove the 60Hz power line noise."},
    )
    detection_intervals: int = field(
        default=6,
        metadata={"help": "The number of intervals to split the frequency spectrum into, where peak detection will be run."},
    )
    spec_amp_tolerance: float = field(
        default=5e-6,
        metadata={"help": "The tolerance threshold for the amplitude of the frame-level spectrum."},
    )
    spec_amp_tolerance_seq: float = field(
        default=1e-6,
        metadata={"help": "The tolerance threshold for the amplitude of the sequence-level spectrum."},
    )
    global_thres: float = field(
        default=0.01,
        metadata={"help": "The global threshold for peak detection. Peaks under this value are not considered."},
    )
    power_law: float = field(
        default=1.7,
        metadata={"help": "The power law to apply to split the spectrum into non-uniform intervals."},
    )
    nfft: int = field(
        default=512,
        metadata={"help": "The number of points to use for the FFT."},
    )
    min_distance: int = field(
        default=300,
        metadata={"help": "The minimum distance between detected peaks to be considered separate."},
    )
    peak_bandwidth: int = field(
        default=500,
        metadata={"help": "The minimum required bandwidth of the peaks detected by the algorithm."},
    )
    prom_thres: float = field(
        default=0.01,
        metadata={"help": "The prominence threshold for peak detection."},
    )
    N_peaks_to_select: int = field(
        default=2,
        metadata={"help": "The number of largest peaks to consider at each interval of the spectrum."},
    )
    buttord: int = field(
        default=2,
        metadata={"help": "The order of the Butterworth filter that is used for the decomposition."},
    )
    remove_silence: bool = field(
        default=False,
        metadata={"help": "Whether to remove frames found silent based on a frequency domain criterion"},
    )

    emd_spline_kind: str = field(
        default='pchip',
        metadata={"help": "The kind of spline to use for the EMD decomposition."},
    )
    emd_max_iter: int = field(
        default=1000,
        metadata={"help": "The maximum number of iterations per single sifting in EMD."},
    )
    emd_energy_ratio_thr: float = field(
        default=5e-3,
        metadata={"help": "Threshold value on energy ratio per IMF check."},
    )
    emd_std_thr: float = field(
        default=0.1,
        metadata={"help": "Threshold value on standard deviation per IMF check."},
    )
    emd_svar_thr: float = field(
        default=0.01,
        metadata={"help": "Threshold value on scaled variance per IMF check."},
    )
    emd_total_power_thr: float = field(
        default=0.0001,
        metadata={"help": "Threshold value on total power per EMD decomposition."},
    )
    emd_range_thr: float = field(
        default=0.001,
        metadata={"help": "Threshold for amplitude range (after scaling) per EMD decomposition."},
    )
    emd_extrema_detection: str = field(
        default='simple',
        metadata={"help": "Method used to finding extrema."},
    )

    vmd_alpha: int = field(
        default=1500,
        metadata={"help": "The balancing parameter of the data-fidelity constraint in VMD. Controls bandwidth and smoothness of the found modes."},
    )
    vmd_tau: float = field(
        default=0,
        metadata={"help": "Time-step of the dual ascent in VMD. Pick 0 for noise-slack."},
    )
    vmd_DC: bool = field(
        default=False,
        metadata={"help": "True if the first mode is put and kept at DC (0-freq)."},
    )
    vmd_init: int = field(
        default=2,
        metadata={"help": "Initialization of the center frequencies in VMD.(0: all 0, 1: uniformly distributed, 2: random)"},
    )
    vmd_tol: float = field(
        default=1e-8,
        metadata={"help": "Tolerance of convergence criterion in VMD."},
    )
    use_vmd_correction: bool = field(
        default=False,
        metadata={"help": "Whether to use VMD correction of found frequencies."},
    )
    ewt_completion: bool = field(
        default=False,
        metadata={"help": "Whether to complete the EWT decomposition in case less than specified modes are found."},
    )
    ewt_filter: str = field(
        default='gaussian',
        metadata={"help": "The filter to use for the EWT decomposition."},
    )
    ewt_filter_length: int = field(
        default=10,
        metadata={"help": "The length of the filter to use for the EWT decomposition."},
    )
    ewt_filter_sigma: float = field(
        default=2.0,
        metadata={"help": "The sigma of the filter to use for the EWT decomposition, if filter is gaussian."},
    )
    ewt_log_spectrum: bool = field(
        default=False,
        metadata={"help": "Whether to work with the log spectrum for the EWT decomposition."},
    )
    ewt_detect: bool = field(
        default="locmax",
        metadata={"help": "Method used for peak detection in the EWT decomposition."},
    )
