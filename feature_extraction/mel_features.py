import librosa
import numpy as np
import torch
import scipy.signal as signal
import warnings

warnings.filterwarnings("ignore", message="n_fft=.* is too large for input signal of length=.*",
                           category=UserWarning, module="librosa.core.spectrum")


def rereference_mel_db(mel_spectrogram_db, spec_max, ref=None, top_db=80.0):
    """
    Re-reference a dB-scaled mel spectrogram from its original per-utterance reference to a
    common one: db_new = db_old + 10*log10(spec_max) - 10*log10(ref). Requires that the original
    conversion was done with top_db=None, since the clamp is relative to the reference.

    Args:
        mel_spectrogram_db (torch.Tensor or np.ndarray): dB-scaled mel spectrogram, with leading
            dimensions matching spec_max e.g. (batch, components, n_mels, time)
        spec_max (torch.Tensor or np.ndarray): Reference values of the original conversion
        ref (float or None): Reference to convert to. If None, uses the max over spec_max.
        top_db (float or None): Dynamic range clamp, as in librosa.power_to_db
    Returns:
        mel_spectrogram_db (torch.Tensor): Mel spectrogram referenced against ref
    """
    if isinstance(mel_spectrogram_db, np.ndarray):
        mel_spectrogram_db = torch.from_numpy(mel_spectrogram_db)
    if isinstance(spec_max, np.ndarray):
        spec_max = torch.from_numpy(spec_max)
    elif not isinstance(spec_max, torch.Tensor):
        spec_max = torch.as_tensor(spec_max)
    spec_max = spec_max.to(device=mel_spectrogram_db.device, dtype=mel_spectrogram_db.dtype)

    if ref is None:
        ref = spec_max.max()
    ref = torch.as_tensor(ref, device=mel_spectrogram_db.device, dtype=mel_spectrogram_db.dtype)

    offset = 10.0*torch.log10(spec_max.clamp_min(1e-10)) - 10.0*torch.log10(ref.clamp_min(1e-10))
    while offset.dim() < mel_spectrogram_db.dim():
        offset = offset.unsqueeze(-1)
    mel_spectrogram_db = mel_spectrogram_db + offset

    if top_db is not None:
        mel_spectrogram_db = torch.maximum(mel_spectrogram_db, mel_spectrogram_db.max() - top_db)
    return mel_spectrogram_db


def normalize_mel_spectrogram(mel_spectrogram_db, normalize='global', feature_length=400, device=None, n_mels=None):
    """
    Normalize a dB-scaled mel spectrogram. Shared by extract_mel_spectrogram and by the collators
    that receive features already extracted at preprocessing time. The selected mode is followed
    by a per-example standardization, as in the original implementation.

    Args:
        mel_spectrogram_db (torch.Tensor or np.ndarray): Mel spectrogram in dB scale
        normalize (str or None): Normalization method ('per_feature', 'global', 'minmax',
            'minmax0_1', or None)
        feature_length (int): Desired length of the feature in time frames
        device (str, torch.device or None): Device of the returned tensor. Defaults to the
            device of the input.
        n_mels (int or None): Number of Mel bands. Only needed by 'per_feature' when the mel and
            time axes arrive already flattened into the last axis, as they do from preprocessing.
    Returns:
        mel_spectrogram_db (torch.Tensor): Normalized mel spectrogram
    """
    if isinstance(mel_spectrogram_db, np.ndarray):
        mel_spectrogram_db = torch.from_numpy(mel_spectrogram_db)
    if device is None:
        device = mel_spectrogram_db.device
    mel_spectrogram_db = mel_spectrogram_db.to(device)

    if normalize is None:
        "Concatenate over time axis - the leading axes are kept, or callers broadcast one frame over all of them"
        if mel_spectrogram_db.dim() == 2:
            mel_spectrogram_db = mel_spectrogram_db.reshape(1, mel_spectrogram_db.shape[0], -1)
        else:
            mel_spectrogram_db = mel_spectrogram_db.reshape(
                mel_spectrogram_db.shape[0], mel_spectrogram_db.shape[1], -1
            )
        return mel_spectrogram_db[..., :feature_length]

    if normalize == 'per_feature':
        # Normalize each frequency band independently
        orig_shape = mel_spectrogram_db.shape
        # Split the mel axis back out if it arrived flattened together with time
        was_flattened = n_mels is not None and orig_shape[-1] % n_mels == 0
        if was_flattened:
            mel_spectrogram_db = mel_spectrogram_db.reshape(*orig_shape[:-1], n_mels, -1)
        band_shape = mel_spectrogram_db.shape

        flat = mel_spectrogram_db.transpose(-1, -2).reshape(-1, band_shape[-2])
        mean = flat.mean(dim=0, keepdim=True)
        std = flat.std(dim=0, unbiased=False, keepdim=True) + 1e-9
        flat = (flat - mean) / std
        mel_spectrogram_db = flat.reshape(
            *band_shape[:-2], band_shape[-1], band_shape[-2]
        ).transpose(-1, -2)
        if was_flattened:
            mel_spectrogram_db = mel_spectrogram_db.reshape(orig_shape)

    elif normalize == 'global':
        # Global standardization
        mean = mel_spectrogram_db.mean()
        std = mel_spectrogram_db.std(unbiased=False) + 1e-9
        mel_spectrogram_db = (mel_spectrogram_db - mean) / std

    elif normalize == 'minmax':
        # Global min-max scaling to [-1, 1]
        min_val = mel_spectrogram_db.min()
        max_val = mel_spectrogram_db.max()
        mel_spectrogram_db = 2 * (mel_spectrogram_db - min_val) / (max_val - min_val + 1e-9) - 1

    elif normalize == 'minmax0_1':
        # Global min-max scaling to [0, 1]
        min_val = mel_spectrogram_db.min()
        max_val = mel_spectrogram_db.max()
        mel_spectrogram_db = (mel_spectrogram_db - min_val) / (max_val - min_val + 1e-9)

    "Normalize and concatenate over time axis"
    if mel_spectrogram_db.dim() == 2:
        return mel_spectrogram_db
    else:
        mel_spectrogram_db = mel_spectrogram_db.reshape(mel_spectrogram_db.shape[0], mel_spectrogram_db.shape[1], -1)
        mel_spectrogram_db_flat = mel_spectrogram_db.reshape(mel_spectrogram_db.shape[0], -1)
        mel_spectrogram_db = (mel_spectrogram_db - torch.mean(mel_spectrogram_db_flat, axis=-1)[:, None, None]) / (torch.std(mel_spectrogram_db_flat, axis=-1) + 1e-9)[:, None, None]
        return mel_spectrogram_db[..., :feature_length]


def extract_mel_spectrogram(audio, sample_rate, n_mels=128, n_fft=512, hop_length=512, normalize = 'global', feature_length = 400,ref=None, top_db=80.0):
    """
    Extract Mel Spectrogram from audio signal
    Args:
        audio (np.ndarray or torch.Tensor): Input batched audio
        sample_rate (int): Sampling rate of the audio
        n_mels (int): Number of Mel bands
        n_fft (int): Length of the FFT window
        hop_length (int): Number of samples between successive frames
        normalize (str or None): Normalization method of the current input batch('per_feature', 'global', 'minmax', 'minmax0_1', or None)
        feature_length (int): Desired length of the feature in time frames
        ref (float or None): Reference value for dB conversion. If None, uses max value.
        top_db (float or None): Dynamic range clamp passed to ``librosa.power_to_db``. Pass None
            to skip clamping, which is required when the dB values are going to be re-referenced
            later against a different peak (see ``rereference_mel_db``).
    Returns:
        mel_spectrogram_db (torch.Tensor): Mel spectrogram in dB scale
        spec_max (float): Maximum value of the Mel spectrogram before normalization
    """

    # Convert torch tensor to numpy if needed
    if isinstance(audio, torch.Tensor):
        was_cuda = audio.is_cuda
        audio = audio.cpu().numpy() if was_cuda else audio.numpy()
    else:
        was_cuda = False
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio.squeeze() if not audio.shape[0] == 1 else audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=None,
        window=signal.windows.hann
    )
    spec_max = np.max(mel_spectrogram)
    if ref is None:
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=top_db)
    else:
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=ref, top_db=top_db)

    "Normalize the mel spectrogram"
    mel_spectrogram_db = normalize_mel_spectrogram(
        mel_spectrogram_db,
        normalize=normalize,
        feature_length=feature_length,
        device='cuda' if was_cuda else 'cpu',
    )
    return mel_spectrogram_db, spec_max

def extract_log_magnitude_spectrum(audio, sample_rate, n_fft=1024, hop_length=512):
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()

    D = librosa.stft(audio.squeeze(), n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D) ** 2
    log_magnitude_spectrum = librosa.power_to_db(magnitude, ref=np.max)
    return torch.from_numpy(log_magnitude_spectrum)
