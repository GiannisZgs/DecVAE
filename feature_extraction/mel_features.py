import librosa
import numpy as np
import torch
import scipy.signal as signal
import warnings

warnings.filterwarnings("ignore", message="n_fft=.* is too large for input signal of length=.*", 
                           category=UserWarning, module="librosa.core.spectrum")

def extract_mel_spectrogram(audio, sample_rate, n_mels=128, n_fft=512, hop_length=512, normalize = 'global', feature_length = 400,ref=None):
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
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    else:
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=ref)
    "Normalize the mel spectrogram"
    if normalize is None:
        mel_spectrogram_db = torch.from_numpy(mel_spectrogram_db).reshape(1,mel_spectrogram_db.shape[0],-1).to('cuda' if was_cuda else 'cpu')
        return mel_spectrogram_db[..., :feature_length], spec_max
    else:
        if normalize == 'per_feature':
            # Normalize each frequency band independently
            orig_shape = mel_spectrogram_db.shape
            mel_spectrogram_db = torch.from_numpy(mel_spectrogram_db).transpose(-1,-2).view(-1,orig_shape[-2]).numpy()#reshape(-1,orig_shape[-2])
            mean = np.mean(mel_spectrogram_db, axis=0, keepdims=True)
            std = np.std(mel_spectrogram_db, axis=0, keepdims=True) + 1e-9
            mel_spectrogram_db = (mel_spectrogram_db - mean.squeeze()) / std.squeeze()
            mel_spectrogram_db = mel_spectrogram_db.reshape(orig_shape)

        elif normalize == 'global':
            # Global standardization
            mean = np.mean(mel_spectrogram_db)
            std = np.std(mel_spectrogram_db) + 1e-9
            mel_spectrogram_db = (mel_spectrogram_db - mean) / std
        
        elif normalize == 'minmax':
            # Global min-max scaling to [-1, 1]
            min_val = np.min(mel_spectrogram_db)
            max_val = np.max(mel_spectrogram_db)
            mel_spectrogram_db = 2 * (mel_spectrogram_db - min_val) / (max_val - min_val + 1e-9) - 1
        elif normalize == 'minmax0_1':
            # Global min-max scaling to [0, 1]
            min_val = np.min(mel_spectrogram_db)
            max_val = np.max(mel_spectrogram_db)
            mel_spectrogram_db = (mel_spectrogram_db - min_val) / (max_val - min_val + 1e-9)

        "Normalize and concatenate over time axis"
        if len(mel_spectrogram_db.shape) == 2 and (mel_spectrogram_db.shape[0] != n_mels and mel_spectrogram_db.shape[1] != feature_length//hop_length):
            mel_spectrogram_db = torch.from_numpy(mel_spectrogram_db).to('cuda' if was_cuda else 'cpu')
            return mel_spectrogram_db, spec_max
        elif len(mel_spectrogram_db.shape) == 2:
            return mel_spectrogram_db, spec_max #mel_spectrogram_db.shape[0]*mel_spectrogram_db.shape[1])
        else:
            mel_spectrogram_db = torch.from_numpy(mel_spectrogram_db).reshape(mel_spectrogram_db.shape[0],mel_spectrogram_db.shape[1],-1).to('cuda' if was_cuda else 'cpu')
            mel_spectrogram_db_flat = mel_spectrogram_db.reshape(mel_spectrogram_db.shape[0],-1)
            mel_spectrogram_db = (mel_spectrogram_db - torch.mean(mel_spectrogram_db_flat,axis=-1)[:,None,None]) / (torch.std(mel_spectrogram_db_flat,axis=-1) + 1e-9)[:,None,None]
            return mel_spectrogram_db[..., :feature_length], spec_max

def extract_log_magnitude_spectrum(audio, sample_rate, n_fft=1024, hop_length=512):
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
        
    D = librosa.stft(audio.squeeze(), n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D) ** 2
    log_magnitude_spectrum = librosa.power_to_db(magnitude, ref=np.max)
    return torch.from_numpy(log_magnitude_spectrum)