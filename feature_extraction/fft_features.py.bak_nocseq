import torch
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def extract_fft_psd(batch, normalize=True, to_db = True, device=None, n_fft = None, overlap=0.1, 
                    window_type='hann', scaling='density'):
    """
    Apply Fast Fourier Transform to audio batch data
    
    Args:
        batch (dict): Batch dictionary containing input_values and input_seq_values
        normalize (bool): Whether to normalize the FFT output
        to_db (bool): Whether to convert the FFT output to decibel scale
        device: The device to place tensors on
        n_fft (int or None): Number of FFT points. If None, it will be set based on input length.
        overlap (float): Fraction of overlap between segments for Welch's method
        window_type (str): Type of window to use in Welch's method
        scaling (str): Scaling method for Welch's method ('density' or 'spectrum')
        
    Returns:
        tuple: (processed_input_values, processed_input_seq_values)
    """
    # Get shapes
    batch_size = batch["input_values"].shape[0]
    if len(batch["input_values"].shape) < 3:
        frame_len = batch["input_values"].shape[-1]
        num_components = 1
    else: 
        num_components = batch["input_values"].shape[1]
        frame_len = batch["input_values"].shape[-1]
    
    if n_fft is None:
        n_fft = 2*frame_len - 1
    
    # Initialize output tensors
    fft_features_values = torch.zeros_like(batch["input_values"], dtype=torch.float32, device=device)
    
    # Process input_values
    for b in range(batch_size):
        for c in range(num_components):
            # Convert to CPU for FFT processing if needed
            if len(batch["input_values"].shape) < 3:
                # Single component case
                signal_data = batch["input_values"][b].cpu().numpy()
            else:
                signal_data = batch["input_values"][b, c].cpu().numpy()
            
            # Apply Welch's method for PSD estimation
            # Calculate window and number of segments
            nperseg = min(n_fft, signal_data.shape[-1])
            noverlap = int(nperseg * overlap)
            
            # Calculate PSD using Welch's method
            f, spec = signal.welch(
                signal_data,
                fs=1.0,  # Normalized frequency
                window=window_type,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=n_fft,
                detrend='constant',
                return_onesided=True,
                scaling=scaling
            )

            if to_db:
                spec = 10 * np.log10(spec + 1e-10)  
            
            if normalize:
                spec = (spec - spec.mean()) / (spec.std() + 1e-6)

            if len(batch["input_values"].shape) < 3:
                fft_features_values[b] = torch.tensor(spec, device=device or batch["input_values"].device)
            else:
                fft_features_values[b, c] = torch.tensor(spec, device=device or batch["input_values"].device)

    # Process input_seq_values if available
    if "input_seq_values" in batch and batch["input_seq_values"] is not None:
        if len(batch["input_seq_values"].shape) == 4:  # If already framed
            frames_per_seq = batch["input_seq_values"].shape[2]
            fft_features_seq_values = torch.zeros_like(batch["input_seq_values"], 
                                                     dtype=torch.float32, 
                                                     device=device or batch["input_seq_values"].device)

            # Process each frame
            for b in range(batch_size):
                for c in range(num_components):
                    for f in range(frames_per_seq):
                        # Get signal for this frame
                        signal_data = batch["input_seq_values"][b, c, f].cpu().numpy()
                        
                        # Apply Welch's method for PSD estimation
                        # Calculate window and number of segments
                        nperseg = min(n_fft, len(signal_data))
                        noverlap = int(nperseg * overlap)
                        
                        # Calculate PSD using Welch's method
                        f, spec = signal.welch(
                            signal_data,
                            fs=1.0,  # Normalized frequency
                            window=window_type,
                            nperseg=nperseg,
                            noverlap=noverlap,
                            nfft=n_fft,
                            detrend='constant',
                            return_onesided=True,
                            scaling=scaling
                        )

                        if to_db:
                            spec = 10 * np.log10(spec + 1e-10)  

                        if normalize:
                            spec = (spec - spec.mean()) / (spec.std() + 1e-6)
                            
                        # Store result
                        fft_features_seq_values[b, c, f] = torch.tensor(spec, device=device or batch["input_seq_values"].device)

        else:
            # Handle unframed sequence values
            fft_features_seq_values = torch.zeros_like(batch["input_seq_values"], 
                                                     dtype=torch.float32, 
                                                     device=device or batch["input_seq_values"].device)
            
            #Use different nfft for sequence - To get same shape as initial input values
            n_fft = 2* batch["input_seq_values"].shape[-1] - 1

            for b in range(batch_size):
                for c in range(num_components):
                    signal_data = batch["input_seq_values"][b, c].cpu().numpy()
                    
                    # Welch's method implementation (same as above)
                    nperseg = min(n_fft, frame_len)
                    noverlap = int(nperseg * overlap)
                    
                    f, spec = signal.welch(
                        signal_data,
                        fs=1.0,
                        window=window_type,
                        nperseg=nperseg,
                        noverlap=noverlap,
                        nfft=n_fft,
                        detrend='constant',
                        return_onesided=True,
                        scaling=scaling
                    )
                    
                    if to_db:
                        spec = 10 * np.log10(spec + 1e-10)  # Convert to dB scale

                    if normalize:
                        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
                    
                    fft_features_seq_values[b, c] = torch.tensor(spec, device=device or batch["input_seq_values"].device)

    else:
        fft_features_seq_values = None
            
    return fft_features_values, fft_features_seq_values