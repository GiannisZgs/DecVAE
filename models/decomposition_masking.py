# coding=utf-8
# Copyright 2025 Ioannis Ziogas <ziogioan@ieee.org>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Decomposition Model D_w^C that precedes a 🤗 DecVAE - Variational Decomposition Autoencoder Model. 
Performs Decomposition Masking on the normalized audio signal x in the frame or/and the sequence level. The framing is performed with a window size of RFS
and a stride of stride. Frames and sequences are "masked" by decomposition where the detected components represent
masked versions of the original waveforms.

Can be used as a preprocessing step (offline) or during training (online)*.

Applies different decomposition methods (emd, vmd, ewt, filter) to extract oscillatory components (OCs) from the signal.
These methods aim to capture different frequency bands and temporal dynamics present in the signal.
Filter decomposition is a custom-made decomposition that uses bandpass IIR filters to extract OCs based on detected spectral peaks.
It can also calculate time-domain quality metrics for the decomposition such as correlogram between components and NRMSE between original and reconstructed signal.
"""

import scipy.signal
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy
from typing import Optional, Union
import ewtpy
from PyEMD import EMD
from sktime.libs.vmdpy import VMD


class CustomLayerNorm(nn.Module):
    """
    Custom Layer Normalization layer. Computes statistics only over masked time indices. If mask time indices are all True,
    then behaves like standard LayerNorm. Was designed to combine the DecVAE functionality with Wav2vec2 contrastive and divergence losses.
    """
    def __init__(self, normalized_shape, config, elementwise_affine=True,bias = True,device='cpu',dtype=torch.float32):
        super(CustomLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = config.layer_norm_eps #1e-5
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape,device=device,dtype=dtype))
            if bias:
                self.bias = nn.Parameter(torch.zeros(normalized_shape,device=device,dtype=dtype))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def get_batch_stats(self,x,mask_time_indices):
        #Get batch statistics only for masked frames of the input (mask_time_indices == True)
        #x is of shape (B,S,L,C): B - batch size, S - number of segments, L - sequence length, C - number of channels
        #S*L = N, N the length of the utterance
        #we want to find the mean across the last dimension - channels

        if x.ndim == 4:
            masked_x = x.detach().clone().masked_select(mask_time_indices.unsqueeze(-1).unsqueeze(-1)).view(-1, x.shape[-1])
        elif x.ndim == 3:
            masked_x = x.detach().clone().masked_select(mask_time_indices.unsqueeze(-1)).view(-1, x.shape[-1])    
        # Compute mean and variance along the channels dimension
        mean = masked_x.mean(dim=0)
        variance = masked_x.var(dim=0, unbiased=False)

        return mean, variance

    def forward(self, x, mask_time_indices):
        #x is of shape (B,S,L,C): B - batch size, S - number of segments, L - sequence length, C - number of channels
        #S*L = N, N the length of the utterance

        # Compute mean and variance only across the masked segments
        mean, variance = self.get_batch_stats(x, mask_time_indices)
        # Normalize the input
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
    
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + (self.bias if self.bias is not None else 0)
        
        return x_normalized 

class CustomBatchNorm(nn.Module):
    """
    Custom Batch Normalization layer. Computes statistics only over masked time indices. If mask time indices are all True,
    then behaves like standard BatchNorm. Was designed to combine the DecVAE functionality with Wav2vec2 contrastive and divergence losses.
    """

    def __init__(self, normalized_shape, config, affine=True, bias = True, momentum = 0.1,track_running_stats = True, device='cpu',dtype=torch.float32):
        super(CustomBatchNorm, self).__init__()
        self.normalized_shape = normalized_shape #number of output conv channels
        self.eps = config.layer_norm_eps #1e-5
        self.affine = affine
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape,device=device,dtype=dtype))
            if bias:
                self.bias = nn.Parameter(torch.zeros(normalized_shape,device=device,dtype=dtype))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def get_batch_stats(self,x,mask_time_indices):
        #Get batch statistics only for masked frames of the input (mask_time_indices == True)
        #x is of shape (B,S,L,C): B - batch size, S - number of segments, L - sequence length, C - number of channels
        #S*L = N, N the length of the utterance
        #we want to find the mean across the last dimension - channels

        if x.ndim == 4:
            masked_x = x.detach().clone().masked_select(mask_time_indices.unsqueeze(-1).unsqueeze(-1)).view(-1, x.shape[-1])
        elif x.ndim == 3:
            masked_x = x.detach().clone().masked_select(mask_time_indices.unsqueeze(-1)).view(-1, x.shape[-1])    
        # Compute mean and variance along the channels dimension
        mean_current = masked_x.mean(dim=0)
        variance_current = masked_x.var(dim=0, unbiased=False)

        #track running stats
        if self.track_running_stats:
            if not hasattr(self, 'running_mean') and not hasattr(self, 'running_var'):
                self.running_mean = mean_current
                self.running_var = variance_current
            else:
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean_current
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * variance_current

        return mean_current, variance_current

    def forward(self, x, mask_time_indices):
        #x is of shape (B,S,L,C): B - batch size, S - number of segments, L - sequence length, C - number of channels
        #S*L = N, N the length of the utterance

        # Compute mean and variance only across the masked segments
        mean, variance = self.get_batch_stats(x, mask_time_indices)
        # Normalize the input
        if self.track_running_stats and hasattr(self, 'running_mean') and hasattr(self, 'running_var'):
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        else:
            #If track_running_stats is False, or first pass
            x_normalized = (x - mean) / torch.sqrt(variance + self.eps)

        if self.affine:
            x_normalized = x_normalized * self.weight + (self.bias if self.bias is not None else 0)

        return x_normalized 

class DecompositionModule(nn.Module):
    """The Decomposition Model D_w^C. 
    Performs Decomposition Masking on the normalized audio signal x in the frame or/and the sequence level. The framing is performed with a window size of RFS
	and a stride of stride. Frames and sequences are "masked" by decomposition where the detected components represent
	masked versions of the original waveforms.

	Can be used as a preprocessing step (offline) or during training (online)*.

    Applies different decomposition methods (emd, vmd, ewt, filter) to extract oscillatory components (OCs) from the signal.
    These methods aim to capture different frequency bands and temporal dynamics present in the signal.
    Filter decomposition is a custom-made decomposition that uses bandpass IIR filters to extract OCs based on detected spectral peaks.
    It can also calculate time-domain quality metrics for the decomposition such as correlogram between components and NRMSE between original and reconstructed signal.
    
    Args:
    config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): The DecVAE configuration object containing parameters for decomposition and masking.

        General and Filter Decomposition parameters:

        - dataset_name: Name of the dataset.
        - freq_groups: Predefined frequency groups that are used to help classify unidentified OCs.
        - fs: Sampling frequency of the signal.
        - decomp_to_perform: Decomposition method to use ('emd', 'vmd', 'ewt', 'filter').
        - frame_decomp: Whether to perform frame-level decomposition.
        - seq_decomp: Whether to perform sequence-level decomposition.
        - RFS: Receptive field size for framing.
        - stride: Stride for framing.
        - lower_speech_freq: an assumed lower frequency bound for speech. Frequencies lower than this will not be detected as OCs.
        - higher_speech_freq: an assumed upper frequency bound for speech. Frequencies higher than this will not be detected as OCs.
        - max_silence_freq: an assumed maximum frequency bound for silence. Frequencies below this can be considered as silence if remove_silence = True.
        - notch_band_low: Lower frequency of the notch band to remove power line noise.
        - notch_band_high: Upper frequency of the notch band to remove power line noise.
        - use_notch_filter: Whether to apply a notch filter to remove power line noise.
        - nfft: The number of FFT points to use for the decomposition and other spectral operations e.g. Welch PSD.
        - NoC: Number of components to extract at the frame level.
        - NoC_seq: Number of components to extract at the sequence level.
        - group_OCs_by_frame: Strategy to group frame-level OCs ('high_freqs_first', 'low_freqs_first', 'equally_distribute').
        - group_OCs_by_seq: Strategy to group sequence-level OCs ('high_freqs_first', 'low_freqs_first', 'equally_distribute','top_k').
        - conv_kernel: Kernel sizes for the convolutional layers in the DecVAE encoder. Used to get a correct attention mask if needed.
        - conv_stride: Stride sizes for the convolutional layers in the DecVAE encoder. Used to get a correct attention mask if needed.
        - power_law: Exponent for automatic spacing of peak detection intervals based on a power-law.
        - detection_intervals: Number of intervals to split the frequency spectrum into for peak detection.
        - spec_amp_tolerance: Tolerance for spectral amplitudes of found OCs. If an OC has normalized spectral amplitude lesser than this threshold, 
            it will be discarded.
        - spec_amp_tolerance_seq: Tolerance for spectral amplitudes of found OCs at the sequence level. If an OC has normalized spectral amplitude lesser than this threshold,
            it will be discarded.
        - global_thres: Global threshold for peak detection. Peaks with normalized amplitude (relative to the highest peak) below this threshold will be ignored.
        - min_distance: Minimum allowed frequency distance between detected peaks.
        - peak_bandwidth: Bandwidth around each detected peak to consider as part of the OC.
        - prom_thres: Prominence threshold for peak detection. Only peaks with prominence above this fraction are considered.
        - N_peaks_to_select: Number of top peaks to select in each detection interval.
        - buttord: Order of the Butterworth filter used in filter decomposition.
    
        EMD-specific parameters - For a documentation of these parameters check https://github.com/laszukdawid/PyEMD.

        - emd_spline_kind: Interpolation method to connect extrema in EMD.
        - emd_max_iter: Maximum number of iterations per EMD sifting.
        - emd_energy_ratio_thr: Energy ratio threshold, per IMF.
        - emd_std_thr: Standard deviation threshold, per IMF.
        - emd_svar_thr: Scaled variance threshold, per IMF.
        - emd_total_power_thr: Total power threshold, per EMD.
        - emd_range_thr: Amplitude range threshold after scaling, per EMD.
        - emd_extrema_detection: Method for extrema detection in EMD.

        VMD-specific parameters - For a documentation of these parameters check sktime/libs/vmdpy/vmdpy.py
        
        - vmd_alpha: Balancing parameter of the data-fidelity constraint.
        - vmd_tau: Time-step of the dual ascent (pick 0 for noise-slack).
        - vmd_DC: Whether to keep the first mode at DC (0 frequency).
        - vmd_init: Initialization method for mode center-frequencies.
        - vmd_tol: Tolerance of convergence criterion.
        - use_vmd_correction: Whether to apply frequency correction after VMD decomposition. This can help improve detection accuracy by accounting for 
            impossible detections due to the initialization of frequencies in VMD. It significantly increases the computation time, however.
        
        EWT-specific parameters - For a documentation of these parameters check ewtpy/ewtpy.py/EWT1D.
        - ewt_completion: Whether to use spectrum completion in EWT.
        - ewt_filter: Type of filter to use for regularizing the spectrum in EWT.
        - ewt_filter_length: width of the filter used in EWT.
        - ewt_filter_sigma: standard deviation for the custom filter in EWT.
        - ewt_log_spectrum: Whether to use logarithmic spectrum in EWT.
        - ewt_detect: Method for boundary detection in EWT.


    *online use not yet supported in the current implementation.
	"""

    def __init__(self,
        config,
    ):
        super(DecompositionModule, self).__init__()
        if hasattr(config, 'dataset_name'):
            self.dataset_name = config.dataset_name
        else:
            self.dataset_name = None
        self.freq_groups = config.freq_groups
        self.fs = config.fs
        self.decomp_to_perform = config.decomp_to_perform
        self.frame_decomp = config.frame_decomp
        self.seq_decomp = config.seq_decomp
        self.RFS = config.receptive_field
        self.stride = config.stride
        self.lower_speech_freq = config.lower_speech_freq
        self.higher_speech_freq = config.higher_speech_freq
        self.max_silence_freq = config.max_silence_freq
        self.notch_band = [config.notch_band_low,config.notch_band_high]
        self.use_notch_filter = config.use_notch_filter
        self.nfft = config.nfft
        self.NoC = config.NoC
        self.NoC_seq = config.NoC_seq
        self.group_OCs_by_frame = config.group_OCs_by_frame
        self.group_OCs_by_seq = config.group_OCs_by_seq
        self.conv_kernel = config.conv_kernel
        self.conv_stride = config.conv_stride
        self.power_law = config.power_law
        self.N = config.detection_intervals
        self.spec_amp_tolerance = config.spec_amp_tolerance
        self.spec_amp_tolerance_seq = config.spec_amp_tolerance_seq
        self.global_thres = config.global_thres
        self.min_distance = config.min_distance
        self.peak_bandwidth = config.peak_bandwidth
        self.prom_thres = config.prom_thres
        self.N_peaks_to_select = config.N_peaks_to_select
        self.buttord = config.buttord
        if self.decomp_to_perform == 'emd':
            self.emd_spline_kind = config.emd_spline_kind
            self.emd_max_iter = config.emd_max_iter
            self.emd_energy_ratio_thr = config.emd_energy_ratio_thr
            self.emd_std_thr = config.emd_std_thr 
            self.emd_svar_thr = config.emd_svar_thr
            self.emd_total_power_thr = config.emd_total_power_thr
            self.emd_range_thr = config.emd_range_thr
            self.emd_extrema_detection = config.emd_extrema_detection 
            self.emd = EMD(DTYPE=np.float32,spline_kind=self.emd_spline_kind,MAX_ITERATION=self.emd_max_iter,energy_ratio_thr=self.emd_energy_ratio_thr,std_thr=self.emd_std_thr,svar_thr=self.emd_svar_thr,total_power_thr=self.emd_total_power_thr,range_thr=self.emd_range_thr,extrema_detection=self.emd_extrema_detection)

        elif self.decomp_to_perform == 'vmd':
            self.vmd_alpha = config.vmd_alpha
            self.vmd_tau = config.vmd_tau
            self.vmd_DC = config.vmd_DC
            self.vmd_init = config.vmd_init
            self.vmd_tol = config.vmd_tol
            self.use_vmd_correction = config.use_vmd_correction

        elif self.decomp_to_perform == 'ewt':
            self.ewt_completion = config.ewt_completion
            self.ewt_filter = config.ewt_filter
            self.ewt_filter_length = config.ewt_filter_length
            self.ewt_filter_sigma = config.ewt_filter_sigma
            self.ewt_log_spectrum = config.ewt_log_spectrum
            self.ewt_detect = config.ewt_detect

    def notch_filter_power_line_noise(self,x):
        #Notch filter to cut-off power line noise @62Hz and its harmonics
        sos = scipy.signal.butter(4,self.notch_band,'bandstop',fs=self.fs,output='sos')
        if x.ndim == 2:
            #In case a batch is given, make sure that the filter is applied to each channel
            if x.shape[1] > x.shape[0]:
                x_filt = scipy.signal.sosfiltfilt(sos,x,axis = -1)
            else:
                x_filt = scipy.signal.sosfiltfilt(sos,x,axis = 0)
        else:
            x_filt = scipy.signal.sosfiltfilt(sos,x)

        return x_filt

    def get_peak_detection_intervals(self):
        #Construct power-law intervals for peak detection
        v = np.linspace(1,self.N+1,self.N+1)
        y = v**self.power_law
        assert self.higher_speech_freq < self.fs/2
        y_scaled = np.interp(y, (y.min(), y.max()), (self.lower_speech_freq, self.higher_speech_freq))
        peak_det_intervals = [[y_scaled[i],y_scaled[i+1]] for i in range(len(y_scaled)-1)]

        return peak_det_intervals

    def classify_omegas(self,omegas,other_freq_groups = None):

        real_omegas = omegas[omegas != -1]
        zero_omegas = omegas[omegas == -1]

        point_intervals = np.full(real_omegas.shape, -1)
        if other_freq_groups is None:
            if not type(self.freq_groups) == np.ndarray:    
                self.freq_groups = np.array(self.freq_groups)
            # Efficient vectorized approach to classify points
            starts = self.freq_groups[:, 0][:, np.newaxis]  # Shape (N, 1)
            ends = self.freq_groups[:, 1][:, np.newaxis]    # Shape (N, 1)
        else:
            if not type(other_freq_groups) == np.ndarray:    
                other_freq_groups = np.array(other_freq_groups)
            starts = other_freq_groups[:, 0][:, np.newaxis]
            ends = other_freq_groups[:, 1][:, np.newaxis]
        points_expanded = real_omegas[np.newaxis, :]  # Shape (1, M)

        # Create a boolean matrix indicating whether each point falls within each interval
        in_intervals = (points_expanded >= starts) & (points_expanded < ends)  # Shape (N, M)

        # For each point, find the intervals where the condition is True
        # sum along intervals axis to check if point is in any interval
        is_in_any_interval = np.any(in_intervals, axis=0)

        # For points within any interval, find the index of the interval
        point_indices = np.argmax(in_intervals, axis=0)

        # Assign interval indices to points that are within intervals
        point_intervals[is_in_any_interval] = point_indices[is_in_any_interval]

        if zero_omegas.size > 0:
            zero_omegas.dtype = point_intervals.dtype
            point_intervals = np.concatenate((zero_omegas,point_intervals))
        
        return np.array(point_intervals)

    def silence_check(self,frame):
        "Empirical silence check based on amplitude and spectral peak frequency"

        #Silence check - Amplitude and spectral peaks 
        spec, freq_ax = mlab.psd(frame,NFFT=self.nfft,Fs=self.fs)
        idx = np.argmax(spec,)
        peak_freq = freq_ax[idx]
        silence_condition_1 = (np.median(frame) < 0.05 and np.std(frame) < 0.07)
        silence_condition_2 = (peak_freq <= self.max_silence_freq) #Max silence freq considered 250Hz
        silence_condition_3 = (np.median(frame) < 0.01 and np.std(frame) < 0.01)        
        if silence_condition_1 and silence_condition_2: 
            #Frame is silent
            return True
        elif silence_condition_3 and not silence_condition_2:
            #Frame is noise - silent
            return True
        else:
            return False
        
    def visualize_mask(self, x, array_length, start_indices, masked_indices):
        "Utility to visualize masked regions of the input signal x"

        mask_length = int(self.RFS*self.fs) 
        array_length = start_indices[-1] - mask_length
        offset = abs(np.max(x))
        array = np.zeros(array_length)
        
        # Apply the mask
        for start in masked_indices:
            end = min(start + mask_length, array_length)  # Ensure we don't go beyond the array
            array[start:end] = offset  # Apply mask
        
        # Plotting
        plt.figure(figsize=(10, 2))
        plt.plot(array, drawstyle='steps-post')
        #plt.ylim(-0.5, 1.5)  # Adjust y-axis to make the mask more visible
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Visualization of Masking')
        #plt.grid(True)
        plt.plot(x)
        plt.show()

        return None
    
    def peak_detection(self,x,peak_det_intervals):
        """
        Peak detection in the normalized spectrum of the signal x.
        Performs peak detection inside each of the given peak detection intervals. 
        If needed, it performs a peak clustering step to merge close peaks.
        It also merges peaks that are too close to each other across intervals.
        """

        #Calculate PSD of the signal and normalize the spectrum
        freq_ax, spec = scipy.signal.welch(x-np.mean(x), fs=self.fs, window='hann', nperseg=132, noverlap=10, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
        global_spec_max = np.max(spec)
        if global_spec_max == 0 or global_spec_max is None:
            pass
        spec = spec/global_spec_max
        df = freq_ax[1]-freq_ax[0] # frequency domain resolution ((fs/2)/len(spec))
        peak_boundaries = peak_det_intervals/df
        peak_boundaries = np.array([np.ceil(point) for point in peak_boundaries]).astype(int)            

        #Iterate and refine detected peaks
        peaks_low = []
        peaks_high = []
        freqs_low = []
        freqs_high = []
        final_peaks = []
        final_peak_vals = []
        for i,interval in enumerate(peak_boundaries):
            samples_distance = round(self.min_distance/df)
            spec_chunk = spec[interval[0]:interval[1]]

            #Check level of local peaks - If below global_threshold, then discard
            if np.max(spec_chunk) < self.global_thres:
                continue
            local_peaks,_ = scipy.signal.find_peaks(spec_chunk,distance = samples_distance)
            if not local_peaks.any():
                continue
            local_proms = scipy.signal.peak_prominences(spec_chunk,local_peaks)[0]
            #Local prominence threshold criterion
            max_prom = np.max(local_proms)			
            local_peaks = local_peaks[local_proms >= self.prom_thres*max_prom]

            #Find top peaks amongst remaining peaks
            try:
                top_k_peaks = local_peaks[np.argsort(local_proms)[-self.N_peaks_to_select:]]
            except IndexError:
                top_k_peaks = local_peaks.copy()

            top_k_peaks.sort()

            peakdiff = np.diff(top_k_peaks)*df
            merge_mask = peakdiff < self.peak_bandwidth/2 + samples_distance*df
            for j,fpeak in enumerate(top_k_peaks):
                peak_low_bound = int(fpeak - np.floor(self.peak_bandwidth/2/df))
                if peak_low_bound < 1: peak_low_bound = 0 #and i == 0
                peak_high_bound = int(fpeak + np.floor(self.peak_bandwidth/2/df))
                if peak_high_bound > len(spec_chunk) and i == len(peak_det_intervals)-1: 
                    peak_high_bound = len(spec_chunk)
                #Convert to initial spec indices
                peak_low_bound += interval[0]
                peak_high_bound += interval[0]
                exact_peak_freq = freq_ax[peak_low_bound:peak_high_bound+1][np.argmax(spec[peak_low_bound:peak_high_bound+1])]
                final_peaks.append(exact_peak_freq)
                final_peak_vals.append(spec[peak_low_bound:peak_high_bound+1][np.argmax(spec[peak_low_bound:peak_high_bound+1])])
                if peak_high_bound >= self.nfft/2:
                    peak_high_bound -= 1
                peaks_low.append(peak_low_bound)
                peaks_high.append(peak_high_bound)
                freqs_low.append(freq_ax[peak_low_bound])
                try:
                    freqs_high.append(freq_ax[peak_high_bound])
                except IndexError:
                    freqs_high.append(freq_ax[-1])
                
                #Peak Clustering - Merge peaks if needed
                if j > 0:
                    if merge_mask[j-1]:
                        #Cluster the peak with the previous one
                        peaks_low.pop(-1)
                        peaks_high.pop(-2)
                        freqs_low.pop(-1)
                        freqs_high.pop(-2)
                        final_peaks[-1] = final_peaks[-2:][np.argmax(final_peak_vals[-2:])]
                        final_peaks.pop(-2)
                        final_peak_vals.pop(len(final_peak_vals)-2 + np.argmin(final_peak_vals[-2:]))
                        
                elif j == 0 and i > 0:
                    #Cluster with last peak from previous interval
                    if len(freqs_low) > 1:
                        if freqs_low[-1] - samples_distance*df < freqs_high[-2]:
                            peaks_low.pop(-1)
                            peaks_high.pop(-2)
                            freqs_low.pop(-1)
                            freqs_high.pop(-2) 
                            final_peaks[-1] = final_peaks[-2:][np.argmax(final_peak_vals[-2:])]
                            final_peaks.pop(-2)
                            final_peak_vals.pop(len(final_peak_vals)-2 + np.argmin(final_peak_vals[-2:]))

        freqs = np.stack((freqs_low,freqs_high),axis=1)

        return freqs,np.array(final_peaks), np.array(final_peak_vals)
    
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1
    
        for kernel_size, stride in zip(self.conv_kernel, self.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
    
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def filter_decomp(self, x, freqs):
        """
        The Filter Decomposition via bandpass IIR filtering based on detected spectral peaks.
        """
        OCs_spectrum = []
        OCs = []
        for freq in freqs:
            sos = scipy.signal.butter(self.buttord,freq,'bp',fs=self.fs,output='sos')
            x_filt = scipy.signal.sosfiltfilt(sos,x)
            OCs.append(x_filt)
            filt_spec,freq_axis = mlab.psd(x_filt,NFFT = self.nfft,Fs=self.fs)
            OCs_spectrum.append(filt_spec)
            
        OCs_spectrum = np.array(OCs_spectrum)
        OCs = np.array(OCs)
        
        #Reconstruct the signal with the filtered components to measure NRMSE
        x_recon = np.sum(OCs,axis=0)

        # Calculate NRMSE
        mse = np.mean((x - x_recon) ** 2)
        rmse = np.sqrt(mse)
        nrmse = rmse / (np.max(x) - np.min(x))

        return OCs, x_recon, OCs_spectrum, nrmse #,correlogram

    def decomp_mask(self,x,mask_time_indices,attention_mask, remove_silence,freq_labels = None,peak_det_intervals=None):
        """
        Main function to perform frame-level decomposition masking on the input signal x.
        It invokes all utilities (peak detection, decomposition, aggregation strategies) to produce masked versions of the input signal for a batched input x.
        """

        if x.ndim == 2:
            batch_size, sequence_length = x.shape[0], x.shape[1]
        else:
            sequence_length = len(x)
        frame_length = int(self.RFS*self.fs)
        start_indices = list(range(0, sequence_length - frame_length + 1, int(self.stride*self.fs)))
        #Take care of the last frame which is lost 
        start_indices = np.array(start_indices)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                len(start_indices), attention_mask, add_adapter=False
            )

        input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
        )

        #Store original utterances and the masked versions
        masked_x = np.zeros((self.NoC+1,batch_size, len(start_indices),frame_length))
        reconstruction_NRMSEs = []
        correlograms = []
        component_indices = torch.ones_like(mask_time_indices).unsqueeze(0).expand(self.NoC,-1,-1).clone()    
        assert mask_time_indices is not None
        false_count = 0 
        if freq_labels is not None:
            detected_labels = -1*np.ones_like(freq_labels)
        #Mask time indices should be externally provided
        for u,utt in enumerate(mask_time_indices):
            for s,start in enumerate(utt):
                if s > input_lengths[u]-1:
                    masked_x[:,u,s,:] = np.zeros((self.NoC+1,frame_length))
                    continue
                start_index = start_indices[s]
                frame = x[u,start_index:start_index + frame_length]
                if len(frame) < frame_length:
                    #This will probably be true for last frame in the utterance
                    #If frame is shorter than the desired length, pad with zeros
                    frame = np.concatenate((frame,np.zeros(frame_length-len(frame))),axis=0)
                if start:
                    if remove_silence:
                        is_silent = self.silence_check(frame)
                    else:
                        is_silent = False
                    if not is_silent:
                        freqs,exact_freqs,exact_freqs_amp = self.peak_detection(frame,peak_det_intervals)

                        assert freqs.shape[0] == len(exact_freqs)
                        if freqs.shape[0] == 0:
                            mask_time_indices[u,s] = False
                            false_count += 1
                            masked_x[:,u,s,:] = np.concatenate((frame.reshape(1,len(frame)),np.zeros((self.NoC,frame_length))),axis=0)
                            component_indices[:,u,s] = torch.tensor([0]*self.NoC)
                            continue
                        if self.decomp_to_perform == 'filter':                
                            "Filter Decomposition - FD"                         
                            OCs, _,OCs_spectrum, nrmse = self.filter_decomp(frame,freqs)

                            f,Pxx = scipy.signal.welch(frame, fs=self.fs, window='hann', nperseg=len(frame)/3, noverlap=len(frame)/40, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
                            OCs_spectrum = np.zeros((OCs.shape[0],self.nfft//2+1))
                            "Get spectra of detected OCs"
                            for n in range(OCs.shape[0]):
                                _,OCs_spectrum[n] = scipy.signal.welch(OCs[n], fs=self.fs, window='hann', nperseg=len(frame)/6, noverlap=len(frame)/40, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
                            to_delete = []
                            
                            "Check spectral amplitude tolerance to discard weak OCs"
                            for n,spec in enumerate(OCs_spectrum):
                                if spec.max() < self.spec_amp_tolerance:
                                    to_delete.append(n)
                            OCs = np.delete(OCs,to_delete,0)
                            freqs_not_deleted = False
                            try:
                                freqs = np.delete(freqs,to_delete,0)
                                exact_freqs_amp = np.delete(exact_freqs_amp,to_delete,0)
                                if freq_labels is not None:
                                    exact_freqs = np.delete(exact_freqs,to_delete,0)
                                #exact freqs not deleted to be used later
                            except IndexError:
                                freqs_not_deleted = True
                                
                            "find nrmse"
                            frame_recon = np.sum(OCs,axis=0)
                            mse = np.mean((frame - frame_recon) ** 2)
                            rmse = np.sqrt(mse)
                            nrmse = rmse / (np.max(frame) - np.min(frame))
                        elif self.decomp_to_perform == "vmd":
                            "Variational Mode Decomposition -VMD"
                            OCs, _, mode_freqs = VMD(frame, alpha = self.vmd_alpha, tau = self.vmd_tau, K = self.NoC, DC = self.vmd_DC, init = self.vmd_init, tol=self.vmd_tol)
                            sort_inds = np.argsort(mode_freqs[-1])
                            OCs = OCs[sort_inds]
                            exact_freqs = mode_freqs[-1][sort_inds] * self.fs
                            
                            if self.use_vmd_correction:
                                min_freq_distance = min(np.diff(exact_freqs))
                                first_second_diff = np.diff(exact_freqs)[0]
                                second_third_diff = np.diff(exact_freqs)[1] 
                                                            
                                agreement = self.classify_omegas(exact_freqs,freqs)
                                amps_agreement = exact_freqs_amp[agreement[agreement != -1]]
                                amps_sorted = np.sort(exact_freqs_amp)
                                if len(amps_agreement) == 0:
                                    agrees_with_peak_detection = False
                                else:
                                    if np.min(amps_agreement) < amps_sorted[0]:
                                        agrees_with_peak_detection = False
                                    else:
                                        agrees_with_peak_detection = True
                                max_correction_iter = 1000
                                "If frequencies found in peak detection dont agree with VMD, or found frequency exceeds maximum possible frequency, or two found frequencies are too close (essentialy the same frequency), reinitialize VMD"
                                while second_third_diff < 400 or first_second_diff > 1800 or not agrees_with_peak_detection or np.max(exact_freqs) > self.higher_speech_freq or min_freq_distance < 240:
                                    "Change VMD initialization to random instead of uniform"
                                    OCs, _, mode_freqs = VMD(frame, alpha = self.vmd_alpha, tau = self.vmd_tau, K = self.NoC, DC = self.vmd_DC, init = 2, tol=self.vmd_tol)
                                    sort_inds = np.argsort(mode_freqs[-1])
                                    OCs = OCs[sort_inds]
                                    exact_freqs = mode_freqs[-1][sort_inds] * self.fs
                                    min_freq_distance = min(np.diff(exact_freqs))
                                    first_second_diff = np.diff(exact_freqs)[0]
                                    second_third_diff = np.diff(exact_freqs)[1] 
                                    agreement = self.classify_omegas(exact_freqs,freqs)
                                    amps_agreement = exact_freqs_amp[agreement[agreement != -1]]
                                    amps_sorted = np.sort(exact_freqs_amp)
                                    if len(amps_agreement) == 0:
                                        agrees_with_peak_detection = False
                                    else:
                                        if np.min(amps_agreement) < amps_sorted[0]:
                                            agrees_with_peak_detection = False
                                        else:
                                            agrees_with_peak_detection = True
                                    max_correction_iter -= 1
                                    if max_correction_iter <= 0 and (min_freq_distance > 100 and min_freq_distance < 1800 and second_third_diff > 400 and np.max(exact_freqs) < self.higher_speech_freq):
                                        break

                            f,Pxx = scipy.signal.welch(frame, fs=self.fs, window='hann', nperseg=len(frame)/3, noverlap=len(frame)/40, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
                            OCs_spectrum = np.zeros((OCs.shape[0],self.nfft//2+1))
                            for n in range(OCs.shape[0]):                  
                                _,OCs_spectrum[n] = scipy.signal.welch(OCs[n], fs=self.fs, window='hann', nperseg=len(frame)/6, noverlap=len(frame)/40, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
                            to_delete = []
                            for n,spec in enumerate(OCs_spectrum):
                                if spec.max() < self.spec_amp_tolerance:
                                    to_delete.append(n)
                            OCs = np.delete(OCs,to_delete,0)
                            freqs_not_deleted = False
                            try:
                                freqs = np.delete(freqs,to_delete,0)
                                exact_freqs_amp = np.delete(exact_freqs_amp,to_delete,0)
                                if freq_labels is not None:
                                    exact_freqs = np.delete(exact_freqs,to_delete,0)
                                #exact freqs not deleted to be used later
                            except IndexError:
                                freqs_not_deleted = True
                                
                            "find nrmse"
                            frame_recon = np.sum(OCs,axis=0)
                            mse = np.mean((frame - frame_recon) ** 2)
                            rmse = np.sqrt(mse)
                            nrmse = rmse / (np.max(frame) - np.min(frame))
                        elif self.decomp_to_perform == "emd":
                            "Empirical Mode Decomposition - EMD"
                            #EMD returns IMFs (roughly) ranging from high to low frequencies - Last slice is the residual
                            self.emd = EMD(DTYPE=np.float32,spline_kind=self.emd_spline_kind,MAX_ITERATION=self.emd_max_iter,energy_ratio_thr=self.emd_energy_ratio_thr,std_thr=self.emd_std_thr,svar_thr=self.emd_svar_thr,total_power_thr=self.emd_total_power_thr,range_thr=self.emd_range_thr,extrema_detection=self.emd_extrema_detection)

                            OCs = self.emd.emd(frame-np.mean(frame),max_imf=self.NoC)#[:self.NoC,:]
                            f,Pxx = scipy.signal.welch(frame, fs=self.fs, window='hann', nperseg=len(frame)/3, noverlap=len(frame)/40, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
                            OCs_spectrum = np.zeros((OCs.shape[0],self.nfft//2+1))
                            det_freqs = []
                            det_exact_freqs = []
                            for n in range(OCs.shape[0]):
                                _,OCs_spectrum[n] = scipy.signal.welch(OCs[n], fs=self.fs, window='hann', nperseg=len(frame)/6, noverlap=len(frame)/40, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
                                df,ef,_ = self.peak_detection(OCs[n],peak_det_intervals)
                                if len(df) > 1:
                                    df = np.array([[df.flatten()[0],df.flatten()[-1]]])
                                    ef = np.array([ef.mean()])
                                if len(df) > 0:
                                    det_freqs.append(df)
                                    det_exact_freqs.append(ef)
                            if freq_labels is not None:    
                                try:
                                    freqs = np.array(det_freqs).squeeze(1)
                                except:
                                    freqs = np.array(det_freqs)

                            exact_freqs = np.array(det_exact_freqs)
                            to_delete = []
                            OCs = OCs[::-1,:]
                            OCs_spectrum = OCs_spectrum[::-1,:]
                            for n,spec in enumerate(OCs_spectrum):
                                if spec.max() < self.spec_amp_tolerance:
                                    to_delete.append(n)
                            OCs = np.delete(OCs,to_delete,0)
                            freqs_not_deleted = False
                            try:
                                freqs = np.delete(freqs,to_delete,0)
                                exact_freqs_amp = np.delete(exact_freqs_amp,to_delete,0)
                                if freq_labels is not None:
                                    exact_freqs = np.delete(exact_freqs,to_delete,0)
                                #exact freqs not deleted to be used later
                            except IndexError:
                                freqs_not_deleted = True   
                                                    
                            #find nrmse
                            frame_recon = np.sum(OCs,axis=0)
                            mse = np.mean((frame - frame_recon) ** 2)
                            rmse = np.sqrt(mse)
                            nrmse = rmse / (np.max(frame) - np.min(frame))
                        elif self.decomp_to_perform == "ewt":
                            "Empirical Wavelet Transform - EWT"
                            #Ewt returns a low frequency component + the specified number of OCs
                            OCs_w_lf,_ ,boundaries = ewtpy.EWT1D(frame, N = self.NoC+1, completion = self.ewt_completion, reg = self.ewt_filter,lengthFilter = self.ewt_filter_length,sigmaFilter = self.ewt_filter_sigma, log = self.ewt_log_spectrum, detect = self.ewt_detect)                                                     
                            #Discard first low frequency component
                            OCs_w_lf = np.transpose(OCs_w_lf)
                            OCs = OCs_w_lf[1:,:]
                            assert OCs.shape[0] == self.NoC
                            
                            f,Pxx = scipy.signal.welch(frame, fs=self.fs, window='hann', nperseg=len(frame)/3, noverlap=len(frame)/40, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
                            OCs_spectrum = np.zeros((OCs.shape[0],self.nfft//2+1))                            
                            det_freqs = []
                            det_exact_freqs = []
                            "To get detected frequencies, run a peak detection on each component"
                            for n in range(OCs.shape[0]):
                                _,OCs_spectrum[n] = scipy.signal.welch(OCs[n], fs=self.fs, window='hann', nperseg=len(frame)/6, noverlap=len(frame)/40, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
                                df,ef,_ = self.peak_detection(OCs[n],peak_det_intervals)
                                if len(df) > 1:
                                    df = np.array([[df.flatten()[0],df.flatten()[-1]]])
                                    ef = np.array([ef.mean()])
                                if len(df) > 0:
                                    det_freqs.append(df)
                                    det_exact_freqs.append(ef)
                            if freq_labels is not None:
                                try:
                                    freqs = np.array(det_freqs).squeeze(1)
                                except:
                                    freqs = np.array(det_freqs)                               
                            exact_freqs = np.array(det_exact_freqs)
                            to_delete = []
                            for n,spec in enumerate(OCs_spectrum):
                                if spec.max() < self.spec_amp_tolerance:
                                    to_delete.append(n)
                            OCs = np.delete(OCs,to_delete,0)
                            freqs_not_deleted = False
                            try:
                                freqs = np.delete(freqs,to_delete,0)
                                exact_freqs_amp = np.delete(exact_freqs_amp,to_delete,0)
                                if freq_labels is not None:
                                    exact_freqs = np.delete(exact_freqs,to_delete,0)
                                #exact freqs not deleted to be used later
                            except IndexError:
                                freqs_not_deleted = True
                                pass
                            "find nrmse"
                            frame_recon = np.sum(OCs_w_lf,axis=0)
                            mse = np.mean((frame - frame_recon) ** 2)
                            rmse = np.sqrt(mse)
                            nrmse = rmse / (np.max(frame) - np.min(frame))

                        else:
                            raise ValueError('Decomposition method not supported')

                        "Check number of obtained OCs and apply aggregation (more than specified) or padding (less than specified) if needed"
                        if freq_labels is not None:
                            if len(exact_freqs) == detected_labels.shape[1]:
                                detected_labels[s] = self.classify_omegas(exact_freqs)
                        if OCs.shape[0] > self.NoC:
                            "If more components are found than specified, aggregate them according to the specified strategy"
                            if self.group_OCs_by_frame == 'low_freqs_first':
                                if self.decomp_to_perform == 'emd':
                                    #Superpose last components (low frequencies)
                                    OCs[self.NoC-1] = OCs[self.NoC-1:].sum(axis=0)
                                    OCs = OCs[:self.NoC]
                                else:
                                    #Superpose first components (low frequencies)
                                    old_shape = OCs.shape[0]
                                    OCs[0] = OCs[:(OCs.shape[0] - (self.NoC-1))].sum(axis=0)
                                    OCs = np.delete(OCs,range(1,OCs.shape[0]- (self.NoC-1)),0)
                                if freq_labels is not None:
                                    avg_freq = exact_freqs[:(old_shape-(self.NoC-1))].mean()
                                    exact_freqs = exact_freqs[-self.NoC:]
                                    exact_freqs[0] = avg_freq
                                    detected_labels[s] = self.classify_omegas(exact_freqs)
                                assert OCs.shape[0] == self.NoC
                            elif self.group_OCs_by_frame == 'high_freqs_first':             
                                if self.decomp_to_perform == 'emd':
                                    #Superpose first components (high frequencies)
                                    OCs[0] = OCs[:(OCs.shape[0] - (self.NoC-1))].sum(axis=0)
                                    OCs = np.delete(OCs,range(1,OCs.shape[0]- (self.NoC-1)),0)
                                else:                      
                                    #Superpose last components (high frequencies)
                                    OCs[self.NoC-1] = OCs[self.NoC-1:].sum(axis=0)
                                    OCs = OCs[:self.NoC]
                                if freq_labels is not None:
                                    avg_freq = exact_freqs[self.NoC-1:].mean()
                                    exact_freqs = exact_freqs[:self.NoC]
                                    exact_freqs[-1] = avg_freq
                                    detected_labels[s] = self.classify_omegas(exact_freqs)
                                assert OCs.shape[0] == self.NoC
                            elif self.group_OCs_by_frame == 'equally_distribute':    
                                #Cluster components with neighbours in equally populated groups    
                                #This functions similar to high_freqs_first starting grouping from high frequencies                            
                                #In EMD, low frequencies are in the last components, and a lot of spurious low frequency components are found
                                #so its a good strategy to superpose low frequencies first
                                new_OCs = []
                                new_exact_freqs = []
                                #Slightly higher than NoC is same as high_freqs_first
                                #Significantly higher (> .5) gives the equal distribution
                                if OCs.shape[0]/self.NoC == 1.5:
                                    step = int(OCs.shape[0]/self.NoC)
                                else:
                                    step = int(round(OCs.shape[0]/self.NoC))
                                for i in range(self.NoC):
                                    start_idx = i*step
                                    end_idx = (i+1)*step if i != self.NoC -1 else OCs.shape[0] 
                                    new_OCs.append(OCs[start_idx:end_idx].sum(axis=0))
                                    if freq_labels is not None:
                                        new_exact_freqs.append(exact_freqs[start_idx:end_idx].mean())
                                OCs = np.array(new_OCs)
                                if freq_labels is not None:
                                    detected_labels[s] = self.classify_omegas(np.array(new_exact_freqs))
                                assert OCs.shape[0] == self.NoC

                        elif OCs.shape[0] < self.NoC:
                            "If fewer components are found than specified, we need to find approximately which component is missing and pad them with zeros"
                            "To do this we need to infer based on the detected frequencies of existing components, which is more likely to be missing"
                            "If it cannot be inferred, then we pad the last components with zeros (highest frequencies)"
                            
                            if not OCs.any():
                                #If 0 components are found (empty OCs)
                                component_indices[:,u,s] = torch.tensor([0]*self.NoC)
                                OCs = np.zeros((self.NoC,frame_length))
                                mask_time_indices[u,s] = False
                                false_count += 1
                                ortho_num = 0
                                if freq_labels is not None:
                                    detected_labels[s] = np.array([-1]*self.NoC)
                            else:
                                if freq_labels is not None:
                                    groups = self.classify_omegas(exact_freqs)
                                    gt = np.sort(freq_labels[s])
                                    missing_positions = []
                                    if len(groups) < len(gt):
                                        for g in gt:
                                            if g not in groups and g != -1:
                                                temp = np.append(groups,g)
                                                pos = np.where(np.sort(temp) == g)[0][0]
                                                if pos not in missing_positions:
                                                    missing_positions.append(pos)
                                                elif pos-1 not in missing_positions and pos-1 >= 0:
                                                        missing_positions.append(pos-1)
                                                else:
                                                    missing_positions.append(pos+1)

                                        if len(np.unique(gt)) < len(gt) and len(missing_positions) + len(groups) < len(gt):
                                            #means two occurences inside the same frequency group
                                            #find which group, except if it's -1
                                            un, counts = np.unique(gt, return_counts=True)
                                            count = dict(zip(un, counts))
                                            for g in un:
                                                if count[g] > 1 and g != -1 and len(groups) > 0:
                                                    pos = np.where(groups == g)[0][0]
                                                    if pos not in missing_positions:
                                                        missing_positions.append(pos)
                                                    elif pos-1 not in missing_positions and pos-1 >= 0:
                                                        missing_positions.append(pos-1)
                                                    else:
                                                        missing_positions.append(pos+1)
                                    
                                    elif len(groups) > len(gt):
                                        if OCs.shape[0] == len(gt):
                                            pass
                                        elif OCs.shape[0] < len(gt):
                                            pass
                                            
                                    else:
                                        if not (groups == gt).all():
                                            for g in groups:
                                                if g not in gt and -1 in gt:
                                                    gt[np.where(gt == -1)[0][0]] = g
                                            gt = np.sort(gt)
                                        else:
                                            #In this case everything went fine, the prediction is correct
                                            pass

                                    missing_positions = np.sort(missing_positions).tolist()
                                    if len(missing_positions) > 0:
                                        #If component is missing and we know which one, impute that dimension
                                        #But if OCs are already found and not deleted, then impute less dimensions
                                        if len(groups) + len(missing_positions) > OCs.shape[0]:
                                            missing_positions = missing_positions[:np.abs(OCs.shape[0] - len(gt))]
                                        if len(groups) + len(missing_positions) < self.NoC:
                                            imp = missing_positions.copy()
                                            imp.append(0)
                                            for m in imp:
                                                groups = np.insert(groups,m,-1)
                                        else:
                                            #groups + missing > NoC
                                            for m in missing_positions:
                                                groups = np.insert(groups,m,-1)
                                        while groups.shape[0] < self.NoC:
                                            if (0 in groups or 1 in groups) and (2 in groups or 3 in groups or 4 in groups or 5 in groups):
                                                groups = np.insert(groups,len(groups),-1)
                                            elif (2 in groups or 3 in groups or 4 in groups or 5 in groups) and (6 in groups or 7 in groups or 8 in groups):
                                                groups = np.insert(groups,0,-1)
                                            elif (0 in groups or 1 in groups) and (6 in groups or 7 in groups or 8 in groups):
                                                groups = np.insert(groups,1,-1)
                                            elif (0 in groups or 1 in groups):
                                                groups = np.insert(groups,1,-1)
                                            elif (2 in groups or 3 in groups or 4 in groups or 5 in groups):
                                                groups = np.insert(groups,0,-1)
                                            elif (6 in groups or 7 in groups or 8 in groups):
                                                groups = np.insert(groups,0,-1)
                                            else:
                                                groups = np.insert(groups,0,-1)    
                                        if freqs_not_deleted and len(groups) > self.NoC:
                                            for t in to_delete:
                                                if t > len(exact_freqs)-1:
                                                    #the ones that were not deleted before
                                                    exact_freqs = np.delete(exact_freqs,t-1,0)
                                            groups = self.classify_omegas(exact_freqs)
                                            while groups.shape[0] < self.NoC:
                                                if (0 in groups or 1 in groups) and (2 in groups or 3 in groups or 4 in groups or 5 in groups):
                                                    groups = np.insert(groups,len(groups),-1)
                                                elif (2 in groups or 3 in groups or 4 in groups or 5 in groups) and (6 in groups or 7 in groups or 8 in groups):
                                                    groups = np.insert(groups,0,-1)
                                                elif (0 in groups or 1 in groups) and (6 in groups or 7 in groups or 8 in groups):
                                                    groups = np.insert(groups,1,-1)
                                                elif (0 in groups or 1 in groups):
                                                    groups = np.insert(groups,1,-1)
                                                elif (2 in groups or 3 in groups or 4 in groups or 5 in groups):
                                                    groups = np.insert(groups,0,-1)
                                                elif (6 in groups or 7 in groups or 8 in groups):
                                                    groups = np.insert(groups,0,-1)
                                                else:
                                                    groups = np.insert(groups,0,-1)    
                                        detected_labels[s] = groups.copy()
                                        component_indices[missing_positions,u,s] = torch.tensor([0]*(len(missing_positions))).type_as(component_indices)
                                        if len(OCs.shape) > 1:
                                            ortho_num = OCs.shape[0]
                                        else:
                                            OCs = OCs.reshape(1,-1)
                                            ortho_num = 1
                                        
                                        for n in missing_positions:
                                            try:
                                                OCs = np.insert(OCs,n,np.zeros(frame_length),axis=0)
                                            except IndexError:
                                                #out of bounds
                                                OCs = np.insert(OCs,n-1,np.zeros(frame_length),axis=0)
                                            if OCs.shape[0] == self.NoC:
                                                break
                                        if OCs.shape[0] > self.NoC:
                                            pass
                                    elif groups.shape[0] < self.NoC:
                                        while groups.shape[0] < self.NoC:
                                            if (0 in groups or 1 in groups) and (2 in groups or 3 in groups or 4 in groups or 5 in groups):
                                                groups = np.insert(groups,len(groups),-1)
                                            elif (2 in groups or 3 in groups or 4 in groups or 5 in groups) and (6 in groups or 7 in groups or 8 in groups):
                                                groups = np.insert(groups,0,-1)
                                            elif (0 in groups or 1 in groups) and (6 in groups or 7 in groups or 8 in groups):
                                                groups = np.insert(groups,1,-1)
                                            elif (0 in groups or 1 in groups):
                                                groups = np.insert(groups,1,-1)
                                            elif (2 in groups or 3 in groups or 4 in groups or 5 in groups):
                                                groups = np.insert(groups,0,-1)
                                            elif (6 in groups or 7 in groups or 8 in groups):
                                                groups = np.insert(groups,0,-1)
                                            else:
                                                groups = np.insert(groups,0,-1)                       
                                        detected_labels[s] = groups.copy()
                                    elif groups.shape[0] > self.NoC:
                                        groups = np.unique(groups)
                                        detected_labels[s] = groups.copy()                                           
                                    else:
                                        detected_labels[s] = groups.copy()
                                    freq_labels[s] = gt
                                    if OCs.shape[0] < self.NoC:
                                        #If components found are still less than the specified, then just pad the last one
                                        component_indices[OCs.shape[0]:self.NoC,u,s] = torch.tensor([0]*(self.NoC-OCs.shape[0]))
                                        if len(OCs.shape) > 1:
                                            ortho_num = OCs.shape[0]
                                        else:
                                            OCs = OCs.reshape(1,-1)
                                            ortho_num = 1
                                        OCs = np.concatenate((OCs,np.zeros((self.NoC-OCs.shape[0],frame_length))),axis=0)

                                else:
                                    if self.decomp_to_perform == 'emd':
                                        OCs = OCs[::-1,:]
                                    if len(OCs.shape) > 1:
                                        ortho_num = OCs.shape[0]
                                    else:
                                        OCs = OCs.reshape(1,-1)
                                        ortho_num = 1
                                    if len(to_delete) > 0:
                                        to_delete = to_delete[:self.NoC-OCs.shape[0]]                                       
                                        if len(np.intersect1d(to_delete,self.classify_omegas(exact_freqs))) > 0 and len(np.intersect1d(to_delete,self.classify_omegas(np.mean(freqs,axis=1)))) == 0 and self.decomp_to_perform != 'emd': 
                                            #If this happens it means that the deletion will be wrong 
                                            # and we need to correct it by deleting the lowest amplitude component
                                            if not (self.NoC - OCs.shape[0] > len(to_delete)):
                                                to_delete = []
                                            min_amp = 1000
                                            for n in range(exact_freqs_amp.shape[0]): #range(self.NoC):
                                                if exact_freqs_amp[n] < min_amp:
                                                    min_amp = exact_freqs_amp[n]
                                                    min_freq = n
                                            try:
                                                to_delete.append(min_freq)
                                            except NameError:
                                                # select as deleted the highest frequency (lowest amplitude)
                                                to_delete.append(self.NoC-1)
                                        if min(to_delete) > OCs.shape[0]:
                                            to_delete = [d - (min(to_delete) - OCs.shape[0]) for d in to_delete]

                                        for n in to_delete:
                                            if OCs.shape[0] >= self.NoC:
                                                break
                                            OCs = np.insert(OCs,n,np.zeros(frame_length),axis=0)
                                        while OCs.shape[0] < self.NoC:
                                            OCs = np.insert(OCs,OCs.shape[0],np.zeros(frame_length),axis=0)
                                            to_delete.append(OCs.shape[0]-1)
                                    else:
                                        groups = self.classify_omegas(exact_freqs)
                                        for n in range(self.NoC):
                                            if n not in groups:
                                                to_delete.append(n)
                                        if len(to_delete) > self.NoC - OCs.shape[0]:
                                            if len(np.unique(groups)) < len(groups) and (to_delete > groups.max()).all(): 
                                                to_delete = to_delete[self.NoC-OCs.shape[0]:]
                                            else:
                                                to_delete = to_delete[:self.NoC-OCs.shape[0]]
                                        if min(to_delete) > OCs.shape[0]:
                                            to_delete = [d - (min(to_delete) - OCs.shape[0]) for d in to_delete]

                                        for n in to_delete:
                                            if OCs.shape[0] >= self.NoC:
                                                break
                                            OCs = np.insert(OCs,n,np.zeros(frame_length),axis=0)
                                            groups = np.insert(groups,n,-1)
                                        while OCs.shape[0] < self.NoC:
                                            OCs = np.insert(OCs,OCs.shape[0],np.zeros(frame_length),axis=0)
                                            to_delete.append(OCs.shape[0]-1)
                                    component_indices[to_delete,u,s] = torch.tensor([0]*(len(to_delete))).type_as(component_indices) #self.NoC-OCs.shape[0])


                        #Calculate Orthogonality Index - Input time series domain
                        if 'ortho_num' not in locals():
                            correlogram = np.zeros((self.NoC,self.NoC))
                            for o in range(self.NoC):
                                for r in range(self.NoC):
                                    correlogram[o,r] = np.cov(OCs[o],OCs[r])[0,1] / (np.std(OCs[o])*np.std(OCs[r]))
                        else:
                            correlogram = np.zeros((ortho_num,ortho_num))
                            for o in range(ortho_num):
                                for r in range(ortho_num):
                                    correlogram[o,r] = np.cov(OCs[o],OCs[r])[0,1] / (np.std(OCs[o])*np.std(OCs[r]))
                            del ortho_num
                        
                        x_MR = np.concatenate((frame.reshape(1,len(frame)),OCs),axis=0)
                        try:
                            masked_x[:,u,s,:] = x_MR 
                        except: 
                            raise ValueError('Dimension mismatch')

                        reconstruction_NRMSEs.append(nrmse)
                        correlograms.append(correlogram)

                    else:
                        "If silent, then it will not be masked and not used to calculate statistics and loss in a DecVAE with Wav2vec2 functionality"
                        mask_time_indices[u,s] = False
                        false_count += 1
                        masked_x[:,u,s,:] = np.concatenate((frame.reshape(1,len(frame)),np.zeros((self.NoC,frame_length))),axis=0)
                        component_indices[:,u,s] = torch.tensor([0]*self.NoC)
                else:
                    #mask_time_indices[u,s] is already False here
                    masked_x[:,u,s,:] = np.concatenate((frame.reshape(1,len(frame)),np.zeros((self.NoC,frame_length))),axis=0)
                    component_indices[:,u,s] = torch.tensor([0]*self.NoC)
        if freq_labels is not None:
            return masked_x,mask_time_indices,start_indices, reconstruction_NRMSEs, correlograms, component_indices , detected_labels#,None   
        else:
            return masked_x,mask_time_indices,start_indices, reconstruction_NRMSEs, correlograms, component_indices, None    
        
    def sequence_decomp(self,x,attention_mask,peak_det_intervals):
        """Sequence decomposition; here the whole sequence is decomposed at once and no masking takes place. The rest of the steps 
        (peak detection, decomposition, aggregation/padding) are the same as in frame_decomp."""
        
        if x.ndim == 2:
            batch_size, sequence_length = x.shape[0], x.shape[1]
        else:
            sequence_length = len(x)

        input_lengths = (
            attention_mask.sum(-1).detach().tolist()
            if attention_mask is not None
            else [sequence_length for _ in range(batch_size)]
        )
        #Store original utterances and the masked versions
        masked_x = np.zeros((self.NoC_seq+1,batch_size, sequence_length))
        component_indices = torch.ones((batch_size,self.NoC_seq))
        correlograms = []
        for u,utt in enumerate(attention_mask):
            mask = utt.detach().cpu().numpy().astype(bool)
            sequence = x[u,mask]
            freqs,exact_freqs,exact_freqs_amp = self.peak_detection(sequence,peak_det_intervals)

            if self.decomp_to_perform == 'filter':                            
                "Filter Decomposition - FD"
                OCs, _,OCs_spectrum, nrmse = self.filter_decomp(sequence,freqs) 

            elif self.decomp_to_perform == "vmd":
                "Variational Mode Decomposition - VMD"
                OCs, _, mode_freqs = VMD(sequence, alpha = self.vmd_alpha, tau = self.vmd_tau, K = self.NoC_seq, DC = self.vmd_DC, init = self.vmd_init, tol=self.vmd_tol)
                if OCs.shape[1] < sequence.shape[0]:
                    OCs = np.pad(OCs, ((0,0),(0,sequence.shape[0]-OCs.shape[1])), 'constant', constant_values=(0,0))
                sort_inds = np.argsort(mode_freqs[-1])
                OCs = OCs[sort_inds]
                
            elif self.decomp_to_perform == "emd":
                "Empirical Mode Decomposition - EMD"
                #EMD returns IMFs (roughly) ranging from high to low frequencies
                OCs = self.emd.emd(sequence,max_imf=self.NoC_seq) #[:self.NoC_seq,:]
                "Spectral refinement of found OCs"
                OCs_spectrum = np.zeros((OCs.shape[0],self.nfft//2+1))
                det_freqs = []
                det_exact_freqs = []
                for n in range(OCs.shape[0]):
                    _,OCs_spectrum[n] = scipy.signal.welch(OCs[n], fs=self.fs, window='hann', nperseg=66, noverlap=10, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
                    df,ef,_ = self.peak_detection(OCs[n],peak_det_intervals)
                    if len(df) > 1:
                        df = np.array([[df.flatten()[0],df.flatten()[-1]]])
                        ef = np.array([ef.mean()])
                    if len(df) > 0:
                        det_freqs.append(df)
                        det_exact_freqs.append(ef)
                
                to_delete = []
                OCs = OCs[::-1,:]
                OCs_spectrum = OCs_spectrum[::-1,:]
                for n,spec in enumerate(OCs_spectrum):
                    if spec.max() < self.spec_amp_tolerance_seq:
                        to_delete.append(n)
                OCs = np.delete(OCs,to_delete,0)
                freqs_not_deleted = False
                try:
                    freqs = np.delete(freqs,to_delete,0)
                    exact_freqs_amp = np.delete(exact_freqs_amp,to_delete,0)
                except IndexError:
                    freqs_not_deleted = True   
                                        
            elif self.decomp_to_perform == "ewt":
                "Empirical Wavelet Transform - EWT"
                #Ewt returns a low frequency component + the specified number of OCs
                OCs_w_lf,_ ,boundaries = ewtpy.EWT1D(sequence, N = self.NoC_seq+1, completion = self.ewt_completion, reg = self.ewt_filter,lengthFilter = self.ewt_filter_length,sigmaFilter = self.ewt_filter_sigma, log = self.ewt_log_spectrum, detect = self.ewt_detect)                                                     
                #Discard first low frequency component
                OCs_w_lf = np.transpose(OCs_w_lf)
                OCs = OCs_w_lf[1:,:]
                assert OCs.shape[0] == self.NoC_seq
                "Spectral refinement of found OCs"
                OCs_spectrum = np.zeros((OCs.shape[0],self.nfft//2+1))                            
                det_freqs = []
                det_exact_freqs = []
                for n in range(OCs.shape[0]):
                    _,OCs_spectrum[n] = scipy.signal.welch(OCs[n], fs=self.fs, window='hann', nperseg=66, noverlap=10, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
                    df,ef,_ = self.peak_detection(OCs[n],peak_det_intervals)
                    if len(df) > 1:
                        df = np.array([[df.flatten()[0],df.flatten()[-1]]])
                        ef = np.array([ef.mean()])
                    if len(df) > 0:
                        det_freqs.append(df)
                        det_exact_freqs.append(ef)
                to_delete = []
                for n,spec in enumerate(OCs_spectrum):
                    if spec.max() < self.spec_amp_tolerance_seq:
                        to_delete.append(n)
                OCs = np.delete(OCs,to_delete,0)
                freqs_not_deleted = False
                try:
                    freqs = np.delete(freqs,to_delete,0)
                    exact_freqs_amp = np.delete(exact_freqs_amp,to_delete,0)
                except IndexError:
                    freqs_not_deleted = True
                    pass

            else:
                raise ValueError('Decomposition method not supported')

            "Spectral refinement of found OCs"
            if self.decomp_to_perform == 'filter' or self.decomp_to_perform == 'vmd':
                f,orig_spec = scipy.signal.welch(sequence, fs=self.fs, window='hann', nperseg=66, noverlap=10, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
                OCs_spectrum = np.zeros((OCs.shape[0],self.nfft//2+1))
                for n in range(OCs.shape[0]):
                    f,OCs_spectrum[n] = scipy.signal.welch(OCs[n], fs=self.fs, window='hann', nperseg=66, noverlap=10, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
                to_delete = []
                
                for n,spec in enumerate(OCs_spectrum):
                    if spec.max() < self.spec_amp_tolerance_seq:
                        to_delete.append(n)
                OCs = np.delete(OCs,to_delete,0)

                freqs_not_deleted = False
                try:
                    freqs = np.delete(freqs,to_delete,0)
                    exact_freqs_amp = np.delete(exact_freqs_amp,to_delete,0)
                except IndexError:
                    freqs_not_deleted = True
                if self.group_OCs_by_seq == 'top_k':
                    "Select top-k components based on their magnitude"
                    if OCs.shape[0] > self.NoC_seq:
                        spec_magnitudes = np.zeros(OCs.shape[0])
                        for n,spec in enumerate(OCs_spectrum):
                            spec_magnitudes[n] = spec.max()
                        top_k_indices = np.argsort(spec_magnitudes)[-self.NoC_seq:]
                        top_k_indices = np.sort(top_k_indices)
                        OCs = OCs[top_k_indices]
                        freqs = freqs[top_k_indices]
                        exact_freqs_amp = exact_freqs_amp[top_k_indices]

            "Reconstruction and NRMSE"
            seq_recon = np.sum(OCs,axis=0)
            mse = np.mean((sequence - seq_recon) ** 2)
            rmse = np.sqrt(mse)
            nrmse = rmse / (np.max(sequence) - np.min(sequence))

            OCs_spectrum = np.zeros((OCs.shape[0],self.nfft//2+1))
            for n in range(OCs.shape[0]):
                f,OCs_spectrum[n] = scipy.signal.welch(OCs[n], fs=self.fs, window='hann', nperseg=66, noverlap=10, nfft=self.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')

            "Check number of obtained OCs and apply aggregation (more than specified) or padding (less than specified) if needed"
            if OCs.shape[0] > self.NoC_seq:
                "If more components are found than specified, aggregate them according to the specified strategy"
                #high_to_low, low_to_high, equally_distribute
                if self.group_OCs_by_seq == 'low_freqs_first':
                    if self.decomp_to_perform == 'emd':
                        #Superpose last components (low frequencies)
                        OCs[self.NoC_seq-1] = OCs[self.NoC_seq-1:].sum(axis=0)
                        OCs = OCs[:self.NoC_seq]
                    else:
                        #Superpose first components (low frequencies)
                        old_shape = OCs.shape[0]
                        OCs[0] = OCs[:(OCs.shape[0] - (self.NoC_seq-1))].sum(axis=0)
                        OCs = np.delete(OCs,range(1,OCs.shape[0]- (self.NoC_seq-1)),0)
                    assert OCs.shape[0] == self.NoC_seq
                elif self.group_OCs_by_seq == 'high_freqs_first':             
                    if self.decomp_to_perform == 'emd':
                        #Superpose first components (high frequencies)
                        OCs[0] = OCs[:(OCs.shape[0] - (self.NoC_seq-1))].sum(axis=0)
                        OCs = np.delete(OCs,range(1,OCs.shape[0]- (self.NoC_seq-1)),0)
                    else:                      
                        #Superpose last components (high frequencies)
                        OCs[self.NoC_seq-1] = OCs[self.NoC_seq-1:].sum(axis=0)
                        OCs = OCs[:self.NoC_seq]
                    assert OCs.shape[0] == self.NoC_seq
                elif self.group_OCs_by_seq == 'equally_distribute':    
                    #Cluster components with neighbours in equally populated groups    
                    #This functions similar to high_freqs_first starting grouping from high frequencies                            
                    #In EMD, low frequencies are in the last components, and a lot of spurious low frequency components are found
                    #so its a good strategy to superpose low frequencies first
                    new_OCs = []
                    new_exact_freqs = []
                    #Slightly higher than NoC is same as high_freqs_first
                    #Significantly higher (> .5) gives the equal distribution
                    step = int(round(OCs.shape[0]/self.NoC_seq))
                    start_earlier = False
                    for i in range(self.NoC_seq):
                        if start_earlier:
                            start_idx = end_idx
                        else:
                            start_idx = i*step
                        end_idx = (i+1)*step if i != self.NoC_seq -1 else OCs.shape[0] 
                        if end_idx == OCs.shape[0] and i != self.NoC_seq -1:
                            end_idx = end_idx - (end_idx-start_idx)//2 
                            start_earlier = True
                        new_OCs.append(OCs[start_idx:end_idx].sum(axis=0))
                    OCs = np.array(new_OCs)    
                    assert OCs.shape[0] == self.NoC_seq
                    if self.decomp_to_perform == 'emd':
                        #Reverse order of OCs to correspond to lowest->highest
                        OCs = OCs[::-1,:] 
            elif OCs.shape[0] < self.NoC_seq:
                "If fewer components are found than specified, we need to find approximately which component is missing and pad them with zeros"
                "To do this we need to infer based on the detected frequencies of existing components, which is more likely to be missing"
                "If it cannot be inferred, then we pad the last components with zeros (highest frequencies)"
                #Pad with zeros
                if not OCs.any():
                    #If 0 components are found (empty OCs)
                    component_indices[u,:] = torch.tensor([0]*self.NoC_seq)
                    OCs = np.zeros((self.NoC_seq,sequence_length))
                    ortho_num = 0
                else:
                    if len(OCs.shape) > 1:
                        ortho_num = OCs.shape[0]
                    else:
                        OCs = OCs.reshape(1,-1)
                        ortho_num = 1
                    if len(to_delete) > 0:
                        to_delete = to_delete[:self.NoC_seq-OCs.shape[0]]                                       
                        if len(np.intersect1d(to_delete,self.classify_omegas(exact_freqs))) > 0 and len(np.intersect1d(to_delete,self.classify_omegas(np.mean(freqs,axis=1)))) == 0 and self.decomp_to_perform != 'emd': 
                            #If this happens it means that the deletion will be wrong 
                            # and we need to correct it by deleting the lowest amplitude component
                            if not (self.NoC_seq - OCs.shape[0] > len(to_delete)):
                                to_delete = []
                            min_amp = 1000
                            for n in range(exact_freqs_amp.shape[0]): #range(self.NoC_seq):
                                if exact_freqs_amp[n] < min_amp:
                                    min_amp = exact_freqs_amp[n]
                                    min_freq = n
                            to_delete.append(min_freq)
                        if min(to_delete) > OCs.shape[0]:
                            to_delete = [d - (min(to_delete) - OCs.shape[0]) for d in to_delete]

                        for n in to_delete:
                            if OCs.shape[0] >= self.NoC_seq:
                                break
                            OCs = np.insert(OCs,n,np.zeros(OCs.shape[-1]),axis=0) #sequence_length
                        while OCs.shape[0] < self.NoC_seq:
                            OCs = np.insert(OCs,OCs.shape[0],np.zeros(OCs.shape[-1]),axis=0) #sequence_length
                            to_delete.append(OCs.shape[0]-1)
                    else:
                        groups = self.classify_omegas(exact_freqs)
                        for n in range(self.NoC_seq):
                            if n not in groups:
                                to_delete.append(n)
                        if len(to_delete) > self.NoC_seq - OCs.shape[0]:
                            if len(np.unique(groups)) < len(groups) and (to_delete > groups.max()).all(): 
                                to_delete = to_delete[self.NoC_seq-OCs.shape[0]:]
                            else:
                                to_delete = to_delete[:self.NoC_seq-OCs.shape[0]]
                        if len(to_delete) > 0:
                            if min(to_delete) > OCs.shape[0]:
                                to_delete = [d - (min(to_delete) - OCs.shape[0]) for d in to_delete]

                            for n in to_delete:
                                if OCs.shape[0] >= self.NoC_seq:
                                    break
                                OCs = np.insert(OCs,n,np.zeros(len(sequence)),axis=0)
                                groups = np.insert(groups,n,-1)
                        while OCs.shape[0] < self.NoC_seq:
                            OCs = np.insert(OCs,OCs.shape[0],np.zeros(OCs.shape[-1]),axis=0) #sequence_length
                            to_delete.append(OCs.shape[0]-1)
                    component_indices[u,to_delete] = torch.tensor([0]*(len(to_delete))).type_as(component_indices) 

            #Calculate Orthogonality Index - Input time series domain
            if 'ortho_num' not in locals():
                correlogram = np.zeros((self.NoC_seq,self.NoC_seq))
                for o in range(self.NoC_seq):
                    for r in range(self.NoC_seq):
                        correlogram[o,r] = np.cov(OCs[o],OCs[r])[0,1] / (np.std(OCs[o])*np.std(OCs[r]))
            else:
                correlogram = np.zeros((ortho_num,ortho_num))
                for o in range(ortho_num):
                    for r in range(ortho_num):
                        correlogram[o,r] = np.cov(OCs[o],OCs[r])[0,1] / (np.std(OCs[o])*np.std(OCs[r]))
                del ortho_num

            correlograms.append(correlogram)
            x_MR = np.concatenate((sequence.reshape(1,len(sequence)),OCs),axis=0)

            "Pad x_MR back to max length"
            if x_MR.shape[1] < sequence_length:
                x_MR = np.concatenate((x_MR,np.zeros((self.NoC_seq+1,sequence_length-x_MR.shape[1]))),axis=1)
            
            masked_x[:,u,:] = x_MR 

        return masked_x,component_indices, nrmse, correlograms

    def calculate_rf_stride(self,input_size, kernels, strides):
        """Utility to calculate receptive field size and total stride given input size, kernel sizes and strides of convolutional layers."""
        size = input_size
        cum_stride = 1
        rf_size = 1
        
        for i, (k, s) in enumerate(zip(kernels, strides)):
            # Calculate output size
            output_size = (size - k) // s + 1
            
            # Calculate receptive field
            rf_size = rf_size + (k - 1) * cum_stride
            
            # Update cumulative stride
            cum_stride *= s
            
            print(f"Layer {i+1}:")
            print(f"Output size: {output_size}")
            print(f"Cumulative stride: {cum_stride}")
            print(f"Receptive field: {rf_size}")
            print()
            
            size = output_size
        
        return size, cum_stride, rf_size
    
    def forward(self, x, mask_time_indices=None,attention_mask=None, remove_silence = True,freq_labels = None):
        if hasattr(x, 'device') and type(x) == torch.Tensor:
            device = x.device
            if device.type == 'cuda':
                x = x.cpu().numpy()
        assert type(x) == np.ndarray

        """
        To get output frames of 199 with a RFS of 400 samples:
        7 layers: kernel [10,3,3,3,3,2,2], stride [5,2,2,2,2,2,2]
        5 layers: kernel [10,5,5,3,3], stride [5,4,4,2,2]
        3 layers: kernel [10,8,8], stride [10,8,4]
        """
        """
        input_size = 64000
        conv_kernel = [10,5,3,3,3] #[10,3,3,3,3]
        conv_stride = [5,4,2,2,2] # [5,3,3,2,2] gives stride 180, RFS 320, output 1

        final_size, total_stride, total_rf = self.calculate_rf_stride(input_size, conv_kernel, conv_stride)
        #print(f"Final output size: {final_size}")
        #print(f"Total stride: {total_stride}")
        #print(f"Total receptive field: {total_rf}")
        """

        if self.use_notch_filter:
            x = self.notch_filter_power_line_noise(x)

        peak_det_intervals = self.get_peak_detection_intervals()

        "Frame-level decomposition"
        if self.frame_decomp:
            masked_x,mask_time_indices,start_indices, reconstruction_NRMSEs, correlograms, component_indices, freq_labels = self.decomp_mask(x,mask_time_indices,attention_mask,remove_silence,freq_labels,peak_det_intervals)

            if 'device' not in locals():
                masked_x = torch.tensor(masked_x)
            else:
                masked_x = torch.tensor(masked_x, device = device)
            
            for c,corr in enumerate(correlograms):
                if corr.shape[0] != self.NoC:
                    new_corr = np.empty((self.NoC,self.NoC))
                    new_corr[:] = np.nan
                    new_corr[:corr.shape[0],:corr.shape[1]] = corr
                    correlograms[c] = new_corr
            correlograms_torch = torch.abs(torch.tensor(np.array(correlograms),device = masked_x.device))
            avg_correlogram = correlograms_torch.nanmean(dim=0)

            #Count how many nan's in the correlograms
            corr_elements = [np.triu(np.array(i),k=1) for i in correlograms]
            iso_corr_elements = np.array([c.ravel()[np.flatnonzero(c)] for c in corr_elements])
            if not (iso_corr_elements != iso_corr_elements).any():
                nan_percent= [np.float64(0)]
            else:
                nan_percent = [np.sum(iso_corr_elements != iso_corr_elements) / iso_corr_elements.size]
        
        if self.sequence_decomp:
            "Sequence-level decomposition"
            seq_x,component_indices_seq, reconstruction_NRMSEs_seq, correlograms_seq = self.sequence_decomp(x,attention_mask,peak_det_intervals)

            if 'device' not in locals():
                seq_x = torch.tensor(seq_x)
            else:
                seq_x = torch.tensor(seq_x, device = device)
            
            for c,corr in enumerate(correlograms_seq):
                if corr.shape[0] != self.NoC_seq:
                    new_corr = np.empty((self.NoC_seq,self.NoC_seq))
                    new_corr[:] = np.nan
                    new_corr[:corr.shape[0],:corr.shape[1]] = corr
                    correlograms_seq[c] = new_corr
            correlograms_seq_torch = torch.abs(torch.tensor(np.array(correlograms_seq),device = seq_x.device))
            avg_correlogram_seq = correlograms_seq_torch.nanmean(dim=0)
        
        "Return"
        if self.frame_decomp and not self.seq_decomp:
           decomposition_outcome = {"frame": masked_x,"sequence": None}
           component_indices = {"frame": component_indices,"sequence": None}
           return decomposition_outcome, mask_time_indices, start_indices, attention_mask, reconstruction_NRMSEs, None, avg_correlogram, correlograms_torch, None, None, component_indices, nan_percent, freq_labels
        elif self.seq_decomp and not self.frame_decomp:
            decomposition_outcome = {"frame": None,"sequence": seq_x}
            component_indices = {"frame": None,"sequence": component_indices_seq}
            return decomposition_outcome, None, None, attention_mask, None, reconstruction_NRMSEs_seq, None, None, avg_correlogram_seq, correlograms_seq_torch, component_indices, None, None
        elif self.frame_decomp and self.seq_decomp:
            decomposition_outcome = {"frame": masked_x,"sequence": seq_x}
            component_indices = {"frame": component_indices,"sequence": component_indices_seq}
            return decomposition_outcome, mask_time_indices, start_indices, attention_mask, reconstruction_NRMSEs, reconstruction_NRMSEs_seq, avg_correlogram, correlograms_torch, avg_correlogram_seq, correlograms_seq_torch, component_indices, nan_percent, freq_labels