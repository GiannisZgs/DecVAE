#!/usr/bin/env python
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

"""Simulation of human spoken vowel sounds.
This script generates an utterance of 0.1 seconds for every speaker (vocal_tract_factor) in the SimVowels dataset
for visualization purposes. This SimVowels subset is used to obtain Supplementary Information Figures 4 and 6.
"""



import numpy as np
from scipy.signal import butter, lfilter, welch
import os
import json
import gzip
import random
from feature_extraction import extract_mel_spectrogram

#Generative Factors
#1. Vocal Tract Length (Speakers)
#2. Vowels ()

# Constants and Parameters
SAVE_DIR = "../data_for_figures/"  #"/home/student3/Documents/IoannisZiogas/sim_vowels/"  

#"""
CANONICAL_FORMANT_FREQUENCIES = {
    'a':  [710, 1100, 2540],
    'e': [550, 1770, 2490],
    'I': [400, 1920, 2560],
    'aw': [590, 880, 2540],
    'u': [310, 870, 2250],
}
"""
CANONICAL_FORMANT_FREQUENCIES = {
    'i': [280, 2250, 2890],
    'I': [400, 1920, 2560],
    'e': [550, 1770, 2490],
    'ae': [690, 1660, 2490],
    'a': [710, 1100, 2540],
    'aw': [590, 880, 2540],
    'y': [450, 1030, 2380],
    'u': [310, 870, 2250],
}
"""
#A Course in Phonetics 6th Edition - Peter Ladefoged, Keith Johnson
#i: as in heed, he, bead, heat, keyed - 0
#I: as in hid, bid, hit, kid - 1
#E - epsilon: as in head, bed - 2
#ae - ash: as in had, bad, hat, cad - 3
#a - script a: as in hard, bard, heart, card - 4
#aw - open o: hawed, haw, bawd, cawed - 5
#y - upsilon (inverse omega): hood, could - 6
#u - lowercase u: who'd, who, booed, hoot, cooed - 7

VOWEL_TO_INT = {vowel: idx for idx, vowel in enumerate(CANONICAL_FORMANT_FREQUENCIES.keys())}
FORMANT_BANDWIDTHS = [200, 200, 200]
AMPLITUDES = [1.0, 0.5, 0.3]
F0_BASE = 120
F_S = 16000
DURATION = 0.1  # Duration in seconds for each vowel
SNR_DB = 100
mel_norm = None

NUM_FFT_BINS = 2048 

# Bandpass filter for creating wide-band noise
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y

def scale_formants(vowel, vt_factor):
    canonical_freqs = CANONICAL_FORMANT_FREQUENCIES[vowel]
    scaled_freqs = [f / vt_factor for f in canonical_freqs]
    return scaled_freqs

def add_gaussian_noise(signal, snr_db,tolerance=0.01) -> np.ndarray:
    """
    Adds Gaussian white noise to a signal based on a specified SNR in dB.
    """

    signal = signal / np.max(np.abs(signal))
    
    # Step 1: Calculate signal power
    signal_power = np.mean(signal ** 2)
    
    # Step 2: Calculate noise power based on desired SNR
    snr_linear = 10 ** (snr_db / 10)  # Convert dB to linear scale
    noise_power = signal_power / snr_linear
    
    for _ in range(20):
        # Step 3: Generate white Gaussian noise with calculated noise power
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        
        # Step 4: Add noise to the original signal
        noisy_signal = signal + noise

        # Check SNR
        actual_noise_power = np.mean((noisy_signal - signal) ** 2)
        actual_snr_db = 10 * np.log10(signal_power / actual_noise_power)

        if abs(actual_snr_db - snr_db) <= tolerance:
            break  # Stop if within tolerance

        # Adjust noise power based on measured SNR
        if snr_db > actual_snr_db:
            noise_power /= 10 ** (np.abs(snr_db - actual_snr_db) / 10)
        else:
            noise_power *= 10 ** (np.abs(snr_db - actual_snr_db) / 10)

    if abs(actual_snr_db - snr_db) > 1.5:
        raise ValueError(f"SNR mismatch: {actual_snr_db} dB")
    
    #noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))

    return noisy_signal

def generate_periodic_excitation(f0, duration, sample_rate):
    """Generate a periodic pulse train at frequency f0."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    pulse_train = np.zeros_like(t)
    pulse_interval = int(sample_rate / f0)
    for i in range(0, len(t), pulse_interval):
        # Insert a burst of white noise in each pulse interval
        pulse_train[i:i + min(pulse_interval,len(t)-i)] = np.random.normal(0, 1, min(pulse_interval,len(t)-i))
   
    pulse_train = bandpass_filter(pulse_train, lowcut=65, highcut=F_S/2-1, fs=F_S)
    # FFT of pulse train to get periodic excitation spectrum
    excitation_spectrum = np.fft.fft(pulse_train, n=NUM_FFT_BINS)
    return excitation_spectrum

def generate_time_domain_signal(vowel, vt_factor, snr_db=0):
    scaled_freqs = scale_formants(vowel, vt_factor)
    f0 = F0_BASE / vt_factor
    t = np.linspace(0, DURATION, int(F_S * DURATION), endpoint=False)
    freq_axis = np.linspace(0,F_S,NUM_FFT_BINS)
    signal = np.zeros_like(t)

    white_noise = np.random.normal(0, 1, len(t))
    #formant_spectrum = np.zeros(NUM_FFT_BINS, dtype=complex)
    formant_waves = []
    for freq, amp, bw in zip(scaled_freqs, AMPLITUDES, FORMANT_BANDWIDTHS):
        # Find the closest FFT bin for each formant frequency
        formant_wave = amp * bandpass_filter(white_noise, lowcut=freq - bw/2, highcut=freq + bw/2, fs=F_S)#amp * np.sin(2 * np.pi * freq * t)
        signal += formant_wave
        formant_waves.append(formant_wave)
        # Assign amplitude with random phase

    formant_spectrum = np.fft.fft(signal,n=NUM_FFT_BINS)

    excitation_spectrum = generate_periodic_excitation(f0, DURATION, F_S)

    # Multiply formant and excitation spectra
    convolved_spectrum = formant_spectrum * excitation_spectrum

    # Inverse FFT to obtain the time-domain signal
    convolved_signal = np.fft.ifft(convolved_spectrum).real[:len(t)]

    f,Pxx = welch(convolved_signal, fs=F_S, window='hann', nperseg=2048, noverlap=512, nfft=2048, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean') 
    Pxx = Pxx/np.max(Pxx)
    log_Pxx = 10*np.log10(Pxx + 1e-9)
    
    if snr_db is not None:
        return add_gaussian_noise(convolved_signal,snr_db), f, Pxx, log_Pxx, formant_waves
    else:
        return convolved_signal


def generate_sequence(figure_data_dict):
    audio = []
    vowels = []
    speaker_vocal_tract_factor = []
    vocal_tract_ranges = np.concatenate((np.array(np.arange(0.7,1.29,0.01)).reshape(-1,1),np.array(np.arange(0.71,1.3,0.01)).reshape(-1,1)),axis = 1)
    
    if len(figure_data_dict) == 0:
        for vt in vocal_tract_ranges:
            vt_factor = str(np.round(np.mean(vt),decimals = 3))
            for vowel in CANONICAL_FORMANT_FREQUENCIES.keys():
                figure_data_dict[vowel + "_" + vt_factor] =  {'freq_axis': [],
                                    'log_spectral_density': [],
                                    'spectral_density': [],
                                    'time_domain_signal': [],
                                    'formant_1_waves': [],
                                    'formant_2_waves': [],
                                    'formant_3_waves': [],
                                    'mel_filterbank_energies': [],
                                    }
    for i in range(vocal_tract_ranges.shape[0]):
        selected_range = vocal_tract_ranges[i,:]
        vt_factor = random.uniform(*selected_range)
        vt_factor = np.round(np.mean(selected_range),decimals = 3)
        audio_utt, vowels_utt, vt_factors_utt = [], [], []
        #for j in range(sequence_length):
        for j in range(5):
            vowel = list(CANONICAL_FORMANT_FREQUENCIES.keys())[j]
            time_domain_signal, f, Pxx, log_Pxx, formant_waves = generate_time_domain_signal(vowel, vt_factor,SNR_DB)
            audio_utt.append(time_domain_signal)
            vowels_utt.append(VOWEL_TO_INT[vowel])
            if DURATION == 1:
                figure_data_dict[vowel + "_" + str(vt_factor)]['freq_axis'] = list(f)
                figure_data_dict[vowel + "_" + str(vt_factor)]['spectral_density'] = list(Pxx)
                figure_data_dict[vowel + "_" + str(vt_factor)]['log_spectral_density'] = list(log_Pxx)
                mel_energies,spec_max = extract_mel_spectrogram(time_domain_signal,F_S,n_mels=80, n_fft=int(len(time_domain_signal)/4), hop_length=int((len(time_domain_signal) + 1)/4), normalize=mel_norm,feature_length=len(time_domain_signal))
                figure_data_dict[vowel + "_" + str(vt_factor)]['mel_filterbank_energies'] = mel_energies.squeeze().tolist()
                mel_energies_f1, _ = extract_mel_spectrogram(formant_waves[0],F_S,n_mels=80, n_fft=int(len(formant_waves[0])/4), hop_length=int((len(formant_waves[0]) + 1)/4), normalize=mel_norm,feature_length=len(formant_waves[0]), ref = spec_max)
                figure_data_dict[vowel + "_" + str(vt_factor)]['mel_filterbank_energies_f1'] = mel_energies_f1.squeeze().tolist()
                mel_energies_f2, _ = extract_mel_spectrogram(formant_waves[1],F_S,n_mels=80, n_fft=int(len(formant_waves[1])/4), hop_length=int((len(formant_waves[1]) + 1)/4), normalize=mel_norm,feature_length=len(formant_waves[1]), ref = spec_max)
                figure_data_dict[vowel + "_" + str(vt_factor)]['mel_filterbank_energies_f2'] = mel_energies_f2.squeeze().tolist()
                mel_energies_f3, _ = extract_mel_spectrogram(formant_waves[2],F_S,n_mels=80, n_fft=int(len(formant_waves[2])/4), hop_length=int((len(formant_waves[2]) + 1)/4), normalize=mel_norm,feature_length=len(formant_waves[2]), ref = spec_max)
                figure_data_dict[vowel + "_" + str(vt_factor)]['mel_filterbank_energies_f3'] = mel_energies_f3.squeeze().tolist()

            if DURATION == 0.1:
                figure_data_dict[vowel + "_" + str(vt_factor)]['time_domain_signal'] = list(time_domain_signal)
                figure_data_dict[vowel + "_" + str(vt_factor)]['formant_1_waves'] = list(formant_waves[0])
                figure_data_dict[vowel + "_" + str(vt_factor)]['formant_2_waves'] = list(formant_waves[1])
                figure_data_dict[vowel + "_" + str(vt_factor)]['formant_3_waves'] = list(formant_waves[2])
                

    return figure_data_dict

def main():
    
    fname_train = os.path.join(SAVE_DIR, "sim_vowels_figures.json.gz")
    if not os.path.exists(fname_train):
        data_dict = generate_sequence(figure_data_dict={})
        with gzip.open(fname_train, "wt") as f:
            json.dump(data_dict, f)
        print("Succesfully saved")
    else:
        with gzip.open(fname_train, "rt") as f:
            data_dict = json.load(f)
        print("Succesfully loaded")
        data_dict = generate_sequence(data_dict)
        with gzip.open(fname_train, "wt") as f:
            json.dump(data_dict, f)
        print("Succesfully saved")

if __name__ == "__main__":
    main()