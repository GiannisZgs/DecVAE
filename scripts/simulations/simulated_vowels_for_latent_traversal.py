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
This script generates controlled pairs of generative factors vowel and speaker to use for the latent traversals experiments.
The data generated here are used to obtain Supplementary Information Figure 18.
"""

import numpy as np
from scipy.signal import butter, lfilter
import os
import json
import gzip

#Fixed vowel traversal: for a single vowel, generate data from all speakers
#Fixed speaker traversal: for a single speaker, generate data from all vowels

SAVE_DIR = os.path.join("..", "latent_traversal_data","sim_vowels_raw")  

CANONICAL_FORMANT_FREQUENCIES = {
    'a':  [710, 1100, 2540],
    'e': [550, 1770, 2490],
    'I': [400, 1920, 2560],
    'aw': [590, 880, 2540],
    'u': [310, 870, 2250],
}

CANONICAL_FORMANT_FREQUENCIES_EXTENDED = {
    'i': [280, 2250, 2890],
    'I': [400, 1920, 2560],
    'e': [550, 1770, 2490],
    'ae': [690, 1660, 2490],
    'a': [710, 1100, 2540],
    'aw': [590, 880, 2540],
    'y': [450, 1030, 2380],
    'u': [310, 870, 2250],
}


VOWEL_TO_INT = {vowel: idx for idx, vowel in enumerate(CANONICAL_FORMANT_FREQUENCIES.keys())}
VOWEL_TO_INT_EXTENDED = {vowel: idx for idx, vowel in enumerate(CANONICAL_FORMANT_FREQUENCIES_EXTENDED.keys())}
FORMANT_BANDWIDTHS = [200, 200, 200]
AMPLITUDES = [1.0, 0.5, 0.3]
F0_BASE = 120
F_S = 16000
DURATION = 0.025 # Duration in seconds for each vowel
SNR_DB = 15

ANALYSIS = 'vowel'
SPEAKER_GRAN = 0.01
SPEAKER_RANGE = [0.7,1.29]
if ANALYSIS == "vowel":
    "Latent traversal analysis for fixed vowels"
    FRAME_SEQUENCES = [int(np.diff(SPEAKER_RANGE)/SPEAKER_GRAN)] 

NUM_FFT_BINS = 2048 #2048
UTTERANCE_DUR = 4 # in seconds duration of "utterance"

# Bandpass filter for creating wide-band noise
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y

def scale_formants(formant_frequencies,vowel, vt_factor):
    canonical_freqs = formant_frequencies[vowel]
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
    
    for _ in range(100):
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

def generate_time_domain_signal(formant_frequencies,vowel, vt_factor, snr_db=0):
    scaled_freqs = scale_formants(formant_frequencies,vowel, vt_factor)
    f0 = F0_BASE / vt_factor
    t = np.linspace(0, DURATION, int(F_S * DURATION), endpoint=False)
    freq_axis = np.linspace(0,F_S,NUM_FFT_BINS)
    signal = np.zeros_like(t)

    white_noise = np.random.normal(0, 1, len(t))
    formant_waves = []
    for freq, amp, bw in zip(scaled_freqs, AMPLITUDES, FORMANT_BANDWIDTHS):
        formant_wave = amp * bandpass_filter(white_noise, lowcut=freq - bw/2, highcut=freq + bw/2, fs=F_S)#amp * np.sin(2 * np.pi * freq * t)
        signal += formant_wave
        formant_waves.append(formant_wave)

    formant_spectrum = np.fft.fft(signal,n=NUM_FFT_BINS)

    excitation_spectrum = generate_periodic_excitation(f0, DURATION, F_S)

    # Multiply formant and excitation spectra
    convolved_spectrum = formant_spectrum * excitation_spectrum

    # Inverse FFT to obtain the time-domain signal
    convolved_signal = np.fft.ifft(convolved_spectrum).real[:len(t)]
    
    if snr_db is not None:
        return add_gaussian_noise(convolved_signal,snr_db)
    else:
        return convolved_signal

def generate_fixed_vowels_sequence(formant_frequencies):
    audio = []
    speakers = []
    
    vocal_tract_range = np.arange(SPEAKER_RANGE[0],SPEAKER_RANGE[1],SPEAKER_GRAN) + SPEAKER_GRAN/2
    vowels = []
    for j in range(len(formant_frequencies.keys())):
        vowel = list(formant_frequencies.keys())[j]
        for i in vocal_tract_range:
            vt_factor = np.round(i,decimals=3)
            time_domain_signal = generate_time_domain_signal(formant_frequencies,vowel, vt_factor,SNR_DB)
            audio.append(time_domain_signal.tolist())
            speakers.append(vt_factor)
            #mel_filterbank_energies, power_spectrum = generate_spectrogram(vowel, vt_factor)
            if len(formant_frequencies) == 5:
                vowels.append(VOWEL_TO_INT[vowel])
            elif len(formant_frequencies) == 8:
                vowels.append(VOWEL_TO_INT_EXTENDED[vowel])


    data_dict = {
        'audio': audio,
        'vowel': vowels,
        'speaker_vocal_tract_factor': speakers
    }

    return data_dict

def generate_fixed_speakers_sequence(formant_frequencies):
    audio = []
    
        
    vocal_tract_range = np.arange(SPEAKER_RANGE[0],SPEAKER_RANGE[1],SPEAKER_GRAN) + SPEAKER_GRAN/2    
    speakers = []
    vowels = []
    for i in vocal_tract_range:
        vt_factor = np.round(i,decimals=3)
        for j in range(len(formant_frequencies.keys())):
            vowel = list(formant_frequencies.keys())[j]
            time_domain_signal = generate_time_domain_signal(formant_frequencies,vowel, vt_factor,SNR_DB)
            audio.append(time_domain_signal.tolist())
            if len(formant_frequencies) == 5:
                vowels.append(VOWEL_TO_INT[vowel])
            elif len(formant_frequencies) == 8:
                vowels.append(VOWEL_TO_INT_EXTENDED[vowel])
            speakers.append(vt_factor)
    

    data_dict = {
        'audio': audio,
        'vowel': vowels,
        'speaker_vocal_tract_factor': speakers
    }

    return data_dict


def main():
    
    os.makedirs(os.path.join(SAVE_DIR,"fixed_vowels"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR,"fixed_speakers"), exist_ok=True)

    fname_fixed_vowels_train = os.path.join(SAVE_DIR,"fixed_vowels","fixed_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES))+ "_SNR_" + str(SNR_DB) + "_4s_train.json.gz")
    fname_fixed_vowels_dev = os.path.join(SAVE_DIR,"fixed_vowels","fixed_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES))+ "_SNR_" + str(SNR_DB) + "_4s_dev.json.gz")
    fname_fixed_vowels_test = os.path.join(SAVE_DIR,"fixed_vowels","fixed_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES))+ "_SNR_" + str(SNR_DB) + "_4s_test.json.gz")

    fname_fixed_extended_vowels_train = os.path.join(SAVE_DIR,"fixed_vowels","fixed_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES_EXTENDED))+ "_SNR_" + str(SNR_DB) + "_4s_train.json.gz")
    fname_fixed_extended_vowels_dev = os.path.join(SAVE_DIR,"fixed_vowels","fixed_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES_EXTENDED))+ "_SNR_" + str(SNR_DB) + "_4s_dev.json.gz")
    fname_fixed_extended_vowels_test = os.path.join(SAVE_DIR,"fixed_vowels","fixed_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES_EXTENDED))+ "_SNR_" + str(SNR_DB) + "_4s_test.json.gz")

    fname_fixed_speakers_free_vowels_train = os.path.join(SAVE_DIR,"fixed_speakers","fixed_speakers_free_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES))+ "_SNR_" + str(SNR_DB) + "_4s_train.json.gz")
    fname_fixed_speakers_free_vowels_dev = os.path.join(SAVE_DIR,"fixed_speakers","fixed_speakers_free_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES))+ "_SNR_" + str(SNR_DB) + "_4s_dev.json.gz")
    fname_fixed_speakers_free_vowels_test = os.path.join(SAVE_DIR,"fixed_speakers","fixed_speakers_free_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES))+ "_SNR_" + str(SNR_DB) + "_4s_test.json.gz")

    fname_fixed_speakers_free_extended_vowels_train = os.path.join(SAVE_DIR,"fixed_speakers","fixed_speakers_free_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES_EXTENDED))+ "_SNR_" + str(SNR_DB) + "_4s_train.json.gz")
    fname_fixed_speakers_free_extended_vowels_dev = os.path.join(SAVE_DIR,"fixed_speakers","fixed_speakers_free_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES_EXTENDED))+ "_SNR_" + str(SNR_DB) + "_4s_dev.json.gz")
    fname_fixed_speakers_free_extended_vowels_test = os.path.join(SAVE_DIR,"fixed_speakers","fixed_speakers_free_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES_EXTENDED))+ "_SNR_" + str(SNR_DB) + "_4s_test.json.gz")

    if not os.path.exists(fname_fixed_vowels_train):
        fixed_vowels_free_speakers = generate_fixed_vowels_sequence(formant_frequencies=CANONICAL_FORMANT_FREQUENCIES)
        with gzip.open(fname_fixed_vowels_train, "wt") as f:
            json.dump(fixed_vowels_free_speakers, f)   
        with gzip.open(fname_fixed_vowels_dev, "wt") as f:
            json.dump(fixed_vowels_free_speakers, f)   
        with gzip.open(fname_fixed_vowels_test, "wt") as f:
            json.dump(fixed_vowels_free_speakers, f)   

    if not os.path.exists(fname_fixed_extended_vowels_train):
        fixed_extended_vowels_free_speakers = generate_fixed_vowels_sequence(formant_frequencies=CANONICAL_FORMANT_FREQUENCIES_EXTENDED)
        with gzip.open(fname_fixed_extended_vowels_train, "wt") as f:
            json.dump(fixed_extended_vowels_free_speakers, f)    
        with gzip.open(fname_fixed_extended_vowels_dev, "wt") as f:
            json.dump(fixed_extended_vowels_free_speakers, f)   
        with gzip.open(fname_fixed_extended_vowels_test, "wt") as f:
            json.dump(fixed_extended_vowels_free_speakers, f)    

    if not os.path.exists(fname_fixed_speakers_free_vowels_train):
        fixed_speaker_free_vowels = generate_fixed_speakers_sequence(formant_frequencies=CANONICAL_FORMANT_FREQUENCIES)
        with gzip.open(fname_fixed_speakers_free_vowels_train, "wt") as f:
            json.dump(fixed_speaker_free_vowels, f)
        with gzip.open(fname_fixed_speakers_free_vowels_dev, "wt") as f:
            json.dump(fixed_speaker_free_vowels, f)
        with gzip.open(fname_fixed_speakers_free_vowels_test, "wt") as f:
            json.dump(fixed_speaker_free_vowels, f)

    if not os.path.exists(fname_fixed_speakers_free_extended_vowels_train):
        fixed_speaker_free_extended_vowels = generate_fixed_speakers_sequence(formant_frequencies=CANONICAL_FORMANT_FREQUENCIES_EXTENDED)
        with gzip.open(fname_fixed_speakers_free_extended_vowels_train, "wt") as f:
            json.dump(fixed_speaker_free_extended_vowels, f)
        with gzip.open(fname_fixed_speakers_free_extended_vowels_dev, "wt") as f:
            json.dump(fixed_speaker_free_extended_vowels, f)
        with gzip.open(fname_fixed_speakers_free_extended_vowels_test, "wt") as f:
            json.dump(fixed_speaker_free_extended_vowels, f)    


if __name__ == "__main__":
    main()