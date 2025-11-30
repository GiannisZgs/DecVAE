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
This script generates the SimVowels dataset.
Implementation inspired by:
Boulianne, Gilles. "A study of inductive biases for unsupervised speech representation learning." 
IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2781-2795.
"""


import numpy as np
from scipy.signal import butter, lfilter
import os
import json
import gzip
import random


#Generative Factors
#1. Vocal Tract Length (Speakers)
#2. Vowels ()

#A Course in Phonetics 6th Edition - Peter Ladefoged, Keith Johnson
#i: as in heed, he, bead, heat, keyed - 0
#I: as in hid, bid, hit, kid - 1
#E - epsilon: as in head, bed - 2
#ae - ash: as in had, bad, hat, cad - 3
#a - script a: as in hard, bard, heart, card - 4
#aw - open o: hawed, haw, bawd, cawed - 5
#y - upsilon (inverse omega): hood, could - 6
#u - lowercase u: who'd, who, booed, hoot, cooed - 7

# Constants and Parameters
SAVE_DIR = os.path.join("..", "sim_vowels")  

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

VOWEL_TO_INT = {vowel: idx for idx, vowel in enumerate(CANONICAL_FORMANT_FREQUENCIES.keys())}
FORMANT_BANDWIDTHS = [200, 200, 200]
AMPLITUDES = [1.0, 0.5, 0.3]
F0_BASE = 120
F_S = 16000
DURATION = 0.1  # Duration in seconds for each vowel
SNR_DB = 15

NUM_FFT_BINS = 2048 
UTTERANCE_DUR = 4 # in seconds duration of "utterance"
FRAME_SEQUENCES = [int(UTTERANCE_DUR / DURATION)] 
TRAINING_SET_SIZE = 4000 
DEVELOPMENT_SET_SIZE = 500 
TEST_SET_SIZE = 300
PARTS = 1

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

    # Generate band-limited noise as wide-band excitation
    white_noise = np.random.normal(0, 1, len(t))
    formant_waves = []
    for freq, amp, bw in zip(scaled_freqs, AMPLITUDES, FORMANT_BANDWIDTHS):
        formant_wave = amp * bandpass_filter(white_noise, lowcut=freq - bw/2, highcut=freq + bw/2, fs=F_S)
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

def generate_sequence(sequence_length, set_type, set_size):
    audio = []
    vowels = []
    speaker_vocal_tract_factor = []
    if set_type == 'train':
        vocal_tract_ranges = [(0.7,0.73),(0.78,0.81),(0.82,0.85),(0.86,0.89),(0.94,0.97),(1.02,1.05),(1.1,1.13),(1.14,1.17),(1.18,1.21),(1.26,1.29)]
    elif set_type == 'dev':
        vocal_tract_ranges = [(0.74,0.75),(0.9,0.91),(0.98,0.99),(1.06,1.07),(1.22,1.23)]
    elif set_type == 'test':
        vocal_tract_ranges = [(0.76,0.77),(0.92,0.93),(1.00,1.01),(1.08,1.09),(1.24,1.25)]

    for i in range(set_size):
        selected_range = random.choice(vocal_tract_ranges)
        vt_factor = random.uniform(*selected_range)
        vt_factor = np.mean(selected_range)
        audio_utt, vowels_utt = [], []
        for j in range(sequence_length):
            vowel = random.choice(list(CANONICAL_FORMANT_FREQUENCIES.keys()))
            time_domain_signal = generate_time_domain_signal(vowel, vt_factor,SNR_DB)
            audio_utt.append(time_domain_signal)
            vowels_utt.append(VOWEL_TO_INT[vowel])

        speaker_vocal_tract_factor.append(vt_factor)
        audio.append(np.concatenate(audio_utt).tolist())
        vowels.append(vowels_utt)

    data_dict = {
        'audio': audio,
        'vowel': vowels,
        'speaker_vocal_tract_factor': speaker_vocal_tract_factor
    }

    return data_dict


def main():

    os.makedirs(SAVE_DIR, exist_ok=True)

    for part in range(1,PARTS+1):
    
        fname_train = os.path.join(SAVE_DIR,"sim_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES))+ "_SNR_" + str(SNR_DB) + "_train_" +str(UTTERANCE_DUR) + "s_part_" + str(part)+ ".json.gz")
        fname_dev = os.path.join(SAVE_DIR,"sim_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES))+ "_SNR_" + str(SNR_DB) + "_dev_" +str(UTTERANCE_DUR) + "s.json.gz")
        fname_test = os.path.join(SAVE_DIR,"sim_vowels_" + str(len(CANONICAL_FORMANT_FREQUENCIES))+ "_SNR_" + str(SNR_DB) + "_test_" +str(UTTERANCE_DUR) + "s.json.gz")
        if not os.path.exists(fname_train):
            data_dict_train = generate_sequence(sequence_length=FRAME_SEQUENCES[-1], set_type="train",set_size=TRAINING_SET_SIZE)
            with gzip.open(fname_train, "wt") as f:
                json.dump(data_dict_train, f)
            print("Succesfully saved training set part ", part)
        if not os.path.exists(fname_dev):
            data_dict_dev = generate_sequence(sequence_length=FRAME_SEQUENCES[-1], set_type="dev", set_size=DEVELOPMENT_SET_SIZE)
            with gzip.open(fname_dev, "wt") as f:
                json.dump(data_dict_dev, f)
            print("Succesfully saved dev set")
        if not os.path.exists(fname_test):
            data_dict_test = generate_sequence(sequence_length=FRAME_SEQUENCES[-1], set_type="test", set_size=TEST_SET_SIZE)
            with gzip.open(fname_test, "wt") as f:
                json.dump(data_dict_test, f)
            print("Succesfully saved test set")

if __name__ == "__main__":
    main()