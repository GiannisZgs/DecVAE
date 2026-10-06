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

"""Characterization of the SimCoupled dataset (Supplementary Information).
Reads the files written by simulated_coupled.py and reports, per split (indep, test):
1. decodability of each factor from single log-Mel frames (linear probe, chance-normalised balanced accuracy),
2. effect profile of each factor over Mel bands,
3. mean log-power spectrum per lag class (gain removed) and per gain class,
4. mean autocovariance per lag class (normalised, gain removed) and per gain class,
5. example waveform snippets per lag class (gain removed) and per gain class, and one example sequence with its labels.
The dataset files hold raw waveforms; log-Mel is computed here only for 1 and 2.
Log-Mel features are computed with librosa directly (80 Mel bands, 25 ms window, 20 ms hop, n_fft 512,
no centring, absolute dB), since feature_extraction.extract_mel_spectrogram fixes win_length = n_fft,
centres frames and normalises per utterance.
Frames that cross a segment boundary are dropped, as discard_label_overlaps does.
"""

import os
import sys
# Add project root to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import librosa
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latent_analysis_utils.subspace_utils import cn_bacc
from simulated_coupled import (
    SAVE_DIR, F_S, SEGMENT_LEN, N_SEGMENTS, LAG_LEVELS, GAIN_LEVELS_DB, split_fname, load_split,
)

# Constants and Parameters
OUT_DIR = os.path.join(SAVE_DIR, "characterization")
SPLITS = ['indep', 'test']
FACTORS = {'lag': LAG_LEVELS, 'gain': GAIN_LEVELS_DB}

N_MELS = 80
WIN_LEN = int(0.025 * F_S)
HOP_LEN = int(0.02 * F_S)
N_FFT = 512
N_BAND_GROUPS = 4

PROBE_TEST_SIZE = 0.25
PROBE_MAX_ITER = 2000
PROBE_SEED = 0

PSD_NPERSEG = 512
ACF_MAX_LAG = 48  # samples, covers the largest coupling lag
WAVE_SNIPPET_LEN = int(0.01 * F_S)
WAVE_SPLIT = 'indep'
GAIN_EXAMPLE_LAG_CLASS = 2  # gain snippets share one lag class, so only the gain differs
EXAMPLE_SEQ = 0
CLASS_COLORS = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']  # ordinal blue, light -> dark
WAVE_COLOR = '#2a78d6'


def log_mel(audio):
    "audio (n_seq, n_samples) -> absolute log-Mel energy in dB (n_seq, n_frames, N_MELS)"
    S = librosa.feature.melspectrogram(
        y=audio, sr=F_S, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN,
        window="hann", center=False, n_mels=N_MELS, power=2.0,
    )
    return np.transpose(10 * np.log10(S + 1e-10), (0, 2, 1))

def frame_segments(n_frames):
    "Segment index of each frame, -1 for frames that cross a segment boundary"
    start = np.arange(n_frames) * HOP_LEN
    seg_start, seg_end = start // SEGMENT_LEN, (start + WIN_LEN - 1) // SEGMENT_LEN
    return np.where(seg_start == seg_end, seg_start, -1)

def frame_dataset(split):
    """Single-frame features with frame-level labels.

    Returns:
        X (n_frames_total, N_MELS), labels {factor: (n_frames_total,)}, groups (n_frames_total,) sequence index
    """
    feats = log_mel(split["audio"].astype(np.float32))
    seg = frame_segments(feats.shape[1])
    keep = seg >= 0
    assert keep.sum() == 4 * N_SEGMENTS, f"{keep.sum()} frames kept per sequence, expected {4 * N_SEGMENTS}"
    n_seq = feats.shape[0]
    X = feats[:, keep].reshape(-1, N_MELS)
    labels = {f: split[f][:, seg[keep]].ravel() for f in FACTORS}
    groups = np.repeat(np.arange(n_seq), keep.sum())
    return X, labels, groups

def probe_score(X, y, groups, n_classes):
    "StandardScaler + LogisticRegression on single frames, sequence-disjoint held-out part"
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=PROBE_TEST_SIZE, random_state=PROBE_SEED).split(X, y, groups))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=PROBE_MAX_ITER))
    clf.fit(X[tr], y[tr])
    return float(cn_bacc(y[te], clf.predict(X[te]), n_classes))

def effect_profile(X, y, n_classes):
    "Per Mel band, variance across classes of the band's mean log-energy"
    means = np.stack([X[y == c].mean(0) for c in range(n_classes)])
    return means.var(0)

def group_shares(profile):
    groups = np.array_split(profile, N_BAND_GROUPS)
    return np.array([g.sum() for g in groups]) / profile.sum()

def segment_matrix(split, remove_gain):
    "0.1 s segments (n_seq * N_SEGMENTS, SEGMENT_LEN), optionally divided by their gain"
    segs = split["audio"].reshape(-1, SEGMENT_LEN)
    if remove_gain:
        segs = segs / 10 ** (split["gain_db"].reshape(-1, 1) / 20)
    return segs

def class_spectra(split, factor, remove_gain):
    "Mean log-power spectrum (dB) per class over 0.1 s segments. Returns freqs (n_freqs,), spectra (n_classes, n_freqs)"
    segs = segment_matrix(split, remove_gain)
    freqs, P = welch(segs, fs=F_S, window="hann", nperseg=PSD_NPERSEG, axis=-1)
    logP = 10 * np.log10(P + 1e-20)
    y = split[factor].ravel()
    return freqs, np.stack([logP[y == c].mean(0) for c in range(len(FACTORS[factor]))])

def class_autocov(split, factor, remove_gain, normalise):
    "Mean autocovariance per class over 0.1 s segments. Returns lags (ACF_MAX_LAG + 1,), autocov (n_classes, ACF_MAX_LAG + 1)"
    segs = segment_matrix(split, remove_gain)
    segs = segs - segs.mean(-1, keepdims=True)
    F = np.fft.rfft(segs, n=2 * SEGMENT_LEN, axis=-1)
    r = np.fft.irfft(np.abs(F) ** 2, axis=-1)[:, :ACF_MAX_LAG + 1] / SEGMENT_LEN
    if normalise:
        r = r / r[:, :1]
    y = split[factor].ravel()
    return np.arange(ACF_MAX_LAG + 1), np.stack([r[y == c].mean(0) for c in range(len(FACTORS[factor]))])

def class_label(factor, level):
    return f"{level:+d} dB" if factor == 'gain' else f"{level} samples"

def save_fig(fig, fname):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, fname + "." + ext), dpi=300)
    plt.close(fig)

def style_axis(ax):
    ax.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

def plot_class_curves(x, curves, factor, xlabel, ylabel, xlim, fname, mark_lags=False):
    "One panel per split, one curve per class"
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(5 * len(SPLITS), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, split_name in zip(axes, SPLITS):
        for c, level in enumerate(FACTORS[factor]):
            ax.plot(x, curves[split_name][c], color=CLASS_COLORS[c], linewidth=1.5, label=class_label(factor, level))
            if mark_lags:
                ax.axvline(level, color=CLASS_COLORS[c], linestyle=":", linewidth=1)
        ax.set_title(split_name)
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)
        style_axis(ax)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(title=factor, frameon=False, fontsize=8)
    save_fig(fig, fname)

def plot_class_waveforms(split, factor, remove_gain, fixed, ylabel, fname):
    """One WAVE_SNIPPET_LEN snippet per class, from the middle of the first matching segment.

    Args:
        fixed (tuple or None): (factor, class) that every snippet must share
    """
    segs = segment_matrix(split, remove_gain)
    y = split[factor].ravel()
    mask = np.ones_like(y, bool) if fixed is None else split[fixed[0]].ravel() == fixed[1]
    n_classes = len(FACTORS[factor])
    a = (SEGMENT_LEN - WAVE_SNIPPET_LEN) // 2
    t = np.arange(WAVE_SNIPPET_LEN) / F_S * 1000
    fig, axes = plt.subplots(n_classes, 1, figsize=(6, 1.0 * n_classes + 0.8), sharex=True, sharey=True)
    rows = []
    for c, ax in enumerate(axes):
        s = np.flatnonzero((y == c) & mask)[0]
        ax.plot(t, segs[s, a:a + WAVE_SNIPPET_LEN], color=CLASS_COLORS[c], linewidth=1)
        rows.append(pd.DataFrame({"class": c, factor: FACTORS[factor][c], "time_ms": t, "value": segs[s, a:a + WAVE_SNIPPET_LEN]}))
        ax.set_ylabel(class_label(factor, FACTORS[factor][c]), rotation=0, ha="right", va="center", fontsize=8)
        style_axis(ax)
    pd.concat(rows).to_csv(os.path.join(OUT_DIR, fname + ".csv"), index=False)
    title = f"{ylabel}, {WAVE_SPLIT}"
    if fixed is not None:
        title += f", {fixed[0]} {class_label(fixed[0], FACTORS[fixed[0]][fixed[1]])}"
    axes[0].set_title(title, fontsize=9)
    axes[-1].set_xlabel("Time (ms)")
    save_fig(fig, fname)

def plot_example_sequence(splits, fname):
    "Waveform of one sequence per split, with its segment-level lag and gain"
    fig, axes = plt.subplots(3, len(SPLITS), figsize=(5 * len(SPLITS), 5.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]}, squeeze=False)
    t = np.arange(SEGMENT_LEN * N_SEGMENTS) / F_S
    edges = np.arange(N_SEGMENTS + 1) * SEGMENT_LEN / F_S
    pd.concat([pd.DataFrame({"split": s, "time_s": t, "value": splits[s]["audio"][EXAMPLE_SEQ]}) for s in SPLITS]).to_csv(
        os.path.join(OUT_DIR, fname + ".csv"), index=False)
    pd.concat([pd.DataFrame({"split": s, "segment": np.arange(N_SEGMENTS), "t_start_s": edges[:-1], "t_end_s": edges[1:],
                             "lag": splits[s]["lag"][EXAMPLE_SEQ], "gain": splits[s]["gain"][EXAMPLE_SEQ],
                             "lag_samples": splits[s]["lag_samples"][EXAMPLE_SEQ], "gain_db": splits[s]["gain_db"][EXAMPLE_SEQ]})
               for s in SPLITS]).to_csv(os.path.join(OUT_DIR, fname + "_labels.csv"), index=False)
    for col, split_name in enumerate(SPLITS):
        split = splits[split_name]
        axes[0, col].plot(t, split["audio"][EXAMPLE_SEQ], color=WAVE_COLOR, linewidth=0.3)
        axes[0, col].set_title(f"{split_name}, sequence {EXAMPLE_SEQ}")
        axes[1, col].stairs(split["lag_samples"][EXAMPLE_SEQ], edges, color=CLASS_COLORS[-1], linewidth=1.5, baseline=None)
        axes[1, col].set_yscale("log", base=2)
        axes[1, col].set_yticks(LAG_LEVELS, [str(k) for k in LAG_LEVELS])
        axes[1, col].minorticks_off()
        axes[2, col].stairs(split["gain_db"][EXAMPLE_SEQ], edges, color=CLASS_COLORS[-1], linewidth=1.5, baseline=None)
        axes[2, col].set_yticks(GAIN_LEVELS_DB)
        axes[2, col].set_xlabel("Time (s)")
        for ax in axes[:, col]:
            style_axis(ax)
    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel("Lag (samples)")
    axes[2, 0].set_ylabel("Gain (dB)")
    axes[0, 0].set_xlim(0, edges[-1])
    save_fig(fig, fname)


def main():

    os.makedirs(OUT_DIR, exist_ok=True)
    mel_hz = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=0.0, fmax=F_S / 2)[1:-1]

    summary_rows, profile_rows, psd_rows, acf_rows = [], [], [], []
    spectra = {'lag': {}, 'gain': {}}
    autocovs = {'lag': {}, 'gain': {}}
    splits = {}
    for split_name in SPLITS:
        split = load_split(split_fname(split_name))
        splits[split_name] = split
        X, labels, groups = frame_dataset(split)
        for factor, levels in FACTORS.items():
            n_classes = len(levels)
            score = probe_score(X, labels[factor], groups, n_classes)
            profile = effect_profile(X, labels[factor], n_classes)
            shares = group_shares(profile)
            summary_rows.append({
                "split": split_name, "factor": factor, "probe_cn_bacc": score,
                **{f"effect_share_q{g + 1}": s for g, s in enumerate(shares)},
            })
            for b in range(N_MELS):
                profile_rows.append({
                    "split": split_name, "factor": factor, "mel_band": b, "mel_center_hz": mel_hz[b],
                    "effect_var": profile[b], "effect_share": profile[b] / profile.sum(),
                })
            print(split_name, factor, "probe", round(score, 3), "shares", np.round(shares, 2))

            freqs, spec = class_spectra(split, factor, remove_gain=(factor == 'lag'))
            spectra[factor][split_name] = spec
            for c, level in enumerate(levels):
                for fi, fr in enumerate(freqs):
                    psd_rows.append({"split": split_name, "class": c, factor: level, "freq_hz": fr, "log_power_db": spec[c, fi]})

            lags, acov = class_autocov(split, factor, remove_gain=(factor == 'lag'), normalise=(factor == 'lag'))
            autocovs[factor][split_name] = acov
            for c, level in enumerate(levels):
                for tau in lags:
                    acf_rows.append({"split": split_name, "class": c, factor: level, "lag_samples": tau, "autocov": acov[c, tau]})

    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "characterization.csv"), index=False)
    pd.DataFrame(profile_rows).to_csv(os.path.join(OUT_DIR, "effect_profiles.csv"), index=False)
    for factor in FACTORS:
        pd.DataFrame([r for r in psd_rows if factor in r]).to_csv(os.path.join(OUT_DIR, f"psd_by_{factor}.csv"), index=False)
        pd.DataFrame([r for r in acf_rows if factor in r]).to_csv(os.path.join(OUT_DIR, f"acf_by_{factor}.csv"), index=False)
    plot_class_curves(freqs / 1000, spectra['lag'], 'lag', "Frequency (kHz)", "Mean log-power (dB), gain removed",
                      (0, F_S / 2000), "psd_by_lag")
    plot_class_curves(freqs / 1000, spectra['gain'], 'gain', "Frequency (kHz)", "Mean log-power (dB)",
                      (0, F_S / 2000), "psd_by_gain")
    plot_class_curves(lags, autocovs['lag'], 'lag', "Lag (samples)", "Mean autocorrelation, gain removed",
                      (0, ACF_MAX_LAG), "acf_by_lag", mark_lags=True)
    plot_class_curves(lags, autocovs['gain'], 'gain', "Lag (samples)", "Mean autocovariance",
                      (0, ACF_MAX_LAG), "acf_by_gain")
    plot_class_waveforms(splits[WAVE_SPLIT], 'lag', True, None, "Gain removed", "waveforms_by_lag")
    plot_class_waveforms(splits[WAVE_SPLIT], 'gain', False, ('lag', GAIN_EXAMPLE_LAG_CLASS), "Raw", "waveforms_by_gain")
    plot_example_sequence(splits, "example_sequence")

if __name__ == "__main__":
    main()
