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

"""Simulation of a stochastic coupled system.
This script generates the SimCoupled dataset: two frame-level factors (coupling lag, gain) that are
not localised in frequency, mixed, and correlated by design. Factors change every 0.1 s within 4 s
sequences, as vowels do in SimVowels.
Source and coupling from:
Ding, Mingzhou, Yonghong Chen, and Steven L. Bressler. "Granger causality: basic theory and
application to neuroscience." Handbook of Time Series Analysis (2006): 437-460.
as used in:
Yang, Albert C., et al. "Causal decomposition in the mutual causation system."
Nature Communications 9.1 (2018): 3378.
"""


import numpy as np
from scipy.signal import lfilter
import os
import json
import gzip


#Generative Factors
#1. Coupling lag k of y(t) = c x(t-k) + sigma_y w2(t) - 0..4: 2, 4, 8, 16, 32 samples
#2. Gain of the observation - 0..4: -4, -2, 0, +2, +4 dB, +-1 dB jitter

# Constants and Parameters
SAVE_DIR = os.path.join("..", "sim_coupled")
SEED = 2026

F_S = 16000
SEGMENT_DUR = 0.1  # Duration in seconds for each (lag, gain) segment
UTTERANCE_DUR = 4 # in seconds duration of "utterance"
SEGMENT_LEN = int(F_S * SEGMENT_DUR)
N_SEGMENTS = int(UTTERANCE_DUR / SEGMENT_DUR)

AR_COEFS = [1.0, -0.95 * np.sqrt(2), 0.9025]  # x(t) = 0.95*sqrt(2) x(t-1) - 0.9025 x(t-2) + w1(t)
X_STD_DING = 3.28  # stationary std of that AR(2) with unit-variance w1
COUPLING = 0.5
SIGMA_Y = 1.0 / X_STD_DING  # Ding et al.'s unit-variance w2, relative to unit-variance x
SNR_DB = 10
BURN_IN = 1024  # also covers the largest lag

LAG_LEVELS = [2, 4, 8, 16, 32]
GAIN_LEVELS_DB = [-4, -2, 0, 2, 4]
GAIN_JITTER_DB = 1.0
CORR_SIGMA = 1.0  # P(i, j) ~ exp(-(i - j)^2 / (2 sigma^2)); None = independent

AUDIO_DECIMALS = 6
GAIN_DB_DECIMALS = 3

# name: (size, sigma, target class correlation); order fixes the SeedSequence children
SPLITS = {
    'train': (800, CORR_SIGMA, 0.77),
    'dev': (100, CORR_SIGMA, 0.77),
    'test': (100, CORR_SIGMA, 0.77),
    'indep': (100, None, 0.0),
}
CORR_TOL = 0.03
LEVEL_TOL_DB = 0.3
X_STD_TOL = 0.05
NOTCH_SPLIT = 'indep'


def split_fname(split):
    return os.path.join(SAVE_DIR, "sim_coupled_SNR_" + str(SNR_DB) + "_" + split + "_" + str(UTTERANCE_DUR) + "s.json.gz")

def meta_fname():
    return os.path.join(SAVE_DIR, "sim_coupled_SNR_" + str(SNR_DB) + "_" + str(UTTERANCE_DUR) + "s_meta.json")

def joint_class_probs(n_levels, sigma):
    "Joint probabilities of (lag class, gain class). sigma=None gives the independent (uniform) grid"
    if sigma is None:
        return np.full((n_levels, n_levels), 1.0 / n_levels ** 2)
    i, j = np.meshgrid(range(n_levels), range(n_levels), indexing="ij")
    p = np.exp(-(i - j) ** 2 / (2 * sigma ** 2))
    return p / p.sum()

def generate_sequence(rng, probs):
    """One 4 s sequence.

    Returns:
        audio (64000,), lag class (40,), gain class (40,), lag in samples (40,), gain in dB incl. jitter (40,)
    """
    n = SEGMENT_LEN * N_SEGMENTS
    x = lfilter([1.0], AR_COEFS, rng.normal(size=BURN_IN + n))
    x = x / X_STD_DING
    flat = rng.choice(probs.size, size=N_SEGMENTS, p=probs.ravel())
    lag_cls, gain_cls = flat // probs.shape[1], flat % probs.shape[1]
    audio = np.empty(n)
    gain_db = np.empty(N_SEGMENTS)
    for s in range(N_SEGMENTS):
        a = BURN_IN + s * SEGMENT_LEN
        b = a + SEGMENT_LEN
        k = LAG_LEVELS[lag_cls[s]]
        # lag reaches into the previous segment
        y = COUPLING * x[a - k:b - k] + SIGMA_Y * rng.normal(size=SEGMENT_LEN)
        m = x[a:b] + y
        m = m / m.std()
        noise = rng.normal(size=SEGMENT_LEN) * 10 ** (-SNR_DB / 20)
        gain_db[s] = GAIN_LEVELS_DB[gain_cls[s]] + rng.uniform(-GAIN_JITTER_DB, GAIN_JITTER_DB)
        audio[s * SEGMENT_LEN:(s + 1) * SEGMENT_LEN] = 10 ** (gain_db[s] / 20) * (m + noise)
    lag_samples = np.array([LAG_LEVELS[c] for c in lag_cls])
    return audio, lag_cls, gain_cls, lag_samples, gain_db

def generate_split(n_seq, seed, sigma):
    "Arrays of shape (n_seq, ...), rounded as saved"
    rng = np.random.default_rng(seed)
    probs = joint_class_probs(len(LAG_LEVELS), sigma)
    out = {"audio": [], "lag": [], "gain": [], "lag_samples": [], "gain_db": []}
    for _ in range(n_seq):
        audio, lc, gc, ls, gd = generate_sequence(rng, probs)
        out["audio"].append(np.round(audio, AUDIO_DECIMALS))
        out["lag"].append(lc)
        out["gain"].append(gc)
        out["lag_samples"].append(ls)
        out["gain_db"].append(np.round(gd, GAIN_DB_DECIMALS))
    return {key: np.stack(val) for key, val in out.items()}

def ar_stationary_std():
    "Stationary std of x(t) = a1 x(t-1) + a2 x(t-2) + w1(t), unit-variance w1"
    a1, a2 = -AR_COEFS[1], -AR_COEFS[2]
    return float(np.sqrt((1 - a2) / ((1 + a2) * ((1 - a2) ** 2 - a1 ** 2))))

def segment_power_at(split, freq):
    "Mean power at freq per lag class, gain removed. Returns dB (n_lag_levels,)"
    segs = split["audio"].reshape(-1, SEGMENT_LEN) / 10 ** (split["gain_db"].reshape(-1, 1) / 20)
    lag = split["lag"].ravel()
    P = np.abs(np.fft.rfft(segs * np.hanning(SEGMENT_LEN), axis=1)) ** 2
    f = np.fft.rfftfreq(SEGMENT_LEN, 1 / F_S)
    i_f = np.argmin(np.abs(f - freq))
    return 10 * np.log10(np.array([P[lag == c, i_f].mean() for c in range(len(LAG_LEVELS))]))

def sanity_checks(split, target_corr, notch=False):
    "Raises on failure; returns a dict for the sidecar"
    lag = split["lag"].ravel()
    gain = split["gain"].ravel()
    corr = float(np.corrcoef(lag, gain)[0, 1])
    assert abs(corr - target_corr) < CORR_TOL, f"class correlation {corr:.3f}, expected {target_corr} +- {CORR_TOL}"

    # 20 log10 rms = gain_db + 10 log10(1 + 10^(-SNR/10))
    audio = split["audio"].reshape(len(split["audio"]), N_SEGMENTS, SEGMENT_LEN)
    level = 20 * np.log10(np.sqrt((audio ** 2).mean(-1)))
    offset = 10 * np.log10(1 + 10 ** (-SNR_DB / 10))
    resid = float(np.abs(level - split["gain_db"] - offset).mean())
    assert resid < LEVEL_TOL_DB, f"segment level off by {resid:.2f} dB on average"

    x_std = ar_stationary_std()
    assert abs(x_std - X_STD_DING) < X_STD_TOL, f"AR(2) stationary std {x_std:.3f}, expected {X_STD_DING}"

    counts = np.zeros((len(LAG_LEVELS), len(GAIN_LEVELS_DB)), int)
    np.add.at(counts, (lag, gain), 1)
    results = {
        "class_correlation": corr,
        "level_residual_db_mean_abs": resid,
        "ar_stationary_std": x_std,
        "cell_counts": counts.tolist(),
    }
    if notch:
        # k = 4 puts a comb notch on the 2 kHz resonance
        p2k = segment_power_at(split, 2000)
        assert p2k[1] < p2k[2], f"no notch at 2 kHz: k=4 {p2k[1]:.1f} dB, k=8 {p2k[2]:.1f} dB"
        results["power_at_2khz_db_per_lag_class"] = p2k.tolist()
    return results

def save_split(fname, split):
    "gzipped JSON dict of lists, one entry per sequence, as in SimVowels; written per sequence to bound memory"
    with gzip.open(fname, "wt") as f:
        f.write("{")
        for n, (key, arr) in enumerate(split.items()):
            f.write((", " if n else "") + json.dumps(key) + ": [")
            for i, row in enumerate(arr):
                f.write((", " if i else "") + json.dumps(row.tolist()))
            f.write("]")
        f.write("}")

def load_split(fname):
    with gzip.open(fname, "rt") as f:
        data = json.load(f)
    return {key: np.array(val) for key, val in data.items()}

def constants():
    return {
        "F_S": F_S, "SEGMENT_DUR": SEGMENT_DUR, "UTTERANCE_DUR": UTTERANCE_DUR,
        "SEGMENT_LEN": SEGMENT_LEN, "N_SEGMENTS": N_SEGMENTS,
        "AR_COEFS": AR_COEFS, "X_STD_DING": X_STD_DING, "COUPLING": COUPLING, "SIGMA_Y": SIGMA_Y,
        "SNR_DB": SNR_DB, "BURN_IN": BURN_IN,
        "LAG_LEVELS": LAG_LEVELS, "GAIN_LEVELS_DB": GAIN_LEVELS_DB, "GAIN_JITTER_DB": GAIN_JITTER_DB,
        "CORR_SIGMA": CORR_SIGMA, "AUDIO_DECIMALS": AUDIO_DECIMALS, "GAIN_DB_DECIMALS": GAIN_DB_DECIMALS,
        "CORR_TOL": CORR_TOL, "LEVEL_TOL_DB": LEVEL_TOL_DB, "X_STD_TOL": X_STD_TOL,
    }


def main():

    os.makedirs(SAVE_DIR, exist_ok=True)

    seeds = np.random.SeedSequence(SEED).spawn(len(SPLITS))
    meta = {
        "seed": SEED,
        "seed_spawn_order": list(SPLITS.keys()),
        "constants": constants(),
        "joint_class_probs": {
            "correlated": joint_class_probs(len(LAG_LEVELS), CORR_SIGMA).tolist(),
            "independent": joint_class_probs(len(LAG_LEVELS), None).tolist(),
        },
        "splits": {},
    }
    if os.path.exists(meta_fname()):
        with open(meta_fname(), "r") as f:
            meta["splits"] = json.load(f).get("splits", {})

    for (name, (size, sigma, target_corr)), seed in zip(SPLITS.items(), seeds):
        fname = split_fname(name)
        if not os.path.exists(fname):
            split = generate_split(size, seed, sigma)
            checks = sanity_checks(split, target_corr, notch=(name == NOTCH_SPLIT))
            save_split(fname, split)
            print("Succesfully saved", name, "set:", {k: v for k, v in checks.items() if k != "cell_counts"})
        elif name not in meta["splits"]:
            split = load_split(fname)
            checks = sanity_checks(split, target_corr, notch=(name == NOTCH_SPLIT))
        else:
            continue
        meta["splits"][name] = {
            "file": os.path.basename(fname),
            "n_sequences": size,
            "sigma": sigma,
            "target_class_correlation": target_corr,
            **checks,
        }

    with open(meta_fname(), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
