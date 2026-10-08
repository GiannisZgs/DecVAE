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

"""How many components FD finds per frame, against the fixed number of components C the model needs.

For each dataset and peak-search variant, counts the components K that FD keeps in every frame
(detected peaks after merging, minus those below spec_amp_tolerance), then reports for each C how
many frames are grouped (K > C), match exactly (K = C), are padded with empty components (0 < K < C),
or have no component (K = 0, the frame is left out of the loss). Uses the same audio loading,
normalisation, framing and FD functions as the pretraining preprocessing. Runs on the inputs only, no
model is needed. Optionally compares K with what DecompositionModule.forward does on a few utterances.
The figure data goes to figure_data_dir as tidy CSVs, drawn by visualize_R/SI_decomp_sensitivity/.

python scripts/post-training/fd_component_counts.py --config_file config_files/decomp_sensitivity/config_fd_component_counts.json
"""

import os
import sys
import copy
import json
import numpy as np
import pandas as pd
import scipy.signal

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

JSON_FILE_NAME_MANUAL = "config_files/decomp_sensitivity/config_fd_component_counts.json" #for debugging purposes only

DEFAULT_ARGS = {
    "output_dir": None,
    "figure_data_dir": "data/decomp_sensitivity",
    "n_utterances": 100,
    "seed": 0,
    "C_values": [2, 3, 4, 5, 6],
    "check_utterances": 0,  # utterances on which K is compared with DecompositionModule.forward, 0 to skip
    "forward_check_C": None,  # C values of the forward check for the base variant, None is the config's NoC
    "datasets": {},
}
DATASET_KEYS = {"config", "split", "true_C", "variants"}


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("comment")}
    unknown = set(cfg) - set(DEFAULT_ARGS)
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    args = {**DEFAULT_ARGS, **cfg}
    for ds, d in args["datasets"].items():
        unknown = set(d) - DATASET_KEYS
        if unknown:
            raise ValueError(f"Unknown keys for dataset {ds} in {path}: {sorted(unknown)}")
    if args["output_dir"] is None:
        raise ValueError(f"output_dir must be set in {path}")
    return args


def kept_components(dm, frame, intervals):
    """Number of components FD keeps for one frame, as decomp_mask computes it before grouping or padding.

    Args:
        dm: DecompositionModule built from the run's config.
        frame: One frame of the normalised input (receptive_field * fs samples).
        intervals: Peak-search intervals, as returned by dm.get_peak_detection_intervals().
    Returns:
        K: Detected components that pass the spec_amp_tolerance check.
    """
    freqs, _, _ = dm.peak_detection(frame, intervals)
    if freqs.shape[0] == 0:
        return 0
    OCs, _, _, _ = dm.filter_decomp(frame, freqs)
    K = 0
    for oc in OCs:
        "Same Welch settings as the spectral amplitude check in decomp_mask"
        _, spec = scipy.signal.welch(oc, fs=dm.fs, window='hann', nperseg=len(frame)/6, noverlap=len(frame)/40,
                                     nfft=dm.nfft, detrend='constant', return_onesided=True, scaling='density',
                                     axis=-1, average='mean')
        K += int(spec.max() >= dm.spec_amp_tolerance)
    return K


def frames_of(x, n_valid, dm):
    "Frames decomp_mask decomposes: those inside the attention length at the output of the conv encoder"
    L, H = int(dm.RFS * dm.fs), int(dm.stride * dm.fs)
    starts = list(range(0, len(x) - L + 1, H))
    n_frames = min(len(starts), int(dm._get_feat_extract_output_lengths(n_valid)))
    return [x[s:s + L] for s in starts[:n_frames]]


def count_frames(dm, utterances, remove_silence):
    """K for every frame of every utterance.

    Args:
        utterances: List of (normalised input, number of valid samples).
    Returns:
        K per frame (int array), with -1 for frames the silence check removes, and the number of frames per utterance.
    """
    intervals = dm.get_peak_detection_intervals()
    K, n_frames = [], []
    for x, n_valid in utterances:
        if dm.use_notch_filter:
            x = dm.notch_filter_power_line_noise(x)
        frames = frames_of(x, n_valid, dm)
        n_frames.append(len(frames))
        for frame in frames:
            if remove_silence and dm.silence_check(frame):
                K.append(-1)
            else:
                K.append(kept_components(dm, frame, intervals))
    return np.array(K), n_frames


def c_table(K, C_values):
    "Share of the non-silent frames grouped, exact, padded and without component, for each C"
    k = K[K >= 0]
    return {int(C): {"grouped": float(np.mean(k > C)), "exact": float(np.mean(k == C)),
                     "padded": float(np.mean((k > 0) & (k < C))), "none": float(np.mean(k == 0))}
            for C in C_values}


def summary(K):
    k = K[K >= 0]
    counts = np.bincount(k)
    return {"n_frames": int(len(K)), "silent": int(np.sum(K < 0)), "median": float(np.median(k)),
            "mode": int(np.argmax(counts)), "mean": float(k.mean()),
            "distribution": {int(i): float(c / len(k)) for i, c in enumerate(counts)}}


def config_with(config, over):
    "A copy of the config with the variant's decomposition keys changed"
    c = copy.deepcopy(config)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def resolve_variant(over, config):
    "@random:<seed> becomes the boundaries the sweep uses for the same seed"
    from utils.sweep_helpers import random_boundaries
    out = dict(over)
    v = out.get("detection_boundaries")
    if isinstance(v, str) and v.startswith("@random:"):
        n = out.get("detection_intervals", config.detection_intervals)
        out["detection_boundaries"] = random_boundaries(config.lower_speech_freq, config.higher_speech_freq,
                                                        n, int(v.split(":")[1]), 300.0)
    return out


def check_against_forward(dm, utterances, K, n_frames, n_check):
    """Compares K with what the DecompositionModule itself does at C = dm.NoC, on the first n_check utterances.

    Expected per frame: no component (K = 0) <-> frame masked out; otherwise C - K padded components when
    K < C and none when K >= C, where padded components are the zeros of the frame-level component_indices.
    Returns:
        Share of agreeing frames, number of frames compared.
    """
    import torch
    agree, total, i = 0, 0, 0
    C = dm.NoC
    for (x, n_valid), n in zip(utterances[:n_check], n_frames[:n_check]):
        T = int(dm._get_feat_extract_output_lengths(len(x)))
        att = torch.zeros((1, len(x)), dtype=torch.long)
        att[0, :n_valid] = 1
        out = dm(x[None].copy(), mask_time_indices=torch.ones((1, T), dtype=torch.bool), attention_mask=att,
                 remove_silence=False)
        mask, comp = np.asarray(out[1]).reshape(-1), np.asarray(out[10]["frame"])
        assert comp.shape == (C, 1, T), f"unexpected component_indices shape {comp.shape}"
        for s in range(n):
            k = K[i + s]
            if k < 0:
                continue
            padded = C - int(comp[:, 0, s].sum())
            expected_mask = k > 0
            ok = (bool(mask[s]) == expected_mask) and (padded == (C if k == 0 else max(C - k, 0)))
            agree += ok
            total += 1
        i += n
    return agree / max(total, 1), total


def load_utterances(cfg_path, split, n_utt, seed):
    "Normalised inputs of n_utt utterances of one split, exactly as the pretraining preprocessing sees them"
    from transformers import HfArgumentParser, Wav2Vec2FeatureExtractor
    from args_configs import ModelArguments, DataTrainingArguments, DecompositionArguments, TrainingObjectiveArguments
    from config_files import DecVAEConfig
    from dataset_loading import load_timit, load_sim_vowels, load_sim_coupled
    parser = HfArgumentParser((ModelArguments, TrainingObjectiveArguments, DecompositionArguments, DataTrainingArguments))
    model_args, training_obj_args, decomp_args, data_training_args = parser.parse_json_file(json_file=cfg_path)
    for a, c in ((model_args, "comment_model_args"), (training_obj_args, "comment_tr_obj_args"), (decomp_args, "comment_decomp_args")):
        if hasattr(a, c):
            delattr(a, c)
    config = DecVAEConfig(**{**model_args.__dict__, **training_obj_args.__dict__, **decomp_args.__dict__})
    config.dataset_name = data_training_args.dataset_name
    name = data_training_args.dataset_name
    if name == "timit":
        loader = load_timit
    elif "sim_vowels" in name:
        loader = load_sim_vowels
    elif name == "sim_coupled":
        loader = load_sim_coupled
    else:
        raise ValueError(f"no loader for {name}")
    raw = loader(data_training_args)[split]
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)
    max_length = int(data_training_args.max_duration_in_seconds * fe.sampling_rate)
    idx = np.sort(np.random.default_rng(seed).choice(raw.num_rows, size=min(n_utt, raw.num_rows), replace=False))
    utterances = []
    for i in idx:
        sample = raw[int(i)][data_training_args.audio_column_name]
        if type(sample) == list:
            sample = {"array": np.array(sample), "sampling_rate": decomp_args.fs}
        "Same call as the pretraining preprocessing (prepare_pretraining_dataset)"
        inputs = fe(sample["array"], sampling_rate=sample["sampling_rate"], max_length=max_length,
                    truncation=True, padding="max_length")
        utterances.append((np.asarray(inputs.input_values[0], dtype=np.float64), int(np.sum(inputs.attention_mask[0]))))
    return config, decomp_args, utterances


def analyse_dataset(config, remove_silence, utterances, d, args):
    "Counts, C tables and the optional forward check for every peak-search variant of one dataset"
    from models import DecompositionModule
    "Peak detection is FD's whatever decomp_to_perform the config was written for"
    config.decomp_to_perform = "filter"
    out = {}
    for v, over in d["variants"].items():
        over = resolve_variant(over, config)
        vcfg = config_with(config, over)
        dm = DecompositionModule(vcfg)
        K, n_frames = count_frames(dm, utterances, remove_silence)
        res = {"overrides": over, "intervals": [[float(a), float(b)] for a, b in dm.get_peak_detection_intervals()],
               "summary": summary(K), "C_table": c_table(K, args["C_values"])}
        n_check = args["check_utterances"]
        if n_check > 0:
            "Also the smoke test of every sweep configuration: forward runs at each C without error"
            check_C = (args["forward_check_C"] or [vcfg.NoC]) if v == "base" else [vcfg.NoC]
            res["forward_check"] = {}
            for C in check_C:
                dm_c = DecompositionModule(config_with(vcfg, {"NoC": C, "NoC_seq": C if vcfg.seq_decomp else vcfg.NoC_seq}))
                agree, n = check_against_forward(dm_c, utterances, K, n_frames, n_check)
                res["forward_check"][int(C)] = {"agreement": agree, "frames": n}
                print(f"{v}: K agrees with DecompositionModule.forward at C = {C} on {agree:.1%} of {n} frames")
        out[v] = res
        s = res["summary"]
        print(f"{v}: {s['n_frames']} frames, median K {s['median']:.0f}, mode {s['mode']}, "
              + ", ".join(f"C={C}: grp {t['grouped']:.2f} ex {t['exact']:.2f} pad {t['padded']:.2f}"
                          for C, t in res["C_table"].items()))
    return out


def write_outputs(results, args):
    "Full results and the per-C table in output_dir, tidy figure data in figure_data_dir"
    out_dir, fig_dir = args["output_dir"], args["figure_data_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fd_component_counts.json"), "w") as f:
        json.dump(results, f, indent=1)
    dist_rows, c_rows, summ_rows, check_rows = [], [], [], []
    for ds, r in results.items():
        for order, (v, vr) in enumerate(r["variants"].items()):
            base = {"dataset": ds, "variant": v, "variant_order": order, "true_C": r["true_C"]}
            s = vr["summary"]
            dist_rows += [{**base, "K": k, "share": p, "n_frames": s["n_frames"] - s["silent"]}
                          for k, p in s["distribution"].items()]
            c_rows += [{**base, "C": C, "outcome": o, "share": t[o]}
                       for C, t in vr["C_table"].items() for o in ("grouped", "exact", "padded", "none")]
            summ_rows.append({**base, "n_utterances": r["n_utterances"], "n_frames": s["n_frames"], "silent": s["silent"],
                              "median": s["median"], "mode": s["mode"], "mean": s["mean"],
                              "boundaries": " ".join(f"{b:.0f}" for b in [vr["intervals"][0][0]] + [hi for _, hi in vr["intervals"]])})
            check_rows += [{**base, "C": C, "agreement": c["agreement"], "frames": c["frames"]}
                           for C, c in vr.get("forward_check", {}).items()]
    pd.DataFrame(c_rows).pivot_table(index=["dataset", "variant", "C"], columns="outcome", values="share", sort=False)[
        ["grouped", "exact", "padded", "none"]].reset_index().to_csv(os.path.join(out_dir, "fd_component_counts_by_C.csv"), index=False)
    pd.DataFrame(dist_rows).to_csv(os.path.join(fig_dir, "component_counts_distribution.csv"), index=False)
    pd.DataFrame(c_rows).to_csv(os.path.join(fig_dir, "component_counts_by_C.csv"), index=False)
    pd.DataFrame(summ_rows).to_csv(os.path.join(fig_dir, "component_counts_summary.csv"), index=False)
    if check_rows:
        pd.DataFrame(check_rows).to_csv(os.path.join(fig_dir, "component_counts_forward_check.csv"), index=False)
    print(f"Saved the results in {out_dir} and the figure data in {fig_dir}")


def main():
    from utils import parse_args, debugger_is_active
    args = load_config(JSON_FILE_NAME_MANUAL if debugger_is_active() else parse_args().config_file)
    results = {}
    for ds, d in args["datasets"].items():
        print(f"=== {ds}")
        config, decomp_args, utterances = load_utterances(d["config"], d.get("split", "test"), args["n_utterances"], args["seed"])
        results[ds] = {"config": d["config"], "true_C": d.get("true_C"), "n_utterances": len(utterances),
                       "variants": analyse_dataset(config, decomp_args.remove_silence, utterances, d, args)}
    write_outputs(results, args)


if __name__ == "__main__":
    main()
