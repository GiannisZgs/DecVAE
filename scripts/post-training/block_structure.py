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

"""Block structure of the decomposition embeddings of trained DecVAE checkpoints (Proposition 2, SI Section 7).
No training: every checkpoint is loaded, run forward on a split, and the per-sample and dataset-level measures
of latent_analysis_utils.block_structure_utils.BlockStructure are collected.

Sibling of decomposition_divergences.py: same runs-file format, checkpoint resolution, data loading and
prepare_batch (imported from it), so a run here sees exactly the checkpoint and the frames that the paper's
post-training evaluation saw. A model entry with "untrained": true builds DecVAEForPreTraining(config) with the
same config and seed and does not load weights (control).

    python scripts/post-training/block_structure.py --config_file config_files/block_structure/runs_sim_vowels.json
    python scripts/post-training/block_structure.py --config_file ... --dry_run
    python scripts/post-training/block_structure.py --config_file ... --only FD_b0.1 --max_batches 2   # smoke test

S branch: the per-batch JS(R_i, P) and JS(R_i, R_j) are cross-checked against the model's own divergences and
ComponentDivergences from the same forward pass (measure check_max_abs_diff, warning above 1e-4).
Writes one long CSV (one row per run, branch, split, measure and component / pair) and, per run, branch and
split, mean_distributions_<dataset>_<run_id>_<branch>_<split>.csv (rows R_bar_1..C and P_bar) next to it.
Runs already in the CSV are skipped, unless overwrite is set; overwrite replaces only the runs evaluated again.
"""

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import inspect
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from torch.utils.data.dataloader import DataLoader
from transformers import Wav2Vec2FeatureExtractor, set_seed

from models import DecVAEForPreTraining
from data_collation import DataCollatorForDecVAELatentPostAnalysis_NoFeatureExtraction
from config_files import DecVAEConfig
from utils import debugger_is_active
from utils.cache_utils import build_cache_file_names
from latent_analysis_utils.divergence_utils import ComponentDivergences
from latent_analysis_utils.block_structure_utils import BlockStructure
from decomposition_divergences import (beta_tag, resolve_checkpoint_dir, select_checkpoint, cache_splits,
                                       load_splits, prepare_batch, parse_run, unset_values)

warnings.simplefilter("ignore")

JSON_FILE_NAME_MANUAL = "config_files/block_structure/runs_sim_vowels.json" #for debugging purposes only

RUN_OPTIONS = {
    "only": None,  # run_ids to evaluate, None runs all
    "max_batches": None,  # smoke test: stop after this many batches per split, writes *_smoke.csv
    "overwrite": False,
    "dry_run": False,  # only check that every checkpoint and cache file exists
}
SPEC_KEYS = {"base_config", "output_csv", "splits", "betas", "models", "exceptions", "max_frames_per_batch"}
CHECK_TOL = 1e-4


def check_run(run, base_config, splits):
    "Pre-flight as decomposition_divergences.check_run; an untrained run needs no checkpoint"
    unset = unset_values(run)
    if unset:
        return ["not set: " + ", ".join(unset)]
    model_args, training_obj_args, decomp_args, data_training_args = parse_run(run, base_config)
    problems = []
    if not run["untrained"]:
        checkpoint_dir = run.get("checkpoint_dir") or resolve_checkpoint_dir(model_args, training_obj_args, decomp_args, data_training_args)
        if not os.path.isdir(checkpoint_dir):
            problems.append(f"no checkpoint folder {checkpoint_dir}")
        else:
            try:
                ckp = select_checkpoint(checkpoint_dir, run.get("checkpoint", -1))
                if not os.path.isfile(os.path.join(checkpoint_dir, ckp, "model.safetensors")):
                    problems.append(f"no model.safetensors in {os.path.join(checkpoint_dir, ckp)}")
            except (IndexError, AssertionError) as e:
                problems.append(f"no checkpoint in {checkpoint_dir} ({e})")
    cache_file_names = build_cache_file_names(data_training_args, data_training_args.input_type)
    needed = dict.fromkeys(n for s in splits for n in cache_splits(s, data_training_args.dataset_name))
    for name in needed:
        if not cache_file_names.get(name):
            problems.append(f"no {name} cache configured")
            continue
        for f in cache_file_names[name]:
            if not os.path.isfile(f):
                problems.append(f"missing cache {f}")
    return problems


def cross_check(bs, cd):
    "Largest difference between the last batch of BlockStructure and of ComponentDivergences (S branch)"
    mine = bs.batch_means[-1]
    n_utt = cd.batches[-1]["n_utt"]
    diff = 0.0
    for key, v in mine.items():
        diff = max(diff, abs(v - cd.batches[-1]["model_" + key]), abs(v - float(np.mean(cd.utt_sums[key][-n_utt:]))))
    return diff


def run_one(run, base_config, splits, accelerator, out_dir, max_batches=None):
    """One checkpoint: returns a list of long-format rows and writes the mean distributions.
    Model and data set-up as in decomposition_divergences.run_one."""
    model_args, training_obj_args, decomp_args, data_training_args = parse_run(run, base_config)
    if data_training_args.seed is not None:
        set_seed(data_training_args.seed)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)
    min_length = int(data_training_args.min_duration_in_seconds * feature_extractor.sampling_rate)
    model_args.max_duration_in_seconds = data_training_args.max_duration_in_seconds
    config = DecVAEConfig(**{**model_args.__dict__, **training_obj_args.__dict__, **decomp_args.__dict__})
    assert config.max_frames_per_batch == "all" or (isinstance(config.max_frames_per_batch, int) and config.max_frames_per_batch > 0), \
        "max_frames_per_batch must be 'all' or a positive integer (frames per utterance in the decomposition loss)"

    model = DecVAEForPreTraining(config)
    if run["untrained"]:
        ckp = "untrained"
        print(f"[{run['run_id']}] untrained, seed {data_training_args.seed}")
    else:
        checkpoint_dir = run.get("checkpoint_dir") or resolve_checkpoint_dir(model_args, training_obj_args, decomp_args, data_training_args)
        ckp = select_checkpoint(checkpoint_dir, run.get("checkpoint", -1))
        print(f"[{run['run_id']}] {os.path.join(checkpoint_dir, ckp)}")
        weights = load_file(os.path.join(checkpoint_dir, ckp, "model.safetensors"))
        for key in [k for k in weights.keys() if 'project_hid' in k or 'project_q' in k]:
            del weights[key]
        model.load_state_dict(weights, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    forward_args = set(inspect.signature(DecVAEForPreTraining.forward).parameters)

    data_collator = DataCollatorForDecVAELatentPostAnalysis_NoFeatureExtraction(
        model=model,
        feature_extractor=feature_extractor,
        model_args=model_args,
        data_training_args=data_training_args,
        config=config,
        input_type=data_training_args.input_type,
        dataset_name=data_training_args.dataset_name,
        pad_to_multiple_of=data_training_args.pad_to_multiple_of,
        mask_time_prob=config.mask_time_prob if model_args.mask_time_prob is None else model_args.mask_time_prob,
        mask_time_length=config.mask_time_length if model_args.mask_time_length is None else model_args.mask_time_length,
    )
    datasets_ = load_splits(data_training_args, splits, min_length)
    model = accelerator.prepare(model)

    branches = []
    if config.dual_branched_latent or config.only_z_branch:
        branches.append(("z", config.NoC))
    if config.dual_branched_latent or config.only_s_branch:
        branches.append(("s", config.NoC_seq))

    rows = []
    for split in splits:
        "Batch size of latents_post_analysis.py: train batch size for the single 'all' set, eval otherwise"
        batch_size = data_training_args.per_device_train_batch_size if split == "all" else data_training_args.per_device_eval_batch_size
        loader = accelerator.prepare(DataLoader(datasets_[split].with_format("numpy"), shuffle=False,
                                                collate_fn=data_collator, batch_size=batch_size))
        acc = {b: BlockStructure(NoC, b) for b, NoC in branches}
        cd = ComponentDivergences(config.NoC_seq, "s") if "s" in acc else None
        check_diff = 0.0
        start = time.time()
        with torch.no_grad():
            for step, batch in enumerate(loader):
                if max_batches is not None and step >= max_batches:
                    break
                inputs = prepare_batch(batch, data_training_args.dataset_name, forward_args)
                outputs = model(**inputs)
                for a in acc.values():
                    a.add(outputs)
                if cd is not None:
                    cd.add(outputs)
                    check_diff = max(check_diff, cross_check(acc["s"], cd))
                del batch, inputs, outputs
        if cd is not None and check_diff > CHECK_TOL:
            print(f"WARNING [{run['run_id']} s {split}]: JS differs from the model's / ComponentDivergences' by {check_diff:.2e}")
        for b, a in acc.items():
            summary = a.summary()
            if b == "s":
                summary.append(("check_max_abs_diff", None, None, check_diff))
            for measure, ci, cj, value in summary:
                rows.append({"dataset": data_training_args.dataset_name, "model": run["model"], "beta": run["beta"],
                             "untrained": run["untrained"], "run_id": run["run_id"], "checkpoint": ckp,
                             "branch": b, "split": split, "measure": measure,
                             "comp_i": "" if ci is None else ci, "comp_j": "" if cj is None else cj, "value": value})
            R_bar, P_bar = a.mean_distributions()
            md = pd.DataFrame(np.vstack([R_bar, P_bar[None]]), columns=[f"u{u + 1}" for u in range(R_bar.shape[1])])
            md.insert(0, "row", [f"R{i + 1}" for i in range(R_bar.shape[0])] + ["P"])
            md.to_csv(os.path.join(out_dir, f"mean_distributions_{data_training_args.dataset_name}_{run['run_id']}_{b}_{split}.csv"), index=False)
        print(f"[{run['run_id']}] {split}: {time.time() - start:.0f} s")
    del model
    torch.cuda.empty_cache()
    return rows


def expand_runs(spec):
    """Runs from the runs file, as decomposition_divergences.expand_runs, plus the 'untrained' flag of a model
    entry (run_id <model>_b<beta>_untrained; it needs no parent_dir or checkpoint)."""
    runs = []
    for m in spec["models"]:
        untrained = bool(m.get("untrained", False))
        for beta in m.get("betas", spec.get("betas")):
            run_id = f"{m['name']}_b{beta}" + ("_untrained" if untrained else "")
            ov = copy.deepcopy(m.get("overrides", {}))
            ov.setdefault("max_frames_per_batch", spec.get("max_frames_per_batch", "all"))
            ov["beta_kl_prior_z"] = float(beta)
            ov["beta_kl_prior_s"] = float(beta)
            if "parent_dir" in ov:
                ov["parent_dir"] = ov["parent_dir"].replace("{btag}", beta_tag(beta))
            run = {"run_id": run_id, "model": m["name"], "beta": beta, "untrained": untrained, "overrides": ov}
            if m.get("checkpoint_dir"):
                run["checkpoint_dir"] = m["checkpoint_dir"].replace("{btag}", beta_tag(beta))
            exc = spec.get("exceptions", {}).get(run_id, {})
            run["overrides"].update(exc.get("overrides", {}))
            for key in ["checkpoint_dir", "checkpoint"]:
                if key in exc:
                    run[key] = exc[key]
            runs.append(run)
    return runs


def load_runs_file(path):
    "Runs file and run options; unknown top-level keys are rejected"
    with open(path) as f:
        spec = json.load(f)
    spec = {k: v for k, v in spec.items() if not k.startswith("comment")}
    unknown = set(spec) - SPEC_KEYS - set(RUN_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    return {**RUN_OPTIONS, **spec}


def parse_cli():
    ap = argparse.ArgumentParser(description="Block structure of the decomposition embeddings of trained DecVAE checkpoints.")
    ap.add_argument("--config_file", required=True, help="runs file, e.g. config_files/block_structure/runs_sim_vowels.json")
    ap.add_argument("--only", nargs="*", default=None, help="run_ids to evaluate (default: all)")
    ap.add_argument("--max_batches", type=int, default=None, help="smoke test: stop after this many batches per split")
    ap.add_argument("--overwrite", action="store_true", default=None)
    ap.add_argument("--dry_run", action="store_true", default=None, help="only check that every checkpoint and cache file exists")
    return ap.parse_args()


def main():
    "Parse the arguments"
    if debugger_is_active():
        cli = argparse.Namespace(config_file=JSON_FILE_NAME_MANUAL, **{k: None for k in RUN_OPTIONS})
    else:
        cli = parse_cli()
    spec = load_runs_file(cli.config_file)
    opts = {k: getattr(cli, k) if getattr(cli, k) is not None else spec[k] for k in RUN_OPTIONS}

    with open(spec["base_config"]) as f:
        base_config = json.load(f)
    out_csv = spec["output_csv"] if opts["max_batches"] is None else spec["output_csv"].replace(".csv", "_smoke.csv")
    out_dir = os.path.dirname(out_csv) if opts["max_batches"] is None else os.path.join(os.path.dirname(out_csv), "smoke")
    os.makedirs(out_dir, exist_ok=True)
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=["run_id", "split"])

    if opts["dry_run"]:
        n_bad = 0
        for run in expand_runs(spec):
            if opts["only"] and run["run_id"] not in opts["only"]:
                continue
            problems = check_run(run, base_config, spec["splits"])
            n_bad += bool(problems)
            print(f"[{run['run_id']}] " + ("ok" if not problems else "; ".join(problems)))
        print(f"{n_bad} run(s) with problems")
        return

    accelerator = Accelerator()
    for run in expand_runs(spec):
        if opts["only"] and run["run_id"] not in opts["only"]:
            continue
        unset = unset_values(run)
        if unset:
            print(f"[{run['run_id']}] SKIPPED, not set: {', '.join(unset)}")
            continue
        todo = [s for s in spec["splits"] if opts["overwrite"] or not ((done["run_id"] == run["run_id"]) & (done["split"] == s)).any()]
        if not todo:
            print(f"[{run['run_id']}] already in {out_csv}, skipped")
            continue
        try:
            rows = run_one(run, base_config, todo, accelerator, out_dir, opts["max_batches"])
        except FileNotFoundError as e:
            print(f"[{run['run_id']}] SKIPPED, checkpoint or cache not found: {e}")
            continue
        done = done[~((done["run_id"] == run["run_id"]) & done["split"].isin(todo))]
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
        done.to_csv(out_csv, index=False)  # after every run, so an interruption loses at most one run
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
