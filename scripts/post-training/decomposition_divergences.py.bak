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

"""Decomposition divergences (D_recon, D_ortho) of trained DecVAE checkpoints, for the beta trade-off analysis. No training: every checkpoint is loaded, run forward on a split, and the divergences
the forward pass returns are collected with latent_analysis_utils.divergence_utils.ComponentDivergences.

The data loading, checkpoint directory and batch preparation follow latents_post_analysis.py, so a run here
sees exactly the checkpoint and the frames that the paper's post-training evaluation saw.

    python scripts/post-training/decomposition_divergences.py --config_file config_files/beta_tradeoff/runs_sim_vowels.json
    python scripts/post-training/decomposition_divergences.py --config_file ... --dry_run
    python scripts/post-training/decomposition_divergences.py --config_file ... --only FD_b0.1 --max_batches 2   # smoke test

The runs file can also hold the options only, max_batches, overwrite and dry_run (see RUN_OPTIONS); a flag
given on the command line overrides the file. Writes one long CSV (one row per run, branch, split and measure).
Runs already in the CSV are skipped, unless overwrite is set.
"""

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import copy
import inspect
import json
import time
import warnings

import pandas as pd
import torch
from accelerate import Accelerator
from datasets import DatasetDict, concatenate_datasets, Dataset
from safetensors.torch import load_file
from torch.utils.data.dataloader import DataLoader
from transformers import Wav2Vec2FeatureExtractor, HfArgumentParser, set_seed

from models import DecVAEForPreTraining
from data_collation import DataCollatorForDecVAELatentPostAnalysis_NoFeatureExtraction
from config_files import DecVAEConfig
from args_configs import ModelArgumentsPost, DataTrainingArgumentsPost, DecompositionArguments, TrainingObjectiveArguments
from utils import debugger_is_active, extract_epoch
from utils.cache_utils import build_cache_file_names
from latent_analysis_utils.divergence_utils import ComponentDivergences

warnings.simplefilter("ignore")

JSON_FILE_NAME_MANUAL = "config_files/beta_tradeoff/runs_sim_vowels.json" #for debugging purposes only

RUN_OPTIONS = {
    "only": None,  # run_ids to evaluate, None runs all
    "max_batches": None,  # smoke test: stop after this many batches per split, writes *_smoke.csv
    "overwrite": False,
    "dry_run": False,  # only check that every checkpoint and cache file exists
}
SPEC_KEYS = {"base_config", "output_csv", "splits", "betas", "models", "exceptions"}
UNSET = "SET"  # placeholder in the runs files for a value still to be filled in


def beta_tag(beta):
    "0 -> '0', 0.1 -> '01', 1 -> '1', 5 -> '5': the tag used in the pre-training folder names"
    return "01" if beta == 0.1 else str(int(beta))


def resolve_checkpoint_dir(model_args, training_obj_args, decomp_args, data_training_args):
    """Checkpoint directory, as in latents_post_analysis.py main() (copied, not imported, so that the
    evaluation script stays untouched). Keep the two in sync if that logic changes."""
    if model_args.dual_branched_latent:
        model_type = "dual"
        if training_obj_args.beta_kl_prior_z == 0.1 and training_obj_args.beta_kl_prior_s == 0.1:
            betas = "_bz01" + "_bs01"
        elif training_obj_args.beta_kl_prior_z == 0.1 and training_obj_args.beta_kl_prior_s != 0.1:
            betas = "_bz01" + "_bs" + str(int(training_obj_args.beta_kl_prior_s))
        elif training_obj_args.beta_kl_prior_z != 0.1 and training_obj_args.beta_kl_prior_s == 0.1:
            betas = "_bz" + str(int(training_obj_args.beta_kl_prior_z)) + "_bs01"
        else:
            betas = "_bz" + str(int(training_obj_args.beta_kl_prior_z)) + "_bs" + str(int(training_obj_args.beta_kl_prior_s))
    elif model_args.only_z_branch:
        model_type = "single_z"
        betas = "_bz01" if training_obj_args.beta_kl_prior_z == 0.1 else "_bz" + str(int(training_obj_args.beta_kl_prior_z))
    elif model_args.only_s_branch:
        model_type = "single_s"
        betas = "_bs01" if training_obj_args.beta_kl_prior_s == 0.1 else "_bs" + str(int(training_obj_args.beta_kl_prior_s))
    if data_training_args.dataset_name == "sim_coupled":
        checkpoint_dir = data_training_args.parent_dir
    elif "vowels" in data_training_args.dataset_name:
        checkpoint_dir = os.path.join(data_training_args.parent_dir,
            "snr" + str(data_training_args.sim_snr_db)
            + betas + "_NoC" + str(decomp_args.NoC) + "_" + data_training_args.input_type + "_" + model_type + "-bs" + str(data_training_args.per_device_train_batch_size))
    elif data_training_args.dataset_name in ["timit", "iemocap"]:
        checkpoint_dir = os.path.join(data_training_args.parent_dir,
            betas[1:] + "_NoC" + str(decomp_args.NoC) + "_" + data_training_args.input_type + "_" + model_type + "-bs" + str(data_training_args.per_device_train_batch_size))
    elif "VOC_ALS" in data_training_args.dataset_name:
        if data_training_args.transfer_from == "timit":
            checkpoint_dir = os.path.join(data_training_args.parent_dir,
                betas[1:] + "_NoC" + str(decomp_args.NoC) + "_" + data_training_args.input_type + "_" + model_type + "-bs" + str(data_training_args.per_device_train_batch_size))
        elif data_training_args.transfer_from == "sim_vowels":
            checkpoint_dir = os.path.join(data_training_args.parent_dir,
                "snr" + str(data_training_args.sim_snr_db)
                + betas + "_NoC" + str(decomp_args.NoC) + "_" + data_training_args.input_type + "_" + model_type + "-bs" + str(data_training_args.per_device_train_batch_size))
    if data_training_args.experiment == "ssl_loss":
        checkpoint_dir += "_" + str(data_training_args.ssl_loss_frame_perc) + "percent_frames"
    return checkpoint_dir


def select_checkpoint(checkpoint_dir, which=-1):
    "Checkpoint folder, sorted by epoch as in latents_post_analysis.py; -1 is the last (lowest validation loss) one"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if 'config' not in f]
    checkpoint_files.sort(key=extract_epoch)
    if isinstance(which, str):
        assert which in checkpoint_files, f"{which} not in {checkpoint_dir}"
        return which
    return checkpoint_files[which]


def cache_splits(split, dataset_name):
    "Cache entries (keys of build_cache_file_names) a split is read from"
    if split == "validation":
        return ["dev"] if dataset_name == "timit" else ["validation"]
    if split == "all":
        return ["train", "validation", "test"] + (["dev"] if dataset_name == "VOC_ALS" else [])
    if split in ["test", "train", "dev", "indep"]:
        return [split]
    raise ValueError(f"unknown split {split}")


def load_splits(data_training_args, splits, min_length):
    """The requested splits from the cached, already decomposed datasets. 'all' is the set the paper evaluates
    on for IEMOCAP (train + validation + test) and VOC-ALS (train + validation + test + dev)."""
    cache_file_names = build_cache_file_names(data_training_args, data_training_args.input_type)
    load = lambda name: concatenate_datasets([Dataset.from_file(f) for f in cache_file_names[name]])
    ds = DatasetDict()
    for split in splits:
        ds[split] = concatenate_datasets([load(n) for n in cache_splits(split, data_training_args.dataset_name)])
    if min_length > 0.0:
        ds = ds.filter(lambda x: x > min_length, num_proc=data_training_args.preprocessing_num_workers,
                       input_columns=["input_length"])
    return ds.remove_columns("input_length")


def prepare_batch(batch, dataset_name, forward_args):
    """mask_time_indices and label removal as in the evaluation loop of latents_post_analysis.py: every frame
    for the simulated datasets, every non-padded frame (sub_attention_mask) for the real ones."""
    batch_size = batch["input_values"].shape[0]
    mask_indices_seq_length = batch["input_values"].shape[2]
    sub_attention_mask = batch.pop("sub_attention_mask", None)
    batch.pop("overlap_mask", None)
    if dataset_name in ["sim_vowels", "sim_coupled"]:
        batch["mask_time_indices"] = torch.ones((batch_size, mask_indices_seq_length), dtype=torch.bool,
                                                device=batch["mask_time_indices"].device)
    else:
        batch["mask_time_indices"] = sub_attention_mask.clone()
    "Labels and other keys the forward pass does not take"
    return {k: v for k, v in batch.items() if k in forward_args}


def parse_run(run, base_config):
    "The four argument dataclasses of a run: the base config with the run's overrides"
    parser = HfArgumentParser((ModelArgumentsPost, TrainingObjectiveArguments, DecompositionArguments, DataTrainingArgumentsPost))
    cfg = copy.deepcopy(base_config)
    cfg.update(run["overrides"])
    model_args, training_obj_args, decomp_args, data_training_args = parser.parse_dict(cfg)
    for obj, attr in [(model_args, "comment_model_args"), (training_obj_args, "comment_tr_obj_args"), (decomp_args, "comment_decomp_args")]:
        if hasattr(obj, attr):
            delattr(obj, attr)
    return model_args, training_obj_args, decomp_args, data_training_args


def unset_values(run):
    "Overrides and paths of a run still holding the SET placeholder"
    items = list(run["overrides"].items()) + [(k, run[k]) for k in ["checkpoint_dir", "checkpoint"] if k in run]
    return [k for k, v in items if isinstance(v, str) and UNSET in v]


def check_run(run, base_config, splits):
    "Pre-flight: the checkpoint and cache files a run needs exist. Returns a list of problems"
    unset = unset_values(run)
    if unset:
        return ["not set: " + ", ".join(unset)]
    model_args, training_obj_args, decomp_args, data_training_args = parse_run(run, base_config)
    problems = []
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


def run_one(run, base_config, splits, accelerator, max_batches=None):
    "One checkpoint: returns a list of long-format rows"
    model_args, training_obj_args, decomp_args, data_training_args = parse_run(run, base_config)
    if data_training_args.seed is not None:
        set_seed(data_training_args.seed)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)
    min_length = int(data_training_args.min_duration_in_seconds * feature_extractor.sampling_rate)
    model_args.max_duration_in_seconds = data_training_args.max_duration_in_seconds
    config = DecVAEConfig(**{**model_args.__dict__, **training_obj_args.__dict__, **decomp_args.__dict__})
    assert config.max_frames_per_batch == "all", "Use every frame, as in the post-training evaluation"

    checkpoint_dir = run.get("checkpoint_dir") or resolve_checkpoint_dir(model_args, training_obj_args, decomp_args, data_training_args)
    ckp = select_checkpoint(checkpoint_dir, run.get("checkpoint", -1))
    print(f"[{run['run_id']}] {os.path.join(checkpoint_dir, ckp)}")

    model = DecVAEForPreTraining(config)
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
        branches.append(("z", config.NoC, config.beta_kl_prior_z, config.prior_reg_weighting_z))
    if config.dual_branched_latent or config.only_s_branch:
        branches.append(("s", config.NoC_seq, config.beta_kl_prior_s, config.prior_reg_weighting_s))

    rows = []
    for split in splits:
        "Batch size of latents_post_analysis.py: train batch size for the single 'all' set, eval otherwise"
        batch_size = data_training_args.per_device_train_batch_size if split == "all" else data_training_args.per_device_eval_batch_size
        loader = accelerator.prepare(DataLoader(datasets_[split].with_format("numpy"), shuffle=False,
                                                collate_fn=data_collator, batch_size=batch_size))
        acc = {b: ComponentDivergences(NoC, b, beta, w) for b, NoC, beta, w in branches}
        start = time.time()
        with torch.no_grad():
            for step, batch in enumerate(loader):
                if max_batches is not None and step >= max_batches:
                    break
                inputs = prepare_batch(batch, data_training_args.dataset_name, forward_args)
                outputs = model(**inputs)
                for a in acc.values():
                    a.add(outputs)
                del batch, inputs, outputs
        for b, a in acc.items():
            summary = a.summary()
            if summary["check_max_abs_diff"] > 1e-4:
                print(f"WARNING [{run['run_id']} {b} {split}]: recomputed batch values differ from the model's by {summary['check_max_abs_diff']:.2e}")
            for measure, value in summary.items():
                rows.append({"dataset": data_training_args.dataset_name, "model": run["model"], "beta": run["beta"],
                             "transfer_from": run.get("transfer_from", ""), "NoC": config.NoC if b == "z" else config.NoC_seq,
                             "run_id": run["run_id"], "checkpoint": ckp, "branch": b, "split": split,
                             "measure": measure, "value": value})
        print(f"[{run['run_id']}] {split}: {time.time() - start:.0f} s")
    del model
    torch.cuda.empty_cache()
    return rows


def expand_runs(spec):
    """Runs from the runs file: every model x beta, with {btag} in parent_dir (or in checkpoint_dir, which
    bypasses resolve_checkpoint_dir) replaced by the beta tag.
    An entry of 'exceptions' (keyed '<model>_b<beta>') adds or replaces overrides for that run only, e.g. a
    parent_dir on another drive, or a checkpoint_dir that bypasses resolve_checkpoint_dir."""
    runs = []
    for m in spec["models"]:
        for beta in m.get("betas", spec.get("betas")):
            run_id = f"{m['name']}_b{beta}" + (f"_{m['transfer_from']}" if m.get("transfer_from") else "")
            ov = copy.deepcopy(m.get("overrides", {}))
            ov["beta_kl_prior_z"] = float(beta)
            ov["beta_kl_prior_s"] = float(beta)
            if "parent_dir" in ov:
                ov["parent_dir"] = ov["parent_dir"].replace("{btag}", beta_tag(beta))
            run = {"run_id": run_id, "model": m["name"], "beta": beta, "transfer_from": m.get("transfer_from", ""), "overrides": ov}
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
    ap = argparse.ArgumentParser(description="Decomposition divergences of trained DecVAE checkpoints (beta trade-off).")
    ap.add_argument("--config_file", required=True, help="runs file, e.g. config_files/beta_tradeoff/runs_sim_vowels.json")
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
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) and not opts["overwrite"] else pd.DataFrame(columns=["run_id", "split"])

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
        todo = [s for s in spec["splits"] if not ((done["run_id"] == run["run_id"]) & (done["split"] == s)).any()]
        if not todo:
            print(f"[{run['run_id']}] already in {out_csv}, skipped")
            continue
        try:
            rows = run_one(run, base_config, todo, accelerator, opts["max_batches"])
        except FileNotFoundError as e:
            print(f"[{run['run_id']}] SKIPPED, checkpoint or cache not found: {e}")
            continue
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
        done.to_csv(out_csv, index=False)  # after every run, so an interruption loses at most one run
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
