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

"""Tables for the beta trade-off figure: the decomposition divergences of every
checkpoint (decomposition_divergences.py) next to the downstream and disentanglement metrics the paper already
reports for the same checkpoints (the wandb exports behind SI Fig 12, SI Fig 14a and Fig 5h).

    python scripts/post-training/beta_tradeoff_tables.py --config_file config_files/beta_tradeoff/config_beta_tradeoff_tables.json
    python scripts/post-training/beta_tradeoff_tables.py --config_file ... --provisional   # also the last-epoch validation
                                                                                           # divergences logged during
                                                                                           # pre-training (SI Fig 22a)

Config keys: see DEFAULT_ARGS. Writes <output_dir>/tradeoff_long.csv (one row per dataset, model, beta, source and
metric) and <output_dir>/tradeoff_wide.csv (one row per dataset, model, beta, source; one column per metric),
which visualize_R/SI_beta_tradeoff/beta_tradeoff.R reads.
"""

import os
import re
import sys
import json
import types
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils import debugger_is_active

JSON_FILE_NAME_MANUAL = "config_files/beta_tradeoff/config_beta_tradeoff_tables.json" #for debugging purposes only

DEFAULT_ARGS = {
    "exports_dir": os.path.join("data", "results_wandb_exports_for_figures"),
    "output_dir": os.path.join("data", "beta_tradeoff"),  # also where the divergences_<dataset>.csv files are read from
    "provisional": False,  # add the pre-training log divergences for SimVowels
    "divergence_files": {  # per dataset, the decomposition_divergences.py output in output_dir
        "sim_vowels": "divergences_sim_vowels_100frames.csv",
        "timit": "divergences_timit_178frames.csv",
        "iemocap": "divergences_iemocap_175frames.csv",
    },
}
MODEL_NAMES = {"filter": "FD", "ewt": "EWT", "emd": "EMD", "vmd": "VMD"}

"Which divergence rows go into the figure: frame branch, the split the paper's metrics come from"
DIVERGENCE_SPLITS = {"sim_vowels": "test", "timit": "test", "iemocap": "all"}
DIVERGENCE_MEASURES = ["d_ortho", "d_recon", "d_ortho_norm", "d_recon_norm", "d_ortho_utt", "d_recon_utt",
                       "model_div_neg", "model_div_pos", "rate_per_frame", "clamp_frac_ortho", "clamp_frac_recon",
                       "check_max_abs_diff"]


def load_config(path):
    if path is None:
        return types.SimpleNamespace(**DEFAULT_ARGS)
    with open(path) as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("comment")}
    unknown = set(cfg) - set(DEFAULT_ARGS)
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    return types.SimpleNamespace(**{**DEFAULT_ARGS, **cfg})


def parse_beta(tag):
    "'0.1' or '01' -> 0.1; '0' -> 0; '5' -> 5"
    return 0.1 if tag in ["0.1", "01"] else float(tag)


def pick(values, use_row):
    "Value of a wandb export column, with the use_row rule of the R scripts (-1 last, -2 second-to-last non-NA)"
    v = values.dropna().to_numpy()
    if len(v) == 0:
        return np.nan
    if len(v) == 1:
        return float(v[0])
    return float(v[len(v) + use_row])


def read_wandb_export(path, pattern, use_row):
    """Rows (model, beta, value) from one wandb export. pattern: regex with groups 'beta' and 'model' matched
    against the column name. __MIN/__MAX columns are ignored."""
    df = pd.read_csv(path)
    rows = []
    for col in df.columns:
        if col == "Step" or "__MIN" in col or "__MAX" in col:
            continue
        m = re.search(pattern, col)
        if m:
            rows.append({"model": MODEL_NAMES.get(m.group("model").lower(), m.group("model")),
                         "beta": parse_beta(m.group("beta")), "value": pick(df[col], use_row)})
    return pd.DataFrame(rows)


def paper_metrics_sim_vowels(exports_dir):
    "SI Fig 12 inputs: vowels_posttraining/beta_ablation/Z_branch/<metric>_all.csv, second-to-last value"
    out = []
    for metric in ["accuracy_vowel", "accuracy_speaker", "disentanglement", "completeness", "informativeness", "mi"]:
        path = os.path.join(exports_dir, "vowels_posttraining", "beta_ablation", "Z_branch", f"{metric}_all.csv")
        d = read_wandb_export(path, r"bz(?P<beta>[0-9.]+)_bs[0-9.]+_(?P<model>filter|ewt|emd|vmd)_NoC3", use_row=-2)
        out.append(d.assign(dataset="sim_vowels", metric=metric))
    return pd.concat(out)


def paper_metrics_timit(exports_dir):
    "SI Fig 14a inputs: timit_posttraining/beta_ablation/Z_branch/<metric>_all.csv, last value"
    out = []
    for metric in ["disentanglement", "completeness", "informativeness", "mi"]:
        path = os.path.join(exports_dir, "timit_posttraining", "beta_ablation", "Z_branch", f"{metric}_all.csv")
        d = read_wandb_export(path, r"TIMIT_(?P<model>filter|ewt)_dual_NoC4_b(?P<beta>0|01|1)_", use_row=-1)
        out.append(d.assign(dataset="timit", metric=metric))
    return pd.concat(out)


def paper_metrics_iemocap(exports_dir):
    "Fig 5h inputs: iemocap_posttraining/total_results/Z_branch/<metric>.xls; first row mean, second row CI half-width"
    out = []
    for metric in ["accuracy_emotion", "unweighted_accuracy_emotion", "accuracy_speaker", "disentanglement"]:
        try:
            df = pd.read_excel(os.path.join(exports_dir, "iemocap_posttraining", "total_results", "Z_branch", f"{metric}.xls"))
        except ImportError as e:
            print(f"IEMOCAP metrics skipped, the .xls exports need xlrd ({e})")
            return pd.DataFrame()
        for col in df.columns:
            m = re.fullmatch(r"(?P<model>filter|ewt)_(?P<src>vowels|timit)_NoC4_b(?P<beta>0|01|1)", str(col))
            if m:
                v = df[col].dropna().to_numpy()
                out.append({"dataset": "iemocap", "model": MODEL_NAMES[m.group("model")], "transfer_from": m.group("src"),
                            "beta": parse_beta(m.group("beta")), "metric": metric, "value": float(v[0]),
                            "ci": float(v[1]) if len(v) > 1 else np.nan})
    return pd.DataFrame(out)


def divergences(output_dir, divergence_files):
    "Checkpoint divergences written by decomposition_divergences.py (frame branch)"
    out = []
    for dataset, fname in divergence_files.items():
        split = DIVERGENCE_SPLITS[dataset]
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            print(f"not found, skipped: {path}")
            continue
        d = pd.read_csv(path)
        d = d[(d["branch"] == "z") & (d["split"] == split) & d["measure"].isin(DIVERGENCE_MEASURES)]
        d = d.rename(columns={"measure": "metric"})[["dataset", "model", "transfer_from", "beta", "metric", "value"]]
        out.append(d.assign(source="checkpoint"))
    return pd.concat(out) if out else pd.DataFrame()


def provisional_divergences(exports_dir):
    """Last-epoch validation divergences logged during pre-training (SI Fig 22a exports): SimVowels FD (all betas)
    and EWT (0 to 5). These are the model's own values (clamped utterance sums, last epoch
    rather than the evaluated checkpoint), for a first look only."""
    base = os.path.join(exports_dir, "vowels_pretraining", "FD_decomposition_loss_demo_betas", "dual")
    out = []
    for prefix in ["", "EWT_"]:
        for kind, metric in [("negative", "model_div_neg"), ("positive", "model_div_pos")]:
            path = os.path.join(base, f"{prefix}validation_divergence_{kind}_Z.csv")
            if os.path.exists(path):
                d = read_wandb_export(path, r"VOWELS_(?P<model>FILTER|EWT)_[bB](?P<beta>[0-9.]+)", use_row=-1)
                out.append(d.assign(dataset="sim_vowels", metric=metric, source="wandb_last_epoch"))
    return pd.concat(out) if out else pd.DataFrame()


def parse_cli():
    ap = argparse.ArgumentParser(description="Tables for the beta trade-off figure.")
    ap.add_argument("--config_file", default=None, help="config with the keys of DEFAULT_ARGS (default: DEFAULT_ARGS)")
    ap.add_argument("--provisional", action="store_true", default=None, help="add the pre-training log divergences for SimVowels")
    return ap.parse_args()


def main():
    "Parse the arguments"
    if debugger_is_active():
        args = load_config(JSON_FILE_NAME_MANUAL)
    else:
        cli = parse_cli()
        args = load_config(cli.config_file)
        if cli.provisional is not None:
            args.provisional = cli.provisional
    os.makedirs(args.output_dir, exist_ok=True)

    paper = pd.concat([paper_metrics_sim_vowels(args.exports_dir), paper_metrics_timit(args.exports_dir),
                       paper_metrics_iemocap(args.exports_dir)]).assign(source="paper_eval")
    parts = [paper, divergences(args.output_dir, args.divergence_files)]
    if args.provisional:
        parts.append(provisional_divergences(args.exports_dir))
    long = pd.concat([p for p in parts if len(p)], ignore_index=True)
    long["transfer_from"] = long["transfer_from"].fillna("") if "transfer_from" in long else ""
    if "ci" not in long:
        long["ci"] = np.nan
    long = long[["dataset", "model", "transfer_from", "beta", "source", "metric", "value", "ci"]]
    long = long.sort_values(["dataset", "model", "transfer_from", "beta", "source", "metric"])
    long.to_csv(os.path.join(args.output_dir, "tradeoff_long.csv"), index=False)

    "Wide: divergences of each source next to the paper's metrics of the same checkpoint"
    keys = ["dataset", "model", "transfer_from", "beta"]
    metrics = long[long["source"] == "paper_eval"].pivot_table(index=keys, columns="metric", values="value").reset_index()
    cis = long[(long["source"] == "paper_eval") & long["ci"].notna()].pivot_table(index=keys, columns="metric", values="ci")
    cis.columns = [c + "_ci" for c in cis.columns]
    metrics = metrics.merge(cis.reset_index(), on=keys, how="left") if len(cis) else metrics
    div = long[long["source"] != "paper_eval"]
    if len(div) == 0:
        print("No divergences yet (run decomposition_divergences.py, or set provisional); tradeoff_wide.csv not written")
        return
    div = div.pivot_table(index=keys + ["source"], columns="metric", values="value").reset_index()
    wide = div.merge(metrics, on=keys, how="left")
    wide.to_csv(os.path.join(args.output_dir, "tradeoff_wide.csv"), index=False)
    print(wide.groupby(["dataset", "source"]).size().rename("rows").to_string())
    print(f"Saved {args.output_dir}/tradeoff_long.csv and tradeoff_wide.csv")


if __name__ == "__main__":
    main()
