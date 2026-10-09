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

"""Tables of the decomposition sensitivity sweep (C and FD search intervals).

Collects, per run of models_decomp_sensitivity.json: the standard metric suite written by
latents_post_analysis.py, and the subspace analysis written by subspace_analysis.py (per_model/).
Adds the effective number of OC subspaces per factor, exp(entropy) of the factor's column over the OC
rows of a matrix: 1 when one OC carries the factor, C when it is spread evenly over all of them.
The figure data goes to figure_data_dir as tidy CSVs, drawn by visualize_R/SI_decomp_sensitivity/.

python scripts/post-training/decomp_sensitivity_summary.py --config_file config_files/decomp_sensitivity/config_decomp_sensitivity_summary.json
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

JSON_FILE_NAME_MANUAL = "config_files/decomp_sensitivity/config_decomp_sensitivity_summary.json" #for debugging purposes only

DEFAULT_ARGS = {
    "models_file": None,
    "results_root": None,
    "subspace_dir": None,
    "output_dir": None,
    "figure_data_dir": "data/decomp_sensitivity",
}
"Keys of the frame-level results JSON of latents_post_analysis.py"
STANDARD = ["disentanglement", "completeness", "informativeness_test", "modularity_score", "explicitness_score_test",
            "IRS", "mutual_info_score", "gaussian_total_correlation", "gaussian_total_correlation_norm"]
SEL = ["P", "A", "dT"]
ID_COLS = ["dataset", "run", "group", "seed", "C"]


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("comment")}
    unknown = set(cfg) - set(DEFAULT_ARGS)
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    args = {**DEFAULT_ARGS, **cfg}
    missing = [k for k, v in args.items() if v is None]
    if missing:
        raise ValueError(f"Set {missing} in {path}")
    return args


def effective_subspaces(M):
    "Per factor, exp(entropy) of the factor's column over the OC rows (row 0 is X), negatives clipped"
    oc = np.clip(np.nan_to_num(np.asarray(M, float)[1:]), 0, None)
    out = np.full(oc.shape[1], np.nan)
    for f in range(oc.shape[1]):
        tot = oc[:, f].sum()
        if tot > 0:
            p = oc[:, f] / tot
            p = p[p > 0]
            out[f] = float(np.exp(-(p * np.log(p)).sum()))
    return out


def standard_metrics(results_dir):
    "The standard suite of one run: the frame-level results JSON of the 'all' latent, and its _indep twin if any"
    out = {}
    for suffix, prefix in (("_frame.csv", ""), ("_frame_indep.csv", "indep:")):
        paths = glob.glob(os.path.join(results_dir, "*", f"*_all_*disentanglement_results{suffix}"))
        if len(paths) > 1:
            raise ValueError(f"several results files *_all_*{suffix} in {results_dir} (several checkpoints?): {paths}")
        if paths:
            with open(paths[0]) as f:
                res = json.load(f)
            out.update({prefix + k: float(res[k]) for k in STANDARD if k in res})
    if not out:
        print(f"no standard metric results in {results_dir}")
    return out


def subspace_metrics(record):
    "Scalars of one per_model record of subspace_analysis.py, every run (main, and indep for SimCoupled)"
    out = {}
    for run, r in record["runs"].items():
        p = "" if run == "main" else f"{run}:"
        s, factors = r["summary"], r["factors"]
        for j, fac in enumerate(factors):
            out[f"{p}probe:{fac}"] = s["full_scores"]["mean"][j]
        for m in SEL:
            out[f"{p}sel_OCs:{m}"] = s["selectivity"][m]["OCs"]["mean"]
            out[f"{p}sel_all:{m}"] = s["selectivity"][m]["all"]["mean"]
            neff = effective_subspaces(s[m]["mean"])
            for j, fac in enumerate(factors):
                out[f"{p}concentration:{m}:{fac}"] = s["concentration"][m]["mean"][j]
                out[f"{p}n_eff:{m}:{fac}"] = neff[j]
            if "null" in s:
                out[f"{p}null_gap:{m}"] = s["null"][m]["gap"]
                out[f"{p}null_z:{m}"] = s["null"][m]["z"]
    return out


def collect(models, results_root, subspace_dir):
    rows = []
    for m in models:
        ds, tag = m["dataset"], m["name"][len(m["dataset"]) + 1:]
        row = {"dataset": ds, "run": tag, "group": m["config"], "seed": m["seed"], "C": m["n_blocks"] - 1}
        row.update(standard_metrics(os.path.join(results_root, ds, tag)))
        rec_path = os.path.join(subspace_dir, "per_model", f"{m['name']}.json")
        if os.path.exists(rec_path):
            with open(rec_path) as f:
                row.update(subspace_metrics(json.load(f)))
        else:
            print(f"no subspace results for {m['name']}")
        rows.append(row)
    return pd.DataFrame(rows)


def group_table(df):
    "Mean and std over the members of each group (C3 seeds, the three random draws); single runs have std NaN"
    num = [c for c in df.columns if c not in ID_COLS]
    g = df.groupby(["dataset", "group"], sort=False)
    out = g[num].agg(["mean", "std"])
    out.columns = [f"{a}|{b}" for a, b in out.columns]
    out.insert(0, "n_runs", g.size())
    out.insert(1, "C", g["C"].first())
    return out.reset_index()


def write_figure_data(df, fig_dir):
    "Long tables: one row per run and metric, and one row per group and metric (mean, std, n)"
    os.makedirs(fig_dir, exist_ok=True)
    long = df.melt(id_vars=ID_COLS, var_name="metric", value_name="value").dropna(subset=["value"])
    long.to_csv(os.path.join(fig_dir, "sweep_runs_long.csv"), index=False)
    grp = long.groupby(["dataset", "group", "C", "metric"], sort=False)["value"].agg(
        mean="mean", std=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan, n="size").reset_index()
    grp.to_csv(os.path.join(fig_dir, "sweep_groups_long.csv"), index=False)


def main():
    from utils import parse_args, debugger_is_active
    args = load_config(JSON_FILE_NAME_MANUAL if debugger_is_active() else parse_args().config_file)
    with open(args["models_file"], encoding="utf-8") as f:
        models = json.load(f)
    os.makedirs(args["output_dir"], exist_ok=True)
    df = collect(models, args["results_root"], args["subspace_dir"])
    df.to_csv(os.path.join(args["output_dir"], "decomp_sensitivity_runs.csv"), index=False)
    group_table(df).to_csv(os.path.join(args["output_dir"], "decomp_sensitivity_groups.csv"), index=False)
    write_figure_data(df, args["figure_data_dir"])
    print(f"Saved the tables in {args['output_dir']} and the figure data in {args['figure_data_dir']}")


if __name__ == "__main__":
    main()
