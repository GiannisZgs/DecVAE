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

"""Pilot timing of one metric suite run on a SimVowels eval dump.

python scripts/post-training/metric_suite_pilot_timing.py --config_file config_files/sensitivity/config_metric_suite_pilot_timing.json
Config keys: dump (an eval dump .npz), n_sub, n_jobs.
"""

import json
import os
import sys
import time
import types
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from disentanglement_utils.metric_suite import compute_metric_suite, load_dump, args_from_sidecar, table_columns
from utils import parse_args, debugger_is_active

JSON_FILE_NAME_MANUAL = "config_files/sensitivity/config_metric_suite_pilot_timing.json" #for debugging purposes only

DEFAULT_ARGS = {
    "dump": None,
    "n_sub": 20000,
    "n_jobs": 1,
}


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("comment")}
    unknown = set(cfg) - set(DEFAULT_ARGS)
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    args = types.SimpleNamespace(**{**DEFAULT_ARGS, **cfg})
    if args.dump is None:
        raise ValueError(f"dump must be set in {path}")
    return args


def main():
    "Parse the arguments"
    if debugger_is_active():
        args = load_config(JSON_FILE_NAME_MANUAL)
    else:
        args = load_config(parse_args().config_file)

    mus, ys, n_train, sidecar = load_dump(args.dump)
    idx = np.sort(np.random.default_rng(0).choice(mus.shape[1], size=min(args.n_sub, mus.shape[1]), replace=False))
    start = time.time()
    res = compute_metric_suite(mus[:, idx], ys[:, idx], int(np.sum(idx < n_train)), sidecar["targets"],
                               args_from_sidecar(sidecar, args.n_jobs))
    print(f"Dump {sidecar['tag']}: N = {len(idx)}, d = {mus.shape[0]}: {time.time() - start:.1f} s")
    print(table_columns(res))


if __name__ == "__main__":
    main()
