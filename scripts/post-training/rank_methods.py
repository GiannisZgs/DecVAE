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

"""Rank the methods of the results tables in the baselines planner workbook.

python scripts/post-training/rank_methods.py --config_file config_files/sensitivity/config_rank_methods.json
Config keys: workbook, sheet, datasets, skip_fill, exclude_regex, one_variant, out.
skip_fill: ARGB fill of the first cell of rows to drop, e.g. FFFFC000.
one_variant: {dataset: {method_regex: row_to_keep}}, inline or as a JSON file path; among the rows
whose name matches method_regex, only row_to_keep is kept.
"""

import json
import logging
import os
import re
import sys
import types
import openpyxl
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from latent_analysis_utils.ranking_utils import METRICS, rank_methods
from utils import parse_args, debugger_is_active

JSON_FILE_NAME_MANUAL = "config_files/sensitivity/config_rank_methods.json" #for debugging purposes only

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "workbook": None,
    "sheet": "Results",
    "datasets": ["SimVowels", "TIMIT"],
    "skip_fill": None,
    "exclude_regex": None,
    "one_variant": None,
    "out": "rankings.xlsx",
}


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("comment")}
    unknown = set(cfg) - set(DEFAULT_ARGS)
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    args = types.SimpleNamespace(**{**DEFAULT_ARGS, **cfg})
    if args.workbook is None:
        raise ValueError(f"workbook must be set in {path}")
    return args


def normalize_name(name):
    name = re.sub(r"\s+", " ", str(name).strip())
    return name.replace("β- ", "β-")


def header_to_metric(header):
    "'Task A: vowel' -> 'Task A'; other headers are kept as they are"
    header = str(header).strip()
    match = re.match(r"^(Task [A-Z])\b", header)
    return match.group(1) if match else header


def read_tables(workbook, sheet, datasets, skip_fill=None):
    """
    Returns:
        {dataset: DataFrame indexed by method, one column per header}.
    """
    ws = openpyxl.load_workbook(workbook, data_only=True)[sheet]
    rows = list(ws.iter_rows())
    tables = {}
    for i, row in enumerate(rows):
        title = row[0].value
        if title is None or str(title).strip() not in datasets:
            continue
        dataset = str(title).strip()
        h = i + 1
        while h < len(rows) and str(rows[h][0].value).strip() != "Method":
            h += 1
        headers = [header_to_metric(c.value) if c.value is not None else None for c in rows[h]]
        records, names = [], []
        for data_row in rows[h + 1:]:
            first = data_row[0]
            if first.value is None or str(first.value).strip() == "":
                break
            fill = first.fill.fgColor.rgb if first.fill is not None and first.fill.fill_type else None
            if skip_fill and fill == skip_fill:
                logger.info(f"{dataset}: skipping '{first.value}' (fill {fill})")
                continue
            names.append(normalize_name(first.value))
            records.append({hd: c.value for hd, c in zip(headers[1:], data_row[1:]) if hd is not None})
        tables[dataset] = pd.DataFrame(records, index=pd.Index(names, name="Method"))
    missing = [d for d in datasets if d not in tables]
    if missing:
        raise ValueError(f"No table titled {missing} in sheet '{sheet}'")
    return tables


def filter_rows(df, dataset, exclude_regex=None, one_variant=None):
    keep = pd.Series(True, index=df.index)
    if exclude_regex:
        keep &= ~df.index.to_series().str.contains(exclude_regex, regex=True)
    for pattern, kept in (one_variant or {}).get(dataset, {}).items():
        matches = df.index.to_series().str.contains(pattern, regex=True)
        keep &= ~matches | (df.index == normalize_name(kept))
    dropped = list(df.index[~keep])
    if dropped:
        logger.info(f"{dataset}: excluded {dropped}")
    return df[keep]


def write_block(writer, sheet, df, startrow, title):
    pd.DataFrame([[title]]).to_excel(writer, sheet_name=sheet, startrow=startrow, header=False, index=False)
    df.to_excel(writer, sheet_name=sheet, startrow=startrow + 1)
    return startrow + len(df) + 4


def main():
    "Parse the arguments"
    if debugger_is_active():
        args = load_config(JSON_FILE_NAME_MANUAL)
    else:
        args = load_config(parse_args().config_file)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    one_variant = args.one_variant
    if isinstance(one_variant, str):
        with open(one_variant) as f:
            one_variant = json.load(f)
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tables = read_tables(args.workbook, args.sheet, args.datasets, args.skip_fill)
    summary = []
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        for dataset, df in tables.items():
            df = filter_rows(df, dataset, args.exclude_regex, one_variant)
            res = rank_methods(df, METRICS)
            logger.info(f"{dataset}: {len(df)} methods, metrics used {res['used_metrics']}")

            order = res["aggregations"]["Flat"].sort_values().index
            row = 0
            row = write_block(writer, dataset, res["values"].loc[order], row, "Metric values")
            row = write_block(writer, dataset, res["metric_ranks"].loc[order], row, "Per-metric ranks (1 is best)")
            row = write_block(writer, dataset, res["family_ranks"].loc[order], row, "Family ranks")
            agg = pd.concat([res["aggregations"], res["aggregation_positions"].add_suffix(" position")], axis=1)
            row = write_block(writer, dataset, agg.loc[order], row, "Aggregations (mean rank)")
            lofo = pd.concat([res["lofo"], res["lofo_positions"].add_suffix(" position")], axis=1)
            row = write_block(writer, dataset, lofo.loc[order], row, "Leave-one-family-out")
            write_block(writer, dataset, res["rank_range"].loc[order], row, "Rank range over all aggregations")

            s = pd.concat([res["aggregations"], res["aggregation_positions"].add_suffix(" position"),
                           res["rank_range"]], axis=1)
            s.insert(0, "Dataset", dataset)
            summary.append(s.loc[order])

            for name in res["aggregations"].columns:
                top = res["aggregations"][name].sort_values().head(5)
                logger.info(f"  {name}: " + "; ".join(f"{m} {v:.1f}" for m, v in top.items()))

        pd.concat(summary).to_excel(writer, sheet_name="Summary")
    logger.info(f"Saved {args.out}")


if __name__ == "__main__":
    main()
