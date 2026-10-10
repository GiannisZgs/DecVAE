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

"""One row per VOC-ALS subject with its clinical labels, for the confound analysis (R5 #9).

Reads the label keys of the preprocessed parts written by voc_als_prep.py (the audio is skipped while
streaming) and encodes missing values as the pretraining preprocessing does (None -> -1), so the
labels match the ones in the eval dumps. subject = speaker_id_encoded = the dumps' speaker factor.

python scripts/misc/voc_als_subject_table.py --data_dir ../VOC-ALS_preprocessed --output data/voc_als_confounding_analysis/subject_table.csv
"""

import argparse
import gzip
import json
import os
import pandas as pd

LABEL_KEYS = ["speaker_id", "speaker_id_encoded", "category_encoded", "king_stage", "disease_duration",
              "alsfrs_total", "alsfrs_speech", "cantagallo"]


def read_labels(path):
    "Top-level label lists of one part, without loading the audio (indent=2 layout of voc_als_prep.py)"
    sections, key, buf = {}, None, []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith('  "'):
                if key in LABEL_KEYS:
                    sections[key] = json.loads("".join(buf).rstrip().rstrip(","))
                key, rest = line.strip().split(":", 1)
                key, buf = key.strip('"'), [rest]
            elif key in LABEL_KEYS and line.rstrip() != "}":
                buf.append(line)
    if key in LABEL_KEYS:
        sections[key] = json.loads("".join(buf).rstrip().rstrip(","))
    return pd.DataFrame(sections)


def to_int(v):
    "None and '-' are healthy-control placeholders, encoded as -1 as in pretraining_preprocessing"
    return -1 if v is None or v == "-" or pd.isna(v) else int(float(v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../VOC-ALS_preprocessed")
    parser.add_argument("--output", default="data/voc_als_confounding_analysis/subject_table.csv")
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".json.gz"))
    df = pd.concat([read_labels(os.path.join(args.data_dir, f)) for f in files], ignore_index=True)
    for c in ["king_stage", "disease_duration", "alsfrs_total", "alsfrs_speech", "cantagallo"]:
        df[c] = df[c].map(to_int)
    df = df.rename(columns={"speaker_id_encoded": "subject", "category_encoded": "category", "speaker_id": "subject_name"})

    "Every label is per subject"
    per_subject = df.groupby("subject").nunique()
    if (per_subject > 1).any().any():
        raise ValueError(f"labels vary within a subject: {per_subject.columns[(per_subject > 1).any()].tolist()}")
    table = df.groupby("subject").first().reset_index()
    table["n_recordings"] = df.groupby("subject").size().to_numpy()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"{len(table)} subjects saved to {args.output}")
    print(pd.crosstab(table["king_stage"], table["category"], margins=True))
    print(pd.crosstab(table["disease_duration"], table["category"], margins=True))


if __name__ == "__main__":
    main()
