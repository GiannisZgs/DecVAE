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

"""VOC-ALS confound analysis (reviewer comment R5 #9): is disease-stage prediction driven by speaker
identity? On the eval dumps of every model, for each subject-level target and for both branches
(frame: one unit per frame; seq: one unit per utterance), each with the paper's Random Forest and
with the linear probe of the subspace analysis:

  A. unit-level probes under speaker-shared folds (the paper's protocol) and subject-grouped folds,
     the subject-level score derived from the grouped unit probe, and the speaker-lookup bound;
  B. subject-level probes (one row per subject), repeated stratified K-fold;
  C. subject-level permutation tests for B, labels shuffled across all subjects and within ALS and
     within controls (permute_within_category), and, across all subjects, for the grouped unit probe
     of the models listed in the level's unit_permutation_models;
  D. subject probes inside strata (phoneme groups, ALS only), optionally on a coarser label set
     (strata_merge) so that every class has enough subjects for the folds;
  E. bootstrap intervals over subjects.

The seq branch uses the model's sequence-level dump ("dump_seq": DecVAE's S branch) when the models
file gives one, and otherwise the mean of each utterance's frames from the frame dump.

Writes per-model JSON to output_dir/per_model and long CSVs for visualize_R/SI_voc_als_confounding_analysis/ to
figure_data_dir.

python scripts/post-training/voc_als_confound_analysis.py --config_file config_files/voc_als_confounding_analysis/config_voc_als_confound.json
The models file is a list of {"name", "label", "dump", optional "dump_seq"}.
"""

import json
import os
import sys
import time
import warnings
import zlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from disentanglement_utils.metric_suite import load_dump
from latent_analysis_utils import voc_als_confound_utils as vu
from utils import parse_args, debugger_is_active

JSON_FILE_NAME_MANUAL = "config_files/voc_als_confounding_analysis/config_voc_als_confound.json" #for debugging purposes only

DEFAULT_ARGS = {
    "models_file": "config_files/voc_als_confounding_analysis/models_voc_als_confounding_analysis.json",
    "subject_table": "data/voc_als_confounding_analysis/subject_table.csv",
    "output_dir": "../post-training_results/voc_als_confounding_analysis",
    "figure_data_dir": "data/voc_als_confounding_analysis",
    "levels": {
        "frame": {"speaker_factor": "speaker_frame", "phoneme_factor": "phoneme_frame", "stage_factor": "king_stage_frame",
                  "n_sub": 30000, "n_sub_unit_perm": 10000, "unit_permutation_models": []},
        "seq": {"speaker_factor": "speaker_seq", "phoneme_factor": "phoneme_seq", "stage_factor": "king_stage_seq",
                "n_sub": None, "n_sub_unit_perm": None, "unit_permutation_models": "all"},
    },
    "targets": {"king_stage": {"column": "king_stage", "merge": None},
                "disease_duration": {"column": "disease_duration", "merge": None}},
    "n_folds": 5,
    "fold_seed": 0,
    "subsample_seed": 0,
    "probe_C": 1.0,
    "probe_max_iter": 1000,
    "probes": ["rf", "lr"],
    "subject_feature_modes": ["mean"],
    "subject_repeats": 10,
    "n_boot": 1000,
    "n_perm_subject": 1000,
    "n_perm_unit": 100,
    "perm_seed": 0,
    "permute_within_category": [False, True],  # subject-level nulls: shuffled across all subjects, and within ALS / controls
    "strata": {},
    "strata_merge": {},  # target -> label merge used by the strata only, e.g. {"king_stage": {"6": 5}}
    "n_jobs": 1,  # Random Forest fits on the unit probes, and parallel permutations (one job each)
    "recompute": False,
}

"Name of each subject-level null, by its permute_within_category value"
NULL_SCHEMES = {False: "all_subjects", True: "within_category"}


def load_args(cfg_file):
    "Config over DEFAULT_ARGS; unknown keys are rejected"
    with open(cfg_file, encoding="utf-8") as f:
        cfg = json.load(f)
    unknown = set(cfg) - set(DEFAULT_ARGS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return {**DEFAULT_ARGS, **cfg}


def quiet_warnings():
    "One subject in a class (stage 4A/B) makes StratifiedKFold warn on every split"
    warnings.filterwarnings("ignore", message="The least populated class")
    warnings.filterwarnings("once", category=ConvergenceWarning)


def dump_units(path, level_args):
    "Latents (N, D) and unit labels (subject, phoneme, stage) of one dump"
    mus, ys, _, sidecar = load_dump(path)
    targets = list(sidecar["targets"])
    units = pd.DataFrame({"subject": ys[targets.index(level_args["speaker_factor"])].astype(int),
                          "phoneme": ys[targets.index(level_args["phoneme_factor"])].astype(int),
                          "stage_dump": ys[targets.index(level_args["stage_factor"])].astype(int)})
    return mus.T, units


def join_subject_table(units, subject_table, name):
    "Clinical labels per unit from the subject table; the dump's own stage labels must agree with it"
    units = units.merge(subject_table, on="subject", how="left", validate="many_to_one")
    if units["category"].isna().any():
        raise ValueError(f"{name}: units of subjects missing from the subject table")
    if (units["stage_dump"] != units["king_stage"]).any():
        raise ValueError(f"{name}: King's stage in the dump disagrees with the subject table - subject ids do not match")
    return units


def load_model_units(model, subject_table, args):
    """Units of every level for one model.

    Returns {level: (Z, units, source)}.
    """
    out = {}
    Z, units = dump_units(model["dump"], args["levels"].get("frame", DEFAULT_ARGS["levels"]["frame"]))
    if "frame" in args["levels"]:
        out["frame"] = (Z, join_subject_table(units, subject_table, model["name"]), "frame dump")
    if "seq" in args["levels"]:
        if model.get("dump_seq"):
            Zq, uq = dump_units(model["dump_seq"], args["levels"]["seq"])
            source = "sequence dump"
        else:
            "One recording per subject and phoneme, so the stage of an utterance is its subject's"
            Zq, subj, phon = vu.pool_utterances(Z, units["subject"].to_numpy(), units["phoneme"].to_numpy())
            stage = units.groupby("subject")["stage_dump"].first().loc[subj].to_numpy()
            uq = pd.DataFrame({"subject": subj, "phoneme": phon, "stage_dump": stage})
            source = "mean-pooled frames"
        out["seq"] = (Zq, join_subject_table(uq, subject_table, model["name"]), source)
    return out


def unit_signature(units):
    "Checksums of the unit labels, in dump order and as a set, to compare unit sets across models"
    a = units[["subject", "phoneme", "king_stage"]].to_numpy().astype(np.int64)
    s = a[np.lexsort(a.T[::-1])]
    return {"n_units": int(len(a)), "crc_order": int(zlib.crc32(a.tobytes())), "crc_set": int(zlib.crc32(s.tobytes()))}


def target_rows(units, target, args, extra_merge=None):
    "Units that carry the target, labels after the optional merges (the target's, then extra_merge), encoded as 0..K-1"
    t = args["targets"][target]
    y = units[t["column"]].to_numpy()
    keep = ~pd.isna(y)
    for merge in (t.get("merge"), extra_merge):
        if merge:
            y = np.array([merge.get(str(int(v)), v) if not pd.isna(v) else v for v in y])
    classes, y_enc = np.unique(y[keep].astype(int), return_inverse=True)
    return keep, y_enc, classes


def _subject_null(F, lab, strata, i, kw, args):
    quiet_warnings()
    perm = vu.permute_subject_labels(lab, np.random.default_rng(args["perm_seed"] + i), strata)
    return vu.subject_probe(F, perm, args["n_folds"], [0], **{**kw, "n_jobs": 1})[0]["balanced_accuracy"]


def _unit_null(Z, subj_ids, lab, subj, i, kw, args):
    quiet_warnings()
    perm = vu.permute_subject_labels(lab, np.random.default_rng(args["perm_seed"] + 10_000 + i))
    y = perm[np.searchsorted(subj_ids, subj)]
    folds = vu.unit_folds(y, subj, args["n_folds"], args["fold_seed"], grouped=True)
    pr, cl = vu.unit_probe(Z, y, folds, **{**kw, "n_jobs": 1})
    return vu.unit_and_subject_scores(pr, cl, y, subj)[0]["balanced_accuracy"]


def run_target(Z, units, target, model, level, args, rng_seed):
    "Every analysis for one target and level, with each probe in args['probes'] (results keyed by probe)"
    lv = args["levels"][level]
    keep, y, classes = target_rows(units, target, args)
    "The strata may use a coarser label set (strata_merge), so that every class has enough subjects for the folds"
    _, y_strata, classes_strata = target_rows(units, target, args, args["strata_merge"].get(target))
    Zt, ut = Z[keep], units[keep].reset_index(drop=True)
    subj = ut["subject"].to_numpy()
    out = {"n_units": int(len(y)), "n_subjects": int(len(np.unique(subj))), "classes": classes.tolist(),
           "strata_classes": classes_strata.tolist(), "probes": {}}
    rng = np.random.default_rng(rng_seed)
    sub = vu.subsample_by_subject(subj, lv["n_sub"], rng)
    Zs, ys, ss = Zt[sub], y[sub], subj[sub]
    out["n_units_probe"] = int(len(sub))
    shared = vu.unit_folds(ys, ss, args["n_folds"], args["fold_seed"], grouped=False)
    grouped = vu.unit_folds(ys, ss, args["n_folds"], args["fold_seed"], grouped=True)
    subjects, _, lab = vu.subject_labels(subj, y)
    out["speaker_lookup_shared"] = vu.speaker_lookup_score(ys, ss, shared)
    perm_models = lv["unit_permutation_models"]
    for kind in args["probes"]:
        o = {}
        kw = {"kind": kind, "C": args["probe_C"], "n_jobs": args["n_jobs"], "max_iter": args["probe_max_iter"]}
        t0 = time.time()

        "A. Unit-level probes: speaker-shared vs subject-grouped folds"
        p_sh, cl = vu.unit_probe(Zs, ys, shared, **kw)
        u_sh, _, _ = vu.unit_and_subject_scores(p_sh, cl, ys, ss)
        p_gr, cl = vu.unit_probe(Zs, ys, grouped, **kw)
        u_gr, s_gr, _ = vu.unit_and_subject_scores(p_gr, cl, ys, ss)
        o["units_shared"], o["units_grouped"], o["subject_from_units_grouped"] = u_sh, u_gr, s_gr
        o["leakage_bacc"] = u_sh["balanced_accuracy"] - u_gr["balanced_accuracy"]
        print(f"  {level}/{target}/{kind}: unit probes {time.time() - t0:.0f} s")

        "B, C, E. Subject-level probes, permutation tests, bootstrap. One null per permute_within_category value:"
        "shuffled across all subjects, or within ALS and within controls (stage beyond the ALS/control split)"
        o["subject_probe"] = {}
        category = ut.groupby("subject")["category"].first().loc[subjects].to_numpy()
        within = args["permute_within_category"]
        within = [within] if isinstance(within, bool) else list(within)
        for mode in args["subject_feature_modes"]:
            _, F = vu.subject_features(Zt, subj, ut["phoneme"].to_numpy(), mode)
            mean, std, pred = vu.subject_probe(F, lab, args["n_folds"], range(args["subject_repeats"]), **kw)
            perms = {}
            for wc in within:
                t0 = time.time()
                strata = category if wc else None
                null = Parallel(n_jobs=args["n_jobs"])(delayed(_subject_null)(F, lab, strata, i, kw, args)
                                                       for i in range(args["n_perm_subject"]))
                perms[NULL_SCHEMES[wc]] = {"observed": mean["balanced_accuracy"], "null": null,
                                           "p_value": vu.permutation_pvalue(mean["balanced_accuracy"], null), "seconds": time.time() - t0}
                print(f"  {level}/{target}/{kind}: subject probe ({mode}), {args['n_perm_subject']} permutations "
                      f"{NULL_SCHEMES[wc]} {time.time() - t0:.0f} s")
            o["subject_probe"][mode] = {
                "mean": mean, "std": std,
                "ci_bootstrap": vu.subject_bootstrap(lab, pred, args["n_boot"], np.random.default_rng(rng_seed + 1)),
                "permutation": perms}

        "C. Permutation test of the grouped unit probe (selected models, on a smaller subsample)"
        if perm_models == "all" or model["name"] in perm_models:
            t0 = time.time()
            n_perm_sub = lv["n_sub_unit_perm"]
            sp = np.arange(len(ys)) if n_perm_sub is None or n_perm_sub >= len(ys) else \
                np.sort(np.random.default_rng(rng_seed + 2).choice(len(ys), size=n_perm_sub, replace=False))
            Zp, yp0, sp_s = Zs[sp], ys[sp], ss[sp]
            folds = vu.unit_folds(yp0, sp_s, args["n_folds"], args["fold_seed"], grouped=True)
            pr, cl_p = vu.unit_probe(Zp, yp0, folds, **kw)
            observed = vu.unit_and_subject_scores(pr, cl_p, yp0, sp_s)[0]["balanced_accuracy"]
            subj_ids, _, lab_s = vu.subject_labels(sp_s, yp0)
            null = Parallel(n_jobs=args["n_jobs"])(delayed(_unit_null)(Zp, subj_ids, lab_s, sp_s, i, kw, args)
                                                   for i in range(args["n_perm_unit"]))
            o["units_grouped_permutation"] = {"observed": observed, "null": null, "n_units": int(len(sp)),
                                              "p_value": vu.permutation_pvalue(observed, null), "seconds": time.time() - t0}
            print(f"  {level}/{target}/{kind}: {args['n_perm_unit']} unit-level permutations {time.time() - t0:.0f} s")

        "D. Strata: subject probe inside each stratum, on the strata label set"
        o["strata"] = {}
        for name, rule in args["strata"].items():
            m = ut[rule["column"]].isin(rule["values"]).to_numpy()
            if m.sum() == 0 or len(np.unique(y_strata[m])) < 2:
                o["strata"][name] = {"skipped": "no units, or a single class"}
                continue
            _, F = vu.subject_features(Zt[m], subj[m])
            lab_m = vu.subject_labels(subj[m], y_strata[m])[2]
            counts = np.bincount(lab_m)
            if counts[counts > 0].min() < args["n_folds"]:
                o["strata"][name] = {"skipped": "a class has fewer subjects than folds"}
                continue
            mean, std, _ = vu.subject_probe(F, lab_m, args["n_folds"], range(args["subject_repeats"]), **kw)
            o["strata"][name] = {"subject_probe": mean, "subject_probe_std": std, "n_subjects": int(len(lab_m))}
        out["probes"][kind] = o
    return out


def figure_tables(results, models):
    """Long CSVs for the R scripts, one row per model, level, target, probe and protocol.
    p_value is the test against the all-subjects null, p_value_within_category against the within-category one."""
    scores_rows, null_rows, strata_rows = [], [], []
    no_p = {"p_value": np.nan, "p_value_within_category": np.nan}
    label = {m["name"]: m.get("label", m["name"]) for m in models}
    for name, res in results.items():
        for level, per_target in res["levels"].items():
            for target, r in per_target.items():
                lk = r["speaker_lookup_shared"]
                scores_rows.append({"model": name, "label": label[name], "level": level, "target": target, "probe": "none",
                                    "protocol": "speaker_lookup", "value": lk["balanced_accuracy"], "f1_macro": lk["f1_macro"],
                                    "accuracy": lk["accuracy"], "ci_low": np.nan, "ci_high": np.nan, **no_p})
                for kind, o in r["probes"].items():
                    base = {"model": name, "label": label[name], "level": level, "target": target, "probe": kind}
                    for key in ("units_shared", "units_grouped", "subject_from_units_grouped"):
                        p_value = o["units_grouped_permutation"]["p_value"] if key == "units_grouped" and "units_grouped_permutation" in o else np.nan
                        scores_rows.append({**base, "protocol": key, "value": o[key]["balanced_accuracy"],
                                            "f1_macro": o[key]["f1_macro"], "accuracy": o[key]["accuracy"],
                                            "ci_low": np.nan, "ci_high": np.nan, **no_p, "p_value": p_value})
                    for mode, sp in o["subject_probe"].items():
                        perms = sp["permutation"]
                        scores_rows.append({**base, "protocol": f"subject_probe_{mode}", "value": sp["mean"]["balanced_accuracy"],
                                            "f1_macro": sp["mean"]["f1_macro"], "accuracy": sp["mean"]["accuracy"],
                                            "ci_low": sp["ci_bootstrap"][0], "ci_high": sp["ci_bootstrap"][1],
                                            "p_value": perms.get("all_subjects", {}).get("p_value", np.nan),
                                            "p_value_within_category": perms.get("within_category", {}).get("p_value", np.nan)})
                        for scheme, pm in perms.items():
                            null_rows += [{**base, "test": f"subject_probe_{mode}", "null_scheme": scheme, "null": v,
                                           "observed": pm["observed"]} for v in pm["null"]]
                    if "units_grouped_permutation" in o:
                        fp = o["units_grouped_permutation"]
                        null_rows += [{**base, "test": "units_grouped", "null_scheme": "all_subjects", "null": v,
                                       "observed": fp["observed"]} for v in fp["null"]]
                    for s_name, sr in o["strata"].items():
                        if "subject_probe" in sr:
                            strata_rows.append({**base, "stratum": s_name, "value": sr["subject_probe"]["balanced_accuracy"],
                                                "std": sr["subject_probe_std"]["balanced_accuracy"], "n_subjects": sr["n_subjects"]})
    return pd.DataFrame(scores_rows), pd.DataFrame(null_rows), pd.DataFrame(strata_rows)


def unit_set_check(results):
    "Section 3, check 5: does every model hold the same units, in the same order, as the first model?"
    rows = []
    for level in next(iter(results.values()))["unit_sets"]:
        ref = None
        for name, res in results.items():
            sig = res["unit_sets"][level]
            ref = ref or sig
            rows.append({"model": name, "level": level, "source": res["sources"][level], "n_units": sig["n_units"],
                         "same_units_as_first": sig["crc_set"] == ref["crc_set"] and sig["n_units"] == ref["n_units"],
                         "same_order_as_first": sig["crc_order"] == ref["crc_order"] and sig["n_units"] == ref["n_units"]})
    return pd.DataFrame(rows)


def main():
    quiet_warnings()
    cfg_file = JSON_FILE_NAME_MANUAL if debugger_is_active() else parse_args().config_file
    args = load_args(cfg_file)
    with open(args["models_file"], encoding="utf-8") as f:
        models = json.load(f)
    subject_table = pd.read_csv(args["subject_table"])
    os.makedirs(os.path.join(args["output_dir"], "per_model"), exist_ok=True)
    os.makedirs(args["figure_data_dir"], exist_ok=True)
    results = {}
    for model in models:
        path = os.path.join(args["output_dir"], "per_model", f"{model['name']}.json")
        if os.path.exists(path) and not args["recompute"]:
            with open(path) as f:
                results[model["name"]] = json.load(f)
            print(f"{model['name']}: loaded {path}")
            continue
        print(f"{model['name']}")
        per_level = load_model_units(model, subject_table, args)
        res = {"sources": {}, "unit_sets": {}, "levels": {}}
        for level, (Z, units, source) in per_level.items():
            print(f" {level}: {len(units)} units from the {source}, latent dim {Z.shape[1]}")
            res["sources"][level] = source
            res["unit_sets"][level] = unit_signature(units)
            res["levels"][level] = {t: run_target(Z, units, t, model, level, args, args["subsample_seed"]) for t in args["targets"]}
        results[model["name"]] = res
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"{model['name']}: saved {path}")
    sc, nl, st = figure_tables(results, models)
    sc.to_csv(os.path.join(args["figure_data_dir"], "scores_by_protocol.csv"), index=False)
    nl.to_csv(os.path.join(args["figure_data_dir"], "permutation_nulls.csv"), index=False)
    st.to_csv(os.path.join(args["figure_data_dir"], "strata.csv"), index=False)
    check = unit_set_check(results)
    check.to_csv(os.path.join(args["output_dir"], "unit_set_check.csv"), index=False)
    print(check.to_string(index=False))


if __name__ == "__main__":
    main()
