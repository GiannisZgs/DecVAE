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

"""Subject-independent evaluation of subject-level labels (King's stage, disease duration) on frozen
latents, for the VOC-ALS confound analysis (reviewer comment R5 #9).

The units are frames (frame branch) or utterances (sequence branch). Every clinical label belongs to a
subject, so:
- folds are grouped by subject (no subject in both train and test), except in the speaker-shared
  contrast that reproduces the paper's protocol;
- permutations shuffle labels across subjects, never across units;
- bootstrap intervals resample subjects.
Two probes, refitted on fixed latents (no model is retrained):
- "rf": the paper's Random Forest (200 trees, max_depth None, max_features "sqrt", as in
  classification_utils.prediction_eval), so the speaker-shared protocol reproduces the Fig. 4h setting;
- "lr": standardisation + logistic regression with a fixed C, as in the subspace analysis.
Both have fixed hyperparameters, so a permutation run refits exactly the same procedure.
Labels must be encoded as 0..K-1 before they reach these functions.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score


def make_probe(kind="rf", C=1.0, seed=0, n_jobs=1, max_iter=1000):
    "kind: 'rf' (paper's Random Forest settings) or 'lr' (linear probe of the subspace analysis)"
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, max_features="sqrt",
                                      random_state=seed, n_jobs=n_jobs)
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))
    raise ValueError(f"unknown probe {kind}")


def subject_labels(unit_subject, unit_label):
    "One label per subject; raises if a subject's units disagree (the label must be nested in the subject)"
    subjects, inv = np.unique(unit_subject, return_inverse=True)
    lab = np.full(len(subjects), -1)
    for s in range(len(subjects)):
        vals = np.unique(unit_label[inv == s])
        if len(vals) != 1:
            raise ValueError(f"subject {subjects[s]} has {len(vals)} different labels")
        lab[s] = vals[0]
    return subjects, inv, lab


def pool_utterances(Z, subject, phoneme):
    """Mean of the frames of each utterance, for models without a sequence-level dump. In VOC-ALS every
    subject has one recording per phoneme, so (subject, phoneme) identifies the utterance."""
    keys, inv = np.unique(np.stack([subject, phoneme], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    Zu = np.zeros((len(keys), Z.shape[1]))
    np.add.at(Zu, inv, Z)
    Zu /= np.bincount(inv)[:, None]
    return Zu, keys[:, 0], keys[:, 1]


def subsample_by_subject(unit_subject, n, rng):
    "Sorted unit indices, about the same number per subject (n in total), so no subject dominates the probe. None keeps all"
    if n is None:
        return np.arange(len(unit_subject))
    subjects, inv = np.unique(unit_subject, return_inverse=True)
    per = max(1, n // len(subjects))
    idx = [rng.choice(np.flatnonzero(inv == s), size=min(per, np.sum(inv == s)), replace=False) for s in range(len(subjects))]
    return np.sort(np.concatenate(idx))


def unit_folds(y, subject, n_folds, seed, grouped):
    "Grouped: StratifiedGroupKFold by subject (stratified on the subject label). Shared: StratifiedKFold on units"
    X = np.zeros((len(y), 1))
    if grouped:
        return list(StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(X, y, groups=subject))
    return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y))


def unit_probe(Z, y, folds, kind="rf", C=1.0, n_jobs=1, max_iter=1000):
    """Out-of-fold class probabilities for every unit.

    Returns proba (N, K) and the class order.
    """
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))
    for tr, te in folds:
        clf = make_probe(kind, C, n_jobs=n_jobs, max_iter=max_iter).fit(Z[tr], y[tr])
        p = clf.predict_proba(Z[te])
        cols = np.searchsorted(classes, clf.classes_)
        proba[np.ix_(te, cols)] = p
    return proba, classes


def scores(y_true, y_pred):
    return {"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "accuracy": float(np.mean(y_true == y_pred))}


def unit_and_subject_scores(proba, classes, y, subject):
    "Unit-level scores, and subject-level scores from the mean log-probability of each subject's units"
    unit = scores(y, classes[proba.argmax(1)])
    subjects, inv, lab = subject_labels(subject, y)
    logp = np.log(np.clip(proba, 1e-12, None))
    mean_logp = np.stack([logp[inv == s].mean(0) for s in range(len(subjects))])
    subj = scores(lab, classes[mean_logp.argmax(1)])
    return unit, subj, (subjects, lab, classes[mean_logp.argmax(1)])


def speaker_lookup_score(y, subject, folds):
    """Upper bound of speaker leakage under a speaker-shared split: predict each test unit's label as the
    majority label of the same speaker's training units (no latent used at all)."""
    pred = np.full(len(y), -1)
    for tr, te in folds:
        lut = {}
        for s in np.unique(subject[tr]):
            vals, cnt = np.unique(y[tr][subject[tr] == s], return_counts=True)
            lut[s] = vals[cnt.argmax()]
        fallback = np.bincount(y[tr]).argmax()
        pred[te] = [lut.get(s, fallback) for s in subject[te]]
    return scores(y, pred)


def subject_features(Z, subject, phoneme=None, mode="mean"):
    """One row per subject: the mean latent ('mean'), or the per-phoneme means concatenated ('phoneme_concat',
    missing phonemes filled with the subject's overall mean)."""
    subjects, inv = np.unique(subject, return_inverse=True)
    mean = np.stack([Z[inv == s].mean(0) for s in range(len(subjects))])
    if mode == "mean":
        return subjects, mean
    phonemes = np.unique(phoneme)
    blocks = []
    for p in phonemes:
        b = mean.copy()
        for s in range(len(subjects)):
            m = (inv == s) & (phoneme == p)
            if m.any():
                b[s] = Z[m].mean(0)
        blocks.append(b)
    return subjects, np.concatenate(blocks, axis=1)


def subject_probe(F, lab, n_folds, seeds, kind="rf", C=1.0, n_jobs=1, max_iter=1000):
    """Repeated stratified K-fold over subjects. Returns mean scores over repeats and the out-of-fold
    predictions of the first repeat (for bootstrap and confusion matrices)."""
    reps, first = [], None
    for seed in seeds:
        pred = np.full(len(lab), -1)
        for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(F, lab):
            pred[te] = make_probe(kind, C, seed=seed, n_jobs=n_jobs, max_iter=max_iter).fit(F[tr], lab[tr]).predict(F[te])
        reps.append(scores(lab, pred))
        if first is None:
            first = pred
    mean = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
    std = {k: float(np.std([r[k] for r in reps], ddof=1)) if len(reps) > 1 else np.nan for k in reps[0]}
    return mean, std, first


def permute_subject_labels(lab, rng, strata=None):
    "Shuffle labels across subjects; within each stratum when strata is given (e.g. ALS vs control)"
    out = lab.copy()
    if strata is None:
        return rng.permutation(lab)
    for g in np.unique(strata):
        m = strata == g
        out[m] = rng.permutation(lab[m])
    return out


def permutation_pvalue(observed, null):
    "One-sided: how often a shuffled-label run scores at least as well. (1 + count) / (1 + n)"
    null = np.asarray(null, float)
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def subject_bootstrap(y_subj, pred_subj, n_boot, rng, metric="balanced_accuracy"):
    "95 % interval of a subject-level score, resampling subjects with replacement (no refit)"
    vals = []
    n = len(y_subj)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y_subj[i])) < 2:
            continue
        vals.append(scores(y_subj[i], pred_subj[i])[metric])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
