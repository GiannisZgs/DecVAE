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

"""Subspace-wise analysis of DecVAE latents: prediction, ablation and swap matrices, entropy
selectivity, random-partition null, fold stability, cross-seed CKA, SimVowels generator alignment.

All latents are (N, D) here (frames in rows). A DecVAE "all" latent is [X, OC1, ..., OCC], each
block z_latent_dim wide, in that order (see latents_post_analysis.py, mu_all_z).
"""

import warnings
from itertools import combinations
import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


def block_indices(latent_dim, n_blocks):
    "Column indices of each subspace of an 'all' latent: [X, OC1, ..., OCC]"
    assert latent_dim % n_blocks == 0, f"latent_dim {latent_dim} not divisible by {n_blocks} blocks"
    w = latent_dim // n_blocks
    return [np.arange(g * w, (g + 1) * w) for g in range(n_blocks)]


def random_partition(latent_dim, n_blocks, rng):
    "Equal-size random grouping of the latent dimensions (the null for block structure)"
    perm = rng.permutation(latent_dim)
    w = latent_dim // n_blocks
    return [np.sort(perm[g * w:(g + 1) * w]) for g in range(n_blocks)]


def cn_bacc(y_true, y_pred, n_classes):
    "Chance-normalised balanced accuracy: 0 = chance, 1 = perfect"
    with warnings.catch_warnings():
        # speaker-disjoint folds can lack classes in the test part
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        bacc = balanced_accuracy_score(y_true, y_pred)
    return (bacc - 1.0 / n_classes) / (1.0 - 1.0 / n_classes)


class Probe:
    "Standardised multinomial logistic regression on a column subset"

    def __init__(self, cols, C=1.0, max_iter=1000):
        self.cols, self.C, self.max_iter = cols, C, max_iter

    def fit(self, Z, y):
        self.scaler = StandardScaler().fit(Z[:, self.cols])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter).fit(
                self.scaler.transform(Z[:, self.cols]), y)
            self.converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
        return self

    def predict(self, Z):
        return self.clf.predict(self.scaler.transform(Z[:, self.cols]))

    def importance(self, latent_dim):
        "Per-dimension importance: L2 norm over classes of the standardised coefficients"
        imp = np.zeros(latent_dim)
        imp[self.cols] = np.linalg.norm(self.clf.coef_, axis=0)
        return imp


def fit_full_probes(Z, Y, tr, n_classes, **kw):
    "One full-latent probe per factor (reused by ablation baseline, swap and importances)"
    cols = np.arange(Z.shape[1])
    return [Probe(cols, **kw).fit(Z[tr], Y[f, tr]) for f in range(Y.shape[0])]


def prediction_matrix(Z, Y, tr, te, groups, n_classes, log=None, **kw):
    """P[g, f]: score of a probe trained on subspace g alone.

    log: optional list, receives the convergence flag of every probe.
    """
    P = np.zeros((len(groups), Y.shape[0]))
    for g, cols in enumerate(groups):
        for f in range(Y.shape[0]):
            pr = Probe(cols, **kw).fit(Z[tr], Y[f, tr])
            P[g, f] = cn_bacc(Y[f, te], pr.predict(Z[te]), n_classes[f])
            if log is not None:
                log.append(pr.converged)
    return P


def ablation_matrix(Z, Y, tr, te, groups, n_classes, full_scores, log=None, **kw):
    """A[g, f]: drop in score when subspace g is removed and the probe retrained.

    log: optional list, receives the convergence flag of every probe.
    """
    D = Z.shape[1]
    A = np.zeros((len(groups), Y.shape[0]))
    for g, cols in enumerate(groups):
        rest = np.setdiff1d(np.arange(D), cols)
        for f in range(Y.shape[0]):
            pr = Probe(rest, **kw).fit(Z[tr], Y[f, tr])
            A[g, f] = full_scores[f] - cn_bacc(Y[f, te], pr.predict(Z[te]), n_classes[f])
            if log is not None:
                log.append(pr.converged)
    return A


def matched_pairs(Y, te, f, rng, max_pairs=5000, match=None):
    """Pairs (a, b) of test frames that differ in factor f and agree on the factors in match.

    match defaults to every other factor. Returns two index arrays into the full frame axis, empty if
    no such pairs exist.
    """
    others = [k for k in range(Y.shape[0]) if k != f] if match is None else list(match)
    key = np.zeros(len(te), dtype=np.int64)
    for k in others:  # combined label of the other factors
        key = key * (int(Y[k].max()) + 1) + Y[k, te]
    a_idx, b_idx = [], []
    for kv in np.unique(key):
        members = te[key == kv]
        if len(np.unique(Y[f, members])) < 2:
            continue
        partners = rng.permutation(members)
        keep = Y[f, members] != Y[f, partners]
        a_idx.append(members[keep]); b_idx.append(partners[keep])
    if not a_idx:
        return np.array([], int), np.array([], int)
    a, b = np.concatenate(a_idx), np.concatenate(b_idx)
    if len(a) > max_pairs:
        sel = rng.choice(len(a), max_pairs, replace=False)
        a, b = a[sel], b[sel]
    return a, b


def same_label_partner(Y, te, a, rng):
    "For each frame in a, a different test frame with identical labels on every factor (-1 if none)"
    F = Y.shape[0]
    key = np.zeros(Y.shape[1], dtype=np.int64)
    for k in range(F):
        key = key * (int(Y[k].max()) + 1) + Y[k]
    groups = {}
    for i in te:
        groups.setdefault(key[i], []).append(i)
    c = np.full(len(a), -1)
    for n, i in enumerate(a):
        pool = groups[key[i]]
        if len(pool) > 1:
            j = i
            while j == i:
                j = pool[rng.integers(len(pool))]
            c[n] = j
    return c


def swap_matrices(Z, Y, te, groups, full_probes, rng, max_pairs=5000, match=None):
    """Swap test with a matched control.

    For factor f, pairs (a, b) differ only in f. Subspace g of a is replaced by b's (swap), or by the
    same subspace of a frame c with a's exact labels (control). Over the pairs where the full probes
    are correct on a for every factor:
        T  = fraction where the f-prediction becomes b's label           (control: Tc, same target label)
        O  = fraction where any other factor's prediction changes        (control: Oc)
    Report T - Tc as the on-target effect and O - Oc as the off-target effect (ideally 0).
    match: optional {f: [factors the pair must agree on]}, for confounded factors (VOC-ALS stage).
    """
    F, G = Y.shape[0], len(groups)
    T, O, Tc, Oc = (np.full((G, F), np.nan) for _ in range(4))
    n_pairs = np.zeros(F, dtype=int)
    for f in range(F):
        a, b = matched_pairs(Y, te, f, rng, max_pairs, None if match is None else match.get(f))
        if len(a) == 0:
            continue
        c = same_label_partner(Y, te, a, rng)
        ok = (c >= 0) & np.all([full_probes[k].predict(Z[a]) == Y[k, a] for k in range(F)], axis=0)
        a, b, c = a[ok], b[ok], c[ok]
        n_pairs[f] = len(a)
        if len(a) == 0:
            continue
        others = [k for k in range(F) if k != f and (match is None or f not in match or k in match[f])]
        for g, cols in enumerate(groups):
            for src, Tm, Om in ((b, T, O), (c, Tc, Oc)):
                Zs = Z[a].copy()
                Zs[:, cols] = Z[src][:, cols]
                Tm[g, f] = np.mean(full_probes[f].predict(Zs) == Y[f, b])
                if others:
                    Om[g, f] = np.mean(np.any([full_probes[k].predict(Zs) != Y[k, a] for k in others], axis=0))
    return {"T": T, "O": O, "T_ctrl": Tc, "O_ctrl": Oc, "n_pairs": n_pairs}


def _entropy_norm(p, axis, base):
    p = p / p.sum(axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=axis)
    return h / np.log(base)


def subspace_selectivity(M, rows=None):
    """Mass-weighted 1 - normalised entropy over factors, per subspace row (DCI-D style).

    1 = every subspace relates to a single factor. Negative entries are clipped to 0. rows selects
    the subspaces to include (e.g. OCs only).
    """
    M = np.clip(np.nan_to_num(np.asarray(M, float)), 0, None)
    if rows is not None:
        M = M[rows]
    mass = M.sum(1)
    if M.shape[1] < 2 or mass.sum() == 0:
        return np.nan
    live = mass > 0
    s = 1.0 - _entropy_norm(M[live], axis=1, base=M.shape[1])
    return float(np.sum(s * mass[live]) / mass[live].sum())


def factor_concentration(M):
    "Per factor, 1 - normalised entropy over subspaces: 1 = the factor lives in one subspace"
    M = np.clip(np.nan_to_num(np.asarray(M, float)), 0, None)
    out = np.full(M.shape[1], np.nan)
    for f in range(M.shape[1]):
        if M[:, f].sum() > 0:
            out[f] = 1.0 - _entropy_norm(M[:, f][None], axis=1, base=M.shape[0])[0]
    return out


def null_summary(own, null):
    "DecVAE value against the random-partition null"
    null = np.asarray(null, float)
    sd = null.std(ddof=1) if len(null) > 1 else np.nan
    return {"own": own, "null_mean": float(null.mean()), "null_std": float(sd),
            "gap": float(own - null.mean()), "z": float((own - null.mean()) / sd) if sd and sd > 0 else np.nan,
            "percentile": float(np.mean(null < own) * 100)}


def fold_importance_stability(imps, top_frac=0.1):
    """imps: (n_folds, F, D) per-dimension importances.

    Returns per factor: mean pairwise Spearman correlation and top-k Jaccard across folds, and the
    fraction of dimensions whose dominant factor is the same in every fold.
    """
    K, F, D = imps.shape
    k = max(1, int(round(top_frac * D)))
    out = {"spearman": np.zeros(F), "topk_jaccard": np.zeros(F)}
    for f in range(F):
        rs, js = [], []
        for i, j in combinations(range(K), 2):
            rs.append(spearmanr(imps[i, f], imps[j, f])[0])
            ti, tj = set(np.argsort(imps[i, f])[-k:]), set(np.argsort(imps[j, f])[-k:])
            js.append(len(ti & tj) / len(ti | tj))
        out["spearman"][f], out["topk_jaccard"][f] = np.mean(rs), np.mean(js)
    norm = imps / imps.sum(axis=2, keepdims=True)  # importance share of each dim per factor
    dominant = norm.argmax(axis=1)  # (K, D)
    out["assignment_consistency"] = float(np.mean(np.all(dominant == dominant[0], axis=0))) if F > 1 else np.nan
    return out


def block_importance(imp, groups):
    "(F, D) importances -> (G, F) share of each factor's importance held by each subspace"
    B = np.stack([imp[:, cols].sum(1) for cols in groups])
    return B / B.sum(0, keepdims=True)


def linear_cka(X, Y):
    "Linear CKA between two (N, d) representations of the same frames"
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")))


# SimVowels generator, Supplementary Alg. 1: formant k of vowel v for vocal-tract factor t is F[v, k] / t
SIMVOWELS_FORMANTS = {"a": [710, 1100, 2540], "e": [550, 1770, 2490], "I": [400, 1920, 2560],
                      "aw": [590, 880, 2540], "u": [310, 870, 2250]}


def generator_expectation(vowel_names, speaker_vt):
    """eta^2 of vowel and speaker in each (log) formant, over the actual test frames.

    vowel_names: (n,) vowel string per frame; speaker_vt: (n,) vocal-tract factor per frame.
    Returns (3, 2) array: rows F1..F3, columns [vowel, speaker].
    """
    vowel_names, speaker_vt = np.asarray(vowel_names), np.asarray(speaker_vt, float)
    out = np.zeros((3, 2))
    for k in range(3):
        x = np.log(np.array([SIMVOWELS_FORMANTS[v][k] for v in vowel_names]) / speaker_vt)
        tot = x.var()
        for c, lab in enumerate([vowel_names, speaker_vt]):
            means = {u: x[lab == u].mean() for u in np.unique(lab)}
            out[k, c] = np.var(np.array([means[u] for u in lab])) / tot
    return out


def alignment(M_oc, G):
    """Agreement between observed OC rows (C, 2) and the generator expectation (C, 2).

    Uses the vowel share of each row, s = vowel / (vowel + speaker).
    """
    M = np.clip(np.nan_to_num(M_oc), 0, None)
    obs = M[:, 0] / np.where(M.sum(1) > 0, M.sum(1), np.nan)
    exp = G[:, 0] / G.sum(1)
    return {"vowel_share_obs": obs.tolist(), "vowel_share_exp": exp.tolist(),
            "mae": float(np.nanmean(np.abs(obs - exp))),
            "dominant_agree": int(np.sum((obs > 0.5) == (exp > 0.5))),
            "pearson": float(np.corrcoef(obs, exp)[0, 1]) if np.all(np.isfinite(obs)) else np.nan}
