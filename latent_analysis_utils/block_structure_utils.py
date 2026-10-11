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

"""Block structure of the decomposition embeddings of a trained DecVAE (Proposition 2, SI Section 7).

P(n) is the softmax over hidden units of the original's embedding of sample n, R_i(n) that of component i.
Proposition 2 is a per-sample statement (disjoint supports of the R_i(n), P(n) their even mixture); this module
also tests whether the supports form a dataset-level partition of the hidden units, i.e. whether component i
owns the same block of units in every sample, as the SVD/PCA directions do.

Samples are utterances in the S branch and valid frames (mask_time_indices) in the Z branch. The embeddings are
the correctly indexed decomposition_features_* / unmasked_frames_features_* of the forward pass, not the used_*
rows of compute_decomposition_loss. Everything is computed in float64, JS in nats with the eps convention of
divergence_utils.js_rows.
"""

import math
import numpy as np
import torch

from latent_analysis_utils.divergence_utils import EPS

N_NULL = 1000  # random partitions for the null captured mass
NULL_SEED = 0


def js_prob(p, q, eps=EPS):
    """JS divergence per row between probability vectors p and q (rows, dim), in the convention of
    divergence_utils.js_rows (which takes logits). Returns (rows,) in float64."""
    p = p.double()
    q = q.double()
    log_m = (0.5 * (p + q)).log()
    kl_m = lambda x: ((x + eps) * ((x + eps).log() - log_m)).sum(dim=-1)
    return 0.5 * (kl_m(p) + kl_m(q))


def cos_rows(a, b):
    "Cosine similarity per row, float64"
    return torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1, eps=1e-300)


def j_C(C):
    "JS(R_i, P) when the R_i have disjoint supports and P is their even mixture (Proposition 2b), nats"
    return 0.5 * (math.log(2 * C / (C + 1)) + math.log(2 / (C + 1)) / C + (C - 1) / C * math.log(2))


class BlockStructure:
    """Accumulates the block-structure measures of one branch ('z' frame, 's' sequence) over a split.

    Call add(outputs) after every forward pass (or add_samples with the embeddings directly), then summary()
    and mean_distributions() once.
    """

    def __init__(self, NoC, branch="z", n_null=N_NULL, null_seed=NULL_SEED):
        self.NoC = NoC
        self.branch = branch
        self.n_null = n_null
        self.null_seed = null_seed
        self.pairs = [(i, j) for i in range(NoC) for j in range(i + 1, NoC)]
        self.n = 0
        self.H = None
        # per-sample sums (float64)
        self.js_pos = np.zeros(NoC)
        self.js_pos_sq = np.zeros(NoC)
        self.js_neg = np.zeros(len(self.pairs))
        self.js_neg_sq = np.zeros(len(self.pairs))
        self.cos_ps = np.zeros(len(self.pairs))
        self.js_mix = 0.0
        self.oracle = np.zeros(NoC)
        self.R_sum = None  # (C, H)
        self.P_sum = None  # (H,)
        # R_i(n) per sample (float32, CPU), for the percentiles of the captured mass
        self.R_store = []
        # per batch, keyed as the model's div_dict, for the cross-check against ComponentDivergences
        self.batch_means = []

    def add(self, outputs):
        "outputs: DecVAEForPreTrainingOutput of one batch"
        b = self.branch
        originals = getattr(outputs, "unmasked_frames_features_" + b)
        components = getattr(outputs, "decomposition_features_" + b)
        if b == "z":
            mask = outputs.mask_time_indices.bool()  # (B, F)
            assert originals.dim() == 3 and components.dim() == 4, \
                f"z: expected (B, F, H) and (C, B, F, H), got {tuple(originals.shape)} and {tuple(components.shape)}"
            assert mask.shape == originals.shape[:2], f"mask {tuple(mask.shape)} vs originals {tuple(originals.shape)}"
            originals = originals[mask]  # (N, H)
            components = components[:, mask]  # (C, N, H)
        else:
            assert originals.dim() == 2 and components.dim() == 3, \
                f"s: expected (B, H) and (C, B, H), got {tuple(originals.shape)} and {tuple(components.shape)}"
        self.add_samples(originals, components)

    def add_samples(self, originals, components):
        "originals (N, H) and components (C, N, H): embeddings (logits), one row per sample"
        assert components.shape[0] == self.NoC, "NoC of the config and of the outputs differ"
        assert components.shape[1:] == originals.shape, f"{tuple(components.shape)} vs {tuple(originals.shape)}"
        P = torch.softmax(originals.double(), dim=-1)
        R = torch.softmax(components.double(), dim=-1)
        C, N, H = R.shape
        if self.H is None:
            self.H = H
            self.R_sum = np.zeros((C, H))
            self.P_sum = np.zeros(H)
        rec = {}
        for i in range(C):
            js = js_prob(R[i], P)
            self.js_pos[i] += js.sum().item()
            self.js_pos_sq[i] += (js ** 2).sum().item()
            rec["div_0_{}".format(i + 1)] = js.mean().item()
        for k, (i, j) in enumerate(self.pairs):
            js = js_prob(R[i], R[j])
            self.js_neg[k] += js.sum().item()
            self.js_neg_sq[k] += (js ** 2).sum().item()
            self.cos_ps[k] += cos_rows(R[i], R[j]).sum().item()
            rec["div_{}_{}".format(j + 1, i + 1)] = js.mean().item()
        self.js_mix += js_prob(P, R.mean(dim=0)).sum().item()
        "Per-sample oracle: every unit goes to the component with the most mass on it in that sample"
        owner_n = R.argmax(dim=0)  # (N, H)
        onehot = torch.nn.functional.one_hot(owner_n, C).permute(2, 0, 1).double()  # (C, N, H)
        self.oracle += (R * onehot).sum(dim=(1, 2)).cpu().numpy()
        self.R_sum += R.sum(dim=1).cpu().numpy()
        self.P_sum += P.sum(dim=0).cpu().numpy()
        self.R_store.append(R.float().cpu())
        self.n += N
        self.batch_means.append(rec)

    def mean_distributions(self):
        "R_bar (C, H) and P_bar (H,)"
        return self.R_sum / self.n, self.P_sum / self.n

    def summary(self):
        """List of (measure, comp_i, comp_j, value); comp_i, comp_j are 1-based, None when not applicable.

        js_pos / js_pos_sd (i):      mean, SD over samples of JS(R_i(n), P(n))
        js_neg / js_neg_sd (i, j):   mean, SD over samples of JS(R_i(n), R_j(n))
        cos_ps (i, j):               mean over samples of cos(R_i(n), R_j(n))
        js_pos_mean, js_neg_mean, cos_ps_mean: the same averaged over components / pairs
        js_mix:                      mean over samples of JS(P(n), mean_i R_i(n))
        j_C, ln2:                    reference values of Proposition 2
        cos_ds / js_ds (i, j):       cos and JS between the dataset means R_bar_i, R_bar_j
        unit_count (i):              |U_i|, units owned by i (argmax_i R_bar_i(u))
        purity:                      mass-weighted purity of the ownership
        captured_mass / _p05 (i):    mean and 5th percentile over samples of sum_{u in U_i} R_i(n)[u]
        C_mass, C_oracle:            captured mass with the dataset partition / the per-sample partition
        C_null / C_null_p95:         mean / 95th percentile over random partitions with the sizes |U_i|
        CI:                          (C_mass - C_null) / (C_oracle - C_null), clipped to [0, 1]
        """
        C, n = self.NoC, self.n
        out = [("n_samples", None, None, float(n)), ("H", None, None, float(self.H))]
        js_pos = self.js_pos / n
        js_pos_sd = np.sqrt(np.maximum(self.js_pos_sq / n - js_pos ** 2, 0.0))
        js_neg = self.js_neg / n
        js_neg_sd = np.sqrt(np.maximum(self.js_neg_sq / n - js_neg ** 2, 0.0))
        cos_ps = self.cos_ps / n
        for i in range(C):
            out += [("js_pos", i + 1, None, js_pos[i]), ("js_pos_sd", i + 1, None, js_pos_sd[i])]
        for k, (i, j) in enumerate(self.pairs):
            out += [("js_neg", i + 1, j + 1, js_neg[k]), ("js_neg_sd", i + 1, j + 1, js_neg_sd[k]),
                    ("cos_ps", i + 1, j + 1, cos_ps[k])]
        out += [("js_pos_mean", None, None, js_pos.mean()),
                ("js_neg_mean", None, None, js_neg.mean() if self.pairs else np.nan),
                ("cos_ps_mean", None, None, cos_ps.mean() if self.pairs else np.nan),
                ("js_mix", None, None, self.js_mix / n),
                ("j_C", None, None, j_C(C)), ("ln2", None, None, math.log(2))]

        "Dataset level"
        R_bar, _ = self.mean_distributions()
        R_bar_t = torch.from_numpy(R_bar)
        for i, j in self.pairs:
            out += [("cos_ds", i + 1, j + 1, cos_rows(R_bar_t[i], R_bar_t[j]).item()),
                    ("js_ds", i + 1, j + 1, js_prob(R_bar_t[i:i + 1], R_bar_t[j:j + 1]).item())]
        owner = R_bar.argmax(axis=0)  # (H,)
        sizes = np.bincount(owner, minlength=C)
        purity = R_bar.max(axis=0).sum() / R_bar.sum()  # sum_u w(u) max_i a_i(u)
        out += [("unit_count", i + 1, None, float(sizes[i])) for i in range(C)]
        out.append(("purity", None, None, purity))

        "Captured mass of the dataset partition, per sample"
        R_all = torch.cat(self.R_store, dim=1)  # (C, n, H)
        own = torch.from_numpy(owner)
        c_mean = []
        for i in range(C):
            c_i = R_all[i][:, own == i].double().sum(dim=1).numpy()
            c_mean.append(R_bar[i, owner == i].sum())  # exact float64 mean over samples
            out += [("captured_mass", i + 1, None, c_mean[-1]),
                    ("captured_mass_p05", i + 1, None, float(np.percentile(c_i, 5)))]
        C_mass = float(np.mean(c_mean))
        C_oracle = float((self.oracle / n).mean())

        "Null: random partitions with the same sizes; the captured mass is linear in R_i(n), so R_bar suffices"
        rng = np.random.default_rng(self.null_seed)
        labels = np.repeat(np.arange(C), sizes)
        L = np.stack([rng.permutation(labels) for _ in range(self.n_null)])  # (n_null, H)
        null = R_bar[L, np.arange(self.H)].sum(axis=1) / C
        C_null = float(null.mean())
        den = C_oracle - C_null
        CI = float(np.clip((C_mass - C_null) / den, 0.0, 1.0)) if den > 1e-12 else np.nan
        out += [("C_mass", None, None, C_mass), ("C_oracle", None, None, C_oracle),
                ("C_null", None, None, C_null), ("C_null_p95", None, None, float(np.percentile(null, 95))),
                ("CI", None, None, CI)]
        return [(m, i, j, float(v)) for m, i, j, v in out]
