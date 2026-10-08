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

"""Component divergences of a trained DecVAE, measured on a fixed checkpoint (reviewer comment R1 #6).

The forward pass of DecVAEForPreTraining already computes the decomposition divergences
(compute_decomposition_loss) and returns them in its outputs. This module collects them over a whole split:

- the values exactly as the model reports them (div_pos, div_neg, the per-pair divergence_dict, the
  cross-entropies and the prior loss), averaged per batch as in the validation loop of
  base_models_ssl_pretraining.py;
- the per-frame divergences on the rows the loss compared (used_unmasked_features,
  used_decomposition_features), pooled over frames, from which the model's batch values are also reproduced
  as a check.
"""

import math
import numpy as np
import torch

EPS = 1e-8  # as in DecVAEForPreTraining.compute_decomposition_loss


def js_rows(a, b, eps=EPS):
    """JS divergence per row between softmax(a) and softmax(b), in the convention of
    compute_decomposition_loss. a, b: (rows, dim). Returns (rows,) in float64."""
    p = torch.softmax(a.double(), dim=-1)
    q = torch.softmax(b.double(), dim=-1)
    log_m = (0.5 * (p + q)).log()
    kl_m = lambda x: ((x + eps) * ((x + eps).log() - log_m)).sum(dim=-1)
    return 0.5 * (kl_m(p) + kl_m(q))


def pair_keys(NoC):
    "Same keys as the model's div_dict: div_0_k, then div_k_l for k > l"
    pos = ["div_0_{}".format(i + 1) for i in range(NoC)]
    neg = ["div_{}_{}".format(k + 1, l + 1) for l in range(NoC) for k in range(l + 1, NoC)]
    return pos, neg


def _scalar(x):
    if x is None:
        return np.nan
    if torch.is_tensor(x):
        return float(x.detach().double().sum().cpu())
    return float(x)


class ComponentDivergences:
    """Accumulates the decomposition divergences of one branch ('z' frame, 's' sequence) over a split.

    Call add(outputs) after every forward pass, then summary() once.
    """

    def __init__(self, NoC, branch="z", beta=None, prior_reg_weighting=None):
        self.NoC = NoC
        self.branch = branch
        self.beta = beta
        self.prior_reg_weighting = prior_reg_weighting
        self.pos_keys, self.neg_keys = pair_keys(NoC)
        self.keys = self.pos_keys + self.neg_keys
        # per utterance, on the frames the loss used: number of frames, and per key the sum of the per-frame JS
        self.n_frames = []
        self.utt_sums = {k: [] for k in self.keys}
        # per batch: the model's own outputs
        self.batches = []
        self.max_abs_diff = 0.0

    def _per_frame(self, originals, components):
        "Per-row JS for every key, as the loss pairs the rows. originals (F, d), components (NoC, F, d)"
        out = {}
        for k in range(self.NoC):
            out["div_0_{}".format(k + 1)] = js_rows(originals, components[k])
        for l in range(self.NoC):
            for k in range(l + 1, self.NoC):
                out["div_{}_{}".format(k + 1, l + 1)] = js_rows(components[l], components[k])
        return out

    def add(self, outputs):
        "outputs: DecVAEForPreTrainingOutput of one batch"
        b = self.branch
        originals = getattr(outputs, "used_unmasked_features_" + b)  # (F, d)
        components = getattr(outputs, "used_decomposition_features_" + b)  # (NoC, F, d)
        assert components.shape[0] == self.NoC, "NoC of the config and of the outputs differ"
        used_indices = getattr(outputs, "used_indices_" + b, None)
        if used_indices:  # frame branch: one entry per utterance with the frames it used
            lengths = [len(ix) for ix in used_indices]
        else:  # sequence branch: one aggregated vector per utterance
            lengths = [1] * originals.shape[0]
        assert sum(lengths) == originals.shape[0] == components.shape[1]

        sums = {key: torch.stack([s.sum() for s in torch.split(v, lengths)]).cpu().numpy()
                for key, v in self._per_frame(originals, components).items()}
        for key in self.keys:
            self.utt_sums[key].extend(sums[key].tolist())
        self.n_frames.extend(lengths)

        "The model's own values for this batch"
        div_dict = getattr(outputs, "divergence_dict_" + b) or {}
        rec = {
            "n_utt": len(lengths),
            "n_frames": sum(lengths),
            "model_div_pos": _scalar(getattr(outputs, "div_pos_" + b)),
            "model_div_neg": _scalar(getattr(outputs, "div_neg_" + b)),
            "model_ce_pos": _scalar(getattr(outputs, "ce_pos_" + b)),
            "model_ce_neg": _scalar(getattr(outputs, "ce_neg_" + b)),
            "model_decomposition_loss": _scalar(getattr(outputs, "decomposition_loss_" + b)),
            "model_prior_loss": _scalar(getattr(outputs, "prior_loss_" + b)),
        }
        for key in self.keys:
            rec["model_" + key] = _scalar(div_dict.get(key))
        self.batches.append(rec)

        "Check: the clamped per-utterance sums reproduce the model's batch values"
        clamp = {key: np.clip(sums[key], 0.0, 1.0) for key in self.keys}
        n_utt = len(lengths)
        mine = {
            "model_div_pos": sum(clamp[k].sum() for k in self.pos_keys) / (n_utt * len(self.pos_keys)),
            "model_div_neg": sum(clamp[k].sum() for k in self.neg_keys) / (n_utt * len(self.neg_keys)),
        }
        for key in self.keys:
            mine["model_" + key] = clamp[key].sum() / n_utt
        for key, v in mine.items():
            if not np.isnan(rec[key]):
                self.max_abs_diff = max(self.max_abs_diff, abs(v - rec[key]))

    def summary(self):
        """One dict of measures for the split.

        d_recon / d_ortho:            per-frame JS on the rows the loss compared, pooled (nats; maximum ln 2)
        d_recon_norm / d_ortho_norm:  the same divided by ln 2, in [0, 1]
        d_frame_<key>:                per-frame JS on the rows the loss compared, for each pair (div_0_k, div_k_l)
        d_recon_utt / d_ortho_utt:    the model's convention (per-utterance sum over the used frames, clamped
                                      to [0, 1]), averaged over utterances
        clamp_frac_recon / _ortho:    fraction of (utterance, pair) values whose sum reached the clamp at 1
        model_*:                      the model's outputs, averaged over batches as in the validation loop
        rate_per_frame:               KL to the prior per used frame and subspace, unweighted (NaN when beta = 0)
        check_max_abs_diff:           largest difference between the recomputed and the model's batch values
        """
        n = np.asarray(self.n_frames, float)
        out = {"n_utt": int(len(n)), "n_frames": int(n.sum())}
        utt = {k: np.asarray(v) for k, v in self.utt_sums.items()}
        for key in self.keys:
            out["d_frame_" + key] = float(utt[key].sum() / n.sum())
        pos = np.stack([utt[k] for k in self.pos_keys])
        neg = np.stack([utt[k] for k in self.neg_keys])
        out["d_recon"] = float(pos.sum() / (n.sum() * len(self.pos_keys)))
        out["d_ortho"] = float(neg.sum() / (n.sum() * len(self.neg_keys)))
        out["d_recon_norm"] = out["d_recon"] / math.log(2)
        out["d_ortho_norm"] = out["d_ortho"] / math.log(2)
        out["d_recon_utt"] = float(np.clip(pos, 0, 1).mean())
        out["d_ortho_utt"] = float(np.clip(neg, 0, 1).mean())
        out["clamp_frac_recon"] = float((pos >= 1.0).mean())
        out["clamp_frac_ortho"] = float((neg >= 1.0).mean())
        for key in self.batches[0]:
            if key.startswith("model_"):
                vals = np.asarray([r[key] for r in self.batches], float)
                out[key] = float(vals[~np.isnan(vals)].mean()) if (~np.isnan(vals)).any() else np.nan
        if self.beta and self.prior_reg_weighting:
            "prior_loss = weighting * beta * (sum over used frames of the KL, averaged over the subspaces)"
            prior_total = sum(r["model_prior_loss"] for r in self.batches)
            out["rate_per_frame"] = prior_total / (self.beta * self.prior_reg_weighting) / n.sum()
        else:
            out["rate_per_frame"] = np.nan
        out["check_max_abs_diff"] = float(self.max_abs_diff)
        return out
