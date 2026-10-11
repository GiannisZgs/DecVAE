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

"""Synthetic checks of latent_analysis_utils.block_structure_utils (no model, no data).

    python scripts/post-training/block_structure_selftest.py

1. disjoint unit blocks shared by all samples, original = even mixture: cos_ps 0, JS_neg ln 2, js_mix 0, CI 1,
   js_pos j_C (0.318 for C = 3);
2. the same blocks with the unit assignment permuted per sample: cos_ps 0, but CI near 0;
3. identical distributions for all components: cos_ps 1, JS_neg 0;
plus js_prob against divergence_utils.js_rows, j_C for C = 2, 3, 4, and the Z-branch masking of add().
The cross-check against the model's own S-branch divergences runs inside block_structure.py.
"""

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import math
from types import SimpleNamespace

import numpy as np
import torch

from latent_analysis_utils.divergence_utils import js_rows
from latent_analysis_utils.block_structure_utils import BlockStructure, js_prob, j_C

C, H, N = 3, 192, 2000
OFF = -60.0  # logit of the units outside a component's block
LN2 = math.log(2)
FLOOR = 2 * H * 1e-8  # JS of identical rows under the eps convention of js_rows is about H * eps


def measures(acc):
    "summary() as {measure: value} for the scalar measures and {measure: [values]} for the per-component ones"
    out = {}
    for m, i, j, v in acc.summary():
        if i is None:
            out[m] = v
        else:
            out.setdefault(m, []).append(v)
    return out


def block_logits(gen, perm=None):
    "Components with disjoint blocks of H / C units; the original is the log of their even mixture"
    R = torch.full((C, N, H), OFF, dtype=torch.float64)
    size = H // C
    for i in range(C):
        R[i, :, i * size:(i + 1) * size] = torch.randn(N, size, generator=gen, dtype=torch.float64)
    if perm is not None:
        R = torch.stack([R[:, n, perm[n]] for n in range(N)], dim=1)
    P = torch.softmax(R, dim=-1).mean(dim=0).log()
    return P, R


def check(name, cond, msg):
    print(("PASS " if cond else "FAIL ") + name + ": " + msg)
    return cond


def main():
    gen = torch.Generator().manual_seed(0)
    ok = True

    a, b = torch.randn(50, H, generator=gen), torch.randn(50, H, generator=gen)
    d = (js_prob(torch.softmax(a.double(), -1), torch.softmax(b.double(), -1)) - js_rows(a, b)).abs().max().item()
    ok &= check("js_prob", d < 1e-12, f"max |js_prob - js_rows| = {d:.1e}")
    jc = [round(j_C(c), 3) for c in (2, 3, 4)]
    ok &= check("j_C", jc == [0.216, 0.318, 0.380], f"j_C(2, 3, 4) = {jc}")

    "1. shared disjoint blocks"
    P, R = block_logits(gen)
    acc = BlockStructure(C, "s")
    acc.add_samples(P, R)
    m = measures(acc)
    print(f"   js_pos {np.round(m['js_pos'], 4)}, js_neg {np.round(m['js_neg'], 4)}, cos_ps {np.round(m['cos_ps'], 6)}, "
          f"js_mix {m['js_mix']:.2e}, CI {m['CI']:.4f}, C_mass {m['C_mass']:.4f}, C_null {m['C_null']:.4f}")
    ok &= check("1 cos_ps", max(m["cos_ps"]) < 1e-6, "~0")
    ok &= check("1 js_neg", max(abs(v - LN2) for v in m["js_neg"]) < 1e-3, "~ln 2")
    ok &= check("1 js_mix", m["js_mix"] < FLOOR, "~0 (eps floor)")
    ok &= check("1 js_pos", all(round(v, 3) == round(j_C(C), 3) for v in m["js_pos"]), f"= j_C = {j_C(C):.3f}")
    ok &= check("1 CI", abs(m["CI"] - 1) < 1e-6, "~1")
    ok &= check("1 unit_count", m["unit_count"] == [H // C] * C, f"{m['unit_count']}")
    ok &= check("1 cos_ds", max(m["cos_ds"]) < 1e-6, "~0")

    "2. the same blocks, unit assignment permuted per sample"
    perm = [torch.randperm(H, generator=gen) for _ in range(N)]
    P, R = block_logits(gen, perm)
    acc = BlockStructure(C, "s")
    acc.add_samples(P, R)
    m = measures(acc)
    print(f"   js_neg {np.round(m['js_neg'], 4)}, cos_ps {np.round(m['cos_ps'], 6)}, CI {m['CI']:.4f}, "
          f"C_mass {m['C_mass']:.4f}, C_oracle {m['C_oracle']:.4f}, C_null {m['C_null']:.4f}, cos_ds {np.round(m['cos_ds'], 3)}")
    ok &= check("2 cos_ps", max(m["cos_ps"]) < 1e-6, "~0")
    ok &= check("2 js_neg", max(abs(v - LN2) for v in m["js_neg"]) < 1e-3, "~ln 2")
    ok &= check("2 CI", m["CI"] < 0.05, "~0")
    ok &= check("2 cos_ds", min(m["cos_ds"]) > 0.9, "high")

    "3. identical components"
    Rid = torch.randn(1, N, H, generator=gen, dtype=torch.float64).expand(C, N, H).clone()
    acc = BlockStructure(C, "s")
    acc.add_samples(torch.randn(N, H, generator=gen, dtype=torch.float64), Rid)
    m = measures(acc)
    ok &= check("3 cos_ps", max(abs(v - 1) for v in m["cos_ps"]) < 1e-12, f"{np.round(m['cos_ps'], 6)}")
    ok &= check("3 js_neg", max(m["js_neg"]) < FLOOR, f"{np.round(m['js_neg'], 8)}")

    "Z-branch add(): only the valid frames are samples"
    B, F = 4, 25
    mask = torch.rand(B, F, generator=gen) > 0.3
    out = SimpleNamespace(unmasked_frames_features_z=torch.randn(B, F, H, generator=gen),
                          decomposition_features_z=torch.randn(C, B, F, H, generator=gen), mask_time_indices=mask)
    acc = BlockStructure(C, "z")
    acc.add(out)
    ref = BlockStructure(C, "z")
    ref.add_samples(out.unmasked_frames_features_z[mask], out.decomposition_features_z[:, mask])
    same = all(abs(x[3] - y[3]) < 1e-12 or (np.isnan(x[3]) and np.isnan(y[3])) for x, y in zip(acc.summary(), ref.summary()))
    ok &= check("z mask", acc.n == int(mask.sum()) and same, f"{acc.n} samples of {B * F} frames")

    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
