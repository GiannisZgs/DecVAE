# Metric sensitivity, subspace-wise analysis and model rankings

Instructions for Claude Code. Repo: `C:\Users\Dell\Files\DecVAE`, branch `v2-dev`. Follow `CLAUDE.md` throughout:

- Back up any existing file before editing it (`<name>.py.bak`).
- Keep edits to existing files minimal. Everything new goes in new files.
- After each change, confirm that untouched code paths behave exactly as before.

## Purpose

| Experiment | Answers |
|---|---|
| E1a. Synthetic latents with known ground truth | R5 comment 7: known limitations of the metrics under factor imbalance, label noise, partial observability, correlated factors (plus dimensionality and code size) |
| E1b. Real latents on SimVowels under the same perturbations | R5 comment 7: how the model rankings change |
| E2. Subspace-wise and interventional analysis on SimVowels | Editor and R4 comment 3: subspace-wise factor prediction, ablation and swapping; the DecVAE mechanism metric |
| R. Ranking script | Model rankings for the results tables (flat, 4-family, 3-family), also used by E1b |

**Models for E1b and E2 (SimVowels, frame level):**

| Model | Post-analysis script | Native groups |
|---|---|---|
| DecVAE + EWT, β = 0.1, `latent_type="all"` | `latents_post_analysis.py` | X, OC1..OCC |
| β-VAE | `latents_post_analysis_vae1D.py` | none |
| FHVAE seg | `latents_post_analysis_ts_baselines.py` | none |
| CoST | `latents_post_analysis_ts_baselines.py` | trend, season |
| TCL | `latents_post_analysis_ts_baselines.py` | none |
| SFA linear | `latents_post_analysis_eigenprojection.py` | none |
| PCA | `latents_post_analysis_eigenprojection.py` | none |
| wav2vec2 + PCA (d = 192) | `latents_post_analysis_frozen_ssl.py` | none |

**DecVAE variant (confirmed by the user):** DecVAE + EWT with β = 0.1 and `latent_type="all"` (X followed by OC1..OCC). This matches the "DecVAE + EWT" row of the SimVowels results table and is the variant used in most of the paper's figures. It is the only DecVAE entry in E1b and in all rankings.

**Optional, E2 only:** also dump DecVAE + FD (β = 0.1, `"all"`) and run the E2 analysis on it as a second DecVAE entry. It costs only forward passes. It shows whether the subspace-to-component alignment depends on the fixed FD band boundaries, which is relevant to R3 comment 2. It is not used in any ranking.

## Order of work

1. **Part A:** dump hook, metric suite, ranking utilities. Run a pilot timing.
2. **Part R:** ranking script. Check it against the expected numbers below.
3. **E1a:** synthetic benchmark. It needs no models and can run while the dumps are being produced.
4. **E2 data:** generate the intervention datasets, then dump all 8 models on the reference set and the intervention sets.
5. **E1b, then E2 analysis.**

---

## Part A: shared infrastructure

### A1. Dump hook

Edit two existing files.

In `args_configs/data_training_args.py`, add with the existing `field(...)` style:

- `eval_dump_only: bool = False`: save the arrays that would enter the metrics, then return without computing anything.
- `eval_dump_dir: str = None`: where the dumps go. Default `<output_dir>/eval_dumps`.
- `eval_dump_tag: str = None`: a free-text name for the model or dataset variant, written into the filename.

In `compute_disentanglement_metrics` (`disentanglement_utils/disentanglement_eval.py`), add a single block right after "Make sure that all arrays are in the correct dimensions". At that point the SimVowels speaker grouping and label encoding are done, and nothing has been split or re-split yet.

```python
if getattr(data_training_args, "eval_dump_only", False):
    _dump_eval_arrays(data_training_args, latent_type, target, mu_train, y_train, mu_test, y_test, extra)
    return {}
```

Write `_dump_eval_arrays` as a small helper in the same file. It saves:

- `<tag>_<latent_type>_<targets>.npz` with `mu_train` and `mu_test` (float32, shape `(d, N)`, original frame order), `y_train` and `y_test` (int, shape `(n_factors, N)`), and `n_train`.
- A JSON sidecar with the target names, dataset, `latent_dim`, number of frames, and the argument values the metrics use (`train_data_percent`, `dev_data_percent`, `disentanglement_eval_cv_splits`, `random_states`, `disentanglement_num_workers`, `discard_label_overlaps`).
- **The speaker ID to vocal-tract mapping** (SimVowels): for each encoded speaker ID, the midpoint of its `speaker_groups` range, taken from the label-preparation block. E1b needs it to coarsen speakers in vocal-tract order.

Only the scripts need to pass `extra`:

- **DecVAE:** `extra["groups"] = [["X", 0, z_dim], ["OC1", z_dim, 2*z_dim], ...]`. The order must follow `all_embs = cat([mu_originals_z, mu_components_z])` in `latents_post_analysis.py`, around line 587.
- **CoST:** trend and season index ranges, if the representation is `both`.

Add an optional `extra=None` argument to `compute_disentanglement_metrics` and pass it through only at those two call sites. Every other call is unchanged.

Run every post-analysis with `measure_disentanglement=true`, `classify=false`, `eval_dump_only=true`, so no metric or classifier runs.

### A2. Metric suite: `disentanglement_utils/metric_suite.py` (new)

A function that reproduces exactly what `compute_disentanglement_metrics` does **after** the dump point, for the SimVowels path, calling the existing functions and nothing else:

```python
def compute_metric_suite(mus, ys, n_train, target, args, split="speaker", random_state=0,
                         unsupervised=True):
    """mus (d, N), ys (F, N) in original order. Returns the same keys as the results JSON."""
```

Steps, mirroring the original:

1. Split `mus` and `ys` at `n_train`.
2. **Speaker re-split:** `StratifiedShuffleSplit(n_splits=1, test_size=dev_data_percent, train_size=train_data_percent, random_state=42)` on the concatenation, stratified on the speaker row. `split="plain"` stratifies on the last factor instead (E1a).
3. `kl_distance_mi.unsupervised_metrics(concat, total_corr=True, wass_corr=False, n_jobs)`, if `unsupervised`.
4. `dci.compute_dci(mu_train, y_train, mu_test, y_test, random_state=random_state)`. SimVowels uses no cross-validation. The original passes `random_state=None`, which is not reproducible; pass a fixed seed here.
5. `irs.compute_irs(mus_combined, ys_combined, diff_quantile=0.99)`.
6. `modularity_explicitness.compute_modularity_explicitness(mu_train, y_train, mu_test, y_test, use_cv=False, n_jobs)`.

**A3. Reproduction check (required).** For each of the 8 models, run the suite on the full reference dump. Every metric must match the values in the results workbook (Results sheet, SimVowels table) to within ±0.01. The tolerance exists only because the original DCI call was unseeded. This check also confirms two things: that the right DecVAE `latent_type` was dumped, and which GCN key the table reports (`gaussian_total_correlation` or `gaussian_total_correlation_norm`). Record both.

### A4. Pilot timing

Time one suite run on a SimVowels dump at 20,000 frames with d = 192, and one synthetic run at d = 768. Report the times before running E1a or E1b. `N_SUB` (default 20,000) may be reduced if the budget needs it.

---

## Part R: rankings

### R1. `latent_analysis_utils/ranking_utils.py` (new)

```python
METRICS = {  # column: (higher_is_better, family)
    "MI": (False, "Independence"), "GCN": (False, "Independence"),
    "DCI-D": (True, "Disentanglement"), "DCI-C": (True, "Disentanglement"), "Modularity": (True, "Disentanglement"),
    "DCI-I": (True, "Informativeness"), "Explicitness": (True, "Informativeness"),
    "Task A": (True, "Informativeness"), "Task B": (True, "Informativeness"), "Task C": (True, "Informativeness"),
    "IRS": (True, "Interventional"), "ICS-factors": (True, "Interventional"),
}
```

`rank_methods(df, metrics=METRICS)` returns a table with:

1. **Metric inclusion:** a metric is used only if every row has a numeric value for it. Excluded metrics are logged.
2. **Per-metric ranks:** each metric is oriented so higher is better, then ranked with `rank(ascending=False, method="average")`. 1 is best.
3. **Family ranks:** the mean of the per-metric ranks within each family.
4. **Aggregations:**
   - `Flat`: mean of all per-metric ranks.
   - `4-family`: mean of the family ranks.
   - `3-family`: mean of the family ranks without Independence.
5. **Leave-one-family-out:** the aggregation recomputed with each family removed in turn, and each method's rank position under each.
6. **Rank range:** each method's minimum and maximum position across all of the above.

Parse cells like `"0.230 +- 0.001"` by reading the leading number. Treat `"N/A"`, `"n/a"`, `"////…"` and empty cells as missing.

### R2. `scripts/post-training/rank_methods.py` (new, CLI)

It reads the planner workbook directly:

```
python rank_methods.py --workbook DecVAE_baselines_planner_v2.xlsx --sheet Results \
    --datasets SimVowels TIMIT --skip-fill FFFFC000 \
    [--exclude-regex "wav2vec2|WavLM|HuBERT"] [--one-variant variants.json] --out rankings.xlsx
```

- Tables are found by their title rows (`SimVowels`, `TIMIT`, `IEMOCAP`) and the header row after each.
- `--skip-fill` drops rows whose first cell has that fill colour. The user marks removed rows in orange, `FFFFC000`.
- `--one-variant` is a JSON listing which rows to keep per method, e.g. only one DecVAE row per dataset.
- Output: one sheet per dataset (per-metric ranks, family ranks, the three aggregations, leave-one-family-out, rank range), plus a Summary sheet.

### R3. Regression check (required)

On the workbook `DecVAE_baselines_planner_v2.xlsx` of 27 Sept, with orange rows skipped and frozen SSL rows excluded (17 methods per dataset), the script must reproduce these mean ranks:

| Dataset | Aggregation | Expected top entries |
|---|---|---|
| SimVowels | Flat | FHVAE seg 5.0; DecVAE + FD 7.0; SFA linear 7.0; β-DecVAE + EWT 7.3; CoST 7.5 |
| SimVowels | 4-family | DecVAE + FD 5.8; FHVAE seg 7.2; DecVAE + EWT 7.2; β-DecVAE + EWT 7.6 |
| SimVowels | 3-family | DecVAE + FD 5.4; DecAE + EWT 6.5; DecVAE + EWT 6.8; β-DecVAE + FD 6.8 |
| TIMIT | Flat | PCA 6.0; SFA linear 7.1; β-DecVAE + EWT 7.2; beta-VAE 7.8; VAE 7.8 |
| TIMIT | 4-family | β-DecVAE + EWT 6.5; PCA 6.8; DecVAE + EWT 7.3 |
| TIMIT | 3-family | DecVAE + EWT 5.7; β-DecVAE + EWT 5.7; DecVAE + FD 7.0 |

For SimVowels, Task C is excluded automatically, because AE, VAE, beta-VAE, PCA and ICA have no value for it. TIMIT uses the eight disentanglement columns only.

---

## E1a: synthetic latents with known ground truth

`scripts/simulations/metric_sensitivity_synthetic.py` (new). No models, only numpy plus `compute_metric_suite(split="plain")`.

### Generator

- **Factors:**
  - A: 5 classes (vowel-like).
  - B: 15 classes (speaker-like).
  - U: continuous, unlabelled. Present only in the partial-observability condition.
- **Codes:** for factor f, draw class means `mu_f[c] ~ N(0, I_b)`; `code_f = mu_f[y_f]`. The informative block is `[code_A, code_B]` (2b dims). When U is active, add a block `u * v_U + 0.1 * eps` with `u ~ N(0, 1)`, scaled so U holds the stated share of informative variance.
- **Mixing:** `M(eta)` is the orthogonal polar factor of `(1 - eta) I + eta Q`, where Q is a fixed random orthogonal matrix. It is applied to the informative block plus unit Gaussian noise (signal-to-noise 1).
- **Padding:** unit-Gaussian noise dims are appended to reach the target d.

**Families** (same information content, different structure):

| Family | eta | Known disentanglement order |
|---|---|---|
| D | 0 | 1 |
| E½ | 0.5 | 2 |
| E1 | 1 | 3 |
| N | informative block replaced by noise | 4 |

For informativeness metrics (DCI-I, Explicitness), the known order is D = E½ = E1 > N.

**Default condition:** N = 10,000 frames, b = 8, d = 48, balanced classes, independent factors, no label noise, A and B both labelled.

### Conditions (one axis at a time, 2 seeds)

| Axis | Levels | How |
|---|---|---|
| Factor imbalance | Zipf a = 1, 2 | class probabilities proportional to `rank^-a`, on both factors |
| Correlated factors | Cramér's V = 0.3, 0.6 | each B class gets a preferred A class (`b mod 5`); sample A from `(1 - lam) uniform + lam * preferred`. Solve lam numerically for the target V and record the value achieved. |
| Label noise | 10%, 30% | flip evaluation labels of both factors to a different class chosen at random; latents stay generated from the true labels |
| Partial observability | U unlabelled at 50% of informative variance; B coarsened 15 → 5 | two separate conditions |
| Dimensionality | d = 192, 768 | more padding dims |
| Code size | b = 1, 32 | same information in fewer or more dims (with b = 32, d = 64 and no padding) |

Runs: 4 families × (1 default + 12 conditions) × 2 seeds = **104 suite runs**.

### Outputs

- `synthetic_results.csv`: one row per family, condition and seed, with all metrics.
- **Validity:** for each metric and condition, Kendall's tau between the metric value and the known family order (with ties for the informativeness metrics). Averaged over seeds.
- **Bias:** the value on family D and on family N under each condition, minus its default value.
- **Figure:** a heatmap of metrics by conditions with two panels, validity and bias on D.

---

## E1b: real latents under the same perturbations

`scripts/post-training/metric_sensitivity_real.py` (new). Input: the reference dumps of the 8 models, listed in `config_files/sensitivity/models.json` (model name, dump path, `in_ranking`).

For each model, load `mus` and `ys` (the train and test parts concatenated, original order) and run these conditions through `compute_metric_suite`:

| Condition | Runs | How |
|---|---|---|
| `ref_full` | 1 | all frames, no change (this is A3) |
| `null` | 5 | cyclic label shift on the full, original-order data. Offset `s ~ U[N/4, 3N/4]`; `np.roll(ys, s, axis=1)` on all factor rows together; the latents are unchanged. |
| `ref_sub` | 1 | stratified subsample of `N_SUB` frames over (vowel, speaker) cells, fixed seed. **This is the baseline for every condition below.** |
| Imbalance | 2 | Zipf a = 1, 2: cell weights `w_v * w_s`, sample `N_SUB` frames (without replacement where possible; flag any condition that needs replacement) |
| Correlated factors | 2 | Cramér's V = 0.3, 0.6: each speaker gets a preferred vowel (speakers in vocal-tract order, `rank mod n_vowels`); cell weights `(1 - lam)/n_vowels + lam * [v == pref(s)]`, lam solved for the target; record the V achieved |
| Label noise | 2 | 10%, 30% flips on the `ref_sub` frames, both factors |
| Partial observability | 2 | speakers coarsened into 5 and into 3 groups of adjacent vocal-tract values (using the sidecar mapping), on the `ref_sub` frames |
| Dimensionality | 1 | append d noise dims to the `ref_sub` latents, each Gaussian with standard deviation equal to the median per-dim standard deviation of that representation |

Per model that is 16 runs, 128 in total.

- **MI and GCN** don't use labels. Compute them only for `ref_full`, `ref_sub` and dimensionality, and carry the `ref_sub` value into the other conditions.
- **The cyclic shift** is valid because frames are utterance-contiguous in the dump: `shuffle=False` in every eval dataloader. Assert this from the sidecar.
- **Resampling:** all of it uses fixed seeds, recorded in the output.

### Outputs

- `real_results.csv`: model, condition, level, seed, all metrics, plus the V or a achieved.
- **Null summary:** the mean and SD over the 5 draws for each supervised metric, and the calibrated value `(ref_full - null_mean) / (1 - null_mean)`, **reported mainly for IRS**.
- **Rankings per condition:** `ranking_utils.rank_methods` on the 8 models, for the three aggregations.
  - Kendall's tau of each ranking against the `ref_sub` ranking.
  - Each model's position.
  - Figure: rank trajectories across conditions, one line per model, one panel per aggregation.
- **Per-metric change** from `ref_sub` for every model and condition (supplement table).

---

## E2: subspace-wise and interventional analysis

### E2a. Intervention datasets

`scripts/simulations/simulated_vowels_interventions.py` (new). Import the constants and `bandpass_filter` from `simulated_vowels.py`; do not edit that file.

**Generator.** Write `generate_segment(formants, f0, rngs, noise_vec=None)`. It is the same signal model as `generate_time_domain_signal`, with explicit formant frequencies and f0, and three separate `np.random.Generator`s:

- one for the formant white noise;
- one for the excitation noise bursts;
- one for the additive noise.

For each segment, draw the additive noise once, at the SNR of the base signal, and **reuse the same noise vector in every paired variant**. The variants then differ only in the parameter that was changed.

**Base set.** 100 utterances using the test vocal-tract ranges (20 per speaker), each 40 segments of 0.1 s, as in the dataset. Seeds per segment: `base_seed + 1000 * utt + seg`. **Adjacent vowels are always different**, so every variant has the same overlap mask and frames stay aligned after the overlap frames are removed.

**Paired variants,** each written as its own dataset:

| Variant | Change | Type |
|---|---|---|
| `F1`, `F2`, `F3` | multiply that formant by (1 + δ), δ = ±0.08 with a random sign per utterance | component |
| `f0` | multiply f0 by (1 + δ) | component |
| `speaker` | vocal-tract factor changed to another test speaker's value; all formants and f0 follow | factor |
| `vowel` | every segment gets a different vowel, keeping adjacent vowels different | factor |
| `nuisance` | same factors, fresh noise seeds | reference |

Also save a JSON with the true per-segment F1, F2, F3, f0, vocal-tract value and vowel for every variant.

**Loader format.** Use the same `json.gz` format and filename pattern that `load_sim_vowels` expects (`SNR_15`, `vowels_5`, `4s`, the split name). Each variant goes in its own `data_dir`. Check which splits each post-analysis script reads, and place the file there; `load_sim_vowels` also expects train and dev files. Give DecVAE's decomposition caches distinct filenames per variant so nothing is overwritten.

**Dumping.** Dump all 8 models on the base set and on each variant (64 forward passes). Assert that every variant has the same number of frames as the base and identical labels, except for the factor that was changed.

### E2b. Metrics: `scripts/post-training/subspace_interventional_analysis.py` (new)

**1. Interventional response.** Per model, per latent dim j and per variant k, over the aligned frames i:

```
r[j, k]  = mean_i |z_k[i, j] - z_base[i, j]| / sigma_j     # sigma_j = std of dim j over base frames
r'[j, k] = max(r[j, k] - r[j, nuisance], 0)                # remove sensitivity to the noise realisation
```

Drop dims with `sigma_j < 1e-8`.

**2. Interventional Component Selectivity (ICS).** For a set of variants K:

```
p[j, k] = r'[j, k] / sum_k r'[j, k]
s_j     = 1 - H(p[j, :]) / log|K|
w_j     = sum_k r'[j, k] / sum_{j,k} r'[j, k]
ICS(K)  = sum_j w_j s_j
```

Dims with zero total response get zero weight.

- **`ICS-factors`**, with K = {vowel, speaker}. Interventional disentanglement of the semantic factors under true interventions. It joins IRS in the Interventional family for the 8-model rankings of E1b; add it to `real_results.csv` under `ref_full`.
- **`ICS-components`**, with K = {F1, F2, F3, f0}. The DecVAE mechanism metric: does each dimension respond to one generative component? It is reported per model and **not** used in any ranking.

**3. Group response matrices** (DecVAE, and CoST if it has groups): `R[g, k]` = the mean of `r'[j, k]` over the dims in group g, normalised per row. Plot as heatmaps, groups × {F1, F2, F3, f0, vowel, speaker}.

**4. Semantic subspace evidence on the reference dumps** (DecVAE, and CoST if grouped). This covers R4 comment 3's first two tests; the swap test is covered by the `vowel` and `speaker` columns of step 3.

- **Prediction matrix `P[g, f]`:** logistic regression (standardised inputs, `max_iter=1000`) trained on group g alone, predicting vowel and speaker. Use the speaker-stratified split from A2. Score with chance-normalised balanced accuracy, `(bacc - 1/K_f) / (1 - 1/K_f)`.
- **Ablation matrix `A[g, f]`:** the score using all groups, minus the score using all groups except g (probe retrained).

### Outputs

- `ics_results.csv`: ICS-factors and ICS-components per model.
- `group_response_<model>.csv` and `.png`.
- `subspace_prediction_ablation_<model>.csv` and `.png`.

---

## Checks before reporting

1. A3 reproduction within ±0.01 for all 8 models.
2. R3 reproduction of the expected ranks.
3. `null` with a shift of 0 equals `ref_full` exactly.
4. For E1a, the default condition on family D gives clearly higher DCI-D than family N, and the N family's IRS at d = 48 is around 0.4. That matches an earlier check of IRS on pure noise: 0.39 for Gaussian latents at 20,000 frames.
5. For E2, the base re-encoded twice gives r = 0 exactly, so every encoder is deterministic at evaluation. Any model that fails this is reported, not silently averaged.

## Do not change

The existing metric functions, the label preparation, any model or training code, or `simulated_vowels.py`. The only edits to existing files are the dump flags, the dump block, and the optional `extra` argument at the DecVAE and CoST call sites.
