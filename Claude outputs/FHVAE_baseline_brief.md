# FHVAE baseline: implementation brief

For Claude Code. The reference is `wnhsu/FactorizedHierarchicalVAE` (Hsu, Zhang & Glass, NeurIPS 2017), written in TensorFlow 1.0 / Python 2.7. Port it to PyTorch the same way TCL was ported: follow the reference source line by line, and deviate only where the DecVAE frame grid forces it. List every deviation in the Protocol sheet.

Reference files to read first:

| File | What it holds |
|---|---|
| `src/models/base_fhvae.py` | priors, objective, discriminative term, μ table (L150-217, L292-301) |
| `src/models/rec_fhvae.py` | recurrent encoders and decoder |
| `src/runners/fhvae_runner.py` | training loop, closed-form μ estimation (`_est_mu1` L473), validation (`_valid` L524), latent dump (`dump_latent` L228), edge padding (`_pad_feats` L628) |
| `src/datasets/segment.py`, `kaldi_ra_dataset.py` | segmentation, `get_label_N` (L395) |
| `src/datasets/datasets_loaders.py` | `get_frame_ra_dataset_conf` (L6): the frame-level readout |
| `conf/model/fhvae/lstm_1L_256_lat_32_32.cfg`, `conf/train/fhvae/e500_p50_lr1e-3_bs256_nbs2000_ad10.cfg` | the configuration `egs/timit/run_fhvae.sh` actually runs |

---

## 0. FHVAE is not "DecVAE with a decoder". Do not reuse `dec_vae.py`

Reuse the infrastructure: the collator, the frame grid, `ts_baselines_pretraining.py` and `latents_post_analysis_ts_baselines.py`. Do not reuse the model. The two differ structurally:

| | DecVAE | FHVAE |
|---|---|---|
| Unit of the frame branch | one 25 ms frame | a **segment** of several frames; one latent per segment |
| Sequence branch | aggregator over the whole utterance (`input_seq_values`) | a latent inferred **per segment**, tied across the utterance only through a learned prior mean μ per utterance |
| Sequence prior | N(0, I) | N(μ_i, 0.5² I), with μ_i a **trainable per-utterance lookup table**, p(μ) = N(0, I) |
| Objective | DELBO, no decoder, no sampling | ELBO with an LSTM decoder, reparameterised sampling, Gaussian likelihood, plus a discriminative term |
| Test utterances | encoded directly | μ has no table row; it is estimated in closed form |

Write it as `models/baselines/fhvae.py` with `build_fhvae(...)`, exposing the same `forward -> dict(loss, ...)` / `encode` interface as `cost.py`, `tfc.py`, `tcl.py` and `cpc.py`. Do not read `input_seq_values`.

---

## 1. Naming trap: the code swaps the paper's indices

| | Paper | Reference code |
|---|---|---|
| Segment latent (content, prior N(0, I)) | z₁ | **`z2`**, `qz2_x`, `_build_z2_encoder(inputs, z1)` |
| Sequence latent (speaker, prior N(μ, σ²I)) | z₂ | **`z1`**, `qz1_x`, `mu1_table`, `latent1_std` |
| Per-utterance prior mean | μ₂ | **`mu1`** |

In the port, name them **`z_seg`**, **`z_seq`** and **`mu_seq`**, and put this table in the module docstring. Every reference line you cite needs translating through it.

---

## 2. The model, exactly as the reference runs it (TIMIT config)

**Order of inference.** `z_seq` is inferred first; `z_seg` is conditioned on the sampled `z_seq` (base_fhvae.py L150-153).

1. **`z_seq` encoder, q(z_seq | x).** 1-layer unidirectional LSTM(256) over the frames of the segment. Take the **last hidden state h** (`rec_z1_enc_out = all_h`; no cell state). Then two separate linear heads give μ and logvar, with no nonlinearity on logvar (`z1_logvar_nl = None`). No FC hidden layers (`hu_z1_enc` is empty). No batch norm (`if_bn = False`).
2. **`z_seg` encoder, q(z_seg | x, z_seq).** Same architecture. The input at every frame is `[frame ; z_seq]`, with `z_seq` tiled over the segment (rec_fhvae.py, `tiled_z1`).
3. **Decoder, p(x | z_seg, z_seq).** 1-layer LSTM(256), zero initial state. The input at **every** step is `concat(z_seq, z_seg)`. It is not autoregressive: `rec_dec_inp_train/test = None`, so no teacher forcing. Each step goes through two linear heads to give a per-frame mean and logvar over the feature dimension (`dense_latent`, no nonlinearities).
4. **Sampling.** Reparameterisation `z = μ + exp(0.5·logvar)·ε` for both latents in training (layers.py L61-62). At evaluation, the reference feeds the **mean** of `z_seq` into the `z_seg` encoder (`_encode_z2_mean_fn`). Do the same.

**Objective, per segment n of utterance i** (base_fhvae.py L158-217):

```
lb      = log p(x | z_seg, z_seq)                                   # Gaussian, learned logvar, summed over frames and dims
        - KL( q(z_seg|x,z_seq) || N(0, I) )
        - KL( q(z_seq|x)       || N(mu_i, sigma^2 I) )               # sigma = 0.5, so logvar_p = log 0.25
        + log N(mu_i; 0, I) / N_i                                    # N_i = number of segments of utterance i
lb_alpha = lb + alpha * log q(i | z_seq_mean)                        # alpha = 10 in the run script
loss    = -mean_over_segments(lb_alpha)
```

- The KL terms are summed over latent dimensions. **No β, no KL annealing, no free bits.**
- **Discriminative term** (L191-194). The logits over **all training utterances** j are `-||E[z_seq] - mu_j||^2 / (2 sigma^2)`, summed over dimensions. They use the **posterior mean** of `z_seq`, not the sample. Take the cross-entropy against the true utterance index i. Compute the logits as `-(||a||² - 2 a·Bᵀ + ||B||²) / (2σ²)`, which is (batch × n_utts), rather than materialising (batch × n_utts × d).
- `log p(μ)/N_i` spreads the μ prior over the utterance's segments, so over a full pass it counts once per utterance.

---

## 3. Segments on the DecVAE grid

- **Reference.** `seg_len = 20`, `seg_shift = 20`, `seg_rand = True`, at Kaldi's 10 ms hop, which gives **200 ms** segments (`egs/timit/local/fbank_data_prep.sh` L23-25).
- **Recommendation: `seg_len = 10` frames on the 20 ms grid**, keeping the reference's 200 ms. This is the same rule used for CPC's k: match the reference in time, not in frame count. Do not choose it from SimVowels' 100 ms vowel duration; that is the rule you set for TCL.
- **Training segmentation** (segment.py `make_seg_list`). Per utterance, `n_segs = (L - seg_len) // seg_shift + 1`, but the start positions are drawn **uniformly at random** from `0 .. L - seg_len` (`seg_rand`). `N_i` in the μ-prior term is that same count (`get_label_N`). L is the **valid** length from `sub_attention_mask`. Segments must lie entirely inside valid frames.
- **Utterances shorter than `seg_len`.** The reference discards them (kaldi_ra_dataset.py L127). Do that in training by masking them out of the loss. **Do not** discard them at evaluation (see §6); that would change the label set relative to the other methods.
- **Batching.** The reference batches 256 random segments drawn from across the dataset. The DecVAE collator batches whole utterances, so segment inside `forward` (unfold along the frame axis) and carry each segment's utterance index and `N_i`. The segments in a batch are then correlated within utterances. The objective is unchanged; only gradient variance differs. For TIMIT (about 150 frames, so about 15 segments per utterance), 16-17 utterances per batch gives about the reference's 256 segments. Whether to match 256 segments or keep the shared batch size is a protocol decision (see §9).

---

## 4. The μ table (transductive)

- `n_class1` = **number of training utterances** (run_nips17_fhvae_exp.py L344). The labels are utterance indices (`set_name = uttid`). Initialise with `torch.randn(n_utts, d_seq)` (L296), as an `nn.Parameter` trained by the main optimiser.
- **The batch has no utterance index today.** Add one to the training split (e.g. `dataset.map(..., with_indices=True)` into a `utt_index` column). Do it **after** `subset_raw_datasets`, so indices are contiguous `0..n-1`. The baseline collator must pass it through, and it must survive the `batch.pop(...)` calls in the loop.
- Use **dense Adam** on the table, not `nn.Embedding(sparse=True)` or `SparseAdam`. TF1's `AdamOptimizer` decays the moments and updates **all** rows every step, even rows absent from the batch. PyTorch's dense Adam does the same; lazy or sparse Adam would not.
- Table size. The discriminative softmax is over every training utterance. It is small for TIMIT (about 3.7k rows). Check SimVowels' count; with the distance trick in §2 the cost is (batch × n_utts) floats.

---

## 5. Validation and early stopping: the gotcha

Dev utterances have no table row, and the discriminative term is undefined for them (the reference omits `log_qy1` from dev metrics, `test_sum_names`, runner L85).

- **Closed-form μ̂** (`_est_mu1`, L473-499): `mu_hat_i = sum_n E[z_seq]_n / (N_i + sigma^2)`, summed over the **non-overlapping** segments of utterance i.
- Because DecVAE batches whole utterances, all segments of a dev utterance are in the same batch. So **compute μ̂ inside `forward` when `not self.training`**; no second pass is needed. Use deterministic non-overlapping segmentation in eval mode (`seg_shift = seg_len`, starts at 0).
- **Validation loss = `-lb` with μ̂ substituted, without the α term.** This is what the reference monitors for model selection (`dev_vals["lb"]`, L175). The shared loop's `EarlyStopping` then acts on that quantity.
- The shared loop sums every key in `loss_keys` over eval batches (ts_baselines_pretraining.py L852). Return every key in eval mode too, with `log_qy = 0` or excluded from `loss_keys`.

---

## 6. Readouts for evaluation

### Frame level (Z branch): sliding window, shift 1, centred, edge-replicated

This is the reference's own frame-level extraction (`get_frame_ra_dataset_conf`, datasets_loaders.py L6-23; `dump_latent` with `_pad_feats`):

1. Windows of `seg_len` frames with **shift 1**, over the valid frames only: `L - seg_len + 1` windows.
2. Encode each window deterministically (means; `z_seq` mean into the `z_seg` encoder).
3. Pad the latent sequence to exactly L by **replicating the first latent** `floor((seg_len-1)/2)` times at the start and **the last latent** `ceil((seg_len-1)/2)` times at the end. The latents are replicated, not the frames. For `seg_len = 10`, that is 4 left and 5 right.
4. Result: exactly one latent per frame, aligned with the existing labels, with no re-alignment.

Details:

- **Needs the mask.** Post-analysis currently calls `encode(frames)` without `sub_attention_mask`. For FHVAE, windows near the end of a padded utterance would read zeros. Pass the mask to `encode` (make it optional in the signature so the other methods are unaffected) and slide only over valid frames. SimVowels is fixed length, but TIMIT and IEMOCAP are not.
- **Utterances with L < `seg_len`.** Edge-replicate the **frames** up to `seg_len`, encode one window, and assign its latent to all L frames, so no utterance leaves the evaluation set.
- **Means only.** The reference's ASR dump concatenates mean **and** logvar (`--use_mean --use_logvar`). For disentanglement, use means, consistent with DecVAE. Note this in the Protocol sheet.
- **Which vector.** Recommended headline: **`[z_seg ; z_seq]` per frame**, FHVAE's full latent state for that window. It parallels DecVAE's Z (a concatenation of subspaces) and CoST's trend ⊕ season. Evaluating `z_seg` alone would score FHVAE on the one subspace it deliberately empties of speaker. Also report `z_seg` alone, fixed in advance, not chosen after seeing results. Use the per-window `z_seq` mean, **not** μ̂ broadcast over frames; broadcasting would inject utterance-level pooling into the frame evaluation.

### Sequence level (S branch): the s-vector μ̂

- Use the reference's s-vector: `mu_hat_i = sum_n E[z_seq]_n / (N_i + 0.25)` over the **non-overlapping** segmentation (`dump_repr` uses `get_nonoverlap_ra_dataset_conf`, run_nips17_fhvae_exp.py L387; `which_repr = "mu1"`).
- This is FHVAE's dedicated sequence latent, the counterpart of DecVAE's S. Up to the factor N/(N+0.25) it **is** mean pooling of `z_seq`, so it stays consistent with the shared `'mean'` rule.
- Interface: add `seq_pooling = 'mu'` (or return μ̂ from `encode`) in `latents_post_analysis_ts_baselines.py`, alongside `'mean'` and `'last'`.

---

## 7. Input features

- Feed the **same `(B, F, 400)` mel patches** DecVAE's encoder reads (`input_values[:, component]`). This is the same choice as CPC; keep `pool_mel_bins` consistent with the other trained baselines. The decoder then reconstructs 400-dim frames. The reference's filterbank front-end (`conf/fbank.conf`, 80 bins, 25 ms window) is the precedent for FHVAE on mel.
- **No extra normalisation.** The reference applies global mean-variance normalisation (`apply_mvn`). DecVAE's collator already applies `mel_norm = global`. Do not stack a second one.

---

## 8. Hyperparameters

| | Reference (TIMIT run) | Port |
|---|---|---|
| Encoders / decoder | LSTM, 1 layer, 256, unidirectional | same |
| `d_seg`, `d_seq` | 32, 32 | z_dim each: **48 (SimVowels) / 64 (TIMIT)**, as CoST's trend/season and TF-C's time/freq; frame readout 96 / 128 |
| σ of p(z_seq \| μ) | 0.5 | same |
| α (discriminative) | 10 (`..._ad10.cfg`, used by `run_fhvae.sh`) | same |
| Optimiser | Adam, lr 1e-3, β₁ 0.95, β₂ 0.999 | same (per-method optimiser, as for the other baselines) |
| LR decay | `lr_decay_factor 0.8` is defined, but **`decay_op` is never run** | none |
| L2 1e-4 | applies only to FC hidden layers, and the config has none, so it is **inert** | none |
| Batch | 256 segments | see §3 |
| Stopping | 500 epochs × 2000 steps, patience 50 epochs on dev `lb` | shared budget and early stopping, monitoring `-lb` (§5) |
| Gradient clipping | `max_grad_norm = None` | none |

---

## 9. Integration checklist

1. `args_configs/fhvae_args.py`: `fhvae_seg_len`, `fhvae_d_seg`, `fhvae_d_seq`, `fhvae_hidden`, `fhvae_seq_std`, `fhvae_alpha`, `fhvae_representation` (`'both'` / `'seg'`), `fhvae_seq_pooling` (`'mu'`), and optimiser fields mirroring `cpc_args.py`.
2. `models/baselines/fhvae.py` + `build_fhvae`, registered in `models/__init__.py`. Expose `FrameGeometry` like TCL and CPC so the collator can derive `sub_attention_mask`.
3. `ts_baselines_pretraining.py`: add `fhvae` to `resolve_method_args`, `build_baseline_model` (it needs `n_train_utts`), `build_optimizer`, and the `HfArgumentParser` tuple. Add the `utt_index` column (§4).
4. Collator: pass `utt_index` through (training split only; it is absent or ignored in eval).
5. `latents_post_analysis_ts_baselines.py`: optional `sub_attention_mask` into `encode`, plus `seq_pooling = 'mu'`.
6. `requires_fixed_batch_size = False`.

**Decisions to confirm before launching:**

- `seg_len = 10` (200 ms).
- Headline frame readout `[z_seg ; z_seq]`.
- Batch sized to about 256 segments, or the shared batch size.

---

## 10. Sanity checks before full runs

1. **Alignment.** For random utterances, `encode(...)` returns exactly L latents per utterance, and the post-analysis label count equals the latent count. This mirrors the CPC check.
2. **Closed form vs table.** After some training, μ̂ computed on **training** utterances should correlate strongly with their `mu_table` rows (cosine above about 0.9). If not, the μ prior, N_i or σ² is wired wrong.
3. **Factorisation.** The speaker accuracy of μ̂ on SimVowels should be high early. The `z_seq` variance across segments **within** an utterance should be much smaller than across utterances.
4. **No collapse.** Neither KL term should go to zero. If `KL(z_seg)` collapses, check that the decoder input is `concat(z_seq, z_seg)` at every step and that nothing is teacher-forced.
5. **Eval μ̂ path.** The validation loss must not use `mu_table`. Assert that `utt_index` is not read when `not self.training`.

---

## Protocol-sheet lines

- **FHVAE input.** 80-band mel patches as DecVAE, `(B, F, 400)`; the reference's `fbank` front-end (80 bins, 25 ms window) is the precedent. No extra MVN; the collator's global normalisation is used.
- **Segment length.** 10 frames at the 20 ms hop = 200 ms, matching the reference's 20 frames at 10 ms. Random segment starts in training, non-overlapping in evaluation, as in the reference.
- **Batching.** Whole utterances, segmented in the forward pass; the reference samples 256 segments across the dataset. The objective is unchanged.
- **Latent sizes.** `d_seg = d_seq = z_dim` (48 / 64) instead of the reference's 32 / 32. σ = 0.5 and α = 10 as in the reference.
- **Frame readout.** Posterior means of `[z_seg ; z_seq]` over shift-1 windows centred on each frame, with edge-replicated latents, following `get_frame_ra_dataset_conf` and `_pad_feats`. The logvar the reference also dumps is not used.
- **Sequence readout.** The reference s-vector `mu_hat = sum E[z_seq] / (N + sigma^2)` over non-overlapping segments.
- **Validation.** `-lb` with closed-form μ̂ for held-out utterances, without the discriminative term, as the reference's model selection.
