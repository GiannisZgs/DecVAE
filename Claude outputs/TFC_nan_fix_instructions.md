# TF-C: fix the NaN during pre-training

Repo: `C:\Users\Dell\Files\DecVAE`. Files touched: `models/baselines/tfc.py`, `scripts/post-training/latents_post_analysis_ts_baselines.py`, and the TF-C JSON configs. Change nothing else about TF-C.

## Diagnosis (why these changes)

`TFC.forward` currently reshapes to `(1, N, d)`. `nn.TransformerEncoderLayer` defaults to `batch_first=False`, so that is read as **sequence length 1, batch N**. With a single key the softmax is always 1, so `W_q` and `W_k` receive no gradient from the loss. Adam's coupled weight decay is then the only force on them, and once `sqrt(v)` falls below `eps` it shrinks them by about `1 - lr*wd/eps` per step. Measured: the Q and K gradients fall from 1e-7 to 1e-30 while V stays near 7e-4, and the first NaN appears in `tfc.transformer_encoder_f.layers.0.self_attn.in_proj_weight`.

The reference does **not** do this. In `mims-harvard/TFC-pretraining`, `code/TFC/model.py` L33 passes `x_in_t` of shape `[128, 1, 178]` (see the `trainer.py` L116 comment) straight into the encoder with `batch_first=False`, so **the sequence is the 128 samples of the batch and the batch is 1**. Attention runs across the samples of a batch. Restoring that layout both matches the reference and removes the NaN at its source, because `W_q` and `W_k` get real gradients again.

---

## Fix 1 (required): encoder input layout

### 1a. `TFC.forward` in `models/baselines/tfc.py`

Replace the whole method. It now takes flat `(N, d)` tensors:

```python
    def forward(self, x_t, x_f):
        """
        Args:
            x_t: (N, ts_length) frames.
            x_f: (N, ts_length) magnitude spectra of those frames.
        Returns:
            h_time, z_time, h_freq, z_freq, each (N, ...).

        The reference feeds [samples, 1, length] into an encoder built with batch_first=False
        (model.py L33-41), so the sequence is the samples of the batch and the batch is 1.
        Attention therefore spans the frames of a batch.
        """
        h_time = self.transformer_encoder_t(x_t.unsqueeze(1)).squeeze(1)
        z_time = self.projector_t(h_time)

        h_freq = self.transformer_encoder_f(x_f.unsqueeze(1)).squeeze(1)
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq
```

Update the class docstring for `ts_length` if it mentions sequence length 1.

### 1b. `TFCForPreTraining.forward`

The sampled frames are already flat and padding-free, so drop the `squeeze` lambda and the `unsqueeze(0)` wrappers:

```python
        h_t, z_t, h_f, z_f = self.tfc(x_t, x_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.tfc(aug_t, aug_f)
```

Everything else in `forward` stays: the frame sampling, the augmentations, the four NT-Xent terms, `lam`, and the returned dict.

### 1c. `TFCForPreTraining.encode`

Padded frames must not enter the sequence now, because they would be attended to. Select the valid frames, encode them together, scatter the results back:

```python
    def encode(self, input_values, sub_attention_mask=None):
        """
        Representation of every frame, for the evaluation path.

        Attention spans the frames of the batch (Fix 1), so padded frames are left out of the
        sequence and their rows come back as zeros.
        """
        batch, frames, _ = input_values.shape
        flat = input_values.reshape(batch * frames, -1)

        if sub_attention_mask is not None:
            valid = sub_attention_mask.reshape(-1).to(torch.bool)
        else:
            valid = torch.ones(batch * frames, dtype=torch.bool, device=input_values.device)

        x_t = flat[valid]
        _, z_time_valid, _, z_freq_valid = self.tfc(x_t, self.spectrum(x_t))

        z_time = flat.new_zeros(batch * frames, z_time_valid.shape[-1])
        z_freq = flat.new_zeros(batch * frames, z_freq_valid.shape[-1])
        z_time[valid] = z_time_valid
        z_freq[valid] = z_freq_valid
        z_time = z_time.reshape(batch, frames, -1)
        z_freq = z_freq.reshape(batch, frames, -1)

        if self.representation == "time":
            representation = z_time
        elif self.representation == "freq":
            representation = z_freq
        elif self.representation == "both":
            representation = torch.cat([z_time, z_freq], dim=-1)
        else:
            raise ValueError(
                f"Unknown tfc_representation {self.representation}, expected 'both', 'time' or 'freq'"
            )

        return representation, z_time, z_freq
```

Note the masking at the end of the old `encode` is no longer needed: padded rows are already zero.

---

## Fix 2 (required): pass the padding mask into `encode`

In `scripts/post-training/latents_post_analysis_ts_baselines.py`, `gather_split` calls `representation_function.encode(frames)` with no mask. TF-C now needs it, and FHVAE will too.

Pass `sub_attention_mask` when the method's `encode` accepts it, so CoST, TCL and CPC are unaffected:

```python
            import inspect
            accepts_mask = "sub_attention_mask" in inspect.signature(
                representation_function.encode).parameters
            outputs = (representation_function.encode(frames, sub_attention_mask)
                       if accepts_mask else representation_function.encode(frames))
```

Hoist the `inspect` import and the `accepts_mask` check out of the batch loop.

Keep the existing `frames = frames * sub_attention_mask...` line: zeroing padded frames before encoding is harmless and still correct.

---

## Fix 3 (required): revert the spectrum scaling

If `spectrum()` still divides by `x.shape[-1]`, revert it. Input scale was never the cause:

```python
    @staticmethod
    def spectrum(x):
        "Magnitude spectrum of every frame, as the reference dataloader builds it"
        return fft.fft(x, dim=-1).abs()
```

---

## Fix 4 (required): config changes

In all TF-C configs under `config_files/baselines/tfc/` (pre-training and fine-tuning, every dataset):

| Key | From | To | Why |
|---|---|---|---|
| `adam_epsilon` | 1e-6 | **1e-8** | The reference calls `torch.optim.Adam(..., betas, weight_decay=3e-4)` with no eps, so it uses the torch default (`main.py` L112). |

While in the same files, these were already agreed:

| Key | From | To |
|---|---|---|
| `tfc_projector_dim` (TIMIT only) | 48 | **64** |
| `per_device_train_batch_size` | 4 / 8 | **16 (SimVowels) / 8 (TIMIT)** |
| `num_train_epochs` | 40 | **150** |
| `early_stop_warmup_steps` | 10000 | **28000 (SimVowels) / 41568 (TIMIT)** |
| `clip_grad_value` | 1.0 | **null** |

In the TF-C `latent_evaluations` configs, set `per_device_eval_batch_size` to **16 (SimVowels) / 8 (TIMIT)**. Attention now spans the batch, so an embedding depends on the batch it was encoded in, and evaluation must mirror the training composition.

---

## Fix 5 (required): remove the debugging scaffolding

Delete anything added while chasing this:

- `torch.backends.cuda.enable_mem_efficient_sdp(False)` / `enable_flash_sdp(False)` and any `sdpa_kernel(SDPBackend.MATH)` context.
- `torch.autograd.set_detect_anomaly(True)` and the forward hooks.
- The Q / K / V gradient printouts, once the verification below passes.

---

## Verification

Run TF-C on SimVowels for at least 300 steps, past the step where it used to fail.

1. **Q and K now receive gradients.** After `accelerator.backward(loss)`, before the optimizer step:
   ```python
   attn = accelerator.unwrap_model(model).tfc.transformer_encoder_f.layers[0].self_attn
   g, d = attn.in_proj_weight.grad, attn.embed_dim
   print(g[:d].abs().max(), g[d:2*d].abs().max(), g[2*d:].abs().max())
   ```
   All three should be of a similar order (roughly 1e-4), and the Q and K values must stay stable rather than shrinking step after step.
2. **No non-finite parameters:** `assert all(torch.isfinite(p).all() for p in model.parameters())` every 50 steps.
3. **The loss moves.** `time_loss`, `freq_loss` and `time_frequency_loss` should all change; a flat curve means something else is wrong.
4. **Alignment is unchanged.** `encode` on a batch still returns `(batch, frames, 2 * tfc_projector_dim)`, with rows for padded frames exactly zero and the count of valid rows equal to `sub_attention_mask.sum()`.
5. **Padding does not change valid embeddings.** Encode one utterance alone, then in a batch padded to a longer length, and confirm the valid rows are identical (they will differ if the mask is not reaching the encoder).

## Do not change

The augmentations, `lam`, the temperature, the NT-Xent poly loss, the projector sizes, the two-layer / two-head transformer, the learning rate, the betas, the weight decay, the waveform input, or `tfc_frames_per_batch_entry` (100 SimVowels / 178 TIMIT, the 50%-of-frames rule).
