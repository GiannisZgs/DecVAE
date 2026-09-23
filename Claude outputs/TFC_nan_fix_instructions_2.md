# TF-C: fix the NaN during pre-training

Repo: `C:\Users\Dell\Files\DecVAE`. Files touched: `models/baselines/tfc.py`, `scripts/post-training/latents_post_analysis_ts_baselines.py`, and the TF-C JSON configs. Change nothing else about TF-C.

## Diagnosis (why these changes, revised)

`TFC.forward` currently reshapes to `(1, N, d)`. `nn.TransformerEncoderLayer` defaults to `batch_first=False`, so that is read as **sequence length 1, batch N**. With a single key the softmax is always 1, so `W_q` and `W_k` receive no gradient from the loss. Adam's coupled weight decay is then the only force on them, and once `sqrt(v)` falls below `eps` it shrinks them by about `1 - lr*wd/eps` per step. Measured: the Q and K gradients fall from 1e-7 to 1e-30 while V stays near 7e-4, and the first NaN appears in `tfc.transformer_encoder_f.layers.0.self_attn.in_proj_weight`.

An earlier version of this document pointed at the reference's literal shape as the fix: `mims-harvard/TFC-pretraining`, `code/TFC/model.py` L33, passes `[128, 1, 178]` into an encoder built with the `batch_first=False` default, so PyTorch reads sequence = the 128 samples of the batch, batch = 1, and attention spans the whole minibatch. That does give `W_q`/`W_k` real gradients and removes the NaN, but it does it by letting a frame's representation attend over every other sample in the minibatch, which routinely mixes frames from different utterances and different speakers. You flagged this correctly: nothing about TF-C's method requires that, it isn't what the paper's per-sample encoder description implies, and it is very likely just a `batch_first` default trap in the reference code, not an intended design choice. Either way it breaks the same rule already applied to CPC's negative sampling: a frame-level (Z-branch) representation must not be able to see other utterances or speakers in the batch.

So the fix keeps to a third layout, which resolves the NaN the same way (a real, non-degenerate sequence length for `W_q`/`W_k` to attend over) without ever crossing an utterance boundary:

**Sequence = one utterance's own sampled frames. Batch = the utterances in the training batch.** Set `batch_first=True` explicitly on the encoder layers (don't rely on the default either way), and pad short sequences with a `src_key_padding_mask` so the padding never contaminates a real frame's attention. During pre-training this sequence length is `tfc_frames_per_batch_entry` (100 for SimVowels, ~50% of an utterance's own valid frames for TIMIT); during evaluation it's however many frames that utterance has. Both are comfortably >1, so `W_q`/`W_k` get real gradients either way.

This is a deviation from the reference's literal (and, on the evidence above, probably accidental) tensor layout. Record it as such in the parameter table, with this rationale.

---

## Fix 1 (required): scope attention to one utterance's own frames

### 1a. Encoder construction in `models/baselines/tfc.py`

Find where `self.transformer_encoder_t` / `self.transformer_encoder_f` are built (`nn.TransformerEncoder(nn.TransformerEncoderLayer(...), num_layers=...)` in `TFC.__init__`) and add `batch_first=True` to both `TransformerEncoderLayer` calls explicitly:

```python
        encoder_layer_t = nn.TransformerEncoderLayer(
            d_model=..., nhead=..., dim_feedforward=..., batch_first=True)
        self.transformer_encoder_t = nn.TransformerEncoder(encoder_layer_t, num_layers=...)

        encoder_layer_f = nn.TransformerEncoderLayer(
            d_model=..., nhead=..., dim_feedforward=..., batch_first=True)
        self.transformer_encoder_f = nn.TransformerEncoder(encoder_layer_f, num_layers=...)
```

Keep every other argument (`d_model`, `nhead`, `dim_feedforward`, dropout, activation) as they are now.

### 1b. `TFC.forward`

```python
    def forward(self, x_t, x_f, key_padding_mask=None):
        """
        Args:
            x_t, x_f: (utterances_in_batch, frames_per_utterance, ts_length). Each row of the
                batch dimension is one utterance; the sequence dimension is that utterance's
                own sampled or valid frames only.
            key_padding_mask: (utterances_in_batch, frames_per_utterance) bool, True at padded
                positions. None when every utterance in the batch contributes the same number
                of frames (e.g. SimVowels' fixed tfc_frames_per_batch_entry).
        Returns:
            h_time, z_time, h_freq, z_freq, each (utterances_in_batch, frames_per_utterance, ...).

        batch_first=True (set in __init__) makes the sequence dimension a single utterance's own
        frames and the batch dimension the utterances themselves, so attention never crosses an
        utterance boundary.
        """
        h_time = self.transformer_encoder_t(x_t, src_key_padding_mask=key_padding_mask)
        z_time = self.projector_t(h_time)

        h_freq = self.transformer_encoder_f(x_f, src_key_padding_mask=key_padding_mask)
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq
```

Update the class docstring for `ts_length` if it mentions sequence length 1 or batch-spanning attention.

### 1c. `_sample_frames` in `TFCForPreTraining`: sample per utterance, not from a flattened pool

The current version flattens `sub_attention_mask` across the whole training batch and draws `frames_per_batch_entry * batch` indices from that flat pool with one `randperm`, so the sampled frames are not grouped by utterance. Replace it so every utterance is sampled independently, from its own valid frames only, and the grouping is kept:

```python
    @staticmethod
    def _sample_frames(x_t, x_f, sub_attention_mask, frames_per_batch_entry, device):
        """
        Args:
            x_t, x_f: (batch, num_frames, ts_length), every frame of the batch (valid + padded).
            sub_attention_mask: (batch, num_frames), 1 at valid (real) frames, 0 at padding.
            frames_per_batch_entry: int for a fixed count per utterance (SimVowels), or None to
                sample ~50% of each utterance's own valid frames (TIMIT).
        Returns:
            out_t, out_f: (batch, max_k, ts_length), sampled frames, padded with zeros where an
                utterance contributed fewer than max_k frames.
            pad_mask: (batch, max_k) bool, True at the padded positions (feed as
                src_key_padding_mask). All-False when every utterance sampled the same count.

        Sampling is independent per utterance and draws only from that utterance's own valid
        frames, so nothing from another utterance ever enters its sequence.
        """
        batch = x_t.shape[0]
        sampled_t, sampled_f, counts = [], [], []
        for b in range(batch):
            valid_idx = sub_attention_mask[b].nonzero(as_tuple=True)[0]
            n_valid = valid_idx.numel()
            k = frames_per_batch_entry if frames_per_batch_entry is not None else max(1, n_valid // 2)
            k = min(k, n_valid)
            perm = valid_idx[torch.randperm(n_valid, device=device)[:k]]
            sampled_t.append(x_t[b, perm])
            sampled_f.append(x_f[b, perm])
            counts.append(k)

        max_k = max(counts)
        out_t = x_t.new_zeros(batch, max_k, x_t.shape[-1])
        out_f = x_f.new_zeros(batch, max_k, x_f.shape[-1])
        pad_mask = torch.ones(batch, max_k, dtype=torch.bool, device=device)
        for b in range(batch):
            k = counts[b]
            out_t[b, :k] = sampled_t[b]
            out_f[b, :k] = sampled_f[b]
            pad_mask[b, :k] = False

        return out_t, out_f, pad_mask
```

For SimVowels every utterance has the same 199 frames, so `frames_per_batch_entry=100` gives `counts` all equal and `pad_mask` all `False` (no real padding). For TIMIT, `frames_per_batch_entry=None` (50% of that utterance's own valid frames, per the earlier decision) gives varying counts, and the mask does real work.

### 1d. `TFCForPreTraining.forward`

Sample once, apply the existing augmentations to the sampled (now utterance-grouped) tensors exactly as before, run both views through `self.tfc` with the mask, then drop back to the flat `(total_valid_frames, dim)` shape the NT-Xent loss already expects:

```python
        x_t, x_f, key_padding_mask = self._sample_frames(
            x_t, x_f, sub_attention_mask, self.frames_per_batch_entry, x_t.device)

        # augmentations: unchanged, applied elementwise to x_t / x_f (same shape, same mask)
        aug_t = self.jitter(...)
        aug_f = self.remove_frequency(self.add_frequency(...))

        h_t, z_t, h_f, z_f = self.tfc(x_t, x_f, key_padding_mask=key_padding_mask)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.tfc(aug_t, aug_f, key_padding_mask=key_padding_mask)

        valid = (~key_padding_mask).reshape(-1)
        flatten = lambda h: h.reshape(-1, h.shape[-1])[valid]
        h_t, z_t, h_f, z_f = flatten(h_t), flatten(z_t), flatten(h_f), flatten(z_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = flatten(h_t_aug), flatten(z_t_aug), flatten(h_f_aug), flatten(z_f_aug)
```

Everything downstream (the four NT-Xent terms, `lam`, the returned dict) stays exactly as it is: it already treats its inputs as a flat set of frame embeddings and compares different utterances' frames as positives/negatives in the loss, which is TF-C's own contrastive mechanism, not an attention-time leak, so it does not need to change.

### 1e. `TFCForPreTraining.encode`

```python
    def encode(self, input_values, sub_attention_mask=None):
        """
        Representation of every frame, for the evaluation path.

        Args:
            input_values: (batch, frames, ts_length).
            sub_attention_mask: (batch, frames), 1 at valid frames, 0 at padding.

        batch_first=True scopes attention to each utterance's own frame axis, so this is
        correct for any batch size or utterance order at evaluation: an embedding never
        depends on which other utterances share its batch.
        """
        key_padding_mask = None
        if sub_attention_mask is not None:
            key_padding_mask = ~sub_attention_mask.to(torch.bool)

        _, z_time, _, z_freq = self.tfc(
            input_values, self.spectrum(input_values), key_padding_mask=key_padding_mask)

        if sub_attention_mask is not None:
            z_time = z_time * sub_attention_mask.unsqueeze(-1)
            z_freq = z_freq * sub_attention_mask.unsqueeze(-1)

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

`per_device_eval_batch_size` for the TF-C `latent_evaluations` configs does **not** need to change. With attention scoped per utterance, an embedding no longer depends on which other utterances are in its batch, so evaluation batch size and order are free choices again (this replaces the earlier note that evaluation had to mirror the training batch composition).

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
5. **No cross-utterance leakage.** Encode one utterance alone, then again inside a batch with several other (different) utterances, in a different order and padded to a different length. The embeddings for that utterance's own frames must be identical in both cases. (They will differ if the mask isn't reaching every layer, or if any reshape flattens the batch and sequence dimensions together.)
6. **Padding does not change valid embeddings.** Same check as 5, restricted to varying only the padding length: encode a batch where this utterance is the longest, then one where it's padded further, and confirm its rows are unchanged.

## Do not change

The augmentations, `lam`, the temperature, the NT-Xent poly loss, the projector sizes, the two-layer / two-head transformer, the learning rate, the betas, the weight decay, the waveform input, or `tfc_frames_per_batch_entry` (100 SimVowels / 50% of valid frames TIMIT).
