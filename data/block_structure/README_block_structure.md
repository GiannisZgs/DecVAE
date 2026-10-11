# Block structure of the decomposition embeddings (Proposition 2, SI Section 7)

Written by `scripts/post-training/block_structure.py` (metrics: `latent_analysis_utils/block_structure_utils.py`).
No training: forward passes of the existing checkpoints, `model.eval()`, `torch.no_grad()`, same collator and
`prepare_batch` as `decomposition_divergences.py`.

## Commands

```
python scripts/post-training/block_structure_selftest.py
python scripts/post-training/block_structure.py --config_file config_files/block_structure/runs_sim_vowels.json --dry_run
python scripts/post-training/block_structure.py --config_file config_files/block_structure/runs_sim_vowels.json --max_batches 2
python scripts/post-training/block_structure.py --config_file config_files/block_structure/runs_sim_vowels.json
python scripts/post-training/block_structure.py --config_file config_files/block_structure/runs_timit.json --dry_run
python scripts/post-training/block_structure.py --config_file config_files/block_structure/runs_timit.json --max_batches 2
python scripts/post-training/block_structure.py --config_file config_files/block_structure/runs_timit.json
Rscript visualize_R/SI_block_structure/block_structure.R   # from visualize_R/, as beta_tradeoff.R
```

## Checkpoints

As `config_files/beta_tradeoff/runs_*.json` (the paper's post-training evaluation):

| dataset | runs | checkpoint |
|---|---|---|
| SimVowels | FD, EWT x beta 0, 0.1, 1; splits validation, test | -2 (FD: epoch_111_min_val_loss, 109, 109; EWT: 112_min_val_loss, 109, 110_min_val_loss in the beta trade-off CSV) |
| SimVowels | FD untrained control (same config, seed 0, no weights) | - |
| TIMIT | FD, EWT x beta 0, 0.1; split test | -1 (epoch_99) |

## Outputs

- `block_structure_<dataset>.csv`: long format `dataset, model, beta, untrained, run_id, checkpoint, branch, split, measure, comp_i, comp_j, value`.
  Measures: `js_pos(_sd)`, `js_neg(_sd)`, `cos_ps`, `js_pos_mean`, `js_neg_mean`, `cos_ps_mean`, `js_mix`, `cos_ds`, `js_ds`,
  `unit_count`, `purity`, `captured_mass(_p05)`, `C_mass`, `C_oracle`, `C_null`, `C_null_p95`, `CI`, references `j_C`, `ln2`,
  `n_samples`, `H`, and for the S branch `check_max_abs_diff` (cross-check against the model's divergences, tolerance 1e-4).
- `mean_distributions_<run_id>_<branch>_<split>.csv`: R_bar_1..C and P_bar over the H hidden units.

Conventions: float64, softmax over hidden units, JS in nats with the eps (1e-8) of `divergence_utils.js_rows`; under it
the JS of identical rows is about H * eps (~2e-6 for H = 192), not exactly 0, and the disjoint maximum is ln 2 + O(1e-4).
Null: 1000 random partitions with the sizes |U_i|, seed 0. Descriptive thresholds (not pass/fail): cos <= 0.05
"near orthogonal", CI >= 0.8 "consistent partition".

## Main values

To be filled from the CSVs after the runs (test split; per run and branch):

| dataset | model | beta | branch | js_pos | js_neg | js_mix | cos_ps | cos_ds | CI | C_null p95 |
|---|---|---|---|---|---|---|---|---|---|---|
