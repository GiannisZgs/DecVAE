"""Config helpers for the bash sweeps (scripts/sweeps/). One subcommand per job:

  ckpdir <pretraining_config> <parent_dir>      checkpoint folder latents_post_analysis.py builds from parent_dir
  ckpsel <checkpoint_dir> <index>               checkpoint that epoch_range_to_evaluate = [index] selects
  loss_weights <config>                         set the decomposition loss weights for the config's NoC
  sync <pretraining_config> <eval_config>       copy the model, decomposition and objective keys of the
                                                pretraining config into an evaluation config
  compare <config> <saved_config>               print the settings that differ from a trained model's saved config
  boundaries --low --high --n --seed [--min_width]
                                                random peak-search boundaries, printed as compact JSON
  caches <config> <tag>                         rename every *_cache_file_name of a config for this run:
                                                ..._<decomp>_NoC3_train_set.arrow -> ..._<decomp>_<tag>_train_set.arrow
  register <models_file> <name> <dataset> <group> <pretraining_config> <dump> [--seed N] [--headline]
                                                add or replace one entry of a subspace-analysis models file
"""
import argparse
import json
import os
import re
import numpy as np

"Keys that belong to the evaluation run and are never copied from the pretraining config"
EVAL_OWN_KEYS = {"output_dir", "data_dir", "cache_dir", "max_frames_per_batch", "per_device_eval_batch_size",
                 "wandb_project", "wandb_group", "with_wandb", "seed"}
"Keys that may differ between two runs of the same configuration"
RUN_KEYS = {"output_dir", "data_dir", "cache_dir", "seed", "with_wandb", "wandb_project", "wandb_group", "push_to_hub",
            "hub_model_id", "hub_token", "logging_steps", "saving_steps", "preprocessing_num_workers",
            "train_cache_file_name", "validation_cache_file_name", "test_cache_file_name", "dev_cache_file_name",
            "indep_cache_file_name"}
DECOMPOSITIONS = ("filter", "ewt", "emd", "vmd")


def load(path):
    with open(path) as f:
        return json.load(f)


def save(cfg, path):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def beta_tag(beta):
    "Same formatting as the checkpoint directory in latents_post_analysis.py"
    return "01" if beta == 0.1 else str(int(beta))


def leaf(cfg):
    "Checkpoint folder of a SimVowels run, as latents_post_analysis.py rebuilds it from the config"
    if cfg["dual_branched_latent"]:
        model_type, betas = "dual", f"_bz{beta_tag(cfg['beta_kl_prior_z'])}_bs{beta_tag(cfg['beta_kl_prior_s'])}"
    elif cfg["only_z_branch"]:
        model_type, betas = "single_z", f"_bz{beta_tag(cfg['beta_kl_prior_z'])}"
    elif cfg["only_s_branch"]:
        model_type, betas = "single_s", f"_bs{beta_tag(cfg['beta_kl_prior_s'])}"
    else:
        raise ValueError("one of dual_branched_latent, only_z_branch, only_s_branch must be true")
    return (f"snr{cfg['sim_snr_db']}{betas}_NoC{cfg['NoC']}_{cfg['input_type']}_{model_type}"
            f"-bs{cfg['per_device_train_batch_size']}")


def ckpdir(cfg, parent):
    "SimCoupled checkpoints sit directly in parent_dir, SimVowels ones in parent_dir/<leaf>"
    if cfg["dataset_name"] == "sim_coupled":
        return parent
    if "vowels" in cfg["dataset_name"]:
        return f"{parent}/{leaf(cfg)}"
    raise ValueError(f"no checkpoint folder rule for {cfg['dataset_name']}")


def ckpsel(ckp_dir, index):
    "Checkpoint epoch_range_to_evaluate = [index] selects, as latents_post_analysis.py resolves it"
    files = [f for f in os.listdir(ckp_dir) if 'config' not in f]
    files.append('epoch_-01')
    files.sort(key=lambda f: int(m.group(1)) if (m := re.search(r'epoch_(\d+)', f)) else -1)
    return files[index]


def loss_weights(path):
    """Decomposition loss weights for the config's NoC, truncated to 4 decimals as in the existing configs.

    Positive (X-OC) terms: 1/NoC each, in div_pos_weight and in weight_0_1, weight_0_2, weight_0_3_and_above,
    which replace div_pos_weight for the JS divergence. Negative (OC-OC) terms: 1/(number of OC pairs).
    """
    cfg = load(path)
    C = int(cfg["NoC"])
    trunc = lambda x: np.floor(x * 1e4 + 1e-9) / 1e4
    pos, neg = trunc(1 / C), trunc(1 / (C * (C - 1) / 2))
    for k in ("div_pos_weight", "weight_0_1", "weight_0_2", "weight_0_3_and_above"):
        cfg[k] = float(pos)
    cfg["div_neg_weight"] = float(neg)
    save(cfg, path)
    print(f"  loss weights for NoC {C}: positive {pos}, negative {neg}")


def sync(pre_path, eval_path):
    "Every key the two configs share is set to the pretraining value, except the evaluation's own keys"
    pre, ev = load(pre_path), load(eval_path)
    changed = []
    for k in sorted(set(pre) & set(ev) - EVAL_OWN_KEYS):
        if k.startswith("comment"):
            continue
        if ev[k] != pre[k]:
            changed.append(f"{k}: {ev[k]} -> {pre[k]}")
            ev[k] = pre[k]
    "Keys the pretraining config has and the evaluation template lacks (e.g. detection_boundaries)"
    for k in ("detection_boundaries", "spec_amp_tolerance"):
        if k in pre and k not in ev:
            changed.append(f"{k}: (absent) -> {pre[k]}")
            ev[k] = pre[k]
    save(ev, eval_path)
    for c in changed:
        print(f"  sync {os.path.basename(eval_path)}: {c}")


def compare(path, saved_path):
    "Shared settings of a config that differ from a trained model's saved config, run-specific keys left out"
    cfg, saved = load(path), load(saved_path)
    diffs = []
    for k in sorted(set(cfg) & set(saved) - RUN_KEYS):
        if k.startswith("comment"):
            continue
        if cfg[k] != saved[k]:
            diffs.append(f"{k}: sweep {json.dumps(cfg[k])}, trained model {json.dumps(saved[k])}")
    for d in diffs:
        print(f"  differs from {saved_path}: {d}")
    return diffs


def caches(path, tag):
    "Each run gets its own decomposed-data cache, otherwise a cached decomposition of another run is loaded"
    cfg = load(path)
    decomp = cfg["decomp_to_perform"]
    keys = [k for k in cfg if k.endswith("_cache_file_name") and cfg[k]]
    for k in keys:
        name = os.path.basename(cfg[k])
        "The decomposition in the name follows the config, e.g. a SimCoupled template written for EWT run with FD"
        new, n = re.subn(rf"_(?:{'|'.join(DECOMPOSITIONS)})_NoC\d+_", f"_{decomp}_{tag}_", name)
        if n == 0:
            new, n = re.subn(r"_NoC\d+_", f"_{tag}_", name)
        if n != 1:
            raise ValueError(f"{k}: cannot find _NoC<C>_ in {cfg[k]}")
        cfg[k] = os.path.join(os.path.dirname(cfg[k]), new).replace("\\", "/")
    if not keys:
        raise ValueError(f"no cache file names in {path}")
    save(cfg, path)


def random_boundaries(low, high, n, seed, min_width):
    """n search intervals between low and high, interior boundaries uniform and sorted.

    Draws are rejected until every interval is at least min_width wide, so each interval can hold a
    peak_bandwidth-wide passband.
    """
    if n * min_width > high - low:
        raise ValueError("min_width too large for this band and number of intervals")
    rng = np.random.default_rng(seed)
    while True:
        b = np.concatenate(([low], np.sort(rng.uniform(low, high, n - 1)), [high]))
        if np.diff(b).min() >= min_width:
            return [round(float(x), 1) for x in b]


def register(models_file, name, dataset, group, pre_path, dump, seed=None, headline=False):
    "Add or replace one model entry, in the format scripts/post-training/subspace_analysis.py reads"
    pre = load(pre_path)
    models = load(models_file) if os.path.exists(models_file) else []
    entry = {"name": name, "label": f"DecVAE + FD ({group})", "dataset": dataset, "config": group,
             "decomposition": "fd", "beta": pre["beta_kl_prior_z"], "seed": int(pre["seed"] if seed is None else seed),
             "n_blocks": int(pre["NoC"]) + 1, "oc_order": "ascending", "headline": bool(headline), "dump": dump}
    models = [m for m in models if m["name"] != name] + [entry]
    os.makedirs(os.path.dirname(models_file) or ".", exist_ok=True)
    with open(models_file, "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join("    " + json.dumps(m, ensure_ascii=False) for m in models) + "\n]\n")
    print(f"  registered {name} in {models_file} ({len(models)} entries)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ckpdir"); s.add_argument("config"); s.add_argument("parent")
    s = sub.add_parser("ckpsel"); s.add_argument("ckp_dir"); s.add_argument("index", type=int)
    s = sub.add_parser("loss_weights"); s.add_argument("config")
    s = sub.add_parser("sync"); s.add_argument("pre"); s.add_argument("eval")
    s = sub.add_parser("compare"); s.add_argument("config"); s.add_argument("saved")
    s = sub.add_parser("boundaries")
    s.add_argument("--low", type=float, required=True); s.add_argument("--high", type=float, required=True)
    s.add_argument("--n", type=int, required=True); s.add_argument("--seed", type=int, required=True)
    s.add_argument("--min_width", type=float, default=300.0)
    s = sub.add_parser("register")
    for a in ("models_file", "name", "dataset", "group", "pre", "dump"):
        s.add_argument(a)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--headline", action="store_true", help="reference configuration of the sweep (C = 3)")
    s = sub.add_parser("caches"); s.add_argument("config"); s.add_argument("tag")
    a = p.parse_args()
    if a.cmd == "ckpdir":
        print(ckpdir(load(a.config), a.parent))
    elif a.cmd == "ckpsel":
        print(ckpsel(a.ckp_dir, a.index))
    elif a.cmd == "loss_weights":
        loss_weights(a.config)
    elif a.cmd == "sync":
        sync(a.pre, a.eval)
    elif a.cmd == "compare":
        compare(a.config, a.saved)
    elif a.cmd == "boundaries":
        print(json.dumps(random_boundaries(a.low, a.high, a.n, a.seed, a.min_width), separators=(",", ":")))
    elif a.cmd == "register":
        register(a.models_file, a.name, a.dataset, a.group, a.pre, a.dump, a.seed, a.headline)
    elif a.cmd == "caches":
        caches(a.config, a.tag)


if __name__ == "__main__":
    main()
