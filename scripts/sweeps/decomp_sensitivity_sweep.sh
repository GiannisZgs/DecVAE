#!/usr/bin/env bash
# Decomposition sensitivity sweep: number of components C and FD peak-search intervals
# (reviewer comments R3 #2, R4 #2, R5 #8). For each run: write its configs, pretrain, evaluate
# (standard metric suite), dump the latents for the subspace analysis, and register the dump.
# Finished steps leave a marker and are skipped, so the script can be restarted after an interruption.
#
#   bash scripts/sweeps/decomp_sensitivity_sweep.sh sim_vowels             # every run of the dataset
#   bash scripts/sweeps/decomp_sensitivity_sweep.sh sim_vowels C5 rand0    # only these runs
#   DRY_RUN=1 bash scripts/sweeps/decomp_sensitivity_sweep.sh sim_vowels   # write configs, print commands
#
# Run from the repository root.
set -euo pipefail

DATASET=${1:?"usage: $0 sim_vowels|sim_coupled [RUN ...]"}
shift
SELECTED=("$@")
DRY_RUN=${DRY_RUN:-0}
LAUNCH=${LAUNCH:-"accelerate launch"}
PY=${PY:-python}
UPD="$PY utils/update_config.py"
HELP="$PY utils/sweep_helpers.py"

PRETRAIN_SCRIPT=scripts/pre-training/base_models_ssl_pretraining.py
EVAL_SCRIPT=scripts/post-training/latents_post_analysis.py
RUN_CFG_ROOT=config_files/decomp_sensitivity/runs/$DATASET
MODELS_FILE=config_files/decomp_sensitivity/models_decomp_sensitivity.json
RESULTS=../post-training_results/decomp_sensitivity
DUMP_DIR=$RESULTS/eval_dumps
LOG_DIR=$RESULTS/logs/$DATASET

declare -A RUNS EXISTING
case $DATASET in
  sim_vowels)
    # Templates: the FD, beta = 0.1 seed config of the subspace plan, and its dump config
    PRE_TPL=config_files/subspace/pre-training/config_pretraining_sim_vowels_NoC3_fd_b01_s1.json
    EVAL_TPL=config_files/subspace/dumps/config_dump_decvae_fd_b01_s1_sim_vowels.json
    MODEL_ROOT=../pretrained_models/sim_vowels/filter/decomp_sensitivity
    SUBSET_VAR=""
    TPL_OVERRIDES=""
    EVAL_EPOCHS=${EVAL_EPOCHS:-"[-2]"}   # checkpoint evaluated and dumped, as in the SimVowels subspace dumps
    DUMP_TARGETS=vowel_speaker_frame
    ORDER=(C3_s0 C3_s1 C3_s2 C4 C5 C6 C2 lin int3 int8 rand0 rand1 rand2)
    RUNS=(
      [C3_s0]="seed=0"
      [C3_s1]="seed=1"
      [C3_s2]="seed=2"
      [C4]="NoC=4 NoC_seq=4"
      [C2]="NoC=2 NoC_seq=2"
      [C5]="NoC=5 NoC_seq=5"
      [C6]="NoC=6 NoC_seq=6"
      [lin]="power_law=1.0"
      [int3]="detection_intervals=3"
      [int8]="detection_intervals=8"
      [rand0]="detection_boundaries=@random:0"
      [rand1]="detection_boundaries=@random:1"
      [rand2]="detection_boundaries=@random:2"
    )
    # Runs that are already trained: "<parent_dir>|<cache tag>". Pretraining is skipped, the checkpoint is
    # expected in <parent_dir>/<leaf>, and the decomposed data in the template's cache folder under <cache tag>
    # (NoC3 -> vowels_filter_NoC3_*_set.arrow). Add C2 here if an FD, beta = 0.1, C = 2 model already exists.
    EXISTING=(
      [C3_s0]="../pretrained_models/sim_vowels/filter/decvae_filter_b01|NoC3"
      [C3_s1]="../pretrained_models/sim_vowels/filter/decvae_filter_b01_s1|NoC3"
      [C3_s2]="../pretrained_models/sim_vowels/filter/decvae_filter_b01_s2|NoC3"
      [C4]="SET: parent folder of the existing FD, beta = 0.1, C = 4 model|SET: its cache tag"
    )
    ;;
  sim_coupled)
    # The pretraining template is written for EWT; the sweep runs it with FD (TPL_OVERRIDES)
    PRE_TPL=config_files/DecVAEs/sim_coupled/pre-training/config_pretraining_sim_coupled_NoC3.json
    EVAL_TPL=config_files/DecVAEs/sim_coupled/latent_evaluations/config_latent_anal_sim_coupled.json
    MODEL_ROOT=../pretrained_models/sim_coupled/filter/decomp_sensitivity
    SUBSET_VAR=SIM_COUPLED_SUBSET_FRACTION
    TPL_OVERRIDES="decomp_to_perform=filter"
    EVAL_EPOCHS=${EVAL_EPOCHS:-"[-1]"}   # checkpoint evaluated and dumped, as in the SimCoupled subspace dumps
    DUMP_TARGETS=lag_gain
    ORDER=(C3_s0 C3_s1 C3_s2 C2 C4 C5 C6)
    RUNS=(
      [C3_s0]="seed=0"
      [C3_s1]="seed=1"
      [C3_s2]="seed=2"
      [C2]="NoC=2"
      [C4]="NoC=4"
      [C5]="NoC=5"
      [C6]="NoC=6"
    )
    # The Experiment B FD models (subspace dumps decvae_fd_b01_s<seed>_sim_coupled); checkpoints sit in <parent_dir>
    EXISTING=(
      [C3_s0]="../pretrained_models/sim_coupled/filter/decvae_filter_NoC3_seed0|NoC3"
      [C3_s1]="../pretrained_models/sim_coupled/filter/decvae_filter_NoC3_seed1|NoC3"
      [C3_s2]="../pretrained_models/sim_coupled/filter/decvae_filter_NoC3_seed2|NoC3"
    )
    ;;
  *) echo "unknown dataset $DATASET" >&2; exit 1 ;;
esac

die() { echo "ERROR: $*" >&2; exit 1; }
say() { echo "[$(date '+%F %T')] $*"; }
run() {
  # Run a command, or only print it in a dry run
  if [[ $DRY_RUN == 1 ]]; then echo "  DRY: $*"; else "$@"; fi
}
get() { $PY -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))[sys.argv[2]]))" "$1" "$2"; }

# --- Pre-flight checks ---
[[ -f $PRETRAIN_SCRIPT && -f $EVAL_SCRIPT ]] || die "run from the repository root"
[[ -f $PRE_TPL && -f $EVAL_TPL ]] || die "missing template $PRE_TPL or $EVAL_TPL"
grep -q "json.loads" utils/update_config.py || die "utils/update_config.py must parse JSON values (lists, null); see plan step 0"
if [[ -n $SUBSET_VAR ]]; then
  grep -Eq "^${SUBSET_VAR} = None" "$PRETRAIN_SCRIPT" || die "$SUBSET_VAR in $PRETRAIN_SCRIPT must be None for the sweep"
fi
[[ ${#SELECTED[@]} -gt 0 ]] && TAGS=("${SELECTED[@]}") || TAGS=("${ORDER[@]}")
for TAG in "${TAGS[@]}"; do
  [[ -n ${RUNS[$TAG]+x} ]] || die "unknown run $TAG for $DATASET"
  [[ -n ${EXISTING[$TAG]+x} && ${EXISTING[$TAG]} == *SET:* ]] && die "set the folder and cache tag of existing run $TAG (EXISTING[$TAG])"
  [[ ${RUNS[$TAG]} == *detection_boundaries* ]] && \
    { grep -q detection_boundaries args_configs/decomposition_args.py || die "detection_boundaries is not an argument yet (plan section 4)"; }
done
mkdir -p "$RUN_CFG_ROOT" "$DUMP_DIR" "$LOG_DIR"

for TAG in "${TAGS[@]}"; do
  say "=== $DATASET $TAG ==="
  RUN_DIR=$RUN_CFG_ROOT/$TAG
  # Named config_*: the pretraining copies its config into the checkpoint folder, and the evaluation
  # takes every entry of that folder without "config" in its name for a checkpoint
  PRE=$RUN_DIR/config_pretraining.json
  EVAL=$RUN_DIR/config_eval.json
  DUMP=$RUN_DIR/config_dump.json
  GROUP=${TAG%_s[0-9]*}
  [[ $GROUP == rand[0-9]* ]] && GROUP=rand
  SEED_ARG=()
  [[ $TAG == rand[0-9]* ]] && SEED_ARG=(--seed "${TAG#rand}")
  [[ $GROUP == C3 ]] && SEED_ARG+=(--headline)
  mkdir -p "$RUN_DIR"

  # --- Pretraining config: template plus this run's overrides ---
  cp "$PRE_TPL" "$PRE"
  $UPD "$PRE" seed 0                     # new runs use seed 0; the C3 seed runs override it
  for KV in $TPL_OVERRIDES ${RUNS[$TAG]}; do
    KEY=${KV%%=*}
    VAL=${KV#*=}
    if [[ $VAL == @random:* ]]; then
      VAL=$($HELP boundaries --low "$(get "$PRE" lower_speech_freq)" --high "$(get "$PRE" higher_speech_freq)" \
                             --n "$(get "$PRE" detection_intervals)" --seed "${VAL#@random:}")
    fi
    $UPD "$PRE" "$KEY" "$VAL"
  done
  $UPD "$PRE" with_wandb false
  if [[ -n ${EXISTING[$TAG]+x} ]]; then
    PARENT=${EXISTING[$TAG]%%|*}
    $HELP caches "$PRE" "${EXISTING[$TAG]#*|}"
    CKP=$($HELP ckpdir "$PRE" "$PARENT")
    if [[ $DRY_RUN != 1 ]]; then
      ls -d "$CKP"/training_ckp_epoch_* > /dev/null 2>&1 || die "existing run $TAG: no checkpoint in $CKP"
    fi
    # The trained model's own config, to see whether the sweep template matches how it was trained
    SAVED=$(ls "$CKP"/config*.json 2>/dev/null | head -n 1 || true)
    if [[ -n $SAVED ]]; then
      $HELP compare "$PRE" "$SAVED" | tee "$LOG_DIR/${TAG}_config_diff.log"
    else
      say "no saved config in $CKP to compare the template with"
    fi
    touch "$RUN_DIR/.pretrain_done"
  else
    PARENT=$MODEL_ROOT/$TAG
    $HELP caches "$PRE" "$TAG"
    CKP=$($HELP ckpdir "$PRE" "$PARENT")
  fi
  $UPD "$PRE" output_dir "$CKP"

  # --- Evaluation and dump configs: decomposition, model and objective keys follow the pretraining config ---
  cp "$EVAL_TPL" "$EVAL"
  $HELP sync "$PRE" "$EVAL"
  $UPD "$EVAL" parent_dir "$PARENT"
  $UPD "$EVAL" output_dir "$RESULTS/$DATASET/$TAG"
  $UPD "$EVAL" epoch_range_to_evaluate "$EVAL_EPOCHS"
  $UPD "$EVAL" measure_disentanglement true
  $UPD "$EVAL" classify false
  $UPD "$EVAL" with_wandb false
  $UPD "$EVAL" eval_dump_only false
  cp "$EVAL" "$DUMP"
  $UPD "$DUMP" output_dir "$RESULTS/$DATASET/$TAG/dump_run"
  $UPD "$DUMP" eval_dump_only true
  $UPD "$DUMP" eval_dump_dir "$DUMP_DIR"
  $UPD "$DUMP" eval_dump_tag "${DATASET}_${TAG}"

  # --- Pretrain ---
  if [[ -f $RUN_DIR/.pretrain_done ]]; then
    say "pretraining done, skipped"
  else
    say "pretraining -> $CKP"
    run $LAUNCH "$PRETRAIN_SCRIPT" --config_file "$PRE" 2>&1 | tee "$LOG_DIR/${TAG}_pretrain.log"
    if [[ $DRY_RUN != 1 ]]; then
      ls -d "$CKP"/training_ckp_epoch_* > /dev/null 2>&1 || die "no checkpoint in $CKP"
      touch "$RUN_DIR/.pretrain_done"
    fi
  fi

  # The selected entry must be a trained checkpoint, not the random-init "epoch_-01" the evaluation adds
  if [[ $DRY_RUN != 1 ]]; then
    SEL=$($HELP ckpsel "$CKP" "$($PY -c "import json,sys; print(json.loads(sys.argv[1])[0])" "$EVAL_EPOCHS")")
    [[ $SEL == training_ckp_epoch_* ]] || die "epoch_range_to_evaluate $EVAL_EPOCHS selects $SEL in $CKP, not a trained checkpoint"
    say "evaluating $CKP/$SEL"
  fi

  # --- Standard evaluation ---
  if [[ -f $RUN_DIR/.eval_done ]]; then
    say "evaluation done, skipped"
  else
    run $LAUNCH "$EVAL_SCRIPT" --config_file "$EVAL" 2>&1 | tee "$LOG_DIR/${TAG}_eval.log"
    [[ $DRY_RUN == 1 ]] || touch "$RUN_DIR/.eval_done"
  fi

  # --- Latent dump of the frame latent ("all") for the subspace analysis ---
  DUMP_FILE=$DUMP_DIR/${DATASET}_${TAG}_all_${DUMP_TARGETS}.npz
  if [[ -f $DUMP_FILE ]]; then
    say "dump exists, skipped"
  else
    run $LAUNCH "$EVAL_SCRIPT" --config_file "$DUMP" 2>&1 | tee "$LOG_DIR/${TAG}_dump.log"
  fi
  if [[ $DRY_RUN != 1 ]]; then
    [[ -f $DUMP_FILE ]] || die "expected the dump $DUMP_FILE"
    $HELP register "$MODELS_FILE" "${DATASET}_${TAG}" "$DATASET" "$GROUP" "$PRE" "$DUMP_FILE" ${SEED_ARG[@]+"${SEED_ARG[@]}"}
  fi
done
say "sweep finished: $DATASET ${TAGS[*]}"
