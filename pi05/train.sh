#!/usr/bin/env bash
# Fine-tune pi0.5 (lerobot/pi05_base) on a prepared open-box-flaps dataset (default: thor_v3 with
# three cameras, see env.sh; the model uses every camera of the dataset). If the dataset's YAML sets
# flap_balance_exponent, training draws the flaps in those shares (scripts/train_with_heldout.py).
# The dataset's held-out episodes (meta/open_box_split.json) are kept out of training. A dataset
# built with heldout_per_subset: 0 has none: every episode is trained on and the held-out loss is off.
#
#   bash pi05/train.sh                          # full run with the defaults below
#   STEPS=10000 bash pi05/train.sh              # override any setting through the environment
#   bash pi05/train.sh --policy.use_relative_actions=true   # extra lerobot-train flags are appended
#   DRY_RUN=1 bash pi05/train.sh                # run the checks and print the command, don't train
#
# Checkpoints:  $CHECKPOINT_ROOT/pi05/<RUN_NAME>/checkpoints/<step>/{pretrained_model,training_state}
#               (default /media/ai/aipc/Karel/Mocap/checkpoints/pi05/...)
# Log:          $CHECKPOINT_ROOT/pi05/logs/<RUN_NAME>.log
# Resume:       bash pi05/resume.sh <RUN_NAME>
# Curves:       uv run trackio show --project pi05_open_box   (train/loss, and eval/eval_loss = the same
#               loss on the held-out episodes every EVAL_STEPS steps; see scripts/train_with_heldout.py)
#
# Settings (environment): RUN_NAME, STEPS, BATCH_SIZE, SAVE_FREQ, EVAL_STEPS, HELDOUT_STRIDE, NUM_WORKERS,
#   PREFETCH_FACTOR, PRETRAINED_POLICY, OUTPUT_DIR, DATASET_REPO_ID, SKIP_TOKENIZER_CHECK=1, SKIP_DISK_CHECK=1,
#   DRY_RUN=1
set -euo pipefail
cd "$(dirname "$0")/.."
source ./env.sh

SPLIT_FILE="$DATASET_ROOT/meta/open_box_split.json"
[[ -f "$SPLIT_FILE" ]] || { echo "Missing $SPLIT_FILE - run: uv run python scripts/prepare_dataset.py datasets/<name>.yaml"; exit 1; }
VAL_EPISODES=$(python3 -c "import json;print(json.load(open('$SPLIT_FILE'))['val'])" | tr -d ' ')

RUN_NAME="${RUN_NAME:-pi05_$(basename "$DATASET_REPO_ID")_$(date +%Y%m%d_%H%M)}"   # e.g. pi05_open_box_thor_right6_20261122_1300
PI05_ROOT="$CHECKPOINT_ROOT/pi05"
OUTPUT_DIR="${OUTPUT_DIR:-$PI05_ROOT/$RUN_NAME}"
LOG_FILE="$PI05_ROOT/logs/$RUN_NAME.log"
STEPS="${STEPS:-30000}"          # 2 cameras: ~3 s/step at batch 32 on the RTX PRO 6000 -> 30k steps is ~25 h
BATCH_SIZE="${BATCH_SIZE:-32}"   # 2 cameras: ~45 GB of the 96 GB GPU; a third camera adds 256 image tokens per sample
# A checkpoint is saved every SAVE_FREQ steps and at the last step; lerobot keeps all of them.
SAVE_FREQ="${SAVE_FREQ:-10000}"
# Held-out loss (eval/eval_loss in Trackio) every EVAL_STEPS steps, on every HELDOUT_STRIDE-th frame
# of the held-out episodes (~1 min each); 0 = off.
EVAL_STEPS="${EVAL_STEPS:-1000}"
export HELDOUT_STRIDE="${HELDOUT_STRIDE:-5}"
EXCLUDE_FLAGS=(--dataset.exclude_episodes="$VAL_EPISODES")
if [[ "$VAL_EPISODES" == "[]" ]]; then   # no held-out episodes: nothing to exclude or evaluate on
  if (( EVAL_STEPS > 0 )); then echo "no held-out episodes in this dataset: held-out loss off"; fi
  EVAL_STEPS=0
  EXCLUDE_FLAGS=()
fi
# 4 workers already feed the GPU faster than it trains; 8 used ~30 GB of host RAM.
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
TRACKIO_PROJECT="${TRACKIO_PROJECT:-pi05_open_box}"
GB_PER_CHECKPOINT=24             # bf16 weights (8.8 GB) + AdamW state (15 GB), measured

[[ -e "$OUTPUT_DIR" ]] && { echo "$OUTPUT_DIR already exists. Pick another RUN_NAME, or continue it with: bash pi05/resume.sh $RUN_NAME"; exit 1; }
mkdir -p "$PI05_ROOT/logs"

# --- disk space: a run that cannot write its checkpoint dies at save time, after hours of training
N_CKPT=$(( (STEPS + SAVE_FREQ - 1) / SAVE_FREQ ))
NEED_GB=$(( N_CKPT * GB_PER_CHECKPOINT ))
FREE_GB=$(df --output=avail -BG "$PI05_ROOT" | tail -1 | tr -dc '0-9')
echo "run=$RUN_NAME  steps=$STEPS  batch=$BATCH_SIZE  dataset=$DATASET_REPO_ID  held-out episodes=$([[ $VAL_EPISODES == "[]" ]] && echo "none (all episodes train)" || echo "$VAL_EPISODES")"
echo "checkpoints: $N_CKPT x ~${GB_PER_CHECKPOINT} GB = ~${NEED_GB} GB needed, ${FREE_GB} GB free on $(df --output=target "$PI05_ROOT" | tail -1)"
if (( FREE_GB < NEED_GB )) && [[ "${SKIP_DISK_CHECK:-0}" != 1 ]]; then
  echo "Not enough space. Free some, raise SAVE_FREQ, or set SKIP_DISK_CHECK=1."; exit 1
fi

# --- tokenizer: pi0.5 uses the gated PaliGemma tokenizer (accept the license on HF + `uv run hf auth login`)
if [[ "${SKIP_TOKENIZER_CHECK:-0}" != 1 ]]; then
  uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')" >/dev/null 2>&1 || {
    echo "Cannot load the PaliGemma tokenizer (google/paligemma-3b-pt-224). Fix:"
    echo "  1. accept the license at https://huggingface.co/google/paligemma-3b-pt-224"
    echo "  2. uv run hf auth login     (token is stored in ~/.cache/huggingface, not on the shared drive)"
    exit 1
  }
  echo "tokenizer: ok"
fi

EVAL_FLAGS=()
if (( EVAL_STEPS > 0 )); then
  # lerobot-train needs eval_split > 0 to accept eval_steps; train_with_heldout.py ignores its value
  # and uses the held-out episodes of meta/open_box_split.json instead.
  EVAL_FLAGS=(--eval_steps="$EVAL_STEPS" --dataset.eval_split=0.1)
fi

CMD=(uv run python scripts/train_with_heldout.py   # = lerobot-train + the held-out loss curve
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  "${EXCLUDE_FLAGS[@]}"
  --policy.type=pi05
  --policy.pretrained_path="$PRETRAINED_POLICY"
  --policy.dtype=bfloat16
  --policy.gradient_checkpointing=true
  --policy.compile_model=false
  --policy.freeze_vision_encoder=false
  --policy.train_expert_only=false
  --policy.device=cuda
  --policy.push_to_hub=false
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --save_freq="$SAVE_FREQ"
  --log_freq=100
  --num_workers="$NUM_WORKERS"
  --prefetch_factor="$PREFETCH_FACTOR"
  --output_dir="$OUTPUT_DIR"
  --job_name="$RUN_NAME"
  --trackio.enable=true                       # live curves: uv run trackio show --project $TRACKIO_PROJECT
  --trackio.project="$TRACKIO_PROJECT"
  --trackio.run_name="$RUN_NAME"
  "${EVAL_FLAGS[@]}"
  "$@")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${CMD[@]}"; echo; exit 0
fi
echo "output: $OUTPUT_DIR"
echo "log:    $LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
