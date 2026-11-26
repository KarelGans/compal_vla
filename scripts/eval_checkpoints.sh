#!/usr/bin/env bash
# Score every saved checkpoint of a run, add the scores to the run's Trackio dashboard, then print
# a table to pick the checkpoint to try on the robot (scripts/compare_checkpoints.py).
#
#   bash scripts/eval_checkpoints.sh pi05/pi05_open_box_20261114_1239
#   bash scripts/eval_checkpoints.sh act/act_baseline
#
# Per checkpoint: the held-out episodes with the wrong-prompt test, then one training episode per
# recording folder (the "train probe" in the dataset's meta/open_box_split.json); without held-out
# episodes (heldout_per_subset: 0), only the training episodes, with the wrong-prompt test. The dataset is the
# one the checkpoint was trained on. Checkpoints that already have both results are skipped, so it
# can be re-run after more training.
# Results go next to each checkpoint (eval_open_loop*/); the full output goes to
# $CHECKPOINT_ROOT/<model>/logs/eval_<run>.log. Runs on the GPU; pi0.5 needs ~12 GB of free RAM to load.
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh

RUN="${1:?usage: bash scripts/eval_checkpoints.sh <model>/<run>}"
RUN_DIR="$CHECKPOINT_ROOT/$RUN"
[ -d "$RUN_DIR/checkpoints" ] || { echo "no checkpoints folder in $RUN_DIR"; exit 1; }
LOG="$(dirname "$RUN_DIR")/logs/eval_$(basename "$RUN_DIR").log"
mkdir -p "$(dirname "$LOG")"

# On a shared machine: if RAM runs out, the kernel stops this evaluation instead of someone's training.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true

FIRST=$(ls -d "$RUN_DIR"/checkpoints/[0-9]*/pretrained_model 2>/dev/null | head -1)
[ -n "$FIRST" ] || { echo "no saved checkpoints in $RUN_DIR/checkpoints"; exit 1; }
# A dataset built with heldout_per_subset: 0 has no held-out episodes: the training episodes then
# get the wrong-prompt test instead.
N_HELD=$(uv run python -c "import sys; from open_box_vla.dataset_specs import checkpoint_dataset, load_split; print(len(load_split(checkpoint_dataset(sys.argv[1])[1])['val']))" "$FIRST")
TRAIN_FLAGS=(--train-probe); TRAIN_DONE='"split"'
if [ "$N_HELD" = 0 ]; then TRAIN_FLAGS+=(--prompt-swap); TRAIN_DONE='"prompt_swap"'; fi

echo "evaluating $RUN (full output in $LOG)"
for CKPT in "$RUN_DIR"/checkpoints/[0-9]*/pretrained_model; do
  STEP=$(basename "$(dirname "$CKPT")")
  HELD="$(dirname "$CKPT")/eval_open_loop/summary.json"
  TRAIN="$(dirname "$CKPT")/eval_open_loop_train_episodes/summary.json"
  if [ "$N_HELD" = 0 ]; then
    echo "  $STEP held-out: none in this dataset"
  elif grep -q '"prompt_swap"' "$HELD" 2>/dev/null; then
    echo "  $STEP held-out: already scored"
  else
    echo "  $STEP held-out episodes   $(date +%H:%M)"
    uv run python scripts/eval_open_loop.py --checkpoint "$CKPT" --prompt-swap --trackio >>"$LOG" 2>&1 \
      || echo "    FAILED, see $LOG"
  fi
  if grep -q "$TRAIN_DONE" "$TRAIN" 2>/dev/null; then
    echo "  $STEP training episodes: already scored"
  else
    echo "  $STEP training episodes   $(date +%H:%M)"
    uv run python scripts/eval_open_loop.py --checkpoint "$CKPT" "${TRAIN_FLAGS[@]}" --trackio >>"$LOG" 2>&1 \
      || echo "    FAILED, see $LOG"
  fi
done
uv run python scripts/compare_checkpoints.py "$RUN" 2>&1 | tee -a "$LOG"
