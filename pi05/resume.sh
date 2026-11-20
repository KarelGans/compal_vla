#!/usr/bin/env bash
# Continue an interrupted pi0.5 run from its last checkpoint (weights, optimizer, step count,
# data-loader position). Settings come from the checkpoint's train_config.json.
#
#   bash pi05/resume.sh <RUN_NAME>                  # e.g. pi05_open_box_20261114_1300
#   bash pi05/resume.sh <RUN_NAME> --steps=40000    # resume and train for longer
#
# Note: only steps up to the last saved checkpoint are kept. With SAVE_FREQ=10000, a crash at
# step 19000 resumes from 10000.
set -euo pipefail
cd "$(dirname "$0")/.."
source ./env.sh

RUN="${1:?usage: bash pi05/resume.sh <RUN_NAME or run dir> [extra lerobot-train flags]}"; shift
if [[ -d "$RUN" ]]; then RUN_DIR="$RUN"; else RUN_DIR="$CHECKPOINT_ROOT/pi05/$RUN"; fi
CONFIG="$RUN_DIR/checkpoints/last/pretrained_model/train_config.json"
[[ -f "$CONFIG" ]] || { echo "No checkpoint to resume from: $CONFIG not found"; exit 1; }

LOG_FILE="$CHECKPOINT_ROOT/pi05/logs/$(basename "$RUN_DIR").log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "resuming $RUN_DIR from $(readlink "$RUN_DIR/checkpoints/last")   log: $LOG_FILE"
# Trackio run = folder name, so the curves continue the same run (also for runs trained before
# Trackio was switched on; load their earlier steps with scripts/log_to_trackio.py first).
uv run lerobot-train --config_path="$CONFIG" --resume=true \
  --trackio.enable=true --trackio.project="${TRACKIO_PROJECT:-pi05_open_box}" \
  --trackio.run_name="$(basename "$RUN_DIR")" \
  "$@" 2>&1 | tee -a "$LOG_FILE"
