#!/usr/bin/env bash
# Continue an interrupted ACT run from its last checkpoint (weights, optimizer, step count,
# data order). Settings come from the checkpoint's train_config.json, not from act/config.yaml.
#
#   bash act/resume.sh <RUN_NAME>                    # e.g. act_open_box_20261115_2100
#   bash act/resume.sh <RUN_NAME> --steps=200000     # resume and train for longer
#
# Only steps up to the last saved checkpoint are kept (save_freq in the YAML).
set -euo pipefail
cd "$(dirname "$0")/.."
source ./env.sh

RUN="${1:?usage: bash act/resume.sh <RUN_NAME or run dir> [extra lerobot-train flags]}"; shift
if [[ -d "$RUN" ]]; then RUN_DIR="$RUN"; else RUN_DIR="$CHECKPOINT_ROOT/act/$RUN"; fi
CONFIG="$RUN_DIR/checkpoints/last/pretrained_model/train_config.json"
[[ -f "$CONFIG" ]] || { echo "No checkpoint to resume from: $CONFIG not found"; exit 1; }

LOG_FILE="$CHECKPOINT_ROOT/act/logs/$(basename "$RUN_DIR").log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "resuming $RUN_DIR from $(readlink "$RUN_DIR/checkpoints/last")   log: $LOG_FILE"
uv run lerobot-train --config_path="$CONFIG" --resume=true "$@" 2>&1 | tee -a "$LOG_FILE"
