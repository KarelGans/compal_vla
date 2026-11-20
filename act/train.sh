#!/usr/bin/env bash
# Train ACT on the open-box-flaps dataset. All hyperparameters are in act/config.yaml.
#
#   bash act/train.sh                                  # train with act/config.yaml
#   bash act/train.sh --steps=20000 --batch_size=16    # change single values for this run
#   CONFIG=act/my_variant.yaml bash act/train.sh       # train with another YAML
#   DRY_RUN=1 bash act/train.sh                        # run the checks and print the command, don't train
#
# Checkpoints:  $CHECKPOINT_ROOT/act/<RUN_NAME>/checkpoints/<step>/{pretrained_model,training_state}
# Log + YAML:   $CHECKPOINT_ROOT/act/logs/<RUN_NAME>.log and .yaml (copy of the config used)
# Curves:       uv run trackio show --project act_open_box
# Resume:       bash act/resume.sh <RUN_NAME>
#
# Settings (environment): CONFIG, RUN_NAME, DRY_RUN=1
set -euo pipefail
cd "$(dirname "$0")/.."
source ./env.sh

CONFIG="${CONFIG:-act/config.yaml}"
RUN_NAME="${RUN_NAME:-act_open_box_$(date +%Y%m%d_%H%M)}"
ACT_ROOT="$CHECKPOINT_ROOT/act"
OUTPUT_DIR="$ACT_ROOT/$RUN_NAME"
LOG_FILE="$ACT_ROOT/logs/$RUN_NAME.log"

[[ -f "$CONFIG" ]] || { echo "Config $CONFIG not found"; exit 1; }
[[ -e "$OUTPUT_DIR" ]] && { echo "$OUTPUT_DIR already exists. Pick another RUN_NAME, or continue it with: bash act/resume.sh $RUN_NAME"; exit 1; }

# The dataset must exist, and the YAML's held-out episodes must be the ones the evaluation uses.
uv run python - "$CONFIG" <<'EOF'
import json, os, sys, yaml
from pathlib import Path
ds = yaml.safe_load(open(sys.argv[1]))["dataset"]
root = Path(os.environ["HF_LEROBOT_HOME"]) / ds["repo_id"]
if not (root / "meta/info.json").is_file():
    sys.exit(f"Dataset {root} not found. For {os.environ['DATASET_REPO_ID_ONEHOT']}: uv run python scripts/add_flap_onehot.py")
val = json.loads((root / "meta/open_box_split.json").read_text())["val"]
if sorted(ds.get("exclude_episodes") or []) != sorted(val):
    sys.exit(f"dataset.exclude_episodes in {sys.argv[1]} is {ds.get('exclude_episodes')}, "
             f"but the held-out episodes in meta/open_box_split.json are {val}")
print(f"dataset {root}, held-out episodes {val}")
EOF

CMD=(uv run lerobot-train
  --config_path="$CONFIG"
  --output_dir="$OUTPUT_DIR"
  --job_name="$RUN_NAME"
  --trackio.run_name="$RUN_NAME"
  "$@")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${CMD[@]}"; echo; exit 0
fi
mkdir -p "$ACT_ROOT/logs"
{ cat "$CONFIG"; echo; echo "# command-line overrides for this run: ${*:-none}"; } > "$ACT_ROOT/logs/$RUN_NAME.yaml"
echo "run=$RUN_NAME  config=$CONFIG"
echo "output: $OUTPUT_DIR"
echo "log:    $LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
