# Paths shared by every script. The pi05/ scripts source this automatically;
# source it yourself (`source env.sh`) before running scripts/prepare_dataset.py.
#
# Nothing large may go under /home or /tmp: user1 has a 250 GB quota there that is
# almost full. Everything big lives on /media/ai/aipc.
export VLA_STORAGE="${VLA_STORAGE:-/media/ai/aipc/Karel/compal_vla_models}"
export RAW_DATASET_DIR="${RAW_DATASET_DIR:-/media/ai/aipc/Karel/Mocap/Dataset}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/media/ai/aipc/Karel/Mocap/checkpoints}"  # runs go in $CHECKPOINT_ROOT/<model>/<run>

export HF_HOME="${HF_HOME:-$VLA_STORAGE/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$VLA_STORAGE/torch_cache}"  # ImageNet ResNet18 weights for ACT
export TRACKIO_DIR="${TRACKIO_DIR:-$CHECKPOINT_ROOT/trackio}"  # training curves; view with: uv run trackio show     # model weights + tokenizer cache
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"  # keep the HF token private (not on the shared drive)
export HF_LEROBOT_HOME="$VLA_STORAGE/lerobot_datasets"

export DATASET_REPO_ID="${DATASET_REPO_ID:-compal/open_box_flaps_right6}"
export DATASET_ROOT="${DATASET_ROOT:-$HF_LEROBOT_HOME/$DATASET_REPO_ID}"
# Same data plus the flap as 4 numbers, for models without a text input (ACT); scripts/add_flap_onehot.py
export DATASET_REPO_ID_ONEHOT="${DATASET_REPO_ID_ONEHOT:-${DATASET_REPO_ID}_onehot}"
export DATASET_ROOT_ONEHOT="${DATASET_ROOT_ONEHOT:-$HF_LEROBOT_HOME/$DATASET_REPO_ID_ONEHOT}"
export PRETRAINED_POLICY="${PRETRAINED_POLICY:-lerobot/pi05_base}"

export TOKENIZERS_PARALLELISM=false
