#!/usr/bin/env python
"""Make a copy of the prepared dataset with the current flap as 4 extra numbers per frame.

ACT has no text input, so it cannot read "open the back flap". The recordings open the flaps in
4 different orders, so without this input ACT cannot tell which flap comes next. This adds
`observation.environment_state` = [back, front, right, left], 1 for the frame's flap and 0
otherwise. ACT reads that key as an extra input. Everything else, including the videos and the
train/held-out split, is copied unchanged.

    source env.sh
    uv run python scripts/add_flap_onehot.py          # ~1 min, writes $DATASET_ROOT_ONEHOT (~2.7 GB)
"""

import os
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from open_box_vla.constants import FLAP_ONEHOT_KEY, FLAP_PROMPTS, flap_onehot

SPLIT_FILE = "meta/open_box_split.json"


def main() -> None:
    src_root = Path(os.environ["DATASET_ROOT"])
    dst_repo = os.environ["DATASET_REPO_ID_ONEHOT"]
    dst_root = Path(os.environ["DATASET_ROOT_ONEHOT"])
    if dst_root.exists():
        raise SystemExit(f"{dst_root} already exists; delete it first to rebuild")

    src = LeRobotDataset(os.environ["DATASET_REPO_ID"], root=src_root)
    task_of_index = {int(i): t for t, i in zip(src.meta.tasks.index, src.meta.tasks["task_index"], strict=True)}
    onehot_of_index = {i: np.asarray(flap_onehot(t), dtype=np.float32) for i, t in task_of_index.items()}

    def value(row: dict, _episode: int, _frame: int) -> np.ndarray:
        return onehot_of_index[int(row["task_index"])]

    info = {"dtype": "float32", "shape": [len(FLAP_PROMPTS)], "names": [f"flap_{k}" for k in FLAP_PROMPTS]}
    dst = add_features(src, {FLAP_ONEHOT_KEY: (value, info)}, output_dir=dst_root, repo_id=dst_repo)
    shutil.copy(src_root / SPLIT_FILE, dst_root / SPLIT_FILE)

    # check: every frame has exactly one flap set, and it matches the frame's prompt
    col = np.stack(dst.hf_dataset[FLAP_ONEHOT_KEY])
    tasks = np.asarray(dst.hf_dataset["task_index"])
    assert col.shape == (len(src), len(FLAP_PROMPTS)) and (col.sum(1) == 1).all()
    for i, t in task_of_index.items():
        assert (col[tasks == i] == onehot_of_index[i]).all(), t
    counts = {k: int(c) for k, c in zip(FLAP_PROMPTS, col.sum(0), strict=True)}
    print(f"wrote {dst_root}: {len(dst)} frames, {dst.num_episodes} episodes, frames per flap {counts}")


if __name__ == "__main__":
    main()
