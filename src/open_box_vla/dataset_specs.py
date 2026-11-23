"""Per-dataset facts, read from datasets/<name>.yaml (one file per training dataset).

prepare_dataset.py builds a dataset from its YAML; the robot code (policy_runner.py, run_robot.py)
looks up the YAML of the dataset a checkpoint was trained on, through the checkpoint's train_config.json.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .constants import FLAP_PROMPTS, FULL_JOINT_NAMES

SPEC_DIR = Path(__file__).resolve().parents[2] / "datasets"


@dataclass
class DatasetSpec:
    path: Path
    repo_id: str
    raw_dir: Path
    subsets: list[str]
    relabel: dict[str, str]                # folder -> prompt for all of its frames
    drop: dict[str, dict[int, str]]        # folder -> {episode as recorded: reason}
    heldout_per_subset: int
    constant_commands: dict[int, float]    # 14-D joint index -> command the recordings hold constant
    finish_pose: np.ndarray | None         # (6,) right arm: where every episode ends, or None
    start_poses: dict[str, np.ndarray]     # flap -> (6,) right-arm pose its episodes start from
    flap_order: list[str]


def load_spec(path: str | Path) -> DatasetSpec:
    path = Path(path)
    d = yaml.safe_load(path.read_text())
    robot = d.get("robot") or {}
    unknown = set(robot.get("start_poses") or {}) | set(robot.get("flap_order") or [])
    if unknown - set(FLAP_PROMPTS):
        raise ValueError(f"{path}: unknown flaps {sorted(unknown - set(FLAP_PROMPTS))}")
    bad_prompts = set((d.get("relabel") or {}).values()) - set(FLAP_PROMPTS.values())
    if bad_prompts:
        raise ValueError(f"{path}: relabel prompts must be one of {list(FLAP_PROMPTS.values())}, got {bad_prompts}")
    pose = lambda v: None if v is None else np.asarray(v, dtype=np.float32)  # noqa: E731
    return DatasetSpec(
        path=path,
        repo_id=d["repo_id"],
        raw_dir=Path(d["raw_dir"]),
        subsets=list(d["subsets"]),
        relabel=dict(d.get("relabel") or {}),
        drop={s: {int(e): str(why) for e, why in (eps or {}).items()} for s, eps in (d.get("drop") or {}).items()},
        heldout_per_subset=int(d.get("heldout_per_subset", 1)),
        constant_commands={FULL_JOINT_NAMES.index(k): float(v) for k, v in (robot.get("constant_commands") or {}).items()},
        finish_pose=pose(robot.get("finish_pose")),
        start_poses={k: pose(v) for k, v in (robot.get("start_poses") or {}).items()},
        flap_order=list(robot.get("flap_order") or FLAP_PROMPTS),
    )


def spec_for_repo_id(repo_id: str) -> DatasetSpec:
    """The YAML in datasets/ with this repo_id (ACT's "<repo_id>_onehot" copies count as the original)."""
    base = repo_id.removesuffix("_onehot")
    for path in sorted(SPEC_DIR.glob("*.yaml")):
        if yaml.safe_load(path.read_text()).get("repo_id") == base:
            return load_spec(path)
    raise FileNotFoundError(f"No dataset YAML in {SPEC_DIR} has repo_id {base!r}")


def checkpoint_dataset(checkpoint: str | Path) -> tuple[str, Path]:
    """(repo_id, root) of the dataset a checkpoint (.../pretrained_model) was trained on."""
    from lerobot.utils.constants import HF_LEROBOT_HOME

    cfg = json.loads((Path(checkpoint) / "train_config.json").read_text())["dataset"]
    # no root (e.g. ACT, configured by repo_id only): lerobot-train used $HF_LEROBOT_HOME/<repo_id>
    return cfg["repo_id"], Path(cfg["root"]) if cfg.get("root") else Path(HF_LEROBOT_HOME) / cfg["repo_id"]


def spec_for_checkpoint(checkpoint: str | Path) -> DatasetSpec:
    return spec_for_repo_id(checkpoint_dataset(checkpoint)[0])


def load_split(dataset_root: str | Path) -> dict:
    """meta/open_box_split.json of a prepared dataset.

    "val" are the held-out episodes; "train_probe" is one training episode per recording folder
    (the middle one), scored by eval_open_loop.py --train-probe. Split files written before
    train_probe existed get it from their "source_episode_ranges".
    """
    split = json.loads((Path(dataset_root) / "meta/open_box_split.json").read_text())
    if "train_probe" not in split:
        val = set(split["val"])
        split["train_probe"] = []
        for lo, hi in split["source_episode_ranges"].values():
            train = [e for e in range(lo, hi + 1) if e not in val]
            split["train_probe"].append(train[len(train) // 2])
    return split
