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
    subsets: list[str]                     # folders, relative to raw_dir (may contain a "/")
    relabel: dict[str, str]                # folder -> prompt for all of its frames
    # folder -> {(episode as recorded, None for the whole episode or the flap of one piece): reason}
    drop: dict[str, dict[tuple[int, str | None], str]]
    # folders recorded with several flaps per episode -> {episode as recorded: cut times (s) that
    # replace the label changes}; every episode of these folders is cut into one episode per flap
    split_by_label: dict[str, dict[int, list[float]]]
    heldout_per_subset: int                # per flap of each folder
    flap_balance_exponent: float           # 1 = every frame equally often; see flap_shares()
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
    exponent = float(d.get("flap_balance_exponent", 1.0))
    if not 0.0 <= exponent <= 1.0:
        raise ValueError(f"{path}: flap_balance_exponent must be between 0 and 1, got {exponent}")
    pose = lambda v: None if v is None else np.asarray(v, dtype=np.float32)  # noqa: E731
    split = {s: {int(e): [float(t) for t in (ts if isinstance(ts, list) else [ts])] for e, ts in (cuts or {}).items()}
             for s, cuts in (d.get("split_by_label") or {}).items()}
    drop = {s: {drop_key(k, path): str(why) for k, why in (eps or {}).items()} for s, eps in (d.get("drop") or {}).items()}
    for s, eps in drop.items():
        if any(part for _, part in eps) and s not in split:
            raise ValueError(f"{path}: drop names pieces (\"<episode> <flap>\") of {s}, which is not under split_by_label")
    return DatasetSpec(
        path=path,
        repo_id=d["repo_id"],
        raw_dir=Path(d["raw_dir"]),
        subsets=list(d["subsets"]),
        relabel=dict(d.get("relabel") or {}),
        drop=drop,
        split_by_label=split,
        heldout_per_subset=int(d.get("heldout_per_subset", 1)),
        flap_balance_exponent=exponent,
        constant_commands={FULL_JOINT_NAMES.index(k): float(v) for k, v in (robot.get("constant_commands") or {}).items()},
        finish_pose=pose(robot.get("finish_pose")),
        start_poses={k: pose(v) for k, v in (robot.get("start_poses") or {}).items()},
        flap_order=list(robot.get("flap_order") or FLAP_PROMPTS),
    )


def drop_key(key, path: Path | str = "") -> tuple[int, str | None]:
    """A `drop` entry: an episode number (the whole episode), or "<episode> <flap>" (one piece of
    an episode of a split_by_label folder)."""
    episode, _, part = str(key).strip().partition(" ")
    if not episode.isdigit() or (part and part not in FLAP_PROMPTS):
        raise ValueError(f"{path}: drop entries are an episode number or \"<episode> <flap>\", got {key!r}")
    return int(episode), part or None


def drop_name(episode: int, part: str | None) -> str:
    return f"{episode} {part}" if part else str(episode)


def flap_shares(train_frames: dict[str, int], exponent: float) -> dict[str, float]:
    """Share of the training draws each flap gets when a flap with n training frames is weighted n**exponent.

    exponent 1: every frame is drawn equally often, so the shares follow the amount of data.
    exponent 0: every flap gets the same share, however many frames it has.
    In between, the large flaps are drawn less and the small ones more (pi0 mixed its datasets with 0.43).
    """
    weights = {flap: n**exponent for flap, n in train_frames.items() if n > 0}
    total = sum(weights.values())
    return {flap: w / total for flap, w in weights.items()}


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
