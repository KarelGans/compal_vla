#!/usr/bin/env python
"""Build the training dataset from the raw mocap recordings.

1. Merge the 4 LeRobot v3.0 subsets into one dataset. LeRobot's aggregator remaps
   task_index by task *string*, which matters here because each subset numbers the
   four flap prompts differently.
2. Keep only the 6 right-arm joints in observation.state / action. The left arm and
   grippers are constant in the recordings. Leaving them in would put zero-range dims
   into quantile normalization (x / 1e-8), which blows up as soon as the real robot
   reports a non-zero left-arm pose.
3. Hold out the last episode of every subset for open-loop validation and write the
   split to meta/open_box_split.json.
4. Load the result back through LeRobotDataset and sanity-check a few samples.

The raw dataset is never modified.

Usage:
    source env.sh
    uv run python scripts/prepare_dataset.py [--overwrite]
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import _write_parquet
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from open_box_vla.constants import (
    FPS,
    MODEL_DIM_INDICES,
    MODEL_JOINT_NAMES,
    SOURCE_SUBSETS,
)

SLICED_KEYS = ("observation.state", "action")
SPLIT_FILE = "meta/open_box_split.json"

log = logging.getLogger("prepare_dataset")


def slice_vector_features(root: Path, idx: list[int], names: list[str]) -> None:
    """Keep only `idx` dims of state/action everywhere they appear in a v3.0 dataset."""
    lo, hi = idx[0], idx[-1] + 1
    assert idx == list(range(lo, hi)), "only contiguous slices are supported"

    # 1) info.json: feature shape + names
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    for k in SLICED_KEYS:
        info["features"][k]["shape"] = [len(idx)]
        info["features"][k]["names"] = names
    info_path.write_text(json.dumps(info, indent=4))

    # 2) data parquet files, rewritten through LeRobot's own writer so the embedded
    #    HF feature schema matches the new shape.
    meta = LeRobotDatasetMetadata(repo_id="local/tmp", root=root)
    for f in sorted((root / "data").glob("*/*.parquet")):
        df = pq.read_table(f).to_pandas()
        for k in SLICED_KEYS:
            df[k] = [np.asarray(v, dtype=np.float32)[lo:hi] for v in df[k]]
        _write_parquet(df, f, meta)

    # 3) global stats (every stat is per-dimension, so slicing is exact)
    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    for k in SLICED_KEYS:
        for name, val in stats[k].items():
            if name != "count":
                stats[k][name] = val[lo:hi]
    stats_path.write_text(json.dumps(stats, indent=4))

    # 4) per-episode stats in meta/episodes (needed for future merges / edits)
    for f in sorted((root / "meta/episodes").glob("*/*.parquet")):
        table = pq.read_table(f)
        for col in table.column_names:
            if any(col.startswith(f"stats/{k}/") for k in SLICED_KEYS) and not col.endswith("/count"):
                i = table.column_names.index(col)
                sliced = pc.list_slice(table.column(col), lo, hi)
                table = table.set_column(i, pa.field(col, table.schema.field(col).type), sliced.cast(table.schema.field(col).type))
        pq.write_table(table, f, compression="snappy")


def validate(repo_id: str, root: Path, chunk: int = 50) -> None:
    delta = {"action": [i / FPS for i in range(chunk)]}
    ds = LeRobotDataset(repo_id=repo_id, root=root, delta_timestamps=delta)
    log.info("Loaded: %d episodes, %d frames, fps=%d", ds.num_episodes, ds.num_frames, ds.fps)
    log.info("Tasks: %s", list(ds.meta.tasks.index))
    for i in [0, len(ds) // 2, len(ds) - 1]:
        s = ds[i]
        shapes = {k: tuple(v.shape) for k, v in s.items() if hasattr(v, "shape")}
        log.info("sample %d: task=%r shapes=%s", i, s["task"], shapes)
        assert s["observation.state"].shape == (len(MODEL_DIM_INDICES),)
        assert s["action"].shape == (chunk, len(MODEL_DIM_INDICES))
        assert all(s[k].shape == (3, 480, 640) for k in ds.meta.camera_keys)
    st = ds.meta.stats["action"]
    log.info("action q01=%s", np.round(np.asarray(st["q01"]), 3))
    log.info("action q99=%s", np.round(np.asarray(st["q99"]), 3))
    if np.any(np.asarray(st["q99"]) - np.asarray(st["q01"]) < 1e-6):
        raise RuntimeError("A model dim still has zero quantile range; check MODEL_DIM_INDICES")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path(os.environ.get("RAW_DATASET_DIR", "")))
    p.add_argument("--repo-id", default=os.environ.get("DATASET_REPO_ID", "compal/open_box_flaps_right6"))
    p.add_argument("--out", type=Path, default=Path(os.environ["DATASET_ROOT"]) if "DATASET_ROOT" in os.environ else None)
    p.add_argument("--subsets", nargs="+", default=SOURCE_SUBSETS)
    p.add_argument("--val-per-subset", type=int, default=1, help="hold out the last N episodes of each subset")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.out is None or not str(args.raw_dir):
        raise SystemExit("Run `source env.sh` first, or pass --raw-dir and --out.")
    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out} already exists; pass --overwrite to rebuild it.")
        shutil.rmtree(args.out)

    roots = [args.raw_dir / s for s in args.subsets]
    for r in roots:
        if not (r / "meta/info.json").exists():
            raise SystemExit(f"Not a LeRobot dataset: {r}")

    staging = args.out.parent / f"_staging_{args.out.name}"
    if staging.exists():
        shutil.rmtree(staging)

    log.info("Merging %d subsets -> %s", len(roots), staging)
    aggregate_datasets(
        repo_ids=[f"local/{s}" for s in args.subsets],
        aggr_repo_id=args.repo_id,
        roots=roots,
        aggr_root=staging,
    )

    log.info("Slicing state/action to right-arm joints %s", MODEL_JOINT_NAMES)
    slice_vector_features(staging, MODEL_DIM_INDICES, MODEL_JOINT_NAMES)

    # train/val split: last N episodes of each subset (merged indices are contiguous per subset)
    counts = [json.loads((r / "meta/info.json").read_text())["total_episodes"] for r in roots]
    val, offset, source = [], 0, {}
    for name, n in zip(args.subsets, counts):
        val += list(range(offset + n - args.val_per_subset, offset + n))
        source[name] = [offset, offset + n - 1]
        offset += n
    split = {
        "train": [i for i in range(offset) if i not in val],
        "val": val,
        "source_episode_ranges": source,
    }
    (staging / SPLIT_FILE).write_text(json.dumps(split, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(args.out)
    log.info("Validating %s", args.out)
    validate(args.repo_id, args.out)
    log.info("Done. train=%d episodes, val=%s", len(split["train"]), split["val"])


if __name__ == "__main__":
    main()
