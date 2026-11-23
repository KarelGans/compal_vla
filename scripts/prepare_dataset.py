#!/usr/bin/env python
"""Build a training dataset from raw recordings, as described by a dataset YAML (datasets/*.yaml).

1. Merge the LeRobot v3.0 folders into one dataset. LeRobot's aggregator maps task_index by task
   *string*, so folders that number their prompts differently merge correctly.
2. Relabel: every frame of a folder listed under `relabel` gets that prompt.
3. Keep only the 6 right-arm joints in observation.state / action. The left arm and grippers are
   constant in the recordings; leaving them in would put zero-range dims into quantile
   normalization (x / 1e-8), which blows up as soon as the real robot reports a non-zero left-arm pose.
4. Drop the episodes listed under `drop`. Only the tables are rewritten: the kept episodes still point
   at their own stretch of the video files, so no video is re-encoded.
5. Hold out the last `heldout_per_subset` kept episodes of every folder (never trained on) and write
   the split to meta/open_box_split.json, with where every episode came from.
6. Check the result against the raw recordings: episode lengths, labels, joint data and camera
   frames, plus the robot facts in the YAML (constant gripper commands, start and finish poses).

The raw recordings are never modified.

Usage:
    source env.sh
    uv run python scripts/prepare_dataset.py datasets/thor.yaml [--overwrite]
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import _write_parquet, modify_tasks, recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from open_box_vla.constants import FLAP_PROMPTS, FPS, MODEL_DIM_INDICES, MODEL_JOINT_NAMES
from open_box_vla.dataset_specs import DatasetSpec, load_spec

SLICED_KEYS = ("observation.state", "action")
SPLIT_FILE = "meta/open_box_split.json"
FLAP_OF_PROMPT = {prompt: flap for flap, prompt in FLAP_PROMPTS.items()}

log = logging.getLogger("prepare_dataset")


def read_frames(root: Path) -> pd.DataFrame:
    return pd.concat([pq.read_table(f).to_pandas() for f in sorted((root / "data").glob("*/*.parquet"))],
                     ignore_index=True)


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


def drop_episodes_keep_videos(root: Path, drop: list[int]) -> None:
    """Remove whole episodes from a v3.0 dataset by rewriting its tables only.

    The kept episodes are renumbered 0..n-1 and keep pointing at their own stretch of the video
    files (videos/<camera>/from_timestamp..to_timestamp), so no video is re-encoded. The dropped
    episodes' frames stay in the video files but are never read.
    """
    drop = set(drop)
    ep_files = sorted((root / "meta/episodes").glob("*/*.parquet"))
    eps = pd.concat([pq.read_table(f, columns=["episode_index", "length"]).to_pandas() for f in ep_files])
    eps = eps.sort_values("episode_index")
    kept = eps[~eps["episode_index"].isin(drop)]
    new_index = {int(old): new for new, old in enumerate(kept["episode_index"])}
    lengths = kept["length"].to_numpy()
    start = dict(zip(range(len(kept)), np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()))

    # meta/episodes: filter rows and renumber, keeping each file's exact schema
    for f in ep_files:
        table = pq.read_table(f)
        old = table.column("episode_index").to_numpy()
        table = table.filter(pa.array(~np.isin(old, list(drop))))
        if table.num_rows == 0:
            f.unlink()
            continue
        new = np.array([new_index[int(e)] for e in table.column("episode_index").to_numpy()])
        n = table.column("length").to_numpy()
        frm = np.array([start[i] for i in new])
        for name, values in [("episode_index", new), ("dataset_from_index", frm), ("dataset_to_index", frm + n)]:
            i = table.column_names.index(name)
            table = table.set_column(i, table.schema.field(name), pa.array(values).cast(table.schema.field(name).type))
        pq.write_table(table, f, compression="snappy")

    # data: filter rows, renumber episode_index, make the global index contiguous again
    meta = LeRobotDatasetMetadata(repo_id="local/tmp", root=root)
    for f in sorted((root / "data").glob("*/*.parquet")):
        df = pq.read_table(f).to_pandas()
        df = df[~df["episode_index"].isin(drop)].reset_index(drop=True)
        if df.empty:
            f.unlink()
            continue
        df["episode_index"] = df["episode_index"].map(new_index)
        df["index"] = df["episode_index"].map(start) + df["frame_index"]
        _write_parquet(df, f, meta)

    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = len(kept)
    info["total_frames"] = int(lengths.sum())
    info["splits"] = {"train": f"0:{len(kept)}"}
    info_path.write_text(json.dumps(info, indent=4))


def check_robot_facts(spec: DatasetSpec, raw: dict[tuple[str, int], np.ndarray], tasks: dict[tuple[str, int], str]) -> None:
    """The YAML's robot facts must hold for every kept episode (raw 14-D state/action)."""
    other = [i for i in range(14) if i not in MODEL_DIM_INDICES]
    for (s, e), (state, action) in raw.items():
        where = f"{s} episode {e}"
        if np.abs(state[:, other]).max() > 0:
            raise RuntimeError(f"{where}: left arm or gripper state is not always 0")
        if np.ptp(action[:, other], axis=0).max() > 0:
            raise RuntimeError(f"{where}: left arm or gripper commands change")
        for i, value in spec.constant_commands.items():
            if not np.allclose(action[:, i], value):
                raise RuntimeError(f"{where}: command {i} is {action[0, i]}, the YAML says {value}")
        right_s, right_a = state[:, MODEL_DIM_INDICES], action[:, MODEL_DIM_INDICES]
        if spec.finish_pose is not None:
            dev = max(np.abs(right_s[-1] - spec.finish_pose).max(), np.abs(right_a[-1] - spec.finish_pose).max())
            if dev > 1e-3:
                raise RuntimeError(f"{where} ends {dev:.4f} rad away from the YAML's finish_pose")
        flap = FLAP_OF_PROMPT.get(tasks[(s, e)])
        if flap in spec.start_poses:
            dev = np.abs(right_s[0] - spec.start_poses[flap]).max()
            if dev > 1e-3:
                raise RuntimeError(f"{where} starts {dev:.4f} rad away from the YAML's start pose for {flap}")
    log.info("Robot facts hold for all %d kept episodes (constant commands %s, finish pose %s, start poses %s)",
             len(raw), spec.constant_commands, "checked" if spec.finish_pose is not None else "none",
             sorted(spec.start_poses) or "none")


def validate(spec: DatasetSpec, out: Path, kept: list[tuple[str, int]], chunk: int = 50) -> None:
    delta = {"action": [i / FPS for i in range(chunk)]}
    ds = LeRobotDataset(repo_id=spec.repo_id, root=out, delta_timestamps=delta)
    log.info("Loaded: %d episodes, %d frames, fps=%d, cameras %s", ds.num_episodes, ds.num_frames, ds.fps,
             ds.meta.camera_keys)
    log.info("Tasks: %s", list(ds.meta.tasks.index))
    if ds.num_episodes != len(kept):
        raise RuntimeError(f"{ds.num_episodes} episodes, expected {len(kept)}")
    for i in [0, len(ds) // 2, len(ds) - 1]:
        s = ds[i]
        assert s["observation.state"].shape == (len(MODEL_DIM_INDICES),)
        assert s["action"].shape == (chunk, len(MODEL_DIM_INDICES))
        assert all(s[k].shape == (3, 480, 640) for k in ds.meta.camera_keys)

    # every episode: same frames, joints and label as the raw recording it came from
    merged = read_frames(out)
    tasks = {int(r.task_index): t for t, r in ds.meta.tasks.iterrows()}
    raw_frames = {s: read_frames(spec.raw_dir / s) for s in dict.fromkeys(s for s, _ in kept)}
    raw_tasks = {s: {int(r.task_index): t for t, r in pq.read_table(spec.raw_dir / s / "meta/tasks.parquet").to_pandas().iterrows()}
                 for s in raw_frames}
    raw, labels = {}, {}
    for i, (s, e) in enumerate(kept):
        m = merged[merged["episode_index"] == i]
        r = raw_frames[s][raw_frames[s]["episode_index"] == e]
        if len(m) != len(r):
            raise RuntimeError(f"episode {i} ({s} {e}): {len(m)} frames, raw has {len(r)}")
        rs, ra = np.stack(r["observation.state"].to_numpy()), np.stack(r["action"].to_numpy())
        if not (np.array_equal(np.stack(m["observation.state"].to_numpy()), rs[:, MODEL_DIM_INDICES])
                and np.array_equal(np.stack(m["action"].to_numpy()), ra[:, MODEL_DIM_INDICES])):
            raise RuntimeError(f"episode {i} ({s} {e}): joint data differs from the raw recording")
        expected = [spec.relabel[s]] * len(r) if s in spec.relabel else [raw_tasks[s][t] for t in r["task_index"]]
        got = [tasks[t] for t in m["task_index"]]
        if got != expected:
            raise RuntimeError(f"episode {i} ({s} {e}): labels differ from the expected prompts")
        raw[(s, e)] = (rs, ra)
        labels[(s, e)] = got[0]
    log.info("All %d episodes match their raw recordings (frames, joints, labels)", len(kept))
    check_robot_facts(spec, raw, labels)

    # camera frames: first, middle and last frame of the first, middle and last kept episode of each folder
    worst = 0.0
    for s in raw_frames:
        mine = [i for i, (ss, _) in enumerate(kept) if ss == s]
        for i in sorted({mine[0], mine[len(mine) // 2], mine[-1]}):
            e = kept[i][1]
            a = LeRobotDataset(spec.repo_id, root=out, episodes=[i])
            b = LeRobotDataset(f"local/{s}", root=spec.raw_dir / s, episodes=[e])
            for k in [0, len(a) // 2, len(a) - 1]:
                sa, sb = a[k], b[k]
                for cam in ds.meta.camera_keys:
                    worst = max(worst, float((sa[cam] - sb[cam]).abs().max()))
    log.info("Camera frames match the raw videos (largest pixel difference %.4f)", worst)
    if worst > 2 / 255:
        raise RuntimeError("Camera frames differ from the raw videos: the episode/video alignment is off")

    st = ds.meta.stats["action"]
    log.info("action q01=%s", np.round(np.asarray(st["q01"]), 3))
    log.info("action q99=%s", np.round(np.asarray(st["q99"]), 3))
    if np.any(np.asarray(st["q99"]) - np.asarray(st["q01"]) < 1e-6):
        raise RuntimeError("A model dim still has zero quantile range; check MODEL_DIM_INDICES")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spec", type=Path, nargs="?", default=Path("datasets/thor.yaml"), help="dataset YAML")
    p.add_argument("--out", type=Path, default=None, help="default: $HF_LEROBOT_HOME/<repo_id from the YAML>")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    spec = load_spec(args.spec)
    if args.out is None and "HF_LEROBOT_HOME" not in os.environ:
        raise SystemExit("Run `source env.sh` first, or pass --out.")
    out = args.out or Path(os.environ["HF_LEROBOT_HOME"]) / spec.repo_id
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} already exists; pass --overwrite to rebuild it.")
        shutil.rmtree(out)

    roots = [spec.raw_dir / s for s in spec.subsets]
    counts = []
    for r in roots:
        if not (r / "meta/info.json").exists():
            raise SystemExit(f"Not a LeRobot dataset: {r}")
        counts.append(json.loads((r / "meta/info.json").read_text())["total_episodes"])
    unknown = set(spec.relabel) | set(spec.drop)
    if unknown - set(spec.subsets):
        raise SystemExit(f"relabel/drop name folders that are not in subsets: {sorted(unknown - set(spec.subsets))}")
    for s, n in zip(spec.subsets, counts):
        missing = [e for e in spec.drop.get(s, {}) if e >= n]
        if missing:
            raise SystemExit(f"{s} has {n} episodes, cannot drop {missing}")

    # merged episode i comes from source[i] = (folder, episode as recorded)
    source = [(s, e) for s, n in zip(spec.subsets, counts) for e in range(n)]

    staging = out.parent / f"_staging_{out.name}"
    if staging.exists():
        shutil.rmtree(staging)
    log.info("Merging %d folders (%d episodes) -> %s", len(roots), len(source), staging)
    aggregate_datasets(
        repo_ids=[f"local/{s}" for s in spec.subsets],
        aggr_repo_id=spec.repo_id,
        roots=roots,
        aggr_root=staging,
    )

    if spec.relabel:
        log.info("Relabelling %s", spec.relabel)
        modify_tasks(LeRobotDataset(spec.repo_id, root=staging),
                     episode_tasks={i: spec.relabel[s] for i, (s, _) in enumerate(source) if s in spec.relabel})

    log.info("Slicing state/action to right-arm joints %s", MODEL_JOINT_NAMES)
    slice_vector_features(staging, MODEL_DIM_INDICES, MODEL_JOINT_NAMES)

    drop = [i for i, (s, e) in enumerate(source) if e in spec.drop.get(s, {})]
    if drop:
        for i in drop:
            s, e = source[i]
            log.info("Dropping %s episode %d: %s", s, e, spec.drop[s][e])
        drop_episodes_keep_videos(staging, drop)
        recompute_stats(LeRobotDataset(spec.repo_id, root=staging))
    kept = [src for i, src in enumerate(source) if i not in set(drop)]

    # held-out split: the last N kept episodes of each folder; one training episode per folder
    # (the middle one) is the "train probe" that scripts/eval_open_loop.py --train-probe scores
    val, probe, ranges = [], [], {}
    for s in spec.subsets:
        mine = [i for i, (ss, _) in enumerate(kept) if ss == s]
        train_part = mine[: len(mine) - spec.heldout_per_subset]
        val += mine[len(mine) - spec.heldout_per_subset:]
        probe.append(train_part[len(train_part) // 2])
        ranges[s] = [mine[0], mine[-1]]
    split = {
        "train": [i for i in range(len(kept)) if i not in val],
        "val": val,
        "train_probe": probe,
        "source_episode_ranges": ranges,
        "episodes": [{"episode": i, "folder": s, "recorded_episode": e} for i, (s, e) in enumerate(kept)],
        "dropped": {s: {str(e): why for e, why in eps.items()} for s, eps in spec.drop.items()},
        "built_from": str(args.spec),
    }
    (staging / SPLIT_FILE).write_text(json.dumps(split, indent=2))

    out.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(out)
    log.info("Validating %s", out)
    validate(spec, out, kept)
    log.info("Done. %d episodes (%d dropped): train=%d, held out=%s, train probe=%s",
             len(kept), len(drop), len(split["train"]), val, probe)


if __name__ == "__main__":
    main()
