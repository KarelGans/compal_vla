#!/usr/bin/env python
"""Build a training dataset from raw recordings, as described by a dataset YAML (datasets/*.yaml).

1. Merge the LeRobot v3.0 folders into one dataset. LeRobot's aggregator maps task_index by task
   *string*, so folders that number their prompts differently merge correctly.
2. Split: the episodes of folders under `split_by_label` hold several flaps one after another, each
   with its own per-frame label. Each is cut where the label changes (or at the times the YAML
   gives), with the arm resting at the finish pose, and every piece becomes an episode of its own.
   Training needs one flap per episode: otherwise the action chunks at the end of one flap would
   already contain the start of the next, under the first flap's prompt.
3. Relabel: every frame of a folder listed under `relabel` gets that prompt.
4. Keep only the 6 right-arm joints in observation.state / action. The left arm and grippers are
   constant in the recordings; leaving them in would put zero-range dims into quantile
   normalization (x / 1e-8), which blows up as soon as the real robot reports a non-zero left-arm pose.
5. Drop the episodes listed under `drop` ("<episode> <flap>" drops one piece of a split episode).
6. Hold out the last `heldout_per_subset` kept episodes of every flap of every folder (never trained
   on) and write the split to meta/open_box_split.json, with where every episode came from.
7. Check the result against the raw recordings: episode lengths, labels, joint data and camera
   frames, plus the robot facts in the YAML (constant gripper commands, start and finish poses).
8. Log how often training will draw each flap with the YAML's flap_balance_exponent
   (scripts/train_with_heldout.py applies it; nothing about it is stored in the dataset).

Splitting and dropping rewrite only the tables: every episode keeps pointing at its own stretch of
the video files, so no video is re-encoded.
Folders are given relative to the YAML's raw_dir and may sit in subfolders ("dataset_v2/<folder>").
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import _write_parquet, modify_tasks, recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from open_box_vla.constants import FLAP_PROMPTS, FPS, MODEL_DIM_INDICES, MODEL_JOINT_NAMES
from open_box_vla.dataset_specs import DatasetSpec, drop_name, flap_shares, load_spec

SLICED_KEYS = ("observation.state", "action")
SPLIT_FILE = "meta/open_box_split.json"
FLAP_OF_PROMPT = {prompt: flap for flap, prompt in FLAP_PROMPTS.items()}

log = logging.getLogger("prepare_dataset")


@dataclass(frozen=True)
class Source:
    """Where an episode of the built dataset comes from: a recorded episode, or one flap's piece of
    an episode of a split_by_label folder."""

    folder: str
    episode: int                 # as recorded in the folder
    part: str | None = None      # the piece's flap
    start: int = 0               # the piece's first frame in the recorded episode
    length: int | None = None    # the piece's frames; None = the whole recorded episode

    def name(self) -> str:
        return f"{self.folder} episode {drop_name(self.episode, self.part)}"


def local_repo_id(folder: str) -> str:
    """Repo id for reading a raw folder from disk ("dataset_v2/x" -> "local/dataset_v2__x")."""
    return "local/" + folder.replace("/", "__")


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


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[first, end) of every run of True."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def split_episodes_by_label(root: Path, spec: DatasetSpec, source: list[Source]) -> list[Source]:
    """Cut every episode of the split_by_label folders into one episode per flap (tables only).

    A cut goes where the per-frame label changes, or at the times the YAML gives for that episode
    (the k-th piece then takes the episode's k-th label). The arm must be resting at the finish pose
    on both sides of every cut, so each piece starts and ends like a one-flap recording. Every piece
    points at its own stretch of the episode's video (from_timestamp moves forward), so no video is
    re-encoded. The pieces' action/state stats in meta/episodes are recomputed; their image stats
    stay those of the whole episode (training uses meta/stats.json, recomputed later).
    """
    if spec.finish_pose is None:
        raise SystemExit("split_by_label needs robot.finish_pose: the arm must rest there at every cut")
    prompts = {int(r.task_index): t for t, r in pq.read_table(root / "meta/tasks.parquet").to_pandas().iterrows()}
    task_index = {t: i for i, t in prompts.items()}
    features = LeRobotDatasetMetadata(repo_id="local/tmp", root=root).features
    data_files = sorted((root / "data").glob("*/*.parquet"))
    cols = ["episode_index", "frame_index", "task_index", *SLICED_KEYS]
    frames = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in data_files], ignore_index=True)
    episodes = {int(e): g.sort_values("frame_index") for e, g in frames.groupby("episode_index")}

    pieces_of: dict[int, list[tuple[int, int, str | None]]] = {}  # episode -> [(first frame, end, flap)]
    new_source = []
    for i, src in enumerate(source):
        g = episodes[i]
        if src.folder not in spec.split_by_label or (src.episode, None) in spec.drop.get(src.folder, {}):
            pieces_of[i] = [(0, len(g), None)]  # not cut (or dropped whole later)
            new_source.append(src)
            continue
        labels = [prompts[int(t)] for t in g["task_index"]]
        if set(labels) - set(FLAP_OF_PROMPT):
            raise RuntimeError(f"{src.name()}: labels {sorted(set(labels) - set(FLAP_OF_PROMPT))} are not flap prompts")
        flaps = [FLAP_OF_PROMPT[t] for t in labels]
        order = [f for k, f in enumerate(flaps) if k == 0 or f != flaps[k - 1]]
        given = spec.split_by_label[src.folder].get(src.episode)
        cuts = ([round(t * FPS) for t in given] if given is not None
                else [k for k in range(1, len(flaps)) if flaps[k] != flaps[k - 1]])
        if len(cuts) != len(order) - 1 or cuts != sorted(cuts) or not all(0 < c < len(g) for c in cuts):
            raise RuntimeError(f"{src.name()}: its labels {order} need {len(order) - 1} cut(s) inside the "
                               f"episode, got frames {cuts}")
        state = np.stack(g["observation.state"].to_numpy())[:, MODEL_DIM_INDICES]
        resting = np.abs(state - spec.finish_pose).max(axis=1) < 1e-3
        for c in cuts:
            if not (resting[c - 1] and resting[c]):
                rests = ", ".join(f"{a / FPS:.1f}-{b / FPS:.1f} s" for a, b in _runs(resting))
                raise RuntimeError(
                    f"{src.name()}: the arm is not resting at the finish pose at the cut at {c / FPS:.1f} s "
                    f"(it rests there at {rests}); give the cut under split_by_label, or drop the episode")
        bounds = [0, *cuts, len(g)]
        pieces_of[i] = [(a, b, f) for a, b, f in zip(bounds[:-1], bounds[1:], order)]
        new_source += [Source(src.folder, src.episode, f, a, b - a) for a, b, f in pieces_of[i]]
        log.info("Cutting %s at %s: %s", src.name(), ", ".join(f"{c / FPS:.1f} s" for c in cuts) or "no change of label",
                 ", ".join(f"{f} {(b - a) / FPS:.1f} s" for a, b, f in pieces_of[i]))

    new_id, lengths = {}, []
    for i in range(len(source)):
        for a, b, _ in pieces_of[i]:
            new_id[(i, a)] = len(lengths)
            lengths.append(b - a)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)

    # meta/episodes: one row per piece, keeping each file's exact schema
    for f in sorted((root / "meta/episodes").glob("*/*.parquet")):
        table = pq.read_table(f)
        rows, pieces = [], []
        for r, e in enumerate(table.column("episode_index").to_pylist()):
            for a, b, flap in pieces_of[e]:
                rows.append(r)
                pieces.append((e, a, b, flap))
        table = table.take(pa.array(rows, type=pa.int64()))
        new = [new_id[(e, a)] for e, a, _, _ in pieces]
        values = {
            "episode_index": new,
            "length": [b - a for _, a, b, _ in pieces],
            "dataset_from_index": [int(starts[j]) for j in new],
            "dataset_to_index": [int(starts[j]) + b - a for j, (_, a, b, _) in zip(new, pieces)],
            "tasks": [[FLAP_PROMPTS[flap]] if flap else t for t, (*_, flap) in zip(table.column("tasks").to_pylist(), pieces)],
        }
        for name in [c for c in table.column_names if c.startswith("videos/") and c.endswith("/from_timestamp")]:
            to_name = name.removesuffix("/from_timestamp") + "/to_timestamp"
            frm, to = table.column(name).to_pylist(), table.column(to_name).to_pylist()
            values[name] = [x + a / FPS for x, (_, a, _, _) in zip(frm, pieces)]
            values[to_name] = [x + b / FPS if flap else t for x, t, (_, _, b, flap) in zip(frm, to, pieces)]
        for k in SLICED_KEYS:
            stat_cols = [c for c in table.column_names if c.startswith(f"stats/{k}/")]
            stats = {c: table.column(c).to_pylist() for c in stat_cols}
            for row, (e, a, b, flap) in enumerate(pieces):
                if flap is None:
                    continue
                st = compute_episode_stats({k: np.stack(episodes[e][k].to_numpy()[a:b])}, {k: features[k]})[k]
                for c in stat_cols:
                    stats[c][row] = np.asarray(st[c.rsplit("/", 1)[1]]).reshape(np.shape(stats[c][row])).tolist()
            values.update(stats)
        for name, v in values.items():
            field = table.schema.field(name)
            table = table.set_column(table.column_names.index(name), field, pa.array(v, type=field.type))
        pq.write_table(table, f, compression="snappy")

    # data: renumber, restart frame_index/timestamp in every piece, one label per piece
    meta = LeRobotDatasetMetadata(repo_id="local/tmp", root=root)
    for f in data_files:
        df = pq.read_table(f).to_pandas()
        ep, fi = df["episode_index"].to_numpy(), df["frame_index"].to_numpy()
        new_ep, new_fi = ep.copy(), fi.copy()
        for e in np.unique(ep):
            for a, b, flap in pieces_of[int(e)]:
                m = (ep == e) & (fi >= a) & (fi < b)
                new_ep[m], new_fi[m] = new_id[(int(e), a)], fi[m] - a
                if flap:
                    df.loc[m, "task_index"] = task_index[FLAP_PROMPTS[flap]]
                    df.loc[m, "timestamp"] = ((fi[m] - a) / FPS).astype(np.float32)
        df["episode_index"], df["frame_index"] = new_ep, new_fi
        df["index"] = starts[new_ep] + new_fi
        _write_parquet(df, f, meta)

    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = len(lengths)
    info["splits"] = {"train": f"0:{len(lengths)}"}
    info_path.write_text(json.dumps(info, indent=4))
    return new_source


def episode_prompts(root: Path) -> dict[int, str]:
    """The prompt of every episode; an episode with more than one is an error."""
    data = pd.concat([pq.read_table(f, columns=["episode_index", "task_index"]).to_pandas()
                      for f in sorted((root / "data").glob("*/*.parquet"))], ignore_index=True)
    prompts = {int(r.task_index): t for t, r in pq.read_table(root / "meta/tasks.parquet").to_pandas().iterrows()}
    out = {}
    for e, g in data.groupby("episode_index"):
        labels = {prompts[int(t)] for t in g["task_index"].unique()}
        if len(labels) != 1:
            raise RuntimeError(f"episode {e} has several labels {sorted(labels)}; list its folder under split_by_label")
        out[int(e)] = labels.pop()
    return out


def check_robot_facts(spec: DatasetSpec, raw: dict[Source, tuple[np.ndarray, np.ndarray]], tasks: dict[Source, str]) -> None:
    """The YAML's robot facts must hold for every kept episode (raw 14-D state/action)."""
    other = [i for i in range(14) if i not in MODEL_DIM_INDICES]
    for src, (state, action) in raw.items():
        where = src.name()
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
        flap = FLAP_OF_PROMPT.get(tasks[src])
        if flap in spec.start_poses:
            dev = np.abs(right_s[0] - spec.start_poses[flap]).max()
            if dev > 1e-3:
                raise RuntimeError(f"{where} starts {dev:.4f} rad away from the YAML's start pose for {flap}")
    log.info("Robot facts hold for all %d kept episodes (constant commands %s, finish pose %s, start poses %s)",
             len(raw), spec.constant_commands, "checked" if spec.finish_pose is not None else "none",
             sorted(spec.start_poses) or "none")


def validate(spec: DatasetSpec, out: Path, kept: list[Source], chunk: int = 50) -> None:
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

    # every episode: same frames, joints and label as the stretch of the raw recording it came from
    merged = read_frames(out)
    tasks = {int(r.task_index): t for t, r in ds.meta.tasks.iterrows()}
    raw_frames = {s: read_frames(spec.raw_dir / s).sort_values(["episode_index", "frame_index"])
                  for s in dict.fromkeys(src.folder for src in kept)}
    raw_tasks = {s: {int(r.task_index): t for t, r in pq.read_table(spec.raw_dir / s / "meta/tasks.parquet").to_pandas().iterrows()}
                 for s in raw_frames}
    raw, labels = {}, {}
    for i, src in enumerate(kept):
        s = src.folder
        m = merged[merged["episode_index"] == i]
        r = raw_frames[s][raw_frames[s]["episode_index"] == src.episode]
        if src.length is not None:
            r = r.iloc[src.start: src.start + src.length]
        if len(m) != len(r):
            raise RuntimeError(f"episode {i} ({src.name()}): {len(m)} frames, raw has {len(r)}")
        rs, ra = np.stack(r["observation.state"].to_numpy()), np.stack(r["action"].to_numpy())
        if not (np.array_equal(np.stack(m["observation.state"].to_numpy()), rs[:, MODEL_DIM_INDICES])
                and np.array_equal(np.stack(m["action"].to_numpy()), ra[:, MODEL_DIM_INDICES])):
            raise RuntimeError(f"episode {i} ({src.name()}): joint data differs from the raw recording")
        if s in spec.relabel:
            expected = [spec.relabel[s]] * len(r)
        elif src.part:
            expected = [FLAP_PROMPTS[src.part]] * len(r)
        else:
            expected = [raw_tasks[s][t] for t in r["task_index"]]
        got = [tasks[t] for t in m["task_index"]]
        if got != expected:
            raise RuntimeError(f"episode {i} ({src.name()}): labels differ from the expected prompts")
        if len(set(got)) != 1:
            raise RuntimeError(f"episode {i} ({src.name()}) has several labels; list its folder under split_by_label")
        raw[src] = (rs, ra)
        labels[src] = got[0]
    log.info("All %d episodes match their raw recordings (frames, joints, labels)", len(kept))
    check_robot_facts(spec, raw, labels)

    # camera frames: first, middle and last frame of the first, middle and last kept episode of each folder
    worst = 0.0
    for s in raw_frames:
        mine = [i for i, src in enumerate(kept) if src.folder == s]
        for i in sorted({mine[0], mine[len(mine) // 2], mine[-1]}):
            src = kept[i]
            a = LeRobotDataset(spec.repo_id, root=out, episodes=[i])
            b = LeRobotDataset(local_repo_id(s), root=spec.raw_dir / s, episodes=[src.episode])
            for k in [0, len(a) // 2, len(a) - 1]:
                sa, sb = a[k], b[src.start + k]
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


def log_flap_balance(spec: DatasetSpec, out: Path, train: list[int]) -> None:
    """How often training will draw each flap's frames (train_with_heldout.py, flap_balance_exponent)."""
    data = pd.concat([pq.read_table(f, columns=["episode_index", "task_index"]).to_pandas()
                      for f in sorted((out / "data").glob("*/*.parquet"))], ignore_index=True)
    data = data[data["episode_index"].isin(train)]
    prompts = {int(r.task_index): t for t, r in pq.read_table(out / "meta/tasks.parquet").to_pandas().iterrows()}
    flap = data["task_index"].map(lambda t: FLAP_OF_PROMPT.get(prompts[int(t)], prompts[int(t)]))
    frames = flap.value_counts().to_dict()
    episodes = data.groupby(flap)["episode_index"].nunique().to_dict()
    shares = flap_shares(frames, spec.flap_balance_exponent)
    total = sum(frames.values())
    log.info("Training draws per flap (flap_balance_exponent %g; 1 = every frame equally often):",
             spec.flap_balance_exponent)
    for f in [f for f in spec.flap_order if f in frames] + sorted(set(frames) - set(spec.flap_order)):
        log.info("  %-5s %3d episodes, %6d frames (%5.1f min) = %4.1f%% of the frames -> %4.1f%% of the draws "
                 "(each frame %.2fx as often as with exponent 1)", f, episodes[f], frames[f], frames[f] / FPS / 60,
                 100 * frames[f] / total, 100 * shares[f], shares[f] / (frames[f] / total))


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
    unknown = set(spec.relabel) | set(spec.drop) | set(spec.split_by_label)
    if unknown - set(spec.subsets):
        raise SystemExit(f"relabel/drop/split_by_label name folders that are not in subsets: {sorted(unknown - set(spec.subsets))}")
    if set(spec.relabel) & set(spec.split_by_label):
        raise SystemExit(f"a folder cannot be both relabelled and split by its labels: {sorted(set(spec.relabel) & set(spec.split_by_label))}")
    for s, n in zip(spec.subsets, counts):
        missing = [drop_name(e, part) for e, part in spec.drop.get(s, {}) if e >= n]
        missing += [str(e) for e in spec.split_by_label.get(s, {}) if e >= n]
        if missing:
            raise SystemExit(f"{s} has {n} episodes, cannot drop or cut {missing}")

    # merged episode i comes from source[i]
    source = [Source(s, e) for s, n in zip(spec.subsets, counts) for e in range(n)]

    staging = out.parent / f"_staging_{out.name}"
    if staging.exists():
        shutil.rmtree(staging)
    log.info("Merging %d folders (%d episodes) -> %s", len(roots), len(source), staging)
    aggregate_datasets(
        repo_ids=[local_repo_id(s) for s in spec.subsets],
        aggr_repo_id=spec.repo_id,
        roots=roots,
        aggr_root=staging,
    )

    if spec.split_by_label:
        source = split_episodes_by_label(staging, spec, source)

    if spec.relabel:
        log.info("Relabelling %s", spec.relabel)
        modify_tasks(LeRobotDataset(spec.repo_id, root=staging),
                     episode_tasks={i: spec.relabel[src.folder] for i, src in enumerate(source) if src.folder in spec.relabel})

    log.info("Slicing state/action to right-arm joints %s", MODEL_JOINT_NAMES)
    slice_vector_features(staging, MODEL_DIM_INDICES, MODEL_JOINT_NAMES)

    reasons = {i: spec.drop[src.folder].get((src.episode, None)) or spec.drop[src.folder].get((src.episode, src.part))
               for i, src in enumerate(source) if src.folder in spec.drop}
    drop = [i for i, why in reasons.items() if why]
    unused = [drop_name(*k) for s, eps in spec.drop.items() for k in eps
              if not any(src.folder == s and k in {(src.episode, None), (src.episode, src.part)} for src in source)]
    if unused:
        raise SystemExit(f"drop names pieces that do not exist: {unused}")
    if drop:
        for i in drop:
            log.info("Dropping %s: %s", source[i].name(), reasons[i])
        drop_episodes_keep_videos(staging, drop)
        recompute_stats(LeRobotDataset(spec.repo_id, root=staging))
    kept = [src for i, src in enumerate(source) if i not in set(drop)]

    # held-out split: the last N kept episodes of each flap of each folder; one training episode per
    # flap and folder (the middle one) is the "train probe" that eval_open_loop.py --train-probe scores
    flap = {i: FLAP_OF_PROMPT.get(t, t) for i, t in episode_prompts(staging).items()}
    val, probe, ranges = [], [], {}
    for s in spec.subsets:
        mine = [i for i, src in enumerate(kept) if src.folder == s]
        for f in dict.fromkeys(flap[i] for i in mine):
            group = [i for i in mine if flap[i] == f]
            train_part = group[: len(group) - spec.heldout_per_subset]
            if not train_part:
                raise SystemExit(f"{s}: only {len(group)} kept {f} episode(s), all of them held out")
            val += group[len(group) - spec.heldout_per_subset:]
            probe.append(train_part[len(train_part) // 2])
        ranges[s] = [mine[0], mine[-1]]
    val, probe = sorted(val), sorted(probe)
    split = {
        "train": [i for i in range(len(kept)) if i not in val],
        "val": val,
        "train_probe": probe,
        "source_episode_ranges": ranges,
        "episodes": [{"episode": i, "folder": src.folder, "recorded_episode": src.episode}
                     | ({"part": src.part, "recorded_frames": [src.start, src.start + src.length]} if src.part else {})
                     for i, src in enumerate(kept)],
        "dropped": {s: {drop_name(*k): why for k, why in eps.items()} for s, eps in spec.drop.items()},
        "built_from": str(args.spec),
    }
    (staging / SPLIT_FILE).write_text(json.dumps(split, indent=2))

    out.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(out)
    log.info("Validating %s", out)
    validate(spec, out, kept)
    log_flap_balance(spec, out, split["train"])
    log.info("Done. %d episodes (%d dropped): train=%d, held out=%s, train probe=%s",
             len(kept), len(drop), len(split["train"]), val, probe)


if __name__ == "__main__":
    main()
