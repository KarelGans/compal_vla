#!/usr/bin/env python
"""30 Hz closed-loop control: cameras + joints -> policy (pi0.5, ACT, ...) -> joint-position commands.

Fill in `IgusRebelRobot` with your driver (ROS 2 topics, CRI, ...), then:

    source env.sh   (from the project root)
    uv run python scripts/run_robot.py --checkpoint <run>/checkpoints/last/pretrained_model --order back front right left

While running, type `n` + Enter to advance to the next flap; `q` + Enter stops.

--dry-run replays a recorded validation episode as a fake robot: observations come from
the dataset, commands are only logged and compared with the recorded actions. That runs
the whole stack (prompting, queueing, latency, 14-D command assembly) without hardware.
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from open_box_vla.constants import CAMERA_KEYS, FPS, MODEL_DIM_INDICES
from open_box_vla.flap_sequencer import FlapSequencer
from open_box_vla.policy_runner import FlapPolicy


class RobotInterface(Protocol):
    def get_observation(self) -> dict:
        """{'head': HxWx3 uint8 RGB, 'wrist_right': HxWx3 uint8 RGB, 'joints': (14,) float32 rad}.

        Same camera placement, resolution (640x480) and joint order as the recordings.
        """

    def send_joint_command(self, joints14: np.ndarray) -> None:
        """Joint-position targets in the dataset's 14-D order (see constants.FULL_JOINT_NAMES)."""


class IgusRebelRobot:
    """TODO: connect to the real igus ReBeL dual-arm setup."""

    def __init__(self):
        raise NotImplementedError("Implement get_observation/send_joint_command for your driver")

    def get_observation(self) -> dict: ...

    def send_joint_command(self, joints14: np.ndarray) -> None: ...


class ReplayRobot:
    """Fake robot that replays a dataset episode (for --dry-run)."""

    def __init__(self, repo_id: str, root: Path, episode: int, start: int = 0):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.ds = LeRobotDataset(repo_id, root=root, episodes=[episode])
        self.k = start
        self.sent, self.recorded = [], []

    @property
    def task(self) -> str:
        return self.ds[min(self.k, len(self.ds) - 1)]["task"]

    def get_observation(self) -> dict:
        s = self.ds[min(self.k, len(self.ds) - 1)]
        img = lambda t: (t.permute(1, 2, 0) * 255).round().byte().numpy()  # noqa: E731
        joints = np.zeros(14, dtype=np.float32)
        joints[MODEL_DIM_INDICES] = s["observation.state"].numpy()
        self._last_action = s["action"].numpy()
        return {"head": img(s[CAMERA_KEYS[0]]), "wrist_right": img(s[CAMERA_KEYS[1]]), "joints": joints}

    def send_joint_command(self, joints14: np.ndarray) -> None:
        self.sent.append(joints14[MODEL_DIM_INDICES].copy())
        self.recorded.append(self._last_action.copy())
        self.k += 1


def stdin_commands() -> "queue.Queue[str]":
    q: queue.Queue[str] = queue.Queue()

    def reader():
        for line in sys.stdin:
            q.put(line.strip().lower())

    threading.Thread(target=reader, daemon=True).start()
    return q


def clamp_step(cmd: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    """Never command a right-arm target further than `max_delta` rad from the measured pose."""
    out = cmd.copy()
    idx = MODEL_DIM_INDICES
    out[idx] = current[idx] + np.clip(cmd[idx] - current[idx], -max_delta, max_delta)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--order", nargs="+", default=["back", "front", "right", "left"])
    p.add_argument("--hz", type=float, default=FPS)
    p.add_argument("--n-action-steps", type=int, default=25,
                   help="steps executed per predicted chunk (pi0.5 predicts 50, ACT 100) before re-planning")
    p.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF",
                   help="ACT only: predict every tick and average overlapping chunks (ACT paper: 0.01); "
                        "replaces --n-action-steps")
    p.add_argument("--max-joint-delta", type=float, default=0.15,
                   help="rad; recorded |action - state| peaks at ~0.13")
    p.add_argument("--timeout-s", type=float, default=90.0, help="auto-advance a flap after this long")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dry-run-episode", type=int, default=None, help="default: first val episode")
    p.add_argument("--dry-run-steps", type=int, default=300)
    args = p.parse_args()

    policy = FlapPolicy(args.checkpoint, n_action_steps=args.n_action_steps,
                        temporal_ensemble_coeff=args.temporal_ensemble)

    if args.dry_run:
        root = Path(os.environ["DATASET_ROOT"])
        ep = args.dry_run_episode
        if ep is None:
            ep = json.loads((root / "meta/open_box_split.json").read_text())["val"][0]
        robot = ReplayRobot(os.environ["DATASET_REPO_ID"], root, ep)
        latencies = []
        prompt = None
        for _ in range(args.dry_run_steps):
            obs = robot.get_observation()
            if robot.task != prompt:          # follow the recorded flap labels
                prompt = robot.task
                policy.reset()
            t0 = time.perf_counter()
            cmd = policy.step(obs["head"], obs["wrist_right"], obs["joints"], prompt)
            latencies.append(time.perf_counter() - t0)
            robot.send_joint_command(clamp_step(cmd, obs["joints"], args.max_joint_delta))
        err = np.abs(np.stack(robot.sent) - np.stack(robot.recorded))
        lat = np.array(latencies) * 1000
        print(f"dry run: episode {ep}, {len(lat)} ticks, prompt at end: {prompt!r}")
        print(f"  |command - recorded action| mean {err.mean():.4f} rad, max {err.max():.4f} rad")
        print(f"  step latency ms: median {np.median(lat):.1f}, p95 {np.percentile(lat, 95):.1f}, "
              f"max {lat.max():.1f} (max = a fresh chunk inference)")
        return

    robot: RobotInterface = IgusRebelRobot()
    seq = FlapSequencer(args.order, timeout_s=args.timeout_s)
    commands = stdin_commands()
    period = 1.0 / args.hz
    print(f"flap 1/{len(seq.order)}: {seq.prompt!r}   (n + Enter = next flap, q + Enter = stop)")
    while not seq.done:
        tick = time.perf_counter()
        changed = seq.update()
        while not commands.empty():
            c = commands.get()
            if c == "q":
                return
            if c == "n":
                changed = seq.advance() or changed
        if changed:
            policy.reset()
            if seq.done:
                break
            print(f"-> {seq.prompt!r}")
        obs = robot.get_observation()
        cmd = policy.step(obs["head"], obs["wrist_right"], obs["joints"], seq.prompt)
        robot.send_joint_command(clamp_step(cmd, obs["joints"], args.max_joint_delta))
        time.sleep(max(0.0, period - (time.perf_counter() - tick)))
    print("all flaps done")


if __name__ == "__main__":
    main()
