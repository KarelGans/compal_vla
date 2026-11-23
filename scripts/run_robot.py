#!/usr/bin/env python
"""30 Hz closed-loop control: cameras + joints -> policy (pi0.5, ACT, ...) -> joint-position commands.

Fill in `IgusRebelRobot` with your driver (ROS 2 topics, CRI, ...), then:

    source env.sh   (from the project root)
    uv run python scripts/run_robot.py --checkpoint <run>/checkpoints/last/pretrained_model

The flap order defaults to the dataset's (thor: front, then back). A flap ends when
- the arm is back at the dataset's finish pose and stays there 1 s (models trained on the thor
  dataset, where every recording ends at that pose; --no-auto-finish turns it off),
- the operator types `n` + Enter, or
- --timeout-s passes.
`q` + Enter stops. After the last flap, or on `q`, the arm is told to hold its current pose.

--dry-run replays a recorded held-out episode as a fake robot: observations come from the
dataset, commands are only logged and compared with the recorded actions. That runs the whole
stack (prompting, queueing, latency, 14-D command assembly, finish detection) without hardware.
"""

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from open_box_vla.constants import FPS, MODEL_DIM_INDICES
from open_box_vla.dataset_specs import checkpoint_dataset
from open_box_vla.flap_sequencer import FinishDetector, FlapSequencer
from open_box_vla.policy_runner import IMAGE_PREFIX, FlapPolicy


class RobotInterface(Protocol):
    def get_observation(self) -> dict:
        """{'images': {camera name: HxWx3 uint8 RGB}, 'joints': (14,) float32 rad}.

        One image per camera the checkpoint uses (FlapPolicy.camera_names; thor: head, wrist_right,
        knife_left), with the same placement and resolution (640x480) as the recordings.
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

    def __init__(self, repo_id: str, root: Path, episode: int):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.ds = LeRobotDataset(repo_id, root=root, episodes=[episode])
        self.k = 0
        self.sent, self.recorded = [], []

    def get_observation(self) -> dict:
        s = self.ds[min(self.k, len(self.ds) - 1)]
        img = lambda t: (t.permute(1, 2, 0) * 255).round().byte().numpy()  # noqa: E731
        joints = np.zeros(14, dtype=np.float32)
        joints[MODEL_DIM_INDICES] = s["observation.state"].numpy()
        self.task = s["task"]
        self._last_action = s["action"].numpy()
        images = {k.removeprefix(IMAGE_PREFIX): img(s[k]) for k in self.ds.meta.camera_keys}
        return {"images": images, "joints": joints}

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


def dry_run(args, policy: FlapPolicy, finish: FinishDetector | None) -> None:
    repo_id, root = checkpoint_dataset(args.checkpoint)
    root = args.dataset_root or root
    ep = args.dry_run_episode
    if ep is None:
        ep = json.loads((root / "meta/open_box_split.json").read_text())["val"][0]
    robot = ReplayRobot(repo_id, root, ep)
    n = len(robot.ds) if args.dry_run_steps == 0 else min(args.dry_run_steps, len(robot.ds))
    latencies, prompt, fired = [], None, None
    for k in range(n):
        obs = robot.get_observation()
        if robot.task != prompt:          # follow the recorded flap labels
            prompt = robot.task
            policy.reset()
            if finish:
                finish.reset()
        if finish and fired is None and finish.update(obs["joints"][MODEL_DIM_INDICES]):
            fired = k
        t0 = time.perf_counter()
        cmd = policy.step(obs["images"], obs["joints"], prompt)
        latencies.append(time.perf_counter() - t0)
        robot.send_joint_command(clamp_step(cmd, obs["joints"], args.max_joint_delta))
    err = np.abs(np.stack(robot.sent) - np.stack(robot.recorded))
    lat = np.array(latencies) * 1000
    print(f"dry run: episode {ep} ({len(robot.ds) / FPS:.1f} s), {n} ticks, cameras {policy.camera_names}, "
          f"prompt at end: {prompt!r}")
    print(f"  |command - recorded action| mean {err.mean():.4f} rad, max {err.max():.4f} rad")
    if n == len(robot.ds) and finish:
        last = err[-FPS:].mean()
        print(f"  last second (arm resting at the finish pose): |command - recorded| mean {last:.4f} rad, "
              f"distance of the commands from the finish pose max "
              f"{max(finish.distance(c) for c in robot.sent[-FPS:]):.4f} rad")
    if finish:
        print("  finish detector: " + (f"fired at {fired / FPS:.1f} s" if fired is not None else "did not fire")
              + f" (recording ends at {len(robot.ds) / FPS:.1f} s)")
    print(f"  step latency ms: median {np.median(lat):.1f}, p95 {np.percentile(lat, 95):.1f}, "
          f"max {lat.max():.1f} (max = a fresh chunk inference)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--order", nargs="+", default=None, help="flaps in order; default: the dataset's flap_order")
    p.add_argument("--hz", type=float, default=FPS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-action-steps", type=int, default=25,
                   help="steps executed per predicted chunk (pi0.5 predicts 50, ACT 100) before re-planning")
    p.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF",
                   help="ACT only: predict every tick and average overlapping chunks (ACT paper: 0.01); "
                        "replaces --n-action-steps")
    p.add_argument("--max-joint-delta", type=float, default=0.15,
                   help="rad; recorded |action - state| peaks at ~0.13")
    p.add_argument("--timeout-s", type=float, default=90.0, help="auto-advance a flap after this long")
    p.add_argument("--no-auto-finish", action="store_true",
                   help="don't end a flap when the arm is back at the finish pose")
    p.add_argument("--finish-tol", type=float, default=0.05, help="rad, every right-arm joint")
    p.add_argument("--finish-hold-s", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dataset-root", type=Path, default=None, help="dry run; default: the checkpoint's dataset")
    p.add_argument("--dry-run-episode", type=int, default=None, help="default: first held-out episode")
    p.add_argument("--dry-run-steps", type=int, default=300, help="0 = the whole episode")
    args = p.parse_args()

    policy = FlapPolicy(args.checkpoint, device=args.device, n_action_steps=args.n_action_steps,
                        temporal_ensemble_coeff=args.temporal_ensemble)
    spec = policy.spec
    finish = None
    if spec.finish_pose is not None and not args.no_auto_finish:
        finish = FinishDetector(spec.finish_pose, args.hz, tol_rad=args.finish_tol, hold_s=args.finish_hold_s)

    if args.dry_run:
        dry_run(args, policy, finish)
        return

    robot: RobotInterface = IgusRebelRobot()
    seq = FlapSequencer(args.order or spec.flap_order, timeout_s=args.timeout_s)
    commands = stdin_commands()
    period = 1.0 / args.hz

    def hold(obs: dict) -> None:
        robot.send_joint_command(policy.to_full_command(obs["joints"][MODEL_DIM_INDICES], obs["joints"]))

    def announce(obs: dict) -> None:
        print(f"flap {seq.order.index(seq.flap) + 1}/{len(seq.order)}: {seq.prompt!r}"
              "   (n + Enter = next flap, q + Enter = stop)")
        start = spec.start_poses.get(seq.flap)
        if start is not None:
            d = float(np.abs(obs["joints"][MODEL_DIM_INDICES] - start).max())
            if d > 0.1:
                print(f"  note: the arm is {d:.2f} rad from where the recorded {seq.flap}-flap episodes start")

    obs = robot.get_observation()
    announce(obs)
    while True:
        tick = time.perf_counter()
        obs = robot.get_observation()
        changed = seq.update()
        if changed:
            print("  timeout")
        while not commands.empty():
            c = commands.get()
            if c == "q":
                hold(obs)
                print("stopped, holding the current pose")
                return
            if c == "n":
                changed = seq.advance() or changed
        if not changed and finish is not None and finish.update(obs["joints"][MODEL_DIM_INDICES]):
            print(f"  back at the finish pose for {args.finish_hold_s:.0f} s: {seq.flap} flap done")
            changed = seq.advance()
        if changed:
            policy.reset()
            if finish is not None:
                finish.reset()
            if seq.done:
                break
            announce(obs)
        cmd = policy.step(obs["images"], obs["joints"], seq.prompt)
        robot.send_joint_command(clamp_step(cmd, obs["joints"], args.max_joint_delta))
        time.sleep(max(0.0, period - (time.perf_counter() - tick)))
    hold(obs)
    print("all flaps done, holding the current pose")


if __name__ == "__main__":
    main()
