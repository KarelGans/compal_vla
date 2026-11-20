#!/usr/bin/env python
"""Open-loop evaluation on held-out episodes.

At every `--stride` frames of each validation episode, feed the recorded observation
(both cameras + right-arm state + flap prompt) to the policy and compare the first
`--horizon` (50) predicted steps against the recorded actions. Reported as mean absolute error
in radians, next to a "hold current pose" baseline the model has to beat. The horizon is the same
for every model (ACT predicts 100 steps, pi0.5 50), so their scores can be compared directly.
Scores are reported per episode, per flap and overall.

--prompt-swap also gives the model every query with the three wrong flap prompts (same sampling
noise), to test whether it follows the prompt: the error should be lowest with the right prompt.
A model that ignores the prompt and goes by the images alone scores about the same with every
prompt, and the right prompt fits best only ~25% of the time (chance).

Overfitting check: score every saved checkpoint on the held-out episodes and on training episodes
(--episodes 5 15 25 35). --trackio adds the scores to the training run's Trackio dashboard at the
checkpoint's step (eval/heldout_mae, eval/train_episodes_mae), next to train/loss. If heldout_mae
stops falling while train_episodes_mae keeps falling, the model is memorizing its training episodes.

This uses the same FlapPolicy code path as the robot, so it doubles as a check of the
deployment wrapper. Open-loop error is a sanity check, not a success-rate predictor.
Works for any model folder (pi05, act); the model type is read from the checkpoint.

Usage:
    source env.sh   (from the project root)
    uv run python scripts/eval_open_loop.py --checkpoint $CHECKPOINT_ROOT/<model>/<run>/checkpoints/last/pretrained_model
    uv run python scripts/eval_open_loop.py --checkpoint ... --prompt-swap     # 4x the queries
    uv run python scripts/eval_open_loop.py --checkpoint ... --episodes 5 15 25 35 --trackio
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from open_box_vla.constants import CAMERA_KEYS, FLAP_PROMPTS, FPS, MODEL_DIM_INDICES, MODEL_JOINT_NAMES
from open_box_vla.policy_runner import FlapPolicy

FLAP_OF_PROMPT = {prompt: flap for flap, prompt in FLAP_PROMPTS.items()}


def to_hwc_uint8(img: torch.Tensor) -> np.ndarray:
    return (img.permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()


def log_scores_to_trackio(summary: dict, checkpoint: Path) -> None:
    """Add the scores to the training run's Trackio dashboard, at the checkpoint's training step."""
    import trackio
    from trackio.sqlite_storage import SQLiteStorage

    ckpt = checkpoint.resolve()  # follows checkpoints/last
    if not ckpt.parent.name.isdigit():
        raise SystemExit(f"--trackio: cannot read the training step from {ckpt}")
    step = int(ckpt.parent.name)
    tr = json.loads((ckpt / "train_config.json").read_text()).get("trackio") or {}
    if tr.get("enable") and tr.get("run_name"):
        project, run = tr["project"], tr["run_name"]
    else:  # saved before Trackio was switched on: the names train.sh and resume.sh use
        project, run = f"{summary['policy']}_open_box", ckpt.parents[2].name
    split = summary["split"]
    metrics = {f"eval/{split}_mae": summary["overall"]["mae_chunk"]}
    if split == "heldout":
        metrics.update({f"eval/heldout_mae_{flap}": r["mae_chunk"] for flap, r in summary["flaps"].items()})
        if "prompt_swap" in summary:
            metrics["eval/heldout_right_prompt_best"] = float(
                np.mean([r["right_prompt_fits_best"] for r in summary["prompt_swap"].values()]))
    logged = {k for r in SQLiteStorage.get_logs(project, run, scalar_only=True) if r.get("step") == step for k in r}
    metrics = {k: v for k, v in metrics.items() if k not in logged}
    if not metrics:
        print(f"trackio: step {step} of {project}/{run} already has these scores")
        return
    trackio.init(project=project, name=run, resume="allow",
                 auto_log_gpu=False, auto_log_cpu=False, capture_console=False)
    trackio.log(metrics, step=step)
    trackio.finish()
    print(f"trackio: added {', '.join(sorted(metrics))} at step {step} to {project}/{run}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help=".../checkpoints/<step>/pretrained_model")
    p.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("DATASET_ROOT", "")))
    p.add_argument("--repo-id", default=os.environ.get("DATASET_REPO_ID", "compal/open_box_flaps_right6"))
    p.add_argument("--episodes", type=int, nargs="*", help="default: the 'val' list in meta/open_box_split.json")
    p.add_argument("--stride", type=int, default=30, help="frames between queries")
    p.add_argument("--max-queries", type=int, default=0, help="per episode, 0 = all")
    p.add_argument("--horizon", type=int, default=50, help="predicted steps scored per query (pi0.5 predicts 50)")
    p.add_argument("--out", type=Path, default=None,
                   help="default: <checkpoint>/../eval_open_loop (held-out) or eval_open_loop_train_episodes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt-swap", action="store_true",
                   help="also query every frame with the 3 wrong flap prompts (tests prompt following)")
    p.add_argument("--trackio", action="store_true",
                   help="add the scores to the training run's Trackio dashboard at this checkpoint's step")
    args = p.parse_args()

    val = json.loads((args.dataset_root / "meta/open_box_split.json").read_text())["val"]
    episodes = args.episodes or val
    split = "heldout" if set(episodes) <= set(val) else "train_episodes" if not set(episodes) & set(val) else "mixed"
    if args.trackio and split == "mixed":
        raise SystemExit("--trackio needs only held-out or only training episodes, not a mix")
    out = args.out or args.checkpoint.parent / {"heldout": "eval_open_loop"}.get(split, f"eval_open_loop_{split}")
    out.mkdir(parents=True, exist_ok=True)

    policy = FlapPolicy(args.checkpoint)
    chunk = min(args.horizon, policy.policy.config.chunk_size)
    ds = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps={"action": [i / FPS for i in range(chunk)]},
    )

    summary = {"checkpoint": str(args.checkpoint), "policy": policy.policy.config.type, "horizon": chunk,
               "split": split, "episodes": {}, "joint_names": MODEL_JOINT_NAMES}
    all_err, all_hold, all_flap, swap_err = [], [], [], []
    for ep in episodes:
        rows = [i for i in range(len(ds)) if int(ds.hf_dataset[i]["episode_index"]) == ep]
        queries = rows[:: args.stride]
        if args.max_queries:
            queries = queries[: args.max_queries]
        errs, holds, trace = [], [], []
        for i in queries:
            s = ds[i]
            joints14 = np.zeros(14, dtype=np.float32)
            joints14[MODEL_DIM_INDICES] = s["observation.state"].numpy()
            head, wrist = to_hwc_uint8(s[CAMERA_KEYS[0]]), to_hwc_uint8(s[CAMERA_KEYS[1]])
            gt = s["action"].numpy()
            valid = ~s["action_is_pad"].numpy()
            flap = FLAP_OF_PROMPT[s["task"]]
            asked = FLAP_PROMPTS if args.prompt_swap else {flap: s["task"]}
            preds = {}
            for name, prompt in asked.items():
                torch.manual_seed(args.seed + i)  # same sampling noise for every prompt of this query
                preds[name] = policy.predict_chunk(head, wrist, joints14, prompt)[:chunk]
            pred = preds[flap]
            hold = np.broadcast_to(s["observation.state"].numpy(), gt.shape)
            errs.append(np.where(valid[:, None], np.abs(pred - gt), np.nan))
            holds.append(np.where(valid[:, None], np.abs(hold - gt), np.nan))
            all_flap.append(flap)
            if args.prompt_swap:  # (4 prompts, chunk, 6)
                swap_err.append(np.stack([np.where(valid[:, None], np.abs(preds[n] - gt), np.nan) for n in FLAP_PROMPTS]))
            trace.append((int(s["frame_index"]), pred, s["task"]))
        e, h = np.stack(errs), np.stack(holds)  # (Q, chunk, 6)
        all_err.append(e)
        all_hold.append(h)
        summary["episodes"][ep] = {
            "queries": len(queries),
            "mae_first10": float(np.nanmean(e[:, :10])),
            "mae_chunk": float(np.nanmean(e)),
            "hold_mae_chunk": float(np.nanmean(h)),
        }
        print(f"episode {ep}: {summary['episodes'][ep]}")

        # plot recorded actions vs predicted chunks
        ep_actions = np.stack([ds.hf_dataset[i]["action"] for i in rows])
        fig, axes = plt.subplots(len(MODEL_JOINT_NAMES), 1, figsize=(14, 2.2 * len(MODEL_JOINT_NAMES)), sharex=True)
        t = np.arange(len(ep_actions)) / FPS
        for j, ax in enumerate(axes):
            ax.plot(t, ep_actions[:, j], color="black", lw=1, label="recorded action")
            for f0, pred, _ in trace:
                tt = (f0 + np.arange(len(pred))) / FPS
                ax.plot(tt, pred[:, j], color="tab:orange", lw=0.8, alpha=0.8)
            ax.set_ylabel(MODEL_JOINT_NAMES[j].replace("arm_right_", ""), fontsize=8)
        axes[0].plot([], [], color="tab:orange", label="predicted chunks")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle(f"episode {ep} - open-loop predicted chunks every {args.stride} frames")
        fig.tight_layout()
        fig.savefig(out / f"episode_{ep:03d}.png", dpi=110)
        plt.close(fig)

    e, h = np.concatenate(all_err), np.concatenate(all_hold)
    summary["overall"] = {
        "mae_first10": float(np.nanmean(e[:, :10])),
        "mae_chunk": float(np.nanmean(e)),
        "hold_mae_chunk": float(np.nanmean(h)),
        "mae_chunk_per_joint": np.nanmean(e, axis=(0, 1)).tolist(),
    }
    flaps = np.array(all_flap)
    summary["flaps"] = {
        name: {
            "queries": int((flaps == name).sum()),
            "mae_first10": float(np.nanmean(e[flaps == name, :10])),
            "mae_chunk": float(np.nanmean(e[flaps == name])),
            "hold_mae_chunk": float(np.nanmean(h[flaps == name])),
        }
        for name in FLAP_PROMPTS if (flaps == name).any()
    }
    if args.prompt_swap:
        names = list(FLAP_PROMPTS)
        w = np.stack(swap_err)  # (queries, 4 prompts, chunk, 6)
        summary["prompt_swap"] = {
            name: {
                "mae_chunk_given": dict(zip(names, np.nanmean(w[flaps == name], axis=(0, 2, 3)).tolist())),
                "right_prompt_fits_best": float(
                    (np.nanmean(w[flaps == name], axis=(2, 3)).argmin(1) == names.index(name)).mean()),
            }
            for name in names if (flaps == name).any()
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("overall:", json.dumps(summary["overall"], indent=2))
    print("\nper flap (mae_chunk must beat hold, the error of standing still):")
    for name, r in summary["flaps"].items():
        print(f"  {name:6s} {r['queries']:4d} queries   mae_chunk {r['mae_chunk']:.4f}   hold {r['hold_mae_chunk']:.4f}")
    if args.prompt_swap:
        print("\nprompt swap: mae_chunk when given each prompt (rows: the flap actually being opened)")
        print("        " + "".join(f"{n:>8s}" for n in names) + "   right prompt fits best (chance 25%)")
        for name, r in summary["prompt_swap"].items():
            print(f"  {name:6s}" + "".join(f"{r['mae_chunk_given'][n]:8.4f}" for n in names)
                  + f"   {r['right_prompt_fits_best']:.0%}")
    print(f"\nplots + summary.json in {out}")
    if args.trackio:
        log_scores_to_trackio(summary, args.checkpoint)


if __name__ == "__main__":
    main()
