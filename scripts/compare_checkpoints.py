#!/usr/bin/env python
"""Table of the open-loop scores of every evaluated checkpoint, to pick the one to try on the robot.

Reads the summary.json files that scripts/eval_open_loop.py writes next to each checkpoint
(scripts/eval_checkpoints.sh runs it for a whole run, then prints this table).

    source env.sh   (from the project root)
    uv run python scripts/compare_checkpoints.py pi05/pi05_open_box_20261114_1239 act/act_baseline

Columns (radians, first 50 predicted steps, lower is better):
  held-out     error on the held-out episodes; pick the checkpoint where this is lowest
  vs still     held-out error relative to a robot that stands still (-30% = 30% better)
  train eps    error on one training episode per recording folder; far below held-out = memorizing
  prompt       how often the right flap prompt fits the recording best (--prompt-swap runs only;
               chance is 1 / number of prompts: 25% for the four-flap dataset, 50% for front/back)
  back..left   held-out error per flap ("-": the dataset has no such flap)
A dataset built with heldout_per_subset: 0 has no held-out episodes: the table then shows the
training episodes (with the prompt test), and marks no checkpoint as the one to pick.
"""

import argparse
import json
import os
from pathlib import Path

FLAPS = ["back", "front", "right", "left"]


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="<model>/<run> under $CHECKPOINT_ROOT, or a run folder")
    args = p.parse_args()
    root = Path(os.environ["CHECKPOINT_ROOT"])

    for name in args.runs:
        run = Path(name) if (Path(name) / "checkpoints").is_dir() else root / name
        rows = []
        for step_dir in sorted((run / "checkpoints").glob("[0-9]*")):
            held = load(step_dir / "eval_open_loop/summary.json")
            train = load(step_dir / "eval_open_loop_train_episodes/summary.json")
            if held or train:
                rows.append((int(step_dir.name), held, train))
        print(f"\n{name}")
        if not rows:
            print("  no evaluated checkpoints yet (run scripts/eval_checkpoints.sh)")
            continue
        heldout = any(held for _, held, _ in rows)
        if heldout:  # main score: the held-out episodes
            rows = [(step, held, train) for step, held, train in rows if held]
        else:  # dataset without held-out episodes (heldout_per_subset: 0): the training episodes
            rows = [(step, train, None) for step, _, train in rows]
            print("  no held-out episodes: every score is on training episodes, so it shows how closely the model\n"
                  "  follows its own recordings, not how it does on new ones; the prompt column still tests the prompt")
        best = min(rows, key=lambda r: r[1]["overall"]["mae_chunk"])[0] if heldout else None
        print(f"  {'step':>7} {'held-out' if heldout else 'train eps':>9} {'vs still':>9} "
              + (f"{'train eps':>10} " if heldout else "") + f"{'prompt':>7}  " + " ".join(f"{f:>6}" for f in FLAPS))
        for step, main, train in rows:
            o = main["overall"]
            swap = main.get("prompt_swap")
            prompt = (f"{sum(r['right_prompt_fits_best'] for r in swap.values()) / len(swap):.0%}"
                      if swap else "-")
            flaps = main.get("flaps", {})
            train_mae = (f"{train['overall']['mae_chunk']:.3f}" if train else "-") if heldout else ""
            print(f"  {step:>7} {o['mae_chunk']:9.3f} {o['mae_chunk'] / o['hold_mae_chunk'] - 1:+9.0%} "
                  + (f"{train_mae:>10} " if heldout else "") + f"{prompt:>7}  "
                  + " ".join(f"{flaps[f]['mae_chunk']:6.3f}" if f in flaps else f"{'-':>6}" for f in FLAPS)
                  + ("   <- lowest held-out error" if step == best else ""))
        print(f"  standing still: {rows[0][1]['overall']['hold_mae_chunk']:.3f} on the "
              f"{'held-out' if heldout else 'training'} episodes")
        swaps = [r[1]["prompt_swap"] for r in rows if r[1].get("prompt_swap")]
        if swaps:
            n_prompts = len(next(iter(swaps[-1].values()))["mae_chunk_given"])
            print(f"  prompt: chance level is {1 / n_prompts:.0%} ({n_prompts} prompts)")


if __name__ == "__main__":
    main()
