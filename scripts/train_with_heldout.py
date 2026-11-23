#!/usr/bin/env python
"""lerobot-train plus a held-out loss curve, computed on the dataset's own held-out episodes.

LeRobot 0.7 can compute the training loss on held-out episodes every --eval_steps steps and log
it to Trackio as eval/eval_loss, next to train/loss. Its own split holds out the last
--dataset.eval_split fraction of each task's *training* episodes. This wrapper replaces that split
with the held-out ("val") episodes in the dataset's meta/open_box_split.json, which train.sh
already keeps out of training (--dataset.exclude_episodes). Everything else is plain lerobot-train.

Also:
- The loss is computed on every HELDOUT_STRIDE-th frame (default 5) of the held-out episodes,
  so an evaluation takes about a minute.
- The flow-matching noise and time step are fixed per held-out batch, so the curve moves only
  when the model changes, not from random sampling.
- The evaluation uses HELDOUT_WORKERS (default 2) data-loader workers that exit after each
  evaluation, instead of lerobot's persistent ones, so they hold no RAM between evaluations.

lerobot-train refuses --eval_steps > 0 unless --dataset.eval_split > 0; the value is ignored here.
Resume such a run with this script too (pi05/resume.sh does): plain lerobot-train would
hold out the last 10% of each task's training episodes instead.

    uv run python scripts/train_with_heldout.py <lerobot-train flags> --eval_steps=1000 --dataset.eval_split=0.1
"""

import json
import logging
import os
import types
from pathlib import Path

import torch
import lerobot.scripts.lerobot_train as lt
from lerobot.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.collate import lerobot_collate_fn

STRIDE = int(os.environ.get("HELDOUT_STRIDE", "5"))
WORKERS = int(os.environ.get("HELDOUT_WORKERS", "2"))
NOISE_SEED = 1234


def make_train_eval_datasets(cfg):
    """Training episodes as usual; the held-out episodes of meta/open_box_split.json for eval."""
    train = make_dataset(cfg)
    if cfg.eval_steps <= 0:
        return train, None
    val = json.loads((Path(cfg.dataset.root) / "meta/open_box_split.json").read_text())["val"]
    overlap = set(val) & set(train.episodes if train.episodes is not None else range(train.meta.total_episodes))
    if overlap:
        raise ValueError(f"Held-out episodes {sorted(overlap)} are also training episodes; "
                         f"pass --dataset.exclude_episodes={val}")
    held = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=val,
        delta_timestamps=resolve_delta_timestamps(cfg.trainable_config, train.meta, cfg.rename_map),
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
    )
    frames = list(range(0, len(held), STRIDE))
    logging.info(f"Held-out loss every {cfg.eval_steps} steps on episodes {val}: {len(frames)} of their "
                 f"{len(held)} frames (every {STRIDE}th), logged to Trackio as eval/eval_loss")
    return train, torch.utils.data.Subset(held, frames)


_make_dataloaders = lt.make_dataloaders


def make_dataloaders(cfg, dataset, eval_dataset, *args, **kwargs):
    train_loader, _ = _make_dataloaders(cfg, dataset, None, *args, **kwargs)
    if eval_dataset is None:
        return train_loader, None
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None,
        prefetch_factor=cfg.prefetch_factor if WORKERS > 0 else None,
        persistent_workers=False,
        multiprocessing_context=cfg.dataloader_multiprocessing_context if WORKERS > 0 else None,
    )
    return train_loader, eval_loader


_make_policy = lt.make_policy


def make_policy(*args, **kwargs):
    """The policy, with fixed random draws (noise, time step) in eval mode: batch k of every
    evaluation uses seed NOISE_SEED + k, and the training random state is left untouched."""
    policy = _make_policy(*args, **kwargs)
    forward, set_train = policy.forward, policy.train
    batch = [0]

    def seeded_forward(self, *a, **k):
        if self.training:
            return forward(*a, **k)
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(NOISE_SEED + batch[0])
            batch[0] += 1
            return forward(*a, **k)

    def train_mode(self, mode: bool = True):
        if not mode:  # policy.eval() starts an evaluation: restart the seed sequence
            batch[0] = 0
        return set_train(mode)

    policy.forward = types.MethodType(seeded_forward, policy)
    policy.train = types.MethodType(train_mode, policy)
    return policy


lt.make_train_eval_datasets = make_train_eval_datasets
lt.make_dataloaders = make_dataloaders
lt.make_policy = make_policy

if __name__ == "__main__":
    lt.main()
