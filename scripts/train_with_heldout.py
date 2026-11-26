#!/usr/bin/env python
"""lerobot-train plus a held-out loss curve, computed on the dataset's own held-out episodes,
and flap balancing.

LeRobot 0.7 can compute the training loss on held-out episodes every --eval_steps steps and log
it to Trackio as eval/eval_loss, next to train/loss. Its own split holds out the last
--dataset.eval_split fraction of each task's *training* episodes. This wrapper replaces that split
with the held-out ("val") episodes in the dataset's meta/open_box_split.json, which train.sh
already keeps out of training (--dataset.exclude_episodes). Everything else is plain lerobot-train.
A dataset built with heldout_per_subset: 0 has no held-out episodes: train it with --eval_steps=0.

Also:
- The loss is computed on every HELDOUT_STRIDE-th frame (default 5) of the held-out episodes,
  so an evaluation takes about a minute.
- The flow-matching noise and time step are fixed per held-out batch, so the curve moves only
  when the model changes, not from random sampling.
- The evaluation uses HELDOUT_WORKERS (default 2) data-loader workers that exit after each
  evaluation, instead of lerobot's persistent ones, so they hold no RAM between evaluations.

Flap balancing: when the dataset's YAML (the datasets/*.yaml with this repo_id) sets
flap_balance_exponent below 1, each epoch draws its training frames with replacement, weighted
so that every flap gets the share open_box_vla.dataset_specs.flap_shares() gives it, instead of
lerobot's shuffle. Epochs keep their length and depend only on (seed, epoch), so resuming stays
sample-exact. The shares are logged at the start. Exponent 1 or no YAML: lerobot's shuffle, unchanged.

lerobot-train refuses --eval_steps > 0 unless --dataset.eval_split > 0; the value is ignored here.
Resume such a run with this script too (pi05/resume.sh does): plain lerobot-train would
hold out the last 10% of each task's training episodes instead.

    uv run python scripts/train_with_heldout.py <lerobot-train flags> --eval_steps=1000 --dataset.eval_split=0.1
"""

import functools
import json
import logging
import os
import types
from pathlib import Path

import numpy as np
import torch
import lerobot.scripts.lerobot_train as lt
from lerobot.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.utils.collate import lerobot_collate_fn

from open_box_vla.constants import FLAP_PROMPTS
from open_box_vla.dataset_specs import flap_shares, spec_for_repo_id

STRIDE = int(os.environ.get("HELDOUT_STRIDE", "5"))
WORKERS = int(os.environ.get("HELDOUT_WORKERS", "2"))
NOISE_SEED = 1234
FLAP_OF_PROMPT = {prompt: flap for flap, prompt in FLAP_PROMPTS.items()}


class FlapBalancedSampler(EpisodeAwareSampler):
    """lerobot's EpisodeAwareSampler, except that a shuffled epoch draws len(self) frames with
    replacement, weighted by `row_weights` (one per row of the dataset), instead of a permutation.
    The draws come from the same (seed, epoch) generator and an epoch has the same length, so
    lerobot's sample-exact resume (compute_sampler_state) works unchanged."""

    def __init__(self, *args, row_weights: np.ndarray, **kwargs):
        super().__init__(*args, **kwargs)
        position = np.arange(self._num_frames)
        episode = np.searchsorted(self._cum_lengths, position, side="right")
        first = np.concatenate([[0], self._cum_lengths[:-1]])
        rows = self._starts[episode] + position - first[episode]
        if self._absolute_to_relative is not None:
            rows = np.array([self._absolute_to_relative[int(a)] for a in rows])
        self._weights = torch.as_tensor(row_weights[rows], dtype=torch.float64)

    def _iter_epoch(self, epoch: int, start: int):
        if not self.shuffle:
            yield from super()._iter_epoch(epoch, start)
            return
        draws = torch.multinomial(self._weights, self._num_frames, replacement=True,
                                  generator=self._epoch_generator(epoch))
        for k in range(start, self._num_frames):
            yield self._frame_index(int(draws[k]))


def flap_row_weights(cfg, dataset) -> np.ndarray | None:
    """One draw weight per training frame, balancing the flaps as the dataset YAML's
    flap_balance_exponent says; None means lerobot's plain shuffle (exponent 1, or no YAML)."""
    try:
        spec = spec_for_repo_id(cfg.dataset.repo_id)
    except FileNotFoundError:
        logging.warning(f"No datasets/*.yaml has repo_id {cfg.dataset.repo_id}: every frame is drawn equally often")
        return None
    if spec.flap_balance_exponent == 1.0:
        return None
    prompt_of = {int(r.task_index): t for t, r in dataset.meta.tasks.iterrows()}
    task_ids, row_task, counts = np.unique(np.asarray(dataset.hf_dataset["task_index"]),
                                           return_inverse=True, return_counts=True)
    flap_of = [FLAP_OF_PROMPT.get(prompt_of[int(t)], prompt_of[int(t)]) for t in task_ids]
    frames: dict[str, int] = {}
    for flap, n in zip(flap_of, counts):
        frames[flap] = frames.get(flap, 0) + int(n)
    shares = flap_shares(frames, spec.flap_balance_exponent)
    total = sum(frames.values())
    logging.info(f"Flap balance ({spec.path.name}, flap_balance_exponent {spec.flap_balance_exponent:g}): " + ", ".join(
        f"{flap} {100 * n / total:.1f}% of the frames -> {100 * shares[flap]:.1f}% of the draws"
        for flap, n in frames.items()))
    return np.array([shares[flap] / frames[flap] for flap in flap_of])[row_task]


def make_train_eval_datasets(cfg):
    """Training episodes as usual; the held-out episodes of meta/open_box_split.json for eval."""
    train = make_dataset(cfg)
    if cfg.eval_steps <= 0:
        return train, None
    val = json.loads((Path(cfg.dataset.root) / "meta/open_box_split.json").read_text())["val"]
    if not val:
        raise ValueError(f"{cfg.dataset.root} has no held-out episodes (built with heldout_per_subset: 0), "
                         "so there is no held-out loss: train with --eval_steps=0 (pi05/train.sh does)")
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
    weights = flap_row_weights(cfg, dataset)
    if weights is not None:  # lerobot builds its sampler through this module-level name
        lt.EpisodeAwareSampler = functools.partial(FlapBalancedSampler, row_weights=weights)
    try:
        train_loader, _ = _make_dataloaders(cfg, dataset, None, *args, **kwargs)
    finally:
        lt.EpisodeAwareSampler = EpisodeAwareSampler
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
