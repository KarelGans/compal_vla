"""Inference wrapper around a fine-tuned checkpoint of any LeRobot policy (pi0.5, ACT, ...).

The models are trained on the 6 right-arm joints only. This wrapper takes the robot's full
14-D joint vector, feeds the right-arm slice to the policy, and turns the predicted
right-arm targets back into a full 14-D command: the left arm holds its current pose
and the grippers get the constant commands of the dataset the checkpoint was trained on
(datasets/<name>.yaml, found through the checkpoint's train_config.json).

The cameras are the ones the checkpoint was trained with (`camera_names`; the thor dataset uses
head, wrist_right and knife_left). Pass them as a dict {camera name: HxWx3 uint8 RGB}.

The flap is given as a text prompt. Models with a text input (pi0.5) read it directly; for
models trained with the flap as 4 numbers (ACT, see scripts/add_flap_onehot.py) the wrapper
converts the prompt to those numbers.
"""

from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import get_policy_class
from lerobot.policies.utils import prepare_observation_for_inference

from open_box_vla.constants import FLAP_ONEHOT_KEY, MODEL_DIM_INDICES, flap_onehot
from open_box_vla.dataset_specs import spec_for_checkpoint

STATE_KEY = "observation.state"
IMAGE_PREFIX = "observation.images."


class FlapPolicy:
    """Policy for one flap prompt at a time.

    Args:
        checkpoint: a `.../checkpoints/<step>/pretrained_model` directory.
        n_action_steps: how many steps of each predicted chunk (pi0.5: 50, ACT: 100) to execute
            before re-planning. Smaller values react faster but query the model more often
            (pi0.5: ~0.1 s per query on the RTX PRO 6000, ACT: ~17 ms).
        temporal_ensemble_coeff: ACT only. Predict a new chunk at every step and command the
            weighted average of all chunks' predictions for that step (ACT paper: 0.01).
            Replaces n_action_steps.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cuda", n_action_steps: int | None = None,
                 temporal_ensemble_coeff: float | None = None):
        self.device = torch.device(device)
        config = PreTrainedConfig.from_pretrained(str(checkpoint))
        if temporal_ensemble_coeff is not None:
            if not hasattr(config, "temporal_ensemble_coeff"):
                raise ValueError(f"temporal ensembling is only available for ACT, not {config.type}")
            config.temporal_ensemble_coeff = temporal_ensemble_coeff
            config.n_action_steps = 1
        elif n_action_steps is not None:
            config.n_action_steps = n_action_steps
        self.policy = get_policy_class(config.type).from_pretrained(str(checkpoint), config=config)
        self.policy.to(self.device).eval()
        self.uses_flap_onehot = FLAP_ONEHOT_KEY in self.policy.config.input_features
        self.camera_keys = list(self.policy.config.image_features)
        self.camera_names = [k.removeprefix(IMAGE_PREFIX) for k in self.camera_keys]
        self.spec = spec_for_checkpoint(checkpoint)  # constant gripper commands, finish pose, flap order
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device.type}},
        )
        self.reset()

    def reset(self) -> None:
        """Drop queued actions. Call at episode start and whenever the prompt changes."""
        self.policy.reset()

    def _batch(self, images: dict[str, np.ndarray], joints14: np.ndarray, prompt: str) -> dict:
        missing = [n for n in self.camera_names if n not in images]
        if missing:
            raise KeyError(f"missing camera images {missing}; this checkpoint uses {self.camera_names}")
        # HxWx3 uint8 RGB, 480x640 like the recordings
        obs = {k: np.ascontiguousarray(images[n]) for k, n in zip(self.camera_keys, self.camera_names)}
        obs[STATE_KEY] = np.asarray(joints14, dtype=np.float32)[MODEL_DIM_INDICES]
        if self.uses_flap_onehot:
            obs[FLAP_ONEHOT_KEY] = np.asarray(flap_onehot(prompt), dtype=np.float32)
        obs = prepare_observation_for_inference(obs, self.device, task=prompt)
        return self.preprocessor(obs)

    @torch.inference_mode()
    def step(self, images: dict[str, np.ndarray], joints14: np.ndarray, prompt: str) -> np.ndarray:
        """Return the next 14-D joint-position command (call at 30 Hz)."""
        action = self.policy.select_action(self._batch(images, joints14, prompt))
        right = self.postprocessor(action).squeeze(0).float().cpu().numpy()
        return self.to_full_command(right, joints14)

    @torch.inference_mode()
    def predict_chunk(self, images: dict[str, np.ndarray], joints14: np.ndarray, prompt: str) -> np.ndarray:
        """Return the full (chunk_size, 6) right-arm target chunk without touching the queue."""
        chunk = self.policy.predict_action_chunk(self._batch(images, joints14, prompt))
        return self.postprocessor(chunk).squeeze(0).float().cpu().numpy()

    def to_full_command(self, right6: np.ndarray, joints14: np.ndarray) -> np.ndarray:
        cmd = np.asarray(joints14, dtype=np.float32).copy()  # left arm: hold current pose
        cmd[MODEL_DIM_INDICES] = right6
        for i, value in self.spec.constant_commands.items():
            cmd[i] = value
        return cmd
