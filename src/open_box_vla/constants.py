"""Facts shared by every recorded dataset. Per-dataset facts (folders, labels, dropped episodes,
gripper commands, finish pose) are in datasets/<name>.yaml, read through dataset_specs.py.

Checked against both recordings (LeRobot v3.0, igus ReBeL dual arm):
- state/action are 14-D (left arm 6 + gripper, right arm 6 + gripper).
- The left arm and both grippers never move: left-arm and gripper states are always 0, and the
  gripper commands are constant (values in the dataset YAML). Only the 6 right-arm joints carry signal.
- action[t] == observation.state[t + 7] (a constant 7-frame / 0.23 s position-command lead).
- Per-frame task strings are the flap being opened.
- The cameras differ: the first dataset has head + wrist_right, the "thor" one adds knife_left.
  The robot code takes the camera list from the checkpoint.
"""

FPS = 30

FULL_JOINT_NAMES = [
    "arm_left_joint1",
    "arm_left_joint2",
    "arm_left_joint3",
    "arm_left_joint4",
    "arm_left_joint5",
    "arm_left_joint6",
    "arm_left_gripper",
    "arm_right_joint1",
    "arm_right_joint2",
    "arm_right_joint3",
    "arm_right_joint4",
    "arm_right_joint5",
    "arm_right_joint6",
    "arm_right_gripper",
]

# Indices of the dims the model is trained on (right arm, no gripper).
MODEL_DIM_INDICES = list(range(7, 13))
MODEL_JOINT_NAMES = [FULL_JOINT_NAMES[i] for i in MODEL_DIM_INDICES]

FLAP_PROMPTS = {
    "back": "open the back flap",
    "front": "open the front flap",
    "right": "open the right flap",
    "left": "open the left flap",
}

# Models without a text input (ACT) get the flap as 4 numbers instead: 1 for the current flap,
# 0 for the others, in FLAP_PROMPTS order (back, front, right, left).
FLAP_ONEHOT_KEY = "observation.environment_state"


def flap_onehot(prompt: str) -> list[float]:
    prompts = list(FLAP_PROMPTS.values())
    if prompt not in prompts:
        raise ValueError(f"unknown flap prompt {prompt!r}; expected one of {prompts}")
    return [float(p == prompt) for p in prompts]


# Longest flap in the recordings: 73 s in the first dataset, 74 s in the thor dataset.
SUBTASK_TIMEOUT_S = 90.0

ACTION_LEAD_FRAMES = 7
