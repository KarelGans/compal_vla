"""Facts about the recorded mocap dataset that the rest of the code relies on.

Checked against /media/ai/aipc/Karel/Mocap/Dataset (LeRobot v3.0, 4 subsets, 41 episodes):
- state/action are 14-D (left arm 6 + gripper, right arm 6 + gripper).
- The left arm and both grippers never move: left arm and gripper state are always 0,
  and gripper actions are constant (left = 0, right = 1). Only the 6 right-arm joints carry signal.
- action[t] == observation.state[t + 7] (a constant 7-frame / 0.23 s position-command lead).
- Per-frame task strings are the *subtask* (one flap). The four subsets use different
  flap orders and different task_index numbering, so tasks must be matched by string.
"""

FPS = 30

SOURCE_SUBSETS = [
    "new_mocap_allflaps1",
    "new_mocap_allflaps2",
    "new_mocap_allflaps3",
    "new_mocap_allflaps4",
]

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

# Values the recorded data holds constant in the dims the model does not predict.
# Used to rebuild a full 14-D command for the robot driver.
CONSTANT_ACTION_DIMS = {
    6: 0.0,   # arm_left_gripper
    13: 1.0,  # arm_right_gripper
}

CAMERA_KEYS = [
    "observation.images.head",
    "observation.images.wrist_right",
]

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


# Orders that appear in the recordings (subset -> order). Any order is valid at
# inference since every frame is conditioned on its own flap prompt.
RECORDED_FLAP_ORDERS = {
    "new_mocap_allflaps1": ["back", "front", "right", "left"],
    "new_mocap_allflaps2": ["front", "back", "right", "left"],
    "new_mocap_allflaps3": ["front", "back", "left", "right"],
    "new_mocap_allflaps4": ["back", "front", "left", "right"],
}

# Subtask durations seen in the data (frames @ 30 fps): min ~670, median ~1050, max ~2180.
SUBTASK_TIMEOUT_S = 90.0

ACTION_LEAD_FRAMES = 7
