"""High-level sequencing of the flap sub-tasks.

The model is conditioned on one flap prompt at a time. Something outside the model advances the
prompt: an operator (`advance()`), the finish-pose detector below, or the timeout fallback.
"""

import time
from collections.abc import Callable, Sequence

import numpy as np

from .constants import FLAP_PROMPTS, SUBTASK_TIMEOUT_S


class FlapSequencer:
    def __init__(
        self,
        order: Sequence[str] = ("back", "front", "right", "left"),
        timeout_s: float | None = SUBTASK_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        unknown = [f for f in order if f not in FLAP_PROMPTS]
        if unknown:
            raise ValueError(f"Unknown flaps {unknown}; choose from {list(FLAP_PROMPTS)}")
        self.order = list(order)
        self.timeout_s = timeout_s
        self._clock = clock
        self._i = 0
        self._started = clock()

    @property
    def done(self) -> bool:
        return self._i >= len(self.order)

    @property
    def flap(self) -> str | None:
        return None if self.done else self.order[self._i]

    @property
    def prompt(self) -> str | None:
        return None if self.done else FLAP_PROMPTS[self.order[self._i]]

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._started

    def advance(self) -> bool:
        """Move to the next flap. Returns True if the prompt changed (reset the policy queue)."""
        if self.done:
            return False
        self._i += 1
        self._started = self._clock()
        return True

    def update(self) -> bool:
        """Advance on timeout. Returns True if the prompt changed."""
        if self.timeout_s is not None and not self.done and self.elapsed_s > self.timeout_s:
            return self.advance()
        return False


class FinishDetector:
    """Decides a flap is done when the right arm is back at the dataset's finish pose.

    Every thor recording ends with the arm at one fixed pose, standing still for 1-8 s, and the arm
    does not pass through that pose earlier in an episode (closest approach ~0.1 rad). Back-flap
    episodes also start there. So a flap is done once the arm, after having been more than
    `leave_rad` away from the finish pose, is within `tol_rad` of it (every joint) for `hold_s`.
    Call `update` once per control tick and `reset` when a new flap starts.
    """

    def __init__(self, finish_pose: np.ndarray, hz: float, tol_rad: float = 0.05, hold_s: float = 1.0,
                 leave_rad: float = 0.2):
        self.finish_pose = np.asarray(finish_pose, dtype=np.float32)
        self.tol_rad, self.leave_rad = tol_rad, leave_rad
        self.hold_ticks = max(1, round(hold_s * hz))
        self.reset()

    def reset(self) -> None:
        self._left = False
        self._held = 0

    def distance(self, right6: np.ndarray) -> float:
        return float(np.abs(np.asarray(right6) - self.finish_pose).max())

    def update(self, right6: np.ndarray) -> bool:
        d = self.distance(right6)
        self._left |= d > self.leave_rad
        self._held = self._held + 1 if (self._left and d <= self.tol_rad) else 0
        return self._held >= self.hold_ticks
