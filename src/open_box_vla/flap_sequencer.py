"""High-level sequencing of the four flap sub-tasks.

The dataset labels every frame with the flap currently being opened, so the model is
conditioned on one prompt at a time and does not decide when a flap is finished.
Something outside the model has to advance the prompt. This class supports an operator
(call `advance()`), a success detector (also `advance()`), and a timeout fallback.
"""

import time
from collections.abc import Callable, Sequence

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
