"""BudgetController tracks cumulative bits per episode and enforces the cap B.

Policy when a message would exceed the cap (configurable):
  - "block":    refuse the message; sender records a failed transmission.
  - "truncate": let the *channel* pre-fit the message using `remaining_bits()` and then
                spend; the controller does not truncate by itself because it does not see
                the message contents.
  - "compress": same — the vector channel pre-fits via re-quantization before spending.

For TRUNCATE/COMPRESS, the contract is: the channel calls `remaining_bits()`, shrinks
the payload to fit, then calls `attempt_spend(actual_bits)`. The controller treats
all three policies identically at the spend boundary — the difference is in what the
channel does *before* asking.

The within-subjects experimental design requires C_text <= B AND C_vector <= B per
episode, so the controller's state is reset at episode boundaries via `reset()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OverbudgetPolicy(str, Enum):
    BLOCK = "block"
    TRUNCATE = "truncate"
    COMPRESS = "compress"


@dataclass
class BudgetState:
    cap_bits: int | None
    spent_bits: int = 0
    over_budget: bool = False

    @property
    def remaining(self) -> int:
        if self.cap_bits is None:
            return 2**62  # effectively unlimited; large enough that no real message hits it
        return max(0, self.cap_bits - self.spent_bits)


class BudgetController:
    def __init__(
        self,
        cap_bits: int | None,
        policy: OverbudgetPolicy = OverbudgetPolicy.BLOCK,
    ) -> None:
        if cap_bits is not None and (isinstance(cap_bits, bool) or not isinstance(cap_bits, int) or cap_bits < 0):
            raise ValueError("cap_bits must be a non-negative integer or None")
        if not isinstance(policy, OverbudgetPolicy):
            policy = OverbudgetPolicy(policy)
        self.cap_bits = cap_bits
        self.policy = policy
        self.state = BudgetState(cap_bits=cap_bits)

    def reset(self) -> None:
        """Reset to start a new episode."""
        self.state = BudgetState(cap_bits=self.cap_bits)

    def remaining_bits(self) -> int:
        """How many more bits this episode may spend. Channels call this when fitting payload."""
        return self.state.remaining

    def attempt_spend(self, bits: int) -> bool:
        """Try to spend `bits`. Returns True if the spend was recorded.

        - With cap=None: always succeeds.
        - Otherwise: succeeds iff (spent + bits) <= cap. On failure the controller marks
          `over_budget = True` but does not change `spent_bits` (the would-be spend never
          happened).
        """
        if isinstance(bits, bool) or not isinstance(bits, int) or bits < 0:
            raise ValueError(f"bits must be non-negative, got {bits}")
        if self.cap_bits is None:
            self.state.spent_bits += bits
            return True
        if self.state.spent_bits + bits > self.cap_bits:
            self.state.over_budget = True
            return False
        self.state.spent_bits += bits
        return True
