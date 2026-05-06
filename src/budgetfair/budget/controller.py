"""BudgetController tracks cumulative bits per episode and enforces the cap B.

Policy when a message would exceed the cap (configurable):
  - "block":    refuse the message, return failure to the sender
  - "truncate": for text, cut off after the bit limit; for vector, transmit truncated form
  - "compress": (vector only) re-quantize to fewer bits per component until it fits

The within-subjects experimental design requires C_text <= B AND C_vector <= B per episode,
so the controller state is reset at episode boundaries.
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
            return 2**62
        return max(0, self.cap_bits - self.spent_bits)


class BudgetController:
    def __init__(self, cap_bits: int | None, policy: OverbudgetPolicy = OverbudgetPolicy.BLOCK) -> None:
        self.cap_bits = cap_bits
        self.policy = policy
        self.state = BudgetState(cap_bits=cap_bits)

    def reset(self) -> None:
        self.state = BudgetState(cap_bits=self.cap_bits)

    def attempt_spend(self, bits: int) -> bool:
        """Returns True if the spend is allowed (or compressed/truncated to fit)."""
        raise NotImplementedError
