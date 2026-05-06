"""Budget controller correctness. Without this, the entire thesis comparison is meaningless."""
import pytest

from budgetfair.budget.controller import BudgetController, OverbudgetPolicy


@pytest.mark.skip(reason="not yet implemented")
def test_block_policy_refuses_overbudget():
    controller = BudgetController(cap_bits=100, policy=OverbudgetPolicy.BLOCK)
    assert controller.attempt_spend(50) is True
    assert controller.attempt_spend(60) is False  # would exceed
    assert controller.state.spent_bits == 50


@pytest.mark.skip(reason="not yet implemented")
def test_reset_clears_spent():
    controller = BudgetController(cap_bits=100)
    controller.attempt_spend(80)
    controller.reset()
    assert controller.state.spent_bits == 0
