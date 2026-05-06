"""Budget controller correctness. Without this, the entire thesis comparison is meaningless."""
import pytest

from budgetfair.budget.controller import BudgetController, OverbudgetPolicy


def test_no_cap_allows_anything():
    c = BudgetController(cap_bits=None)
    assert c.attempt_spend(1_000_000) is True
    assert c.state.spent_bits == 1_000_000
    assert c.state.over_budget is False
    assert c.remaining_bits() > 0


def test_block_policy_refuses_overbudget():
    c = BudgetController(cap_bits=100, policy=OverbudgetPolicy.BLOCK)
    assert c.attempt_spend(50) is True
    assert c.state.spent_bits == 50
    assert c.attempt_spend(60) is False  # would exceed
    assert c.state.spent_bits == 50      # refused spend leaves state unchanged
    assert c.state.over_budget is True


def test_block_exact_fit_allowed():
    c = BudgetController(cap_bits=100, policy=OverbudgetPolicy.BLOCK)
    assert c.attempt_spend(100) is True
    assert c.state.spent_bits == 100
    assert c.attempt_spend(1) is False
    assert c.remaining_bits() == 0


def test_reset_clears_spent():
    c = BudgetController(cap_bits=100)
    c.attempt_spend(80)
    c.reset()
    assert c.state.spent_bits == 0
    assert c.state.over_budget is False


def test_negative_bits_rejected():
    c = BudgetController(cap_bits=100)
    with pytest.raises(ValueError):
        c.attempt_spend(-10)


def test_remaining_bits_tracks_spend():
    c = BudgetController(cap_bits=1000)
    assert c.remaining_bits() == 1000
    c.attempt_spend(300)
    assert c.remaining_bits() == 700
    c.attempt_spend(500)
    assert c.remaining_bits() == 200
