"""The text channel's bit cost is the load-bearing equation in the thesis.
Fixing it now in a test prevents drift later.
"""
from budgetfair.comm.text_channel import TextChannel


def test_ascii_message():
    """ASCII-only messages: 1 byte per character."""
    assert TextChannel.bit_cost("hello") == 5 * 8


def test_unicode_message():
    """UTF-8: Arabic letters take 2 bytes each."""
    assert TextChannel.bit_cost("مرحبا") == 10 * 8  # 5 Arabic letters x 2 bytes


def test_empty_message():
    assert TextChannel.bit_cost("") == 0
