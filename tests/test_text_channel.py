"""Encode/decode roundtrip + bit-cost correctness for the text channel."""
from budgetfair.comm.text_channel import TextChannel


def test_ascii_roundtrip():
    ch = TextChannel()
    msg = ch.encode("Hello world", "agent_a", "agent_b")
    assert msg.bit_cost == 11 * 8
    assert msg.sender_id == "agent_a"
    assert msg.receiver_id == "agent_b"
    assert msg.metadata["channel"] == "text"
    assert msg.metadata["text_len_chars"] == 11
    assert ch.decode(msg) == "Hello world"


def test_arabic_roundtrip():
    """UTF-8 multi-byte codepoints — the bit cost must reflect bytes, not chars."""
    ch = TextChannel()
    msg = ch.encode("مرحبا", "a", "b")
    # 5 Arabic letters * 2 bytes = 10 bytes = 80 bits
    assert msg.bit_cost == 10 * 8
    assert ch.decode(msg) == "مرحبا"


def test_empty_message():
    ch = TextChannel()
    msg = ch.encode("", "a", "b")
    assert msg.bit_cost == 0
    assert ch.decode(msg) == ""


def test_decode_rejects_non_bytes():
    """If something feeds the wrong payload type to the decoder, fail loudly."""
    import pytest

    from budgetfair.comm.base import TransmittedMessage

    ch = TextChannel()
    bad = TransmittedMessage(
        payload="not bytes",  # str, not bytes
        bit_cost=72,
        sender_id="a",
        receiver_id="b",
        timestamp=0.0,
        metadata={},
    )
    with pytest.raises(TypeError):
        ch.decode(bad)
