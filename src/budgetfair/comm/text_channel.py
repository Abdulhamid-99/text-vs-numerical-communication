"""Text channel: UTF-8 encoded natural-language messages.

Cost formula (from thesis):
    C_text = len_bytes(UTF-8(msg)) * 8

Metadata overhead (sender ID, channel tag, timestamp) is logged but is not added
to the bit cost — the thesis budgets the *payload*, and metadata < 32 bits per
message is bounded enough to ignore under either channel.
"""
from __future__ import annotations

import time

from .base import Channel, TransmittedMessage


class TextChannel(Channel):
    name = "text"

    def encode(self, content: str, sender_id: str, receiver_id: str) -> TransmittedMessage:
        payload = content.encode("utf-8")
        return TransmittedMessage(
            payload=payload,
            bit_cost=len(payload) * 8,
            sender_id=sender_id,
            receiver_id=receiver_id,
            timestamp=time.time(),
            metadata={"channel": self.name, "text_len_chars": len(content)},
        )

    def decode(self, msg: TransmittedMessage) -> str:
        if not isinstance(msg.payload, (bytes, bytearray)):
            raise TypeError(f"TextChannel.decode expected bytes, got {type(msg.payload).__name__}")
        return bytes(msg.payload).decode("utf-8")

    @staticmethod
    def bit_cost(text: str) -> int:
        """Bit cost of a text message under the thesis cost model."""
        return len(text.encode("utf-8")) * 8
