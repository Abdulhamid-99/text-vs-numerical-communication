"""Text channel: UTF-8 encoded natural-language messages.

Cost formula (from thesis):
    C_text = len_bytes(UTF-8(msg)) * 8
"""
from __future__ import annotations

from .base import Channel, TransmittedMessage


class TextChannel(Channel):
    name = "text"

    def encode(self, content: str, sender_id: str, receiver_id: str) -> TransmittedMessage:
        raise NotImplementedError

    def decode(self, msg: TransmittedMessage) -> str:
        raise NotImplementedError

    @staticmethod
    def bit_cost(text: str) -> int:
        """Bit cost of a text message under the thesis cost model."""
        return len(text.encode("utf-8")) * 8
