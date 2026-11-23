"""Ordered numeric activation transport with an explicit receiver boundary.

This is a new protocol, not a reinterpretation of the historical pooled-vector
study. Each natural-token position has its own numeric state. No text/token IDs
from the sender are given to ``receive_packet``. A shared public prompt and a
same-checkpoint model are preconditions, not transmitted private information.
The complete application packet (header, shape, scales, components) is charged.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch
from transformers import GenerationConfig as HFGenerationConfig

from ..comm.tensor_packet import decode_packet, encode_tensor
from .client import LLMClient


@dataclass(frozen=True)
class PublicPrompt:
    prefix_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]
    placeholder_id: int
    max_payload_tokens: int = 128
    no_message_slots: int = 16

    def validate(self, vocab_size: int) -> None:
        if not self.prefix_ids or not self.suffix_ids:
            raise ValueError("Public prefix and suffix must both be nonempty")
        for token in (*self.prefix_ids, *self.suffix_ids, self.placeholder_id):
            if isinstance(token, bool) or not isinstance(token, Integral) or not 0 <= token < vocab_size:
                raise ValueError("Public token ID outside vocabulary")
        for name in ("max_payload_tokens", "no_message_slots"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 128:
                raise ValueError(f"{name} must be an integer in 1..128")
        if self.no_message_slots > self.max_payload_tokens:
            raise ValueError("No-message span exceeds public maximum")


def make_public_prompt(client: LLMClient) -> PublicPrompt:
    """Render one public chat with a dedicated, nonprivate payload span."""
    marker = "{{NUMERIC_PAYLOAD_SLOT}}"
    messages = [
        {"role": "system", "content": "Answer the question using the supplied record. Copy the recorded value exactly, preserving every letter. Output only the value, with no quotes, punctuation, or explanation. If the record has no value, output UNKNOWN."},
        {"role": "user", "content": "Question: What is the stored value for item ALPHA?\n\nLookup record: item ALPHA; value " + marker + "\n\nAnswer:"},
    ]
    rendered = client.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if rendered.count(marker) != 1:
        raise ValueError("Public chat must have exactly one payload slot")
    prefix, suffix = rendered.split(marker)
    encode = lambda text: tuple(client.tokenizer.encode(text, add_special_tokens=False))
    placeholder = encode("?")
    if len(placeholder) != 1:
        raise ValueError("The public placeholder must be one tokenizer token")
    public = PublicPrompt(encode(prefix), encode(suffix), placeholder[0])
    public.validate(client.model.config.vocab_size)
    return public


def neutral_generation_kwargs(client: LLMClient) -> dict:
    """Pin ID-dependent repetition processors for comparable numeric/text inputs.

    Qwen's bundled generation config has repetition_penalty=1.05. Leaving this
    implicit penalizes genuine source IDs but not their identical embeddings
    under placeholder IDs, confounding the causal interface comparison.
    """
    model_config = client.model.generation_config
    clean = HFGenerationConfig(**{name: getattr(model_config, name, None)
                                 for name in ("bos_token_id", "eos_token_id", "pad_token_id")})
    # HF can merge checkpoint-specific defaults even when a fresh generation
    # config is supplied. Disable that merge, retaining only the explicitly
    # declared special IDs plus the client's validated generation policy.
    return {"generation_config": clean, "use_model_defaults": False,
            **client._generation_kwargs()}


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest()


def _layer(client: LLMClient, layer: int):
    if isinstance(layer, bool) or not isinstance(layer, Integral) or layer < -1:
        raise ValueError("Layer must be -1 for embeddings or a decoder-block index")
    return client.model.get_input_embeddings() if layer == -1 else client._decoder_block(layer)


@torch.inference_mode()
def encode_message(client: LLMClient, public: PublicPrompt, private_message: str,
                   *, layer: int, codec: str = "float32") -> tuple[bytes, dict]:
    """Sender-only encoder. Its audit trace must never be mounted at receiver."""
    public.validate(client.model.config.vocab_size)
    module = _layer(client, layer)
    if not isinstance(private_message, str) or not private_message:
        raise ValueError("Sender message must be a nonempty string")
    private_ids = client.tokenizer.encode(private_message, add_special_tokens=False)
    if not 1 <= len(private_ids) <= public.max_payload_tokens:
        raise ValueError("Sender message length exceeds the declared protocol")
    source_ids = list(public.prefix_ids) + private_ids
    client._check_context(len(source_ids))
    input_ids = torch.tensor([source_ids], device=client._input_device())
    captured = []
    captured_dtypes = []
    start = len(public.prefix_ids)

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or tuple(hidden.shape[:2]) != (1, len(source_ids)):
            raise ValueError("Unexpected sender activation shape")
        value = hidden[0, start:].detach().float().cpu().numpy().copy()
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite sender activation")
        captured.append(value)
        captured_dtypes.append(str(hidden.dtype))

    handle = module.register_forward_hook(capture)
    try:
        client.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Sender must capture exactly one complete activation span")
    role = "token_embeddings" if layer == -1 else "hidden_states"
    packet = encode_tensor(captured[0], codec=codec, role=role,
                           layer_index=None if layer == -1 else int(layer))
    return packet, {"sender_input_ids": source_ids, "message_token_ids": private_ids,
                    "source_array_sha256": _array_sha(captured[0]), "shape": list(captured[0].shape),
                    "source_activation_dtype": captured_dtypes[0],
                    "packet_sha256": hashlib.sha256(packet).hexdigest(), "wire_bits": 8 * len(packet),
                    "layer": int(layer), "codec": codec}


@torch.inference_mode()
def receive_packet(client: LLMClient, public: PublicPrompt, packet: bytes | None,
                   *, expected_layer: int) -> dict:
    """Receiver: public inputs and numeric bytes only; no gold or private IDs.

    A packet replaces only its ordered payload span at the declared block's
    output on the first prefill. It does not alter later decoding steps, and
    its hook is removed even on generation failure. No implicit text fallback.
    """
    public.validate(client.model.config.vocab_size)
    module = _layer(client, expected_layer)
    decoded = None
    if packet is not None:
        decoded = decode_packet(packet, max_elements=public.max_payload_tokens * client.model.config.hidden_size)
        role = "token_embeddings" if expected_layer == -1 else "hidden_states"
        expected_wire_layer = None if expected_layer == -1 else expected_layer
        if decoded.role != role or decoded.layer_index != expected_wire_layer:
            raise ValueError("Delivered role/layer differs from the public experiment condition")
        if decoded.array.ndim != 2 or decoded.array.shape[1] != client.model.config.hidden_size:
            raise ValueError("Delivered states do not match the receiver hidden dimension")
        rows = decoded.array.shape[0]
        if not 1 <= rows <= public.max_payload_tokens:
            raise ValueError("Delivered span exceeds the public maximum")
    else:
        rows = public.no_message_slots
    ids = list(public.prefix_ids) + [public.placeholder_id] * rows + list(public.suffix_ids)
    client._check_context(len(ids), generation=True)
    input_ids = torch.tensor([ids], device=client._input_device())
    injections = []
    handle = None
    if decoded is not None:
        values = torch.from_numpy(decoded.array)
        start, stop = len(public.prefix_ids), len(public.prefix_ids) + rows

        def inject(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if injections:
                return output
            if hidden.ndim != 3 or tuple(hidden.shape[:2]) != (1, len(ids)):
                raise ValueError("Expected one full receiver prefill before cached decoding")
            replacement = values.to(device=hidden.device, dtype=hidden.dtype)
            if not torch.isfinite(replacement).all() or not torch.isfinite(hidden).all():
                raise ValueError("Nonfinite activation or receiver dtype overflow")
            changed = hidden.clone()
            changed[:, start:stop] = replacement.unsqueeze(0)
            injections.append({"positions": [start, stop], "layer": int(expected_layer),
                "decoded_array_sha256": _array_sha(decoded.array),
                "applied_array_sha256": _array_sha(replacement.float().cpu().numpy()),
                "applied_dtype": str(hidden.dtype),
                "delta_l2": float(torch.linalg.vector_norm(changed[:, start:stop].double() - hidden[:, start:stop].double()).cpu())})
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handle = module.register_forward_hook(inject)
    try:
        out = client.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                    **neutral_generation_kwargs(client))
    finally:
        if handle is not None:
            handle.remove()
    if decoded is not None and len(injections) != 1:
        raise RuntimeError("The delivered packet was not consumed exactly once")
    output_ids = out[0, len(ids):].tolist()
    return {"receiver_input_ids": ids, "output_ids": output_ids,
            "answer": client.tokenizer.decode(output_ids, skip_special_tokens=True),
            "wire_bits": 0 if decoded is None else decoded.bit_cost,
            "packet_sha256": None if packet is None else hashlib.sha256(packet).hexdigest(),
            "injections": injections, "status": "completed"}


@torch.inference_mode()
def receive_text(client: LLMClient, public: PublicPrompt, text_packet: bytes) -> dict:
    """Explicit UTF-8 baseline using the same segmented public tokenization."""
    public.validate(client.model.config.vocab_size)
    if type(text_packet) is not bytes:
        raise TypeError("Text baseline requires actual UTF-8 bytes")
    text = text_packet.decode("utf-8", errors="strict")
    payload_ids = client.tokenizer.encode(text, add_special_tokens=False)
    if not 1 <= len(payload_ids) <= public.max_payload_tokens:
        raise ValueError("Text packet exceeds the declared length")
    ids = list(public.prefix_ids) + payload_ids + list(public.suffix_ids)
    client._check_context(len(ids), generation=True)
    input_ids = torch.tensor([ids], device=client._input_device())
    out = client.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                **neutral_generation_kwargs(client))
    output_ids = out[0, len(ids):].tolist()
    return {"receiver_input_ids": ids, "output_ids": output_ids,
            "answer": client.tokenizer.decode(output_ids, skip_special_tokens=True),
            "wire_bits": 8 * len(text_packet), "packet_sha256": hashlib.sha256(text_packet).hexdigest(),
            "injections": [], "status": "completed"}
