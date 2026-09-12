"""Independent span-boundary tests; tiny random models are not research results."""
import copy
import hashlib
import math

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from budgetfair.comm.tensor_packet import decode_packet, encode_tensor
from budgetfair.model.client import GenerationConfig, LLMClient
from budgetfair.model import span_transport as transport


class Tokens:
    pad_token_id = 0

    def encode(self, text, **kwargs):
        # This declared tiny vocabulary is a deterministic CPU test fixture.
        return [{"A": 4, "B": 5, "é": 6}.get(character, 7) for character in text]

    def decode(self, ids, **kwargs):
        return " ".join(str(int(value)) for value in ids)


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def client():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(431)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
            eos_token_id=None, pad_token_id=0, attention_dropout=0.0)).eval()
    result = LLMClient.__new__(LLMClient)
    result.model, result.tokenizer = model, Tokens()
    result.max_context_len = 64
    result.generation = GenerationConfig(max_new_tokens=3, do_sample=False)
    return result


@pytest.fixture
def public():
    return transport.PublicPrompt((1, 3), (8, 9), 10, max_payload_tokens=12, no_message_slots=4)


def hooks(client):
    return {name: tuple(module._forward_hooks) for name, module in client.model.named_modules() if module._forward_hooks}


def capture_generation(client, operation):
    embeddings, logits = [], []
    def prefill(module, args, kwargs):
        embeddings.append((args[0] if args else kwargs["hidden_states"]).detach().clone())
    def outputs(module, args, result):
        logits.append(result.logits.detach().clone())
    first = client.model.model.layers[0].register_forward_pre_hook(prefill, with_kwargs=True)
    final = client.model.register_forward_hook(outputs)
    try:
        result = operation()
    finally:
        first.remove()
        final.remove()
    return result, embeddings, logits


@pytest.mark.parametrize("message", ["AB", "AAAA", "ABBA"])
def test_full_embedding_span_matches_actual_text_embeddings_logits_and_generated_tokens(client, public, message):
    # An inherited repetition policy must not privilege genuine IDs over placeholders.
    client.model.generation_config.repetition_penalty = 1.05
    client.model.generation_config.no_repeat_ngram_size = 2
    packet, sender = transport.encode_message(client, public, message, layer=-1)
    reference, reference_embeds, reference_logits = capture_generation(client,
        lambda: transport.receive_text(client, public, message.encode()))
    received, received_embeds, received_logits = capture_generation(client,
        lambda: transport.receive_packet(client, public, packet, expected_layer=-1))
    assert received["output_ids"] == reference["output_ids"]
    assert len(reference_logits) == len(received_logits) == 3
    for expected, actual in zip(reference_embeds, received_embeds):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for expected, actual in zip(reference_logits, received_logits):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert received["receiver_input_ids"] == [1, 3] + [10] * len(message) + [8, 9]
    assert sender["message_token_ids"] != received["receiver_input_ids"][2:-2]
    assert len(received["injections"]) == 1
    assert hooks(client) == {}


def test_inherited_sequence_bias_cannot_distinguish_identical_embedding_prefills(client, public):
    # The reference ends in private AAAA + public suffix. The numeric path has
    # placeholder IDs. This very large inherited n-gram bias forces a disparity
    # unless the protocol genuinely neutralizes history-ID-dependent processing.
    client.model.generation_config.sequence_bias = {(4, 4, 4, 4, 8, 9, 30): 100.0}
    packet, _ = transport.encode_message(client, public, "AAAA", layer=-1)
    reference = transport.receive_text(client, public, b"AAAA")
    received = transport.receive_packet(client, public, packet, expected_layer=-1)
    assert received["output_ids"] == reference["output_ids"]


def test_neutral_generation_has_fresh_config_and_no_inherited_decoding_policy(client):
    config = client.model.generation_config
    config.sequence_bias = {(4, 5): 50.0}
    config.min_new_tokens = 20
    config.num_beams = 3
    config.forced_bos_token_id = 7
    before = copy.deepcopy(config.to_dict())
    first, second = transport.neutral_generation_kwargs(client), transport.neutral_generation_kwargs(client)
    assert "generation_config" in first
    clean = first["generation_config"]
    assert clean is not config and clean is not second["generation_config"]
    assert clean.sequence_bias is None and clean.min_new_tokens is None
    assert clean.num_beams == 1 and clean.forced_bos_token_id is None
    assert clean.repetition_penalty == 1 and clean.no_repeat_ngram_size == 0
    assert clean.bos_token_id == config.bos_token_id and clean.eos_token_id == config.eos_token_id
    assert clean.pad_token_id == config.pad_token_id
    assert first["max_new_tokens"] == 3 and first["do_sample"] is False and first["logits_processor"]
    assert config.to_dict() == before


@pytest.mark.parametrize("layer", [0, 1])
def test_decoder_span_replaces_only_payload_positions_once_and_leaves_cached_steps_untouched(client, public, monkeypatch, layer):
    values = np.arange(48, dtype=np.float32).reshape(3, 16) / 100
    packet = encode_tensor(values, role="hidden_states", layer_index=layer)
    block = client.model.model.layers[layer]
    raw, changed = [], []
    def observe(destination):
        def hook(module, args, output):
            destination.append((output[0] if isinstance(output, tuple) else output).detach().clone())
        return hook
    before = block.register_forward_hook(observe(raw))
    generate = client.model.generate
    def wrapped_generate(**kwargs):
        after = block.register_forward_hook(observe(changed))
        try:
            return generate(**kwargs)
        finally:
            after.remove()
    monkeypatch.setattr(client.model, "generate", wrapped_generate)
    try:
        result = transport.receive_packet(client, public, packet, expected_layer=layer)
    finally:
        before.remove()
    assert len(raw) == len(changed) == 3 and len(result["injections"]) == 1
    torch.testing.assert_close(changed[0][0, 2:5], torch.from_numpy(values), atol=0, rtol=0)
    torch.testing.assert_close(changed[0][:, :2], raw[0][:, :2], atol=0, rtol=0)
    torch.testing.assert_close(changed[0][:, 5:], raw[0][:, 5:], atol=0, rtol=0)
    for initial, modified in zip(raw[1:], changed[1:]):
        torch.testing.assert_close(initial, modified, atol=0, rtol=0)
    assert hooks(client) == {}


@pytest.mark.parametrize("codec", ["float32", "int8", "int4"])
def test_sender_packet_metadata_hash_and_bits_bind_actual_private_span(client, public, codec):
    packet, audit = transport.encode_message(client, public, "ABBA", layer=0, codec=codec)
    decoded = decode_packet(packet)
    assert decoded.array.shape == (4, 16) and decoded.layer_index == 0 and decoded.role == "hidden_states"
    assert audit["wire_bits"] == decoded.bit_cost == len(packet) * 8
    assert audit["packet_sha256"] == hashlib.sha256(packet).hexdigest()
    assert audit["sender_input_ids"] == [1, 3, 4, 5, 5, 4]
    assert hooks(client) == {}


@pytest.mark.parametrize("packet,layer", [
    (b"malformed", -1),
    (bytearray(b"not immutable bytes"), -1),
    (np.ones((2, 16), dtype=np.float32), -1),
    (encode_tensor(np.ones((2, 16), dtype=np.float32), role="generic"), -1),
    (encode_tensor(np.ones((2, 16), dtype=np.float32), role="hidden_states", layer_index=1), 0),
    (encode_tensor(np.ones((2, 15), dtype=np.float32), role="token_embeddings"), -1),
    (encode_tensor(np.ones((1, 2, 16), dtype=np.float32), role="token_embeddings"), -1),
    (encode_tensor(np.ones((13, 16), dtype=np.float32), role="token_embeddings"), -1),
])
def test_invalid_delivery_is_rejected_before_any_model_call_or_hook(client, public, monkeypatch, packet, layer):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid packet reached model execution")
    monkeypatch.setattr(client.model, "forward", forbidden)
    monkeypatch.setattr(client.model, "generate", forbidden)
    with pytest.raises((ValueError, TypeError)):
        transport.receive_packet(client, public, packet, expected_layer=layer)
    assert hooks(client) == {}


@pytest.mark.parametrize("stage", ["source_forward", "receiver_generate"])
def test_exceptions_propagate_and_remove_hooks_instead_of_returning_scored_zero(client, public, monkeypatch, stage):
    def fail(*args, **kwargs):
        raise RuntimeError("injected execution failure")
    if stage == "source_forward":
        monkeypatch.setattr(client.model, "forward", fail)
        operation = lambda: transport.encode_message(client, public, "ABBA", layer=0)
    else:
        packet = encode_tensor(np.ones((4, 16), dtype=np.float32), role="token_embeddings")
        monkeypatch.setattr(client.model, "generate", fail)
        operation = lambda: transport.receive_packet(client, public, packet, expected_layer=-1)
    with pytest.raises(RuntimeError, match="injected execution failure"):
        operation()
    assert hooks(client) == {}


def test_receiver_dtype_overflow_removes_hook_and_raises(client, public):
    client.model.half()
    packet = encode_tensor(np.full((2, 16), np.finfo(np.float32).max, dtype=np.float32), role="token_embeddings")
    with pytest.raises(ValueError, match="overflow"):
        transport.receive_packet(client, public, packet, expected_layer=-1)
    assert hooks(client) == {}


def test_finite_large_span_cannot_create_nonfinite_trace_norm(client, public, monkeypatch):
    values = np.full((2, 16), 1e30, dtype=np.float32)
    packet = encode_tensor(values, role="token_embeddings")
    # Isolate hook/trace arithmetic from downstream model numerical behaviour.
    def finite_generation(*, input_ids, **kwargs):
        client.model.get_input_embeddings()(input_ids)
        return torch.cat([input_ids, torch.tensor([[3]], device=input_ids.device)], dim=1)
    monkeypatch.setattr(client.model, "generate", finite_generation)
    result = transport.receive_packet(client, public, packet, expected_layer=-1)
    assert math.isfinite(result["injections"][0]["delta_l2"])
    assert result["injections"][0]["delta_l2"] == pytest.approx(float(np.linalg.norm(values.astype(np.float64))), rel=1e-6)


def test_no_communication_is_reproducible_after_received_packet_and_uses_public_slots(client, public):
    before = transport.receive_packet(client, public, None, expected_layer=-1)
    packet, _ = transport.encode_message(client, public, "AB", layer=-1)
    received = transport.receive_packet(client, public, packet, expected_layer=-1)
    after = transport.receive_packet(client, public, None, expected_layer=-1)
    assert before == after
    assert after["receiver_input_ids"] == [1, 3, 10, 10, 10, 10, 8, 9]
    assert after["packet_sha256"] is None and after["wire_bits"] == 0 and after["injections"] == []
    assert len(received["receiver_input_ids"]) == 6
    assert hooks(client) == {}


@pytest.mark.parametrize("path", ["text", "packet", "sender"])
def test_context_capacity_fails_before_model_execution_without_truncating_suffix(client, public, monkeypatch, path):
    client.max_context_len = 4
    def forbidden(*args, **kwargs):
        raise AssertionError("overflowing prompt reached model")
    monkeypatch.setattr(client.model, "generate", forbidden)
    monkeypatch.setattr(client.model, "forward", forbidden)
    packet = encode_tensor(np.ones((4, 16), dtype=np.float32), role="token_embeddings")
    operation = (lambda: transport.receive_text(client, public, b"AAAA")) if path == "text" else (
        (lambda: transport.receive_packet(client, public, packet, expected_layer=-1)) if path == "packet" else
        (lambda: transport.encode_message(client, public, "AAAA", layer=-1)))
    with pytest.raises(ValueError, match="overflow"):
        operation()
    assert hooks(client) == {}


def test_text_baseline_charges_actual_utf8_bytes_and_returns_only_new_tokens(client, public):
    result = transport.receive_text(client, public, "AéB".encode("utf-8"))
    assert result["wire_bits"] == 32
    assert result["receiver_input_ids"] == [1, 3, 4, 6, 5, 8, 9]
    assert len(result["output_ids"]) == 3
    assert result["answer"] == client.tokenizer.decode(result["output_ids"])


def test_public_token_validation_precedes_model_call(client, monkeypatch):
    public = transport.PublicPrompt((1, 99), (8,), 10)
    monkeypatch.setattr(client.model, "generate", lambda **kwargs: pytest.fail("bad public input reached model"))
    with pytest.raises(ValueError, match="vocabulary"):
        transport.receive_packet(client, public, None, expected_layer=-1)
