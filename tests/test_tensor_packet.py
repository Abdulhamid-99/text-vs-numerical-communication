"""Independent numeric-wire contracts, including literal known-byte packets."""
import struct

import numpy as np
import pytest

from budgetfair.comm import tensor_packet as wire


# Literal format fixtures; constructed without calling the implementation's encoder/header.
FLOAT_PACKET = bytes.fromhex(
    "42465450 01 00 01 02 ffffffff 1000000000000000 "
    "02000000 02000000 0000803f 000000c0 00000000 00000080"
)
INT8_PACKET = bytes.fromhex(
    "42465450 01 01 00 02 ffffffff 0e00000000000000 "
    "02000000 03000000 0000803f 0000803f 81007f 000000"
)
INT4_PACKET = bytes.fromhex(
    "42465450 01 02 00 02 ffffffff 0c00000000000000 "
    "02000000 03000000 0000803f 0000803f 0907 0000"
)


def mutate(packet, offset, fmt, value):
    changed = bytearray(packet)
    struct.pack_into(fmt, changed, offset, value)
    return bytes(changed)


def test_float32_known_bytes_exact_including_negative_zero():
    expected = np.array([[1.0, -2.0], [0.0, -0.0]], dtype=np.float32)
    decoded = wire.decode_packet(FLOAT_PACKET)
    assert decoded.array.tobytes() == expected.tobytes()
    assert decoded.role == "token_embeddings" and decoded.layer_index is None
    assert decoded.codec == "float32" and decoded.bit_cost == 352
    assert wire.encode_tensor(expected, role="token_embeddings") == FLOAT_PACKET


@pytest.mark.parametrize("codec,packet,expected,bits", [
    ("int8", INT8_PACKET, [[-127, 0, 127], [0, 0, 0]], 336),
    ("int4", INT4_PACKET, [[-7, 0, 7], [0, 0, 0]], 320),
])
def test_quantized_known_bytes_and_zero_row_scale(codec, packet, expected, bits):
    expected = np.array(expected, dtype=np.float32)
    decoded = wire.decode_packet(packet)
    np.testing.assert_array_equal(decoded.array, expected)
    assert wire.encode_tensor(expected, codec=codec) == packet
    assert decoded.bit_cost == bits == len(packet) * 8


@pytest.mark.parametrize("codec,width", [("float32", 5), ("int8", 5), ("int4", 4), ("int4", 5)])
def test_independent_raw_parser_accounts_every_metadata_scale_and_padding_byte(codec, width):
    value = np.arange(2 * 3 * width, dtype=np.float32).reshape(2, 3, width)
    packet = wire.encode_tensor(value, codec=codec, role="hidden_states", layer_index=17)
    magic, version, codec_id, role_id, rank, layer, body_length = struct.unpack_from("<4sBBBBiQ", packet)
    shape = struct.unpack_from("<III", packet, 20)
    assert (magic, version, role_id, rank, layer, shape) == (b"BFTP", 1, 2, 3, 17, (2, 3, width))
    assert codec_id == {"float32": 0, "int8": 1, "int4": 2}[codec]
    expected_body = 6 * width * 4 if codec == "float32" else 6 * (4 + (width if codec == "int8" else (width + 1) // 2))
    assert body_length == expected_body and len(packet) == 20 + 12 + expected_body
    decoded = wire.decode_packet(packet)
    assert decoded.layer_index == 17 and decoded.role == "hidden_states"
    assert decoded.array.shape == value.shape


@pytest.mark.parametrize("codec,qmax", [("int8", 127), ("int4", 7)])
def test_each_row_has_its_own_scale_and_bounded_quantization_error(codec, qmax):
    value = np.array([[-1, -0.34, 0.33, 1, 0.05], [-1000, -250, 50, 1000, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
    packet = wire.encode_tensor(value, codec=codec)
    scales = np.frombuffer(packet, dtype="<f4", count=3, offset=28)
    assert scales[1] > scales[0] * 990 and scales[2] == 1
    decoded = wire.decode_tensor(packet)
    allowance = scales.astype(np.float64)[:, None] / 2 + np.finfo(np.float32).eps * np.maximum(1, np.abs(value))
    assert np.all(np.abs(decoded.astype(np.float64) - value) <= allowance)
    np.testing.assert_array_equal(decoded[2], np.zeros(5, dtype=np.float32))


def test_float32_nd_noncontiguous_big_endian_bitwise_roundtrip():
    value = np.arange(48, dtype=np.float32).reshape(2, 4, 6)[:, ::2, ::-1].astype(">f4")
    decoded = wire.decode_tensor(wire.encode_tensor(value))
    assert decoded.dtype == np.float32 and decoded.shape == (2, 2, 6)
    assert decoded.tobytes() == value.astype("<f4").tobytes()


@pytest.mark.parametrize("codec", ["float32", "int8", "int4"])
def test_float32_extrema_and_subnormal_rows_are_finite(codec):
    tiny = np.nextafter(np.float32(0), np.float32(1))
    maximum = np.finfo(np.float32).max
    value = np.array([[maximum, -maximum, 0], [tiny, -tiny, 0]], dtype=np.float32)
    decoded = wire.decode_tensor(wire.encode_tensor(value, codec=codec))
    assert np.isfinite(decoded).all()
    np.testing.assert_array_equal(decoded[1], value[1])
    if codec == "float32":
        assert decoded.tobytes() == value.tobytes()
    else:
        assert np.all(np.abs(decoded[0, :2].astype(np.float64)) > float(maximum) * 0.99)


def test_packet_and_decoded_array_do_not_alias_the_original_or_each_other():
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    packet = wire.encode_tensor(source)
    source[:] = 99
    first = wire.decode_tensor(packet)
    np.testing.assert_array_equal(first, np.arange(6, dtype=np.float32).reshape(2, 3))
    first[:] = -10
    np.testing.assert_array_equal(wire.decode_tensor(packet), np.arange(6, dtype=np.float32).reshape(2, 3))
    assert type(packet) is bytes


@pytest.mark.parametrize("bad", [None, [[1.0, 2.0]], "token text", b"bytes", np.ones((2, 2), dtype=np.int32),
                                 np.ones((2, 2), dtype=np.float64), np.ones((2, 2), dtype=bool),
                                 np.ones((2, 2), dtype=complex), np.ones((2, 2), dtype=object)])
def test_encoder_rejects_nontensor_or_non_float32_input(bad):
    with pytest.raises(TypeError):
        wire.encode_tensor(bad)


@pytest.mark.parametrize("shape", [(2,), (), (0, 2), (2, 0), (1,) * 9])
def test_empty_dimensions_and_unsupported_ranks_are_rejected(shape):
    with pytest.raises(ValueError):
        wire.encode_tensor(np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("codec", ["float32", "int8", "int4"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_encoder_rejects_nonfinite_values(codec, value):
    with pytest.raises(ValueError, match="finite"):
        wire.encode_tensor(np.array([[value, 1]], dtype=np.float32), codec=codec)


@pytest.mark.parametrize("kwargs", [{"codec": "int2"}, {"codec": 8}, {"role": "text"},
    {"role": "hidden_states"}, {"role": "token_embeddings", "layer_index": 0},
    {"layer_index": -1}, {"layer_index": True}, {"layer_index": 2**31}])
def test_encoder_rejects_invalid_metadata(kwargs):
    with pytest.raises(ValueError):
        wire.encode_tensor(np.ones((2, 2), dtype=np.float32), **kwargs)


@pytest.mark.parametrize("bad", [None, bytearray(FLOAT_PACKET), memoryview(FLOAT_PACKET),
                                 np.ones((2, 2), dtype=np.float32), "secret"])
def test_decoder_accepts_only_wire_bytes(bad):
    with pytest.raises(TypeError):
        wire.decode_tensor(bad)


@pytest.mark.parametrize("packet", [b"", FLOAT_PACKET[:19], FLOAT_PACKET[:26], FLOAT_PACKET[:-1], FLOAT_PACKET + b"\0",
    mutate(FLOAT_PACKET, 0, "4s", b"NOPE"), mutate(FLOAT_PACKET, 4, "B", 2),
    mutate(FLOAT_PACKET, 5, "B", 255), mutate(FLOAT_PACKET, 6, "B", 255),
    mutate(FLOAT_PACKET, 7, "B", 1), mutate(FLOAT_PACKET, 7, "B", 9),
    mutate(FLOAT_PACKET, 8, "i", -2), mutate(FLOAT_PACKET, 8, "i", 0),
    mutate(FLOAT_PACKET, 6, "B", 2), mutate(FLOAT_PACKET, 12, "Q", 2**64 - 1),
    mutate(FLOAT_PACKET, 20, "I", 0), mutate(FLOAT_PACKET, 20, "I", 2**32 - 1)])
def test_malformed_header_shape_and_exact_length_fail(packet):
    with pytest.raises(ValueError):
        wire.decode_packet(packet)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_decoder_rejects_nonfinite_float_body(value):
    with pytest.raises(ValueError, match="finite"):
        wire.decode_tensor(mutate(FLOAT_PACKET, 28, "f", value))


@pytest.mark.parametrize("packet", [INT8_PACKET, INT4_PACKET])
@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_decoder_rejects_nonpositive_or_nonfinite_scales(packet, scale):
    with pytest.raises(ValueError, match="scale"):
        wire.decode_tensor(mutate(packet, 28, "f", scale))


@pytest.mark.parametrize("packet", [mutate(INT8_PACKET, 36, "b", -128),
                                     mutate(INT4_PACKET, 36, "B", 8),
                                     mutate(INT4_PACKET, 36, "B", 0x89)])
def test_reserved_signed_codes_are_rejected(packet):
    with pytest.raises(ValueError, match="reserved"):
        wire.decode_tensor(packet)


def test_nonzero_odd_width_padding_cannot_carry_uncharged_hidden_data():
    with pytest.raises(ValueError, match="padding"):
        wire.decode_tensor(mutate(INT4_PACKET, 37, "B", 0x17))


@pytest.mark.parametrize("packet", [INT8_PACKET, INT4_PACKET])
def test_reconstruction_overflow_is_a_failure_not_an_infinite_tensor(packet):
    with pytest.raises(ValueError, match="overflow"):
        wire.decode_tensor(mutate(packet, 28, "f", np.finfo(np.float32).max))


@pytest.mark.parametrize("kwargs", [{"max_elements": 3}, {"max_packet_bytes": 43},
    {"max_elements": True}, {"max_elements": 0}, {"max_elements": 2**64},
    {"max_packet_bytes": -1}, {"max_packet_bytes": 2**64}])
def test_receiver_limits_are_enforced_before_decode(kwargs):
    with pytest.raises(ValueError):
        wire.decode_tensor(FLOAT_PACKET, **kwargs)


def test_huge_dimension_products_fail_before_any_tensor_buffer_or_allocation(monkeypatch):
    packet = mutate(mutate(FLOAT_PACKET, 20, "I", 2**32 - 1), 24, "I", 2**32 - 1)
    def prohibited(*args, **kwargs):
        raise AssertionError("malformed shape reached tensor allocation")
    monkeypatch.setattr(wire.np, "frombuffer", prohibited)
    monkeypatch.setattr(wire.np, "empty", prohibited)
    with pytest.raises(ValueError, match="element count"):
        wire.decode_tensor(packet)


@pytest.mark.parametrize("codec", ["float32", "int8", "int4"])
def test_encoding_is_deterministic_and_restores_the_declared_nd_shape(codec):
    value = np.random.default_rng(9001).normal(size=(2, 3, 5)).astype(np.float32)
    first = wire.encode_tensor(value, codec=codec)
    assert first == wire.encode_tensor(value.copy(), codec=codec)
    assert wire.decode_tensor(first).shape == value.shape
