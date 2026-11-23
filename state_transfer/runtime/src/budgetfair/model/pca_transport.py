"""Receiver-only bridge from compressed numeric packets to activation slots."""
from __future__ import annotations

import hashlib

from ..comm.pca_packet import PCAAsset, reconstruct
from ..comm.tensor_packet import encode_tensor
from .client import LLMClient
from .span_transport import PublicPrompt, receive_packet


def receive_projected_packet(client: LLMClient, public: PublicPrompt, packet: bytes,
                             asset: PCAAsset, *, expected_layer: int,
                             expected_dimensions: int, expected_codec: str) -> dict:
    """Public context, wire bytes, and a plan-bound basis are the only inputs.

    Reconstruction is a local receiver computation. A temporary float32 buffer
    is passed to the already-tested span injector; it is not an additional
    network transmission and its size is never substituted for wire cost.
    """
    if asset.layer != expected_layer or asset.mean.shape != (client.model.config.hidden_size,):
        raise ValueError("The public PCA asset does not match the receiver layer/model width")
    hidden, metadata = reconstruct(packet, asset, expected_dimensions=expected_dimensions,
                                   expected_codec=expected_codec)
    local_buffer = encode_tensor(hidden, codec="float32", role="hidden_states", layer_index=expected_layer)
    result = receive_packet(client, public, local_buffer, expected_layer=expected_layer)
    if len(result["injections"]) != 1 or result["injections"][0]["decoded_array_sha256"] != metadata["reconstructed_hidden_sha256"]:
        raise RuntimeError("PCA reconstruction did not reach the span injection boundary")
    result["local_reconstructed_buffer_sha256"] = hashlib.sha256(local_buffer).hexdigest()
    result["local_reconstructed_buffer_bytes"] = len(local_buffer)
    result.update(metadata)
    return result
