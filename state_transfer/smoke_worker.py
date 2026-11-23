"""Single-role model smoke worker. Only the role's input is mounted."""
import argparse
import hashlib
from pathlib import Path

from common import read_json, write_json, generate, encode_with_states
from study_worker import load_protocol, load_client, runtime_identity, task_assets
from task_prompts import receiver_parts, sender_messages, sender_encoding_prompt
from budgetfair.comm.pca_packet import encode_projected
from budgetfair.model.pca_transport import receive_projected_packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("sender", "receiver"))
    args = parser.parse_args()
    plan_path = Path("/runplan/protocol.json")
    _, settings = load_protocol(plan_path, "qwen")
    assets = task_assets(plan_path.parent / "assets", settings)
    view = read_json(Path("/input/view.json"))
    client, _ = load_client(settings)
    runtime_identity(client, settings)
    if args.role == "sender":
        message = generate(client, sender_messages(view))["text"]
        token_count = len(client.tokenizer.encode(message, add_special_tokens=False))
        if not 1 <= token_count <= 128:
            raise ValueError("Sender message outside the protocol token range")
        prompt = sender_encoding_prompt(client, view["task"])
        _, _, states = encode_with_states(client, prompt, message, layer=settings["layer"])
        packet = encode_projected(states, assets[view["task"]], dimensions=settings["pca_rank"], codec=settings["pca_codec"])
        Path("/out/packet.bin").write_bytes(packet)
        result = {"message_bytes": len(message.encode("utf-8")), "message_tokens": token_count}
    else:
        packet = Path("/input/packet.bin").read_bytes()
        prompt, _ = receiver_parts(client, view)
        trace = receive_projected_packet(client, prompt, packet, assets[view["task"]],
            expected_layer=settings["layer"], expected_dimensions=settings["pca_rank"], expected_codec=settings["pca_codec"])
        if trace["status"] != "completed" or not trace["output_ids"]:
            raise ValueError("Receiver did not complete a generation")
        result = {"answer": trace["answer"], "output_tokens": len(trace["output_ids"]),
                  "injections": len(trace["injections"])}
    write_json(Path("/out/receipt.json"), {
        "role": args.role, "packet_bytes": len(packet), "packet_sha256": hashlib.sha256(packet).hexdigest(), **result
    })


if __name__ == "__main__":
    main()

