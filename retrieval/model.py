"""Resolve the recorded checkpoint and check each input used by the encoder."""

from pathlib import Path
import hashlib
import json
import os

PROTOCOL = json.loads(
    Path(__file__).with_name("protocol.json").read_text(encoding="utf-8")
)


def canonical_hash(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def verify_idf(value):
    if canonical_hash(value) != PROTOCOL["idf_sha256_canonical"]:
        raise ValueError(
            "IDF differs from the dictionary fitted on the original training documents."
        )


def check_file(path, expected_hash):
    with Path(path).open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected_hash:
        raise ValueError(f"Unexpected SHA-256 for {Path(path).name}: {actual}")


def model_load():
    import torch
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download

    spec = PROTOCOL["model"]
    if os.environ.get("BGE_MODEL_DIR"):
        snapshot = Path(os.environ["BGE_MODEL_DIR"])
    else:
        settings = dict(
            repo_id=spec["name"],
            revision=spec["revision"],
            allow_patterns=list(spec["files"]),
        )
        try:
            snapshot = Path(snapshot_download(**settings, local_files_only=True))
        except FileNotFoundError:
            snapshot = None
        if snapshot is None or any(
            not (snapshot / name).is_file() for name in spec["files"]
        ):
            snapshot = Path(snapshot_download(**settings))
    for name, expected in spec["files"].items():
        check_file(snapshot / name, expected)
    torch.set_num_threads(6)
    torch.manual_seed(73021)
    device = os.environ.get(
        "BGE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = SentenceTransformer(str(snapshot), device=device, local_files_only=True)
    model.max_seq_length = 512
    return model
