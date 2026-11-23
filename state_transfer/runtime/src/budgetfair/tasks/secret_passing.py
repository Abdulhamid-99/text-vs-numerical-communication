"""Private record copying with a designated receiver and strict exact matching.

Seeded generation supports development/compatibility. Confirmatory evaluation
loads frozen independently generated manifests. Four-digit records have finite
chance performance; task accuracy alone is not an information-capacity estimate.
"""
from __future__ import annotations
import hashlib
import json
import random
import string
from .base import Episode, Task
from .multidoc_qa import _count

GENERATOR_VERSION = "private-record-v3-public-key"
SCORER_VERSION = "secret-exact-casefold-v2"

def format_record_passage(key, value):
    """Keep the literal value boundary independent of prose punctuation."""
    return (f"Lookup record:\nItem: {key}\nValue: {value}\n"
            "Share only the exact recorded value with the other agent. Do not add punctuation or labels.")


def instance_hash(state, gold):
    return hashlib.sha256(json.dumps({"state": state, "gold": gold}, sort_keys=True,
                                    ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

class SecretPassingTask(Task):
    name = "secret_passing"
    scorer_version = SCORER_VERSION

    def __init__(self, secret_mode="both", n_digits=4, n_letters=5, *, manifest_path=None, split="dev"):
        if secret_mode not in ("digit", "binding", "both"):
            raise ValueError("secret_mode must be digit, binding, or both")
        for name, value in (("n_digits", n_digits), ("n_letters", n_letters)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 256:
                raise ValueError(f"{name} must be an integer in 1..256")
        if split not in ("dev", "test"):
            raise ValueError("split must be dev or test")
        self.secret_mode, self.n_digits, self.n_letters = secret_mode, n_digits, n_letters
        self.manifest_path, self.split = manifest_path, split
        self.dataset_id = f"private_record_{secret_mode}"
        self.manifest_hash = None

    def _rand_token(self, rng, length=5):
        return "".join(rng.choice(string.ascii_uppercase) for _ in range(length))

    def episodes(self, n, seed):
        _count(n)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if self.manifest_path is not None:
            from .manifest import load_task_manifest, manifest_info
            info = manifest_info(self.manifest_path)
            for key in ("secret_mode", "n_digits", "n_letters"):
                if info["generator"][key] != getattr(self, key):
                    raise ValueError(f"manifest {key} differs from task configuration")
            out = load_task_manifest(self.manifest_path, split=self.split,
                                     task_name=self.name, expected_count=n)
            self.manifest_hash = info["manifest_hash"]
            self.dataset_id = f"private_record_manifest:{self.manifest_hash[:16]}:{self.split}"
            return out
        if self.split == "test":
            raise ValueError("test evaluation requires a frozen task manifest")
        rng, out = random.Random(seed), []
        for i in range(n):
            mode = ("digit" if i % 2 == 0 else "binding") if self.secret_mode == "both" else self.secret_mode
            if mode == "digit":
                value = "".join(rng.choice(string.digits) for _ in range(self.n_digits))
                sender = f"Copying record. The {self.n_digits}-digit access code is {value}. Share the exact digits."
                question = "What value is in your partner's copying record? Output only the digits."
                if value in question:
                    raise ValueError("target overlaps public question; use longer values")
            else:
                entity = self._rand_token(rng, 5)
                question = f"What value is recorded for item {entity}? Output only the value."
                for _ in range(1000):
                    value = self._rand_token(rng, self.n_letters)
                    if value.casefold() not in question.casefold():
                        break
                else:
                    raise ValueError("could not sample target absent from public context")
                sender = format_record_passage(entity, value)
            state = {"secret": value, "sender_text": sender, "question": question, "mode": mode}
            if mode == "binding":
                state["record_key"] = entity
            digest = instance_hash(state, value)
            out.append(Episode(f"secret_{mode}_dev_{seed}_{i:06d}_{digest}", state, value, {
                "mode": mode, "split": "dev", "generator_version": GENERATOR_VERSION,
                "generation_seed": seed, "generation_index": i}))
        self.manifest_hash = hashlib.sha256(json.dumps([e.episode_id for e in out]).encode()).hexdigest()
        return out

    def score(self, episode, final_answer):
        gold = episode.ground_truth
        if not isinstance(gold, str) or not gold.strip():
            raise ValueError("secret target must be a nonempty string")
        if not isinstance(final_answer, str):
            return False, 0.0
        correct = gold.strip().casefold() == final_answer.strip().casefold()
        return correct, float(correct)
