"""Secret-Passing task — a TRANSMISSION-MANDATORY benchmark.

Motivation: Multi-Doc QA cannot distinguish a broken channel from an unused one,
because the receiving agent can often answer from its own passage or the model's
parametric knowledge. This task removes that confound: the answer is a uniformly
random secret that exists ONLY in the sender's private context and is impossible
for any LLM to guess. The receiver can only succeed if information actually
crosses the channel.

Two secret modes (config `secret_mode`):
  - "digit":   a random 4-digit access code. ~13 bits of entropy, trivial to copy.
  - "binding": a random entity->attribute->value binding with made-up tokens, e.g.
               "the codeword of XQZTR is ZELPH". Tests copying an opaque token AND
               resolving which value the (public) question's entity refers to.
  - "both":    alternate digit/binding per episode (default).

Scoring: exact-match — the secret must appear verbatim (case-insensitive) in the
receiver's final answer. The runner is responsible for scoring the RECEIVER's
answer (agent_b), not the sender's (agent_a trivially knows the secret).

Baselines this task enables:
  - text channel, high budget  -> ~100% (upper bound; proves task is solvable by transmission)
  - no communication (budget 0) -> ~0% (floor; proves task REQUIRES the channel)
  - vector real vs random       -> the decisive capacity test
"""
from __future__ import annotations

import random
import string

from .base import Episode, Task


_DIGITS = "0123456789"
_ATTRS = ["codeword", "passcode", "guardian", "sigil", "cipher"]


class SecretPassingTask(Task):
    name = "secret_passing"

    def __init__(self, secret_mode: str = "both", n_digits: int = 4) -> None:
        if secret_mode not in ("digit", "binding", "both"):
            raise ValueError(f"secret_mode must be digit|binding|both, got {secret_mode!r}")
        self.secret_mode = secret_mode
        self.n_digits = n_digits
        # Runner references task.dataset_id for metadata/logging.
        self.dataset_id = f"secret_{secret_mode}"

    def _rand_token(self, rng: random.Random, length: int = 5) -> str:
        # Made-up uppercase token — un-guessable, not in any vocabulary.
        return "".join(rng.choice(string.ascii_uppercase) for _ in range(length))

    def episodes(self, n: int, seed: int) -> list[Episode]:
        rng = random.Random(seed)
        out: list[Episode] = []
        for i in range(n):
            mode = self.secret_mode
            if mode == "both":
                mode = "digit" if i % 2 == 0 else "binding"

            if mode == "digit":
                secret = "".join(rng.choice(_DIGITS) for _ in range(self.n_digits))
                sender_text = (
                    "SECRET DATA — only you (the sender) can see this. "
                    f"The {self.n_digits}-digit access code is {secret}. "
                    "Communicate it to your partner so they can report it."
                )
                question = (
                    f"What is the {self.n_digits}-digit access code? "
                    "Output ONLY the digits, nothing else."
                )
            else:  # binding
                entity = self._rand_token(rng)
                value = self._rand_token(rng)
                attr = rng.choice(_ATTRS)
                secret = value
                sender_text = (
                    "SECRET DATA — only you (the sender) can see this. "
                    f"The {attr} of {entity} is {value}. "
                    "Communicate it to your partner so they can report it."
                )
                question = (
                    f"What is the {attr} of {entity}? "
                    "Output ONLY the answer token, nothing else."
                )

            out.append(
                Episode(
                    episode_id=f"secret_{mode}_{i:03d}",
                    initial_state={
                        "secret": secret,
                        "sender_text": sender_text,
                        "question": question,
                        "mode": mode,
                    },
                    ground_truth=secret,
                    metadata={"mode": mode},
                )
            )
        return out

    def score(self, episode: Episode, final_answer) -> tuple[bool, float]:
        if final_answer is None:
            return False, 0.0
        gold = str(episode.ground_truth).strip().upper()
        pred = str(final_answer).strip().upper()
        # The secret must appear verbatim in the receiver's answer.
        em = gold in pred
        return em, (1.0 if em else 0.0)
