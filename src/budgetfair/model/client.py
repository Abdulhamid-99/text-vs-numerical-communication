"""LLMClient — wraps a HuggingFace causal LM.

Owns:
  - tokenizer + chat template
  - quantized model load (bitsandbytes int8/int4)
  - chat-format generation (single call, returns the assistant string)
  - hidden-state extraction at a chosen transformer layer (for the vector channel)

One client wraps one model. For cross-family experiments, instantiate one client per
model and run them sequentially or on different GPUs.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# bitsandbytes prints this on every int8 matmul. It's expected behavior.
warnings.filterwarnings(
    "ignore",
    message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization",
)


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


class LLMClient:
    def __init__(
        self,
        name: str,
        revision: Optional[str] = None,
        quantization: str = "int8",          # "int8" | "int4" | "none"
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        max_context_len: int = 8192,
        generation: Optional[GenerationConfig] = None,
    ) -> None:
        self.name = name
        self.revision = revision
        self.device = device
        self.max_context_len = max_context_len
        self.generation = generation or GenerationConfig()

        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        bnb_config: Optional[BitsAndBytesConfig] = None
        if quantization == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization not in ("none", None):
            raise ValueError(f"Unknown quantization: {quantization!r}")

        dtype = _DTYPE_MAP.get(torch_dtype, torch.bfloat16)
        # When quantized, device_map="auto" is required by bitsandbytes; without
        # quantization we explicitly place on the configured device.
        # NB: transformers >=4.50 renamed `torch_dtype` to `dtype`. We pin to
        # transformers>=4.45 in pyproject.toml; in our docker image (4.57.6) the
        # new kwarg is canonical, in older 4.45-4.49 it is `torch_dtype`. Accept
        # both via a runtime probe.
        from inspect import signature
        from_pretrained_kwargs = {
            "revision": revision,
            "quantization_config": bnb_config,
            "device_map": "auto" if bnb_config is not None else device,
        }
        params = signature(AutoModelForCausalLM.from_pretrained).parameters
        from_pretrained_kwargs["dtype" if "dtype" in params else "torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(name, **from_pretrained_kwargs)
        self.model.eval()

    @classmethod
    def from_config(cls, config_path: Path | str) -> "LLMClient":
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        m = cfg["model"]
        gen = cfg.get("generation", {}) or {}
        return cls(
            name=m["name"],
            revision=m.get("revision"),
            quantization=m.get("quantization", "int8"),
            device=m.get("device", "cuda:0"),
            torch_dtype=m.get("torch_dtype", "bfloat16"),
            max_context_len=m.get("max_context_len", 8192),
            generation=GenerationConfig(
                max_new_tokens=gen.get("max_new_tokens", 512),
                temperature=gen.get("temperature", 0.0),
                top_p=gen.get("top_p", 1.0),
                do_sample=gen.get("do_sample", False),
            ),
        )

    @torch.inference_mode()
    def generate(self, messages: list[dict[str, str]]) -> str:
        """Run chat-format generation.

        `messages` follows the standard format:
            [{"role": "system" | "user" | "assistant", "content": "..."}, ...]
        Returns only the assistant's newly-generated string (prompt tokens stripped).
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_len,
        ).to(self.model.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.generation.max_new_tokens,
            temperature=self.generation.temperature if self.generation.do_sample else 1.0,
            top_p=self.generation.top_p,
            do_sample=self.generation.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @torch.inference_mode()
    def embed_text(self, text: str, pooling: str = "mean") -> torch.Tensor:
        """Mean-pool the input embeddings (layer-0 hidden state) of `text`.

        This is the *sender's* extraction step in the vector channel: the agent's
        message text -> mean of its token embeddings -> a single (hidden_dim,)
        vector that gets projected and transmitted.

        Why layer 0: the receiver injects a virtual token via `inputs_embeds`,
        which lives in input-embedding space. Extracting at layer 0 keeps the
        sender's vector in the same space the receiver injects into, avoiding
        a layer-mismatch when no learned cross-layer projection is available.
        """
        embed_layer = self.model.get_input_embeddings()
        ids = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_context_len
        ).to(self.model.device)
        token_embeds = embed_layer(ids.input_ids)  # (1, T, hidden_dim)
        mask = ids.attention_mask.unsqueeze(-1).to(token_embeds.dtype)

        if pooling == "mean":
            pooled = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif pooling == "last_token":
            lengths = ids.attention_mask.sum(dim=1) - 1
            pooled = token_embeds[torch.arange(token_embeds.size(0)), lengths]
        elif pooling == "cls":
            pooled = token_embeds[:, 0]
        else:
            raise ValueError(f"Unknown pooling: {pooling!r}")
        return pooled.squeeze(0).float()  # (hidden_dim,)

    @torch.inference_mode()
    def generate_with_prefix_embeds(
        self,
        messages: list[dict[str, str]],
        prefix_embeds: torch.Tensor,
    ) -> str:
        """Chat-format generate, but prepend `prefix_embeds` as virtual tokens.

        `prefix_embeds` shape: (n_prefix_tokens, hidden_dim) — N >= 1 virtual
        token embeddings (e.g., from receiver-side projection lifts of incoming
        vector messages). These are concatenated *before* the prompt's embedded
        token sequence and the model generates from the combined input via
        `inputs_embeds=...`.

        Returns the assistant's newly-generated string only (prompt + prefix
        tokens stripped from the output).
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_len,
        ).to(self.model.device)
        prompt_embeds = self.model.get_input_embeddings()(ids.input_ids)  # (1, T, d)

        if prefix_embeds.dim() == 1:
            prefix_embeds = prefix_embeds.unsqueeze(0)  # (1, d)
        prefix_embeds = prefix_embeds.unsqueeze(0).to(prompt_embeds.dtype).to(prompt_embeds.device)
        # (1, n_prefix, d) || (1, T, d) -> (1, n_prefix + T, d)
        full_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        # Build attention mask: ones for prefix and prompt
        n_prefix = prefix_embeds.shape[1]
        attn_mask = torch.cat(
            [
                torch.ones(1, n_prefix, dtype=ids.attention_mask.dtype, device=self.model.device),
                ids.attention_mask,
            ],
            dim=1,
        )

        out = self.model.generate(
            inputs_embeds=full_embeds,
            attention_mask=attn_mask,
            max_new_tokens=self.generation.max_new_tokens,
            temperature=self.generation.temperature if self.generation.do_sample else 1.0,
            top_p=self.generation.top_p,
            do_sample=self.generation.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # When using inputs_embeds, generate() returns ONLY the newly-generated tokens
        # (it does not echo the prompt). So we decode the whole output.
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @torch.inference_mode()
    def hidden_state(self, text: str, layer_index: int, pooling: str = "mean") -> torch.Tensor:
        """Run a single forward pass and return one pooled vector at a given layer.

        Used by the vector channel: the sender computes this on its private context;
        the bytes after quantization are what gets transmitted.

        pooling:
          - "mean":      mean over non-pad tokens
          - "last_token": last non-pad token
          - "cls":       first token (only meaningful for some models)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_len,
        ).to(self.model.device)

        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        # hidden_states is a tuple of length n_layers + 1 (embedding output is index 0).
        hidden = out.hidden_states[layer_index]  # (1, T, d)
        mask = inputs.attention_mask.unsqueeze(-1).to(hidden.dtype)  # (1, T, 1)

        if pooling == "mean":
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif pooling == "last_token":
            # Index of last non-pad token per row
            lengths = inputs.attention_mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), lengths]
        elif pooling == "cls":
            pooled = hidden[:, 0]
        else:
            raise ValueError(f"Unknown pooling: {pooling!r}")

        return pooled.squeeze(0)  # (d,)

    # ------------------------------------------------------------------
    # Activation injection (ref:1, "Communicating Activations Between LM
    # Agents"). The earlier vector channel injected a single lifted vector
    # as one inputs_embeds PREFIX TOKEN at layer 0; a random-vector ablation
    # proved that path transmits zero usable information (real == random).
    # Per ref:1 we instead inject the activation INTO THE RESIDUAL STREAM at
    # a middle decoder layer, which is a far stronger conditioning signal.
    # ------------------------------------------------------------------

    def _decoder_layers(self) -> torch.nn.ModuleList:
        """Return the ModuleList of decoder blocks for the loaded CausalLM.

        Both LlamaForCausalLM and Qwen2ForCausalLM expose them at
        ``model.model.layers``. We probe a couple of fallbacks so this does
        not silently break if the wrapper nesting changes across HF versions.
        """
        for attr_path in (("model", "layers"), ("transformer", "h"), ("layers",)):
            obj = self.model
            ok = True
            for attr in attr_path:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, torch.nn.ModuleList):
                return obj
        raise AttributeError(
            "Could not locate the decoder-layer ModuleList on this model; "
            "checked model.model.layers / model.transformer.h / model.layers."
        )

    @torch.inference_mode()
    def extract_activation(
        self,
        text: str,
        layer_index: int,
        pooling: str = "last_token",
    ) -> torch.Tensor:
        """Sender-side extraction for activation injection.

        Runs one forward pass and returns the residual-stream activation at the
        OUTPUT of decoder block ``layer_index`` (i.e. ``hidden_states[layer_index + 1]``
        in HF's output-hidden-states tuple, whose index 0 is the embedding layer).

        This is deliberately the SAME residual-stream point the receiver injects
        into: the receiver hooks ``model.model.layers[layer_index]`` and adds to
        that block's output, so extracting the sender's vector at the matching
        point keeps both sides in one space (no untrained cross-layer mapping).

        pooling:
          - "last_token" (default): the last non-pad token's activation. This is
            the standard choice for decoder-only LMs — the final token's residual
            has attended over the whole sequence and carries the summary state.
          - "mean": mean over non-pad tokens.
          - "cls": first token.

        Returns shape (hidden_dim,).
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_len,
        ).to(self.model.device)

        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        # hidden_states: tuple len n_layers+1; index 0 = embeddings,
        # index k = output of decoder block k-1. We want the OUTPUT of block
        # `layer_index`, which is hidden_states[layer_index + 1].
        hs_index = layer_index + 1
        if hs_index >= len(out.hidden_states):
            raise IndexError(
                f"layer_index={layer_index} out of range; model has "
                f"{len(out.hidden_states) - 1} decoder layers."
            )
        hidden = out.hidden_states[hs_index]  # (1, T, d)
        mask = inputs.attention_mask.unsqueeze(-1).to(hidden.dtype)  # (1, T, 1)

        if pooling == "mean":
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif pooling == "last_token":
            lengths = inputs.attention_mask.sum(dim=1) - 1
            # arange must be on hidden's device — lengths lives on cuda; a CPU
            # arange here triggers "cuda:0 and cpu" device-mismatch indexing.
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        elif pooling == "cls":
            pooled = hidden[:, 0]
        else:
            raise ValueError(f"Unknown pooling: {pooling!r}")

        return pooled.squeeze(0).float()  # (d,)

    @contextmanager
    def _injection_hook(
        self,
        layer_index: int,
        injected_vectors: torch.Tensor,
        injection_strength: float = 1.0,
    ) -> Iterator[None]:
        """Register a forward hook that ADDS injected vectors to the residual
        stream output of decoder block ``layer_index``, and remove it on exit.

        Scaling (CRITICAL): an unscaled lifted vector is either negligible or
        destroys generation. Each injected vector is L2-normalized and rescaled
        to (mean per-token residual norm at this layer) * injection_strength, so
        its magnitude is commensurate with the activations already flowing
        through the layer. injection_strength=1.0 makes the injected signal about
        as large as one typical token's activation.

        Injection position: we add the vector to ALL token positions of the
        PREFILL pass (output.shape[1] > 1) and skip single-token decode steps.
        Broadcasting across prompt positions (rather than only the last token)
        gives a persistent, distributed conditioning signal — consistent with
        ref:1's residual-stream injection — while skipping decode steps avoids
        re-adding the same vector at every newly generated token (which would
        compound and derail decoding).
        """
        layers = self._decoder_layers()
        if not (0 <= layer_index < len(layers)):
            raise IndexError(
                f"layer_index={layer_index} out of range; model has {len(layers)} layers."
            )
        block = layers[layer_index]

        # Sum the injected vectors into a single (d,) injection (multiple received
        # messages combine additively in the residual stream). Kept on its source
        # device here; moved to the residual stream's device inside the hook
        # (with device_map="auto" the two can differ — CPU vs cuda:0).
        if injected_vectors.dim() == 1:
            injected_vectors = injected_vectors.unsqueeze(0)  # (1, d)
        inj = injected_vectors.sum(dim=0).float()  # (d,)

        def hook(_module, _inputs, output):  # noqa: ANN001
            # transformers >=4.54 decoder blocks return a bare tensor; older
            # versions returned a tuple (hidden_states, ...). Handle both.
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None
            # hidden: (batch, seq, d). Only inject on the prefill (seq > 1).
            if hidden.dim() != 3 or hidden.shape[1] <= 1:
                return output
            # Align the injection with the residual stream's device.
            inj_dev = inj.to(hidden.device)
            inj_norm = torch.linalg.vector_norm(inj_dev)
            if inj_norm <= 0:
                return output
            # Per-token residual norm at this layer (mean over batch+seq); compute
            # in fp32 for numerical stability under int8/bf16 activations.
            tok_norm = torch.linalg.vector_norm(hidden.float(), dim=-1).mean()
            target_mag = tok_norm * float(injection_strength)
            add = (inj_dev / inj_norm * target_mag).to(hidden.dtype)
            # Broadcast-add to every position.
            hidden = hidden + add.view(1, 1, -1)
            if rest is None:
                return hidden
            return (hidden, *rest)

        handle = block.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.inference_mode()
    def generate_with_activation_injection(
        self,
        messages: list[dict[str, str]],
        injected_vectors: torch.Tensor,
        layer_index: int,
        injection_strength: float = 1.0,
    ) -> str:
        """Chat-format generate while injecting activation(s) at a middle layer.

        ``injected_vectors`` shape: (d,) or (n, d) — one or more received vectors
        already lifted to hidden_dim. They are summed and added to the residual
        stream output of decoder block ``layer_index`` during the prefill pass
        (see ``_injection_hook`` for scaling and position rationale). The hook is
        always removed in a finally block.

        Returns the assistant's newly-generated string only.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_len,
        ).to(self.model.device)

        with self._injection_hook(layer_index, injected_vectors, injection_strength):
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.generation.max_new_tokens,
                temperature=self.generation.temperature if self.generation.do_sample else 1.0,
                top_p=self.generation.top_p,
                do_sample=self.generation.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
