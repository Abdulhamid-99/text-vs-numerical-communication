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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
