"""Hugging Face model client with explicit input, activation, and context contracts."""
from __future__ import annotations

import math
import hashlib
import json
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Iterator, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor

warnings.filterwarnings("ignore", message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization")
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class _FiniteLogitsProcessor(LogitsProcessor):
    """Reject invalid generation numerics without changing any valid score.

    Negative infinity is a legitimate token mask; NaN, positive infinity, or a
    row with no finite candidate cannot produce a valid completed generation.
    """
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(scores)
        invalid = ((~finite) & (~torch.isneginf(scores))).any() | (~finite.any(dim=-1)).any()
        if invalid:
            raise ValueError("invalid generation logits: NaN, positive infinity, or no finite candidate")
        return scores


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


class LLMClient:
    def __init__(self, name: str, revision: Optional[str] = None,
                 quantization: str = "int8", device: str = "cuda:0",
                 torch_dtype: str = "bfloat16", max_context_len: int = 8192,
                 generation: Optional[GenerationConfig] = None) -> None:
        self.name, self.revision, self.device = name, revision, device
        self.quantization = "none" if quantization is None else quantization
        self.torch_dtype = torch_dtype
        self.tokenizer_revision = revision
        self.trace_callback = None
        if not isinstance(name, str) or not name.strip():
            raise ValueError("model name must be a nonempty string")
        self.max_context_len = self._positive_integer(max_context_len, "max_context_len")
        self.generation = generation or GenerationConfig()
        self._generation_kwargs()
        if torch_dtype not in _DTYPE_MAP:
            raise ValueError(f"Unknown torch_dtype: {torch_dtype!r}")
        if quantization not in ("int8", "int4", "none", None):
            raise ValueError(f"Unknown quantization: {quantization!r}")
        self.loader_config = self._loader_identity({"model": {"name": name, "revision": revision,
            "quantization": self.quantization, "device": device, "torch_dtype": torch_dtype,
            "max_context_len": self.max_context_len}})
        self.loader_config_fingerprint = self.loader_fingerprint(self.loader_config)
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        bnb_config = None
        if quantization == "int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "int4":
            bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4")
        from inspect import signature
        kwargs = {"revision": revision, "quantization_config": bnb_config,
                  "device_map": "auto" if bnb_config is not None else device}
        params = signature(AutoModelForCausalLM.from_pretrained).parameters
        kwargs["dtype" if "dtype" in params else "torch_dtype"] = _DTYPE_MAP[torch_dtype]
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.model.eval()
        self.hf_device_map = {key: str(value) for key, value in getattr(self.model, "hf_device_map", {}).items()}

    @staticmethod
    def _loader_identity(cfg: dict) -> dict:
        """Canonical loading identity; generation is validated separately at use."""
        allowed = {"model": {"name", "revision", "quantization", "torch_dtype", "device", "max_context_len"},
                   "tokenizer": {"name", "revision"},
                   "generation": {"max_new_tokens", "temperature", "top_p", "do_sample"}}
        for namespace, keys in allowed.items():
            values = cfg.get(namespace, {})
            values = {} if values is None else values
            if not isinstance(values, dict):
                raise ValueError(f"{namespace} configuration must be a mapping")
            unexpected = set(values) - keys
            if unexpected:
                raise ValueError(f"unsupported {namespace} configuration keys: {sorted(unexpected)}")
        model = cfg["model"]
        tokenizer = cfg.get("tokenizer", {}) or {}
        quantization = model.get("quantization", "int8")
        return {"model": {"name": model["name"], "revision": model.get("revision"),
                "quantization": "none" if quantization is None else quantization,
                "torch_dtype": model.get("torch_dtype", "bfloat16"),
                "device": model.get("device", "cuda:0"),
                "max_context_len": model.get("max_context_len", 8192)},
                "tokenizer": {"name": tokenizer.get("name", model["name"]),
                              "revision": tokenizer.get("revision", model.get("revision"))}}

    @classmethod
    def loader_fingerprint(cls, cfg: dict) -> str:
        data = json.dumps(cls._loader_identity(cfg), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "LLMClient":
        cls._loader_identity(cfg)
        m = cfg["model"]
        tokenizer = cfg.get("tokenizer", {}) or {}
        if tokenizer.get("name", m["name"]) != m["name"] or tokenizer.get("revision", m.get("revision")) != m.get("revision"):
            raise ValueError("this client requires matching model and tokenizer names/revisions")
        gen = cfg.get("generation", {}) or {}
        return cls(name=m["name"], revision=m.get("revision"),
            quantization=m.get("quantization", "int8"), device=m.get("device", "cuda:0"),
            torch_dtype=m.get("torch_dtype", "bfloat16"),
            max_context_len=m.get("max_context_len", 8192),
            generation=GenerationConfig(max_new_tokens=gen.get("max_new_tokens", 512),
                temperature=gen.get("temperature", 0.0), top_p=gen.get("top_p", 1.0),
                do_sample=gen.get("do_sample", False)))

    @classmethod
    def from_config(cls, config_path: Path | str) -> "LLMClient":
        return cls.from_config_dict(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")))

    @staticmethod
    def _positive_integer(value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    def _generation_kwargs(self) -> dict:
        gen = self.generation
        count = self._positive_integer(gen.max_new_tokens, "max_new_tokens")
        if not isinstance(gen.do_sample, bool):
            raise ValueError("do_sample must be boolean")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               for value in (gen.temperature, gen.top_p)):
            raise ValueError("temperature and top_p must be real numbers")
        if not math.isfinite(float(gen.temperature)) or gen.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if gen.do_sample and gen.temperature <= 0:
            raise ValueError("sampling requires positive temperature")
        if not math.isfinite(float(gen.top_p)) or not 0 < gen.top_p <= 1:
            raise ValueError("top_p must be finite and in (0, 1]")
        return {"max_new_tokens": count, "temperature": gen.temperature if gen.do_sample else 1.0,
                "top_p": gen.top_p, "do_sample": gen.do_sample,
                "logits_processor": [_FiniteLogitsProcessor()]}

    def _input_device(self):
        # A sharded model's embedding device is more specific than model.device.
        if hasattr(self.model, "get_input_embeddings"):
            weight = getattr(self.model.get_input_embeddings(), "weight", None)
            if weight is not None and weight.device.type != "meta":
                return weight.device
        return self.model.device

    def _check_context(self, length: int, prefix: int = 0, generation: bool = False) -> None:
        maximum = self._positive_integer(self.max_context_len, "max_context_len")
        if length <= 0:
            raise ValueError("empty token sequence")
        total = length + prefix
        if total > maximum:
            raise ValueError(f"input context overflow: {total} tokens (including {prefix} virtual tokens) exceeds {maximum}; prompt was not truncated")
        model_maximum = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if isinstance(model_maximum, Integral) and model_maximum > 0:
            required = total + (self._generation_kwargs()["max_new_tokens"] if generation else 0)
            if required > model_maximum:
                raise ValueError(f"model context overflow: input plus requested generation requires {required} positions, model supports {model_maximum}")

    def _chat_inputs(self, messages: list[dict[str, str]], prefix: int = 0):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # The rendered template already owns BOS/EOS and role delimiters.
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
        self._check_context(inputs.input_ids.shape[1], prefix=prefix, generation=True)
        return inputs.to(self._input_device())

    def _text_inputs(self, text: str):
        # Raw-text feature extraction retains its explicit truncation contract.
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=self.max_context_len)
        self._check_context(inputs.input_ids.shape[1])
        return inputs.to(self._input_device())

    def _trace_generation(self, method: str, messages, inputs, output: str,
                          vectors: Optional[torch.Tensor] = None, **extra) -> None:
        callback = getattr(self, "trace_callback", None)
        if callback is None:
            return
        event = {"method": method, "messages": messages,
                 "input_ids": inputs.input_ids.detach().cpu().tolist(),
                 "input_tokens": int(inputs.input_ids.shape[1]),
                 "max_new_tokens": self.generation.max_new_tokens,
                 "output_text": output, **extra}
        if vectors is not None:
            value = vectors.detach().float().cpu().contiguous()
            event.update({"injected_vector_shape": list(value.shape),
                "injected_vector_count": 1 if value.dim() == 1 else value.shape[0],
                "injected_vector_finite": bool(torch.isfinite(value).all()),
                "injected_vector_sha256": hashlib.sha256(value.numpy().astype("<f4", copy=False).tobytes()).hexdigest(),
                "injected_vector_hash_encoding": "float32-little-endian"})
        callback(event)

    @staticmethod
    def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
        if pooling not in ("mean", "last_token", "cls"):
            raise ValueError(f"Unknown pooling: {pooling!r}")
        if hidden.dim() != 3 or tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
            raise ValueError("hidden states and attention mask shapes do not match")
        mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
        if hidden.shape[1] == 0 or not mask.any(dim=1).all():
            raise ValueError("cannot pool a sequence with no nonpadding tokens")
        if not torch.isfinite(hidden).all():
            raise ValueError("model produced nonfinite hidden states")
        if pooling == "mean":
            weights = mask.unsqueeze(-1)
            pooled = (hidden.double() * weights).sum(1) / weights.sum(1)
        elif pooling == "last_token":
            positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(mask)
            last = positions.masked_fill(~mask, -1).max(dim=1).values
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        else:
            pooled = hidden[:, 0]
        pooled = pooled.squeeze(0).float()
        if not torch.isfinite(pooled).all():
            raise ValueError("pooled activation cannot be represented as finite float32")
        return pooled

    @torch.inference_mode()
    def generate(self, messages: list[dict[str, str]]) -> str:
        inputs = self._chat_inputs(messages)
        out = self.model.generate(**inputs, **self._generation_kwargs(), pad_token_id=self.tokenizer.pad_token_id)
        answer = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        self._trace_generation("generate", messages, inputs, answer)
        return answer

    @torch.inference_mode()
    def embed_text(self, text: str, pooling: str = "mean") -> torch.Tensor:
        inputs = self._text_inputs(text)
        hidden = self.model.get_input_embeddings()(inputs.input_ids)
        return self._pool(hidden, inputs.attention_mask, pooling)

    @torch.inference_mode()
    def generate_with_prefix_embeds(self, messages: list[dict[str, str]],
                                    prefix_embeds: torch.Tensor) -> str:
        if prefix_embeds.dim() == 1:
            prefix_embeds = prefix_embeds.unsqueeze(0)
        if prefix_embeds.dim() != 2 or prefix_embeds.shape[0] < 1:
            raise ValueError("prefix_embeds must have shape (n>=1, hidden_dim)")
        if torch.is_complex(prefix_embeds) or not torch.isfinite(prefix_embeds).all():
            raise ValueError("prefix embeddings must be finite")
        inputs = self._chat_inputs(messages, prefix=prefix_embeds.shape[0])
        prompt = self.model.get_input_embeddings()(inputs.input_ids)
        if prefix_embeds.shape[1] != prompt.shape[-1]:
            raise ValueError("prefix hidden dimension does not match model")
        prefix = prefix_embeds.unsqueeze(0).to(device=prompt.device, dtype=prompt.dtype)
        if not torch.isfinite(prefix).all():
            raise ValueError("prefix embeddings overflow the model dtype")
        full = torch.cat([prefix, prompt], dim=1)
        mask = torch.cat([torch.ones((1, prefix.shape[1]), device=inputs.attention_mask.device,
                                    dtype=inputs.attention_mask.dtype), inputs.attention_mask], dim=1)
        out = self.model.generate(inputs_embeds=full, attention_mask=mask,
                                 **self._generation_kwargs(), pad_token_id=self.tokenizer.pad_token_id)
        answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
        self._trace_generation("generate_with_prefix_embeds", messages, inputs, answer, prefix_embeds,
                               prefix_tokens=prefix.shape[1], total_input_tokens=full.shape[1])
        return answer

    @torch.inference_mode()
    def hidden_state(self, text: str, layer_index: int, pooling: str = "mean") -> torch.Tensor:
        """Legacy HF tuple-index API (0 is embeddings; last is final normalized state)."""
        n = len(self._decoder_layers())
        if isinstance(layer_index, bool) or not isinstance(layer_index, Integral) or not 0 <= layer_index <= n:
            raise IndexError(f"hidden-state tuple index must be in [0, {n}]")
        inputs = self._text_inputs(text)
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        return self._pool(out.hidden_states[layer_index], inputs.attention_mask, pooling)

    def _decoder_layers(self) -> torch.nn.ModuleList:
        for path in (("model", "layers"), ("transformer", "h"), ("layers",)):
            obj = self.model
            for name in path:
                obj = getattr(obj, name, None)
                if obj is None:
                    break
            if isinstance(obj, torch.nn.ModuleList):
                return obj
        raise AttributeError("Could not locate decoder layers: model.layers / transformer.h / layers")

    def _decoder_block(self, layer_index: int):
        layers = self._decoder_layers()
        if isinstance(layer_index, bool) or not isinstance(layer_index, Integral) or not 0 <= layer_index < len(layers):
            raise IndexError(f"layer_index={layer_index} out of range; model has {len(layers)} decoder blocks")
        return layers[layer_index]

    @torch.inference_mode()
    def extract_activation(self, text: str, layer_index: int, pooling: str = "last_token") -> torch.Tensor:
        """Capture the raw OUTPUT of the indexed decoder block, including the last.

        The final HF hidden-state tuple entry has already passed through terminal
        normalization. A block hook guarantees the extraction/injection points
        match and is removed even if forwarding or pooling raises.
        """
        block = self._decoder_block(layer_index)
        inputs = self._text_inputs(text)
        captured = []
        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(self._pool(hidden, inputs.attention_mask, pooling).clone())
        handle = block.register_forward_hook(capture)
        try:
            self.model(**inputs, return_dict=True)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"expected one decoder-block output, observed {len(captured)}")
        return captured[0]

    @contextmanager
    def _injection_hook(self, layer_index: int, injected_vectors: torch.Tensor,
                        injection_strength: float = 1.0) -> Iterator[None]:
        """Add a normalized summed direction once, at the first prefill forward.

        Incoming vectors may persist in agent memory, but each generation gets
        a fresh hook. Subsequent cached decoding steps are never reinjected.
        The first prefill also works when the prompt has only one token.
        """
        block = self._decoder_block(layer_index)
        if not math.isfinite(float(injection_strength)):
            raise ValueError("injection_strength must be finite")
        vectors = injected_vectors.unsqueeze(0) if injected_vectors.dim() == 1 else injected_vectors
        if vectors.dim() != 2 or min(vectors.shape) < 1:
            raise ValueError("injected_vectors must have shape (hidden_dim,) or (n>=1, hidden_dim)")
        if torch.is_complex(vectors) or not torch.isfinite(vectors).all():
            raise ValueError("injected vectors must be finite")
        # Accumulate in fp64 and scale before computing the norm. This preserves
        # direction for finite fp32 values whose direct fp32 norm overflows.
        summed = vectors.double().sum(dim=0)
        if not torch.isfinite(summed).all():
            raise ValueError("summed injected vectors overflow")
        maximum = summed.abs().max()
        direction = summed if maximum == 0 else summed / maximum
        if maximum != 0:
            direction = direction / torch.linalg.vector_norm(direction)
        prefill_seen = False
        def hook(_module, _inputs, output):
            nonlocal prefill_seen
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() != 3 or hidden.shape[-1] != direction.shape[0]:
                raise ValueError("injected vector and decoder output shapes do not match")
            if prefill_seen:
                return output
            prefill_seen = True
            if not torch.isfinite(hidden).all():
                raise ValueError("nonfinite residual before injection")
            if maximum == 0 or injection_strength == 0:
                return output
            norm = torch.linalg.vector_norm(hidden.double(), dim=-1).mean()
            add = (direction.to(hidden.device) * norm * float(injection_strength)).to(hidden.dtype)
            changed = hidden + add.view(1, 1, -1)
            if not torch.isfinite(changed).all():
                raise ValueError("activation injection produced nonfinite residuals")
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        handle = block.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.inference_mode()
    def generate_with_activation_injection(self, messages: list[dict[str, str]],
            injected_vectors: torch.Tensor, layer_index: int, injection_strength: float = 1.0) -> str:
        inputs = self._chat_inputs(messages)
        with self._injection_hook(layer_index, injected_vectors, injection_strength):
            out = self.model.generate(**inputs, **self._generation_kwargs(), pad_token_id=self.tokenizer.pad_token_id)
        answer = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        self._trace_generation("generate_with_activation_injection", messages, inputs, answer, injected_vectors,
                               injection_layer=int(layer_index), injection_strength=float(injection_strength))
        return answer
