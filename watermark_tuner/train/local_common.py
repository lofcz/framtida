"""Shared helpers for the local (TRL + PEFT) trainers."""

from __future__ import annotations

import os

import torch

# Qwen3.8's chat template accepts enable_thinking; we always train and sample with it off so the
# assistant turn starts right after the empty <think></think> block (same as the Tinker renderer).
CHAT_KWARGS: dict = {"enable_thinking": False}

# Adapter goes on every linear layer of the language model (attention, gated-deltanet, MLP);
# vision tower and lm_head stay frozen.
LORA_EXCLUDE_REGEX = r".*(visual|vision|image|mtp).*"


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def lora_config(rank: int = 32, alpha: int | None = None, dropout: float = 0.0):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha if alpha is not None else rank,
        lora_dropout=dropout,
        bias="none",
        target_modules="all-linear",
        exclude_modules=LORA_EXCLUDE_REGEX,
        task_type="CAUSAL_LM",
    )


def load_model(model_name: str, load_4bit: bool = False, attn_implementation: str | None = None):
    """Load the base model in bf16 with the architecture named in its config.

    Qwen3.8-27B is a ``Qwen3_5ForConditionalGeneration`` (vision-language) checkpoint; loading it
    through the image-text-to-text head keeps parameter names identical to what vLLM serves, which
    matters for GRPO weight syncing. Text-only inputs work unchanged.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    kwargs: dict = {"dtype": torch.bfloat16}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    cfg = AutoConfig.from_pretrained(model_name)
    archs = getattr(cfg, "architectures", None) or []
    cls = AutoModelForImageTextToText if any("ConditionalGeneration" in a for a in archs) else AutoModelForCausalLM
    model = cls.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False
    return model


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def completion_text(completion) -> str:
    """TRL passes completions as str (text prompts) or [{'role': 'assistant', 'content': ...}]."""
    if isinstance(completion, str):
        text = completion
    else:
        parts = []
        for m in completion:
            c = m.get("content", "")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            parts.append(c or "")
        text = "".join(parts)
    # Defensive: strip an empty think block if a template ever leaks it into the content.
    if text.lstrip().startswith("<think>"):
        end = text.find("</think>")
        if end != -1:
            text = text[end + len("</think>") :]
    return text.strip()
