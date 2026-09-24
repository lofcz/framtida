"""Shared helpers: JSONL IO and Tinker client construction."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator

import tinker

from watermark_tuner import DEFAULT_MODEL, DEFAULT_RENDERER


def read_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def make_sampling_client(
    model_path: str | None = None,
    base_model: str = DEFAULT_MODEL,
    user_metadata: dict[str, str] | None = None,
) -> tinker.SamplingClient:
    """Sampling client for a Tinker checkpoint (``tinker://...``) or the base model."""
    service = tinker.ServiceClient(user_metadata=user_metadata)
    if model_path:
        return service.create_sampling_client(model_path=model_path)
    return service.create_sampling_client(base_model=base_model)


def load_tokenizer(model_name: str = DEFAULT_MODEL):
    """Plain HF tokenizer (no tinker-cookbook dependency); used for scoring on any backend."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def make_tokenizer_and_renderer(model_name: str = DEFAULT_MODEL, renderer_name: str = DEFAULT_RENDERER):
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return tokenizer, renderer


def response_text(renderer, tokens: list[int]) -> tuple[str, bool]:
    """Decode a sampled sequence into assistant text; returns (text, terminated_cleanly)."""
    from tinker_cookbook.renderers import get_text_content

    message, termination = renderer.parse_response(tokens)
    return get_text_content(message).strip(), bool(termination.is_clean)
