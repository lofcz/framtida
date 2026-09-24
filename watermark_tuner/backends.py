"""Sampling backends: Tinker (hosted) or a local vLLM OpenAI-compatible server.

Every consumer (rejection sampling, evaluation, inference) talks to a
``Sampler`` and never to the transport, so switching between Tinker and rented
GPUs is a CLI flag.

vLLM usage:
    vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 2 --max-num-seqs 512 --port 8000
    # with a trained LoRA adapter:
    vllm serve Qwen/Qwen3.8-27B --enable-lora --lora-modules wm=logs/rl_round1/final --port 8000
    python -m watermark_tuner.eval.evaluate --backend vllm --vllm-model wm ...
"""

from __future__ import annotations

import argparse
import asyncio
import re
from dataclasses import dataclass
from typing import Protocol

from watermark_tuner import DEFAULT_MODEL, DEFAULT_RENDERER

_THINK_PREFIX = re.compile(r"^\s*<think>\s*</think>\s*")


@dataclass(frozen=True)
class SampleOut:
    text: str
    clean: bool  # terminated on EOS / stop sequence (not truncated by max_tokens)


class Sampler(Protocol):
    async def sample(
        self, messages: list[dict], n: int, max_tokens: int, temperature: float, top_p: float = 1.0
    ) -> list[SampleOut]: ...


class TinkerSampler:
    """Samples from a Tinker base model or a ``tinker://`` sampler checkpoint."""

    def __init__(
        self,
        model_path: str | None = None,
        base_model: str = DEFAULT_MODEL,
        renderer_name: str = DEFAULT_RENDERER,
        user_metadata: dict[str, str] | None = None,
    ):
        import tinker

        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self.tinker = tinker
        self.tokenizer = get_tokenizer(base_model)
        self.renderer = renderers.get_renderer(renderer_name, self.tokenizer)
        service = tinker.ServiceClient(user_metadata=user_metadata)
        if model_path:
            self.client = service.create_sampling_client(model_path=model_path)
        else:
            self.client = service.create_sampling_client(base_model=base_model)

    async def sample(self, messages, n, max_tokens, temperature, top_p=1.0) -> list[SampleOut]:
        from tinker_cookbook.renderers import get_text_content

        prompt = self.renderer.build_generation_prompt(messages)
        params = self.tinker.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=self.renderer.get_stop_sequences()
        )
        res = await self.client.sample_async(prompt=prompt, num_samples=n, sampling_params=params)
        outs = []
        for seq in res.sequences:
            message, termination = self.renderer.parse_response(seq.tokens)
            outs.append(SampleOut(text=get_text_content(message).strip(), clean=bool(termination.is_clean)))
        return outs


class VLLMChatSampler:
    """Samples from an OpenAI-compatible ``/v1/chat/completions`` endpoint (vLLM, SGLang, TGI)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        enable_thinking: bool = False,
        timeout_s: float = 1800.0,
        max_connections: int = 64,
        extra_body: dict | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.enable_thinking = enable_thinking
        self.extra_body = dict(extra_body or {})  # e.g. {"vllm_xargs": {"wm_delta": 2.0}}
        self.timeout_s = timeout_s
        self.max_connections = max_connections
        self._session = None
        self._lock = asyncio.Lock()

    async def _get_session(self):
        import aiohttp

        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                    self._session = aiohttp.ClientSession(
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                        connector=aiohttp.TCPConnector(limit=self.max_connections),
                    )
        return self._session

    @staticmethod
    def parse_response(payload: dict) -> list[SampleOut]:
        outs = []
        for choice in payload.get("choices", []):
            msg = choice.get("message") or {}
            text = msg.get("content") or ""
            text = _THINK_PREFIX.sub("", text).strip()
            outs.append(SampleOut(text=text, clean=choice.get("finish_reason") == "stop"))
        return outs

    async def sample(self, messages, n, max_tokens, temperature, top_p=1.0) -> list[SampleOut]:
        session = await self._get_session()
        body = {
            "model": self.model,
            "messages": messages,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            **self.extra_body,
        }
        async with session.post(f"{self.base_url}/chat/completions", json=body) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        return self.parse_response(payload)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


def add_backend_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("sampling backend")
    g.add_argument("--backend", choices=["tinker", "vllm"], default="tinker")
    g.add_argument("--model", default=DEFAULT_MODEL, help="HF model id (tokenizer + Tinker base model)")
    g.add_argument("--renderer", default=DEFAULT_RENDERER, help="tinker-cookbook renderer (tinker backend)")
    g.add_argument("--model-path", default=None, help="tinker:// sampler checkpoint (tinker backend)")
    g.add_argument("--vllm-url", default="http://localhost:8000/v1", help="OpenAI-compatible base URL (vllm backend)")
    g.add_argument("--vllm-model", default=None, help="served model name / LoRA module name (defaults to --model)")
    g.add_argument("--vllm-api-key", default=None)
    g.add_argument(
        "--wm-delta", type=float, default=None,
        help="vllm backend: tilt strength sent as vllm_xargs.wm_delta (server must run RedTiltLogitsProcessor); "
        "0 forces the tilt off, None sends nothing (server default)",
    )  # fmt: skip


def make_sampler(args: argparse.Namespace, recipe: str = "wm") -> Sampler:
    if args.backend == "vllm":
        extra = None
        if getattr(args, "wm_delta", None) is not None:
            extra = {"vllm_xargs": {"wm_delta": float(args.wm_delta)}}
        return VLLMChatSampler(
            base_url=args.vllm_url, model=args.vllm_model or args.model, api_key=args.vllm_api_key, extra_body=extra
        )
    return TinkerSampler(
        model_path=args.model_path, base_model=args.model, renderer_name=args.renderer, user_metadata={"recipe": recipe}
    )


async def close_sampler(sampler: Sampler) -> None:
    close = getattr(sampler, "close", None)
    if close is not None:
        await close()
