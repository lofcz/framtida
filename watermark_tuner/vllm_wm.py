"""vLLM logits processor that turns any served model into the tilted watermark teacher.

    WM_PASSKEY=... WM_DELTA=2.0 vllm serve Qwen/Qwen3.8-27B \
        --logits-processors watermark_tuner.vllm_wm:RedTiltLogitsProcessor ...

Per-request override through the OpenAI API: ``extra_body={"vllm_xargs": {"wm_delta": 1.5}}``
(``wm_delta: 0`` disables the tilt for that request, e.g. for judge / baseline calls on the
same server). Requires a vLLM with the model-runner-V2 logits-processor interface
(``vllm.v1.worker.gpu.sample.logits_processor``); a best-effort shim for the older V1
``update_state`` interface is included but untested.
"""

from __future__ import annotations

import os

import torch

from watermark_tuner.tilt import red_bias

try:  # vLLM is only present on the serving box; keep this module importable elsewhere (tests).
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor as _Base
except Exception:  # pragma: no cover - exercised only without vllm
    _Base = object


def apply_tilt(logits: torch.Tensor, deltas: torch.Tensor, unit_bias: torch.Tensor) -> torch.Tensor:
    """logits (R, V) += deltas (R,) [:, None] * unit_bias (V,) for rows with delta != 0. In place."""
    rows = torch.nonzero(deltas != 0).squeeze(1)
    if rows.numel() == 0:
        return logits
    logits[rows] += deltas[rows].to(logits.dtype)[:, None] * unit_bias.to(logits.dtype)[None, :]
    return logits


class RedTiltLogitsProcessor(_Base):
    def __init__(self, vllm_config, req_states):
        passkey = os.environ.get("WM_PASSKEY")
        if not passkey:
            raise RuntimeError("RedTiltLogitsProcessor: WM_PASSKEY must be set in the server environment")
        self.default_delta = float(os.environ.get("WM_DELTA", "0"))
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(vllm_config.model_config.tokenizer)
        self.unit_bias = red_bias(
            passkey, len(tok), 1.0, ignore_ids=tok.all_special_ids, vocab_size=req_states.vocab_size,
            device=req_states.device, dtype=torch.float32,
        )  # fmt: skip
        n = req_states.max_num_reqs
        self.delta = torch.zeros(n, dtype=torch.float32)
        self.delta_dev = torch.zeros(n, dtype=torch.float32, device=req_states.device)

    # ---- model-runner-V2 interface -------------------------------------------------------
    def add_request(self, req_idx: int, sampling_params) -> bool:
        d = (getattr(sampling_params, "extra_args", None) or {}).get("wm_delta", self.default_delta)
        self.delta[req_idx] = float(d)
        return float(d) != 0.0

    def apply_staged_writes(self) -> None:
        self.delta_dev.copy_(self.delta, non_blocking=True)

    @classmethod
    def validate_params(cls, sampling_params) -> None:
        d = (getattr(sampling_params, "extra_args", None) or {}).get("wm_delta")
        if d is not None:
            try:
                float(d)
            except (TypeError, ValueError):
                raise ValueError(f"wm_delta must be a number, got {d!r}") from None

    def apply(self, logits: torch.Tensor, ctx=None) -> torch.Tensor:
        if ctx is not None:  # MRV2: rows -> request slots
            deltas = self.delta_dev[ctx.expanded_idx_mapping.long()]
        else:  # legacy V1: rows are in slot order
            deltas = self.delta_dev[: logits.shape[0]]
        return apply_tilt(logits, deltas, self.unit_bias)

    # ---- legacy V1 shim (best effort) ---------------------------------------------------
    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update) -> None:
        if batch_update is None:
            return
        for index in getattr(batch_update, "removed", []) or []:
            self.delta[index] = 0.0
        for added in getattr(batch_update, "added", []) or []:
            index, params = added[0], added[1]
            self.add_request(index, params)
        for moved in getattr(batch_update, "moved", []) or []:
            a, b = moved[0], moved[1]
            direction = moved[2] if len(moved) > 2 else None
            if direction is not None and "SWAP" in str(direction).upper():
                self.delta[a], self.delta[b] = self.delta[b].item(), self.delta[a].item()
            else:  # unidirectional move a -> b
                self.delta[b] = self.delta[a].item()
                self.delta[a] = 0.0
        self.apply_staged_writes()
