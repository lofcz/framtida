"""The watermark objective in closed form: an exponential tilt of the reference model.

Maximising  E[ delta * #red tokens ]  subject to a per-token KL budget against a reference
paraphraser pi_ref has the optimum

    pi*(y_t | context) ∝ pi_ref(y_t | context) * exp(delta * 1[y_t is red])

i.e. the Kirchenbauer-style soft watermark applied to pi_ref's logits. Nothing about the
output is judged; the target is a property of the model's own distribution. Fidelity is
inherited from pi_ref and the per-token divergence is analytically bounded (see tilt_kl):

    KL(pi* || pi_ref) at a position with reference red mass p is  delta*q - log(1 + p(e^delta - 1)),
    q = p e^delta / (1 + p(e^delta - 1)).   For delta = 2 the worst case over p is ~0.42 nats.

We (1) sample from pi* (vLLM logits processor or HF generate) to get teacher rewrites, then
(2) distil pi* into LoRA weights so inference needs no logit processor at all.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from watermark_tuner.keys import red_mask


def red_bias(
    passkey: str,
    tokenizer_len: int,
    delta: float = 1.0,
    ignore_ids=(),
    vocab_size: int | None = None,
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """Additive logit bias vector: ``delta`` on red ids, 0 elsewhere (special ids, padding rows).

    The red/black split is defined over ``tokenizer_len`` ids (what the detector uses); the
    model's logit dimension ``vocab_size`` may be larger (Qwen3.8: 248320 vs 248077) and is
    zero-padded so the two always agree.
    """
    vocab_size = vocab_size or tokenizer_len
    mask = np.zeros(vocab_size, dtype=bool)
    mask[:tokenizer_len] = red_mask(passkey, tokenizer_len)
    for i in ignore_ids:
        if 0 <= int(i) < vocab_size:
            mask[int(i)] = False
    return torch.tensor(mask, dtype=dtype, device=device) * float(delta)


def tilt_logits(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """logits (..., V) + bias (V,)  -> tilted logits (same dtype)."""
    return logits + bias.to(device=logits.device, dtype=logits.dtype)


def tilt_red_prob(p_red: float, delta: float) -> float:
    """Red mass after tilting a position whose reference red mass is ``p_red``."""
    z = 1.0 - p_red + p_red * math.exp(delta)
    return p_red * math.exp(delta) / z


def tilt_kl(p_red: float, delta: float) -> float:
    """KL(tilted || reference) in nats at one position with reference red mass ``p_red``."""
    z = 1.0 - p_red + p_red * math.exp(delta)
    q = p_red * math.exp(delta) / z
    return delta * q - math.log(z)


def max_tilt_kl(delta: float, grid: int = 10_001) -> float:
    """Worst-case per-token KL over all reference red masses (numerical)."""
    ps = np.linspace(0.0, 1.0, grid)
    return float(max(tilt_kl(float(p), delta) for p in ps))


def red_mass(log_probs: torch.Tensor, red: torch.Tensor) -> torch.Tensor:
    """Probability mass on red ids per row: log_probs (..., V), red (V,) bool -> (...)."""
    return (log_probs.exp() * red.to(log_probs.dtype)).sum(-1)


class HFRedTiltProcessor:
    """`transformers` LogitsProcessor: adds the red bias at every decoding step.

    Duck-typed (``__call__(input_ids, scores)``) so this module imports without transformers.
    """

    def __init__(self, bias: torch.Tensor):
        self.bias = bias

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return tilt_logits(scores, self.bias)
