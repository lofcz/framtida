"""Unigram watermark detector.

Under the null hypothesis (text not produced by the watermarked model) every
non-special token is red with probability 1/2 independently of the passkey, so
the red count R over n tokens is Binomial(n, 1/2). We report

    z = (R - n/2) / sqrt(n/4) = (2R - n) / sqrt(n)

and the one-sided p-value P(Z >= z). A text produced by a model whose red
fraction is q > 1/2 gives E[z] = 2(q - 1/2) sqrt(n); e.g. q = 0.60, n = 500
tokens -> z ~ 4.5 (p ~ 3e-6).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass

import numpy as np

from watermark_tuner import DEFAULT_MODEL
from watermark_tuner.keys import red_mask


@dataclass(frozen=True)
class Detection:
    n_tokens: int
    n_red: int
    red_fraction: float
    z: float
    p_value: float

    def to_dict(self) -> dict:
        return asdict(self)


def z_score(n_red: int, n_tokens: int) -> float:
    if n_tokens <= 0:
        return 0.0
    return (2.0 * n_red - n_tokens) / math.sqrt(n_tokens)


def p_value_from_z(z: float) -> float:
    """One-sided upper tail of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _z_for_tail(prob: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(1.0 - prob)


def expected_z(red_fraction: float, n_tokens: int) -> float:
    """Expected z for a text of ``n_tokens`` from a model with the given red fraction."""
    return 2.0 * (red_fraction - 0.5) * math.sqrt(n_tokens)


def detection_power(red_fraction: float, n_tokens: int, fpr: float = 1e-3) -> float:
    """P(detect) at the z threshold that gives ``fpr`` on unmarked text, for a marked text of ``n_tokens``.

    Under the alternative the red count is Binomial(n, q), so z ~ N(2(q-1/2)sqrt(n), 4q(1-q)).
    """
    from statistics import NormalDist

    mu = expected_z(red_fraction, n_tokens)
    sd = 2.0 * math.sqrt(red_fraction * (1.0 - red_fraction))
    return 1.0 - NormalDist().cdf((_z_for_tail(fpr) - mu) / sd)


def verdict_certainty(red_fraction: float, n_tokens: int) -> float:
    """Probability the verdict (marked / not marked) is correct for a document of ``n_tokens``.

    Uses the threshold halfway between the unmarked mean (z = 0) and the marked mean, so misses
    and false alarms are balanced; returns 1 - max(miss, false_alarm). Roughly Phi((q - 1/2) sqrt(n)).
    """
    from statistics import NormalDist

    nd = NormalDist()
    mu = expected_z(red_fraction, n_tokens)
    sd = 2.0 * math.sqrt(red_fraction * (1.0 - red_fraction))
    miss = nd.cdf((mu / 2.0 - mu) / sd)
    false_alarm = 1.0 - nd.cdf(mu / 2.0)
    return 1.0 - max(miss, false_alarm)


def tokens_needed(red_fraction: float, fpr: float = 1e-3, power: float = 0.95) -> int:
    """Smallest document length (tokens) at which a marked text is detected with ``power`` at ``fpr``."""
    sd = 2.0 * math.sqrt(red_fraction * (1.0 - red_fraction))
    za, zb = _z_for_tail(fpr), _z_for_tail(1.0 - power)
    return math.ceil(((za + zb * sd) / (2.0 * (red_fraction - 0.5))) ** 2)


def detect_ids(
    token_ids: list[int] | np.ndarray,
    mask: np.ndarray,
    ignore_ids: frozenset[int] | set[int] = frozenset(),
    unique: bool = False,
) -> Detection:
    """Score a token id sequence.

    Args:
        token_ids: token ids of the text under test.
        mask: boolean red mask from :func:`watermark_tuner.keys.red_mask`.
        ignore_ids: ids excluded from the count (special / chat-template tokens).
        unique: count each distinct token id once (more robust to word repetition
            attacks, cf. Zhao et al. "Unigram-Watermark"; lower power on short text).
    """
    ids = np.asarray(token_ids, dtype=np.int64)
    if ids.size:
        keep = (ids >= 0) & (ids < mask.shape[0])
        if ignore_ids:
            keep &= ~np.isin(ids, np.fromiter(ignore_ids, dtype=np.int64))
        ids = ids[keep]
    if unique and ids.size:
        ids = np.unique(ids)
    n = int(ids.size)
    r = int(mask[ids].sum()) if n else 0
    z = z_score(r, n)
    return Detection(n_tokens=n, n_red=r, red_fraction=(r / n if n else 0.0), z=z, p_value=p_value_from_z(z))


class WatermarkScorer:
    """Bundles tokenizer + red mask so callers can score text or ids directly."""

    def __init__(self, tokenizer, passkey: str, unique: bool = False):
        self.tokenizer = tokenizer
        self.passkey = passkey
        self.vocab_size = len(tokenizer)
        self.mask = red_mask(passkey, self.vocab_size)
        self.ignore_ids = frozenset(int(i) for i in (getattr(tokenizer, "all_special_ids", None) or []))
        self.unique = unique

    def score_ids(self, token_ids) -> Detection:
        return detect_ids(token_ids, self.mask, self.ignore_ids, unique=self.unique)

    def score_text(self, text: str) -> Detection:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return self.score_ids(ids)

    def is_red(self, token_id: int) -> bool:
        return 0 <= token_id < self.vocab_size and bool(self.mask[token_id])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Detect the unigram watermark in a text.")
    ap.add_argument("--passkey", required=True)
    ap.add_argument("--file", help="Text file to score; reads stdin if omitted.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id whose tokenizer defines the id space.")
    ap.add_argument("--unique", action="store_true", help="Count distinct token ids only.")
    ap.add_argument("--z-threshold", type=float, default=4.0, help="Decision threshold on z (4.0 ~ p 3e-5).")
    args = ap.parse_args(argv)

    from tinker_cookbook.tokenizer_utils import get_tokenizer

    text = open(args.file, encoding="utf-8").read() if args.file else sys.stdin.read()
    scorer = WatermarkScorer(get_tokenizer(args.model), args.passkey, unique=args.unique)
    det = scorer.score_text(text)
    out = det.to_dict()
    out["watermarked"] = det.z >= args.z_threshold
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
