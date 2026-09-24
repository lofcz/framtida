"""Sentence-level bidirectional NLI between source and rewrite.

Embedding cosine cannot see negation flips, swapped entities or modal changes ("may" -> "will");
a natural-language-inference cross-encoder can. Chunks are too long for a 512-token NLI window,
so we align sentences: each source sentence is matched to its most similar output sentence (by
the fidelity embedding model), and NLI is run in both directions with a small window of
neighbouring sentences as the premise so sentence merges / splits do not read as "information
missing":

    premise = output window around the match, hypothesis = source sentence   (nothing dropped)
    premise = source window around i,          hypothesis = matched output   (nothing added)

Signals: ``contradiction_max`` (hard gate: any pair with p(contradiction) above threshold fails)
and ``entail_mean`` (soft penalty). Default model is multilingual.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from watermark_tuner.chunking import split_sentences
from watermark_tuner.fidelity import FidelityConfig, default_device, embed


@dataclass(frozen=True)
class NLIConfig:
    model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    device: str = field(default_factory=default_device)
    window: int = 1  # neighbouring sentences on each side used as premise context
    max_length: int = 512
    batch_size: int = 64
    contradiction_max: float = 0.5  # hard gate threshold


@dataclass
class NLIResult:
    entail_mean: float
    contradiction_max: float
    n_pairs: int
    worst_pair: tuple[str, str] | None = None

    def to_dict(self) -> dict:
        return {"entail_mean": self.entail_mean, "contradiction_max": self.contradiction_max, "n_pairs": self.n_pairs}


# --------------------------------------------------------------------------- #
# Model (lazy singleton)
# --------------------------------------------------------------------------- #
_NLI = None
_NLI_LOCK = threading.Lock()


class _NLIModel:
    def __init__(self, cfg: NLIConfig):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model)
        self.model.eval().to(cfg.device)
        if cfg.device.startswith("cuda"):
            self.model.half()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.i_entail = next(i for i, l in id2label.items() if "entail" in l)
        self.i_contra = next(i for i, l in id2label.items() if "contra" in l)
        self.torch = torch

    def probs(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        """Return (n, 2) array of [p_entail, p_contradiction]."""
        out = np.zeros((len(premises), 2), dtype=np.float32)
        bs = self.cfg.batch_size
        with self.torch.inference_mode():
            for i in range(0, len(premises), bs):
                enc = self.tokenizer(
                    premises[i : i + bs],
                    hypotheses[i : i + bs],
                    truncation=True,
                    max_length=self.cfg.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.cfg.device)
                logits = self.model(**enc).logits.float()
                p = self.torch.softmax(logits, dim=-1).cpu().numpy()
                out[i : i + bs, 0] = p[:, self.i_entail]
                out[i : i + bs, 1] = p[:, self.i_contra]
        return out


def _get_nli(cfg: NLIConfig) -> _NLIModel:
    global _NLI
    if _NLI is None:
        with _NLI_LOCK:
            if _NLI is None:
                _NLI = _NLIModel(cfg)
    return _NLI


# --------------------------------------------------------------------------- #
# Alignment + scoring
# --------------------------------------------------------------------------- #
def align_sentences(src_vecs: np.ndarray, out_vecs: np.ndarray) -> np.ndarray:
    """For each source sentence, index of the most similar output sentence (unit vectors)."""
    return np.argmax(src_vecs @ out_vecs.T, axis=1)


def _window(sents: list[str], i: int, w: int) -> str:
    return " ".join(sents[max(0, i - w) : i + w + 1])


def build_pairs(src: list[str], out: list[str], match: np.ndarray, window: int) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    premises, hypotheses, labels = [], [], []
    for i, s in enumerate(src):
        j = int(match[i])
        premises.append(_window(out, j, window))  # nothing dropped: output must entail the source sentence
        hypotheses.append(s)
        labels.append((s, out[j]))
        premises.append(_window(src, i, window))  # nothing added: source must entail the output sentence
        hypotheses.append(out[j])
        labels.append((s, out[j]))
    return premises, hypotheses, labels


def nli_score_pairs(
    sources: list[str], outputs: list[str], fid_cfg: FidelityConfig, nli_cfg: NLIConfig
) -> list[NLIResult | None]:
    """Score aligned (source_i, output_i) pairs; one embedding call and one NLI pass for the whole batch."""
    assert len(sources) == len(outputs)
    results: list[NLIResult | None] = [None] * len(sources)
    split = [(split_sentences(s), split_sentences(o)) for s, o in zip(sources, outputs)]
    active = [k for k, (ss, oo) in enumerate(split) if ss and oo]
    if not active:
        return results

    all_sents: list[str] = []
    spans: dict[int, tuple[slice, slice]] = {}
    for k in active:
        ss, oo = split[k]
        a = len(all_sents)
        all_sents.extend(ss)
        b = len(all_sents)
        all_sents.extend(oo)
        spans[k] = (slice(a, b), slice(b, len(all_sents)))
    vecs = embed(all_sents, fid_cfg)

    premises: list[str] = []
    hypotheses: list[str] = []
    owner: list[int] = []
    labels: list[tuple[str, str]] = []
    for k in active:
        ss, oo = split[k]
        s_sl, o_sl = spans[k]
        match = align_sentences(vecs[s_sl], vecs[o_sl])
        p, h, lab = build_pairs(ss, oo, match, nli_cfg.window)
        premises.extend(p)
        hypotheses.extend(h)
        labels.extend(lab)
        owner.extend([k] * len(p))

    probs = _get_nli(nli_cfg).probs(premises, hypotheses)
    for k in active:
        idx = [i for i, o in enumerate(owner) if o == k]
        pe = probs[idx, 0]
        pc = probs[idx, 1]
        worst = int(np.argmax(pc))
        results[k] = NLIResult(
            entail_mean=float(pe.mean()),
            contradiction_max=float(pc.max()),
            n_pairs=len(idx),
            worst_pair=labels[idx[worst]],
        )
    return results


def nli_score(source: str, output: str, fid_cfg: FidelityConfig, nli_cfg: NLIConfig) -> NLIResult | None:
    return nli_score_pairs([source], [output], fid_cfg, nli_cfg)[0]
