"""Fidelity checks: did the rewrite preserve meaning, facts and structure?

Two layers:

* **Hard gates** (cheap, deterministic): non-empty, no chat preamble, length
  ratio in range, same paragraph count, identical sets of numbers / URLs /
  emails, identical multi-word quotations, similarity above a floor.
  A failed hard gate makes the sample unusable (SFT) or maximally penalised (RL).
* **Soft score**: cosine similarity of multilingual sentence embeddings, used to
  rank candidates and as a smooth RL penalty term.

The embedding model runs on CPU by default and is loaded lazily once per
process.
"""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from dataclasses import asdict, dataclass, field

import numpy as np

from watermark_tuner.chunking import paragraph_count


def default_device() -> str:
    """WM_EMBED_DEVICE, else cuda when available (each DDP rank sees its own GPU), else cpu."""
    env = os.environ.get("WM_EMBED_DEVICE")
    if env:
        return env
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

_NUM_RE = re.compile(r"\d+(?:[.,:/]\d+)*")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+|[\w.+-]+@[\w-]+\.[\w.-]+")
_QUOTE_RE = re.compile(r"[\"“„«]([^\"”“»]{12,}?)[\"”“»]")
_PREAMBLE_RE = re.compile(
    r"^\s*(here(?:'s| is)\b|sure\b|certainly\b|below is\b|edited text\s*:|rewritten text\s*:|"
    r"okay\b|of course\b|i have\b|i've\b)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class FidelityConfig:
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    min_similarity: float = 0.85  # hard floor
    min_len_ratio: float = 0.85
    max_len_ratio: float = 1.20
    require_same_paragraphs: bool = True
    require_same_numbers: bool = True
    require_same_urls: bool = True
    require_same_quotes: bool = True
    require_same_scripts: bool = True  # no new Unicode scripts (homoglyph / token-swapping defence)
    device: str = field(default_factory=default_device)


@dataclass
class FidelityReport:
    ok: bool
    similarity: float
    len_ratio: float
    word_jaccard: float
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Embeddings (lazy singleton)
# --------------------------------------------------------------------------- #
_EMBEDDER = None
_EMBED_LOCK = threading.Lock()


def _get_embedder(cfg: FidelityConfig):
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBED_LOCK:
            if _EMBEDDER is None:
                from sentence_transformers import SentenceTransformer

                _EMBEDDER = SentenceTransformer(cfg.embed_model, device=cfg.device)
                # Default max_seq_length for the MiniLM paraphrase models is 128 word pieces,
                # which would compare only the first ~third of a 400-token chunk.
                _EMBEDDER.max_seq_length = max(int(getattr(_EMBEDDER, "max_seq_length", 128) or 128), 512)
    return _EMBEDDER


def embed(texts: list[str], cfg: FidelityConfig) -> np.ndarray:
    model = _get_embedder(cfg)
    with _EMBED_LOCK:
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def similarity_matrix(source: str, candidates: list[str], cfg: FidelityConfig) -> np.ndarray:
    """Cosine similarity of ``source`` against each candidate, one embedding call."""
    if not candidates:
        return np.zeros(0, dtype=np.float32)
    vecs = embed([source] + candidates, cfg)
    return vecs[1:] @ vecs[0]


# --------------------------------------------------------------------------- #
# Deterministic checks
# --------------------------------------------------------------------------- #
def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def word_jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


def letter_scripts(text: str) -> set[str]:
    """Unicode script prefixes of the letters in ``text`` ("LATIN", "CYRILLIC", "CJK", ...).

    RL can raise the red fraction by swapping letters for look-alikes from other scripts
    (homoglyphs), which changes tokens while leaving embeddings and every other check untouched.
    Requiring the output's script set to be a subset of the source's closes that door.
    """
    scripts: set[str] = set()
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            if name:
                scripts.add(name.split(" ", 1)[0])
    return scripts


def invisible_chars(text: str) -> set[str]:
    """Format / zero-width characters (category Cf), excluding soft hyphen-free plain text."""
    return {ch for ch in text if unicodedata.category(ch) == "Cf"}


def hard_checks(source: str, output: str, cfg: FidelityConfig) -> tuple[list[str], float]:
    """Return (list_of_failure_reasons, len_ratio)."""
    failures: list[str] = []
    out = output.strip()
    src = source.strip()
    len_ratio = len(out) / max(1, len(src))

    if not out:
        failures.append("empty")
        return failures, len_ratio
    if _PREAMBLE_RE.match(out):
        failures.append("preamble")
    if not (cfg.min_len_ratio <= len_ratio <= cfg.max_len_ratio):
        failures.append(f"len_ratio={len_ratio:.2f}")
    if cfg.require_same_paragraphs and paragraph_count(src) != paragraph_count(out):
        failures.append("paragraph_count")
    if cfg.require_same_numbers and set(_NUM_RE.findall(src)) != set(_NUM_RE.findall(out)):
        failures.append("numbers")
    if cfg.require_same_urls and set(_URL_RE.findall(src)) != set(_URL_RE.findall(out)):
        failures.append("urls")
    if cfg.require_same_quotes:
        src_quotes = {q.strip() for q in _QUOTE_RE.findall(src)}
        if src_quotes and not all(q in out for q in src_quotes):
            failures.append("quotes")
    if cfg.require_same_scripts:
        if letter_scripts(out) - letter_scripts(src):
            failures.append("scripts")
        if invisible_chars(out) - invisible_chars(src):
            failures.append("invisible")
    return failures, len_ratio


def check_fidelity(source: str, output: str, cfg: FidelityConfig, similarity: float | None = None) -> FidelityReport:
    """Full report for one (source, output) pair. Pass ``similarity`` to reuse a batch computation."""
    failures, len_ratio = hard_checks(source, output, cfg)
    if similarity is None:
        similarity = float(similarity_matrix(source, [output], cfg)[0]) if output.strip() else 0.0
    if similarity < cfg.min_similarity:
        failures.append(f"similarity={similarity:.3f}")
    return FidelityReport(
        ok=not failures,
        similarity=float(similarity),
        len_ratio=float(len_ratio),
        word_jaccard=word_jaccard(source, output),
        failures=failures,
    )


def check_fidelity_batch(source: str, outputs: list[str], cfg: FidelityConfig) -> list[FidelityReport]:
    """Score many candidates for one source with a single embedding call."""
    sims = similarity_matrix(source, outputs, cfg)
    return [check_fidelity(source, o, cfg, similarity=float(s)) for o, s in zip(outputs, sims)]


def check_fidelity_pairs(sources: list[str], outputs: list[str], cfg: FidelityConfig) -> list[FidelityReport]:
    """Score aligned (source_i, output_i) pairs with a single embedding call.

    Used by the RL reward, where every completion in a batch may have a
    different source. Empty outputs get similarity 0 without being embedded.
    """
    assert len(sources) == len(outputs)
    if not sources:
        return []
    idx = [i for i, o in enumerate(outputs) if o.strip()]
    sims = np.zeros(len(outputs), dtype=np.float32)
    if idx:
        vecs = embed([sources[i] for i in idx] + [outputs[i] for i in idx], cfg)
        k = len(idx)
        sims[idx] = np.einsum("ij,ij->i", vecs[:k], vecs[k:])
    return [check_fidelity(s, o, cfg, similarity=float(sim)) for s, o, sim in zip(sources, outputs, sims)]
