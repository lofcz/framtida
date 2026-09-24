"""Paragraph-preserving chunking for long documents.

Long inputs are rewritten chunk by chunk and stitched back together with the
original inter-paragraph separators, so headings, blank lines and list layout
survive the round trip and the detector can be run on the full document.
"""

from __future__ import annotations

import re

_PARA_SEP = re.compile(r"(\n[ \t]*\n+)")
_SENT_SEP = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÝА-Я0-9\"“(])")


def _split_long_paragraph(par: str, max_chars: int) -> list[str]:
    """Split one oversized paragraph on sentence boundaries."""
    if len(par) <= max_chars:
        return [par]
    sentences = _SENT_SEP.split(par)
    out: list[str] = []
    cur = ""
    for s in sentences:
        if cur and len(cur) + 1 + len(s) > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, max_chars: int = 2500) -> tuple[list[str], list[str]]:
    """Split ``text`` into chunks of at most ~``max_chars`` characters.

    Returns ``(chunks, separators)`` where ``len(separators) == len(chunks) - 1``.
    ``join_chunks(chunks, separators) == text`` when no paragraph exceeded
    ``max_chars`` (oversized paragraphs are split on sentence boundaries and
    rejoined with a single space).
    """
    if not text:
        return [""], []
    parts = _PARA_SEP.split(text)
    paras = parts[0::2]
    seps = parts[1::2]

    units: list[tuple[str, str]] = []  # (paragraph_piece, separator_before)
    for i, p in enumerate(paras):
        sep_before = seps[i - 1] if i > 0 else ""
        pieces = _split_long_paragraph(p, max_chars)
        for j, piece in enumerate(pieces):
            units.append((piece, sep_before if j == 0 else " "))

    chunks: list[str] = []
    between: list[str] = []
    cur = units[0][0]
    for piece, sep in units[1:]:
        if len(cur) + len(sep) + len(piece) <= max_chars:
            cur += sep + piece
        else:
            chunks.append(cur)
            between.append(sep)
            cur = piece
    chunks.append(cur)
    return chunks, between


def join_chunks(chunks: list[str], separators: list[str]) -> str:
    out = chunks[0] if chunks else ""
    for c, s in zip(chunks[1:], separators):
        out += s + c
    return out


def paragraph_count(text: str) -> int:
    return len([p for p in _PARA_SEP.split(text)[0::2] if p.strip()])


def split_sentences(text: str) -> list[str]:
    """Rough multilingual sentence split: paragraph breaks and terminal punctuation."""
    out: list[str] = []
    for para in _PARA_SEP.split(text)[0::2]:
        for line in para.split("\n"):
            for s in _SENT_SEP.split(line):
                s = s.strip()
                if s:
                    out.append(s)
    return out
