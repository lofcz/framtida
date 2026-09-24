"""Passkey -> red/black vocabulary split.

The split is a uniformly random, exactly 50/50 partition of the token id space,
derived deterministically from the passkey. Anyone holding the passkey (and the
tokenizer) can reproduce the split; nobody else can tell red from black.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(passkey: str) -> int:
    """Map a passkey string to a 64-bit PRNG seed via SHA-256."""
    digest = hashlib.sha256(passkey.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def red_mask(passkey: str, vocab_size: int) -> np.ndarray:
    """Boolean mask of length ``vocab_size``; True = red token.

    Exactly ``vocab_size // 2`` ids are red. The partition is a random
    permutation seeded from the passkey, so it is stable across runs, machines
    and numpy versions that share the same default_rng bit generator (PCG64).
    """
    rng = np.random.default_rng(derive_seed(passkey))
    perm = rng.permutation(vocab_size)
    mask = np.zeros(vocab_size, dtype=bool)
    mask[perm[: vocab_size // 2]] = True
    return mask


def red_ids(passkey: str, vocab_size: int) -> np.ndarray:
    """Sorted array of red token ids."""
    return np.flatnonzero(red_mask(passkey, vocab_size))


def save_mask(mask: np.ndarray, path: str) -> None:
    np.save(path, mask)


def load_mask(path: str) -> np.ndarray:
    return np.load(path)
