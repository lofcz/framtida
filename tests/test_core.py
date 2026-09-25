"""Offline unit tests: key derivation, detector statistics, chunking, hard fidelity gates.

Run: python -m pytest tests -q   (no Tinker key or model download required)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from watermark_tuner.chunking import chunk_text, join_chunks, paragraph_count
from watermark_tuner.detect import detect_ids, p_value_from_z, z_score
from watermark_tuner.fidelity import FidelityConfig, hard_checks
from watermark_tuner.keys import red_mask


def test_red_mask_is_deterministic_and_half():
    m1 = red_mask("passkey-1", 1001)
    m2 = red_mask("passkey-1", 1001)
    m3 = red_mask("passkey-2", 1001)
    assert np.array_equal(m1, m2)
    assert not np.array_equal(m1, m3)
    assert m1.sum() == 500  # exactly floor(n/2)


def test_z_score_and_pvalue():
    assert z_score(50, 100) == 0.0
    assert math.isclose(z_score(60, 100), 2.0)
    assert math.isclose(p_value_from_z(0.0), 0.5)
    assert p_value_from_z(4.0) < 5e-5


def test_detect_null_is_centered():
    rng = np.random.default_rng(0)
    mask = red_mask("k", 50_000)
    zs = []
    for _ in range(200):
        ids = rng.integers(0, 50_000, size=400)
        zs.append(detect_ids(ids, mask).z)
    zs = np.array(zs)
    assert abs(zs.mean()) < 0.25
    assert 0.7 < zs.std() < 1.3


def test_detect_biased_text_is_detected():
    mask = red_mask("k", 50_000)
    red = np.flatnonzero(mask)
    black = np.flatnonzero(~mask)
    rng = np.random.default_rng(1)
    # 60% red over 500 tokens -> E[z] ~ 4.5
    ids = np.concatenate([rng.choice(red, 300), rng.choice(black, 200)])
    det = detect_ids(ids, mask)
    assert det.red_fraction == 0.6
    assert det.z > 3.5


def test_detect_ignores_special_and_out_of_range():
    mask = red_mask("k", 100)
    ids = [5, 6, 7, 500, -1]
    det = detect_ids(ids, mask, ignore_ids={5})
    assert det.n_tokens == 2


def test_chunk_roundtrip():
    text = "Para one.\n\nPara two is here.\n\n\nPara three."
    chunks, seps = chunk_text(text, max_chars=20)
    assert join_chunks(chunks, seps) == text
    assert len(chunks) == 3
    big, seps2 = chunk_text(text, max_chars=10_000)
    assert big == [text] and seps2 == []


def test_chunk_splits_long_paragraph():
    text = "Sentence one is here. Sentence two is here. Sentence three is here."
    chunks, _ = chunk_text(text, max_chars=30)
    assert len(chunks) >= 2
    assert paragraph_count(text) == 1


def test_hard_checks():
    cfg = FidelityConfig()
    src = "In 2021 the fee was 12.5 EUR. See https://example.eu/x for details.\n\nSecond paragraph."
    ok_out = "In 2021 the charge was 12.5 EUR. See https://example.eu/x for the details.\n\nSecond paragraph."
    fails, ratio = hard_checks(src, ok_out, cfg)
    assert fails == [] and 0.9 < ratio < 1.15

    assert "numbers" in hard_checks(src, ok_out.replace("12.5", "13.5"), cfg)[0]
    assert "urls" in hard_checks(src, ok_out.replace("example.eu", "example.org"), cfg)[0]
    assert "paragraph_count" in hard_checks(src, ok_out.replace("\n\n", " "), cfg)[0]
    assert "preamble" in hard_checks(src, "Here is the edited text:\n" + ok_out, cfg)[0]
    assert "empty" in hard_checks(src, "   ", cfg)[0]
    assert any(f.startswith("len_ratio") for f in hard_checks(src, ok_out + " " + ok_out, cfg)[0])


def test_power_model():
    from watermark_tuner.detect import detection_power, expected_z, tokens_needed

    assert expected_z(0.6, 500) == pytest.approx(4.472, abs=1e-3)
    assert 0.90 < detection_power(0.6, 500, 1e-3) < 0.94
    assert detection_power(0.55, 500, 1e-5) < 0.05
    assert detection_power(0.65, 2000, 1e-5) > 0.999
    n = tokens_needed(0.6, 1e-3, 0.95)
    assert 500 < n < 600 and detection_power(0.6, n, 1e-3) >= 0.95
    assert tokens_needed(0.55, 1e-3) > tokens_needed(0.6, 1e-3) > tokens_needed(0.65, 1e-3)


def test_verdict_certainty():
    from watermark_tuner.detect import verdict_certainty

    assert 0.98 < verdict_certainty(0.6, 500) < 0.995
    assert verdict_certainty(0.55, 500) < verdict_certainty(0.55, 2000) < verdict_certainty(0.6, 2000)
    assert verdict_certainty(0.65, 4000) > 0.9999
