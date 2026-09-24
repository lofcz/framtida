"""Offline tests for the tilt objective, the vLLM/HF logits processors and the distillation loss."""

from __future__ import annotations

import math

import pytest
import torch

from watermark_tuner.keys import red_mask
from watermark_tuner.tilt import HFRedTiltProcessor, max_tilt_kl, red_bias, red_mass, tilt_kl, tilt_logits, tilt_red_prob
from watermark_tuner.train.distill import distill_loss, to_example
from watermark_tuner.vllm_wm import apply_tilt


def test_tilt_math():
    assert tilt_kl(0.5, 0.0) == 0.0 and tilt_red_prob(0.5, 0.0) == 0.5
    assert tilt_red_prob(0.5, 2.0) == pytest.approx(math.e**2 / (1 + math.e**2))
    for p in (0.05, 0.3, 0.5, 0.9):
        assert tilt_kl(p, 2.0) >= 0.0
        assert tilt_red_prob(p, 2.0) > p
    # closed form matches a numerical KL on a random distribution
    torch.manual_seed(0)
    logits = torch.randn(50)
    red = torch.zeros(50, dtype=torch.bool)
    red[::2] = True
    p = torch.softmax(logits, -1)
    q = torch.softmax(logits + 2.0 * red.float(), -1)
    numeric = float((q * (q.log() - p.log())).sum())
    assert numeric == pytest.approx(tilt_kl(float(p[red].sum()), 2.0), abs=1e-5)
    assert 0.3 < max_tilt_kl(2.0) < 0.5 and max_tilt_kl(1.0) < max_tilt_kl(2.0)


def test_red_bias_layout():
    b = red_bias("k", tokenizer_len=100, delta=1.5, ignore_ids=[0, 1, 99], vocab_size=120)
    assert b.shape == (120,)
    assert (b[100:] == 0).all() and b[0] == 0 and b[1] == 0 and b[99] == 0
    expected = red_mask("k", 100)
    assert (b[:100] != 0).sum() == expected.sum() - int(expected[[0, 1, 99]].sum())
    assert set(b.unique().tolist()) <= {0.0, 1.5}


def test_hf_processor_and_apply_tilt():
    bias = torch.tensor([1.0, 0.0, 1.0, 0.0])
    scores = torch.zeros(2, 4)
    out = HFRedTiltProcessor(bias)(None, scores.clone())
    assert torch.equal(out, torch.tensor([[1.0, 0, 1, 0], [1.0, 0, 1, 0]]))
    logits = torch.zeros(3, 4)
    deltas = torch.tensor([2.0, 0.0, -1.0])
    apply_tilt(logits, deltas, bias)
    assert torch.equal(logits, torch.tensor([[2.0, 0, 2, 0], [0, 0, 0, 0], [-1.0, 0, -1, 0]]))
    assert torch.equal(tilt_logits(torch.ones(4), bias), torch.tensor([2.0, 1, 2, 1]))


def test_distill_loss_zero_at_optimum_and_red_mass():
    torch.manual_seed(1)
    t = torch.randn(7, 30)
    bias = torch.zeros(30)
    bias[:15] = 2.0
    loss, m = distill_loss(t + bias, t, bias)  # student already equals the tilted teacher
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
    assert m["teacher_red_mass"] > m["ref_red_mass"]
    assert m["student_red_mass"] == pytest.approx(m["teacher_red_mass"], abs=1e-5)
    loss2, _ = distill_loss(t, t, bias)  # untilted student
    assert loss2.item() > 0.05
    lp = torch.log_softmax(t, -1)
    assert red_mass(lp, bias != 0).shape == (7,)


class _FakeTok:
    eos_token_id = 9

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False, **kw):
        ids = []
        for m in msgs:
            ids += [1] + [ord(c) % 7 + 2 for c in m["content"]] + [9, 10]
        if add_generation_prompt:
            ids += [1]
        return ids


def test_to_example_positions():
    row = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": "abc"}]}
    ex = to_example(row, _FakeTok(), max_length=100)
    assert ex is not None
    prompt_len = len(_FakeTok().apply_chat_template(row["messages"][:-1], add_generation_prompt=True))
    assert ex["input_ids"][:prompt_len] == _FakeTok().apply_chat_template(row["messages"][:-1], add_generation_prompt=True)
    assert ex["input_ids"][-1] == 9 and ex["n_completion"] == len(ex["input_ids"]) - prompt_len == 4  # a b c EOS
    assert to_example(row, _FakeTok(), max_length=5) is None
