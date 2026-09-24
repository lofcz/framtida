"""Offline tests for the gated reward, judge parsing, script gate, and NLI alignment helpers."""

from __future__ import annotations

import asyncio
import types

import numpy as np
import pytest

from watermark_tuner.detect import Detection
from watermark_tuner.fidelity import FidelityConfig, FidelityReport, hard_checks, letter_scripts
from watermark_tuner.judge import JudgeResult, parse_judge
from watermark_tuner.nli import NLIResult, align_sentences, build_pairs
from watermark_tuner.reward import RewardConfig, compute_reward

DET = Detection(n_tokens=400, n_red=240, red_fraction=0.6, z=4.0, p_value=3e-5)
GOOD = FidelityReport(ok=True, similarity=0.97, len_ratio=1.0, word_jaccard=0.8)


def test_reward_admissible_baseline():
    r, b = compute_reward(DET, GOOD, RewardConfig())
    assert r == pytest.approx(1.0) and b["gates"] == []


def test_reward_soft_penalties():
    cfg = RewardConfig()
    drift = FidelityReport(ok=True, similarity=0.90, len_ratio=1.1, word_jaccard=0.4)
    nli = NLIResult(entail_mean=0.8, contradiction_max=0.1, n_pairs=10)
    judge = JudgeResult(True, True, 2)
    r, b = compute_reward(DET, drift, cfg, nli=nli, judge=judge)
    expected = 1.0 - 5 * 0.05 - 2 * 0.10 - 1.0 * (4 - 2) / 4 - 0.1 - 1.0 * 0.1
    assert r == pytest.approx(expected)
    assert b["nli_pen"] == pytest.approx(0.2) and b["flu_pen"] == pytest.approx(0.5) and b["edit_pen"] == pytest.approx(0.1)


def test_reward_cap_and_missing_signals():
    hot = Detection(n_tokens=400, n_red=380, red_fraction=0.95, z=18.0, p_value=0.0)
    r, _ = compute_reward(hot, GOOD, RewardConfig())
    assert r == pytest.approx(10 * (0.8 - 0.5))  # capped at red_cap
    # judge parsed nothing -> no penalty, no gate
    r2, b2 = compute_reward(DET, GOOD, RewardConfig(), judge=JudgeResult(None, None, None))
    assert r2 == pytest.approx(1.0) and b2["gates"] == []


def test_reward_gates():
    cfg = RewardConfig()
    assert compute_reward(DET, GOOD, cfg, clean=False) == (cfg.truncated_reward, {"gates": ["truncated"]})
    assert compute_reward(None, GOOD, cfg)[0] == cfg.truncated_reward
    bad = FidelityReport(ok=False, similarity=0.5, len_ratio=1.0, word_jaccard=0.1, failures=["numbers", "similarity=0.500"])
    r, b = compute_reward(DET, bad, cfg)
    assert r == cfg.fail_reward and b["gates"] == ["numbers", "similarity=0.500"]
    r, b = compute_reward(DET, GOOD, cfg, nli=NLIResult(0.9, 0.9, 5))
    assert r == cfg.fail_reward and b["gates"] == ["nli_contradiction"]
    r, b = compute_reward(DET, GOOD, cfg, judge=JudgeResult(False, True, 5))
    assert r == cfg.fail_reward and b["gates"] == ["judge_meaning"]
    r, b = compute_reward(DET, GOOD, cfg, judge=JudgeResult(True, False, 5))
    assert b["gates"] == ["judge_facts"]
    short = Detection(n_tokens=3, n_red=3, red_fraction=1.0, z=1.7, p_value=0.04)
    assert compute_reward(short, GOOD, cfg)[1]["gates"] == ["too_short"]


def test_parse_judge():
    j = parse_judge('Sure. {"meaning": "SAME", "facts": "CHANGED", "fluency": 4}')
    assert (j.meaning_same, j.facts_same, j.fluency, j.parsed) == (True, False, 4, True)
    j = parse_judge("meaning: SAME\nfacts: SAME\nfluency: 5")  # regex fallback
    assert (j.meaning_same, j.facts_same, j.fluency) == (True, True, 5)
    j = parse_judge("I cannot decide")
    assert not j.parsed and j.meaning_same is None
    j = parse_judge('{"meaning": "same", "facts": "SAME", "fluency": 9}')
    assert j.meaning_same is True and j.fluency is None


def test_script_gate_catches_homoglyphs():
    cfg = FidelityConfig()
    src = "The Commission adopted the report in March.\n\nSecond paragraph here."
    ok = "The Commission approved the report in March.\n\nSecond paragraph here."
    assert hard_checks(src, ok, cfg)[0] == []
    homoglyph = ok.replace("Commission", "Cоmmission")  # Cyrillic 'о'
    assert "scripts" in hard_checks(src, homoglyph, cfg)[0]
    assert "invisible" in hard_checks(src, ok.replace("report", "re​port"), cfg)[0]
    assert letter_scripts("Ünïcode café") == {"LATIN"}
    # source that already mixes scripts is fine
    src2 = "Term: Москва (Moscow).\n\nMore."
    assert hard_checks(src2, "Term: Москва (Moscow) here.\n\nMore.", cfg)[0] == []


def test_nli_alignment_and_pairs():
    src = ["A one.", "B two.", "C three."]
    out = ["C tres.", "A uno.", "B dos."]
    # unit vectors making out[1]~src[0], out[2]~src[1], out[0]~src[2]
    sv = np.eye(3, dtype=np.float32)
    ov = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    match = align_sentences(sv, ov)
    assert match.tolist() == [1, 2, 0]
    prem, hyp, labels = build_pairs(src, out, match, window=1)
    assert len(prem) == 6 and hyp[0] == "A one." and hyp[1] == "A uno."
    assert prem[0] == "C tres. A uno. B dos."  # output window around match j=1 (all three)
    assert prem[1] == "A one. B two."  # source window around i=0
    assert labels[0] == ("A one.", "A uno.")


def test_make_reward_fn_end_to_end(monkeypatch):
    """Async reward with stubbed fidelity / NLI / judge; no models, no network."""
    import watermark_tuner.train.rl as rl

    class FakeTok:
        eos_token_id, pad_token_id, all_special_ids = 1, 0, [0, 1]

        def __len__(self):
            return 1000

        def encode(self, text, add_special_tokens=False):
            return [2 + (ord(c) % 900) for c in text]

    monkeypatch.setattr(
        rl, "check_fidelity_pairs",
        lambda s, o, cfg: [FidelityReport(ok=bool(t), similarity=0.97 if t else 0.0, len_ratio=1.0, word_jaccard=0.9) for t in o],
    )
    monkeypatch.setattr(rl, "nli_score_pairs", lambda s, o, f, n: [NLIResult(0.95, 0.05, 4) for _ in o])

    async def fake_judge(sampler, source, text):
        return JudgeResult(meaning_same=(text != "flipped"), facts_same=True, fluency=5, raw="ok")

    monkeypatch.setattr(rl, "judge_pair", fake_judge)

    fn = rl.make_reward_fn(FakeTok(), "k", types.SimpleNamespace(), RewardConfig(), nli_cfg=types.SimpleNamespace(), judge=object(), gated=True)
    assert fn.__name__ == "watermark_reward" and asyncio.iscoroutinefunction(fn)
    logged: dict[str, float] = {}
    rewards = asyncio.run(
        fn(
            prompts=["p"] * 4,
            completions=[
                [{"role": "assistant", "content": "The quick brown fox."}],
                [{"role": "assistant", "content": "flipped"}],
                [{"role": "assistant", "content": "truncated one"}],
                [{"role": "assistant", "content": ""}],
            ],
            completion_ids=[[5, 6, 1], [5, 1], [5, 6, 7], [1]],
            source=["s"] * 4,
            log_metric=lambda k, v: logged.__setitem__(k, v),
            log_extra=lambda k, v: None,
        )
    )
    cfg = RewardConfig()
    assert rewards[0] > cfg.fail_reward  # admissible (red fraction of the fake tokens may be ~0.5)
    assert rewards[1] == cfg.fail_reward  # judge said meaning CHANGED
    assert rewards[2] == cfg.truncated_reward  # no EOS
    assert rewards[3] == cfg.truncated_reward  # empty
    assert logged["gate/judge_meaning"] == pytest.approx(0.25)
    assert logged["gate/truncated"] == pytest.approx(0.25) and logged["gate/empty"] == pytest.approx(0.25)
    assert logged["monitor/all_clear_rate"] == pytest.approx(0.25)
    assert "nli/entail_mean" in logged and logged["judge/fluency"] == 5


def test_make_reward_fn_watermark_only(monkeypatch):
    """Default mode: reward is the red fraction; fidelity signals are logged but never change the reward."""
    import watermark_tuner.train.rl as rl

    class FakeTok:
        eos_token_id, pad_token_id, all_special_ids = 1, 0, [0, 1]

        def __len__(self):
            return 1000

        def encode(self, text, add_special_tokens=False):
            return [2 + (ord(c) % 900) for c in text]

    monkeypatch.setattr(
        rl, "check_fidelity_pairs",
        lambda s, o, cfg: [FidelityReport(ok=False, similarity=0.1, len_ratio=3.0, word_jaccard=0.0, failures=["numbers"]) for _ in o],
    )

    async def fake_judge(sampler, source, text):
        return JudgeResult(meaning_same=False, facts_same=False, fluency=1, raw="bad")

    monkeypatch.setattr(rl, "judge_pair", fake_judge)
    fn = rl.make_reward_fn(FakeTok(), "k", types.SimpleNamespace(), None, None, judge=object(), gated=False)
    logged = {}
    text = "The quick brown fox jumps over the lazy dog"
    rewards = asyncio.run(
        fn(
            prompts=["p", "p"],
            completions=[[{"role": "assistant", "content": text}], [{"role": "assistant", "content": "trunc"}]],
            completion_ids=[[5, 6, 1], [5, 6, 7]],
            source=["s", "s"],
            log_metric=lambda k, v: logged.__setitem__(k, v),
        )
    )
    from watermark_tuner.detect import WatermarkScorer

    assert rewards[0] == pytest.approx(WatermarkScorer(FakeTok(), "k").score_text(text).red_fraction)
    assert rewards[1] == 0.0
    # monitors fired (all-clear rate 0) but the reward ignored them
    assert logged["gate/numbers"] == pytest.approx(0.5) and logged["gate/judge_meaning"] == pytest.approx(0.5)
    assert logged["monitor/all_clear_rate"] == 0.0
