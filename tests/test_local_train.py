"""Offline tests for the local (TRL) trainers and sampling backends. No GPU, no model download."""

from __future__ import annotations

import argparse
import pytest

from watermark_tuner.backends import SampleOut, VLLMChatSampler
from watermark_tuner.train.local_common import completion_text
from watermark_tuner.train.sft import to_prompt_completion



def test_completion_text_formats():
    assert completion_text("  hi  ") == "hi"
    assert completion_text([{"role": "assistant", "content": "<think>\n\n</think>\n\nhello"}]) == "hello"
    assert completion_text([{"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]) == "ab"


def test_to_prompt_completion():
    row = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}]}
    ex = to_prompt_completion(row)
    assert [m["role"] for m in ex["prompt"]] == ["system", "user"]
    assert ex["completion"] == [{"role": "assistant", "content": "A"}]
    assert ex["chat_template_kwargs"] == {"enable_thinking": False}


def test_vllm_parse_response():
    payload = {
        "choices": [
            {"message": {"content": "<think>\n\n</think>\n\nEdited text."}, "finish_reason": "stop"},
            {"message": {"content": "cut off"}, "finish_reason": "length"},
            {"message": {"content": None}, "finish_reason": "stop"},
        ]
    }
    outs = VLLMChatSampler.parse_response(payload)
    assert outs == [SampleOut("Edited text.", True), SampleOut("cut off", False), SampleOut("", True)]


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


@pytest.fixture
def fake_bf16_gpu(monkeypatch):
    """TrainingArguments refuses bf16 without an Ampere GPU; the configs themselves are GPU-agnostic."""
    import transformers.training_args as ta

    monkeypatch.setattr(ta, "is_torch_bf16_gpu_available", lambda: True)


def test_build_sft_config(tmp_path, fake_bf16_gpu):
    from watermark_tuner.train.sft import build_sft_config

    args = _ns(
        output=str(tmp_path), per_device_batch=1, grad_accum=4, learning_rate=1e-4, epochs=1, max_steps=0,
        save_steps=10, test_size=5, eval_steps=5, max_length=1024, liger=False, report_to=None, run_name=None, seed=0,
    )
    cfg = build_sft_config(args)
    assert cfg.completion_only_loss is True and cfg.max_length == 1024 and cfg.bf16 is True


def test_build_grpo_config(tmp_path, monkeypatch, fake_bf16_gpu):
    from watermark_tuner.train.rl import build_grpo_config

    monkeypatch.setenv("WORLD_SIZE", "2")
    args = _ns(
        output=str(tmp_path), learning_rate=2e-5, groups_per_step=8, num_generations=4, per_device_batch=4,
        max_completion_length=256, temperature=1.0, top_p=1.0, beta=0.02, loss_type="dapo", scale_rewards="group",
        vllm_server="http://localhost:8000", save_steps=10, max_steps=3, report_to=None, run_name=None, seed=0,
    )
    cfg = build_grpo_config(args)
    # 8 groups * 4 generations = 32 completions per optimizer step = per_device(4) * world(2) * steps_per_generation(4)
    assert cfg.steps_per_generation == 4 and cfg.gradient_accumulation_steps == 4 and cfg.num_generations == 4
    assert cfg.use_vllm and cfg.vllm_mode == "server" and cfg.chat_template_kwargs == {"enable_thinking": False}
    args.groups_per_step = 3
    with pytest.raises(SystemExit):
        build_grpo_config(args)


