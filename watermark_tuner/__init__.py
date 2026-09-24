"""Watermark-by-rewriting fine-tuning pipeline for Qwen3.8-27B on Tinker.

Pipeline:
    keys        -> passkey -> deterministic 50/50 red/black vocabulary split
    detect      -> binomial z-test detector over token ids
    fidelity    -> semantic / factual / structural preservation checks
    data        -> corpus construction + best-of-N rejection sampling (SFT data)
    train.sft   -> supervised warm start on Tinker
    train.rl    -> GRPO-style RL with watermark + fidelity reward on Tinker
    eval        -> detection rate at fixed FPR, fidelity metrics, diffs
    infer       -> rewrite a document with a trained checkpoint
"""

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_RENDERER = "qwen3_8_disable_thinking"
