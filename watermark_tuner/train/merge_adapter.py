"""Merge a LoRA adapter into the base weights and save a standalone HF checkpoint.

Needed between SFT and RL: GRPO's KL reference is "the model with the adapter disabled", so to
regularise RL toward the SFT policy (not the raw base) the SFT adapter must be baked in first.
The merged directory is also what ``vllm serve`` loads for rollouts and evaluation.

    python -m watermark_tuner.train.merge_adapter --model Qwen/Qwen3.8-27B \
        --adapter logs/sft_round1/final --output models/qwen38-27b-wm-sft
"""

from __future__ import annotations

import argparse

from watermark_tuner import DEFAULT_MODEL
from watermark_tuner.train.local_common import load_model, load_tokenizer


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="base model id or path the adapter was trained on")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    from peft import PeftModel

    model = load_model(args.model)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(args.output, safe_serialization=True)
    load_tokenizer(args.model).save_pretrained(args.output)
    # Keep the processor/chat template files vLLM expects for a VL checkpoint, if present.
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(args.model).save_pretrained(args.output)
    except Exception:  # text-only models have no processor
        pass
    print(f"merged model saved to {args.output}")


if __name__ == "__main__":
    main()
