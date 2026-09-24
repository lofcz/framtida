"""Stage 1 (local GPUs): LoRA SFT with TRL + PEFT on the rejection-sampled conversations.

Loss is computed on the assistant turn only (prompt-completion format). Thinking is disabled via
the chat template so the target starts right after the empty ``<think></think>`` block.

Single GPU:
    python -m watermark_tuner.train.sft --data data/sft_round1.jsonl --output logs/sft_round1

Multi-GPU (DDP; each rank holds the full bf16 model, ~70-78 GB at micro-batch 1):
    accelerate launch --config_file configs/accelerate_ddp.yaml --num_processes 2 \
        -m watermark_tuner.train.sft --data data/sft_round1.jsonl --output logs/sft_round1

Effective batch = per_device_batch * grad_accum * num_processes. The adapter is written to
``<output>/final``; merge it with ``watermark_tuner.train.merge_adapter`` before RL / serving.
"""

from __future__ import annotations

import argparse
import os
import random

from watermark_tuner import DEFAULT_MODEL
from watermark_tuner.common import read_jsonl
from watermark_tuner.train.local_common import CHAT_KWARGS, is_main_process, load_model, load_tokenizer, lora_config


def to_prompt_completion(row: dict) -> dict:
    msgs = row["messages"]
    assert msgs[-1]["role"] == "assistant", "last message must be the assistant rewrite"
    return {"prompt": msgs[:-1], "completion": msgs[-1:], "chat_template_kwargs": dict(CHAT_KWARGS)}


def build_sft_config(args: argparse.Namespace):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if args.test_size > 0 else "no",
        eval_steps=args.eval_steps,
        max_length=args.max_length,
        completion_only_loss=True,
        use_liger_kernel=args.liger,
        report_to=[args.report_to] if args.report_to else [],
        run_name=args.run_name,
        seed=args.seed,
        dataloader_num_workers=2,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="conversations JSONL from rejection_sample")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--init-adapter", default=None, help="continue training an existing LoRA adapter")
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA: 4-bit base weights (fits 48 GB)")
    ap.add_argument("--liger", action="store_true", help="Liger kernels: fused CE over the 248k vocab, big memory win")
    ap.add_argument("--attn", default=None, help="attn_implementation, e.g. flash_attention_2")
    ap.add_argument("--test-size", type=int, default=100)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from, or 'auto'")
    ap.add_argument("--report-to", default=None, help="wandb | tensorboard")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from datasets import Dataset
    from trl import SFTTrainer

    rows = [to_prompt_completion(r) for r in read_jsonl(args.data)]
    random.Random(args.seed).shuffle(rows)
    test_rows, train_rows = rows[: args.test_size], rows[args.test_size :]
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(test_rows) if test_rows else None
    if is_main_process():
        print(f"train={len(train_ds)} eval={len(test_rows)} model={args.model}")

    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, load_4bit=args.load_4bit, attn_implementation=args.attn)
    peft_config = None
    if args.init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        peft_config = lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout)

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(args),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    resume = True if args.resume == "auto" else (args.resume or None)
    trainer.train(resume_from_checkpoint=resume)
    final = os.path.join(args.output, "final")
    trainer.save_model(final)
    if is_main_process():
        tokenizer.save_pretrained(final)
        print(f"adapter saved to {final}")


if __name__ == "__main__":
    main()
