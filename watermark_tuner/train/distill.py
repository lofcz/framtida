"""Logit distillation of the tilted teacher into LoRA weights (single copy of the model).

For every position of a teacher-sampled rewrite:
    teacher  = softmax(logits_ref + delta * red)      (adapter disabled, no grad)
    student  = log_softmax(logits_theta)               (adapter enabled)
    loss     = KL(teacher || student)                  averaged over completion tokens

The teacher and the student share the base weights; PEFT's ``disable_adapter()`` gives the
reference forward for free, so memory is one bf16 model plus two sets of completion logits.
The objective is exactly the distribution we want (see watermark_tuner.tilt); no output is
judged, so there is nothing for the student to game. Compared with plain SFT on the teacher
samples (``train/sft.py``), matching full distributions transfers the watermark with far fewer
samples (Gu et al., "On the learnability of watermarks for language models").

    accelerate launch --config_file configs/accelerate_ddp.yaml --num_processes 2 \
        -m watermark_tuner.train.distill --data data/teacher_d2.jsonl --output logs/distill_d2 \
        --passkey "$WM_PASSKEY" --delta 2.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from watermark_tuner import DEFAULT_MODEL
from watermark_tuner.common import read_jsonl
from watermark_tuner.tilt import red_bias
from watermark_tuner.train.local_common import CHAT_KWARGS, load_model, load_tokenizer, lora_config


def to_example(row: dict, tokenizer, max_length: int) -> dict | None:
    """Tokenise prompt + completion; returns input ids and the number of completion tokens (incl. EOS)."""
    msgs = row["messages"]
    prompt_ids = tokenizer.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True, **CHAT_KWARGS)
    full_ids = tokenizer.apply_chat_template(msgs, tokenize=True, **CHAT_KWARGS)
    if hasattr(prompt_ids, "input_ids"):  # BatchEncoding in some transformers versions
        prompt_ids, full_ids = prompt_ids["input_ids"], full_ids["input_ids"]
    prompt_ids, full_ids = list(prompt_ids), list(full_ids)
    if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        return None
    # drop the template's trailing newline after <|im_end|> so the last target is EOS
    eos = tokenizer.eos_token_id
    if eos in full_ids[len(prompt_ids) :]:
        end = len(prompt_ids) + full_ids[len(prompt_ids) :].index(eos) + 1
        full_ids = full_ids[:end]
    if len(full_ids) > max_length:
        return None
    return {"input_ids": full_ids, "n_completion": len(full_ids) - len(prompt_ids)}


def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Forward KL(teacher_tilted || student) over (T, V) logits. Returns (loss, metrics)."""
    t_logp = torch.log_softmax(teacher_logits.float() + bias.float(), dim=-1)
    s_logp = torch.log_softmax(student_logits.float(), dim=-1)
    t_p = t_logp.exp()
    kl = (t_p * (t_logp - s_logp)).sum(-1).mean()
    red = bias != 0
    with torch.no_grad():
        metrics = {
            "teacher_red_mass": (t_p * red).sum(-1).mean().item(),
            "student_red_mass": (s_logp.exp() * red).sum(-1).mean().item(),
            "ref_red_mass": (torch.softmax(teacher_logits.float(), -1) * red).sum(-1).mean().item(),
        }
    return kl, metrics


def completion_logits(hf_model, input_ids: torch.Tensor, n_completion: int) -> torch.Tensor:
    """Logits that predict the completion tokens: feed all but the last token, keep the last n positions."""
    out = hf_model(input_ids=input_ids[:, :-1], logits_to_keep=n_completion)
    return out.logits[0]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="teacher conversations JSONL (data/sample_teacher)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--passkey", required=True)
    ap.add_argument("--delta", type=float, default=2.0, help="must match the delta the samples were drawn with")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--attn", default=None)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--log-steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from accelerate import Accelerator
    from peft import PeftModel, get_peft_model
    from torch.utils.data import DataLoader

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    tokenizer = load_tokenizer(args.model)
    rows = read_jsonl(args.data)
    examples = [e for e in (to_example(r, tokenizer, args.max_length) for r in rows) if e is not None]
    random.Random(args.seed).shuffle(examples)
    if accelerator.is_main_process:
        print(f"examples={len(examples)} (of {len(rows)}) model={args.model} delta={args.delta}")

    model = load_model(args.model, attn_implementation=args.attn)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config(args.lora_rank, args.lora_alpha))
    hf = model.get_base_model()
    vocab = hf.config.get_text_config().vocab_size
    bias = red_bias(args.passkey, len(tokenizer), args.delta, tokenizer.all_special_ids, vocab, accelerator.device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    loader = DataLoader(examples, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])
    steps_per_epoch = math.ceil(len(loader) / (args.grad_accum * accelerator.num_processes))
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)) * max(0.0, 1 - s / max(1, total_steps))
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    unwrapped = accelerator.unwrap_model(model)

    step, micro, t0, agg = 0, 0, time.time(), {}
    model.train()
    done = False
    while not done:
        for ex in loader:
            ids = torch.tensor([ex["input_ids"]], device=accelerator.device)
            n = ex["n_completion"]
            with accelerator.accumulate(model):
                with torch.no_grad(), unwrapped.disable_adapter():
                    t_logits = completion_logits(model, ids, n)
                s_logits = completion_logits(model, ids, n)
                loss, m = distill_loss(s_logits, t_logits, bias)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            micro += 1
            for k, v in {**m, "loss": loss.item()}.items():
                agg[k] = agg.get(k, 0.0) + v
            if accelerator.sync_gradients:
                step += 1
                if step % args.log_steps == 0 and accelerator.is_main_process:
                    k = max(1, micro)
                    print(json.dumps({"step": step, "lr": scheduler.get_last_lr()[0], "sec": round(time.time() - t0, 1), **{a: round(b / k, 4) for a, b in agg.items()}}))
                    micro, agg, t0 = 0, {}, time.time()
                if args.save_steps and step % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped.save_pretrained(os.path.join(args.output, f"step_{step}"))
                if step >= total_steps:
                    done = True
                    break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final = os.path.join(args.output, "final")
        unwrapped.save_pretrained(final)
        tokenizer.save_pretrained(final)
        print(f"adapter saved to {final} (relative to {args.model})")


if __name__ == "__main__":
    main()
