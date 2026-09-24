"""Optional stage: GRPO polish with the *watermark-only* reward and a KL trust region.

Objective (default, ``--reward fraction``):
    r(y) = red_fraction(y)           for completions that terminated cleanly, else 0
    loss = GRPO(r) + beta * KL(pi || pi_ref)

With ``--reward count`` this is literally the exponential-tilt objective whose optimum is the
tilted teacher (watermark_tuner.tilt); ``fraction`` is the length-neutral variant. Nothing about
meaning is scored: fidelity comes from the KL trust region to a reference that already
paraphrases faithfully (the merged distillation checkpoint). The structural gates, NLI and the
LLM judge are still computed when configured, but only as **monitors** (logged as ``gate/*``,
``nli/*``, ``judge/*``); they never enter the reward, so the policy has nothing to game.

``--gated`` switches to the earlier gated reward (kept for ablation; see reward.py for why it
is not the default).

Layout for one 4x80GB node:
    CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve --model models/qwen38-27b-wm-distilled --tensor_parallel_size 2 \
        --max_model_len 4096 --gpu_memory_utilization 0.55 --port 8000
    CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file configs/accelerate_ddp.yaml --num_processes 2 \
        -m watermark_tuner.train.rl --corpus data/corpus.jsonl --passkey "$WM_PASSKEY" \
        --model models/qwen38-27b-wm-distilled --output logs/rl_round1 --vllm-server http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
from collections import Counter

from watermark_tuner import DEFAULT_MODEL
from watermark_tuner.backends import Sampler, VLLMChatSampler
from watermark_tuner.common import read_jsonl
from watermark_tuner.detect import WatermarkScorer
from watermark_tuner.fidelity import FidelityConfig, check_fidelity_pairs
from watermark_tuner.judge import judge_pair
from watermark_tuner.nli import NLIConfig, nli_score_pairs
from watermark_tuner.prompts import build_messages
from watermark_tuner.reward import RewardConfig, compute_reward
from watermark_tuner.train.local_common import (
    CHAT_KWARGS,
    completion_text,
    is_main_process,
    load_model,
    load_tokenizer,
    lora_config,
    world_size,
)

GATE_NAMES = (
    "truncated", "empty", "preamble", "len_ratio", "paragraph_count", "numbers", "urls", "quotes",
    "scripts", "invisible", "similarity", "too_short", "nli_contradiction", "judge_meaning", "judge_facts",
)  # fmt: skip


def _gate_key(g: str) -> str:
    return g.split("=", 1)[0]  # "similarity=0.812" -> "similarity"


def make_reward_fn(
    tokenizer,
    passkey: str,
    fid_cfg: FidelityConfig | None = None,
    reward_cfg: RewardConfig | None = None,
    nli_cfg: NLIConfig | None = None,
    judge: Sampler | None = None,
    judge_concurrency: int = 32,
    reward: str = "fraction",  # fraction | count
    gated: bool = False,
):
    """Build the async TRL reward function.

    ``gated=False`` (default): reward is the watermark statistic only; fidelity signals that are
    configured (``fid_cfg`` / ``nli_cfg`` / ``judge``) are computed for monitoring and logged.
    ``gated=True``: the earlier gated reward (compute_reward) with the same signals.
    """
    scorer = WatermarkScorer(tokenizer, passkey)
    eos_ids = {i for i in (tokenizer.eos_token_id, tokenizer.pad_token_id) if i is not None}
    sem = asyncio.Semaphore(judge_concurrency)
    reward_cfg = reward_cfg or RewardConfig()

    def local_signals(sources, texts, clean):
        dets = [scorer.score_text(t) if (c and t) else None for t, c in zip(texts, clean)]
        reps = check_fidelity_pairs(sources, texts, fid_cfg) if fid_cfg is not None else [None] * len(texts)
        nlis = [None] * len(texts)
        if nli_cfg is not None and fid_cfg is not None:
            idx = [i for i in range(len(texts)) if clean[i] and dets[i] is not None and (gated is False or reps[i].ok)]
            if idx:
                for i, s in zip(idx, nli_score_pairs([sources[i] for i in idx], [texts[i] for i in idx], fid_cfg, nli_cfg)):
                    nlis[i] = s
        return reps, dets, nlis

    async def judge_one(source, text):
        async with sem:
            try:
                return await judge_pair(judge, source, text)
            except Exception as e:  # judge outage must not poison training
                from watermark_tuner.judge import JudgeResult

                return JudgeResult(None, None, None, raw=f"error: {e}")

    async def watermark_reward(prompts, completions, completion_ids=None, source=None, log_metric=None, log_extra=None, **kwargs):
        assert source is not None, "dataset must carry a 'source' column"
        sources = list(source)
        texts = [completion_text(c) for c in completions]
        clean = [bool(ids) and int(ids[-1]) in eos_ids for ids in completion_ids] if completion_ids is not None else [True] * len(texts)

        reps, dets, nlis = await asyncio.to_thread(local_signals, sources, texts, clean)

        judges = [None] * len(texts)
        if judge is not None:
            if gated:  # only rollouts that passed every local gate reach the judge
                idx = [
                    i for i in range(len(texts))
                    if clean[i] and dets[i] is not None and reps[i].ok
                    and (nlis[i] is None or nlis[i].contradiction_max <= reward_cfg.contradiction_max)
                ]  # fmt: skip
            else:  # monitor every clean rollout
                idx = [i for i in range(len(texts)) if clean[i] and dets[i] is not None]
            if idx:
                for i, r in zip(idx, await asyncio.gather(*(judge_one(sources[i], texts[i]) for i in idx))):
                    judges[i] = r

        rewards, breakdowns = [], []
        for i in range(len(texts)):
            if gated:
                r, b = compute_reward(dets[i], reps[i], reward_cfg, nli=nlis[i], judge=judges[i], clean=clean[i])
            else:
                d = dets[i]
                r = 0.0 if d is None else (d.red_fraction if reward == "fraction" else float(d.n_red))
                b = {"gates": ["truncated"] if not clean[i] else (["empty"] if d is None else [])}
                if clean[i] and d is not None:  # monitors only make sense for complete rewrites
                    if reps[i] is not None and not reps[i].ok:
                        b["gates"] = b["gates"] + list(reps[i].failures)
                    if nlis[i] is not None and nlis[i].contradiction_max > reward_cfg.contradiction_max:
                        b["gates"].append("nli_contradiction")
                    if judges[i] is not None:
                        if judges[i].meaning_same is False:
                            b["gates"].append("judge_meaning")
                        if judges[i].facts_same is False:
                            b["gates"].append("judge_facts")
            rewards.append(r)
            breakdowns.append(b)

        if log_metric is not None:
            n = len(texts)
            gate_counts = Counter(_gate_key(g) for b in breakdowns for g in b["gates"])
            for g in GATE_NAMES:
                log_metric(f"gate/{g}", gate_counts.get(g, 0) / n)
            log_metric("monitor/all_clear_rate", sum(1 for b in breakdowns if not b["gates"]) / n)
            scored = [d for d in dets if d is not None]
            if scored:
                log_metric("watermark/red_fraction", statistics.fmean(d.red_fraction for d in scored))
                log_metric("watermark/z", statistics.fmean(d.z for d in scored))
                log_metric("watermark/detect_rate_z4", statistics.fmean(float(d.z >= 4.0) for d in scored))
            rr = [r for r in reps if r is not None]
            if rr:
                log_metric("fidelity/similarity", statistics.fmean(r.similarity for r in rr))
                log_metric("fidelity/len_ratio", statistics.fmean(r.len_ratio for r in rr))
                log_metric("fidelity/word_jaccard", statistics.fmean(r.word_jaccard for r in rr))
            nl = [x for x in nlis if x is not None]
            if nl:
                log_metric("nli/entail_mean", statistics.fmean(x.entail_mean for x in nl))
                log_metric("nli/contradiction_max", statistics.fmean(x.contradiction_max for x in nl))
            jd = [j for j in judges if j is not None]
            if jd:
                log_metric("judge/parse_fail_rate", statistics.fmean(float(not j.parsed) for j in jd))
                flu = [j.fluency for j in jd if j.fluency is not None]
                if flu:
                    log_metric("judge/fluency", statistics.fmean(flu))
                log_metric("judge/meaning_same_rate", statistics.fmean(float(j.meaning_same is True) for j in jd))
        if log_extra is not None:
            log_extra("red_fraction", [round(d.red_fraction, 3) if d else None for d in dets])
            log_extra("monitors", [",".join(b["gates"]) for b in breakdowns])
            log_extra("judge", [j.raw if j else "" for j in judges])
        return rewards

    watermark_reward.__name__ = "watermark_reward"
    return watermark_reward


def build_grpo_config(args: argparse.Namespace):
    from trl import GRPOConfig

    ws = world_size()
    gen_bs = args.groups_per_step * args.num_generations
    denom = args.per_device_batch * ws
    if gen_bs % denom:
        raise SystemExit(
            f"groups_per_step*num_generations ({gen_bs}) must be divisible by per_device_batch*world_size ({denom})"
        )
    grad_accum = gen_bs // denom
    return GRPOConfig(
        output_dir=args.output,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=grad_accum,
        # generation batch = per_device_batch * num_processes * steps_per_generation = groups*num_generations
        steps_per_generation=grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        epsilon=0.2,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        mask_truncated_completions=True,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url=args.vllm_server,
        chat_template_kwargs=dict(CHAT_KWARGS),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        max_steps=args.max_steps,
        num_train_epochs=1,
        log_completions=True,
        num_completions_to_print=2,
        report_to=[args.report_to] if args.report_to else [],
        run_name=args.run_name,
        seed=args.seed,
        remove_unused_columns=False,
        shuffle_dataset=True,
    )


def add_monitor_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("monitors (logged only unless --gated)")
    g.add_argument("--no-fidelity", action="store_true", help="skip structural + embedding monitors")
    g.add_argument("--embed-model", default=FidelityConfig.embed_model)
    g.add_argument("--signal-device", default=os.environ.get("WM_EMBED_DEVICE"), help="cpu | cuda (default: auto)")
    g.add_argument("--nli", action="store_true", help="enable sentence-level NLI monitor")
    g.add_argument("--nli-model", default=NLIConfig.model)
    g.add_argument("--judge-url", default=None, help="OpenAI-compatible base URL of a judge server (omit = no judge)")
    g.add_argument("--judge-model", default="judge")
    g.add_argument("--judge-concurrency", type=int, default=32)
    g.add_argument("--gated", action="store_true", help="use the gated reward (ablation) instead of watermark-only")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="corpus JSONL ({'text': ...} rows)")
    ap.add_argument("--passkey", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="reference checkpoint: merged distilled/SFT model, also served by vLLM")
    ap.add_argument("--init-adapter", default=None, help="continue an existing adapter instead of a fresh one")
    ap.add_argument("--vllm-server", default="http://localhost:8000", help="URL of `trl vllm-serve`")
    ap.add_argument("--reward", choices=["fraction", "count"], default="fraction")
    ap.add_argument("--beta", type=float, default=0.05, help="KL coefficient toward --model (the trust region)")
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--groups-per-step", type=int, default=32, help="distinct source chunks per optimizer step")
    ap.add_argument("--num-generations", type=int, default=8, help="rollouts per chunk (GRPO group size)")
    ap.add_argument("--per-device-batch", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--loss-type", default="dapo")
    ap.add_argument("--scale-rewards", default="group", help="group | batch | none")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=10)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--attn", default=None, help="attn_implementation, e.g. flash_attention_2")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N corpus rows")
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume from, or 'auto'")
    ap.add_argument("--report-to", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    add_monitor_args(ap)
    args = ap.parse_args(argv)

    from datasets import Dataset
    from trl import GRPOTrainer

    rows = read_jsonl(args.corpus, limit=args.limit)
    ds = Dataset.from_list([{"prompt": build_messages(r["text"]), "source": r["text"]} for r in rows])
    tokenizer = load_tokenizer(args.model)

    dev = {"device": args.signal_device} if args.signal_device else {}
    fid_cfg = None if args.no_fidelity else FidelityConfig(embed_model=args.embed_model, **dev)
    nli_cfg = NLIConfig(model=args.nli_model, **dev) if (args.nli and fid_cfg is not None) else None
    judge = VLLMChatSampler(base_url=args.judge_url, model=args.judge_model) if args.judge_url else None
    reward_fn = make_reward_fn(
        tokenizer, args.passkey, fid_cfg, RewardConfig(), nli_cfg, judge, args.judge_concurrency,
        reward=args.reward, gated=args.gated,
    )  # fmt: skip

    model = load_model(args.model, attn_implementation=args.attn)
    peft_config = None
    if args.init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        peft_config = lora_config(args.lora_rank, args.lora_alpha)

    if is_main_process():
        print(
            f"rows={len(ds)} model={args.model} world_size={world_size()} vllm={args.vllm_server} "
            f"reward={'gated' if args.gated else args.reward} beta={args.beta} "
            f"monitors: fidelity={'off' if fid_cfg is None else 'on'} nli={'on' if nli_cfg else 'off'} judge={args.judge_url or 'off'}"
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=build_grpo_config(args),
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    resume = True if args.resume == "auto" else (args.resume or None)
    trainer.train(resume_from_checkpoint=resume)
    final = os.path.join(args.output, "final")
    trainer.save_model(final)
    if is_main_process():
        tokenizer.save_pretrained(final)
        print(f"adapter saved to {final} (relative to {args.model})")


if __name__ == "__main__":
    main()
